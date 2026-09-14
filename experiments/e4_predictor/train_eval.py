#!/usr/bin/env python3
"""E4 -- Predictor upgrade and confidence calibration (Experiments.md E4, [P1]).

The pilot's frequency-table predictor gets 55-62% recall at m=2k. Two things
must improve: (a) higher recall at LOWER m (m is a bandwidth multiplier --
Experiments.md's own §0 reframing means every extra candidate byte matters),
and (b) calibrated per-expert confidence, which is what E7's precision-gating
needs to decide how many bytes to spend on each candidate.

Method (Experiments.md E4 steps 1-5), all against already-collected b1
traces -- no GPU required, "trains in minutes on CPU/MPS" per the doc itself,
verified true on this machine (no CUDA at all):
  1. Predictor v2: a small per-expert-sigmoid MLP, one per (model, d, layer)
     -- see mlp.py.
  2. Ablate the input: ids / +gate weights / +router entropy / +hidden state
     -- see features.py. The hidden-state variant needs a GPU-collected file
     (collect_hidden_states.py) this machine doesn't have; it's skipped with
     an explicit message when absent, not silently dropped.
  3. Recall-vs-bytes curve: sweep m from k to min(4k, n_experts) on the
     winning variant, reusing already-trained models (no retraining per m).
  4. Calibration: ECE + reliability diagram + temperature scaling on the
     winning variant's held-out sigmoid probabilities -- see calibration.py.
  5. Cross-domain transfer: fit on one domain (chat/code), test on the
     other, and vice versa, using the winning variant at d=1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/

import features as feat  # noqa: E402
import mlp as mlpmod  # noqa: E402
import calibration as calib  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402
from common.trace_io import (  # noqa: E402
    REPO_ROOT, load_traces, train_test_split_reqs, build_transition_tables,
    explode_topk, expert_counts, recall_at_m as pilot_recall_at_m,
)

C_NONRESIDENT = 25.0
XDOMAIN_VARIANT = "ids_gate_entropy"  # doesn't need a hidden-state file, always available


def load_hidden_states(path: str):
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    obj = torch.load(p, map_location="cpu")
    return {k: v.numpy() for k, v in obj.items()}


def train_counts_by_layer(decode_train_df: pd.DataFrame):
    exploded = explode_topk(decode_train_df)
    counts = expert_counts(exploded)
    return {int(l): sub.set_index("expert")["count"] for l, sub in counts.groupby("layer")}


def ablation_sweep(decode_df, train_reqs, test_reqs, n_experts, n_layers, d, variants,
                    hidden_states, device, epochs, hidden_dim, seed, m):
    """Returns (rows, cache, skipped). cache[(variant, layer)] holds the
    trained model + cached test logits/Y/resident_mask so downstream steps
    (recall-vs-bytes, calibration) don't retrain."""
    tr_counts = train_counts_by_layer(decode_df[decode_df.req.isin(train_reqs)])
    rows, cache, skipped = [], {}, []

    for variant in variants:
        pooled = {"hit_o": 0, "tot_o": 0, "hit_n": 0, "tot_n": 0}
        n_layers_used, feature_dim = 0, None
        for layer in range(n_layers - d):
            try:
                train_ds = feat.build_layer_dataset(decode_df, layer, d, variant, n_experts,
                                                     reqs_filter=train_reqs, hidden_states=hidden_states)
                test_ds = feat.build_layer_dataset(decode_df, layer, d, variant, n_experts,
                                                    reqs_filter=test_reqs, hidden_states=hidden_states)
            except feat.MissingHiddenStateError as e:
                print(f"[e4] SKIPPING variant={variant}: {e}", flush=True)
                skipped.append(variant)
                break
            if train_ds is None or test_ds is None or len(train_ds.X) < 50 or len(test_ds.X) < 20:
                continue
            feature_dim = train_ds.feature_dim
            X_train, X_test = mlpmod.standardize_train_test(train_ds.X, test_ds.X)
            model = mlpmod.train_predictor(X_train, train_ds.Y, n_experts, hidden=hidden_dim,
                                            epochs=epochs, seed=seed, device=device)
            logits = mlpmod.predict_logits(model, X_test, device=device)
            res_mask = feat.resident_mask(tr_counts.get(layer + d, pd.Series(dtype=float)), n_experts, C_NONRESIDENT)
            hit_o, tot_o = mlpmod.topm_hits_and_total(logits, test_ds.Y, m)
            hit_n, tot_n = mlpmod.topm_hits_and_total(logits, test_ds.Y, m, candidate_mask=~res_mask)
            pooled["hit_o"] += hit_o
            pooled["tot_o"] += tot_o
            pooled["hit_n"] += hit_n
            pooled["tot_n"] += tot_n
            n_layers_used += 1
            cache[(variant, layer)] = {"model": model, "Y_test": test_ds.Y, "logits_test": logits,
                                        "resident_mask": res_mask}
        if variant in skipped:
            continue
        rows.append({
            "variant": variant, "d": d, "m": m,
            "recall_overall": pooled["hit_o"] / pooled["tot_o"] if pooled["tot_o"] else float("nan"),
            "recall_nonresident": pooled["hit_n"] / pooled["tot_n"] if pooled["tot_n"] else float("nan"),
            "n_layers_used": n_layers_used,
            "feature_dim": feature_dim,
            "param_count": mlpmod.param_count(mlpmod.TinyMLP(feature_dim, n_experts, hidden_dim)) if feature_dim else None,
        })
    return rows, cache, skipped


def baseline_frequency_table(decode_df, train_reqs, test_reqs, n_experts, k, n_layers, d, m):
    """Pilot's own predictor, unmodified -- reuses elp_probe.analyze.recall_at_m
    directly so the comparison is apples-to-apples with what the pilot report
    already claimed, not a reimplementation that could silently drift.

    m MUST be an exact multiple of k: pilot's recall_at_m takes an integer
    "mult" and computes m internally as `mult * k` (its own m_mults are
    always (1,"k")/(2,"2k")/(4,"4k")). Passing a non-multiple would force
    `mult = m/k` to be a float, and `order[:mult*k]` -- a numpy slice --
    raises TypeError on a float index even when it's numerically a whole
    number (e.g. 8.0), so this assertion turns that into a clear message
    instead of a confusing crash two calls deep in library code."""
    assert k > 0 and m % k == 0, f"m={m} must be an exact multiple of k={k} for the pilot's recall_at_m"
    mult = m // k
    tables = build_transition_tables(decode_df, train_reqs, n_experts, n_layers, [d])
    train_exp = explode_topk(decode_df[decode_df.req.isin(train_reqs)])
    train_counts = expert_counts(train_exp)
    res = pilot_recall_at_m(decode_df, test_reqs, tables, train_counts, n_experts, k, n_layers,
                             [d], [(mult, "m")], c_for_nonresident=C_NONRESIDENT)
    return res.get(f"d{d}_mm", {"recall_overall": float("nan"), "recall_nonresident": float("nan")})


def recall_vs_bytes(cache, variant, spec, m_grid, precision="fp16"):
    out = []
    for m in m_grid:
        pooled_hit, pooled_tot = 0, 0
        for (v, _layer), entry in cache.items():
            if v != variant:
                continue
            hit, tot = mlpmod.topm_hits_and_total(entry["logits_test"], entry["Y_test"], m,
                                                    candidate_mask=~entry["resident_mask"])
            pooled_hit += hit
            pooled_tot += tot
        recall = pooled_hit / pooled_tot if pooled_tot else float("nan")
        out.append({"m": int(m), "bytes_per_layer": m * spec.expert_bytes(precision),
                    "recall_nonresident": recall})
    return out


def calibration_for_variant(cache, variant):
    probs_all, labels_all = [], []
    for (v, _layer), entry in cache.items():
        if v != variant:
            continue
        probs_all.append(1 / (1 + np.exp(-entry["logits_test"])))
        labels_all.append(entry["Y_test"])
    if not probs_all:
        return None
    probs = np.concatenate(probs_all, axis=0).ravel()
    labels = np.concatenate(labels_all, axis=0).ravel()
    ece, bins = calib.expected_calibration_error(probs, labels)
    logits_flat = np.concatenate([entry["logits_test"] for (v, _l), entry in cache.items() if v == variant],
                                  axis=0).ravel()
    T = calib.fit_temperature(logits_flat, labels)
    probs_scaled = 1 / (1 + np.exp(-logits_flat / T))
    ece_after, bins_after = calib.expected_calibration_error(probs_scaled, labels)
    return {"ece_before": ece, "bins_before": bins, "temperature": T,
            "ece_after": ece_after, "bins_after": bins_after, "n_pairs": int(len(probs))}


def cross_domain_transfer(decode_df, n_experts, n_layers, variant, hidden_states, device, epochs,
                           hidden_dim, seed, m):
    directions = [("chat", "code"), ("code", "chat")]
    out = {}
    for src, dst in directions:
        src_reqs = set(decode_df[decode_df.domain == src]["req"].unique().tolist())
        dst_reqs = set(decode_df[decode_df.domain == dst]["req"].unique().tolist())
        if not src_reqs or not dst_reqs:
            out[f"{src}_to_{dst}"] = {"error": f"no requests found for domain(s) {src}/{dst}"}
            continue
        pooled = {"hit_n": 0, "tot_n": 0}
        tr_counts = train_counts_by_layer(decode_df[decode_df.req.isin(src_reqs)])
        for layer in range(n_layers - 1):
            train_ds = feat.build_layer_dataset(decode_df, layer, 1, variant, n_experts,
                                                 reqs_filter=src_reqs, hidden_states=hidden_states)
            test_ds = feat.build_layer_dataset(decode_df, layer, 1, variant, n_experts,
                                                reqs_filter=dst_reqs, hidden_states=hidden_states)
            if train_ds is None or test_ds is None or len(train_ds.X) < 50 or len(test_ds.X) < 20:
                continue
            X_train, X_test = mlpmod.standardize_train_test(train_ds.X, test_ds.X)
            model = mlpmod.train_predictor(X_train, train_ds.Y, n_experts, hidden=hidden_dim,
                                            epochs=epochs, seed=seed, device=device)
            logits = mlpmod.predict_logits(model, X_test, device=device)
            res_mask = feat.resident_mask(tr_counts.get(layer + 1, pd.Series(dtype=float)), n_experts, C_NONRESIDENT)
            hit, tot = mlpmod.topm_hits_and_total(logits, test_ds.Y, m, candidate_mask=~res_mask)
            pooled["hit_n"] += hit
            pooled["tot_n"] += tot
        out[f"{src}_to_{dst}"] = {
            "recall_nonresident": pooled["hit_n"] / pooled["tot_n"] if pooled["tot_n"] else float("nan"),
            "n_test_reqs": len(dst_reqs), "n_train_reqs": len(src_reqs),
        }
    return out


def plot_figures(model_tag, recall_bytes_rows, calib_result, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if recall_bytes_rows:
        fig, ax = plt.subplots(figsize=(5, 4))
        xs = [r["bytes_per_layer"] / 1e6 for r in recall_bytes_rows]
        ys = [r["recall_nonresident"] for r in recall_bytes_rows]
        ax.plot(xs, ys, marker="o")
        for r in recall_bytes_rows:
            ax.annotate(f"m={r['m']}", (r["bytes_per_layer"] / 1e6, r["recall_nonresident"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("bytes fetched per layer (MB, fp16, m candidates)")
        ax.set_ylabel("Recall@m, nonresident")
        ax.set_title(f"{model_tag}: recall vs bandwidth cost")
        path = out_dir / f"recall_vs_bytes_{model_tag}.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    if calib_result:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for bins, label, style in [(calib_result["bins_before"], "before T-scaling", "o-"),
                                    (calib_result["bins_after"], "after T-scaling", "s--")]:
            xs = [b["conf"] for b in bins if b["conf"] is not None]
            ys = [b["acc"] for b in bins if b["acc"] is not None]
            ax.plot(xs, ys, style, label=label)
        ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="perfectly calibrated")
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_ylabel("empirical accuracy (bin)")
        ax.set_title(f"{model_tag}: reliability diagram")
        ax.legend(fontsize=8)
        path = out_dir / f"reliability_{model_tag}.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="olmoe,mixtral")
    ap.add_argument("--b1-dirs", default="")
    ap.add_argument("--variants", default=",".join(feat.VARIANTS))
    ap.add_argument("--hidden-state-file", default="",
                     help="comma list model=path, e.g. olmoe=out/hidden_states_olmoe.pt "
                          "(from collect_hidden_states.py). Missing/absent -> ablation (d) skipped.")
    ap.add_argument("--hidden-dim", type=int, default=512, help="TinyMLP hidden width")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                     help="default cpu: for these tiny per-layer MLPs, CPU was faster in practice than MPS "
                          "(dispatch/transfer overhead dominates at this model size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall-bytes-points", type=int, default=8)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    ap.add_argument("--out-dir", default="experiments/e4_predictor/out")
    args = ap.parse_args()

    models = args.models.split(",")
    b1_overrides = {}
    for pair in args.b1_dirs.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            b1_overrides[k] = v
    hs_overrides = {}
    for pair in args.hidden_state_file.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            hs_overrides[k] = v

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")

    per_model = {}
    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        b1_dir = Path(b1_overrides.get(model_tag, REPO_ROOT / spec.trace_dir_default))
        if not (b1_dir / "traces.jsonl.zst").exists():
            print(f"[e4] ERROR: no traces at {b1_dir}/traces.jsonl.zst", flush=True)
            sys.exit(1)
        print(f"[e4] {model_tag}: loading traces...", flush=True)
        df = load_traces(b1_dir / "traces.jsonl.zst")
        decode_df = df[df.phase == "decode"].reset_index(drop=True)
        n_experts, n_layers, k = spec.n_experts, spec.n_layers, spec.k
        train_reqs, test_reqs = train_test_split_reqs(df, seed=args.seed)
        hidden_states = load_hidden_states(hs_overrides.get(model_tag, ""))
        if hidden_states is None:
            print(f"[e4] {model_tag}: no hidden-state file -- ablation (d) will be skipped "
                  f"(run collect_hidden_states.py --model {model_tag} on the GPU server, then pass "
                  f"--hidden-state-file {model_tag}=<path>)", flush=True)

        print(f"[e4] {model_tag}: ablation sweep (d=1, m=k={k}, {len(variants)} variants, "
              f"{n_layers-1} layer-transitions)...", flush=True)
        ablation_rows, cache, skipped = ablation_sweep(
            decode_df, train_reqs, test_reqs, n_experts, n_layers, 1, variants,
            hidden_states, args.device, args.epochs, args.hidden_dim, args.seed, k)

        print(f"[e4] {model_tag}: pilot frequency-table baseline (d=1, m=k)...", flush=True)
        baseline = baseline_frequency_table(decode_df, train_reqs, test_reqs, n_experts, k, n_layers, 1, k)

        valid_rows = [r for r in ablation_rows if not np.isnan(r["recall_nonresident"])]
        winner = max(valid_rows, key=lambda r: r["recall_nonresident"]) if valid_rows else None
        winning_variant = winner["variant"] if winner else None

        pp_gain = ((winner["recall_nonresident"] - baseline["recall_nonresident"]) * 100
                   if winner and not np.isnan(baseline.get("recall_nonresident", float("nan"))) else float("nan"))
        if np.isnan(pp_gain):
            verdict = "INCONCLUSIVE -- baseline or predictor recall unavailable (check trace/layer coverage)."
        elif pp_gain >= 8.0:
            verdict = (f"MLP EARNS ITS PLACE: {winning_variant} beats the frequency table by "
                       f"{pp_gain:+.1f}pp recall@m=k,nonresident (>= the 8pp exit-criteria bar).")
        else:
            verdict = (f"NEGATIVE RESULT (reported honestly, per Experiments.md's own exit criteria): "
                       f"best variant ({winning_variant}) only reaches {pp_gain:+.1f}pp over the frequency "
                       f"table -- the router's selected set already carries nearly all the available signal.")
        print(f"[e4] {model_tag} VERDICT: {verdict}", flush=True)

        m_grid = sorted(set(int(round(x)) for x in
                             np.linspace(k, min(4 * k, n_experts), args.recall_bytes_points)))
        rb_rows = recall_vs_bytes(cache, winning_variant, spec, m_grid, precision=args.precision) if winning_variant else []

        calib_result = calibration_for_variant(cache, winning_variant) if winning_variant else None

        print(f"[e4] {model_tag}: cross-domain transfer ({XDOMAIN_VARIANT}, d=1, m=k)...", flush=True)
        xdomain = cross_domain_transfer(decode_df, n_experts, n_layers, XDOMAIN_VARIANT, hidden_states,
                                         args.device, args.epochs, args.hidden_dim, args.seed, k)

        fig_paths = plot_figures(model_tag, rb_rows, calib_result, out_dir / "figures")

        per_model[model_tag] = {
            "n_layers": n_layers, "n_experts": n_experts, "k": k,
            "n_train_reqs": len(train_reqs), "n_test_reqs": len(test_reqs),
            "ablation_table": ablation_rows, "skipped_variants": skipped,
            "baseline_frequency_table": baseline,
            "winning_variant": winning_variant, "pp_gain_over_baseline": pp_gain, "verdict": verdict,
            "recall_vs_bytes": rb_rows, "precision_used": args.precision,
            "calibration": calib_result,
            "cross_domain_transfer": xdomain,
            "figures": fig_paths,
        }

    summary = {"models": models, "variants": variants, "hidden_dim": args.hidden_dim,
               "epochs": args.epochs, "device": args.device, "seed": args.seed,
               "c_nonresident_pct": C_NONRESIDENT, "per_model": per_model}
    summary_path = out_dir / "predictor_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[e4] wrote {summary_path}", flush=True)

    print("[e4] rendering RESULTS.md...", flush=True)
    from render_results_md import render
    render(summary_path, Path(__file__).resolve().parent / "RESULTS.md")


if __name__ == "__main__":
    main()
