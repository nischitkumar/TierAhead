#!/usr/bin/env python3
"""E2 -- Batching sweet spot: locality erosion vs arithmetic intensity
(Experiments.md E2, [P1]).

E1 says bigger batches help hide latency (more compute per byte). The
pilot's batch-8 erosion check said bigger batches hurt locality (wider
expert union). This experiment finds the crossover: B* that minimizes
per-token time, and whether that optimum is a narrow or wide window, as a
function of expert granularity (n_experts/layer).

Method (Experiments.md E2 steps 1-4), all against already-collected b1
(single-request) decode traces -- "you do not need to re-run the models --
sample B token-steps from distinct requests, take the union of selected
experts per layer" -- so, like E1, no GPU or network is required to run
this script itself.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1_roofline"))  # e1_roofline/

from common.model_specs import MODEL_SPECS, BATCH_GRID_E2  # noqa: E402
from common.eunion import e_union_curve, analytic_e_union_uniform  # noqa: E402
from common.trace_io import (  # noqa: E402
    load_decode_df, load_traces, train_test_split_reqs, expert_counts,
    explode_topk, build_transition_tables,
)
import roofline as rl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M_MULTS = [(1, "k"), (2, "2k")]  # matches pilot; 4k flagged degenerate for small n_experts models


# ---------- steps 1-2: E_union(B) and bytes/token ----------

def e_union_by_batch_all_layers(model_tag: str, b1_dir: Path, batch_grid, n_trials=500, seed=0):
    decode_df = load_decode_df(b1_dir)
    n_layers = int(decode_df["layer"].max()) + 1
    per_layer = {}
    for layer in range(n_layers):
        curve = e_union_curve(decode_df, layer, batch_grid, n_trials=n_trials, seed=seed)
        per_layer[layer] = {b: est.mean for b, est in curve.items()}
    mean_by_batch = {b: float(np.mean([per_layer[l][b] for l in per_layer])) for b in batch_grid}
    return mean_by_batch, per_layer, n_layers


def bytes_per_token(spec, batch: int, residency_pct: float, precision: str, e_union_val: float) -> float:
    """The key E2 quantity: fetched bytes amortized across every token in the
    batch that needed the fetch. bytes_per_token = E_union(B)*size*(1-r)/B."""
    return rl.bytes_per_layer(spec, batch, residency_pct, precision, e_union_val) / batch


# ---------- step 3: rho(B), B* ----------
#
# Two different quantities matter here and must NOT be conflated (an earlier
# version of this file did, by computing total_tpot_ms from the already-
# per-token-divided time -- caught by inspecting a real run's numbers, where
# it made total_tpot monotonically DEcrease with batch, so an SLO could never
# bind):
#   - per-token time (THROUGHPUT): batch-step time / B. Amortizes a fetch
#     across every token in the batch that needed it -- this is what E2 wants
#     to minimize, and it improves monotonically with B (more compute/byte
#     per token, per E1's reframing).
#   - total_tpot_ms (LATENCY): the batch-step time itself, NOT divided by B.
#     Every request in the batch waits for the same layer forward pass to
#     finish regardless of batch size, so this is the actual wall-clock time
#     between successive tokens any individual request experiences -- and it
#     gets WORSE as B grows (more tokens' worth of compute/bytes packed into
#     one step). This is the real erosion cost that bounds how large a batch
#     a latency-SLO'd deployment can accumulate before dispatching, and it is
#     what a p99 TPOT SLO must constrain.

def batch_step_components_ms(spec, batch: int, residency_pct: float, precision: str, bw_gbps: float,
                              e_union_val: float, calib=None):
    """Undivided (whole-batch) compute and transfer time for one layer's
    decode step. Returns (t_compute_batch_ms, t_transfer_batch_ms, provenance)."""
    t_compute_batch_ms, provenance = rl.get_t_compute_ms(spec, batch, calib)
    bytes_layer = rl.bytes_per_layer(spec, batch, residency_pct, precision, e_union_val)  # NOT /batch
    t_transfer_batch_ms = rl.t_transfer_ms(bytes_layer, bw_gbps)
    return t_compute_batch_ms, t_transfer_batch_ms, provenance


def per_token_time_ms(spec, batch: int, residency_pct: float, precision: str, bw_gbps: float,
                       e_union_val: float, calib=None):
    """Per-layer, per-token time (THROUGHPUT metric): max(compute-bound,
    transfer-bound) batch-step time, divided by batch. Matches the roofline's
    rho framing -- when rho<1 the fetch hides behind compute; when rho>=1
    it's transfer-bound. rho itself is scale-invariant to the /batch (both
    terms divide equally), so it's identical whether computed here or from
    the undivided batch-step components."""
    t_compute_batch_ms, t_transfer_batch_ms, provenance = batch_step_components_ms(
        spec, batch, residency_pct, precision, bw_gbps, e_union_val, calib)
    t_compute_token_ms = t_compute_batch_ms / batch
    t_transfer_token_ms = t_transfer_batch_ms / batch
    return max(t_compute_token_ms, t_transfer_token_ms), t_compute_token_ms, t_transfer_token_ms, provenance


def find_b_star(spec, batch_grid, residency_pct, precision, bw_gbps, e_union_by_batch, calib=None,
                 slo_ms: float | None = None):
    """B* = argmin per-token (throughput) time over batch_grid, subject to an
    optional p99 TPOT SLO on total_tpot_ms (n_layers * undivided batch-step
    time -- the actual per-request latency, see module note above). slo_ms
    is a configurable assumption (Experiments.md doesn't pin a number) --
    pass None to disable the constraint and just report the unconstrained
    throughput-argmin alongside every candidate's feasibility.
    """
    rows = []
    for batch in batch_grid:
        tc_batch_ms, tt_batch_ms, prov = batch_step_components_ms(
            spec, batch, residency_pct, precision, bw_gbps, e_union_by_batch[batch], calib)
        pt_ms = max(tc_batch_ms, tt_batch_ms) / batch
        total_tpot_ms = spec.n_layers * max(tc_batch_ms, tt_batch_ms)
        feasible = slo_ms is None or total_tpot_ms <= slo_ms
        rows.append({"batch": batch, "per_token_time_ms": pt_ms,
                     "compute_component_ms": tc_batch_ms / batch, "transfer_component_ms": tt_batch_ms / batch,
                     "total_tpot_ms": total_tpot_ms,
                     "feasible_under_slo": feasible, "compute_provenance": prov,
                     "rho": rl.rho(tt_batch_ms, tc_batch_ms)})
    df = pd.DataFrame(rows)
    feasible_df = df[df.feasible_under_slo] if slo_ms is not None else df
    if len(feasible_df) == 0:
        b_star_row = df.loc[df["per_token_time_ms"].idxmin()]
        note = f"NO batch satisfies the {slo_ms} ms p99 TPOT SLO -- reporting unconstrained argmin instead"
    else:
        b_star_row = feasible_df.loc[feasible_df["per_token_time_ms"].idxmin()]
        note = f"argmin per-token time among batches satisfying the {slo_ms} ms SLO" if slo_ms is not None \
            else "unconstrained argmin (no SLO given)"
    return int(b_star_row["batch"]), df, note


# ---------- step 4: predictor recall for batched targets (new metric) ----------

def batched_recall(df: pd.DataFrame, train_reqs: set, test_reqs: set, n_experts: int, k: int,
                    n_layers: int, batch_grid, m_mults=DEFAULT_M_MULTS, d: int = 1,
                    c_for_nonresident: float = 25.0, n_trials: int = 200, seed: int = 0):
    """Recall@m for the UNION of experts a synthetic batch of B independent
    requests needs at layer l+d, given each request's own top-m prediction at
    layer l+d (per-request ranking, batch-shared prefetch set). This is the
    metric Experiments.md E2 step 4 calls out as new: 'recall degrades when
    you must predict a union of experts rather than top-k for one token.'

    IMPORTANT interpretation note (found while testing this function, not in
    the original design doc): raw recall_overall is NOT directly comparable
    across batch sizes on its own. Each of the B rows contributes its own
    top-m candidate set, so the aggregate candidate pool (pred_union) grows
    with B mechanically -- a bigger "net" catches more fish even with zero
    predictive skill. Every result entry therefore also reports
    baseline_random_batched = E[|union of B iid uniform m-subsets|]/n_experts
    (via common.eunion.analytic_e_union_uniform), the batch-scaled analogue
    of the pilot's baseline_random = m/n_experts. Compare recall_overall
    AGAINST this baseline (the "lift"), not against the m=1,batch=1 number,
    exactly as the pilot compared its predictor against static-popularity
    and random rather than reading recall in isolation.
    """
    decode = df[df.phase == "decode"]
    tables = build_transition_tables(df, train_reqs, n_experts, n_layers, [d])
    test = decode[decode.req.isin(test_reqs)]
    pivot = test.pivot_table(index=["req", "tok"], columns="layer", values="topk", aggfunc="first")
    train_exp = explode_topk(decode[decode.req.isin(train_reqs)])
    train_counts = expert_counts(train_exp)

    resident_by_layer = {}
    for layer, sub in train_counts.groupby("layer"):
        cnt = sub.set_index("expert")["count"].reindex(range(n_experts), fill_value=0)
        top_n = max(1, int(np.ceil(n_experts * c_for_nonresident / 100.0)))
        resident_by_layer[int(layer)] = set(cnt.sort_values(ascending=False).index[:top_n].tolist())

    rng = np.random.default_rng(seed)
    ms = [(min(mult * k, n_experts), name) for mult, name in m_mults]
    results = {}

    for m, name in ms:
        for batch in batch_grid:
            hit_overall, tot_overall, hit_nonres, tot_nonres = 0, 0, 0, 0
            for layer in range(n_layers - d):
                if layer not in tables[d] or layer not in pivot.columns or (layer + d) not in pivot.columns:
                    continue
                log_p = np.log(tables[d][layer])
                pairs = pivot[[layer, layer + d]].dropna()
                n_rows = len(pairs)
                if n_rows == 0:
                    continue
                b = min(batch, n_rows)
                resident = resident_by_layer.get(layer + d, set())
                s_l_all = pairs[layer].to_numpy()
                s_ld_all = pairs[layer + d].to_numpy()

                for _ in range(n_trials):
                    idx = rng.choice(n_rows, size=b, replace=False)
                    pred_union, actual_union = set(), set()
                    for i in idx:
                        scores = log_p[list(s_l_all[i]), :].sum(axis=0)
                        top_m = set(np.argsort(-scores)[:m].tolist())
                        pred_union |= top_m
                        actual_union |= set(s_ld_all[i])
                    nonres_actual = actual_union - resident
                    hit_overall += len(pred_union & actual_union)
                    tot_overall += len(actual_union)
                    hit_nonres += len(pred_union & nonres_actual)
                    tot_nonres += len(nonres_actual)

            baseline_batched = analytic_e_union_uniform(n_experts, m, batch) / n_experts
            recall_overall_val = hit_overall / tot_overall if tot_overall else float("nan")
            results[f"m{name}_B{batch}"] = {
                "m": m, "batch": batch,
                "recall_overall": recall_overall_val,
                "recall_nonresident": hit_nonres / tot_nonres if tot_nonres else float("nan"),
                "baseline_random_batched": baseline_batched,
                "lift_over_random_batched": (recall_overall_val - baseline_batched
                                              if not np.isnan(recall_overall_val) else float("nan")),
            }
    return results


# ---------- plotting ----------

def plot_e2_figures(model_tag, e_union_mean, bstar_table_df, batched_recall_res, batch_grid, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax1 = plt.subplots(figsize=(5, 4))
    bs = sorted(e_union_mean.keys())
    ax1.plot(bs, [e_union_mean[b] for b in bs], marker="o", color="tab:blue", label="E_union(B)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch B")
    ax1.set_ylabel("mean distinct experts/layer", color="tab:blue")
    ax1.set_title(f"{model_tag}: batching sweet spot")
    ax2 = ax1.twinx()
    ax2.plot(bstar_table_df["batch"], bstar_table_df["per_token_time_ms"], marker="s",
             color="tab:red", label="per-token time (ms)")
    ax2.set_ylabel("per-token time (ms)", color="tab:red")
    ax2.set_yscale("log")
    b_star = bstar_table_df.loc[bstar_table_df["per_token_time_ms"].idxmin(), "batch"]
    ax1.axvline(b_star, color="black", linestyle="--", alpha=0.6, label=f"B*={b_star}")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    path = out_dir / f"batch_sweetspot_{model_tag}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    if batched_recall_res:
        fig2, ax = plt.subplots(figsize=(5, 4))
        by_m = {}
        for key, v in batched_recall_res.items():
            by_m.setdefault(v["m"], []).append((v["batch"], v["recall_nonresident"]))
        for m, pts in by_m.items():
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"m={m}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch B")
        ax.set_ylabel("Recall@m, nonresident (batched-union target)")
        ax.set_title(f"{model_tag}: recall degrades as batch widens the target union")
        ax.legend()
        path2 = out_dir / f"recall_vs_batch_{model_tag}.pdf"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        written.append(str(path2))

    return written


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="olmoe,mixtral")
    ap.add_argument("--b1-dirs", default="")
    ap.add_argument("--calib", default="", help="optional calib_compute.json from E1's measure_compute.py")
    ap.add_argument("--residency-pct", type=float, default=25.0, help="fixed r for the rho(B)/B* analysis")
    ap.add_argument("--bw-gbps", type=float, default=64.0, help="fixed link BW for the rho(B)/B* analysis")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    ap.add_argument("--slo-ms", type=float, default=None,
                     help="optional p99 TPOT SLO (ms) constraining B*; unset = unconstrained argmin")
    ap.add_argument("--n-trials", type=int, default=500, help="E_union(B) resampling trials")
    ap.add_argument("--recall-n-trials", type=int, default=200, help="batched-recall resampling trials")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-recall", action="store_true", help="skip the (slower) step-4 batched recall metric")
    ap.add_argument("--out-dir", default="experiments/e2_batching/out")
    args = ap.parse_args()

    models = args.models.split(",")
    b1_overrides = {}
    for pair in args.b1_dirs.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            b1_overrides[k] = v

    calib = None
    if args.calib and Path(args.calib).exists():
        calib = json.loads(Path(args.calib).read_text())
        print(f"[e2] loaded compute calibration from {args.calib}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model = {}
    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        b1_dir = Path(b1_overrides.get(model_tag, REPO_ROOT / spec.trace_dir_default))
        if not (b1_dir / "traces.jsonl.zst").exists():
            print(f"[e2] ERROR: no traces at {b1_dir}/traces.jsonl.zst", flush=True)
            sys.exit(1)

        print(f"[e2] {model_tag}: computing E_union(B) for B={BATCH_GRID_E2} (resampling, no GPU)...", flush=True)
        e_union_mean, e_union_per_layer, n_layers = e_union_by_batch_all_layers(
            model_tag, b1_dir, BATCH_GRID_E2, n_trials=args.n_trials, seed=args.seed)
        print(f"[e2]   E_union(B) = {e_union_mean}", flush=True)

        bpt = {b: bytes_per_token(spec, b, args.residency_pct, args.precision, e_union_mean[b])
               for b in BATCH_GRID_E2}

        print(f"[e2] {model_tag}: finding B* at r={args.residency_pct}% bw={args.bw_gbps}GB/s "
              f"precision={args.precision}...", flush=True)
        b_star, bstar_df, bstar_note = find_b_star(
            spec, BATCH_GRID_E2, args.residency_pct, args.precision, args.bw_gbps,
            e_union_mean, calib=calib, slo_ms=args.slo_ms)
        print(f"[e2]   B*={b_star} ({bstar_note})", flush=True)

        batched_recall_res = {}
        if not args.skip_recall:
            print(f"[e2] {model_tag}: predictor recall for batched targets "
                  f"(this is the slow step, {args.recall_n_trials} trials/layer/batch)...", flush=True)
            full_df = load_traces(b1_dir / "traces.jsonl.zst")
            train_reqs, test_reqs = train_test_split_reqs(full_df, seed=args.seed)
            n_experts = spec.n_experts
            batched_recall_res = batched_recall(
                full_df, train_reqs, test_reqs, n_experts, spec.k, n_layers, BATCH_GRID_E2,
                n_trials=args.recall_n_trials, seed=args.seed)

        fig_paths = plot_e2_figures(model_tag, e_union_mean, bstar_df, batched_recall_res, BATCH_GRID_E2,
                                     out_dir / "figures")

        bstar_df.to_csv(out_dir / f"bstar_table_{model_tag}.csv", index=False)
        per_model[model_tag] = {
            "n_layers": n_layers,
            "e_union_by_batch": e_union_mean,
            "bytes_per_token": bpt,
            "b_star": b_star,
            "b_star_note": bstar_note,
            "b_star_table": bstar_df.to_dict(orient="records"),
            "batched_recall": batched_recall_res,
            "figures": fig_paths,
            "residency_pct_used": args.residency_pct,
            "bw_gbps_used": args.bw_gbps,
            "precision_used": args.precision,
            "slo_ms_used": args.slo_ms,
        }

    # cross-model claim check: "fine-grained MoEs have a wide viable batch
    # window, coarse MoEs have none" -- auto-derived, not asserted
    viable_window = {}
    for model_tag, res in per_model.items():
        n_viable = sum(1 for row in res["b_star_table"] if row["rho"] < 1.0)
        viable_window[model_tag] = {"n_viable_batches_of_grid": n_viable, "grid_size": len(BATCH_GRID_E2)}

    claim_text = _render_claim(per_model, viable_window)

    summary = {
        "models": models,
        "batch_grid": BATCH_GRID_E2,
        "m_mults": DEFAULT_M_MULTS,
        "per_model": per_model,
        "viable_batch_window": viable_window,
        "claim_check": claim_text,
        "n_trials": args.n_trials,
        "recall_n_trials": args.recall_n_trials,
        "seed": args.seed,
    }
    summary_path = out_dir / "batching_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[e2] wrote {summary_path}", flush=True)

    print("[e2] rendering RESULTS.md...", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_results_md import render  # noqa: E402
    render(summary_path, Path(__file__).resolve().parent / "RESULTS.md")


def _render_claim(per_model, viable_window):
    n_experts_by_model = {m: MODEL_SPECS[m].n_experts for m in per_model}
    ranked = sorted(n_experts_by_model.items(), key=lambda kv: -kv[1])  # most fine-grained first
    lines = []
    for model_tag, n_experts in ranked:
        vw = viable_window[model_tag]
        lines.append(f"{model_tag} ({n_experts} experts/layer): {vw['n_viable_batches_of_grid']}/"
                     f"{vw['grid_size']} swept batch sizes land in the latency-hideable "
                     f"(rho<1) regime at the analysis's fixed (r,bw,precision).")
    if len(ranked) >= 2:
        fine, coarse = ranked[0], ranked[-1]
        fine_n = viable_window[fine[0]]["n_viable_batches_of_grid"]
        coarse_n = viable_window[coarse[0]]["n_viable_batches_of_grid"]
        if fine_n > coarse_n:
            verdict = (f"CONFIRMED at these settings: {fine[0]} (finer-grained) has a wider viable "
                       f"batch window ({fine_n} pts) than {coarse[0]} (coarser) ({coarse_n} pts).")
        elif fine_n == coarse_n:
            verdict = (f"INCONCLUSIVE at these settings: both models show {fine_n} viable batch "
                       f"points -- try a different (r, bw, precision) operating point before concluding.")
        else:
            verdict = (f"REFUTED at these settings: {coarse[0]} (coarser) has a wider viable batch "
                       f"window ({coarse_n} pts) than {fine[0]} (finer) ({fine_n} pts).")
        lines.append(verdict)
    return lines


if __name__ == "__main__":
    main()
