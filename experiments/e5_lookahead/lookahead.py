#!/usr/bin/env python3
"""E5 -- Lookahead depth and pipelined prefetch (Experiments.md E5, [P1]).

One layer of lookahead buys one layer of compute time to hide a fetch. E1
shows that window is 50-100x too small for coarse experts. The only way to
extend it is deeper lookahead: predict layer l+d and start fetching d layers
early. But prediction accuracy decays with d, and NAIVELY chaining a
one-step predictor d times compounds that decay. This experiment measures
both the decay curve (direct predictors, one per d, not chained) and the
chained baseline, converts the result into a feasible-fetch-size curve (E1's
compute-time model + this experiment's recall-vs-depth), and simulates a
bounded prefetch queue to see how deep lookahead interacts with link
bandwidth.

No GPU required -- reuses E4's predictor/feature machinery directly
(features.py, mlp.py) against already-collected traces. "Effort 10h (reuses
E4 machinery)" per Experiments.md; verified true here too.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                  # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e4_predictor"))  # e4_predictor/

import features as feat  # noqa: E402
import mlp as mlpmod  # noqa: E402
import chained  # noqa: E402
import feasible_fetch as ff  # noqa: E402
import queue_sim  # noqa: E402
import roofline as rl  # noqa: E402 -- e1_roofline/ already on sys.path via feasible_fetch's own insert
from common.model_specs import MODEL_SPECS, LINK_BW_GBPS_GRID  # noqa: E402
from common.trace_io import REPO_ROOT, load_traces, train_test_split_reqs, explode_topk, expert_counts  # noqa: E402

DIRECT_VARIANT = "ids_gate_entropy"
C_NONRESIDENT = 25.0
RECALL_THRESHOLD = 0.60  # matches the pilot's own STRONG_GO threshold (Pilot.md verdict())


def train_counts_by_layer(decode_train_df):
    counts = expert_counts(explode_topk(decode_train_df))
    return {int(l): sub.set_index("expert")["count"] for l, sub in counts.groupby("layer")}


def direct_recall_for_depth(decode_df, train_reqs, test_reqs, n_experts, n_layers, d, tr_counts,
                             hidden_dim, epochs, seed, m):
    pooled_hit, pooled_tot = 0, 0
    n_layers_used = 0
    for layer in range(n_layers - d):
        train_ds = feat.build_layer_dataset(decode_df, layer, d, DIRECT_VARIANT, n_experts, reqs_filter=train_reqs)
        test_ds = feat.build_layer_dataset(decode_df, layer, d, DIRECT_VARIANT, n_experts, reqs_filter=test_reqs)
        if train_ds is None or test_ds is None or len(train_ds.X) < 50 or len(test_ds.X) < 20:
            continue
        X_train, X_test = mlpmod.standardize_train_test(train_ds.X, test_ds.X)
        model = mlpmod.train_predictor(X_train, train_ds.Y, n_experts, hidden=hidden_dim, epochs=epochs, seed=seed)
        logits = mlpmod.predict_logits(model, X_test)
        res_mask = feat.resident_mask(tr_counts.get(layer + d, pd.Series(dtype=float)), n_experts, C_NONRESIDENT)
        hit, tot = mlpmod.topm_hits_and_total(logits, test_ds.Y, m, candidate_mask=~res_mask)
        pooled_hit += hit
        pooled_tot += tot
        n_layers_used += 1
    return (pooled_hit / pooled_tot if pooled_tot else float("nan")), n_layers_used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="olmoe,mixtral")
    ap.add_argument("--b1-dirs", default="")
    ap.add_argument("--d-values", default="1,2,4,8")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall-threshold", type=float, default=RECALL_THRESHOLD)
    ap.add_argument("--fetch-batch", type=int, default=1, help="batch used for the feasible-fetch-size curve's t_compute")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    ap.add_argument("--queue-depths", default="1,2,4,8,16")
    ap.add_argument("--calib", default="", help="optional calib_compute.json from E1/E3")
    ap.add_argument("--out-dir", default="experiments/e5_lookahead/out")
    args = ap.parse_args()

    models = args.models.split(",")
    d_values = [int(x) for x in args.d_values.split(",")]
    q_grid = [int(x) for x in args.queue_depths.split(",")]
    b1_overrides = {}
    for pair in args.b1_dirs.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            b1_overrides[k] = v
    calib = None
    if args.calib and Path(args.calib).exists():
        calib = json.loads(Path(args.calib).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model = {}
    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        b1_dir = Path(b1_overrides.get(model_tag, REPO_ROOT / spec.trace_dir_default))
        if not (b1_dir / "traces.jsonl.zst").exists():
            print(f"[e5] ERROR: no traces at {b1_dir}/traces.jsonl.zst", flush=True)
            sys.exit(1)
        print(f"[e5] {model_tag}: loading traces...", flush=True)
        df = load_traces(b1_dir / "traces.jsonl.zst")
        decode_df = df[df.phase == "decode"].reset_index(drop=True)
        n_experts, n_layers, k = spec.n_experts, spec.n_layers, spec.k
        train_reqs, test_reqs = train_test_split_reqs(df, seed=args.seed)
        tr_counts = train_counts_by_layer(decode_df[decode_df.req.isin(train_reqs)])

        print(f"[e5] {model_tag}: direct predictors for d={d_values} ({DIRECT_VARIANT})...", flush=True)
        direct_recall, direct_layers_used = {}, {}
        for d in d_values:
            r, n_used = direct_recall_for_depth(decode_df, train_reqs, test_reqs, n_experts, n_layers, d,
                                                 tr_counts, args.hidden_dim, args.epochs, args.seed, k)
            direct_recall[d] = r
            direct_layers_used[d] = n_used
            print(f"[e5]   d={d}: direct recall@k,nonresident = {r:.3f} ({n_used} layer-transitions)", flush=True)

        print(f"[e5] {model_tag}: chained baseline (composing one d=1 'ids' model d times)...", flush=True)
        ids_models = chained.train_ids_d1_models(decode_df, train_reqs, n_experts, n_layers,
                                                  args.hidden_dim, args.epochs, args.seed)
        resident_mask_by_layer = {l + 1: feat.resident_mask(tr_counts.get(l + 1, pd.Series(dtype=float)), n_experts, C_NONRESIDENT)
                                   for l in range(n_layers - 1)}
        chained_recall = {}
        for d in d_values:
            res_mask_at_d = {l + d: feat.resident_mask(tr_counts.get(l + d, pd.Series(dtype=float)), n_experts, C_NONRESIDENT)
                              for l in range(n_layers - d)}
            r = chained.chained_recall(decode_df, test_reqs, ids_models, n_experts, k, n_layers, d,
                                        res_mask_at_d, m=k)
            chained_recall[d] = r
            print(f"[e5]   d={d}: chained recall@k,nonresident = {r:.3f}", flush=True)

        useful_d = ff.max_useful_depth(direct_recall, threshold=args.recall_threshold)
        feasible_rows = []
        for d in d_values:
            for bw in LINK_BW_GBPS_GRID:
                bytes_fetchable, prov = ff.feasible_bytes_at_depth(model_tag, d, args.fetch_batch, bw, calib)
                feasible_rows.append({"d": d, "bw_gbps": bw, "bytes_fetchable": bytes_fetchable,
                                       "compute_provenance": prov,
                                       "expert_bytes": spec.expert_bytes(args.precision)})

        t_layer_ms, compute_prov = rl.get_t_compute_ms(spec, args.fetch_batch, calib)
        expert_total_bytes_worstcase = spec.expert_bytes(args.precision) * spec.k  # batch=1 -> e_union==k exactly
        queue_rows = {}
        for bw in LINK_BW_GBPS_GRID:
            service_ms = expert_total_bytes_worstcase / (bw * 1e6)
            queue_rows[bw] = queue_sim.sweep_queue_depth(t_layer_ms, service_ms, q_grid)

        per_model[model_tag] = {
            "n_layers": n_layers, "n_experts": n_experts, "k": k,
            "direct_recall_by_d": direct_recall, "direct_layers_used": direct_layers_used,
            "chained_recall_by_d": chained_recall,
            "max_useful_depth": useful_d, "recall_threshold": args.recall_threshold,
            "feasible_fetch": feasible_rows,
            "queue_sim": {"t_layer_ms": t_layer_ms, "compute_provenance": compute_prov,
                          "expert_total_bytes_worstcase": expert_total_bytes_worstcase,
                          "by_bw": {str(bw): {str(q): v for q, v in rows.items()} for bw, rows in queue_rows.items()}},
        }

    summary = {"models": models, "d_values": d_values, "hidden_dim": args.hidden_dim, "epochs": args.epochs,
               "seed": args.seed, "fetch_batch": args.fetch_batch, "precision": args.precision,
               "queue_depths": q_grid, "per_model": per_model}
    summary_path = out_dir / "lookahead_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[e5] wrote {summary_path}", flush=True)

    print("[e5] plotting figures...", flush=True)
    from plotting import plot_recall_vs_depth, plot_feasible_fetch_size
    fig_paths = []
    for model_tag, m in per_model.items():
        fig_paths += plot_recall_vs_depth(model_tag, m["direct_recall_by_d"], m["chained_recall_by_d"],
                                           out_dir / "figures")
        fig_paths += plot_feasible_fetch_size(model_tag, m["feasible_fetch"], m["direct_recall_by_d"],
                                               out_dir / "figures")
    summary["figures"] = fig_paths
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("[e5] rendering RESULTS.md...", flush=True)
    # Re-insert this file's own directory at the FRONT of sys.path right
    # before this import: e1_roofline/ and e4_predictor/ are also on
    # sys.path (for `roofline`/`features`/`mlp`) and each has its OWN
    # render_results_md.py -- without this, `import render_results_md` picks
    # up whichever of those got inserted last, not this file's own module
    # (this is the exact bug E1/E2 already avoid via the same re-insert
    # pattern immediately before their own equivalent import).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_results_md import render
    render(summary_path, Path(__file__).resolve().parent / "RESULTS.md")


if __name__ == "__main__":
    main()
