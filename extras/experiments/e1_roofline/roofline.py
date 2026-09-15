#!/usr/bin/env python3
"""E1 -- The MoE Tiering Roofline (Experiments.md E1, [P0]).

Answers: for a given (model, batch B, residency r, precision p, link BW),
is expert traffic latency-bound (prefetch can fully hide the fetch) or
bandwidth-bound (only byte reduction helps)?

    rho = t_transfer(B,r,p,BW) / t_compute(B)      rho < 1 -> latency-bound
                                                    rho >= 1 -> bandwidth-bound

t_compute has two possible sources, both written to the output with an
explicit provenance tag (this table-with-provenance is the same discipline
E3 uses, and the reason this experiment is defensible without a full
hardware calibration ladder having run first):
  - "measured_here": from calib_compute.json (measure_compute.py's real
    CUDA-event timings on the target 2x RTX 4060 Ti box), when available.
  - "flop_estimate": analytical 2*B*k*params_per_expert / (peak_flops*mfu),
    used per Experiments.md E1 step 2 ("use a FLOP-based estimate until E3
    lands") when no calibration file is supplied.

No GPU or network required to run this script itself (Experiments.md marks
E1 "No GPU"); measure_compute.py is the separate, optional GPU step that
upgrades the compute-time input from estimated to measured.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.model_specs import (  # noqa: E402
    MODEL_SPECS, LINK_BW_GBPS_GRID, RESIDENCY_PCT_GRID, BATCH_GRID_E1,
    PRECISION_GRID, RTX_4060TI_PEAK_FP16_FLOPS, DEFAULT_DECODE_MFU,
)
from common.eunion import e_union_curve, analytic_e_union_uniform  # noqa: E402
from common.trace_io import load_decode_df  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------- compute-time model ----------

def t_compute_flop_ms(spec, batch: int, peak_flops: float = RTX_4060TI_PEAK_FP16_FLOPS,
                       mfu: float = DEFAULT_DECODE_MFU) -> float:
    """Analytical per-layer compute time (ms) for a batch of `batch` decode
    tokens: 2*B*k*params_per_expert FLOPs (multiply-add over the k routed
    experts' 3 GEMMs), divided by achieved FLOPs/s."""
    flops = 2.0 * batch * spec.k * spec.params_per_expert
    seconds = flops / (peak_flops * mfu)
    return seconds * 1e3


def t_compute_from_calib(calib: dict, model_tag: str, batch: int):
    """Exact lookup into measure_compute.py's output. Returns ms or None."""
    entry = (calib or {}).get(model_tag, {}).get("by_batch", {}).get(str(batch))
    if entry is None:
        return None
    return entry["mean_ms"]


def get_t_compute_ms(spec, batch: int, calib: dict | None = None):
    """Returns (t_compute_ms, provenance_str)."""
    if calib is not None:
        measured = t_compute_from_calib(calib, spec.tag, batch)
        if measured is not None:
            return measured, "measured_here"
    return t_compute_flop_ms(spec, batch), "flop_estimate"


# ---------- transfer-time / bytes model ----------

def bytes_per_layer(spec, batch: int, residency_pct: float, precision: str,
                     e_union_val: float) -> float:
    """bytes_per_layer(B,r,p) = E_union(B) * expert_size(p) * (1 - r)."""
    return e_union_val * spec.expert_bytes(precision) * (1.0 - residency_pct / 100.0)


def t_transfer_ms(bytes_val: float, bw_gbps: float) -> float:
    if bw_gbps <= 0:
        return float("inf")
    seconds = bytes_val / (bw_gbps * 1e9)
    return seconds * 1e3


def rho(t_transfer_val_ms: float, t_compute_val_ms: float) -> float:
    if t_compute_val_ms <= 0:
        return float("inf")
    return t_transfer_val_ms / t_compute_val_ms


def required_bw_gbps(bytes_val: float, t_compute_val_ms: float, target_rho: float = 1.0) -> float:
    """Exact minimum link bandwidth (continuous, not grid-quantized) at which
    rho == target_rho, i.e. the transfer would just finish as compute does."""
    if t_compute_val_ms <= 0:
        return float("inf")
    t_compute_s = t_compute_val_ms / 1e3
    return bytes_val / (target_rho * t_compute_s) / 1e9


# ---------- E_union lookup per model (mean over layers, per Experiments.md's
# "measured expected number of distinct experts activated per layer") ----------

def build_e_union_lookup(model_tag: str, b1_dir: Path, batch_grid, n_trials=500, seed=0):
    """Returns {batch: mean_e_union_across_layers}, plus the full per-layer
    detail for the figure/summary json."""
    decode_df = load_decode_df(b1_dir)
    n_layers = int(decode_df["layer"].max()) + 1
    per_layer = {}
    for layer in range(n_layers):
        curve = e_union_curve(decode_df, layer, batch_grid, n_trials=n_trials, seed=seed)
        per_layer[layer] = {b: est.mean for b, est in curve.items()}
    mean_by_batch = {b: float(np.mean([per_layer[l][b] for l in per_layer])) for b in batch_grid}
    return mean_by_batch, per_layer


# ---------- sweep ----------

def sweep_grid(models, batch_grid, residency_grid, precision_grid, bw_grid,
                e_union_lookups: dict, calib: dict | None = None) -> pd.DataFrame:
    rows = []
    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        e_union_by_batch = e_union_lookups[model_tag]
        for batch in batch_grid:
            t_compute, provenance = get_t_compute_ms(spec, batch, calib)
            e_union_val = e_union_by_batch[batch]
            for precision in precision_grid:
                for residency_pct in residency_grid:
                    bpl = bytes_per_layer(spec, batch, residency_pct, precision, e_union_val)
                    req_bw = required_bw_gbps(bpl, t_compute)
                    for bw in bw_grid:
                        tt = t_transfer_ms(bpl, bw)
                        rows.append({
                            "model": model_tag, "batch": batch, "residency_pct": residency_pct,
                            "precision": precision, "bw_gbps": bw,
                            "e_union": e_union_val, "bytes_per_layer": bpl,
                            "t_transfer_ms": tt, "t_compute_ms": t_compute,
                            "compute_provenance": provenance,
                            "rho": rho(tt, t_compute),
                            "required_bw_gbps_for_rho1": req_bw,
                        })
    return pd.DataFrame(rows)


def provisioning_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per Experiments.md E1 deliverable: 'for each model class, minimum GB/s
    per accelerator and minimum residency % to keep rho<1 at batch {1,8,32}'.
    One row per (model, precision, batch, residency_pct); required_bw_gbps_for_rho1
    is the exact continuous minimum (nearest supported grid tier alongside it)."""
    grid_bw = sorted(set(df["bw_gbps"]))

    def nearest_tier(req):
        # NaN (not None): keeps the column a consistent float64 dtype
        # regardless of row count/mix, so downstream consumers (JSON,
        # render_results_md) only ever need to check for NaN, not both
        # None and NaN depending on how pandas happened to upcast the column.
        feasible = [b for b in grid_bw if b >= req]
        return float(min(feasible)) if feasible else float("nan")

    cols = ["model", "precision", "batch", "residency_pct", "compute_provenance",
            "required_bw_gbps_for_rho1"]
    out = df[cols].drop_duplicates().sort_values(["model", "precision", "batch", "residency_pct"])
    out = out.reset_index(drop=True)
    out["nearest_supported_link_gbps"] = out["required_bw_gbps_for_rho1"].apply(nearest_tier)
    return out


def headline_sentences(df: pd.DataFrame) -> dict:
    """Auto-derive the E1 exit-criteria sentence per model+precision: the
    easiest (highest-batch, highest-residency-in-grid) viable operating point,
    or an explicit statement that byte reduction (not prefetch) is required."""
    out = {}
    max_batch = df["batch"].max()
    max_r = df["residency_pct"].max()
    for (model, precision), sub in df.groupby(["model", "precision"]):
        best = sub[(sub.batch == max_batch) & (sub.residency_pct == max_r)]
        best = best.sort_values("bw_gbps").iloc[0] if len(best) else None
        # find the minimal-requirement viable point across the whole grid
        viable = sub[sub["rho"] < 1.0].sort_values(["bw_gbps", "residency_pct", "batch"])
        key = f"{model}_{precision}"
        if len(viable):
            v = viable.iloc[0]
            out[key] = (
                f"{model} ({precision}): router-guided prefetch is worth something (rho<1) "
                f"once residency >= {v.residency_pct:g}% on a >={v.bw_gbps:g} GB/s link at "
                f"batch >= {v.batch:g} (expert ~{MODEL_SPECS[model].expert_bytes(precision)/1e6:.1f} MB, "
                f"t_compute source: {v.compute_provenance})."
            )
        else:
            req = best["required_bw_gbps_for_rho1"] if best is not None else float("nan")
            out[key] = (
                f"{model} ({precision}): NOT reachable anywhere in the swept grid "
                f"(batch<=64, residency<=75%, link<=128 GB/s) -- even at the most favorable "
                f"point tested this regime needs ~{req:.0f} GB/s to hit rho=1. "
                f"Byte reduction (E7 precision-gating), not prefetch alone, is required here."
            )
    return out


# ---------- achievable-points overlay (pilot's real measured Coverage@C) ----------

def load_pilot_achievable_points(model_tag: str):
    """Pulls the pilot's real measured Coverage@C (at batch=1) so the roofline
    figure can mark 'this residency level is not hypothetical, we measured
    this coverage on held-out requests' per Experiments.md E1 step 5."""
    path_map = {
        "olmoe": REPO_ROOT / "results" / "pilot_summary.json",
        "mixtral": REPO_ROOT / "results" / "mixtral_summary.json",
    }
    p = path_map.get(model_tag)
    if p is None or not p.exists():
        return []
    summary = json.loads(p.read_text())
    points = []
    for c_str, vals in summary.get("coverage_at_c", {}).items():
        points.append({"residency_pct": float(c_str), "deployable_coverage": vals["deployable"],
                        "oracle_coverage": vals["oracle"]})
    return points


# ---------- plotting ----------

def plot_rho_heatmaps(df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in sorted(df["model"].unique()):
        for precision in sorted(df["precision"].unique()):
            sub = df[(df.model == model) & (df.precision == precision)]
            bws = sorted(sub["bw_gbps"].unique())
            fig, axes = plt.subplots(1, len(bws), figsize=(4 * len(bws), 4), squeeze=False)
            achievable = load_pilot_achievable_points(model)
            for ax, bw in zip(axes[0], bws):
                cell = sub[sub.bw_gbps == bw]
                pivot = cell.pivot(index="residency_pct", columns="batch", values="rho")
                pivot = pivot.sort_index()
                r_vals = pivot.index.to_numpy(dtype=float)
                b_vals = pivot.columns.to_numpy(dtype=float)
                z = np.clip(pivot.to_numpy(), 1e-3, 1e6)
                mesh = ax.pcolormesh(b_vals, r_vals, z, norm=LogNorm(vmin=1e-2, vmax=1e3),
                                      cmap="RdBu_r", shading="nearest")
                try:
                    ax.contour(b_vals, r_vals, z, levels=[1.0], colors="black", linewidths=2)
                except Exception:
                    pass
                for pt in achievable:
                    ax.scatter([1], [pt["residency_pct"]], marker="*", s=120,
                               color="lime", edgecolors="black", zorder=5)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("batch B")
                ax.set_ylabel("residency r (%)")
                ax.set_title(f"BW={bw} GB/s")
            fig.suptitle(f"{model} ({precision}) -- rho = t_transfer/t_compute "
                         f"(black line = rho=1; * = pilot's measured Coverage@C)")
            fig.colorbar(mesh, ax=axes[0].tolist(), label="rho (log scale)")
            out_path = out_dir / f"roofline_{model}_{precision}.pdf"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out_path))
    return written


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="olmoe,mixtral")
    ap.add_argument("--b1-dirs", default="",
                     help="comma-separated model=path overrides, e.g. olmoe=results/olmoe/b1. "
                          "Defaults to each model's ModelSpec.trace_dir_default.")
    ap.add_argument("--calib", default="", help="path to calib_compute.json from measure_compute.py "
                                                 "(optional; falls back to flop estimate if omitted/missing)")
    ap.add_argument("--n-trials", type=int, default=500, help="resampling trials for E_union(B) estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="experiments/e1_roofline/out")
    args = ap.parse_args()

    models = args.models.split(",")
    b1_overrides = {}
    for pair in args.b1_dirs.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            b1_overrides[k] = v

    calib = None
    if args.calib:
        calib_path = Path(args.calib)
        if calib_path.exists():
            calib = json.loads(calib_path.read_text())
            print(f"[e1] loaded compute calibration from {calib_path}", flush=True)
        else:
            print(f"[e1] WARNING: --calib {calib_path} not found, falling back to flop_estimate", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[e1] computing E_union(B) from traces (resampling, no GPU)...", flush=True)
    e_union_lookups, e_union_per_layer = {}, {}
    for model_tag in models:
        b1_dir = Path(b1_overrides.get(model_tag, REPO_ROOT / MODEL_SPECS[model_tag].trace_dir_default))
        if not (b1_dir / "traces.jsonl.zst").exists():
            print(f"[e1] ERROR: no traces at {b1_dir}/traces.jsonl.zst -- run trace collection first "
                  f"(see download_models.sh + elp_probe/run_pilot.sh or run_mixtral.sh)", flush=True)
            sys.exit(1)
        mean_by_batch, per_layer = build_e_union_lookup(model_tag, b1_dir, BATCH_GRID_E1,
                                                          n_trials=args.n_trials, seed=args.seed)
        e_union_lookups[model_tag] = mean_by_batch
        e_union_per_layer[model_tag] = per_layer
        print(f"[e1]   {model_tag}: E_union(B) = {mean_by_batch}", flush=True)

    print("[e1] sweeping (B, r, precision, BW) grid...", flush=True)
    df = sweep_grid(models, BATCH_GRID_E1, RESIDENCY_PCT_GRID, PRECISION_GRID,
                     LINK_BW_GBPS_GRID, e_union_lookups, calib=calib)
    df.to_parquet(out_dir / "roofline_sweep.parquet") if _has_pyarrow() else None
    df.to_csv(out_dir / "roofline_sweep.csv", index=False)

    prov_table = provisioning_table(df)
    prov_table.to_csv(out_dir / "provisioning_table.csv", index=False)

    headlines = headline_sentences(df)
    for k, v in headlines.items():
        print(f"[e1] HEADLINE {k}: {v}", flush=True)

    print("[e1] plotting rho heatmaps (Figure 1)...", flush=True)
    fig_paths = plot_rho_heatmaps(df, out_dir / "figures")

    achievable = {m: load_pilot_achievable_points(m) for m in models}

    summary = {
        "models": models,
        "batch_grid": BATCH_GRID_E1,
        "residency_pct_grid": RESIDENCY_PCT_GRID,
        "precision_grid": PRECISION_GRID,
        "bw_gbps_grid": LINK_BW_GBPS_GRID,
        "compute_provenance": {m: get_t_compute_ms(MODEL_SPECS[m], BATCH_GRID_E1[0], calib)[1] for m in models},
        "e_union_by_batch": e_union_lookups,
        "provisioning_table": prov_table.to_dict(orient="records"),
        "headline_sentences": headlines,
        "achievable_points_from_pilot": achievable,
        "figures": fig_paths,
        "n_trials": args.n_trials,
        "seed": args.seed,
    }
    summary_path = out_dir / "roofline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[e1] wrote {summary_path}", flush=True)

    print("[e1] rendering RESULTS.md...", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_results_md import render  # noqa: E402
    render(summary_path, Path(__file__).resolve().parent / "RESULTS.md")


def _has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
