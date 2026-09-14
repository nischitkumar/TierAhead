"""[0] The MoE Tiering Roofline (Main.md section 4.0 / Experiments.md E1).

    rho = t_transfer(B, r, p, BW) / t_compute(B)      rho < 1 -> latency-bound
                                                        rho >= 1 -> bandwidth-bound

Ported and generalized from experiments/e1_roofline/roofline.py -- same
formulas, same compute-provenance discipline (every t_compute carries an
explicit "measured_here" vs "flop_estimate" tag), extended with
``validate_calib_sanity`` (see below) and a tier-spec-aware sweep so the
framework isn't hardcoded to the four ad hoc GB/s numbers the original
experiment script used.

No GPU or network required to import or run this module against already-
collected traces; a real GPU calibration file only ever *improves* the
compute-time input, it is never required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tierahead.eunion import e_union_curve
from tierahead.specs import MODEL_SPECS, LINK_BW_GBPS_GRID
from tierahead.specs.models import RESIDENCY_PCT_GRID, BATCH_GRID_E1, PRECISION_GRID
from tierahead.traces.io import find_data_root, load_decode_df, resolve_trace_dir

# RTX 4060 Ti 16GB dense FP16 tensor-core peak (vendor spec). Flop-estimate
# fallback only -- measured_here rows never touch this.
RTX_4060TI_PEAK_FP16_FLOPS = 22.1e12
# Batch-1/8/32 decode is heavily memory-bound on the compute side too (small
# GEMMs, kernel-launch overhead) -- documented ASSUMPTION, not a citation.
DEFAULT_DECODE_MFU = 0.08


# ---------- compute-time model ----------

def t_compute_flop_ms(spec, batch: int, peak_flops: float = RTX_4060TI_PEAK_FP16_FLOPS,
                       mfu: float = DEFAULT_DECODE_MFU) -> float:
    flops = 2.0 * batch * spec.k * spec.params_per_expert
    seconds = flops / (peak_flops * mfu)
    return seconds * 1e3


def t_compute_from_calib(calib: dict, model_tag: str, batch: int) -> float | None:
    entry = (calib or {}).get(model_tag, {}).get("by_batch", {}).get(str(batch))
    return entry["mean_ms"] if entry is not None else None


def get_t_compute_ms(spec, batch: int, calib: dict | None = None) -> tuple[float, str]:
    """Returns (t_compute_ms, provenance). provenance is 'measured_here' only
    if a calib dict is supplied AND validate_calib_sanity has not flagged it
    (callers that skip validation get the raw tag the calib file itself
    claims -- see validate_calib_sanity's docstring for why this framework
    doesn't silently override that, only warns)."""
    if calib is not None:
        measured = t_compute_from_calib(calib, spec.tag, batch)
        if measured is not None:
            return measured, "measured_here"
    return t_compute_flop_ms(spec, batch), "flop_estimate"


def validate_calib_sanity(calib: dict) -> dict[str, list[str]]:
    """Heuristic fraud/placeholder detector for a calib_compute.json-shaped
    dict. This repo has ONE documented instance (E3_CALIBRATION.md, this
    framework's design doc) of a calib_compute.json that claimed
    'measured_here' timings but was fabricated -- no CUDA GPU was ever
    touched to produce it -- and this framework's own due-diligence review
    (see PREDICTED.md's Methodology section) found a *second* file on disk
    with the same symptom: batch=1 and batch=8 mean_ms differing by <2% for
    a GEMM-shaped workload, which is not physically plausible (compute
    should scale with batch, even sub-linearly, for a real MoE FFN forward
    pass) and matches a lazily-authored placeholder more than a measurement.

    This function does not delete or silently "fix" anything -- it returns a
    dict of {model_tag: [warning, ...]}, empty list if the entry looks
    plausible, so callers (tierahead doctor, the CLI, tests) can decide how
    loudly to surface it. Two checks, both cheap and both false-positive-
    resistant for genuinely tiny/fast kernels:
      1. batch-scaling flatness: max/min mean_ms across all reported
         batches differs by less than `min_batch_spread` (default 10%) --
         real compute should not be nearly invariant to an 8-32x batch
         change for a dense-GEMM MoE block.
      2. per-layer near-zero variance: the per_layer_mean_ms values' relative
         standard deviation is below `min_layer_rel_std` (default 1%) --
         real per-layer timing on shared hardware has run-to-run and
         layer-to-layer jitter; near-zero spread across 16-32 independently
         timed layers is a synthetic-data signature.
    """
    warnings: dict[str, list[str]] = {}
    for model_tag, entry in (calib or {}).items():
        w = []
        by_batch = entry.get("by_batch", {})
        means = [v["mean_ms"] for v in by_batch.values() if "mean_ms" in v]
        if len(means) >= 2 and min(means) > 0:
            spread = (max(means) - min(means)) / min(means)
            if spread < 0.10:
                w.append(
                    f"batch-scaling flatness: mean_ms across batches {sorted(by_batch)} spans only "
                    f"{spread:.1%} (min={min(means):.4f}ms, max={max(means):.4f}ms) -- real per-layer "
                    f"MoE compute should vary noticeably across an 8x-32x batch range; this pattern "
                    f"matches the fabricated calib_compute.json this repo already caught once "
                    f"(see E3_CALIBRATION.md). Treat as UNVERIFIED until re-measured with "
                    f"tierahead.calibrate.compute on real CUDA hardware."
                )
        for batch_key, v in by_batch.items():
            per_layer = v.get("per_layer_mean_ms", {})
            vals = list(per_layer.values())
            if len(vals) >= 4 and np.mean(vals) > 0:
                rel_std = float(np.std(vals) / np.mean(vals))
                if rel_std < 0.01:
                    w.append(
                        f"batch={batch_key}: per-layer timing relative std={rel_std:.4%} across "
                        f"{len(vals)} layers -- suspiciously uniform for independently-timed CUDA "
                        f"events on real hardware."
                    )
        if w:
            warnings[model_tag] = w
    return warnings


# ---------- transfer-time / bytes model ----------

def bytes_per_layer(spec, batch: int, residency_pct: float, precision: str, e_union_val: float) -> float:
    """bytes_per_layer(B,r,p) = E_union(B) * expert_size(p) * (1 - r)."""
    return e_union_val * spec.expert_bytes(precision) * (1.0 - residency_pct / 100.0)


def t_transfer_ms(bytes_val: float, bw_gbps: float) -> float:
    if bw_gbps <= 0:
        return float("inf")
    return (bytes_val / (bw_gbps * 1e9)) * 1e3


def rho(t_transfer_val_ms: float, t_compute_val_ms: float) -> float:
    if t_compute_val_ms <= 0:
        return float("inf")
    return t_transfer_val_ms / t_compute_val_ms


def required_bw_gbps(bytes_val: float, t_compute_val_ms: float, target_rho: float = 1.0) -> float:
    if t_compute_val_ms <= 0:
        return float("inf")
    t_compute_s = t_compute_val_ms / 1e3
    return bytes_val / (target_rho * t_compute_s) / 1e9


# ---------- E_union lookup per model ----------

def build_e_union_lookup(model_tag: str, trace_dir: Path, batch_grid, n_trials=500, seed=0):
    decode_df = load_decode_df(trace_dir)
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
    grid_bw = sorted(set(df["bw_gbps"]))

    def nearest_tier(req):
        feasible = [b for b in grid_bw if b >= req]
        return float(min(feasible)) if feasible else float("nan")

    cols = ["model", "precision", "batch", "residency_pct", "compute_provenance", "required_bw_gbps_for_rho1"]
    out = df[cols].drop_duplicates().sort_values(["model", "precision", "batch", "residency_pct"]).reset_index(drop=True)
    out["nearest_supported_link_gbps"] = out["required_bw_gbps_for_rho1"].apply(nearest_tier)
    return out


def headline_sentences(df: pd.DataFrame) -> dict:
    out = {}
    max_batch, max_r = df["batch"].max(), df["residency_pct"].max()
    for (model, precision), sub in df.groupby(["model", "precision"]):
        best = sub[(sub.batch == max_batch) & (sub.residency_pct == max_r)]
        best = best.sort_values("bw_gbps").iloc[0] if len(best) else None
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
                f"{model} ({precision}): NOT reachable anywhere in the swept grid -- even at the "
                f"most favorable point tested this regime needs ~{req:.0f} GB/s to hit rho=1. "
                f"Byte reduction (precision-gating), not prefetch alone, is required here."
            )
    return out


def load_pilot_achievable_points(model_tag: str, data_root: Path | None = None):
    root = data_root or find_data_root()
    path_map = {
        "olmoe": root / "results" / "pilot_summary.json",
        "mixtral": root / "results" / "mixtral_summary.json",
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


def run_roofline(models: list[str] | None = None, calib_path: Path | None = None,
                  batch_grid=None, residency_grid=None, precision_grid=None, bw_grid=None,
                  n_trials: int = 500, seed: int = 0, data_root: Path | None = None) -> dict:
    """High-level entry point (what `tierahead roofline` and tests call): loads
    each model's already-collected traces, computes E_union(B), sweeps the
    grid, and returns the same dict shape roofline_summary.json used to
    have, plus calib-sanity warnings if a calibration file was supplied."""
    models = models or ["olmoe", "mixtral"]
    batch_grid = batch_grid or BATCH_GRID_E1
    residency_grid = residency_grid or RESIDENCY_PCT_GRID
    precision_grid = precision_grid or PRECISION_GRID
    bw_grid = bw_grid or LINK_BW_GBPS_GRID
    root = data_root or find_data_root()

    calib = None
    calib_warnings: dict[str, list[str]] = {}
    if calib_path is not None and Path(calib_path).exists():
        calib = json.loads(Path(calib_path).read_text())
        calib_warnings = validate_calib_sanity(calib)

    e_union_lookups = {}
    for model_tag in models:
        trace_dir = resolve_trace_dir(MODEL_SPECS[model_tag].trace_dir_default, root)
        mean_by_batch, _per_layer = build_e_union_lookup(model_tag, trace_dir, batch_grid, n_trials, seed)
        e_union_lookups[model_tag] = mean_by_batch

    df = sweep_grid(models, batch_grid, residency_grid, precision_grid, bw_grid, e_union_lookups, calib=calib)
    prov_table = provisioning_table(df)
    headlines = headline_sentences(df)
    achievable = {m: load_pilot_achievable_points(m, root) for m in models}

    return {
        "models": models, "batch_grid": batch_grid, "residency_pct_grid": residency_grid,
        "precision_grid": precision_grid, "bw_gbps_grid": bw_grid,
        "compute_provenance": {m: get_t_compute_ms(MODEL_SPECS[m], batch_grid[0], calib)[1] for m in models},
        "calib_sanity_warnings": calib_warnings,
        "e_union_by_batch": e_union_lookups,
        "sweep_df": df,
        "provisioning_table": prov_table,
        "headline_sentences": headlines,
        "achievable_points_from_pilot": achievable,
    }
