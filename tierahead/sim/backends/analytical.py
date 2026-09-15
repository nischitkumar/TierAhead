"""Analytical backend: pure roofline ratio (tierahead.roofline), no
trace-driven simulation. Fastest, coarsest -- answers "which regime is this
configuration in" (rho<1 vs rho>=1), not "what is the actual TPOT/stall
rate." Always available, no dependencies beyond numpy/pandas.
"""
from __future__ import annotations

from pathlib import Path

from tierahead.roofline.model import run_roofline


def run(model_tag: str, residency_pct: float, bw_gbps: float, precision: str, batch: int = 1,
        calib_path: Path | None = None, data_root: Path | None = None) -> dict:
    result = run_roofline(models=[model_tag], calib_path=calib_path, batch_grid=[batch],
                           residency_grid=[residency_pct], precision_grid=[precision], bw_grid=[bw_gbps],
                           data_root=data_root)
    row = result["sweep_df"].iloc[0]
    return {
        "backend": "analytical", "model": model_tag, "residency_pct": residency_pct, "bw_gbps": bw_gbps,
        "precision": precision, "batch": batch, "rho": float(row["rho"]),
        "regime": "latency_bound (prefetch can hide it)" if row["rho"] < 1 else "bandwidth_bound (only byte reduction helps)",
        "t_compute_ms": float(row["t_compute_ms"]), "t_transfer_ms": float(row["t_transfer_ms"]),
        "compute_provenance": row["compute_provenance"],
        "calib_sanity_warnings": result["calib_sanity_warnings"],
    }
