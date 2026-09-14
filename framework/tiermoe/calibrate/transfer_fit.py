"""Fits the two-parameter transfer-time model t = alpha + size/beta from
measured (size, time) pairs. Pure math, no GPU -- this is the analysis half;
transfer_measure.py is the GPU half that produces the input pairs.

alpha_ms: fixed per-transfer overhead (kernel launch, driver call, DMA
setup). beta_bytes_per_ms: asymptotic (large-transfer) bandwidth. This is
the exact functional form downstream simulation should use for CXL too, with
literature-derived alpha/beta swapped in for PCIe-measured ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


@dataclass(frozen=True)
class TransferFit:
    alpha_ms: float
    beta_bytes_per_ms: float
    r_squared: float
    n_points: int


def fit_affine_transfer_curve(sizes_bytes, times_ms) -> TransferFit:
    sizes = np.asarray(sizes_bytes, dtype=np.float64)
    times = np.asarray(times_ms, dtype=np.float64)
    if len(sizes) != len(times):
        raise ValueError(f"sizes_bytes ({len(sizes)}) and times_ms ({len(times)}) must be the same length")
    if len(sizes) < 2:
        raise ValueError(f"need >=2 (size,time) points to fit a line, got {len(sizes)}")
    if len(set(sizes.tolist())) < 2:
        raise ValueError("all sizes are identical -- cannot fit a slope; pass a sweep of distinct transfer sizes")

    result = linregress(sizes, times)
    slope, intercept = result.slope, result.intercept
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError(
            f"fitted slope={slope!r} is non-positive/non-finite -- transfer time must strictly increase "
            f"with size for beta=1/slope to be meaningful; check the input data"
        )
    return TransferFit(alpha_ms=float(intercept), beta_bytes_per_ms=float(1.0 / slope),
                        r_squared=float(result.rvalue ** 2), n_points=len(sizes))


def fit_from_calib_json(calib: dict) -> TransferFit:
    by_size = calib["by_size_bytes"]
    sizes = [float(k) for k in by_size]
    times = [v["mean_ms"] for v in by_size.values()]
    return fit_affine_transfer_curve(sizes, times)
