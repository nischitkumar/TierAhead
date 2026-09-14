"""Fits Experiments.md E3 step 2's two-parameter transfer-time model,
`t = alpha + size/beta`, from measured (size, time) pairs. Pure math, no
GPU/network required -- this is the analysis half of E3's transfer
sub-step; `measure_transfer.py` is the GPU half that produces the
(size, time) pairs this function consumes.

alpha_ms is the fixed per-transfer overhead (kernel launch, driver call,
DMA setup) -- what you pay even for a 0-byte transfer. beta_bytes_per_ms is
the asymptotic (large-transfer) bandwidth -- what you approach as size grows
large enough that the fixed overhead becomes negligible. This is exactly the
functional form Experiments.md says downstream simulation should also use
for CXL, with literature-derived alpha/beta swapped in for PCIe-measured
ones -- E3's job is to measure it once, for real, on the physical layer
this box actually has (PCIe), not to guess it.
"""
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
    """t = alpha + size/beta is linear in size (t vs size has intercept
    alpha and slope 1/beta), so an ordinary least-squares fit of times_ms on
    sizes_bytes gives beta = 1/slope directly. Raises ValueError (rather
    than silently returning nonsense) on inputs that can't support a
    meaningful fit: fewer than 2 points, all-identical sizes (zero variance
    in the independent variable -- linregress returns slope=nan here), or a
    non-positive fitted slope (transfer time must increase with size for
    beta=1/slope to mean anything as a bandwidth).
    """
    sizes = np.asarray(sizes_bytes, dtype=np.float64)
    times = np.asarray(times_ms, dtype=np.float64)
    if len(sizes) != len(times):
        raise ValueError(f"sizes_bytes ({len(sizes)}) and times_ms ({len(times)}) must be the same length")
    if len(sizes) < 2:
        raise ValueError(f"need >=2 (size,time) points to fit a line, got {len(sizes)}")
    if len(set(sizes.tolist())) < 2:
        raise ValueError(
            "all sizes are identical -- cannot fit a slope (transfer time vs size needs "
            "variation in size); pass a sweep of distinct transfer sizes"
        )

    result = linregress(sizes, times)
    slope, intercept = result.slope, result.intercept
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError(
            f"fitted slope={slope!r} is non-positive/non-finite -- transfer time must strictly "
            f"increase with size for beta=1/slope to be a meaningful bandwidth; check the input "
            f"data (e.g. did every size use the same warmup/measurement window?)"
        )
    beta = 1.0 / slope
    return TransferFit(
        alpha_ms=float(intercept),
        beta_bytes_per_ms=float(beta),
        r_squared=float(result.rvalue ** 2),
        n_points=len(sizes),
    )


def fit_from_calib_json(calib: dict) -> TransferFit:
    """Convenience wrapper for measure_transfer.py's output shape
    ({"by_size_bytes": {size_bytes_str: {"mean_ms": ..., ...}}})."""
    by_size = calib["by_size_bytes"]
    sizes = [float(k) for k in by_size]
    times = [v["mean_ms"] for v in by_size.values()]
    return fit_affine_transfer_curve(sizes, times)
