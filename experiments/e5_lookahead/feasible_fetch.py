"""Feasible-fetch-size curve (Experiments.md E5 step 3): "given recall >=
threshold at depth d, prefetch window = d * t_layer => max bytes fetchable
at BW. Overlay actual expert sizes for the four target models. This figure
directly answers 'which MoE architectures can CXL serve?'"

Reuses E1's own per-layer compute-time model (t_compute) rather than
reimplementing it -- one more instance of this project's rule that
compute-time estimation lives in exactly one place (roofline.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1_roofline"))  # e1_roofline/

from common.model_specs import MODEL_SPECS  # noqa: E402
import roofline as rl  # noqa: E402


def feasible_bytes_at_depth(model_tag: str, d: int, batch: int, bw_gbps: float, calib=None) -> float:
    """Max bytes fetchable within a d-layer lookahead window: d layers'
    worth of compute time (the window a prefetch has to hide behind) times
    the link's bandwidth. Uses E1's t_compute (measured_here if a calib file
    is supplied, else flop_estimate -- same fallback discipline as E1/E2)."""
    spec = MODEL_SPECS[model_tag]
    t_layer_ms, provenance = rl.get_t_compute_ms(spec, batch, calib)
    window_ms = d * t_layer_ms
    bytes_fetchable = bw_gbps * 1e6 * window_ms  # GB/s -> bytes/ms, matches E8's unit convention
    return bytes_fetchable, provenance


def max_useful_depth(recall_by_d: dict[int, float], threshold: float = 0.60) -> int | None:
    """Smallest d in recall_by_d (sorted ascending) at or beyond which recall
    stays >= threshold for every subsequent depth in the swept set -- i.e.
    the point beyond which deeper lookahead keeps paying off, not just the
    first d that happens to clear the bar once. Returns None if no depth
    clears it, meaning the given model needs d beyond what was swept (or
    byte reduction, per E1's own framing, rather than more lookahead)."""
    ds = sorted(recall_by_d)
    for i, d in enumerate(ds):
        if all(recall_by_d[dd] >= threshold for dd in ds[i:]):
            return d
    return None
