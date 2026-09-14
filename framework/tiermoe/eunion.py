"""E_union(B): expected number of *distinct* experts activated per layer
across a synthetic batch of B decode steps.

Ported from experiments/common/eunion.py. Shared by tiermoe.roofline (needs
it for bytes_per_layer(B,r,p)) and any batching analysis: resamples B
token-steps from the already-collected b1 (single-request) decode traces and
takes the union of selected experts per layer -- Experiments.md E2 step 1's
method ("you do not need to re-run the models"). No GPU or model inference
required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EUnionEstimate:
    batch: int
    mean: float
    p05: float
    p95: float
    n_trials: int


def empirical_e_union(topk_sets: list[frozenset], batch: int, n_trials: int = 500,
                       seed: int = 0) -> EUnionEstimate:
    n_rows = len(topk_sets)
    if n_rows == 0:
        return EUnionEstimate(batch, float("nan"), float("nan"), float("nan"), 0)
    batch = min(batch, n_rows)
    rng = np.random.default_rng(seed)
    arr = np.array(topk_sets, dtype=object)
    unions = np.empty(n_trials, dtype=np.int64)
    for t in range(n_trials):
        idx = rng.choice(n_rows, size=batch, replace=False)
        u: set = set()
        for i in idx:
            u |= arr[i]
        unions[t] = len(u)
    return EUnionEstimate(
        batch=batch, mean=float(unions.mean()), p05=float(np.percentile(unions, 5)),
        p95=float(np.percentile(unions, 95)), n_trials=n_trials,
    )


def e_union_curve(decode_df: pd.DataFrame, layer: int, batch_grid: list[int],
                   n_trials: int = 500, seed: int = 0) -> dict[int, EUnionEstimate]:
    sub = decode_df[decode_df.layer == layer]
    topk_sets = [frozenset(t) for t in sub["topk"]]
    return {b: empirical_e_union(topk_sets, b, n_trials=n_trials, seed=seed) for b in batch_grid}


def e_union_lookup(decode_df: pd.DataFrame, batch_grid: list[int], n_trials: int = 500,
                    seed: int = 0) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Returns (mean_by_batch, per_layer) -- mean_by_batch averages the
    per-layer curve into one number per batch (what the roofline sweep
    consumes); per_layer keeps the detail (what a batching-sweep figure
    needs). n_layers is inferred from the trace's own max layer index."""
    n_layers = int(decode_df["layer"].max()) + 1
    per_layer: dict[int, dict[int, float]] = {}
    for layer in range(n_layers):
        curve = e_union_curve(decode_df, layer, batch_grid, n_trials=n_trials, seed=seed)
        per_layer[layer] = {b: est.mean for b, est in curve.items()}
    mean_by_batch = {b: float(np.mean([per_layer[l][b] for l in per_layer])) for b in batch_grid}
    return mean_by_batch, per_layer


def analytic_e_union_uniform(n_experts: int, k: int, batch: int) -> float:
    """Closed-form E[|union|] under the null model of uniform-random expert
    selection (no skew, no correlation) -- reference curve only. Real
    traffic is skewed (Gini 0.07-0.38 per the pilot), so the empirical curve
    sits below this for any non-trivial batch."""
    if n_experts <= 0 or k <= 0 or batch <= 0:
        return 0.0
    p_miss = max(0.0, 1.0 - k / n_experts) ** batch
    return n_experts * (1.0 - p_miss)
