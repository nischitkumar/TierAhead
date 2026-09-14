"""E_union(B): expected number of *distinct* experts activated per layer
across a synthetic batch of B decode steps.

Shared by E1 (needs it to compute bytes_per_layer(B,r,p)) and E2 (whose
entire method is characterizing this function's shape). Per Experiments.md
E2 step 1: "you do not need to re-run the models -- sample B token-steps
from distinct requests, take the union of selected experts per layer."
This resamples from already-collected b1 (single-request) decode traces;
no new model inference is required.
"""
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
    """topk_sets: one frozenset of expert ids per decode row (a single layer's
    activations, one row per (req, tok)). Draws `n_trials` batches of size
    `batch` *without replacement within a trial* (distinct token-steps) and
    returns the distribution of |union|.
    """
    n_rows = len(topk_sets)
    if n_rows == 0:
        return EUnionEstimate(batch, float("nan"), float("nan"), float("nan"), 0)
    batch = min(batch, n_rows)  # can't sample more distinct rows than exist
    rng = np.random.default_rng(seed)
    arr = np.array(topk_sets, dtype=object)
    unions = np.empty(n_trials, dtype=np.int64)
    for t in range(n_trials):
        idx = rng.choice(n_rows, size=batch, replace=False)
        u = set()
        for i in idx:
            u |= arr[i]
        unions[t] = len(u)
    return EUnionEstimate(
        batch=batch,
        mean=float(unions.mean()),
        p05=float(np.percentile(unions, 5)),
        p95=float(np.percentile(unions, 95)),
        n_trials=n_trials,
    )


def e_union_curve(decode_df: pd.DataFrame, layer: int, batch_grid: list[int],
                   n_trials: int = 500, seed: int = 0) -> dict[int, EUnionEstimate]:
    """decode_df must have columns ['layer', 'topk'] (topk = tuple of expert ids)."""
    sub = decode_df[decode_df.layer == layer]
    topk_sets = [frozenset(t) for t in sub["topk"]]
    return {b: empirical_e_union(topk_sets, b, n_trials=n_trials, seed=seed) for b in batch_grid}


def analytic_e_union_uniform(n_experts: int, k: int, batch: int) -> float:
    """Closed-form E[|union|] under the null model that each decode step
    draws k experts uniformly at random (no skew, no correlation). Reference
    curve only -- real traffic is skewed (pilot Gini 0.07-0.38), so the
    empirical curve should sit BELOW this for any non-trivial batch (skew
    means popular experts get re-hit, shrinking the union). Formula:
    for each expert e, P(e not touched by any of B draws) ~= (1 - k/n)^B
    (independence approximation, exact for sampling-with-replacement of the
    *draw event*, good approximation for k<<n).
    """
    if n_experts <= 0 or k <= 0 or batch <= 0:
        return 0.0
    p_miss = max(0.0, 1.0 - k / n_experts) ** batch
    return n_experts * (1.0 - p_miss)
