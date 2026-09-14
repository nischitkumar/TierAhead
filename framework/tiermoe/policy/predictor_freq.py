"""[4] Prefetch engine -- the pilot's own mechanism, promoted to the system's
centerpiece (Main.md section 4.4, "prefetch-freq" policy). Wraps
tiermoe.analyze.build_transition_tables (a conditional frequency table,
additive smoothing, ranked by summed log-probability) as a reusable
predictor object with a stable `.topm(...)` API, so tiermoe.sim's policies
call one thing regardless of whether the backing model is this frequency
table or policy.predictor_mlp's TinyMLP (see predictor_mlp.py's identical
`.topm`-shaped interface).

No GPU, no torch -- pure numpy, works everywhere this framework runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tiermoe.analyze.workload import build_transition_tables


@dataclass
class FrequencyTablePredictor:
    n_experts: int
    n_layers: int
    d_values: tuple[int, ...] = (1,)
    alpha: float = 1.0
    _tables: dict = field(default_factory=dict, repr=False)
    _log_tables: dict = field(default_factory=dict, repr=False)

    def fit(self, df: pd.DataFrame, train_reqs: set) -> "FrequencyTablePredictor":
        self._tables = build_transition_tables(df, train_reqs, self.n_experts, self.n_layers,
                                                list(self.d_values), alpha=self.alpha)
        self._log_tables = {
            d: {l: np.log(p) for l, p in layer_tables.items()}
            for d, layer_tables in self._tables.items()
        }
        return self

    def topm(self, selected_experts_at_l: list[int] | tuple[int, ...], layer: int, d: int, m: int) -> list[int]:
        """Ranks all n_experts candidates for layer+d given the observed
        expert set at `layer`, returns the top-m expert ids (capped at
        n_experts -- Experiments.md's own m=4k-is-degenerate-for-Mixtral
        caveat, handled once here so every caller gets it for free)."""
        table = self._log_tables.get(d, {}).get(layer)
        m = min(m, self.n_experts)
        if table is None:
            # No fitted transition for this (layer, d) -- fall back to
            # uniform ranking (expert id order) rather than crashing; a
            # caller sweeping depths beyond what a small trace supports
            # should degrade gracefully, matching tiermoe.policy.lookahead's
            # own "n_layers_used" accounting for missing layer-transitions.
            return list(range(m))
        scores = table[list(selected_experts_at_l), :].sum(axis=0)
        order = np.argsort(-scores)
        return order[:m].tolist()

    def probabilities(self, selected_experts_at_l, layer: int, d: int) -> np.ndarray:
        """Un-normalized relative scores -> a proper probability vector over
        candidates (softmax of the summed log-scores), for callers (e.g.
        precision_gate) that need a per-expert confidence rather than a
        ranked list. This is NOT the same as a calibrated probability
        (policy.calibration handles that, and only applies to the MLP
        predictor's sigmoid outputs) -- treat this as an uncalibrated
        relative-confidence proxy for the frequency-table policy."""
        table = self._log_tables.get(d, {}).get(layer)
        if table is None:
            return np.full(self.n_experts, 1.0 / self.n_experts)
        scores = table[list(selected_experts_at_l), :].sum(axis=0)
        scores = scores - scores.max()
        p = np.exp(scores)
        return p / p.sum()
