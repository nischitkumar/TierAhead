"""[3] Fixed-residency placement (Main.md section 4.3).

v1 specified a multiple-knapsack ILP over (expert, tier); the pilot showed
the marginal value of solving it exactly is small (Coverage@50% tops out at
74.1%/54.5% for OLMoE/Mixtral -- PILOT_FINDINGS.md section 8), so this
implements exactly what survived that finding: a per-layer fixed-residency
policy ranked by training-split frequency, plus two free rules the pilot and
Main.md section 4.3 justify:

  - shared experts (DeepSeek/Qwen-style) are ALWAYS resident -- they fire on
    every token, so pinning them is unconditionally correct.
  - optional epoch re-ranking every N tokens (OLMoE's churn of 0.447 means a
    once-and-forever ranking decays -- PILOT_FINDINGS.md section 4).

The ILP survives only as ``ilp_upper_bound`` below: an exact 0/1 knapsack
solve (small enough to brute-force per layer at these expert counts) used
ONLY to report the headroom a real optimizer would buy over this fixed
policy -- reproducing the pilot's own negative result on demand rather than
asserting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FixedResidencyPolicy:
    residency_pct: float
    n_experts: int
    n_shared_experts: int = 0
    _resident_by_layer: dict[int, set] = field(default_factory=dict, repr=False)

    def fit(self, train_counts: pd.DataFrame) -> "FixedResidencyPolicy":
        """train_counts: DataFrame with columns [layer, expert, count]
        (tiermoe.traces.expert_counts' output on the TRAIN split only)."""
        shared = set(range(self.n_shared_experts))  # convention: shared experts are IDs [0, n_shared)
        routed_n_experts = self.n_experts - self.n_shared_experts
        top_n_routed = max(0, int(np.ceil(routed_n_experts * self.residency_pct / 100.0)))
        for layer, sub in train_counts.groupby("layer"):
            cnt = sub.set_index("expert")["count"].reindex(range(self.n_experts), fill_value=0)
            routed_ranked = cnt.drop(index=shared, errors="ignore").sort_values(ascending=False)
            resident = shared | set(routed_ranked.index[:top_n_routed].tolist())
            self._resident_by_layer[int(layer)] = resident
        return self

    def is_resident(self, layer: int, expert: int) -> bool:
        return expert in self._resident_by_layer.get(layer, set())

    def resident_set(self, layer: int) -> set:
        return self._resident_by_layer.get(layer, set())

    def effective_residency_pct(self, layer: int) -> float:
        """Actual resident fraction INCLUDING shared experts -- reported
        separately from the configured `residency_pct` (which only sizes the
        routed-expert slice) because Main.md's own capacity accounting for
        shared-expert models needs both numbers."""
        return 100.0 * len(self.resident_set(layer)) / self.n_experts if self.n_experts else 0.0

    def coverage(self, test_counts: pd.DataFrame) -> float:
        """Fraction of TEST-split activations served by this residency set --
        the "deployable" Coverage@C metric (tiermoe.analyze.coverage_at_c
        computes the activation-weighted, bootstrap-CI'd version of this same
        quantity; this is the plain point estimate for simulator use)."""
        hit, tot = 0, 0
        for layer, sub in test_counts.groupby("layer"):
            resident = self.resident_set(int(layer))
            c = sub.set_index("expert")["count"]
            hit += c[c.index.isin(resident)].sum()
            tot += c.sum()
        return float(hit / tot) if tot else float("nan")


def ilp_upper_bound(train_counts: pd.DataFrame, test_counts: pd.DataFrame, n_experts: int,
                     residency_pct: float) -> dict:
    """Exact per-layer knapsack (oracle: ranks by the TEST split's own
    counts, not train) -- the upper bound a perfect placement solver could
    ever achieve at this residency level. At n_experts <= 64 (both shipped
    models), sorting IS the optimal 0/1 knapsack solution when every item
    has unit "size" (one resident slot per expert, same size for all) --
    this is not an approximation, greedy-by-value equals optimal for a
    uniform-weight knapsack. Comparing this to FixedResidencyPolicy.coverage
    reproduces PILOT_FINDINGS.md's own negative result (headroom is small)
    on demand instead of merely asserting it.
    """
    top_n = max(1, int(np.ceil(n_experts * residency_pct / 100.0)))
    hit, tot = 0, 0
    for layer, sub in test_counts.groupby("layer"):
        c = sub.set_index("expert")["count"].reindex(range(n_experts), fill_value=0)
        oracle_top = set(c.sort_values(ascending=False).index[:top_n].tolist())
        hit += c[c.index.isin(oracle_top)].sum()
        tot += c.sum()
    oracle_coverage = float(hit / tot) if tot else float("nan")

    policy = FixedResidencyPolicy(residency_pct=residency_pct, n_experts=n_experts).fit(train_counts)
    deployable_coverage = policy.coverage(test_counts)
    return {
        "residency_pct": residency_pct,
        "deployable_coverage": deployable_coverage,
        "oracle_coverage": oracle_coverage,
        "headroom_pp": (oracle_coverage - deployable_coverage) * 100.0,
    }
