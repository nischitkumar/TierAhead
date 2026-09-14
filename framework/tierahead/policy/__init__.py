"""[3]/[4]/[4b] Placement, prefetch, and precision-gating policies.

Deliberately does NOT eagerly import predictor_mlp/calibration/lookahead at
package level -- those need torch (an optional extra, ``pip install
tierahead[predictor]``), and every other policy (residency, the frequency-table
predictor, precision-gate) must stay importable with only numpy/pandas so
``tierahead.sim`` and ``tierahead.baseline`` never force a torch install for
users who only want the always-available policies.
"""
from .residency import FixedResidencyPolicy
from .predictor_freq import FrequencyTablePredictor
from .precision_gate import PrecisionGatePolicy, PrecisionDecision, expected_delta_ppl

__all__ = [
    "FixedResidencyPolicy", "FrequencyTablePredictor",
    "PrecisionGatePolicy", "PrecisionDecision", "expected_delta_ppl",
]
