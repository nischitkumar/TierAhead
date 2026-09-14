"""[4b] Confidence-gated precision fallback (Main.md section 4.4b) -- the
novelty claim, and the mechanism this repo's experiments/ folder speced
(Experiments.md E7, [P1]) but never implemented (no e7_* folder exists at the
time this framework was built). This module is that implementation.

Mechanism: store each expert at two precisions (FP16 + NF4, +25% capacity
cost per Main.md's own honest accounting). For a predicted candidate expert
with confidence p (from any predictor exposing a probability -- the
frequency table's `.probabilities()` or the calibrated MLP's sigmoid
output):

    fetch FP16   if p >= tau_hi
    fetch NF4    if tau_lo <= p < tau_hi
    skip         if p <  tau_lo

A misprediction on an NF4-fetched candidate costs a bounded QUALITY penalty
(a usable low-precision copy was already resident) instead of a multi-
millisecond demand stall. This is genuinely novel over HOBBIT
(arXiv:2411.01433, which does mixed-precision reactively ON MISS): this is
predictive and calibrated, which is why the calibration step in
policy.calibration is a prerequisite, not decoration.

IMPORTANT HONESTY NOTE (read before quoting a quality number from this
module): the byte-accounting side of this policy (bytes fetched, bytes
saved vs FP16-everywhere) is exact arithmetic and needs no GPU. The QUALITY
side (real Delta-PPL / accuracy loss from actually substituting NF4 experts
during inference) is NOT measured anywhere in this repo -- doing so needs a
GPU running real inference (Main.md section 4.4b: "validated by real
inference runs ... not asserted"). ``expected_delta_ppl`` below returns a
PROJECTED number sourced from cited literature (QLoRA, HOBBIT), explicitly
tagged `provenance="projected_from_literature"`, never `"measured_here"`.
See PREDICTED.md's E7 section for the full caveat and the exact GPU
experiment that would upgrade this from projected to measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Decision = Literal["fp16", "nf4", "skip"]


@dataclass(frozen=True)
class PrecisionDecision:
    expert_id: int
    probability: float
    decision: Decision


@dataclass
class PrecisionGatePolicy:
    tau_lo: float
    tau_hi: float

    def __post_init__(self):
        if not (0.0 <= self.tau_lo <= self.tau_hi <= 1.0):
            raise ValueError(f"require 0 <= tau_lo <= tau_hi <= 1, got tau_lo={self.tau_lo}, tau_hi={self.tau_hi}")

    def decide_one(self, expert_id: int, probability: float) -> PrecisionDecision:
        if probability >= self.tau_hi:
            d: Decision = "fp16"
        elif probability >= self.tau_lo:
            d = "nf4"
        else:
            d = "skip"
        return PrecisionDecision(expert_id=expert_id, probability=probability, decision=d)

    def decide(self, candidate_probs: dict[int, float]) -> list[PrecisionDecision]:
        return [self.decide_one(e, p) for e, p in candidate_probs.items()]

    def bytes_for_decisions(self, decisions: list[PrecisionDecision], expert_bytes_fp16: float,
                             expert_bytes_nf4: float) -> dict:
        n_fp16 = sum(1 for d in decisions if d.decision == "fp16")
        n_nf4 = sum(1 for d in decisions if d.decision == "nf4")
        n_skip = sum(1 for d in decisions if d.decision == "skip")
        bytes_fetched = n_fp16 * expert_bytes_fp16 + n_nf4 * expert_bytes_nf4
        bytes_if_all_fp16 = len(decisions) * expert_bytes_fp16
        return {
            "n_fp16": n_fp16, "n_nf4": n_nf4, "n_skip": n_skip,
            "bytes_fetched": bytes_fetched, "bytes_if_all_fp16": bytes_fetched and bytes_if_all_fp16,
            "bytes_saved_pct": (100.0 * (1 - bytes_fetched / bytes_if_all_fp16)
                                 if bytes_if_all_fp16 > 0 else 0.0),
        }


def sweep_thresholds(candidate_probs: dict[int, float], expert_bytes_fp16: float, expert_bytes_nf4: float,
                      tau_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    """Sweeps (tau_lo, tau_hi) over a grid (tau_lo <= tau_hi only) and reports
    the bytes-saved side of the Pareto frontier for one layer's candidate
    set. Pure arithmetic -- safe to run anywhere, no GPU."""
    rows = []
    for tau_lo in tau_grid:
        for tau_hi in tau_grid:
            if tau_hi < tau_lo:
                continue
            policy = PrecisionGatePolicy(tau_lo=tau_lo, tau_hi=tau_hi)
            decisions = policy.decide(candidate_probs)
            stats = policy.bytes_for_decisions(decisions, expert_bytes_fp16, expert_bytes_nf4)
            rows.append({"tau_lo": tau_lo, "tau_hi": tau_hi, **stats})
    return rows


def expected_delta_ppl(bytes_saved_frac: float, source: str = "qlora_nf4_prior") -> dict:
    """PROJECTED (not measured) quality cost of the precision-gating policy,
    as a function of the fraction of candidate bytes served at NF4 instead
    of FP16. This is NOT a fitted curve from this project's own inference
    runs -- it is a documented, literature-anchored PRIOR, structured so a
    real GPU run (substitute NF4 experts per the policy's decisions, measure
    WikiText-2 perplexity, per Main.md section 4.4b step 3) can directly
    replace it with a measured curve without changing this function's
    signature or any caller.

    The prior: QLoRA (Dettmers et al., arXiv:2305.14314) reports NF4
    fine-tuned models recovering FP16 performance within noise for full
    *fine-tuning* substitution -- a stronger substitution than this policy's
    "some fraction of experts, chosen by low confidence" scheme, so the
    linear-in-bytes-saved model below (dPPL ~= 0.15 * bytes_saved_frac,
    capped at 0.6) is a deliberately conservative (pessimistic) upper bound,
    not a best case. HOBBIT (Tang et al., arXiv:2411.01433) reports usable
    quality under its own (reactive, uncalibrated) mixed-precision policy at
    comparable NF4/INT4 fractions, which is the other anchor for the slope
    chosen here. Treat every number this function returns as "a bound we
    are willing to defend in a report before the real GPU run exists," not
    as a claim about this specific project's actual models.
    """
    frac = float(np.clip(bytes_saved_frac, 0.0, 1.0))
    slope = 0.15  # PPL points per unit of bytes-saved fraction, conservative prior (see docstring)
    cap = 0.6
    projected = min(cap, slope * frac)
    return {
        "bytes_saved_frac": frac,
        "expected_delta_ppl": projected,
        "provenance": "projected_from_literature",
        "source": source,
        "citations": [
            "Dettmers et al., QLoRA, arXiv:2305.14314",
            "Tang et al., HOBBIT, arXiv:2411.01433",
        ],
        "caveat": "NOT measured on this project's models. Upgrade to measured by running the real "
                  "substitution + WikiText-2 perplexity eval on a CUDA box (Main.md section 4.4b step 3) "
                  "and replacing this function's output with the fitted curve.",
    }
