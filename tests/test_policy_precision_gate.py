import pytest

from tierahead.policy.precision_gate import PrecisionGatePolicy, expected_delta_ppl, sweep_thresholds


def test_decide_thresholds():
    gate = PrecisionGatePolicy(tau_lo=0.3, tau_hi=0.7)
    assert gate.decide_one(0, 0.9).decision == "fp16"
    assert gate.decide_one(1, 0.5).decision == "nf4"
    assert gate.decide_one(2, 0.1).decision == "skip"
    assert gate.decide_one(3, 0.7).decision == "fp16"  # boundary: >= tau_hi
    assert gate.decide_one(4, 0.3).decision == "nf4"  # boundary: >= tau_lo


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        PrecisionGatePolicy(tau_lo=0.8, tau_hi=0.2)


def test_bytes_for_decisions_all_fp16_when_tau_lo_is_one():
    gate = PrecisionGatePolicy(tau_lo=1.0, tau_hi=1.0)
    decisions = gate.decide({0: 0.99, 1: 0.5, 2: 0.01})
    stats = gate.bytes_for_decisions(decisions, expert_bytes_fp16=100.0, expert_bytes_nf4=25.0)
    assert stats["n_fp16"] == 0 and stats["n_nf4"] == 0 and stats["n_skip"] == 3


def test_bytes_saved_positive_when_some_nf4():
    gate = PrecisionGatePolicy(tau_lo=0.2, tau_hi=0.8)
    decisions = gate.decide({0: 0.9, 1: 0.5, 2: 0.1})
    stats = gate.bytes_for_decisions(decisions, expert_bytes_fp16=100.0, expert_bytes_nf4=25.0)
    assert stats["n_fp16"] == 1 and stats["n_nf4"] == 1 and stats["n_skip"] == 1
    assert stats["bytes_saved_pct"] > 0


def test_sweep_thresholds_returns_valid_grid():
    rows = sweep_thresholds({0: 0.9, 1: 0.5, 2: 0.1}, expert_bytes_fp16=100.0, expert_bytes_nf4=25.0)
    assert all(r["tau_lo"] <= r["tau_hi"] for r in rows)
    assert len(rows) > 0


def test_expected_delta_ppl_is_projected_not_measured():
    result = expected_delta_ppl(bytes_saved_frac=0.3)
    assert result["provenance"] == "projected_from_literature"
    assert 0 <= result["expected_delta_ppl"] <= 0.6
    assert "citations" in result and len(result["citations"]) >= 1


def test_expected_delta_ppl_monotone_in_bytes_saved():
    lo = expected_delta_ppl(0.1)["expected_delta_ppl"]
    hi = expected_delta_ppl(0.8)["expected_delta_ppl"]
    assert hi >= lo
