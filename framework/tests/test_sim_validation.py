"""Runs Experiments.md E6's own validation gates for real against the pilot's
committed traces -- see tiermoe/sim/validation.py's module docstring and
PREDICTED.md's Verified section for what these results mean for the report.
"""
from tiermoe.sim.validation import (
    find_collapse_concurrency, gate_hybrid_never_worse_than_lru_under_concurrency,
    gate_latency_barely_moves_regime, gate_monotone_in_residency,
    gate_oracle_invariant_to_concurrency, gate_oracle_is_upper_bound,
    gate_tpot_monotone_nondecreasing_in_concurrency, gate_zero_latency_infinite_bw_parity,
    run_validation_suite,
)


def test_gate_oracle_is_upper_bound_olmoe(olmoe_trace_dir):
    result = gate_oracle_is_upper_bound("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_zero_latency_infinite_bw_parity_olmoe(olmoe_trace_dir):
    result = gate_zero_latency_infinite_bw_parity("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_monotone_in_residency_olmoe(olmoe_trace_dir):
    result = gate_monotone_in_residency("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_latency_barely_moves_regime_olmoe(olmoe_trace_dir):
    result = gate_latency_barely_moves_regime("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_oracle_invariant_to_concurrency_olmoe(olmoe_trace_dir):
    result = gate_oracle_invariant_to_concurrency("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_oracle_invariant_to_concurrency_mixtral(mixtral_trace_dir):
    result = gate_oracle_invariant_to_concurrency("mixtral", mixtral_trace_dir)
    assert result["passed"], result


def test_gate_tpot_monotone_nondecreasing_in_concurrency_olmoe(olmoe_trace_dir):
    result = gate_tpot_monotone_nondecreasing_in_concurrency("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_hybrid_never_worse_than_lru_under_concurrency_olmoe(olmoe_trace_dir):
    result = gate_hybrid_never_worse_than_lru_under_concurrency("olmoe", olmoe_trace_dir)
    assert result["passed"], result


def test_gate_hybrid_never_worse_than_lru_under_concurrency_mixtral(mixtral_trace_dir):
    result = gate_hybrid_never_worse_than_lru_under_concurrency("mixtral", mixtral_trace_dir)
    assert result["passed"], result


def test_find_collapse_concurrency_prefetch_freq_collapses_at_low_bandwidth(olmoe_trace_dir):
    # At the default gate/roofline "everything collapses" reference point
    # (residency=25%, bw=32GB/s, depth=1 -- PREDICTED.md Sec 2.1's own
    # documented reproduction of Main.md's rho>=1 claim), prefetch-freq is
    # already indistinguishable from static-c at concurrency=1 itself.
    result = find_collapse_concurrency("olmoe", olmoe_trace_dir, policy="prefetch-freq")
    assert result["collapse_concurrency"] is not None, result
    assert result["collapse_concurrency"] == 1, result


def test_find_collapse_concurrency_none_for_hybrid_at_good_operating_point(olmoe_trace_dir):
    # hybrid-lru-prefetch's whole point (PREDICTED.md Sec 2.5): it should NOT
    # collapse to static-c at the "good" operating point where conditional
    # prefetch alone visibly helps.
    result = find_collapse_concurrency("olmoe", olmoe_trace_dir, residency_pct=25.0, bw_gbps=128.0, depth=4,
                                        concurrency_grid=(1, 2, 4, 8), policy="hybrid-lru-prefetch")
    assert result["collapse_concurrency"] is None, result


def test_full_validation_suite_olmoe(olmoe_trace_dir):
    result = run_validation_suite("olmoe", olmoe_trace_dir)
    assert result["all_passed"], result
    assert "concurrency_collapse_analysis" in result


def test_full_validation_suite_mixtral(mixtral_trace_dir):
    result = run_validation_suite("mixtral", mixtral_trace_dir)
    assert result["all_passed"], result
    assert "concurrency_collapse_analysis" in result
