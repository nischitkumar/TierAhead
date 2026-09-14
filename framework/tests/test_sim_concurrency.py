"""Locks in the OPEN-LOOP admission-control finding tiermoe/sim/concurrency.py's
module docstring describes (its "question 2"): for N independent, always-
emitting periodic traffic sources sharing one link, below the aggregate-
demand-equals-bandwidth boundary DES contention is negligible (latency stays
at the solo value); at or above it, the queue is genuinely unstable (latency
diverges without bound). This is a real, DES-validated result for OPEN-LOOP
traffic (e.g. new session arrivals) -- it is deliberately NOT a validation of
`simulate_policy`'s CLOSED-loop bw/N concurrency model (self-throttling
decode sessions), which is a different question with a different (and
correct, textbook) answer -- see the module docstring's "THE MISTAKE"
section for why conflating the two was wrong and how it was caught.
"""
from tiermoe.sim.concurrency import (
    aggregate_rho, max_sustainable_concurrency, verify_stability_boundary_with_des,
)

# bw_gbps=10 -> 1e7 bytes/ms. job_bytes/period_ms = 1e6 bytes/ms per stream.
# Boundary: N * 1e6 = 1e7 -> N = 10 exactly.
PERIOD_MS = 1.0
JOB_BYTES = 1e6
BW_GBPS = 10.0


def test_max_sustainable_concurrency_matches_hand_calculation():
    assert max_sustainable_concurrency(bytes_per_stream_per_ms=JOB_BYTES / PERIOD_MS, bw_gbps=BW_GBPS) == 10.0


def test_aggregate_rho_scales_linearly_with_concurrency():
    rate = JOB_BYTES / PERIOD_MS
    assert aggregate_rho(rate, BW_GBPS, 1) == 0.1
    assert aggregate_rho(rate, BW_GBPS, 5) == 0.5
    assert aggregate_rho(rate, BW_GBPS, 10) == 1.0


def test_at_or_below_boundary_des_matches_solo_latency():
    # C=10 sits EXACTLY at aggregate_rho=1.0 -- with perfectly deterministic,
    # evenly-staggered arrivals this is the critical (not yet diverging)
    # point, empirically confirmed stable here; strict inequality (C=11+) is
    # where real divergence starts. Included deliberately, not just the
    # comfortably-sub-critical points, to pin down exactly where the
    # boundary this module claims actually sits.
    for c in (1, 4, 8, 9, 10):
        result = verify_stability_boundary_with_des(PERIOD_MS, JOB_BYTES, BW_GBPS, c, sim_ms=3000.0)
        assert result["matches_solo_within_5pct"], result


def test_above_boundary_des_diverges_from_solo():
    for c in (12, 16):
        result = verify_stability_boundary_with_des(PERIOD_MS, JOB_BYTES, BW_GBPS, c, sim_ms=3000.0)
        assert not result["predicted_stable"], result
        assert not result["matches_solo_within_5pct"], result
        # Should be dramatically worse, not just "a bit off" -- this is the
        # actual finding (naive bw/N badly UNDERSTATES this, since bw/N
        # would predict a small finite number here too).
        assert result["des_measured_latency_ms"] > result["ideal_solo_latency_ms"] * 5, result


def test_concurrency_one_is_always_trivially_stable():
    result = verify_stability_boundary_with_des(PERIOD_MS, JOB_BYTES, BW_GBPS, 1, sim_ms=1000.0)
    assert result["predicted_stable"]
    assert result["matches_solo_within_5pct"]
