"""Unit tests for E8's KV-cache formula and link-contention simulator. Pure
Python/numpy -- no GPU, no network, no trace files required. Run with:
    pytest experiments/e8_kv_cotenancy/tests/test_kv_link_sim.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # e8_kv_cotenancy/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/

import kv_link_sim as k8  # noqa: E402
from common.model_specs import MODEL_SPECS, KV_BYTES_PER_ELEMENT  # noqa: E402


# ---------- KV formula ----------

def test_kv_bytes_per_token_matches_hand_calc_olmoe():
    spec = MODEL_SPECS["olmoe"]
    # 2 * 16 layers * 16 kv_heads * 128 head_dim * 2 bytes(fp16) = 131072
    assert spec.kv_bytes_per_token("fp16") == pytest.approx(131072.0)


def test_kv_bytes_per_token_matches_hand_calc_mixtral():
    spec = MODEL_SPECS["mixtral"]
    # 2 * 32 layers * 8 kv_heads(GQA) * 128 head_dim * 2 bytes(fp16) = 131072
    assert spec.kv_bytes_per_token("fp16") == pytest.approx(131072.0)


def test_olmoe_and_mixtral_kv_bytes_per_token_coincidence():
    """Not a bug: 16 layers * 16 kv_heads == 32 layers * 8 kv_heads == 256,
    and both share head_dim=128, so both models land on the identical
    bytes/token despite very different architectures (MHA vs GQA). Locking
    this in as a regression test so a future model_specs.py edit that breaks
    the coincidence is a deliberate, noticed change, not a silent one."""
    assert (MODEL_SPECS["olmoe"].kv_bytes_per_token("fp16")
            == MODEL_SPECS["mixtral"].kv_bytes_per_token("fp16"))


def test_kv_total_bytes_scales_linearly_with_context_len():
    spec = MODEL_SPECS["mixtral"]
    b4k = spec.kv_total_bytes(4096, "fp16")
    b32k = spec.kv_total_bytes(32768, "fp16")
    assert b32k == pytest.approx(b4k * 8)


def test_kv_fp8_is_half_fp16():
    spec = MODEL_SPECS["olmoe"]
    assert spec.kv_bytes_per_token("fp8") == pytest.approx(spec.kv_bytes_per_token("fp16") / 2)
    assert KV_BYTES_PER_ELEMENT["fp8"] < KV_BYTES_PER_ELEMENT["fp16"]


def test_kv_bytes_per_token_rejects_unknown_precision():
    with pytest.raises(ValueError):
        MODEL_SPECS["olmoe"].kv_bytes_per_token("nf4")


# ---------- policy / rate function ----------

def test_expert_first_gives_kv_nothing_when_both_present():
    rate_fn, kind, w = k8.make_rate_fn("expert-first")
    r_e, r_k = rate_fn(True, True, 100.0)
    assert r_e == 100.0 and r_k == 0.0
    assert kind == "expert-first" and w is None


def test_kv_first_gives_expert_nothing_when_both_present():
    rate_fn, kind, w = k8.make_rate_fn("kv-first")
    r_e, r_k = rate_fn(True, True, 100.0)
    assert r_e == 0.0 and r_k == 100.0


def test_weighted_splits_proportionally():
    rate_fn, kind, w = k8.make_rate_fn("weighted:0.3")
    r_e, r_k = rate_fn(True, True, 100.0)
    assert r_k == pytest.approx(30.0)
    assert r_e == pytest.approx(70.0)
    assert w == pytest.approx(0.3)


def test_any_policy_gives_full_link_to_the_only_present_class():
    for policy in ["expert-first", "kv-first", "weighted:0.5"]:
        rate_fn, _, _ = k8.make_rate_fn(policy)
        assert rate_fn(True, False, 64.0) == (64.0, 0.0)
        assert rate_fn(False, True, 64.0) == (0.0, 64.0)
        assert rate_fn(False, False, 64.0) == (0.0, 0.0)


def test_weighted_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        k8.make_rate_fn("weighted:1.5")


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        k8.make_rate_fn("first-come-first-served")


# ---------- simulate_link ----------

def _const_kv_sampler(size_bytes):
    return lambda rng: size_bytes


def test_no_kv_traffic_expert_runs_at_ideal_rate():
    """With kv_rate_per_sec=0, expert jobs never contend for anything --
    each one should complete in exactly bytes/BW (inflation ~= 0)."""
    res = k8.simulate_link(bw_gbps=64.0, policy="expert-first",
                            expert_period_ms=1.0, expert_job_bytes=1e6,
                            kv_rate_per_sec=0.0, kv_job_sampler=_const_kv_sampler(1.0),
                            sim_ms=50.0, seed=0)
    assert len(res["expert"]) > 0
    for _arrival, latency, ideal in res["expert"]:
        assert latency == pytest.approx(ideal, rel=1e-6)


def test_no_expert_traffic_kv_runs_at_ideal_rate_when_isolated():
    """KV jobs spaced far apart relative to their own transfer time (mean
    interarrival 10ms vs a ~0.016ms transfer at this size/BW) shouldn't
    self-queue -- each should complete in exactly bytes/BW."""
    res = k8.simulate_link(bw_gbps=64.0, policy="kv-first",
                            expert_period_ms=0.0, expert_job_bytes=0.0,
                            kv_rate_per_sec=100.0, kv_job_sampler=_const_kv_sampler(1e6),
                            sim_ms=50.0, seed=1)
    assert len(res["kv"]) >= 1
    for _arrival, latency, ideal in res["kv"]:
        assert latency == pytest.approx(ideal, rel=1e-6)


def test_expert_first_fully_shields_expert_from_kv_contention():
    """Under strict expert-first, expert jobs must complete at their ideal
    rate regardless of how much KV traffic is queued -- KV absorbs 100% of
    the contention cost. Expert period (0.05ms) is deliberately much shorter
    than a KV job's own ideal transfer time (5e6/32e6 = 0.15625ms) so that
    virtually every KV job's service window spans multiple expert arrivals --
    without this, a handful of KV samples has a non-trivial chance of never
    overlapping a sparse expert tick, making the test flaky rather than wrong."""
    res = k8.simulate_link(bw_gbps=32.0, policy="expert-first",
                            expert_period_ms=0.05, expert_job_bytes=1e6,
                            kv_rate_per_sec=50.0, kv_job_sampler=_const_kv_sampler(5e6),
                            sim_ms=200.0, seed=2)
    assert len(res["expert"]) > 0
    for _arrival, latency, ideal in res["expert"]:
        assert latency == pytest.approx(ideal, rel=1e-6)
    # and KV must show real inflation (it's absorbing all the contention)
    assert len(res["kv"]) > 0
    inflated = [lat - ideal for _a, lat, ideal in res["kv"]]
    assert all(x > 1e-9 for x in inflated), "every KV job's window should contain >=1 expert burst"


def test_kv_first_fully_shields_kv_from_expert_contention():
    res = k8.simulate_link(bw_gbps=32.0, policy="kv-first",
                            expert_period_ms=2.0, expert_job_bytes=1e6,
                            kv_rate_per_sec=50.0, kv_job_sampler=_const_kv_sampler(5e6),
                            sim_ms=200.0, seed=3)
    for _arrival, latency, ideal in res["kv"]:
        assert latency == pytest.approx(ideal, rel=1e-6)
    inflated = [lat - ideal for _a, lat, ideal in res["expert"]]
    assert any(x > 1e-9 for x in inflated)


def test_weighted_kv_latency_between_the_two_strict_extremes():
    """Monotonicity sanity check: KV's mean latency under a weighted split
    should sit between its (best case) kv-first latency and (worst case)
    expert-first latency, for the same arrival scenario / seed."""
    kwargs = dict(bw_gbps=32.0, expert_period_ms=2.0, expert_job_bytes=1e6,
                  kv_rate_per_sec=50.0, kv_job_sampler=_const_kv_sampler(5e6),
                  sim_ms=300.0, seed=4)
    best = k8.simulate_link(policy="kv-first", **kwargs)
    worst = k8.simulate_link(policy="expert-first", **kwargs)
    mid = k8.simulate_link(policy="weighted:0.5", **kwargs)

    def mean_latency(res, cls):
        return np.mean([lat for _a, lat, _i in res[cls]])

    best_lat, worst_lat, mid_lat = mean_latency(best, "kv"), mean_latency(worst, "kv"), mean_latency(mid, "kv")
    assert best_lat <= mid_lat <= worst_lat + 1e-6


def test_byte_conservation_no_class_exceeds_link_capacity():
    """Sanity/physical check: total bytes actually served (sum of completed
    job sizes) cannot exceed BW*duration -- the link has a hard capacity."""
    bw_gbps, sim_ms = 16.0, 100.0
    res = k8.simulate_link(bw_gbps=bw_gbps, policy="weighted:0.4",
                            expert_period_ms=1.0, expert_job_bytes=1e6,
                            kv_rate_per_sec=20.0, kv_job_sampler=_const_kv_sampler(2e6),
                            sim_ms=sim_ms, seed=5)
    # each record's 3rd field is bytes/BW (ms); recover bytes served via *BW.
    total_bytes = sum(ideal * res["bw_bytes_per_ms"] for _a, _l, ideal in res["expert"]) + \
        sum(ideal * res["bw_bytes_per_ms"] for _a, _l, ideal in res["kv"])
    capacity_bytes = res["bw_bytes_per_ms"] * sim_ms
    assert total_bytes <= capacity_bytes * 1.001  # small float slack


def test_zero_link_bandwidth_never_completes_anything_within_horizon():
    res = k8.simulate_link(bw_gbps=1e-12, policy="expert-first",
                            expert_period_ms=10.0, expert_job_bytes=1e6,
                            kv_rate_per_sec=0.0, kv_job_sampler=_const_kv_sampler(1.0),
                            sim_ms=30.0, seed=6)
    assert len(res["expert"]) == 0  # jobs arrive but never finish transferring


# ---------- pooled_simulate ----------

def test_pooled_simulate_pools_records_across_seeds():
    kwargs = dict(bw_gbps=64.0, policy="expert-first", expert_period_ms=1.0, expert_job_bytes=1e6,
                  kv_rate_per_sec=0.0, kv_job_sampler=_const_kv_sampler(1.0), sim_ms=20.0)
    single = k8.simulate_link(seed=0, **kwargs)
    pooled = k8.pooled_simulate(n_seeds=4, base_seed=0, **kwargs)
    # deterministic expert arrivals -> pooling across seeds just replicates
    # the same count 4x (no randomness in this particular scenario).
    assert len(pooled["expert"]) == 4 * len(single["expert"])


def test_pooled_simulate_uses_distinct_seeds_for_random_traffic():
    """With KV traffic present (random), pooling 5 seeds of a low-rate
    Poisson process should yield noticeably more samples than any single
    seed -- distinct seeds, not the same seed repeated."""
    kwargs = dict(bw_gbps=64.0, policy="kv-first", expert_period_ms=0.0, expert_job_bytes=0.0,
                  kv_rate_per_sec=20.0, kv_job_sampler=_const_kv_sampler(1e5), sim_ms=500.0)
    pooled = k8.pooled_simulate(n_seeds=5, base_seed=0, **kwargs)
    singles = [len(k8.simulate_link(seed=s, **kwargs)["kv"]) for s in range(5)]
    assert len(pooled["kv"]) == sum(singles)
    assert len(set(singles)) > 1, "identical counts across seeds would suggest seeds aren't varying"


# ---------- recommend_policy ----------

def _fake_sweep_df():
    import pandas as pd
    rows = [
        {"model": "m", "bw_gbps": 32.0, "policy": "expert-first", "ttft_p50_ms": 1000.0, "tpot_inflation_p50_ms": 0.0},
        {"model": "m", "bw_gbps": 32.0, "policy": "kv-first", "ttft_p50_ms": 100.0, "tpot_inflation_p50_ms": 50.0},
        {"model": "m", "bw_gbps": 32.0, "policy": "weighted:0.5", "ttft_p50_ms": 400.0, "tpot_inflation_p50_ms": 10.0},
    ]
    return k8.add_pareto_column(pd.DataFrame(rows))


def test_recommend_policy_no_slo_reports_both_extremes_not_a_single_pick():
    df = _fake_sweep_df()
    recs = k8.recommend_policy(df, ttft_slo_ms=None)
    rec = recs["m_32.0gbps"]
    assert rec["recommended_policy"] is None
    assert rec["best_for_ttft"]["policy"] == "kv-first"       # lowest ttft
    assert rec["best_for_tpot"]["policy"] == "expert-first"   # lowest inflation


def test_recommend_policy_with_feasible_slo_picks_min_inflation_among_feasible():
    df = _fake_sweep_df()
    # SLO=500ms rules out expert-first (ttft=1000); among kv-first(100,50) and
    # weighted:0.5(400,10), both feasible -- min inflation is weighted:0.5.
    recs = k8.recommend_policy(df, ttft_slo_ms=500.0)
    rec = recs["m_32.0gbps"]
    assert rec["recommended_policy"] == "weighted:0.5"


def test_recommend_policy_with_infeasible_slo_falls_back_to_best_ttft():
    df = _fake_sweep_df()
    recs = k8.recommend_policy(df, ttft_slo_ms=1.0)  # nothing meets 1ms
    rec = recs["m_32.0gbps"]
    assert rec["recommended_policy"] == "kv-first"  # best achievable TTFT
    assert "NO policy meets" in rec["note"]


def test_recommend_policy_does_not_trivially_always_pick_expert_first():
    """Regression test for the rejected lexicographic tie-break: with a
    generous SLO, the recommendation must be able to move away from
    expert-first even though expert-first always has the lowest possible
    (zero) inflation -- otherwise the recommendation is data-independent."""
    df = _fake_sweep_df()
    recs = k8.recommend_policy(df, ttft_slo_ms=150.0)  # only kv-first qualifies
    assert recs["m_32.0gbps"]["recommended_policy"] == "kv-first"


# ---------- pareto_labels ----------

def test_pareto_labels_identifies_non_dominated_points():
    # A dominates C (A better-or-equal on both, strictly better on one).
    # B is non-dominated (better on y than A, worse on x).
    points = [(1.0, 5.0, "A"), (2.0, 1.0, "B"), (3.0, 6.0, "C")]
    result = set(k8.pareto_labels(points))
    assert result == {"A", "B"}


def test_pareto_labels_all_non_dominated_when_no_point_beats_another():
    points = [(1.0, 5.0, "A"), (5.0, 1.0, "B"), (3.0, 3.0, "C")]
    result = set(k8.pareto_labels(points))
    assert result == {"A", "B", "C"}


def test_pareto_labels_single_point():
    assert k8.pareto_labels([(1.0, 1.0, "only")]) == ["only"]
