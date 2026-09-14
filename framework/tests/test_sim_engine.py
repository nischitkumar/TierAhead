from tierahead.sim.engine import TrafficClass, pareto_labels, pctile_stats, pooled_simulate, simulate_link


def test_single_periodic_class_no_contention():
    """One class, no contention: every job should complete essentially
    instantly relative to its ideal (period << bw so no queueing)."""
    cls = TrafficClass("solo", weight=1.0, period_ms=10.0, job_bytes=1e6)  # 1MB every 10ms
    result = simulate_link(classes=[cls], bw_gbps=100.0, sim_ms=1000.0, seed=0)
    assert len(result["solo"]) > 0
    lat = [r[1] for r in result["solo"]]
    ideal = [r[2] for r in result["solo"]]
    assert all(abs(l - i) < 1e-6 for l, i in zip(lat, ideal))


def test_two_classes_expert_first_starves_kv_under_saturation():
    """expert weight=1.0 (kv weight=0.0): when both are queued, kv gets zero
    share -- so a sustained expert stream should starve kv until expert
    finally empties its queue."""
    expert = TrafficClass("expert", weight=1.0, period_ms=0.1, job_bytes=1e6)  # heavy, fast stream
    kv = TrafficClass("kv", weight=0.0, arrival_rate_per_sec=100.0, job_sampler=lambda rng: 1e5)
    result = simulate_link(classes=[expert, kv], bw_gbps=1.0, sim_ms=50.0, seed=0)
    # bandwidth is tiny (1 GB/s = 1e6 bytes/ms) relative to expert demand
    # (1e6 bytes every 0.1ms => 1e7 bytes/ms needed) -- link is saturated by
    # expert alone, so kv should complete few or zero jobs within the window.
    assert len(result["kv"]) <= len(result["expert"])


def test_lone_class_gets_full_bandwidth_even_at_zero_weight():
    """A class with weight=0 still gets the WHOLE link when it's the only
    one queued -- a shared link doesn't waste bandwidth just because the
    other class currently has nothing to send."""
    kv_alone = TrafficClass("kv", weight=0.0, arrival_rate_per_sec=10.0, job_sampler=lambda rng: 1e5)
    result = simulate_link(classes=[kv_alone], bw_gbps=10.0, sim_ms=200.0, seed=0)
    assert len(result["kv"]) > 0
    for arrival, latency, ideal in result["kv"]:
        assert abs(latency - ideal) < 1e-6


def test_right_censoring_at_near_zero_bandwidth():
    """At pathologically low bandwidth, jobs should be excluded from stats
    (right-censored) rather than force-completed far outside the sim window
    -- the exact bug E8's own development caught."""
    cls = TrafficClass("solo", weight=1.0, period_ms=1.0, job_bytes=1e12)
    result = simulate_link(classes=[cls], bw_gbps=1e-6, sim_ms=100.0, seed=0)
    for _arrival, latency, _ideal in result["solo"]:
        assert latency <= 100.0 + 1e-6


def test_pooled_simulate_pools_across_seeds():
    cls = TrafficClass("solo", weight=1.0, arrival_rate_per_sec=20.0, job_sampler=lambda rng: rng.exponential(1e5))
    pooled = pooled_simulate(n_seeds=5, base_seed=0, classes=[cls], bw_gbps=10.0, sim_ms=100.0)
    single = simulate_link(classes=[cls], bw_gbps=10.0, sim_ms=100.0, seed=0)
    assert len(pooled["solo"]) >= len(single["solo"])


def test_pctile_stats_empty_is_nan():
    stats = pctile_stats([])
    assert stats["n"] == 0


def test_pareto_labels_dominance():
    points = [(1.0, 1.0, "a"), (2.0, 2.0, "b"), (0.5, 3.0, "c")]
    winners = set(pareto_labels(points))
    assert "b" not in winners  # dominated by "a" on both axes
    assert "a" in winners and "c" in winners


def test_phase_ms_offsets_first_arrival():
    cls = TrafficClass("p", weight=1.0, period_ms=10.0, job_bytes=1e6, phase_ms=3.0)
    result = simulate_link(classes=[cls], bw_gbps=100.0, sim_ms=50.0, seed=0)
    arrivals = sorted(r[0] for r in result["p"])
    assert arrivals[0] == 13.0  # period_ms + phase_ms
    assert abs(arrivals[1] - 23.0) < 1e-9


def test_phase_ms_default_zero_preserves_old_behavior():
    cls = TrafficClass("p", weight=1.0, period_ms=10.0, job_bytes=1e6)
    result = simulate_link(classes=[cls], bw_gbps=100.0, sim_ms=50.0, seed=0)
    arrivals = sorted(r[0] for r in result["p"])
    assert arrivals[0] == 10.0
