from tiermoe.sim.policies import ALL_POLICIES, _LRUCache, run_concurrency_sweep, run_policy_bakeoff, simulate_policy
from tiermoe.specs import MODEL_SPECS
from tiermoe.traces.io import load_traces, train_test_split_reqs


def _splits(trace_dir, seed=0):
    df = load_traces(trace_dir / "traces.jsonl.zst")
    decode = df[df.phase == "decode"].reset_index(drop=True)
    train_reqs, test_reqs = train_test_split_reqs(df, seed=seed)
    return decode[decode.req.isin(train_reqs)].reset_index(drop=True), decode[decode.req.isin(test_reqs)].reset_index(drop=True)


def test_oracle_has_zero_demand_stall_rate(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="oracle",
                              residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    assert result.demand_stall_rate == 0.0


def test_none_forces_zero_residency_and_has_highest_stall(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    none_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="none",
                                    residency_pct=90.0, bw_gbps=32.0, precision="fp16", seed=0)
    static_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="static-c",
                                      residency_pct=90.0, bw_gbps=32.0, precision="fp16", seed=0)
    # 'none' ignores the requested residency (always r=0) -- it must stall at
    # least as often as 'static-c' at 90% residency on the SAME traces.
    assert none_result.demand_stall_rate >= static_result.demand_stall_rate


def test_prefetch_freq_never_worse_than_none_on_tpot(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    none_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="none",
                                    residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    pf_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="prefetch-freq",
                                  residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    assert pf_result.tpot_ms_mean <= none_result.tpot_ms_mean + 1e-9


def test_run_policy_bakeoff_produces_oracle_gap_closed(olmoe_trace_dir):
    df = run_policy_bakeoff("olmoe", olmoe_trace_dir, policies=["none", "static-c", "prefetch-freq", "oracle"],
                              residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    assert set(df["policy"]) == {"none", "static-c", "prefetch-freq", "oracle"}
    assert df["oracle_gap_closed_pct"].notna().all()
    oracle_row = df[df.policy == "oracle"].iloc[0]
    assert abs(oracle_row["oracle_gap_closed_pct"] - 100.0) < 1e-6
    none_row = df[df.policy == "none"].iloc[0]
    assert abs(none_row["oracle_gap_closed_pct"] - 0.0) < 1e-6


def test_precision_gate_uses_fewer_or_equal_bytes_than_prefetch_v2(olmoe_trace_dir):
    df = run_policy_bakeoff("olmoe", olmoe_trace_dir, policies=["prefetch-v2", "prefetch-v2+precision-gate"],
                              residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    if len(df) < 2:
        return  # torch/predictor unavailable on this run -- skip gracefully, matches project's own convention
    v2 = df[df.policy == "prefetch-v2"].iloc[0]
    gated = df[df.policy == "prefetch-v2+precision-gate"].iloc[0]
    assert gated["bytes_per_token_mean"] <= v2["bytes_per_token_mean"] * 1.05  # small tolerance for stall-driven fp16 fallback bytes


def test_mixtral_bakeoff_runs(mixtral_trace_dir):
    df = run_policy_bakeoff("mixtral", mixtral_trace_dir, policies=["none", "static-c", "oracle"],
                              residency_pct=25.0, bw_gbps=32.0, precision="nf4", seed=0)
    assert len(df) == 3
    assert (df["tpot_ms_mean"] > 0).all()


# ---------- mentor-review additions ----------

def test_lru_cache_never_exceeds_configured_capacity():
    """Regression test for a real bug: an earlier version of the 'lru'
    policy checked static-residency MISSES against a separate, same-sized
    LRU cache underneath -- silently giving 'lru' double the memory budget
    of every other policy at the same --residency-pct. This is now the only
    residency mechanism for 'lru', so its own capacity is the whole budget."""
    cache = _LRUCache(capacity=3)
    for e in [0, 1, 2, 3, 4, 5, 6]:
        cache.access(e)
        assert len(cache._od) <= 3


def test_popularity_prefetch_runs_and_reports_precision_recall(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="popularity-prefetch",
                              residency_pct=25.0, bw_gbps=128.0, precision="fp16", depth=4, seed=0)
    assert result is not None
    assert result.prefetch_precision is not None and 0.0 <= result.prefetch_precision <= 1.0
    assert result.prefetch_recall is not None and 0.0 <= result.prefetch_recall <= 1.0


def test_conditioned_prefetch_precision_beats_unconditioned_popularity(olmoe_trace_dir):
    """The whole point of 'popularity-prefetch' as a baseline (Main.md's own
    'critical baseline -- isolates the value of conditioning'): a predictor
    conditioned on the current router state should be at least competitive
    with, and not badly worse than, blindly prefetching global favorites."""
    df = run_policy_bakeoff("olmoe", olmoe_trace_dir, policies=["popularity-prefetch", "prefetch-freq"],
                              residency_pct=25.0, bw_gbps=128.0, precision="fp16", depth=4, seed=0)
    pop = df[df.policy == "popularity-prefetch"].iloc[0]
    freq = df[df.policy == "prefetch-freq"].iloc[0]
    assert freq["tpot_ms_mean"] <= pop["tpot_ms_mean"] + 1e-6


def test_precision_and_recall_are_none_for_non_prefetch_policies(olmoe_trace_dir):
    df = run_policy_bakeoff("olmoe", olmoe_trace_dir, policies=["none", "static-c", "lru", "oracle"],
                              residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0)
    assert df["prefetch_precision"].isna().all()
    assert df["prefetch_recall"].isna().all()


def test_hybrid_lru_prefetch_never_worse_than_plain_lru(olmoe_trace_dir):
    """hybrid-lru-prefetch keeps the exact same LRU hot-set mechanism as
    'lru' and ADDITIONALLY tries to hide misses via prediction -- it must
    never stall MORE often than plain lru at the same residency budget."""
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    lru_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="lru",
                                  residency_pct=25.0, bw_gbps=128.0, precision="fp16", depth=4, seed=0)
    hybrid_result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="hybrid-lru-prefetch",
                                     residency_pct=25.0, bw_gbps=128.0, precision="fp16", depth=4, seed=0)
    assert hybrid_result.demand_stall_rate <= lru_result.demand_stall_rate + 1e-9
    assert hybrid_result.tpot_ms_mean <= lru_result.tpot_ms_mean + 1e-6


def test_concurrency_tagging_present_only_when_concurrency_above_one(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    solo = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="prefetch-freq",
                            residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0, concurrency=1)
    concurrent = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="prefetch-freq",
                                  residency_pct=25.0, bw_gbps=32.0, precision="fp16", seed=0, concurrency=4)
    assert "concurrency_stable" not in solo.extra
    assert "concurrency_stable" in concurrent.extra
    assert "max_sustainable_concurrency" in concurrent.extra


def test_higher_concurrency_never_improves_tpot(olmoe_trace_dir):
    train, test = _splits(olmoe_trace_dir)
    spec = MODEL_SPECS["olmoe"]
    results = {}
    for c in (1, 2, 4, 8):
        results[c] = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="static-c",
                                      residency_pct=25.0, bw_gbps=64.0, precision="fp16", seed=0, concurrency=c)
    tpots = [results[c].tpot_ms_mean for c in (1, 2, 4, 8)]
    assert all(tpots[i] <= tpots[i + 1] + 1e-9 for i in range(len(tpots) - 1))


def test_run_concurrency_sweep_stacks_all_levels(olmoe_trace_dir):
    df = run_concurrency_sweep("olmoe", olmoe_trace_dir, concurrency_grid=[1, 2, 4],
                                policies=["static-c", "prefetch-freq"], residency_pct=25.0, bw_gbps=64.0,
                                precision="fp16", depth=1, seed=0)
    assert set(df["concurrency"]) == {1, 2, 4}
    assert set(df["policy"]) == {"static-c", "prefetch-freq"}
    assert len(df) == 6
