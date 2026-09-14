from tiermoe.kv.cotenancy import _kv_weight_for_policy, recommend_policy, run_kv_expert_sweep


def test_kv_weight_for_policy():
    assert _kv_weight_for_policy("expert-first") == 0.0
    assert _kv_weight_for_policy("kv-first") == 1.0
    assert _kv_weight_for_policy("weighted:0.3") == 0.3


def test_kv_weight_for_policy_rejects_out_of_range():
    try:
        _kv_weight_for_policy("weighted:1.5")
        assert False
    except ValueError:
        pass


def test_run_kv_expert_sweep_olmoe(data_root):
    df = run_kv_expert_sweep(["olmoe"], [64.0], ["expert-first", "kv-first"], batch=32,
                               residency_pct=75.0, expert_precision="nf4", sim_seconds=5.0,
                               n_seeds=2, n_trials=50, data_root=data_root)
    assert len(df) == 2
    assert "pareto_optimal" in df.columns
    kv_first = df[df.policy == "kv-first"].iloc[0]
    expert_first = df[df.policy == "expert-first"].iloc[0]
    # kv-first must give KV at least as good (low) a TTFT as expert-first.
    assert kv_first["ttft_p50_ms"] <= expert_first["ttft_p50_ms"] + 1e-6


def test_recommend_policy_reports_pareto_extremes_without_slo(data_root):
    df = run_kv_expert_sweep(["olmoe"], [64.0], ["expert-first", "kv-first", "weighted:0.5"], batch=32,
                               residency_pct=75.0, expert_precision="nf4", sim_seconds=5.0,
                               n_seeds=2, n_trials=50, data_root=data_root)
    recs = recommend_policy(df, ttft_slo_ms=None)
    assert len(recs) >= 1
    for v in recs.values():
        assert v["recommended_policy"] is None  # no SLO given -- Pareto extremes only
        assert "best_for_ttft" in v and "best_for_tpot" in v
