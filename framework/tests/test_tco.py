from tiermoe.tco.pooling import simulate_fleet_demand, stranding_analysis, tco_crossover, tokens_per_sec_per_dollar


def test_simulate_fleet_demand_shape_and_positivity():
    demands = simulate_fleet_demand(n_hosts=50, mean_gb=200.0, seed=0)
    assert len(demands) == 50
    assert (demands > 0).all()


def test_pooling_reduces_stranding():
    demands = simulate_fleet_demand(n_hosts=64, mean_gb=200.0, cv=0.4, seed=1)
    rows = {r["group_size"]: r for r in stranding_analysis(demands, [1, 4, 16])}
    # Pooling MORE hosts together should never increase stranded capacity
    # relative to pooling fewer (Pond's central stranding-reduction claim).
    assert rows[16]["stranded_pooled_gb"] <= rows[4]["stranded_pooled_gb"] + 1e-6
    assert rows[4]["stranded_pooled_gb"] <= rows[1]["stranded_pooled_gb"] + 1e-6
    assert rows[16]["stranding_reduction_pct"] >= rows[1]["stranding_reduction_pct"] - 1e-6


def test_tokens_per_sec_per_dollar_zero_cost_is_inf():
    assert tokens_per_sec_per_dollar(100.0, 0.0, 0.0, 1.0, 1.0) == float("inf")


def test_tco_crossover_sweep_shape():
    rows = tco_crossover(tokens_per_sec_hbm_only=1000.0, tokens_per_sec_hbm_plus_cxl=850.0, gb_hbm_only=80.0,
                          gb_hbm_hybrid=20.0, gb_cxl_hybrid=90.0)
    assert len(rows) == 4
    for r in rows:
        assert r["cost_per_gb_hbm"] > r["cost_per_gb_cxl"]  # HBM must always cost more per GB in this sweep
