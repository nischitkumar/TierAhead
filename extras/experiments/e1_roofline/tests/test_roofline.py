"""Unit tests for E1's roofline model. Pure Python/numpy -- no GPU, no
network, no trace files required. Run with:
    pytest experiments/e1_roofline/tests/test_roofline.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # e1_roofline/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/

import roofline as rl  # noqa: E402
from common.model_specs import MODEL_SPECS, EXPERT_BYTES_FP16_TABLE_MB, NF4_BYTE_FACTOR  # noqa: E402
from common.eunion import empirical_e_union, analytic_e_union_uniform  # noqa: E402


# ---------- model_specs sanity ----------

@pytest.mark.parametrize("model_tag", ["olmoe", "mixtral"])
def test_expert_bytes_matches_experiments_md_table(model_tag):
    spec = MODEL_SPECS[model_tag]
    computed_mb = spec.expert_bytes("fp16") / 1e6
    table_mb = EXPERT_BYTES_FP16_TABLE_MB[model_tag]
    assert computed_mb == pytest.approx(table_mb, rel=0.02), (
        f"{model_tag}: architecture-derived expert size ({computed_mb:.2f} MB) diverges "
        f"from Experiments.md's table ({table_mb} MB) by more than 2% -- check hidden_size/"
        f"expert_intermediate_size in model_specs.py")


def test_nf4_smaller_than_fp16_by_documented_factor():
    for spec in MODEL_SPECS.values():
        fp16 = spec.expert_bytes("fp16")
        nf4 = spec.expert_bytes("nf4")
        assert nf4 < fp16
        assert nf4 / fp16 == pytest.approx(NF4_BYTE_FACTOR, rel=1e-9)


def test_expert_bytes_rejects_unknown_precision():
    spec = MODEL_SPECS["olmoe"]
    with pytest.raises(ValueError):
        spec.expert_bytes("int8")


# ---------- bytes / transfer / compute arithmetic ----------

def test_bytes_per_layer_scales_with_residency():
    spec = MODEL_SPECS["olmoe"]
    e_union_val = 32.0
    full = rl.bytes_per_layer(spec, batch=1, residency_pct=0, precision="fp16", e_union_val=e_union_val)
    half = rl.bytes_per_layer(spec, batch=1, residency_pct=50, precision="fp16", e_union_val=e_union_val)
    none_resident_at_100 = rl.bytes_per_layer(spec, batch=1, residency_pct=100, precision="fp16", e_union_val=e_union_val)
    assert half == pytest.approx(full / 2)
    assert none_resident_at_100 == pytest.approx(0.0)


def test_bytes_per_layer_scales_with_e_union():
    spec = MODEL_SPECS["mixtral"]
    b1 = rl.bytes_per_layer(spec, batch=1, residency_pct=0, precision="fp16", e_union_val=4.0)
    b2 = rl.bytes_per_layer(spec, batch=1, residency_pct=0, precision="fp16", e_union_val=8.0)
    assert b2 == pytest.approx(2 * b1)


def test_t_transfer_halves_when_bandwidth_doubles():
    bytes_val = 1e9  # 1 GB
    t32 = rl.t_transfer_ms(bytes_val, bw_gbps=32)
    t64 = rl.t_transfer_ms(bytes_val, bw_gbps=64)
    assert t32 == pytest.approx(2 * t64)


def test_t_transfer_zero_bw_is_infinite():
    assert rl.t_transfer_ms(1e6, bw_gbps=0) == float("inf")


def test_t_compute_flop_scales_linearly_with_batch():
    spec = MODEL_SPECS["olmoe"]
    t1 = rl.t_compute_flop_ms(spec, batch=1)
    t8 = rl.t_compute_flop_ms(spec, batch=8)
    assert t8 == pytest.approx(8 * t1, rel=1e-9)


def test_rho_monotonic_in_bandwidth_and_residency():
    spec = MODEL_SPECS["mixtral"]
    e_union_val = 8.0  # all experts touched (mixtral has only 8/layer)
    t_compute = rl.t_compute_flop_ms(spec, batch=1)

    bpl_r0 = rl.bytes_per_layer(spec, 1, 0, "fp16", e_union_val)
    bpl_r50 = rl.bytes_per_layer(spec, 1, 50, "fp16", e_union_val)
    rho_bw32_r0 = rl.rho(rl.t_transfer_ms(bpl_r0, 32), t_compute)
    rho_bw64_r0 = rl.rho(rl.t_transfer_ms(bpl_r0, 64), t_compute)
    rho_bw32_r50 = rl.rho(rl.t_transfer_ms(bpl_r50, 32), t_compute)

    assert rho_bw64_r0 < rho_bw32_r0, "doubling bandwidth must not increase rho"
    assert rho_bw32_r50 < rho_bw32_r0, "raising residency must not increase rho"


def test_rho_zero_compute_time_is_infinite():
    assert rl.rho(t_transfer_val_ms=5.0, t_compute_val_ms=0.0) == float("inf")


def test_required_bw_gbps_is_self_consistent_with_rho():
    """The bandwidth required to hit rho==1 should, when plugged back in,
    produce rho very close to 1."""
    spec = MODEL_SPECS["olmoe"]
    bpl = rl.bytes_per_layer(spec, batch=8, residency_pct=25, precision="fp16", e_union_val=20.0)
    t_compute = rl.t_compute_flop_ms(spec, batch=8)
    req_bw = rl.required_bw_gbps(bpl, t_compute)
    tt = rl.t_transfer_ms(bpl, req_bw)
    assert rl.rho(tt, t_compute) == pytest.approx(1.0, rel=1e-6)


# ---------- compute-time provenance ----------

def test_get_t_compute_prefers_measured_over_flop_estimate():
    spec = MODEL_SPECS["olmoe"]
    calib = {"olmoe": {"by_batch": {"8": {"mean_ms": 1.2345}}}}
    val, prov = rl.get_t_compute_ms(spec, batch=8, calib=calib)
    assert prov == "measured_here"
    assert val == pytest.approx(1.2345)


def test_get_t_compute_falls_back_when_batch_not_in_calib():
    spec = MODEL_SPECS["olmoe"]
    calib = {"olmoe": {"by_batch": {"8": {"mean_ms": 1.2345}}}}
    val, prov = rl.get_t_compute_ms(spec, batch=32, calib=calib)  # 32 not calibrated
    assert prov == "flop_estimate"
    assert val == pytest.approx(rl.t_compute_flop_ms(spec, 32))


def test_get_t_compute_falls_back_when_no_calib_at_all():
    spec = MODEL_SPECS["mixtral"]
    val, prov = rl.get_t_compute_ms(spec, batch=1, calib=None)
    assert prov == "flop_estimate"
    assert val > 0


# ---------- sweep_grid / provisioning_table shape ----------

def test_sweep_grid_row_count():
    models = ["olmoe", "mixtral"]
    batch_grid = [1, 8]
    residency_grid = [0, 50]
    precision_grid = ["fp16", "nf4"]
    bw_grid = [32, 64]
    e_union_lookups = {"olmoe": {1: 10.0, 8: 30.0}, "mixtral": {1: 5.0, 8: 7.0}}
    df = rl.sweep_grid(models, batch_grid, residency_grid, precision_grid, bw_grid, e_union_lookups)
    expected = len(models) * len(batch_grid) * len(residency_grid) * len(precision_grid) * len(bw_grid)
    assert len(df) == expected
    assert set(df.columns) >= {"model", "batch", "residency_pct", "precision", "bw_gbps", "rho"}


def test_sweep_grid_100pct_residency_zero_bytes_zero_rho():
    e_union_lookups = {"olmoe": {1: 10.0}}
    df = rl.sweep_grid(["olmoe"], [1], [100], ["fp16"], [32], e_union_lookups)
    assert (df["bytes_per_layer"] == 0).all()
    assert (df["rho"] == 0).all()


def test_provisioning_table_finds_nearest_tier_correctly():
    e_union_lookups = {"olmoe": {1: 10.0}}
    df = rl.sweep_grid(["olmoe"], [1], [0], ["fp16"], [16, 32, 64, 128], e_union_lookups)
    table = rl.provisioning_table(df)
    row = table.iloc[0]
    grid = [16, 32, 64, 128]
    if row["required_bw_gbps_for_rho1"] > 128:
        assert np.isnan(row["nearest_supported_link_gbps"])
    else:
        assert row["nearest_supported_link_gbps"] in grid
        assert row["nearest_supported_link_gbps"] >= row["required_bw_gbps_for_rho1"]


def test_provisioning_table_missing_tier_is_nan_not_none_even_with_mixed_rows():
    """Regression test: a real multi-row sweep mixes rows that DO reach a
    supported tier with rows that DON'T (need >128 GB/s). Found via an actual
    end-to-end run against real traces: pandas upcast a mixed None/int column
    to float64, silently turning `is None` checks in the renderer into a
    literal 'nan' string in RESULTS.md instead of the intended
    '**exceeds 128 GB/s**' callout. nearest_tier() must return NaN (never
    None) so the column stays a consistent, always-float dtype."""
    e_union_lookups = {"olmoe": {1: 10.0, 32: 10.0}}
    # batch=1's tiny compute time makes it unreachable in this grid; batch=32's
    # much larger compute time (linear in batch under the flop model) makes it
    # reachable -- mirrors the real olmoe/fp16 sweep where only batch=32 rows
    # were reachable at any residency.
    df = rl.sweep_grid(["olmoe"], [1, 32], [0], ["fp16"], [16, 32, 64, 128], e_union_lookups)
    table = rl.provisioning_table(df)
    assert table["nearest_supported_link_gbps"].dtype == np.float64
    unreachable = table[table["required_bw_gbps_for_rho1"] > 128]
    reachable = table[table["required_bw_gbps_for_rho1"] <= 128]
    assert len(unreachable) > 0 and len(reachable) > 0, "test setup should produce both cases"
    assert unreachable["nearest_supported_link_gbps"].isna().all()
    assert reachable["nearest_supported_link_gbps"].notna().all()


def test_headline_sentences_flags_unreachable_regime():
    """Construct a deliberately extreme case (huge e_union, no residency) so
    that even the best grid point can't reach rho<1 -- the headline sentence
    must say so explicitly rather than silently reporting a misleading 'best' row."""
    e_union_lookups = {"mixtral": {1: 8.0, 8: 8.0, 32: 8.0}}
    df = rl.sweep_grid(["mixtral"], [1, 8, 32], [0], ["fp16"], [16], e_union_lookups)
    sentences = rl.headline_sentences(df)
    assert "NOT reachable" in sentences["mixtral_fp16"]


def test_headline_sentences_reports_viable_point_when_one_exists():
    e_union_lookups = {"mixtral": {1: 1.0, 8: 1.0, 32: 1.0}}
    df = rl.sweep_grid(["mixtral"], [1, 8, 32], [0, 75], ["fp16"], [16, 128], e_union_lookups)
    sentences = rl.headline_sentences(df)
    assert "NOT reachable" not in sentences["mixtral_fp16"]
    assert "residency >=" in sentences["mixtral_fp16"]


# ---------- E_union resampling (shared with E2) ----------

def test_empirical_e_union_batch1_equals_mean_set_size():
    sets = [frozenset({1, 2}), frozenset({3, 4}), frozenset({1, 2})]
    est = empirical_e_union(sets, batch=1, n_trials=1000, seed=0)
    assert est.mean == pytest.approx(2.0)  # every set here has size 2


def test_empirical_e_union_matches_brute_force_enumeration():
    """4 rows with known sets; B=2 has only C(4,2)=6 combos -- enumerate them
    by hand and check the resampler's mean converges to the exact expectation."""
    sets = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3}), frozenset({0, 3})]
    from itertools import combinations
    exact_unions = [len(sets[i] | sets[j]) for i, j in combinations(range(4), 2)]
    exact_mean = np.mean(exact_unions)
    est = empirical_e_union(sets, batch=2, n_trials=5000, seed=0)
    assert est.mean == pytest.approx(exact_mean, abs=0.05)


def test_empirical_e_union_monotonically_nondecreasing_in_batch():
    rng = np.random.default_rng(0)
    sets = [frozenset(rng.choice(64, size=8, replace=False).tolist()) for _ in range(200)]
    means = [empirical_e_union(sets, b, n_trials=300, seed=1).mean for b in [1, 2, 4, 8, 16]]
    assert all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))


def test_empirical_e_union_batch_capped_at_n_rows():
    sets = [frozenset({0, 1}), frozenset({2, 3})]
    est = empirical_e_union(sets, batch=10, n_trials=50, seed=0)
    assert est.batch == 2  # capped, can't sample 10 distinct rows from a pool of 2


def test_analytic_e_union_uniform_matches_simulation_for_uniform_random_draws():
    """When experts really are drawn uniformly at random (the null model),
    the closed-form approximation should track a Monte-Carlo simulation of
    that same null model reasonably closely."""
    rng = np.random.default_rng(0)
    n_experts, k, batch = 64, 8, 8
    sets = [frozenset(rng.choice(n_experts, size=k, replace=False).tolist()) for _ in range(3000)]
    empirical = empirical_e_union(sets, batch, n_trials=1000, seed=2).mean
    analytic = analytic_e_union_uniform(n_experts, k, batch)
    assert analytic == pytest.approx(empirical, rel=0.1)


def test_analytic_e_union_uniform_edge_cases():
    assert analytic_e_union_uniform(0, 8, 4) == 0.0
    assert analytic_e_union_uniform(64, 8, 0) == 0.0
    assert analytic_e_union_uniform(64, 8, 1) == pytest.approx(8.0)


# ---------- render_results_md: lock in the None/NaN tier-display fix ----------

def _fake_summary(tmp_tiers):
    """Minimal fake roofline_summary.json covering every field render() reads,
    with a controllable list of nearest_supported_link_gbps values (mixing
    None, float('nan'), and a real number, exactly as a real run can)."""
    prov_rows = [{"model": "olmoe", "precision": "fp16", "batch": 1, "residency_pct": 0,
                  "required_bw_gbps_for_rho1": 999.0, "nearest_supported_link_gbps": t,
                  "compute_provenance": "flop_estimate"} for t in tmp_tiers]
    return {
        "compute_provenance": {"olmoe": "flop_estimate"},
        "e_union_by_batch": {"olmoe": {"1": 8.0, "8": 30.0, "32": 50.0}},
        "batch_grid": [1, 8, 32],
        "headline_sentences": {"olmoe_fp16": "test sentence"},
        "bw_gbps_grid": [16, 32, 64, 128],
        "provisioning_table": prov_rows,
        "achievable_points_from_pilot": {"olmoe": []},
        "figures": [],
    }


def test_render_results_md_shows_exceeds_callout_for_none_and_nan_tiers(tmp_path):
    import render_results_md as rmd

    summary_path = tmp_path / "roofline_summary.json"
    summary_path.write_text(json.dumps(_fake_summary([None, float("nan"), 64.0])))
    out_path = tmp_path / "RESULTS.md"
    rmd.render(summary_path, out_path)
    text = out_path.read_text()
    assert text.count("**exceeds 128 GB/s**") == 2, (
        "both the None-valued row and the NaN-valued row must render as "
        "'exceeds 128 GB/s', not as a literal 'nan' string"
    )
    assert "nan |" not in text.lower().replace("**exceeds 128 gb/s**", "")
    assert "| 64" in text
