from tierahead.baseline.compare import (
    capacity_check, capacity_check_with_kv, kv_hbm_pressure, run_baseline_vs_cxl, run_flagship_report,
)


def test_capacity_check_olmoe_fits_in_80gb():
    result = capacity_check("olmoe", precision="fp16", hbm_budget_gb=80.0)
    assert result["fits_in_hbm_alone"]
    assert result["overflow_gb"] == 0.0


def test_capacity_check_mixtral_fp16_does_not_fit_in_80gb():
    """This is the memory-wall arithmetic Main.md leads with: Mixtral-8x7B
    FP16 (~87GB of expert weights alone, before attention/embedding) does
    not fit an 80GB H100."""
    result = capacity_check("mixtral", precision="fp16", hbm_budget_gb=80.0)
    assert not result["fits_in_hbm_alone"]
    assert result["overflow_gb"] > 0


def test_capacity_check_mixtral_nf4_fits_in_80gb():
    result = capacity_check("mixtral", precision="nf4", hbm_budget_gb=80.0)
    assert result["fits_in_hbm_alone"]


def test_run_baseline_vs_cxl_end_to_end_olmoe(data_root):
    result = run_baseline_vs_cxl("olmoe", residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                                   policies=["none", "static-c", "prefetch-freq", "oracle"],
                                   backend="des", seed=0, data_root=data_root)
    assert result["backend"] == "des"
    assert result["baseline_tpot_ms"] > 0
    assert result["hbm_bytes_freed_gb"] > 0
    assert len(result["policies_df"]) == 4
    assert "tpot_overhead_vs_baseline_pct" in result["policies_df"].columns
    assert isinstance(result["headline"], str) and len(result["headline"]) > 0


def test_run_baseline_vs_cxl_mixtral_overflow_headline(data_root):
    """Mixtral FP16 doesn't fit HBM alone -- the headline sentence must say
    so, not silently treat it as a normal tiering trade-off."""
    result = run_baseline_vs_cxl("mixtral", residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                                   policies=["none"], backend="des", hbm_budget_gb=80.0, seed=0, data_root=data_root)
    assert "does NOT fit" in result["headline"]


def test_run_baseline_vs_cxl_explicit_cxlmemsim_backend_raises_clearly(data_root):
    try:
        run_baseline_vs_cxl("olmoe", backend="cxlmemsim", data_root=data_root)
        assert False, "should have raised on this machine"
    except RuntimeError as e:
        assert "cxlmemsim" in str(e).lower()


# ---------- mentor-review addition: KV-cache HBM pressure ----------

def test_kv_hbm_pressure_reduces_available_residency_as_concurrency_grows():
    """KV cache for concurrent long-context requests eats into the SAME HBM
    budget resident experts need -- more concurrent long-context load should
    monotonically shrink the achievable residency ceiling."""
    # n=1,2 both land in the "KV easily fits, residency ceiling clamped at
    # 100%" plateau (~17GB/62GB and ~34GB/80GB of KV are both well under the
    # ~12.9GB experts alone need) -- spanning n=1,4,6 crosses out of that
    # plateau and into "KV alone exceeds the whole budget," where the
    # monotonic decrease is actually visible.
    results = [kv_hbm_pressure("olmoe", hbm_budget_gb=80.0, concurrent_requests=n, context_len=131072)
               for n in (1, 4, 6)]
    ceilings = [r["max_residency_pct_given_kv"] for r in results]
    assert ceilings[0] > ceilings[1] > ceilings[2]
    assert all(0.0 <= c <= 100.0 for c in ceilings)


def test_kv_hbm_pressure_flags_when_kv_alone_exceeds_budget():
    # olmoe kv_bytes_per_token = 131072 B/token (E8_KV_COTENANCY.md's own
    # documented coincidence, shared with mixtral) -> ~17.18GB/request @128k
    # context; 5 concurrent such requests = ~85.9GB > an 80GB HBM budget.
    result = kv_hbm_pressure("olmoe", hbm_budget_gb=80.0, concurrent_requests=5, context_len=131072)
    assert result["kv_alone_exceeds_hbm_budget"]
    assert result["max_residency_pct_given_kv"] == 0.0
    assert result["hbm_left_for_experts_gb"] == 0.0


def test_capacity_check_with_kv_matches_plain_capacity_check_at_zero_kv():
    """context_len effectively 0 concurrent load should reduce to the same
    expert-only-fits verdict as the plain capacity_check."""
    plain = capacity_check("olmoe", "fp16", 80.0)
    with_kv = capacity_check_with_kv("olmoe", "fp16", 80.0, concurrent_requests=1, context_len=1)
    assert with_kv["fits_in_hbm_alone"] == plain["fits_in_hbm_alone"]
    assert abs(with_kv["total_needed_with_kv_gb"] - plain["total_expert_weight_gb"]) < 0.01


def test_capacity_check_with_kv_can_flip_a_fitting_model_to_not_fitting():
    """OLMoE's expert weights alone fit comfortably in 80GB, but stacking
    enough concurrent long-context KV cache on top of them should NOT --
    accounting for KV is not just a residency-ceiling nuance, it can change
    whether the 'normal' baseline is even feasible at all."""
    result = capacity_check_with_kv("olmoe", "fp16", hbm_budget_gb=80.0, concurrent_requests=6, context_len=131072)
    assert result["fits_in_hbm_alone"]  # expert weights alone still fit
    assert not result["fits_in_hbm_with_kv"]  # but not once KV cache is stacked on top
    assert result["overflow_with_kv_gb"] > 0


def test_run_baseline_vs_cxl_caps_residency_when_kv_pressure_bites(data_root):
    result = run_baseline_vs_cxl("olmoe", residency_pct=90.0, bw_gbps=32.0, precision="fp16",
                                   policies=["static-c"], backend="des", seed=0, data_root=data_root,
                                   concurrent_requests=4, context_len=131072, hbm_budget_gb=80.0)
    assert result["kv_pressure"] is not None
    assert result["effective_residency_pct"] < result["residency_pct"]
    assert "KV cache" in result["headline"]


def test_run_baseline_vs_cxl_default_ignores_kv_pressure_when_context_len_zero(data_root):
    """Default context_len=0 preserves old behavior exactly -- no KV
    accounting unless explicitly requested."""
    result = run_baseline_vs_cxl("olmoe", residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                                   policies=["static-c"], backend="des", seed=0, data_root=data_root)
    assert result["kv_pressure"] is None
    assert result["effective_residency_pct"] == result["residency_pct"]


# ---------- mentor-review addition: single-call flagship report (KV + concurrency together) ----------

def test_run_flagship_report_default_grid_matches_single_baseline_call(data_root):
    """concurrency_grid=None (default) must be a strict superset of a single
    run_baseline_vs_cxl(concurrency=1) call -- same numbers, just wrapped."""
    single = run_baseline_vs_cxl("olmoe", residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                                   policies=["none", "static-c", "oracle"], backend="des", seed=0,
                                   data_root=data_root)
    flagship = run_flagship_report("olmoe", residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                                     policies=["none", "static-c", "oracle"], backend="des", seed=0,
                                     data_root=data_root)
    assert flagship["concurrency_grid"] == [1]
    assert flagship["by_concurrency"][1]["headline"] == single["headline"]
    assert len(flagship["combined_policies_df"]) == len(single["policies_df"])


def test_run_flagship_report_stacks_every_concurrency_level(data_root):
    grid = [1, 2, 4]
    result = run_flagship_report("olmoe", residency_pct=25.0, bw_gbps=128.0, precision="fp16", depth=4,
                                   policies=["static-c", "hybrid-lru-prefetch", "oracle"], backend="des", seed=0,
                                   data_root=data_root, concurrency_grid=grid)
    assert result["concurrency_grid"] == grid
    assert set(result["by_concurrency"].keys()) == set(grid)
    assert set(result["headlines"].keys()) == set(grid)
    assert len(result["combined_policies_df"]) == len(grid) * 3
    assert sorted(result["combined_policies_df"]["concurrency"].unique().tolist()) == grid
    # oracle's TPOT must be the same at every concurrency level (see
    # tierahead.sim.validation.gate_oracle_invariant_to_concurrency).
    oracle_tpots = result["combined_policies_df"].loc[
        result["combined_policies_df"].policy == "oracle", "tpot_ms_mean"
    ].unique()
    assert len(oracle_tpots) == 1


def test_run_flagship_report_carries_kv_pressure_through(data_root):
    result = run_flagship_report("olmoe", residency_pct=90.0, bw_gbps=32.0, precision="fp16",
                                   policies=["static-c"], backend="des", seed=0, data_root=data_root,
                                   concurrency_grid=[1, 2], concurrent_requests=4, context_len=131072,
                                   hbm_budget_gb=80.0)
    assert result["kv_pressure"] is not None
    assert result["kv_pressure"]["kv_alone_exceeds_hbm_budget"] is False
    for headline in result["headlines"].values():
        assert "KV cache" in headline
