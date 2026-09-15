import numpy as np
import pandas as pd

from tierahead.analyze.workload import gini, recall_at_m, run_characterization, verdict


def test_gini_zero_for_uniform():
    assert gini(np.ones(10)) < 1e-6


def test_gini_near_one_for_extreme_skew():
    freqs = np.array([1000.0] + [1.0] * 99)
    assert gini(freqs) > 0.8


def test_verdict_thresholds():
    assert verdict(0.65, 0.65, 0.8)[0] == "STRONG_GO"
    assert verdict(0.65, 0.30, 0.5)[0] == "GO_PLACEMENT_LED"
    assert verdict(0.30, 0.65, 0.5)[0] == "GO_PREFETCH_LED"
    assert verdict(0.30, 0.30, 0.5)[0] == "PIVOT"


def test_run_characterization_olmoe_matches_committed_pilot_summary(olmoe_trace_dir, data_root):
    """Reproduces PILOT_FINDINGS.md's committed OLMoE numbers from the SAME
    committed traces via this framework's own (ported) pipeline -- if this
    ever drifts from results/pilot_summary.json, either this port has a bug
    or the committed summary is stale; either way this test should fail
    loudly rather than let the two silently diverge."""
    result = run_characterization(olmoe_trace_dir, seed=0)
    assert abs(result["headline"]["coverage_25pct_deployable"] - 0.5223) < 0.01
    assert abs(result["headline"]["recall_2k_d1_nonresident"] - 0.5532) < 0.01
    assert result["verdict"]["label"] == "PIVOT"


def test_run_characterization_mixtral_matches_committed_summary(mixtral_trace_dir):
    result = run_characterization(mixtral_trace_dir, seed=0)
    assert abs(result["headline"]["coverage_25pct_deployable"] - 0.2866) < 0.01
    assert abs(result["headline"]["recall_2k_d1_nonresident"] - 0.6180) < 0.01
    assert result["verdict"]["label"] == "GO_PREFETCH_LED"


def test_run_characterization_reports_precision_headline(olmoe_trace_dir):
    """Mentor-review addition: precision must be reported alongside recall.
    Sanity range check on real data (the exact hand-computed identity is
    tested separately below on a synthetic case)."""
    result = run_characterization(olmoe_trace_dir, seed=0)
    p = result["headline"]["precision_2k_d1_nonresident"]
    assert 0.0 <= p <= 1.0
    # m=2k always fetches more candidates than the k actually needed, so
    # precision must be strictly below 1 whenever recall is (real traces
    # never hit the degenerate m==n_experts saturation case at m=2k here).
    assert p < 1.0


def test_precision_nonresident_is_not_a_fixed_rescaling_of_recall_nonresident():
    """precision_overall is an exact algebraic rescaling of recall_overall
    by k/m (both denominators are data-independent constants -- see
    recall_at_m's own docstring) -- NOT genuinely new information.
    precision_nonresident is the informative one, because its denominator
    (how many of the k actual experts are non-resident) varies row to row.
    This is a hand-computable synthetic case that proves it's NOT just
    recall_nonresident * k/m: by construction, recall_nonresident=2/3 but
    precision_nonresident=0.25 (2/3 * 1/2 = 0.333 != 0.25)."""
    n_experts, k = 4, 1
    # From expert 0, the fitted table ranks experts 1 and 2 highest (top-2 = {1,2}).
    table = np.array([
        [0.05, 0.50, 0.35, 0.10],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ])
    tables = {1: {0: table}}

    rows = []
    # 4 requests, all starting from expert 0 at layer 0, landing on a
    # different target expert at layer 1: 1 (hit, nonresident), 3 (the
    # resident expert -- excluded from the nonresident denominator
    # entirely), 2 (hit, nonresident), 0 (miss, nonresident).
    for i, target in enumerate([1, 3, 2, 0]):
        rows.append({"req": f"r{i}", "phase": "decode", "layer": 0, "tok": 0, "topk": (0,)})
        rows.append({"req": f"r{i}", "phase": "decode", "layer": 1, "tok": 0, "topk": (target,)})
    df = pd.DataFrame(rows)
    test_reqs = {f"r{i}" for i in range(4)}

    # Expert 3 dominates layer-1 train counts -> the only resident expert at
    # c_for_nonresident=25% (top ceil(4*0.25)=1 expert).
    train_counts = pd.DataFrame([
        {"layer": 1, "expert": 3, "count": 100},
        {"layer": 1, "expert": 0, "count": 1},
        {"layer": 1, "expert": 1, "count": 1},
        {"layer": 1, "expert": 2, "count": 1},
        {"layer": 0, "expert": 0, "count": 1},
    ])

    result = recall_at_m(df, test_reqs, tables, train_counts, n_experts=n_experts, k=k, n_layers=2,
                          d_values=[1], m_mults=[(2, "2k")], c_for_nonresident=25.0)
    entry = result["d1_m2k"]
    assert entry["m"] == 2
    assert abs(entry["recall_nonresident"] - 2 / 3) < 1e-9
    assert abs(entry["precision_nonresident"] - 0.25) < 1e-9
    # The would-be-redundant rescaling (recall * k/m) gives 0.333, not the
    # actual 0.25 -- proving precision_nonresident carries real information.
    assert abs(entry["precision_nonresident"] - entry["recall_nonresident"] * k / 2) > 0.05


def test_precision_overall_is_exact_rescaling_of_recall_overall():
    """The companion identity: precision_overall SHOULD equal
    recall_overall * k/m exactly (both denominators -- k*n_rows and
    m*n_rows -- are data-independent), unlike the nonresident pair above."""
    n_experts, k = 4, 1
    table = np.array([
        [0.05, 0.50, 0.35, 0.10],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ])
    tables = {1: {0: table}}
    rows = []
    for i, target in enumerate([1, 3, 2, 0]):
        rows.append({"req": f"r{i}", "phase": "decode", "layer": 0, "tok": 0, "topk": (0,)})
        rows.append({"req": f"r{i}", "phase": "decode", "layer": 1, "tok": 0, "topk": (target,)})
    df = pd.DataFrame(rows)
    test_reqs = {f"r{i}" for i in range(4)}
    train_counts = pd.DataFrame([
        {"layer": 1, "expert": 3, "count": 100}, {"layer": 1, "expert": 0, "count": 1},
        {"layer": 1, "expert": 1, "count": 1}, {"layer": 1, "expert": 2, "count": 1},
        {"layer": 0, "expert": 0, "count": 1},
    ])
    result = recall_at_m(df, test_reqs, tables, train_counts, n_experts=n_experts, k=k, n_layers=2,
                          d_values=[1], m_mults=[(2, "2k")], c_for_nonresident=25.0)
    entry = result["d1_m2k"]
    assert abs(entry["precision_overall"] - entry["recall_overall"] * k / entry["m"]) < 1e-9
