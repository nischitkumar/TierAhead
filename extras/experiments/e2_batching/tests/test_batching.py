"""Unit tests for E2's batching-sweet-spot logic. Pure Python/numpy/pandas --
no GPU, no network. The predictor-recall tests build a tiny synthetic traces
DataFrame in the exact schema collect.py writes (req/tok/phase/layer/topk),
so they exercise the real elp_probe/src/analyze.py machinery (build_transition_tables)
that E2 reuses, not a mock of it.

Run with:
    pytest experiments/e2_batching/tests/test_batching.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                    # e2_batching/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / ".."))   # (no-op, keeps path stable)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))                    # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e1_roofline"))    # e1_roofline/

import batching_sweep as bs  # noqa: E402
import roofline as rl  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402


# ---------- bytes_per_token ----------

def test_bytes_per_token_is_bytes_per_layer_divided_by_batch():
    spec = MODEL_SPECS["olmoe"]
    for batch in [1, 4, 16]:
        expected = rl.bytes_per_layer(spec, batch, 25.0, "fp16", e_union_val=30.0) / batch
        got = bs.bytes_per_token(spec, batch, 25.0, "fp16", e_union_val=30.0)
        assert got == pytest.approx(expected)


def test_bytes_per_token_decreases_as_batch_grows_when_e_union_saturates():
    """If E_union(B) is flat (e.g. capped at n_experts, as happens for Mixtral's
    8-expert layers at moderate batch), bytes/token must fall roughly as 1/B --
    the amortization effect the whole experiment is about."""
    spec = MODEL_SPECS["mixtral"]
    saturated = spec.n_experts  # every expert touched, regardless of B
    vals = [bs.bytes_per_token(spec, b, 0.0, "fp16", e_union_val=saturated) for b in [1, 2, 4, 8]]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] == pytest.approx(vals[0] / 8, rel=1e-9)


# ---------- per_token_time / find_b_star ----------

def test_total_tpot_is_undivided_batch_step_time_not_per_token_time_times_n_layers():
    """Regression test for a real modeling bug found by inspecting an actual
    run's numbers: total_tpot_ms must be n_layers * the UNDIVIDED batch-step
    time (every request in a batch waits for the same layer forward pass, so
    latency does not shrink with batch), NOT n_layers * per_token_time_ms
    (which IS divided by batch and is a throughput metric). Getting this
    wrong makes total_tpot monotonically DEcrease with batch, so a p99 SLO
    could never bind and B* would always be the largest grid point.

    Construct a compute-bound regime (tiny e_union -> negligible transfer)
    where t_compute_batch scales linearly with batch under the flop model:
    total_tpot_ms must then scale ~linearly with batch too, and must NOT
    equal n_layers * per_token_time_ms (which is batch-invariant here, see
    test_per_token_compute_component_is_batch_invariant_under_flop_model)."""
    spec = MODEL_SPECS["olmoe"]
    tiny_e_union = {1: 1e-6, 8: 1e-6}  # negligible bytes -> compute-bound at every batch
    _, df, _ = bs.find_b_star(spec, [1, 8], residency_pct=0.0, precision="fp16", bw_gbps=128.0,
                               e_union_by_batch=tiny_e_union, calib=None)
    tpot1 = df.loc[df.batch == 1, "total_tpot_ms"].iloc[0]
    tpot8 = df.loc[df.batch == 8, "total_tpot_ms"].iloc[0]
    pt1 = df.loc[df.batch == 1, "per_token_time_ms"].iloc[0]

    assert tpot8 == pytest.approx(8 * tpot1, rel=1e-6), (
        "total_tpot should scale ~linearly with batch in a compute-bound regime "
        "(undivided batch-step time), not stay flat"
    )
    wrong_formula_tpot8 = spec.n_layers * pt1  # what the bug used to compute (batch-invariant, wrong)
    assert tpot8 != pytest.approx(wrong_formula_tpot8, rel=1e-6), (
        "total_tpot_ms must not equal n_layers * per_token_time_ms -- that was the bug"
    )

def test_per_token_compute_component_is_batch_invariant_under_flop_model():
    """Under the pure FLOP estimate, t_compute scales linearly with batch, so
    per-TOKEN compute time should be constant across batch -- this is the
    'more batch = more compute per byte, for free' half of E1's reframing."""
    spec = MODEL_SPECS["olmoe"]
    e_union_by_batch = {1: 20.0, 2: 20.0, 4: 20.0, 8: 20.0}
    tc_tokens = []
    for batch in [1, 2, 4, 8]:
        _, tc_ms, _, prov = bs.per_token_time_ms(spec, batch, 0.0, "fp16", 64.0,
                                                  e_union_by_batch[batch], calib=None)
        assert prov == "flop_estimate"
        tc_tokens.append(tc_ms)
    assert tc_tokens == pytest.approx([tc_tokens[0]] * len(tc_tokens), rel=1e-9)


def test_find_b_star_picks_true_argmin_of_per_token_time():
    spec = MODEL_SPECS["olmoe"]
    batch_grid = [1, 2, 4, 8]
    e_union_by_batch = {b: 64.0 for b in batch_grid}  # saturated union -> bytes/token ~ 1/B
    b_star, df, note = bs.find_b_star(spec, batch_grid, residency_pct=0.0, precision="fp16",
                                       bw_gbps=64.0, e_union_by_batch=e_union_by_batch, calib=None)
    manual_best = df.loc[df["per_token_time_ms"].idxmin(), "batch"]
    assert b_star == int(manual_best)
    assert "argmin" in note or "unconstrained" in note


def test_find_b_star_respects_slo_and_excludes_infeasible_batches():
    spec = MODEL_SPECS["mixtral"]
    batch_grid = [1, 8, 32]
    e_union_by_batch = {b: 8.0 for b in batch_grid}
    # total_tpot = n_layers * per_token_time; find a SLO that only batch=1's
    # (tiny) compute-bound per-token time can satisfy for this synthetic case.
    _, df_full, _ = bs.find_b_star(spec, batch_grid, 0.0, "fp16", 128.0, e_union_by_batch, calib=None,
                                    slo_ms=None)
    tight_slo = float(df_full["total_tpot_ms"].min()) * 1.01  # just above the best row's own TPOT
    b_star, df, note = bs.find_b_star(spec, batch_grid, 0.0, "fp16", 128.0, e_union_by_batch, calib=None,
                                       slo_ms=tight_slo)
    assert df.loc[df.batch == b_star, "feasible_under_slo"].iloc[0]
    assert "SLO" in note


def test_find_b_star_falls_back_when_no_batch_satisfies_slo():
    spec = MODEL_SPECS["olmoe"]
    batch_grid = [1, 8, 32]
    e_union_by_batch = {b: 64.0 for b in batch_grid}
    b_star, df, note = bs.find_b_star(spec, batch_grid, 0.0, "fp16", 16.0, e_union_by_batch, calib=None,
                                       slo_ms=1e-9)
    assert "NO batch satisfies" in note
    assert b_star in batch_grid


# ---------- synthetic trace fixture for batched_recall ----------

def _make_synthetic_traces_df(n_reqs=40, n_layers=2, n_experts=4, k=1, deterministic=True, seed=0):
    """Schema matches collect.py's writer: model/req/domain/batch_id/phase/
    layer/tok/topk/gate_w/router_entropy. Two layers, k=1 for a simple,
    hand-verifiable deterministic transition e -> (e+1) % n_experts."""
    rng = np.random.default_rng(seed)
    rows = []
    for req in range(n_reqs):
        for tok in range(3):  # a few decode steps per request
            e0 = int(rng.integers(0, n_experts))
            e1 = (e0 + 1) % n_experts if deterministic else int(rng.integers(0, n_experts))
            for layer, e in [(0, e0), (1, e1)]:
                rows.append({
                    "model": "synthetic", "req": f"r{req}", "domain": "test", "batch_id": 0,
                    "phase": "decode", "layer": layer, "tok": tok,
                    "topk": (e,), "gate_w": [1.0], "router_entropy": 0.0,
                })
    df = pd.DataFrame(rows)
    df["topk"] = df["topk"].apply(tuple)
    return df


def test_batched_recall_is_perfect_for_a_deterministic_bijective_transition():
    """layer0 -> layer1 is the deterministic bijection e -> (e+1)%n_experts.
    With k=1, m=1, the fitted table should learn this exactly, so each row's
    own top-1 prediction always equals its own actual target -- meaning
    pred_union == actual_union row-for-row, for ANY batch size. recall must
    be exactly 1.0 regardless of B (a property that holds deterministically,
    not just in expectation, so this assertion is not flaky)."""
    df = _make_synthetic_traces_df(n_reqs=60, n_layers=2, n_experts=4, k=1, deterministic=True)
    train_reqs = {f"r{i}" for i in range(0, 40)}
    test_reqs = {f"r{i}" for i in range(40, 60)}
    res = bs.batched_recall(df, train_reqs, test_reqs, n_experts=4, k=1, n_layers=2,
                             batch_grid=[1, 2, 4, 8], m_mults=[(1, "k")], d=1,
                             c_for_nonresident=25.0, n_trials=30, seed=0)
    for key, v in res.items():
        assert v["recall_overall"] == pytest.approx(1.0), f"{key}: expected perfect recall, got {v}"


def test_batched_recall_has_near_zero_lift_for_a_noisy_transition_at_any_batch():
    """Random (unlearnable) layer0->layer1 mapping, m=1 << n_experts=20.

    Naively, raw recall_overall actually RISES with batch here (verified: it
    is not a bug) because the candidate pool (union of each of the B rows'
    own top-m) mechanically grows with B, independent of prediction quality
    -- discovered while writing this test, and now documented in
    batched_recall()'s docstring and surfaced in RESULTS.md via
    `lift_over_random_batched`. The correct invariant for an unlearnable
    transition is that recall tracks its batch-scaled RANDOM baseline (near
    -zero lift) at every batch size, not that raw recall stays flat."""
    df = _make_synthetic_traces_df(n_reqs=300, n_layers=2, n_experts=20, k=1, deterministic=False, seed=1)
    train_reqs = {f"r{i}" for i in range(0, 200)}
    test_reqs = {f"r{i}" for i in range(200, 300)}
    res = bs.batched_recall(df, train_reqs, test_reqs, n_experts=20, k=1, n_layers=2,
                             batch_grid=[1, 8], m_mults=[(1, "k")], d=1,
                             c_for_nonresident=25.0, n_trials=300, seed=1)
    for key in ["mk_B1", "mk_B8"]:
        lift = res[key]["lift_over_random_batched"]
        assert abs(lift) < 0.15, f"{key}: expected near-zero lift over random baseline for an " \
                                  f"unlearnable transition, got lift={lift}"


def test_batched_recall_saturates_when_batch_approaches_n_experts():
    """The flip side of the test above, documented as an expected property
    rather than a surprise: when n_experts is small relative to batch, both
    pred_union and actual_union approach the full expert set, so recall ->
    1.0 regardless of prediction quality. This mirrors PILOT_FINDINGS.md §5's
    'm=4k is degenerate for Mixtral (m==n_experts)' caveat, generalized to
    batch."""
    df = _make_synthetic_traces_df(n_reqs=150, n_layers=2, n_experts=4, k=1, deterministic=False, seed=1)
    train_reqs = {f"r{i}" for i in range(0, 100)}
    test_reqs = {f"r{i}" for i in range(100, 150)}
    res = bs.batched_recall(df, train_reqs, test_reqs, n_experts=4, k=1, n_layers=2,
                             batch_grid=[8], m_mults=[(1, "k")], d=1,
                             c_for_nonresident=25.0, n_trials=300, seed=1)
    assert res["mk_B8"]["recall_overall"] > 0.7, (
        "expected near-saturation recall when batch=8 >> n_experts=4 (union-saturation artifact)")


def test_batched_recall_handles_empty_actual_gracefully():
    """Degenerate n_trials=0 / tiny dataset shouldn't crash; nan is acceptable
    when there's no data to compute a ratio from."""
    df = _make_synthetic_traces_df(n_reqs=6, n_layers=2, n_experts=4, k=1, deterministic=True)
    train_reqs = {"r0", "r1", "r2"}
    test_reqs = {"r3", "r4", "r5"}
    res = bs.batched_recall(df, train_reqs, test_reqs, n_experts=4, k=1, n_layers=2,
                             batch_grid=[1], m_mults=[(1, "k")], d=1, n_trials=5, seed=0)
    assert "mk_B1" in res
    v = res["mk_B1"]
    assert np.isnan(v["recall_overall"]) or 0.0 <= v["recall_overall"] <= 1.0
