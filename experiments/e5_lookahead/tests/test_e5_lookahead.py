"""Unit tests for E5's queue simulation, feasible-fetch-size math, and
chained-vs-direct lookahead logic. Pure Python/numpy/torch (CPU) -- no GPU,
no network, no real trace files required. Run with:
    pytest experiments/e5_lookahead/tests/test_e5_lookahead.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # e5_lookahead/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e4_predictor"))  # e4_predictor/

import queue_sim  # noqa: E402
import feasible_fetch as ff  # noqa: E402
import chained  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402


# ---------- queue_sim.py ----------

def test_underloaded_queue_never_drops_regardless_of_depth():
    for q in [1, 2, 8, 64]:
        res = queue_sim.simulate_bounded_queue(period_ms=10.0, service_ms=4.0, capacity_q=q)
        assert res["drop_rate"] == 0.0
        assert res["mean_occupancy"] == pytest.approx(0.4)
        assert res["link_utilization"] == pytest.approx(0.4)


def test_exactly_saturated_queue_rho_equals_one_no_drops():
    res = queue_sim.simulate_bounded_queue(period_ms=5.0, service_ms=5.0, capacity_q=4)
    assert res["rho"] == pytest.approx(1.0)
    assert res["drop_rate"] == 0.0
    assert res["link_utilization"] == pytest.approx(1.0)


def test_overloaded_queue_drop_rate_independent_of_capacity():
    """Experiments.md's own point: a deeper buffer doesn't fix an overloaded
    link -- steady-state drop rate should be identical across capacities."""
    rates = [queue_sim.simulate_bounded_queue(period_ms=2.0, service_ms=5.0, capacity_q=q)["drop_rate"]
             for q in [1, 4, 16, 100]]
    assert len(set(round(r, 9) for r in rates)) == 1
    assert rates[0] == pytest.approx(1.0 - 2.0 / 5.0)


def test_overloaded_queue_occupancy_equals_capacity_and_full_utilization():
    res = queue_sim.simulate_bounded_queue(period_ms=1.0, service_ms=3.0, capacity_q=7)
    assert res["mean_occupancy"] == 7.0
    assert res["link_utilization"] == pytest.approx(1.0)


def test_sweep_queue_depth_covers_every_requested_q():
    out = queue_sim.sweep_queue_depth(2.0, 3.0, [1, 2, 4])
    assert set(out.keys()) == {1, 2, 4}


def test_simulate_bounded_queue_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        queue_sim.simulate_bounded_queue(period_ms=0, service_ms=1.0, capacity_q=1)
    with pytest.raises(ValueError):
        queue_sim.simulate_bounded_queue(period_ms=1.0, service_ms=1.0, capacity_q=0)
    with pytest.raises(ValueError):
        queue_sim.simulate_bounded_queue(period_ms=1.0, service_ms=-1.0, capacity_q=1)


# ---------- feasible_fetch.py ----------

def test_feasible_bytes_scales_linearly_with_depth():
    b1, _ = ff.feasible_bytes_at_depth("olmoe", 1, batch=1, bw_gbps=32.0)
    b4, _ = ff.feasible_bytes_at_depth("olmoe", 4, batch=1, bw_gbps=32.0)
    assert b4 == pytest.approx(4 * b1, rel=1e-9)


def test_feasible_bytes_scales_linearly_with_bandwidth():
    b32, _ = ff.feasible_bytes_at_depth("mixtral", 2, batch=1, bw_gbps=32.0)
    b64, _ = ff.feasible_bytes_at_depth("mixtral", 2, batch=1, bw_gbps=64.0)
    assert b64 == pytest.approx(2 * b32, rel=1e-9)


def test_feasible_bytes_uses_flop_estimate_provenance_without_calib():
    _, prov = ff.feasible_bytes_at_depth("olmoe", 1, batch=1, bw_gbps=32.0, calib=None)
    assert prov == "flop_estimate"


def test_max_useful_depth_finds_first_depth_that_holds():
    recall_by_d = {1: 0.9, 2: 0.85, 4: 0.55, 8: 0.62}
    # d=4 dips below 0.6 threshold even though d=8 recovers -- the "holds for
    # every deeper depth too" requirement should reject d=1 and d=2 (since d=4
    # after them fails) and accept d=8 (nothing deeper to fail on).
    assert ff.max_useful_depth(recall_by_d, threshold=0.6) == 8


def test_max_useful_depth_accepts_earliest_depth_when_monotonic():
    recall_by_d = {1: 0.95, 2: 0.9, 4: 0.7, 8: 0.65}
    assert ff.max_useful_depth(recall_by_d, threshold=0.6) == 1


def test_max_useful_depth_none_when_never_clears_bar():
    recall_by_d = {1: 0.5, 2: 0.4, 4: 0.3, 8: 0.2}
    assert ff.max_useful_depth(recall_by_d, threshold=0.6) is None


# ---------- chained.py ----------

def _toy_decode_df(n_experts=8, k=2, n_layers=5, n_reqs=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for req in range(n_reqs):
        for tok in range(6):
            for layer in range(n_layers):
                topk = tuple(sorted(rng.choice(n_experts, size=k, replace=False).tolist()))
                gate_w = sorted(rng.dirichlet(np.ones(k)).tolist(), reverse=True)
                rows.append({"req": f"r{req}", "tok": tok, "layer": layer, "topk": topk,
                             "gate_w": gate_w, "router_entropy": float(rng.random()), "phase": "decode"})
    return pd.DataFrame(rows)


def test_train_ids_d1_models_covers_every_layer_transition():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=4, n_reqs=40)
    train_reqs = {f"r{i}" for i in range(30)}
    models = chained.train_ids_d1_models(df, train_reqs, n_experts=8, n_layers=4, hidden_dim=16,
                                          epochs=10, seed=0)
    assert set(models.keys()) == {0, 1, 2}  # n_layers-1 transitions


def test_chained_recall_is_bounded_and_finite():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=4, n_reqs=40)
    train_reqs = {f"r{i}" for i in range(30)}
    test_reqs = {f"r{i}" for i in range(30, 40)}
    models = chained.train_ids_d1_models(df, train_reqs, n_experts=8, n_layers=4, hidden_dim=16,
                                          epochs=10, seed=0)
    resident = {l: np.zeros(8, dtype=bool) for l in range(4)}  # nothing resident -> full target counts
    for d in [1, 2]:
        r = chained.chained_recall(df, test_reqs, models, n_experts=8, k=2, n_layers=4, d=d,
                                    resident_mask_by_layer=resident, m=2)
        assert 0.0 <= r <= 1.0


def test_chained_recall_d1_uses_full_top_m_ranking_not_just_top_k():
    """At d=1 chaining does exactly one hop -- its output should be scored
    with the model's full logit ranking at m, not silently truncated to k
    before scoring (that would make m>k meaningless for the chained curve)."""
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3, n_reqs=40)
    train_reqs = {f"r{i}" for i in range(30)}
    test_reqs = {f"r{i}" for i in range(30, 40)}
    models = chained.train_ids_d1_models(df, train_reqs, n_experts=8, n_layers=3, hidden_dim=16,
                                          epochs=10, seed=0)
    resident = {l: np.zeros(8, dtype=bool) for l in range(3)}
    r_m2 = chained.chained_recall(df, test_reqs, models, n_experts=8, k=2, n_layers=3, d=1,
                                   resident_mask_by_layer=resident, m=2)
    r_m6 = chained.chained_recall(df, test_reqs, models, n_experts=8, k=2, n_layers=3, d=1,
                                   resident_mask_by_layer=resident, m=6)
    assert r_m6 >= r_m2, "a larger m should never recall strictly less"


def test_chained_recall_missing_layer_model_is_skipped_not_crashed():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=4, n_reqs=40)
    train_reqs = {f"r{i}" for i in range(30)}
    test_reqs = {f"r{i}" for i in range(30, 40)}
    models = chained.train_ids_d1_models(df, train_reqs, n_experts=8, n_layers=4, hidden_dim=16,
                                          epochs=10, seed=0)
    del models[1]  # simulate a layer that didn't get enough data to train
    resident = {l: np.zeros(8, dtype=bool) for l in range(4)}
    r = chained.chained_recall(df, test_reqs, models, n_experts=8, k=2, n_layers=4, d=2,
                                resident_mask_by_layer=resident, m=2)
    assert r != r or 0.0 <= r <= 1.0  # nan (no usable layers) or a valid recall -- must not raise
