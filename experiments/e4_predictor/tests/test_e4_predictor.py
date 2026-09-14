"""Unit tests for E4's feature construction, predictor, and calibration
math. Pure Python/numpy/torch (CPU) -- no GPU, no network, no real trace
files required (synthetic DataFrames instead). Run with:
    pytest experiments/e4_predictor/tests/test_e4_predictor.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # e4_predictor/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/

import features as feat  # noqa: E402
import mlp as mlpmod  # noqa: E402
import calibration as calib  # noqa: E402


# ---------- features.py ----------

def _toy_decode_df(n_experts=8, k=2, n_layers=3, n_reqs=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for req in range(n_reqs):
        for tok in range(5):
            for layer in range(n_layers):
                topk = tuple(sorted(rng.choice(n_experts, size=k, replace=False).tolist()))
                gate_w = sorted(rng.dirichlet(np.ones(k)).tolist(), reverse=True)
                rows.append({"req": f"r{req}", "tok": tok, "layer": layer, "topk": topk,
                             "gate_w": gate_w, "router_entropy": float(rng.random()),
                             "domain": "chat" if req % 2 == 0 else "code", "phase": "decode"})
    return pd.DataFrame(rows)


def test_build_layer_dataset_ids_variant_shapes():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    ds = feat.build_layer_dataset(df, layer=0, d=1, variant="ids", n_experts=8)
    assert ds is not None
    assert ds.X.shape == (len(ds.reqs), 8)
    assert ds.Y.shape == (len(ds.reqs), 8)
    assert ds.feature_dim == 8
    # X rows are multi-hot with exactly k=2 ones (ids variant has no weighting)
    assert np.allclose(ds.X.sum(axis=1), 2.0)
    assert np.allclose(ds.Y.sum(axis=1), 2.0)


def test_build_layer_dataset_ids_gate_uses_real_weights_not_binary():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    ds = feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate", n_experts=8)
    nonzero = ds.X[ds.X > 0]
    assert not np.allclose(nonzero, 1.0), "ids_gate should carry real gate weights, not just 0/1"


def test_build_layer_dataset_entropy_variant_adds_one_column():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    ds_gate = feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate", n_experts=8)
    ds_ent = feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate_entropy", n_experts=8)
    assert ds_ent.feature_dim == ds_gate.feature_dim + 1


def test_build_layer_dataset_hidden_variant_without_lookup_raises():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    with pytest.raises(feat.MissingHiddenStateError):
        feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate_entropy_hidden", n_experts=8,
                                  hidden_states=None)


def test_build_layer_dataset_hidden_variant_with_partial_coverage_raises():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3, n_reqs=4)
    # only provide hidden states for req "r0" -- everything else should trigger the guard
    partial = {("r0", tok, 0): np.zeros(4, dtype=np.float32) for tok in range(5)}
    with pytest.raises(feat.MissingHiddenStateError):
        feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate_entropy_hidden", n_experts=8,
                                  hidden_states=partial)


def test_build_layer_dataset_hidden_variant_with_full_coverage_works():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3, n_reqs=4)
    full = {}
    for req in [f"r{i}" for i in range(4)]:
        for tok in range(5):
            full[(req, tok, 0)] = np.ones(4, dtype=np.float32) * 3.0
    ds = feat.build_layer_dataset(df, layer=0, d=1, variant="ids_gate_entropy_hidden", n_experts=8,
                                   hidden_states=full)
    assert ds.feature_dim == 8 + 1 + 4  # ids_gate(8) + entropy(1) + hidden(4)
    assert np.allclose(ds.X[:, -4:], 3.0)


def test_build_layer_dataset_reqs_filter():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3, n_reqs=10)
    subset = {"r0", "r1", "r2"}
    ds = feat.build_layer_dataset(df, layer=0, d=1, variant="ids", n_experts=8, reqs_filter=subset)
    assert set(ds.reqs.tolist()) <= subset


def test_build_layer_dataset_out_of_range_layer_returns_none():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    assert feat.build_layer_dataset(df, layer=5, d=1, variant="ids", n_experts=8) is None


def test_build_layer_dataset_unknown_variant_raises():
    df = _toy_decode_df(n_experts=8, k=2, n_layers=3)
    with pytest.raises(ValueError):
        feat.build_layer_dataset(df, layer=0, d=1, variant="bogus", n_experts=8)


def test_resident_mask_picks_top_c_pct():
    counts = pd.Series({0: 100, 1: 50, 2: 10, 3: 1}, name="count")
    mask = feat.resident_mask(counts, n_experts=4, c_pct=50.0)  # top 50% of 4 = top 2
    assert mask.sum() == 2
    assert mask[0] and mask[1]
    assert not mask[2] and not mask[3]


# ---------- mlp.py ----------

def test_tinymlp_forward_shape():
    model = mlpmod.TinyMLP(in_dim=10, n_experts=6, hidden=16)
    x = __import__("torch").randn(5, 10)
    out = model(x)
    assert out.shape == (5, 6)


def test_train_predictor_reduces_loss_and_recovers_a_learnable_pattern():
    """Deterministic synthetic task: target = same as input's multi-hot (an
    easy, learnable identity-ish mapping) -- a real predictor should reach
    near-perfect recall on this; a broken training loop (e.g. no gradient
    flow) would stay near chance."""
    rng = np.random.default_rng(0)
    n, n_experts, k = 400, 8, 2
    X = np.zeros((n, n_experts), dtype=np.float32)
    Y = np.zeros((n, n_experts), dtype=np.float32)
    for i in range(n):
        ids = rng.choice(n_experts, size=k, replace=False)
        X[i, ids] = 1.0
        Y[i, ids] = 1.0  # target == input
    model = mlpmod.train_predictor(X, Y, n_experts, hidden=32, epochs=200, seed=0)
    logits = mlpmod.predict_logits(model, X)
    recall = mlpmod.recall_at_m(logits, Y, m=k)
    assert recall > 0.95, f"should easily learn an identity mapping, got recall={recall}"


def test_topm_hits_and_total_perfect_prediction():
    Y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
    logits = np.array([[10, -10, 10, -10], [-10, 10, -10, 10]], dtype=np.float32)
    hits, total = mlpmod.topm_hits_and_total(logits, Y, m=2)
    assert hits == 4 and total == 4


def test_topm_hits_and_total_with_candidate_mask_restricts_target():
    Y = np.array([[1, 1, 0, 0]], dtype=np.float32)
    logits = np.array([[10, 10, -10, -10]], dtype=np.float32)  # predicts experts 0,1 correctly
    mask_only_expert_1 = np.array([False, True, False, False])
    hits, total = mlpmod.topm_hits_and_total(logits, Y, m=2, candidate_mask=mask_only_expert_1)
    assert total == 1  # only expert 1 counts as "actual" under the mask
    assert hits == 1   # and it was correctly predicted


def test_recall_at_m_nan_when_no_actual_targets():
    Y = np.zeros((3, 4), dtype=np.float32)
    logits = np.random.randn(3, 4).astype(np.float32)
    assert np.isnan(mlpmod.recall_at_m(logits, Y, m=2))


def test_param_count_matches_manual_calc():
    model = mlpmod.TinyMLP(in_dim=10, n_experts=5, hidden=4)
    # layer1: 10*4+4=44, layer2: 4*5+5=25 -> 69
    assert mlpmod.param_count(model) == 44 + 25


def test_standardize_train_test_zero_means_and_unit_variance_on_train():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(500, 4)).astype(np.float32)
    X_test = rng.normal(loc=5.0, scale=2.0, size=(100, 4)).astype(np.float32)
    Xtr_s, Xte_s = mlpmod.standardize_train_test(X_train, X_test)
    assert np.allclose(Xtr_s.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(Xtr_s.std(axis=0), 1.0, atol=1e-4)
    # test set uses TRAIN mean/std, not its own -- shouldn't be exactly zero-mean
    assert Xte_s.shape == X_test.shape


def test_standardize_handles_constant_column_without_nan_or_inf():
    X_train = np.stack([np.zeros(50), np.ones(50)], axis=1).astype(np.float32)  # column 0 constant
    X_test = X_train.copy()
    Xtr_s, Xte_s = mlpmod.standardize_train_test(X_train, X_test)
    assert np.isfinite(Xtr_s).all() and np.isfinite(Xte_s).all()
    assert np.allclose(Xtr_s[:, 0], 0.0)  # constant column maps to 0, not NaN from a 0/0 divide


def test_small_scale_features_do_not_collapse_recall_vs_large_scale_features_when_standardized():
    """Regression test for a real bug found while building E4: two datasets
    that encode the IDENTICAL underlying pattern, one at O(1) scale (like the
    'ids' variant's 0/1 one-hot) and one at O(0.01) scale (like 'ids_gate's
    softmax-weighted encoding), trained with the SAME fixed epoch budget and
    LR, silently collapsed the small-scale variant's recall to near-zero
    before standardization was added -- not because the information was any
    worse, purely because the optimizer hadn't converged on the tiny-magnitude
    input in time. After standardize_train_test, both scales must reach
    comparable recall on this synthetic identity-mapping task."""
    rng = np.random.default_rng(0)
    n, n_experts, k = 800, 16, 3
    Y = np.zeros((n, n_experts), dtype=np.float32)
    X_large = np.zeros((n, n_experts), dtype=np.float32)
    X_small = np.zeros((n, n_experts), dtype=np.float32)
    for i in range(n):
        ids = rng.choice(n_experts, size=k, replace=False)
        Y[i, ids] = 1.0
        X_large[i, ids] = 1.0            # O(1) scale, like 'ids'
        X_small[i, ids] = 0.02            # O(0.01) scale, like 'ids_gate'

    split = n * 3 // 4
    recalls = {}
    for name, X in [("large", X_large), ("small", X_small)]:
        Xtr, Xte = mlpmod.standardize_train_test(X[:split], X[split:])
        model = mlpmod.train_predictor(Xtr, Y[:split], n_experts, hidden=32, epochs=60, seed=0)
        logits = mlpmod.predict_logits(model, Xte)
        recalls[name] = mlpmod.recall_at_m(logits, Y[split:], m=k)

    assert recalls["small"] > 0.5, f"small-scale features collapsed after standardization: {recalls}"
    assert abs(recalls["large"] - recalls["small"]) < 0.15, (
        f"standardized large- and small-scale encodings of the same pattern should reach "
        f"comparable recall, got {recalls}")


# ---------- calibration.py ----------

def test_ece_zero_for_perfectly_calibrated_probs():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 5000)
    labels = (rng.uniform(0, 1, 5000) < probs).astype(float)  # by construction, calibrated
    ece, bins = calib.expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.03, f"a perfectly-calibrated-by-construction sample should have low ECE, got {ece}"
    assert sum(b["count"] for b in bins) == 5000


def test_ece_high_for_systematically_overconfident_probs():
    n = 2000
    probs = np.full(n, 0.95)
    labels = np.zeros(n)  # claims 95% confident, right 0% of the time
    ece, _ = calib.expected_calibration_error(probs, labels, n_bins=10)
    assert ece > 0.9


def test_fit_temperature_recovers_known_temperature():
    """Generate labels from sigmoid(logits / T_true), then check fit_temperature
    recovers something close to T_true from the same (logits, labels)."""
    rng = np.random.default_rng(0)
    n = 5000
    raw_logits = rng.normal(0, 3, n).astype(np.float32)
    T_true = 2.5
    probs = 1 / (1 + np.exp(-raw_logits / T_true))
    labels = (rng.uniform(0, 1, n) < probs).astype(np.float32)
    T_fit = calib.fit_temperature(raw_logits, labels)
    assert T_fit == pytest.approx(T_true, rel=0.25)


def test_fit_temperature_scaling_improves_or_maintains_ece_on_overconfident_data():
    rng = np.random.default_rng(1)
    n = 4000
    raw_logits = rng.normal(0, 8, n).astype(np.float32)  # very sharp/overconfident logits
    T_true = 4.0
    probs = 1 / (1 + np.exp(-raw_logits / T_true))
    labels = (rng.uniform(0, 1, n) < probs).astype(np.float32)

    probs_before = 1 / (1 + np.exp(-raw_logits))
    ece_before, _ = calib.expected_calibration_error(probs_before, labels)
    T_fit = calib.fit_temperature(raw_logits, labels)
    probs_after = 1 / (1 + np.exp(-raw_logits / T_fit))
    ece_after, _ = calib.expected_calibration_error(probs_after, labels)
    assert ece_after < ece_before
