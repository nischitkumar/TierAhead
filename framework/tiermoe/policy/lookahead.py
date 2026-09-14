"""Lookahead depth (Main.md's E5 scope): direct (one predictor per depth d)
vs naively-chained recall, and the feasible-fetch-size curve
(bytes fetchable within a d-layer compute window).

``feasible_bytes_at_depth`` / ``max_useful_depth`` are pure (only need
tiermoe.roofline's compute-time model) and always importable. The
direct/chained MLP training functions need torch and import it lazily,
matching predictor_mlp.py's own scoping so this module stays importable
without the `predictor` extra for callers that only want the feasible-fetch
curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tiermoe.roofline.model import get_t_compute_ms
from tiermoe.specs import MODEL_SPECS


def feasible_bytes_at_depth(model_tag: str, d: int, batch: int, bw_gbps: float, calib=None) -> tuple[float, str]:
    """Max bytes fetchable within a d-layer lookahead window: d layers'
    worth of compute time times the link's bandwidth."""
    spec = MODEL_SPECS[model_tag]
    t_layer_ms, provenance = get_t_compute_ms(spec, batch, calib)
    window_ms = d * t_layer_ms
    return bw_gbps * 1e6 * window_ms, provenance  # GB/s -> bytes/ms


def max_useful_depth(recall_by_d: dict[int, float], threshold: float = 0.60) -> int | None:
    """Smallest d at or beyond which recall stays >= threshold for every
    subsequent swept depth. None if no depth clears it."""
    ds = sorted(recall_by_d)
    for i, d in enumerate(ds):
        if all(recall_by_d[dd] >= threshold for dd in ds[i:]):
            return d
    return None


def train_ids_d1_models(decode_df, train_reqs, n_experts, n_layers, hidden_dim, epochs, seed):
    """One d=1 'ids'-variant model per layer, for chaining. Lazy torch import."""
    from tiermoe.policy import features as feat
    from tiermoe.policy import predictor_mlp as mlpmod

    models = {}
    for layer in range(n_layers - 1):
        train_ds = feat.build_layer_dataset(decode_df, layer, 1, "ids", n_experts, reqs_filter=train_reqs)
        if train_ds is None or len(train_ds.X) < 50:
            continue
        mean = train_ds.X.mean(axis=0, keepdims=True)
        std = train_ds.X.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X_std = ((train_ds.X - mean) / std).astype(np.float32)
        model = mlpmod.train_predictor(X_std, train_ds.Y, n_experts, hidden=hidden_dim, epochs=epochs, seed=seed)
        models[layer] = {"model": model, "mean": mean, "std": std}
    return models


def chained_recall(decode_df, test_reqs, ids_models: dict, n_experts, k, n_layers, d,
                    resident_mask_by_layer: dict, m: int) -> float:
    """One d=1 'ids' model composed d times -- chaining compounds error
    (Main.md's own prediction, confirmed on real traces, see
    experiments/e5_lookahead/E5_LOOKAHEAD.md: the direct-vs-chained gap
    widens monotonically with d, +0.019 at d=1 to +0.170 at d=8 for OLMoE)."""
    from tiermoe.policy import predictor_mlp as mlpmod
    from tiermoe.policy import features as feat

    pooled_hit, pooled_tot = 0, 0
    for layer in range(n_layers - d):
        if any((layer + step) not in ids_models for step in range(d)):
            continue
        obs_ds = feat.build_layer_dataset(decode_df, layer, d, "ids", n_experts, reqs_filter=test_reqs)
        if obs_ds is None or len(obs_ds.X) == 0:
            continue
        cur = obs_ds.X
        logits = None
        for step in range(d):
            entry = ids_models[layer + step]
            cur_std = ((cur - entry["mean"]) / entry["std"]).astype(np.float32)
            logits = mlpmod.predict_logits(entry["model"], cur_std)
            if step < d - 1:
                idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
                cur = np.zeros_like(cur)
                np.put_along_axis(cur, idx, 1.0, axis=1)
        res_mask = resident_mask_by_layer.get(layer + d)
        if res_mask is None:
            continue
        hit, tot = mlpmod.topm_hits_and_total(logits, obs_ds.Y, m, candidate_mask=~res_mask)
        pooled_hit += hit
        pooled_tot += tot
    return pooled_hit / pooled_tot if pooled_tot else float("nan")


def direct_recall_for_depth(decode_df, train_reqs, test_reqs, n_experts, n_layers, d, tr_counts,
                             hidden_dim, epochs, seed, m, variant: str = "ids_gate_entropy") -> tuple[float, int]:
    from tiermoe.policy import features as feat
    from tiermoe.policy import predictor_mlp as mlpmod

    pooled_hit, pooled_tot, n_layers_used = 0, 0, 0
    for layer in range(n_layers - d):
        train_ds = feat.build_layer_dataset(decode_df, layer, d, variant, n_experts, reqs_filter=train_reqs)
        test_ds = feat.build_layer_dataset(decode_df, layer, d, variant, n_experts, reqs_filter=test_reqs)
        if train_ds is None or test_ds is None or len(train_ds.X) < 50 or len(test_ds.X) < 20:
            continue
        X_train, X_test = mlpmod.standardize_train_test(train_ds.X, test_ds.X)
        model = mlpmod.train_predictor(X_train, train_ds.Y, n_experts, hidden=hidden_dim, epochs=epochs, seed=seed)
        logits = mlpmod.predict_logits(model, X_test)
        res_mask = feat.resident_mask(tr_counts.get(layer + d, pd.Series(dtype=float)), n_experts, 25.0)
        hit, tot = mlpmod.topm_hits_and_total(logits, test_ds.Y, m, candidate_mask=~res_mask)
        pooled_hit += hit
        pooled_tot += tot
        n_layers_used += 1
    return (pooled_hit / pooled_tot if pooled_tot else float("nan")), n_layers_used
