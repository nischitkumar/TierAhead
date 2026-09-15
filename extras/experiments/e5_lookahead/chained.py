"""Direct vs chained lookahead (Experiments.md E5 step 1): "fit separate
conditional tables/MLPs per d -- do NOT chain, chaining compounds error;
measure chained as a comparison."

Direct: one MLP per (layer, d) trained straight from layer l's REAL observed
features (ids+gate+entropy, same as E4's winning-in-practice variant) to
layer l+d's target -- one hop, real information both ends.

Chained: exactly one d=1 MLP per layer, trained on the "ids"-only variant
(deliberately -- see below), composed d times: layer l's real observed ids
feed the d=1 model, its own top-k OUTPUT (not ground truth) becomes the
"observed" input for the next hop, d times over, until reaching l+d. Only
the "ids" variant is usable for chaining because gate weights and router
entropy for a *predicted* (not real) intermediate layer state don't exist --
there's no real router logit to compute them from at a hop that never
actually ran. This is the actual mechanism by which chaining "compounds
error": each hop's uncertainty becomes the next hop's (imperfect) ground
truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e4_predictor"))  # e4_predictor/

import features as feat  # noqa: E402
import mlp as mlpmod  # noqa: E402


def train_ids_d1_models(decode_df, train_reqs, n_experts, n_layers, hidden_dim, epochs, seed):
    """One d=1 'ids'-variant model per layer, plus its standardization stats
    (needed at inference time for both the direct-d=1 case and every hop of
    every chained composition)."""
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
                    resident_mask_by_layer: dict, m: int):
    """Composes ids_models[l], ids_models[l+1], ..., ids_models[l+d-1] to walk
    from layer l's real observed top-k to a predicted state at l+d, d hops.
    Intermediate hops collapse to their own top-k (that hop's "belief"); the
    FINAL hop's full logit ranking (top-m, not just top-k) is what's scored
    against the real target -- exactly mirroring what a direct predictor
    would be asked to produce (a top-m candidate set), so the two are
    comparable on identical terms."""
    pooled_hit, pooled_tot = 0, 0
    for layer in range(n_layers - d):
        if any((layer + step) not in ids_models for step in range(d)):
            continue
        obs_ds = feat.build_layer_dataset(decode_df, layer, d, "ids", n_experts, reqs_filter=test_reqs)
        if obs_ds is None or len(obs_ds.X) == 0:
            continue
        cur = obs_ds.X  # real observed multi-hot ids at layer l
        logits = None
        for step in range(d):
            entry = ids_models[layer + step]
            cur_std = ((cur - entry["mean"]) / entry["std"]).astype(np.float32)
            logits = mlpmod.predict_logits(entry["model"], cur_std)
            if step < d - 1:
                # collapse to this hop's belief (top-k) to bootstrap the next hop --
                # the next model was trained on REAL top-k-shaped inputs, so feeding
                # it anything other than a k-hot vector would be off-distribution.
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
