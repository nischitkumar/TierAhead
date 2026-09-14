"""Feature construction for the MLP predictor (predictor_mlp.TinyMLP).

Ported from experiments/e4_predictor/features.py. Pure numpy/pandas -- no
torch import here, so importing this module never requires the `predictor`
extra; only predictor_mlp.py (which trains/runs the network itself) does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VARIANTS = ["ids", "ids_gate", "ids_gate_entropy", "ids_gate_entropy_hidden"]


class MissingHiddenStateError(Exception):
    """variant='ids_gate_entropy_hidden' needs a hidden-state file from a
    GPU-only collection pass (tierahead.collect.hidden_states, not yet built
    against real weights on this machine). Callers should catch this and
    skip the ablation, not treat it as fatal."""


@dataclass(frozen=True)
class LayerDataset:
    X: np.ndarray
    Y: np.ndarray
    reqs: np.ndarray
    feature_dim: int


def _layer_frame(decode_df: pd.DataFrame, layer: int) -> pd.DataFrame:
    sub = decode_df[decode_df.layer == layer][["req", "tok", "topk", "gate_w", "router_entropy"]]
    return sub.set_index(["req", "tok"])


def _multi_hot(topk_col: pd.Series, n_experts: int, weights_col: pd.Series | None = None) -> np.ndarray:
    n = len(topk_col)
    out = np.zeros((n, n_experts), dtype=np.float32)
    if weights_col is None:
        for i, ids in enumerate(topk_col):
            out[i, list(ids)] = 1.0
    else:
        for i, (ids, ws) in enumerate(zip(topk_col, weights_col)):
            out[i, list(ids)] = np.asarray(ws, dtype=np.float32)
    return out


def build_layer_dataset(decode_df: pd.DataFrame, layer: int, d: int, variant: str, n_experts: int,
                         reqs_filter: set | None = None, hidden_states: dict | None = None) -> LayerDataset | None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    l_frame = _layer_frame(decode_df, layer)
    ld_frame = _layer_frame(decode_df, layer + d)[["topk"]].rename(columns={"topk": "topk_target"})
    joined = l_frame.join(ld_frame, how="inner")
    if reqs_filter is not None:
        joined = joined[joined.index.get_level_values("req").isin(reqs_filter)]
    if len(joined) == 0:
        return None

    if variant == "ids":
        X = _multi_hot(joined["topk"], n_experts)
    else:
        X = _multi_hot(joined["topk"], n_experts, weights_col=joined["gate_w"])
        if variant != "ids_gate":
            entropy = joined["router_entropy"].to_numpy(dtype=np.float32).reshape(-1, 1)
            X = np.concatenate([X, entropy], axis=1)
        if variant == "ids_gate_entropy_hidden":
            if hidden_states is None:
                raise MissingHiddenStateError(
                    "variant='ids_gate_entropy_hidden' requires a hidden-state lookup "
                    "(run tierahead.collect.hidden_states on a CUDA box first)")
            hs_rows, missing = [], 0
            for req, tok in joined.index:
                v = hidden_states.get((req, tok, layer))
                if v is None:
                    missing += 1
                    v = np.zeros(next(iter(hidden_states.values())).shape, dtype=np.float32)
                hs_rows.append(v)
            if missing:
                raise MissingHiddenStateError(
                    f"{missing}/{len(joined)} rows had no hidden-state entry for layer {layer}")
            X = np.concatenate([X, np.stack(hs_rows).astype(np.float32)], axis=1)

    Y = _multi_hot(joined["topk_target"], n_experts)
    reqs = joined.index.get_level_values("req").to_numpy()
    return LayerDataset(X=X.astype(np.float32), Y=Y.astype(np.float32), reqs=reqs, feature_dim=X.shape[1])


def resident_mask(train_counts_layer: pd.Series, n_experts: int, c_pct: float = 25.0) -> np.ndarray:
    cnt = train_counts_layer.reindex(range(n_experts), fill_value=0)
    top_n = max(1, int(np.ceil(n_experts * c_pct / 100.0)))
    top_experts = set(cnt.sort_values(ascending=False).index[:top_n].tolist())
    mask = np.zeros(n_experts, dtype=bool)
    mask[list(top_experts)] = True
    return mask
