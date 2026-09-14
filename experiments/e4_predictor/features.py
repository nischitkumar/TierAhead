"""Feature construction for E4's predictor v2 (Experiments.md E4 step 1-2).

Builds (X, Y) pairs from already-collected decode traces: X encodes what's
known at layer l (which experts fired, how confidently, how uncertain the
router was, optionally the pre-router hidden state), Y is the multi-hot
target of which experts fire at layer l+d. Four ablation variants,
Experiments.md E4 step 2's list exactly:
  (a) "ids"                      -- selected expert IDs only [pilot baseline]
  (b) "ids_gate"                 -- + gate weights
  (c) "ids_gate_entropy"         -- + router entropy
  (d) "ids_gate_entropy_hidden"  -- + pre-router hidden state

No GPU required: (a)-(c) come straight from fields collect.py already logs
(topk, gate_w, router_entropy). (d) needs a hidden-state file produced by
collect_hidden_states.py (GPU-only, optional) -- callers should catch
MissingHiddenStateError and skip that ablation with a clear message rather
than crash the whole sweep, exactly as E1 falls back from measured_here to
flop_estimate rather than failing when calibration data is absent.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

VARIANTS = ["ids", "ids_gate", "ids_gate_entropy", "ids_gate_entropy_hidden"]


class MissingHiddenStateError(Exception):
    """Raised by build_layer_dataset when variant='ids_gate_entropy_hidden'
    is requested but no hidden-state lookup was supplied. Callers should
    catch this and skip the ablation, not treat it as a hard failure --
    hidden states require a separate GPU-only collection pass
    (collect_hidden_states.py) this analysis doesn't assume exists."""


@dataclass(frozen=True)
class LayerDataset:
    X: np.ndarray           # (N, feature_dim) float32
    Y: np.ndarray           # (N, n_experts) float32, multi-hot target at layer l+d
    reqs: np.ndarray        # (N,) the request id each row belongs to (for domain/group filtering)
    feature_dim: int


def _layer_frame(decode_df: pd.DataFrame, layer: int) -> pd.DataFrame:
    sub = decode_df[decode_df.layer == layer][["req", "tok", "topk", "gate_w", "router_entropy"]]
    return sub.set_index(["req", "tok"])


def _multi_hot(topk_col: pd.Series, n_experts: int, weights_col: pd.Series | None = None) -> np.ndarray:
    """topk_col: Series of tuples/lists of expert ids. weights_col (optional):
    parallel Series of per-slot weights (same order as topk_col's ids) -- if
    given, those weights are scattered into the one-hot slots instead of 1.0."""
    n = len(topk_col)
    out = np.zeros((n, n_experts), dtype=np.float32)
    if weights_col is None:
        for i, ids in enumerate(topk_col):
            out[i, list(ids)] = 1.0
    else:
        for i, (ids, ws) in enumerate(zip(topk_col, weights_col)):
            out[i, list(ids)] = np.asarray(ws, dtype=np.float32)
    return out


def build_layer_dataset(decode_df: pd.DataFrame, layer: int, d: int, variant: str,
                         n_experts: int, reqs_filter: set | None = None,
                         hidden_states: dict | None = None) -> LayerDataset | None:
    """Returns None if there aren't enough joined (layer, layer+d) rows to
    bother training on (e.g. layer+d >= n_layers, or reqs_filter excludes
    everything). hidden_states, if given for variant='...hidden', must map
    (req, tok) -> np.ndarray of that request/token's layer-l hidden state
    (see collect_hidden_states.py's output format)."""
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
    elif variant in ("ids_gate", "ids_gate_entropy", "ids_gate_entropy_hidden"):
        X = _multi_hot(joined["topk"], n_experts, weights_col=joined["gate_w"])
        if variant != "ids_gate":
            entropy = joined["router_entropy"].to_numpy(dtype=np.float32).reshape(-1, 1)
            X = np.concatenate([X, entropy], axis=1)
        if variant == "ids_gate_entropy_hidden":
            if hidden_states is None:
                raise MissingHiddenStateError(
                    "variant='ids_gate_entropy_hidden' requires a hidden-state lookup "
                    "(run collect_hidden_states.py on the GPU server first)")
            hs_rows = []
            missing = 0
            for req, tok in joined.index:
                v = hidden_states.get((req, tok, layer))
                if v is None:
                    missing += 1
                    v = np.zeros(next(iter(hidden_states.values())).shape, dtype=np.float32)
                hs_rows.append(v)
            if missing:
                raise MissingHiddenStateError(
                    f"{missing}/{len(joined)} rows had no hidden-state entry for layer {layer} -- "
                    "hidden-state file doesn't cover this trace/layer, refusing to zero-fill silently")
            X = np.concatenate([X, np.stack(hs_rows).astype(np.float32)], axis=1)
    else:  # pragma: no cover -- unreachable, VARIANTS check above already guards this
        raise AssertionError(variant)

    Y = _multi_hot(joined["topk_target"], n_experts)
    reqs = joined.index.get_level_values("req").to_numpy()
    return LayerDataset(X=X.astype(np.float32), Y=Y.astype(np.float32), reqs=reqs, feature_dim=X.shape[1])


def resident_mask(train_counts_layer: pd.Series, n_experts: int, c_pct: float = 25.0) -> np.ndarray:
    """Boolean mask, True = "resident" (in the top c_pct% by train-set popularity
    at this layer) -- same definition elp_probe's recall_at_m uses for its
    'nonresident' recall variant, reused here so E4's numbers are comparable
    to the pilot's own headline metric."""
    cnt = train_counts_layer.reindex(range(n_experts), fill_value=0)
    top_n = max(1, int(np.ceil(n_experts * c_pct / 100.0)))
    top_experts = set(cnt.sort_values(ascending=False).index[:top_n].tolist())
    mask = np.zeros(n_experts, dtype=bool)
    mask[list(top_experts)] = True
    return mask
