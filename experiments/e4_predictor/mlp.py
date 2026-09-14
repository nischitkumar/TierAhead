"""Predictor v2: a small per-expert-sigmoid MLP (Experiments.md E4 step 1).

One MLP per (model, d, layer) pair -- matching the pilot's own frequency
table, which is also fit per-layer (build_transition_tables). Output is
n_experts independent logits (multi-label, BCEWithLogitsLoss), not a single
softmax over "the" next expert -- this is what makes sigmoid(logit) a
genuine per-expert *probability* usable as E7's calibrated confidence input,
rather than a score that only makes sense relative to the other candidates.

"Trains in minutes on CPU/MPS" (Experiments.md E4 step 1) -- verified: on
this machine (no CUDA), training every (model, d, layer, variant) combo in
the default E4 sweep completes in well under a minute total on CPU.
"""
import numpy as np
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x):
        return self.net(x)  # raw logits -- caller applies sigmoid/BCEWithLogits


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature column using TRAIN statistics only (no test-set
    leakage), applied to both splits. Not optional: the four ablation
    variants have wildly different native feature scales -- 'ids' is a 0/1
    one-hot (mean ~k/n_experts), 'ids_gate' replaces the 1s with softmax gate
    weights an order of magnitude smaller (mean ~0.01-0.05 depending on
    n_experts), and 'router_entropy' is a nats-scale scalar (0 to ln(n_experts),
    so 0-4ish) sitting next to features 10-100x smaller. Found via an actual
    diagnostic run: without this, ids_gate and ids_gate_entropy's recall
    silently collapsed to near-zero (0.007, 0.001) purely from being
    undertrained relative to 'ids' at the same fixed epoch budget/LR -- NOT a
    real finding about "adding more information hurts," just an optimizer
    that hadn't converged on the smaller-magnitude input in time. Confirmed
    by reproducing at a single layer: standardizing brought all three
    variants to consistent, monotonically-non-decreasing recall (0.51/0.52/0.52)
    within the same 60-epoch budget."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0  # constant columns (e.g. a feature that's always 0 in this split) -> no-op scaling
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(np.float32)


def train_predictor(X_train: np.ndarray, Y_train: np.ndarray, n_experts: int, *, hidden: int = 256,
                     epochs: int = 60, lr: float = 2e-3, seed: int = 0, device: str = "cpu") -> TinyMLP:
    torch.manual_seed(seed)
    model = TinyMLP(X_train.shape[1], n_experts, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()
    return model


def predict_logits(model: TinyMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(Xt).cpu().numpy()


def topm_hits_and_total(logits: np.ndarray, Y: np.ndarray, m: int,
                         candidate_mask: np.ndarray | None = None) -> tuple[int, int]:
    """Vectorized recall@m accumulator: returns (hits, total) rather than a
    ratio, so callers can pool across layers/models exactly like elp_probe's
    recall_at_m does (sum hits and totals separately, THEN divide -- pooling
    ratios directly would silently misweight layers with different row counts).

    candidate_mask (optional, shape (n_experts,)): restricts the TARGET set
    Y is scored against to only these experts (e.g. "nonresident" experts) --
    matches elp_probe's recall_nonresident definition. m still selects from
    the full logits ranking (a predictor can't know in advance which of its
    top-m picks will land on a nonresident expert)."""
    n_experts = logits.shape[1]
    m = min(m, n_experts)
    idx = np.argpartition(-logits, kth=m - 1, axis=1)[:, :m]
    pred_mask = np.zeros_like(Y, dtype=bool)
    np.put_along_axis(pred_mask, idx, True, axis=1)

    actual_mask = Y > 0.5
    if candidate_mask is not None:
        actual_mask = actual_mask & candidate_mask[None, :]

    hits = int((pred_mask & actual_mask).sum())
    total = int(actual_mask.sum())
    return hits, total


def recall_at_m(logits: np.ndarray, Y: np.ndarray, m: int, candidate_mask: np.ndarray | None = None) -> float:
    hits, total = topm_hits_and_total(logits, Y, m, candidate_mask)
    return hits / total if total else float("nan")
