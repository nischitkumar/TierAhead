"""Predictor v2: a small per-expert-sigmoid MLP (Main.md section 4.4,
"prefetch-v2"). Ported from experiments/e4_predictor/mlp.py.

Requires torch (``pip install tiermoe[predictor]``) -- CPU/MPS only, no CUDA
needed. Experiments.md E4/E5 verified full training sweeps complete in under
a minute on CPU alone; this module is imported lazily by
tiermoe.policy.__init__ specifically so the rest of the framework (roofline,
analyze, sim, tco, precision_gate) never requires torch to be installed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_experts))

    def forward(self, x):
        return self.net(x)  # raw logits


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each feature column on TRAIN statistics only. Not optional --
    see E4_PREDICTOR.md's documented incident: without this, feature
    variants with smaller native scale (gate weights, router entropy)
    silently failed to converge within a fixed epoch budget and looked like
    a (false) finding that "more information hurts." Fixed here structurally
    by always standardizing before training."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
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
        return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def topm_hits_and_total(logits: np.ndarray, Y: np.ndarray, m: int,
                         candidate_mask: np.ndarray | None = None) -> tuple[int, int]:
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


def sigmoid_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))
