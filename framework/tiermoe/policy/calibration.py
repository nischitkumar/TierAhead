"""Confidence calibration (ECE, reliability diagram, temperature scaling) --
ported from experiments/e4_predictor/calibration.py. Guo et al., On
Calibration of Modern Neural Networks, ICML 2017.

This is a prerequisite for precision_gate.py's tau thresholds to mean
anything: an uncalibrated confidence score is decoration, a calibrated one
is a real decision input (Main.md section 4.4b). ``fit_temperature`` needs
torch (L-BFGS); ``expected_calibration_error`` is pure numpy and works on
ANY predictor's probability output, including the frequency table's
``.probabilities()``.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({"lo": float(lo), "hi": float(hi), "count": 0, "acc": None, "conf": None})
            continue
        acc = float(labels[mask].mean())
        conf = float(probs[mask].mean())
        ece += (count / n) * abs(acc - conf)
        bin_stats.append({"lo": float(lo), "hi": float(hi), "count": count, "acc": acc, "conf": conf})
    return float(ece), bin_stats


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Requires torch -- imported lazily here (not at module top) so
    expected_calibration_error stays usable without the `predictor` extra."""
    import torch

    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.float32)
    log_T = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).clamp(min=1e-2).item())
