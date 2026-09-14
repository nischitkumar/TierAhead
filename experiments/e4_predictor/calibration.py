"""Confidence calibration (Experiments.md E4 step 4): reliability diagram,
Expected Calibration Error, and temperature scaling on the predictor's
per-expert sigmoid probabilities.

Guo et al., On Calibration of Modern Neural Networks, ICML 2017 -- the
standard reference for both ECE-via-binning and temperature scaling.

Calibration here is over *flattened (row, candidate-expert) pairs*: for
n_experts candidates and N held-out rows, that's N*n_experts binary
"was this specific expert actually needed" outcomes, each with the model's
claimed probability. This is exactly the quantity E7's precision-gating
policy consumes (a calibrated probability per candidate expert), so
calibrating in this space -- rather than only calibrating "was the top-1
prediction right" -- is the one that's actually load-bearing downstream.
"""
import numpy as np
import torch


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """probs, labels: flat 1D arrays (same length), labels in {0,1}. Returns
    (ece, bin_stats) where bin_stats is a list of per-bin dicts -- the raw
    material for a reliability diagram (plot bin['conf'] vs bin['acc'])."""
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
    """Single scalar T minimizing BCE-with-logits(logits/T, labels) via
    L-BFGS (Guo et al.'s standard temperature-scaling recipe, adapted from
    softmax-NLL to the multi-label BCE loss this predictor actually uses).
    T > 1 means the model is overconfident (raw probabilities should be
    pulled toward 0.5); T < 1 means underconfident. Clamped to >=1e-2 to
    keep the division well-defined regardless of what L-BFGS proposes."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.float32)
    log_T = torch.nn.Parameter(torch.zeros(1))  # optimize in log-space so T=exp(log_T) stays positive
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).clamp(min=1e-2).item())
