"""Shared helpers: MoE gate-module discovery, entropy, small stats.

torch is imported lazily inside the functions that need it (collect.py's
GPU-side hooks) so analyze.py -- which runs anywhere, no GPU/torch required
(Pilot.md §3.3) -- can import `gini` without pulling in torch.
"""
import re


def find_gate_linears(model):
    """Locate each layer's router module (hidden -> n_experts).

    Historically a plain `nn.Linear` at `model.model.layers[i].mlp.gate`
    (Mixtral-style). Current transformers (>=4.5x) refactored OLMoE's router
    into a custom `OlmoeTopKRouter` module at the same name/path, whose
    `forward()` returns `(router_logits, router_scores, router_indices)`
    instead of a bare tensor -- see modeling_olmoe.py. We therefore match by
    **name only** (not module type), and the caller (collect.py's hook)
    handles both a bare-tensor return and a tuple return.
    """
    exact = {}
    for name, mod in model.named_modules():
        if re.search(r"\.mlp\.gate$", name) or re.search(r"\.block_sparse_moe\.gate$", name):
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                exact[int(m.group(1))] = (name, mod)

    if exact:
        return [exact[i] for i in sorted(exact.keys())]

    # fallback: anything named "...gate" under a layer-indexed path
    fallback = {}
    for name, mod in model.named_modules():
        if name.endswith("gate"):
            m = re.search(r"layers\.(\d+)\.", name)
            idx = int(m.group(1)) if m else len(fallback)
            fallback[idx] = (name, mod)
    if fallback:
        return [fallback[i] for i in sorted(fallback.keys())]

    candidates = [n for n, _ in model.named_modules() if "mlp" in n or "gate" in n or "rout" in n]
    raise RuntimeError(
        "No router module found via name matching.\n"
        "Run this to see the real structure:\n"
        "  python3 -c \"from transformers import AutoModelForCausalLM; "
        "m = AutoModelForCausalLM.from_pretrained('allenai/OLMoE-1B-7B-0924', torch_dtype='auto'); "
        "[print(type(mod).__name__, n) for n, mod in m.named_modules() if 'layers.0.' in n and "
        "('mlp' in n or 'gate' in n or 'rout' in n)]\"\n"
        "then edit find_gate_linears() in utils.py to match.\n"
        f"mlp/gate/rout-related module names seen: {candidates}"
    )


def softmax_entropy(logits_row) -> float:
    """Entropy (nats) of softmax(logits) over the full expert distribution."""
    import torch

    p = torch.softmax(logits_row.float(), dim=-1)
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def gini(freqs):
    """Gini coefficient of a non-negative frequency array."""
    import numpy as np
    x = np.sort(np.asarray(freqs, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)
