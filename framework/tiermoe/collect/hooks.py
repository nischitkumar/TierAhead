"""MoE gate-module discovery + entropy. Ported from elp_probe/src/utils.py.

torch is imported lazily inside the functions that need it, so importing
this module (e.g. from a doctor/CLI context that just wants to inspect a
model's structure) doesn't require torch's CUDA build specifically -- CPU
torch is enough to run find_gate_linears against a loaded model, only the
actual collection run needs CUDA.
"""
from __future__ import annotations

import re


def find_gate_linears(model):
    """Locates each layer's router module (hidden -> n_experts). Historically
    a plain nn.Linear at model.model.layers[i].mlp.gate (Mixtral-style);
    current transformers (>=4.5x) refactored OLMoE's router into a custom
    OlmoeTopKRouter module at the same name/path whose forward() returns
    (router_logits, router_scores, router_indices) instead of a bare tensor.
    Matched by NAME only (not module type) so both shapes work; the caller's
    hook handles both a bare-tensor and a tuple return."""
    exact = {}
    for name, mod in model.named_modules():
        if re.search(r"\.mlp\.gate$", name) or re.search(r"\.block_sparse_moe\.gate$", name):
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                exact[int(m.group(1))] = (name, mod)
    if exact:
        return [exact[i] for i in sorted(exact.keys())]

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
        "No router module found via name matching. Run:\n"
        "  python3 -c \"from transformers import AutoModelForCausalLM; "
        "m = AutoModelForCausalLM.from_pretrained('<hf_id>', torch_dtype='auto'); "
        "[print(type(mod).__name__, n) for n, mod in m.named_modules() if 'layers.0.' in n and "
        "('mlp' in n or 'gate' in n or 'rout' in n)]\"\n"
        "then extend the name patterns in find_gate_linears() to match.\n"
        f"mlp/gate/rout-related module names seen: {candidates}"
    )


def softmax_entropy(logits_row) -> float:
    import torch

    p = torch.softmax(logits_row.float(), dim=-1).clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())
