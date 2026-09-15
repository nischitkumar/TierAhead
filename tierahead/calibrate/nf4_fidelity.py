"""NF4 routing-fidelity spot-check (GPU-required, OLMoE only -- the only
shipped model that fits BOTH fp16 and nf4 on a single 24GB GPU to compare
against each other). Measures how often FP16 and NF4 versions of the same
model agree on which experts to route to, via Jaccard overlap of per-
position top-k sets. Cite QLoRA (arXiv:2305.14314) for Mixtral's NF4
fidelity instead -- Mixtral FP16 (~87GB) doesn't fit for a real comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def capture_topk(model, tok, prompts, k: int, max_new: int, seed: int):
    import torch

    from tierahead.collect.hooks import find_gate_linears

    gate_modules = find_gate_linears(model)
    captured = defaultdict(dict)
    state = {"req_idx": 0, "tok_idx": 0}

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            _v, idx = torch.topk(logits[-1], k=min(k, logits.shape[-1]))
            captured[layer_idx][(state["req_idx"], state["tok_idx"])] = frozenset(idx.tolist())
        return hook

    handles = [mod.register_forward_hook(hook_factory(i)) for i, (_n, mod) in enumerate(gate_modules)]
    model.eval()
    with torch.no_grad():
        for req_idx, r in enumerate(prompts):
            torch.manual_seed(seed + req_idx)
            state["req_idx"], state["tok_idx"] = req_idx, 0
            ids = tok(r["prompt"], return_tensors="pt").to(model.device)
            out = model(**ids, use_cache=True)
            past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for t in range(max_new):
                state["tok_idx"] = t
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
                if next_id.item() == tok.eos_token_id:
                    break
    for h in handles:
        h.remove()
    return captured, len(gate_modules)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_model(quant: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tierahead.specs import MODEL_SPECS

    spec = MODEL_SPECS["olmoe"]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, quantization_config=bnb_config, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map="auto")
    return model, tok


def main(argv=None):
    from tierahead.hw import probe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calib_nf4_fidelity.json")
    args = ap.parse_args(argv)

    if not probe().can_run_real_gpu_calibration:
        print("tierahead calibrate nf4_fidelity: refusing -- no CUDA on this machine.", file=sys.stderr)
        sys.exit(1)

    import torch
    from tierahead.collect.prompts import build_workload
    from tierahead.specs import MODEL_SPECS

    prompts = build_workload("sharegpt", args.n_prompts, seed=args.seed)
    spec = MODEL_SPECS["olmoe"]

    model_fp16, tok = load_model("none")
    captured_fp16, n_layers = capture_topk(model_fp16, tok, prompts, spec.k, args.max_new, args.seed)
    del model_fp16
    torch.cuda.empty_cache()

    model_nf4, tok2 = load_model("nf4")
    captured_nf4, _ = capture_topk(model_nf4, tok2, prompts, spec.k, args.max_new, args.seed)
    del model_nf4
    torch.cuda.empty_cache()

    per_layer_jaccard = {}
    for layer in range(n_layers):
        keys = set(captured_fp16.get(layer, {})) & set(captured_nf4.get(layer, {}))
        if not keys:
            continue
        scores = [jaccard(captured_fp16[layer][k], captured_nf4[layer][k]) for k in keys]
        per_layer_jaccard[str(layer)] = float(sum(scores) / len(scores))

    mean_jaccard = float(sum(per_layer_jaccard.values()) / len(per_layer_jaccard)) if per_layer_jaccard else float("nan")
    entry = {"provenance": "measured_here", "model": "olmoe", "n_prompts": len(prompts), "max_new": args.max_new,
             "per_layer_jaccard": per_layer_jaccard, "mean_jaccard_similarity": mean_jaccard,
             "note": "Mixtral NF4 fidelity not independently measured (fp16 Mixtral doesn't fit alongside "
                     "for comparison) -- cite QLoRA arXiv:2305.14314 for that side."}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[calibrate.nf4_fidelity] mean_jaccard={mean_jaccard:.3f}, wrote {out_path}")


if __name__ == "__main__":
    main()
