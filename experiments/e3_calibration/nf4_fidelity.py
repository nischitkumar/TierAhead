#!/usr/bin/env python3
"""E3 step 4 -- NF4 routing-fidelity spot-check (GPU-only; not run on the
machine that wrote this file). Closes pilot open item #2 honestly instead
of leaving it open: "you cannot fit FP16 Mixtral [to compare against its own
NF4 version], but you *can* run FP16 vs NF4 on a small MoE" (Experiments.md
E3 step 4) -- OLMoE fits FP16 on this box (~14GB on a 24GB GPU), so this
loads it twice (once FP16, once NF4-quantized) and measures how often the
two versions agree on which experts to route to.

Method: run the SAME small prompt set through both models (loaded and freed
sequentially, not simultaneously -- OLMoE FP16 + a second full model would
be tight on 24GB otherwise), hook every gate module on both (reusing
`find_gate_linears`, exactly as `collect.py`/`measure_compute.py` already
do), and for every (layer, token) position compute the Jaccard overlap
between the FP16 top-k expert set and the NF4 top-k expert set. Averaged
over tokens, per layer, then over layers -- this answers "does quantizing
the weights change WHICH experts the router picks", which is the thing that
actually matters for whether a predictor trained on FP16 traces (or NF4
ones) transfers to the other precision.

Deliberately does NOT attempt this for Mixtral (that's the whole reason
this experiment is needed at all: Mixtral FP16 doesn't fit on this box to
compare against) -- cite QLoRA (arXiv:2305.14314) for Mixtral's NF4
fidelity instead, per Experiments.md's own instruction.

Usage (on the GPU server):
  python3 nf4_fidelity.py --n-prompts 20 --out out/calib_nf4_fidelity.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ELP_SRC = REPO_ROOT / "elp_probe" / "src"
sys.path.insert(0, str(ELP_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import find_gate_linears  # noqa: E402
from prompts import build_workload  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402

MODEL_TAG = "olmoe"  # the only model this fits both precisions for on this box


def capture_topk(model, tok, prompts, k: int, max_new: int, seed: int):
    """Returns {(layer, req_idx, tok_idx): frozenset(expert_ids)} across
    prefill+decode for every prompt -- same manual-loop style as
    `collect.py`'s Collector (a fresh, lighter-weight capture here rather
    than reusing collect.py's TraceWriter/zstd machinery, since this needs
    only in-memory top-k sets for a small prompt count, not a persisted
    trace file)."""
    gate_modules = find_gate_linears(model)
    n_layers = len(gate_modules)
    captured = defaultdict(dict)  # layer -> {(req_idx, tok_idx): frozenset}
    state = {"req_idx": 0, "tok_idx": 0}

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            # last row = most recent token position (works for both the
            # multi-row prefill pass and the single-row decode step)
            _topv, topi = torch.topk(logits[-1], k=min(k, logits.shape[-1]))
            captured[layer_idx][(state["req_idx"], state["tok_idx"])] = frozenset(topi.tolist())
        return hook

    handles = [mod.register_forward_hook(hook_factory(i)) for i, (_name, mod) in enumerate(gate_modules)]

    model.eval()
    with torch.no_grad():
        for req_idx, r in enumerate(prompts):
            torch.manual_seed(seed + req_idx)
            state["req_idx"] = req_idx
            state["tok_idx"] = 0
            ids = tok(r["prompt"], return_tensors="pt").to(model.device)
            out = model(**ids, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            for t in range(max_new):
                state["tok_idx"] = t
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1:].argmax(-1)
                if next_id.item() == tok.eos_token_id:
                    break

    for h in handles:
        h.remove()
    return captured, n_layers


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_model(quant: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[MODEL_TAG]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, quantization_config=bnb_config,
                                                       device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map="auto")
    return model, tok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-prompts", type=int, default=20,
                     help="small by design -- this is a spot-check, not a held-out-test-set claim "
                          "(see E3_CALIBRATION.md's honest caveat on sample size)")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/calib_nf4_fidelity.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available -- check nvidia-smi/driver/torch install"

    prompts = build_workload("sharegpt", args.n_prompts, seed=args.seed)
    spec = MODEL_SPECS[MODEL_TAG]

    print(f"[nf4_fidelity] loading {spec.hf_id} fp16...", flush=True)
    model_fp16, tok = load_model("none")
    print(f"[nf4_fidelity] capturing top-k on {len(prompts)} prompts (fp16)...", flush=True)
    captured_fp16, n_layers = capture_topk(model_fp16, tok, prompts, spec.k, args.max_new, args.seed)
    del model_fp16
    torch.cuda.empty_cache()

    print(f"[nf4_fidelity] loading {spec.hf_id} nf4...", flush=True)
    model_nf4, tok2 = load_model("nf4")
    print(f"[nf4_fidelity] capturing top-k on {len(prompts)} prompts (nf4)...", flush=True)
    captured_nf4, _ = capture_topk(model_nf4, tok2, prompts, spec.k, args.max_new, args.seed)
    del model_nf4
    torch.cuda.empty_cache()

    per_layer_jaccard = {}
    for layer in range(n_layers):
        keys = set(captured_fp16.get(layer, {})) & set(captured_nf4.get(layer, {}))
        if not keys:
            continue
        scores = [jaccard(captured_fp16[layer][key], captured_nf4[layer][key]) for key in keys]
        per_layer_jaccard[str(layer)] = float(sum(scores) / len(scores))
        print(f"[nf4_fidelity]   layer {layer}: mean Jaccard={per_layer_jaccard[str(layer)]:.3f} "
              f"(n={len(scores)} positions)", flush=True)

    mean_jaccard = float(sum(per_layer_jaccard.values()) / len(per_layer_jaccard)) if per_layer_jaccard else float("nan")

    entry = {
        "provenance": "measured_here",
        "model": MODEL_TAG,
        "n_prompts": len(prompts),
        "max_new": args.max_new,
        "per_layer_jaccard": per_layer_jaccard,
        "mean_jaccard_similarity": mean_jaccard,
        "note": "Mixtral NF4 fidelity not independently measured (fp16 Mixtral doesn't fit on this "
                "box) -- cite QLoRA arXiv:2305.14314 for that side, per Experiments.md's instruction.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[nf4_fidelity] DONE. mean_jaccard={mean_jaccard:.3f}. wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
