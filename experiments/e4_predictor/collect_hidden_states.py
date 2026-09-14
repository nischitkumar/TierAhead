#!/usr/bin/env python3
"""GPU-only, OPTIONAL: collects pre-router hidden states for E4 ablation (d)
"+hidden state" (Experiments.md E4 step 2). NOT run on this machine (no CUDA)
-- features.py's build_layer_dataset raises MissingHiddenStateError and
train_eval.py skips this one ablation cleanly when this file's output isn't
present, exactly like E1 falls back to flop_estimate when calib_compute.json
is missing.

Why this is a separate script instead of a collect.py flag: collect.py's
existing hook fires on the gate module's OUTPUT (router logits) and is
already tuned/sanity-checked for that. Hidden states are the gate module's
INPUT -- a full hidden_size-dim float vector per (req, tok, layer), which is
100-1000x more bytes per record than everything collect.py currently logs
combined. Re-running the full 250-request workload through this hook would
produce tens of GB. Instead: reuse the exact same model-loading and prompt
pipeline as collect.py, but (a) only hook a handful of layers or all layers
(kept identical to collect.py for consistency) via a forward PRE-hook
(captures the input, not the output), and (b) only run it over a small,
fixed subsample of requests (--n-requests, default 40) -- enough for E4's
per-layer MLPs to have a meaningful train/test split, not enough to blow up
disk. Output is a single torch .pt file (a dict keyed by (req, tok, layer)),
not JSONL -- float tensors don't belong in JSON.

Usage (on the GPU server, after collect.py has already produced
results/{model}/b1/ -- this script re-runs the SAME prompts so (req, tok)
keys line up with the existing trace file):
  python3 collect_hidden_states.py --model olmoe --n-requests 40
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ELP_SRC = REPO_ROOT / "elp_probe" / "src"
sys.path.insert(0, str(ELP_SRC))

from utils import find_gate_linears  # noqa: E402
from collect import MODEL_REGISTRY  # noqa: E402


class HiddenStateCollector:
    """Forward PRE-hooks (input, not output) on each gate module. Mirrors
    collect.py's Collector context-tracking (ctx dict updated by the caller
    before each forward pass) so (req, tok) bookkeeping matches exactly."""

    def __init__(self, gate_modules):
        self.store: dict[tuple, torch.Tensor] = {}
        self.ctx = {"req_ids": [None], "phase": "prefill", "tok": 0}
        for i, (_name, mod) in enumerate(gate_modules):
            mod.register_forward_pre_hook(self._hook_factory(i))

    def _hook_factory(self, layer_idx):
        def hook(module, inputs):
            hs = inputs[0].detach().float()
            if hs.dim() == 3:
                hs = hs.reshape(-1, hs.shape[-1])
            phase = self.ctx["phase"]
            req_ids = self.ctx["req_ids"]
            tok0 = self.ctx["tok"]
            if phase == "decode":
                for b in range(hs.shape[0]):
                    self.store[(req_ids[b], tok0, layer_idx)] = hs[b].clone().cpu()
            # prefill hidden states aren't needed -- E4/E5 only train on decode rows.
        return hook


@torch.no_grad()
def run_one(model, tok, collector: HiddenStateCollector, req_id, prompt, max_new: int, seed: int):
    torch.manual_seed(seed)
    collector.ctx.update(req_ids=[req_id], phase="prefill", tok=0)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model(**ids, use_cache=True)
    prefill_len = ids.input_ids.shape[1]
    collector.ctx.update(phase="decode", tok=prefill_len)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    for _ in range(max_new):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        probs = torch.softmax(out.logits[:, -1] / 0.7, dim=-1)
        next_id = torch.multinomial(probs, 1)
        collector.ctx["tok"] += 1
        if next_id.item() == tok.eos_token_id:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="olmoe", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--prompts-file", default="",
                     help="defaults to results/{model}/b1/prompts_used.jsonl -- MUST be the same "
                          "prompt file collect.py used, so (req, tok) keys line up with traces.jsonl.zst")
    ap.add_argument("--n-requests", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available -- this script is GPU-only"

    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts_path = Path(args.prompts_file) if args.prompts_file else REPO_ROOT / f"results/{args.model}/b1/prompts_used.jsonl"
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()][: args.n_requests]
    print(f"[collect_hidden_states] {len(prompts)} requests from {prompts_path}", flush=True)

    hf_id = MODEL_REGISTRY[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
    model.eval()

    gate_modules = find_gate_linears(model)
    collector = HiddenStateCollector(gate_modules)
    print(f"[collect_hidden_states] {len(gate_modules)} gate modules hooked (pre-hook, capturing INPUT)", flush=True)

    for i, r in enumerate(prompts):
        run_one(model, tok, collector, r["req"], r["prompt"], args.max_new, seed=args.seed + i)
        if (i + 1) % 10 == 0:
            print(f"[collect_hidden_states] {i+1}/{len(prompts)} done, "
                  f"{len(collector.store)} (req,tok,layer) entries so far", flush=True)

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / f"out/hidden_states_{args.model}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(collector.store, out_path)
    print(f"[collect_hidden_states] wrote {out_path} ({len(collector.store)} entries, "
          f"~{out_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
