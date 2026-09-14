"""Per-layer/per-expert compute-time calibration (GPU-required). Ported from
experiments/e1_roofline/measure_compute.py. Two strategies:

  --quant none: loads the real model with AutoModelForCausalLM + device_map,
    single GPU, fp16. Used for OLMoE (fits fully in 24GB, no offload needed).
  --quant nf4: does NOT load the real model. Per-layer compute time depends
    only on GEMM shapes/dtype/quantization -- never on trained weight values
    or on loading the full checkpoint -- so this builds ONE standalone MoE
    block (router + experts, SwiGLU FFN) directly on a single GPU with
    bitsandbytes' bnb.nn.Linear4bit in its standard single-device usage
    pattern (construct, then .to(device) triggers real NF4 quantization).
    No accelerate dispatch, no CPU/disk offload -- avoids the three
    unrelated accelerate/bitsandbytes bugs a full quantized Mixtral load
    hits (see this module's git history / experiments/e1_roofline/E1_ROOFLINE.md
    for the specifics). Mixtral's decoder layers are architecturally
    identical across depth, so this one block's timing IS the per-layer
    number for every layer index, not an approximation of it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def load_full_model(model_tag: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tierahead.specs import MODEL_SPECS

    spec = MODEL_SPECS[model_tag]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    return model, tok


def find_moe_block_modules(model):
    from tierahead.collect.hooks import find_gate_linears

    blocks = []
    for name, _gate_mod in find_gate_linears(model):
        parent_name = name.rsplit(".", 1)[0]
        blocks.append((parent_name, model.get_submodule(parent_name)))
    return blocks


class LayerTimer:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.events: dict = {i: [] for i in range(n_layers)}
        self._starts: dict = {}

    def pre_hook_factory(self, layer_idx: int):
        import torch

        def hook(module, inputs):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._starts[layer_idx] = ev
        return hook

    def post_hook_factory(self, layer_idx: int):
        import torch

        def hook(module, inputs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[layer_idx].append((self._starts[layer_idx], end))
        return hook

    def resolve_ms(self):
        import torch

        torch.cuda.synchronize()
        return {i: [s.elapsed_time(e) for s, e in pairs] for i, pairs in self.events.items()}


def calibrate_full_model(model_tag: str, device: str, batches, warmup_steps: int, measure_steps: int, seed: int):
    import torch

    model, tok = load_full_model(model_tag, device)
    blocks = find_moe_block_modules(model)
    n_layers = len(blocks)
    timer = LayerTimer(n_layers)
    handles = []
    for i, (_name, mod) in enumerate(blocks):
        handles.append(mod.register_forward_pre_hook(timer.pre_hook_factory(i)))
        handles.append(mod.register_forward_hook(timer.post_hook_factory(i)))

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    by_batch = {}
    with torch.no_grad():
        for batch in batches:
            torch.manual_seed(seed)
            prompt = "The quick brown fox jumps over the lazy dog. " * 4
            enc = tok([prompt] * batch, return_tensors="pt", padding=True).to(model.device)
            out = model(**enc, use_cache=True)
            past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for _ in range(warmup_steps):
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for layer_idx in timer.events:
                timer.events[layer_idx].clear()
            for _ in range(measure_steps):
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)

            per_layer_ms = timer.resolve_ms()
            all_samples = [ms for v in per_layer_ms.values() for ms in v]
            by_batch[str(batch)] = {
                "mean_ms": float(statistics.mean(all_samples)), "p50_ms": float(statistics.median(all_samples)),
                "p95_ms": float(sorted(all_samples)[max(0, int(0.95 * len(all_samples)) - 1)]),
                "n_samples": len(all_samples),
                "per_layer_mean_ms": {i: statistics.mean(v) for i, v in per_layer_ms.items() if v},
            }
    for h in handles:
        h.remove()
    return by_batch, n_layers


def calibrate_moe_block_standalone(model_tag: str, quant: str, batches, warmup_steps: int, measure_steps: int, seed: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tierahead.specs import MODEL_SPECS

    spec = MODEL_SPECS[model_tag]
    device = "cuda:0"

    class StandaloneMoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.n_experts, self.top_k = spec.n_experts, spec.k
            self.gate = nn.Linear(spec.hidden_size, spec.n_experts, bias=False, dtype=torch.float16, device=device)
            self.w1, self.w2, self.w3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            for _ in range(spec.n_experts):
                if quant == "nf4":
                    import bitsandbytes as bnb
                    self.w1.append(bnb.nn.Linear4bit(spec.hidden_size, spec.expert_intermediate_size, bias=False,
                                                       compute_dtype=torch.float16, quant_type="nf4").to(device))
                    self.w2.append(bnb.nn.Linear4bit(spec.expert_intermediate_size, spec.hidden_size, bias=False,
                                                       compute_dtype=torch.float16, quant_type="nf4").to(device))
                    self.w3.append(bnb.nn.Linear4bit(spec.hidden_size, spec.expert_intermediate_size, bias=False,
                                                       compute_dtype=torch.float16, quant_type="nf4").to(device))
                else:
                    self.w1.append(nn.Linear(spec.hidden_size, spec.expert_intermediate_size, bias=False, dtype=torch.float16, device=device))
                    self.w2.append(nn.Linear(spec.expert_intermediate_size, spec.hidden_size, bias=False, dtype=torch.float16, device=device))
                    self.w3.append(nn.Linear(spec.hidden_size, spec.expert_intermediate_size, bias=False, dtype=torch.float16, device=device))

        def forward(self, hidden_states):
            bsz, seq_len, hdim = hidden_states.shape
            hs = hidden_states.view(-1, hdim)
            routing_weights = F.softmax(self.gate(hs), dim=-1, dtype=torch.float32)
            routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(hs.dtype)
            out = torch.zeros_like(hs)
            mask = F.one_hot(selected, num_classes=self.n_experts).permute(2, 1, 0)
            for e in range(self.n_experts):
                idx, top_x = torch.where(mask[e])
                if top_x.numel() == 0:
                    continue
                cur = hs[top_x]
                cur_out = self.w2[e](F.silu(self.w1[e](cur)) * self.w3[e](cur))
                out.index_add_(0, top_x, (cur_out * routing_weights[top_x, idx, None]).to(hs.dtype))
            return out.view(bsz, seq_len, hdim)

    block = StandaloneMoEBlock().eval()
    by_batch = {}
    with torch.no_grad():
        for batch in batches:
            torch.manual_seed(seed + batch)
            hs = torch.randn(batch, 1, spec.hidden_size, dtype=torch.float16, device=device)
            for _ in range(warmup_steps):
                block(hs)
            torch.cuda.synchronize()
            events = []
            for _ in range(measure_steps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                block(hs)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            samples = [s.elapsed_time(e) for s, e in events]
            by_batch[str(batch)] = {
                "mean_ms": float(statistics.mean(samples)), "p50_ms": float(statistics.median(samples)),
                "p95_ms": float(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]), "n_samples": len(samples),
                "per_layer_mean_ms": {i: float(statistics.mean(samples)) for i in range(spec.n_layers)},
            }
    return by_batch, spec.n_layers


def main(argv=None):
    from tierahead.hw import probe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["olmoe", "mixtral"])
    ap.add_argument("--quant", default="none", choices=["none", "nf4"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calib_compute.json")
    args = ap.parse_args(argv)

    caps = probe()
    if not caps.can_run_real_gpu_calibration:
        print("tierahead calibrate compute: refusing -- no CUDA on this machine. "
              "roofline/sim fall back to flop_estimate automatically without this file.", file=sys.stderr)
        sys.exit(1)

    import torch

    batches = [int(b) for b in args.batches.split(",")]
    t0 = time.time()
    if args.quant == "nf4":
        by_batch, n_layers = calibrate_moe_block_standalone(args.model, args.quant, batches, args.warmup_steps, args.measure_steps, args.seed)
    else:
        by_batch, n_layers = calibrate_full_model(args.model, args.device, batches, args.warmup_steps, args.measure_steps, args.seed)

    entry = {
        "provenance": "measured_here", "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "quant": args.quant, "n_layers": n_layers, "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps, "elapsed_sec": time.time() - t0, "by_batch": by_batch,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib = json.loads(out_path.read_text()) if out_path.exists() else {}
    calib[args.model] = entry
    out_path.write_text(json.dumps(calib, indent=2))
    print(f"[calibrate.compute] wrote {out_path} (model={args.model})")


if __name__ == "__main__":
    main()
