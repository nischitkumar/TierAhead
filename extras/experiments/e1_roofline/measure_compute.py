#!/usr/bin/env python3
"""E1 compute calibration (GPU, optional but recommended).

Experiments.md E1 step 2 says t_compute should come "from E3; use a
FLOP-based estimate until E3 lands." E3 (the full hardware calibration
ladder: compute + transfer + e2e offloading + NF4 fidelity) is a separate,
much larger experiment not in scope here. This script borrows only E3's
*first* sub-step -- real per-layer MoE-block compute time via CUDA events,
at batch {1,8,32} -- so E1's roofline can use a measured number instead of
a FLOP/MFU guess wherever GPU time is actually available. It does not attempt
E3's transfer or e2e measurements.

Two calibration strategies, chosen by --quant:
  --quant none (OLMoE): loads the real model with AutoModelForCausalLM +
    device_map, single GPU, fp16. No quantization, no offload -- this path
    just works, and has on every box it's been tried on.
  --quant nf4 (Mixtral): does NOT load the real model. See
    "Why no full-model load for NF4" below -- this replaces three straight
    accelerate/bitsandbytes crashes (see E1_ROOFLINE.md's design note) with
    a standalone single-GPU MoE block built via bitsandbytes' own standard,
    well-supported usage pattern.

Why no full-model load for NF4:
  Per-layer compute time depends only on GEMM shapes/dtype/quantization, not
  on trained weight VALUES, and not on loading the real ~93GB checkpoint.
  Loading the real model via device_map="auto" + CPU offload (needed because
  Mixtral NF4's ~26GB footprint doesn't fit alone on a single <32GB GPU) hit
  three separate, unrelated accelerate/bitsandbytes bugs in the multi-device
  dispatch + CPU/disk-offload machinery -- none related to what's actually
  being measured. So: build ONE MoE block (router + experts, SwiGLU FFN)
  directly on a single GPU with random weights, using bitsandbytes'
  bnb.nn.Linear4bit in its standard single-device usage pattern (construct,
  then .to(device) triggers real NF4 quantization) -- the same pattern
  PEFT/LoRA use to build quantized base layers, with no accelerate dispatch
  hooks, no CPU/disk offload, and no multi-device planning involved at all.
  Mixtral's decoder layers are architecturally identical (no per-layer size
  variation), so this one block's timing is reported for every layer index
  -- not an approximation of the per-layer values, it IS the per-layer value,
  for every layer.

Usage (on the GPU server, after download_models.sh):
  python3 measure_compute.py --model olmoe
  python3 measure_compute.py --model mixtral --quant nf4
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
ELP_SRC = REPO_ROOT / "elp_probe" / "src"
sys.path.insert(0, str(ELP_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import find_gate_linears  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402


# ---------- fp16 full-model path (OLMoE; no quantization, no offload needed) ----------

class LayerTimer:
    """Forward pre/post hooks on each MoE block, timed with paired CUDA events."""

    def __init__(self, model, n_layers: int):
        self.n_layers = n_layers
        self.events = {i: [] for i in range(n_layers)}
        self._starts = {}

    def pre_hook_factory(self, layer_idx):
        def hook(module, inputs):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._starts[layer_idx] = ev
        return hook

    def post_hook_factory(self, layer_idx):
        def hook(module, inputs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[layer_idx].append((self._starts[layer_idx], end))
        return hook

    def resolve_ms(self):
        torch.cuda.synchronize()
        out = {}
        for layer_idx, pairs in self.events.items():
            ms = [s.elapsed_time(e) for s, e in pairs]
            out[layer_idx] = ms
        return out


def find_moe_block_modules(model):
    """Reuses find_gate_linears' proven name-matching, then walks up from
    '<prefix>.gate' to '<prefix>' (the whole MoE block: router+experts+combine)."""
    gate_entries = find_gate_linears(model)
    blocks = []
    for name, _gate_mod in gate_entries:
        parent_name = name.rsplit(".", 1)[0]  # strip ".gate" (or ".gate_something")
        parent = model.get_submodule(parent_name)
        blocks.append((parent_name, parent))
    return blocks


def load_model(model_tag: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[model_tag]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    return model, tok


@torch.no_grad()
def run_batch(model, tok, batch: int, warmup_steps: int, measure_steps: int, timer: LayerTimer,
              seed: int = 0):
    torch.manual_seed(seed)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    enc = tok([prompt] * batch, return_tensors="pt", padding=True).to(model.device)
    out = model(**enc, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)

    for _ in range(warmup_steps):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
    for layer_idx in timer.events:
        timer.events[layer_idx].clear()

    for _ in range(measure_steps):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)


def calibrate_full_model(model_tag: str, device: str, batches, warmup_steps: int, measure_steps: int,
                          seed: int):
    print(f"[measure_compute] loading {model_tag} quant=none device={device}", flush=True)
    model, tok = load_model(model_tag, device)
    blocks = find_moe_block_modules(model)
    n_layers = len(blocks)
    print(f"[measure_compute] found {n_layers} MoE blocks", flush=True)

    timer = LayerTimer(model, n_layers)
    handles = []
    for i, (_name, mod) in enumerate(blocks):
        handles.append(mod.register_forward_pre_hook(timer.pre_hook_factory(i)))
        handles.append(mod.register_forward_hook(timer.post_hook_factory(i)))

    by_batch = {}
    for batch in batches:
        print(f"[measure_compute] batch={batch}: warmup={warmup_steps} measure={measure_steps}", flush=True)
        run_batch(model, tok, batch, warmup_steps, measure_steps, timer, seed=seed)
        per_layer_ms = timer.resolve_ms()
        all_samples = [ms for layer_samples in per_layer_ms.values() for ms in layer_samples]
        per_layer_mean = {i: statistics.mean(v) for i, v in per_layer_ms.items() if v}
        by_batch[str(batch)] = {
            "mean_ms": float(statistics.mean(all_samples)),
            "p50_ms": float(statistics.median(all_samples)),
            "p95_ms": float(sorted(all_samples)[max(0, int(0.95 * len(all_samples)) - 1)]),
            "n_samples": len(all_samples),
            "per_layer_mean_ms": per_layer_mean,
        }
        print(f"[measure_compute]   mean={by_batch[str(batch)]['mean_ms']:.4f} ms/layer "
              f"p95={by_batch[str(batch)]['p95_ms']:.4f} ms/layer", flush=True)

    for h in handles:
        h.remove()
    return by_batch, n_layers


# ---------- standalone single-GPU MoE block path (Mixtral NF4) ----------

class StandaloneMoEBlock(nn.Module):
    """One MoE block (router + experts, SwiGLU FFN), built directly on a
    single GPU. No accelerate dispatch, no CPU/disk offload, no full-model
    load -- see this module's docstring above for why."""

    def __init__(self, hidden_size: int, intermediate_size: int, n_experts: int, top_k: int,
                 device: str, quant: str, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, n_experts, bias=False, dtype=torch.float16, device=device)
        self.w1, self.w2, self.w3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_experts):
            if quant == "nf4":
                import bitsandbytes as bnb
                # bnb.nn.Linear4bit's standard single-device usage pattern:
                # construct with random fp32/fp16 init, then .to(device)
                # performs the REAL NF4 quantization -- no meta tensors, no
                # offload, no dispatch hooks. This is the same pattern
                # PEFT/LoRA use to build quantized base layers.
                w1 = bnb.nn.Linear4bit(hidden_size, intermediate_size, bias=False,
                                        compute_dtype=torch.float16, quant_type="nf4").to(device)
                w2 = bnb.nn.Linear4bit(intermediate_size, hidden_size, bias=False,
                                        compute_dtype=torch.float16, quant_type="nf4").to(device)
                w3 = bnb.nn.Linear4bit(hidden_size, intermediate_size, bias=False,
                                        compute_dtype=torch.float16, quant_type="nf4").to(device)
            else:
                w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16, device=device)
                w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float16, device=device)
                w3 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16, device=device)
            self.w1.append(w1)
            self.w2.append(w2)
            self.w3.append(w3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Mirrors transformers' MixtralSparseMoeBlock dispatch pattern
        (softmax -> topk -> renormalize -> per-expert masked dispatch loop),
        so kernel-launch overhead and small per-expert batch sizes match real
        decode-time behavior rather than a "run every expert on every token"
        shortcut."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hs)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hs.dtype)

        final_hidden_states = torch.zeros_like(hs)
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)
        for expert_idx in range(self.n_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hs[top_x]
            current_out = self.w2[expert_idx](F.silu(self.w1[expert_idx](current_state)) * self.w3[expert_idx](current_state))
            current_out = current_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_out.to(hs.dtype))
        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


@torch.no_grad()
def calibrate_moe_block_standalone(model_tag: str, quant: str, device: str, batches, warmup_steps: int,
                                    measure_steps: int, seed: int):
    spec = MODEL_SPECS[model_tag]
    print(f"[measure_compute] building standalone MoE block on {device} "
          f"(hidden={spec.hidden_size} intermediate={spec.expert_intermediate_size} "
          f"n_experts={spec.n_experts} top_k={spec.k} quant={quant})", flush=True)
    block = StandaloneMoEBlock(spec.hidden_size, spec.expert_intermediate_size, spec.n_experts,
                                spec.k, device, quant, seed=seed)
    block.eval()

    by_batch = {}
    for batch in batches:
        torch.manual_seed(seed + batch)
        hidden_states = torch.randn(batch, 1, spec.hidden_size, dtype=torch.float16, device=device)

        for _ in range(warmup_steps):
            block(hidden_states)
        torch.cuda.synchronize()

        events = []
        for _ in range(measure_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            block(hidden_states)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s, e in events]

        by_batch[str(batch)] = {
            "mean_ms": float(statistics.mean(samples)),
            "p50_ms": float(statistics.median(samples)),
            "p95_ms": float(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)]),
            "n_samples": len(samples),
            "per_layer_mean_ms": {i: float(statistics.mean(samples)) for i in range(spec.n_layers)},
        }
        print(f"[measure_compute]   batch={batch}: mean={by_batch[str(batch)]['mean_ms']:.4f} ms "
              f"p95={by_batch[str(batch)]['p95_ms']:.4f} ms", flush=True)
    return by_batch, spec.n_layers


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--quant", default="none", choices=["none", "nf4"])
    ap.add_argument("--device", default="auto",
                     help="only used by the fp16 full-model path (--quant none); "
                          "the nf4 standalone-block path always uses cuda:0")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/e1_roofline/out/calib_compute.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available -- check nvidia-smi/driver/torch install"

    batches = [int(b) for b in args.batches.split(",")]
    t0 = time.time()

    if args.quant == "nf4":
        by_batch, n_layers = calibrate_moe_block_standalone(
            args.model, args.quant, "cuda:0", batches, args.warmup_steps, args.measure_steps, args.seed)
    else:
        by_batch, n_layers = calibrate_full_model(
            args.model, args.device, batches, args.warmup_steps, args.measure_steps, args.seed)

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    entry = {
        "provenance": "measured_here",
        "gpu": gpu_names,
        "quant": args.quant,
        "n_layers": n_layers,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "elapsed_sec": time.time() - t0,
        "by_batch": by_batch,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib = {}
    if out_path.exists():
        calib = json.loads(out_path.read_text())
    calib[args.model] = entry
    out_path.write_text(json.dumps(calib, indent=2))
    print(f"[measure_compute] DONE. wrote {out_path} (model={args.model})", flush=True)


if __name__ == "__main__":
    main()
