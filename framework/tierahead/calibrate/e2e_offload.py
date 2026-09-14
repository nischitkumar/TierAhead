"""End-to-end offloading ground truth (GPU-required, OLMoE only). Measures
real TPOT under three policies (none / static-hot / next-layer-topk) that
share a fixed residency fraction r (a hot-set ranked by the pilot's already-
collected trace frequencies). Does NOT evict real expert weights and reroute
compute -- that would need reverse-engineering the exact expert submodule
layout of whatever transformers version is installed, unverifiable in
advance. Instead the model computes normally (fully resident, correct
output, identical across policies), and for every expert a policy considers
"non-resident" at a step, a REAL pinned-CPU->GPU copy of a correctly-sized
dummy buffer is executed and timed with real CUDA events, charged onto that
step's latency -- measuring the real physical cost of the right-sized,
right-cadence memory movement under three scheduling disciplines, without
risking a silently-wrong rewrite of the real compute graph.

next-layer-topk uses a two-pass oracle design: pass 1 (untimed, greedy
decode) records the real routing trajectory; pass 2 (timed, same seed, same
greedy decode -- so the same trajectory recurs) uses pass 1's already-known
NEXT layer's selection to issue an async prefetch one layer early. This
measures the CEILING a perfect predictor could achieve, which is what
tierahead.policy's real (non-oracle) predictors are validated against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compute_hot_sets(n_layers: int, n_experts: int, residency_pct: float, trace_dir: Path) -> dict:
    from tierahead.traces.io import load_decode_df

    decode_df = load_decode_df(trace_dir)
    top_n = max(1, int(round(n_experts * residency_pct / 100.0)))
    hot_sets = {}
    for layer in range(n_layers):
        sub = decode_df[decode_df.layer == layer]
        counts: dict = {}
        for row in sub["topk"]:
            for e in row:
                counts[e] = counts.get(e, 0) + 1
        ranked = sorted(counts, key=lambda e: -counts[e])
        hot_sets[layer] = set(ranked[:top_n])
    return hot_sets


class ShadowBuffers:
    def __init__(self, n_layers: int, expert_bytes: float, device: str):
        import torch

        n_elements = max(1, int(expert_bytes) // 4)
        self.pinned = [torch.empty(n_elements, dtype=torch.float32, pin_memory=True) for _ in range(n_layers)]
        for t in self.pinned:
            t.uniform_()
        self.staging = [torch.empty(n_elements, dtype=torch.float32, device=device) for _ in range(n_layers)]
        self.stream = torch.cuda.Stream()

    def copy_sync(self, layer: int):
        self.staging[layer].copy_(self.pinned[layer], non_blocking=False)

    def copy_async(self, layer: int, n_copies: int = 1):
        import torch

        with torch.cuda.stream(self.stream):
            ev = torch.cuda.Event()
            for _ in range(max(1, n_copies)):
                self.staging[layer].copy_(self.pinned[layer], non_blocking=True)
            ev.record(self.stream)
        return ev


def record_routing(model, tok, prompts, k, max_new, seed, n_layers):
    import torch

    from tierahead.collect.hooks import find_gate_linears

    gate_modules = find_gate_linears(model)
    routing = {l: {} for l in range(n_layers)}
    state = {"req": 0, "step": 0}

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            _v, idx = torch.topk(logits[-1], k=min(k, logits.shape[-1]))
            routing[layer_idx][(state["req"], state["step"])] = frozenset(idx.tolist())
        return hook

    handles = [mod.register_forward_hook(hook_factory(i)) for i, (_n, mod) in enumerate(gate_modules)]
    with torch.no_grad():
        for req_idx, r in enumerate(prompts):
            torch.manual_seed(seed + req_idx)
            state["req"], state["step"] = req_idx, 0
            ids = tok(r["prompt"], return_tensors="pt").to(model.device)
            out = model(**ids, use_cache=True)
            past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for step in range(max_new):
                state["step"] = step
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
                if next_id.item() == tok.eos_token_id:
                    break
    for h in handles:
        h.remove()
    return routing


def timed_decode(model, tok, prompts, k, max_new, seed, n_layers, policy, hot_sets, known_routing, shadow):
    import torch

    from tierahead.collect.hooks import find_gate_linears

    gate_modules = find_gate_linears(model)
    state = {"req": 0, "step": 0}
    pending_prefetch: dict = {}
    step_times_ms = []

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            _v, idx = torch.topk(logits[-1], k=min(k, logits.shape[-1]))
            actual = frozenset(idx.tolist())
            hot = hot_sets.get(layer_idx, set()) if policy != "none" else set()
            misses = actual - hot
            if misses:
                pf_event = pending_prefetch.pop(layer_idx, None)
                if pf_event is not None:
                    pf_event.synchronize()
                else:
                    for _e in misses:
                        shadow.copy_sync(layer_idx)
            if policy == "next-layer-topk" and layer_idx + 1 < n_layers:
                next_actual = known_routing.get(layer_idx + 1, {}).get((state["req"], state["step"]))
                if next_actual is not None:
                    next_misses = next_actual - hot_sets.get(layer_idx + 1, set())
                    if next_misses:
                        pending_prefetch[layer_idx + 1] = shadow.copy_async(layer_idx + 1, n_copies=len(next_misses))
        return hook

    handles = [mod.register_forward_hook(hook_factory(i)) for i, (_n, mod) in enumerate(gate_modules)]
    with torch.no_grad():
        for req_idx, r in enumerate(prompts):
            torch.manual_seed(seed + req_idx)
            state["req"], state["step"] = req_idx, 0
            pending_prefetch.clear()
            ids = tok(r["prompt"], return_tensors="pt").to(model.device)
            out = model(**ids, use_cache=True)
            past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for step in range(max_new):
                state["step"] = step
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
                end.record()
                torch.cuda.synchronize()
                step_times_ms.append(start.elapsed_time(end))
                past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
                if next_id.item() == tok.eos_token_id:
                    break
    for h in handles:
        h.remove()
    return step_times_ms


def _percentiles(samples):
    s = sorted(samples)
    n = len(s)
    return {"mean_ms": sum(s) / n, "p50_ms": s[n // 2], "p95_ms": s[max(0, int(0.95 * n) - 1)], "n_samples": n}


def main(argv=None):
    from tierahead.hw import probe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--residency-pcts", default="25,50")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="calib_e2e.json")
    args = ap.parse_args(argv)

    if not probe().can_run_real_gpu_calibration:
        print("tierahead calibrate e2e_offload: refusing -- no CUDA on this machine.", file=sys.stderr)
        sys.exit(1)

    import torch
    from tierahead.collect.prompts import build_workload
    from tierahead.specs import MODEL_SPECS
    from tierahead.traces.io import resolve_trace_dir

    spec = MODEL_SPECS["olmoe"]
    data_root = Path(args.data_root) if args.data_root else None
    trace_dir = resolve_trace_dir(spec.trace_dir_default, data_root)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = build_workload("sharegpt", args.n_prompts, seed=args.seed)
    n_layers, expert_bytes = spec.n_layers, spec.expert_bytes("fp16")
    shadow = ShadowBuffers(n_layers, expert_bytes, "cuda:0")

    known_routing = record_routing(model, tok, prompts, spec.k, args.max_new, args.seed, n_layers)

    residency_levels = {}
    for r in [float(x) for x in args.residency_pcts.split(",")]:
        hot_sets = compute_hot_sets(n_layers, spec.n_experts, r, trace_dir)
        tpot_by_policy = {}
        for policy in ["none", "static-hot", "next-layer-topk"]:
            step_times = timed_decode(model, tok, prompts, spec.k, args.max_new, args.seed, n_layers,
                                        policy, hot_sets, known_routing, shadow)
            tpot_by_policy[policy] = _percentiles(step_times)["mean_ms"]
        residency_levels[str(r)] = {"tpot_ms_by_policy": tpot_by_policy,
                                     "hot_set_sizes": {str(l): len(hs) for l, hs in hot_sets.items()}}

    entry = {"provenance": "measured_here", "model": "olmoe",
             "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
             "n_prompts": len(prompts), "max_new": args.max_new, "residency_levels": residency_levels}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[calibrate.e2e_offload] wrote {out_path}")


if __name__ == "__main__":
    main()
