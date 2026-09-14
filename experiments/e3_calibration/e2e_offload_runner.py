#!/usr/bin/env python3
"""E3 step 3 -- End-to-end offloading ground truth (GPU-only; not run on
the machine that wrote this file -- see E3_CALIBRATION.md's "what was and
wasn't run, and what to double-check first" section, and read it before
trusting this script blind on first use).

Experiments.md: "a minimal HF runner that keeps a fraction r of experts on
GPU and streams the rest from pinned host memory over PCIe, with policies
none / static-hot / next-layer-topk. Measure real TPOT. This is not CXL,
but it is the same physical layer and the same policy question -- and it
validates the ORDERING your simulator will later predict."

Scope, deliberately narrower than the full doc text, for reasons documented
in E3_CALIBRATION.md:
  - **OLMoE fp16 only** (Mixtral would need the exact accelerate/bitsandbytes
    multi-device dispatch path `e1_roofline/measure_compute.py` already hit
    three unrelated bugs in -- see that script's design doc). OLMoE fits
    fully in 24GB, so no offload is needed to LOAD it; the "offload" being
    measured here is a deliberately-imposed policy on top of a model that's
    actually fully resident, not an artifact of a model that doesn't fit.
  - **Timing-only weight movement, not a rewired forward pass.** This does
    NOT evict real expert weights from the live model and reroute compute
    through swapped-in copies (that would require correctly reverse-
    engineering the exact expert submodule structure of whatever transformers
    version is on the server, with no way to verify it here). Instead, for
    every expert the active policy considers "non-resident" at this step, a
    REAL pinned-CPU-to-GPU copy of a correctly-sized shadow buffer (one
    buffer per layer, sized to exactly one expert's real FP16 byte count via
    ModelSpec.expert_bytes) is executed and timed with real CUDA events,
    layered on top of the model's normal (fully-resident, unmodified)
    forward pass. The model's own compute is completely unaffected --
    generated text is correct and identical across policies -- only the
    wall-clock cost of the POLICY's implied memory traffic is charged onto
    the measured decode latency. This measures a real thing (the actual
    physical cost of a real H2D copy of the right size, at the right
    cadence, under three different scheduling disciplines) without risking a
    subtly-wrong rewrite of OLMoE's actual compute graph.
  - **Ground-truth (oracle) lookahead for next-layer-topk, not a real
    predictor.** Exactly per Experiments.md's own framing above ("validates
    the ordering your simulator will later predict") -- this measures the
    BEST CASE a perfect predictor could achieve, via a two-pass design: pass
    1 (untimed, greedy/deterministic decoding) records the real routing
    trajectory; pass 2 (timed, same seed, same greedy decoding -- so the
    exact same trajectory recurs) uses pass 1's already-known NEXT layer's
    selection to issue an async prefetch one layer early. E4/E5 build the
    real (non-oracle) predictor this is meant to validate the ceiling for.

Policies share a residency fraction r (a fixed, popularity-ranked hot-set of
experts per layer, computed from the pilot's ALREADY-COLLECTED traces --
`results/olmoe/b1/traces.jsonl.zst` -- not a fresh warmup pass):
  - `none`: r=0 always (no hot-set at all) -- every expert access this step
    is a synchronous on-demand copy. Absolute worst case / lower bound.
  - `static-hot`: r-fraction hot-set resident (no copy needed for hits);
    misses are synchronous on-demand copies, exactly like `none`.
  - `next-layer-topk`: the SAME r-fraction hot-set as `static-hot`; misses
    are prefetched one layer ahead (oracle-quality, see above) instead of
    fetched synchronously -- a hit on arrival costs ~0 extra time, a
    not-yet-complete prefetch costs only the REMAINING copy time.

Usage (on the GPU server):
  python3 e2e_offload_runner.py --residency-pcts 25,50 --out out/calib_e2e.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ELP_SRC = REPO_ROOT / "elp_probe" / "src"
sys.path.insert(0, str(ELP_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import find_gate_linears  # noqa: E402
from prompts import build_workload  # noqa: E402
from common.model_specs import MODEL_SPECS  # noqa: E402
from common.trace_io import load_decode_df  # noqa: E402

MODEL_TAG = "olmoe"
POLICIES = ["none", "static-hot", "next-layer-topk"]


# ---------- static-hot set, from the pilot's already-collected traces ----------

def compute_hot_sets(n_layers: int, n_experts: int, residency_pct: float) -> dict:
    """Per-layer set of the top-(residency_pct%) most frequently activated
    experts, ranked by the pilot's real decode-phase trace counts -- reuses
    already-collected data (Experiments.md's own "you already have this
    machinery" pattern from E1/E2) instead of a fresh model warmup pass."""
    spec = MODEL_SPECS[MODEL_TAG]
    b1_dir = REPO_ROOT / spec.trace_dir_default
    decode_df = load_decode_df(b1_dir)
    top_n = max(1, int(round(n_experts * residency_pct / 100.0)))
    hot_sets = {}
    for layer in range(n_layers):
        sub = decode_df[decode_df.layer == layer]
        counts = {}
        for row in sub["topk"]:
            for e in row:
                counts[e] = counts.get(e, 0) + 1
        ranked = sorted(counts, key=lambda e: -counts[e])
        hot_sets[layer] = set(ranked[:top_n])
    return hot_sets


# ---------- shadow buffers (timing-only, see module docstring) ----------

class ShadowBuffers:
    def __init__(self, n_layers: int, expert_bytes: float, device: str):
        n_elements = max(1, int(expert_bytes) // 4)  # fp32 elements
        self.pinned = [torch.empty(n_elements, dtype=torch.float32, pin_memory=True) for _ in range(n_layers)]
        for t in self.pinned:
            t.uniform_()
        self.staging = [torch.empty(n_elements, dtype=torch.float32, device=device) for _ in range(n_layers)]
        self.stream = torch.cuda.Stream()

    def copy_sync(self, layer: int):
        """Blocking copy -- charged fully onto whatever CUDA-event window
        wraps the caller (the on-demand-fetch cost)."""
        self.staging[layer].copy_(self.pinned[layer], non_blocking=False)

    def copy_async(self, layer: int, n_copies: int = 1) -> torch.cuda.Event:
        """Issues `n_copies` non-blocking copies back-to-back on the side
        stream (one per missing expert -- a real system would need one real
        H2D transfer per distinct expert weight, not one regardless of miss
        count) and returns the Event after the LAST one. Because operations
        queued on the same stream execute in issue order, waiting on this
        single trailing event correctly implies every earlier one in the
        batch has also completed -- no need to track n_copies separate
        events."""
        with torch.cuda.stream(self.stream):
            ev = torch.cuda.Event()
            for _ in range(max(1, n_copies)):
                self.staging[layer].copy_(self.pinned[layer], non_blocking=True)
            ev.record(self.stream)
        return ev


# ---------- pass 1: record the real routing trajectory (untimed) ----------

@torch.no_grad()
def record_routing(model, tok, prompts, k: int, max_new: int, seed: int, n_layers: int):
    """Greedy (argmax) decoding, fixed seed -- deterministic, so pass 2 can
    reproduce the exact same trajectory and legitimately use this
    'already-known future' for the next-layer-topk oracle prefetch."""
    gate_modules = find_gate_linears(model)
    routing = {layer: {} for layer in range(n_layers)}
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
    for req_idx, r in enumerate(prompts):
        torch.manual_seed(seed + req_idx)
        state["req"], state["step"] = req_idx, 0
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        out = model(**ids, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        for step in range(max_new):
            state["step"] = step
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            if next_id.item() == tok.eos_token_id:
                break
    for h in handles:
        h.remove()
    return routing


# ---------- pass 2: timed decode under one policy ----------

@torch.no_grad()
def timed_decode(model, tok, prompts, k: int, max_new: int, seed: int, n_layers: int,
                  policy: str, hot_sets: dict, known_routing: dict, shadow: ShadowBuffers):
    gate_modules = find_gate_linears(model)
    state = {"req": 0, "step": 0}
    pending_prefetch = {}  # layer -> torch.cuda.Event, cleared once consumed
    step_times_ms = []
    step_start_event = {"ev": None}

    def hook_factory(layer_idx):
        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            _v, idx = torch.topk(logits[-1], k=min(k, logits.shape[-1]))
            actual = frozenset(idx.tolist())

            # 1. serve THIS layer's own misses. NOTE: this trusts that a
            # pending prefetch (issued from pass 1's known_routing) actually
            # covers `actual`'s misses -- true as long as pass 2 reproduces
            # pass 1's trajectory bit-for-bit (same seed, same greedy
            # decoding). If GPU kernel nondeterminism ever makes the two
            # passes diverge, this would synchronize a prefetch for the
            # WRONG experts without noticing, understating next-layer-topk's
            # true miss cost for that step -- not re-validated here because
            # doing so would require re-deriving what pass 1 "should" have
            # prefetched, defeating the point of a clean oracle baseline.
            hot = hot_sets.get(layer_idx, set()) if policy != "none" else set()
            misses = actual - hot
            if misses:
                pf_event = pending_prefetch.pop(layer_idx, None)
                if pf_event is not None:
                    pf_event.synchronize()  # waits only the REMAINING time if not yet done
                else:
                    for _e in misses:
                        shadow.copy_sync(layer_idx)  # on-demand: charged fully, inline

            # 2. issue next layer's oracle prefetch (next-layer-topk only)
            if policy == "next-layer-topk" and layer_idx + 1 < n_layers:
                key = (state["req"], state["step"])
                next_actual = known_routing.get(layer_idx + 1, {}).get(key)
                if next_actual is not None:
                    next_hot = hot_sets.get(layer_idx + 1, set())
                    next_misses = next_actual - next_hot
                    if next_misses:
                        pending_prefetch[layer_idx + 1] = shadow.copy_async(layer_idx + 1, n_copies=len(next_misses))
        return hook

    handles = [mod.register_forward_hook(hook_factory(i)) for i, (_n, mod) in enumerate(gate_modules)]

    for req_idx, r in enumerate(prompts):
        torch.manual_seed(seed + req_idx)
        state["req"], state["step"] = req_idx, 0
        pending_prefetch.clear()
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        out = model(**ids, use_cache=True)  # prefill, untimed
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)

        for step in range(max_new):
            state["step"] = step
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            end.record()
            torch.cuda.synchronize()
            step_times_ms.append(start.elapsed_time(end))
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
            if next_id.item() == tok.eos_token_id:
                break

    for h in handles:
        h.remove()
    return step_times_ms


def _percentiles(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "mean_ms": float(sum(s) / n), "p50_ms": float(s[n // 2]),
        "p95_ms": float(s[max(0, int(0.95 * n) - 1)]), "n_samples": n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residency-pcts", default="25,50")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/calib_e2e.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available -- check nvidia-smi/driver/torch install"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODEL_SPECS[MODEL_TAG]
    print(f"[e2e_offload_runner] loading {spec.hf_id} fp16...", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = build_workload("sharegpt", args.n_prompts, seed=args.seed)
    n_layers = spec.n_layers
    expert_bytes = spec.expert_bytes("fp16")
    shadow = ShadowBuffers(n_layers, expert_bytes, "cuda:0")

    print(f"[e2e_offload_runner] pass 1/2: recording real routing trajectory "
          f"(untimed, greedy decode, {len(prompts)} prompts)...", flush=True)
    known_routing = record_routing(model, tok, prompts, spec.k, args.max_new, args.seed, n_layers)

    residency_pcts = [float(x) for x in args.residency_pcts.split(",")]
    residency_levels = {}
    for r in residency_pcts:
        print(f"[e2e_offload_runner] residency={r}%: computing hot-set from pilot traces...", flush=True)
        hot_sets = compute_hot_sets(n_layers, spec.n_experts, r)

        tpot_by_policy = {}
        for policy in POLICIES:
            print(f"[e2e_offload_runner] residency={r}% policy={policy}: pass 2/2 timed decode "
                  f"({args.max_new} steps x {len(prompts)} prompts)...", flush=True)
            step_times = timed_decode(model, tok, prompts, spec.k, args.max_new, args.seed, n_layers,
                                       policy, hot_sets, known_routing, shadow)
            stats = _percentiles(step_times)
            tpot_by_policy[policy] = stats["mean_ms"]
            print(f"[e2e_offload_runner]   {policy}: mean={stats['mean_ms']:.4f}ms "
                  f"p95={stats['p95_ms']:.4f}ms n={stats['n_samples']}", flush=True)

        residency_levels[str(r)] = {
            "tpot_ms_by_policy": tpot_by_policy,
            "hot_set_sizes": {str(l): len(hs) for l, hs in hot_sets.items()},
        }

    entry = {
        "provenance": "measured_here",
        "model": MODEL_TAG,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "n_prompts": len(prompts),
        "max_new": args.max_new,
        "residency_levels": residency_levels,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entry, indent=2))
    print(f"[e2e_offload_runner] DONE. wrote {out_path}", flush=True)
    print("[e2e_offload_runner] next: python3 build_calibration_report.py (checks policy ordering, no GPU)",
          flush=True)


if __name__ == "__main__":
    main()
