"""ELP-Probe trace collector (Pilot.md §3.2).

Two model targets, two boxes:
  - OLMoE-1B-7B, FP16, on the 1x RTX 4500 Ada 24GB box (~14GB weights).
  - Mixtral-8x7B, NF4 (--quant nf4), on either 2x RTX 4060 Ti 16GB box
    (~26GB weights per Pilot.md §1.6 -- fits the 32GB pooled VRAM of a
    single node with headroom for KV cache/activations). device_map="auto"
    spans both GPUs in one process automatically; no distributed/multi-node
    code needed since both GPUs sit on the same PCIe bus.

Manual prefill/decode loop (not model.generate) so phase tag ("prefill" vs
"decode") and token counter stay exact — the predictor needs this alignment.
Hooking the gate Linear (not the MoE block) captures raw router logits before
top-k dispatch.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
from utils import find_gate_linears, softmax_entropy  # noqa: E402
from prompts import build_workload, save_prompt_list  # noqa: E402

MODEL_REGISTRY = {
    "olmoe": "allenai/OLMoE-1B-7B-0924",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # gated: accept license + huggingface-cli login
}


class TraceWriter:
    """Buffered zstd-compressed JSONL writer with periodic flush."""

    def __init__(self, path: Path, flush_every: int = 20000):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "wb")
        self._cctx = zstd.ZstdCompressor(level=9)
        self._compressor = self._cctx.stream_writer(self._fh)
        self.buf = []
        self.flush_every = flush_every
        self.total_written = 0

    def add(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        payload = "\n".join(json.dumps(r) for r in self.buf) + "\n"
        self._compressor.write(payload.encode())
        self.total_written += len(self.buf)
        self.buf = []

    def close(self):
        self.flush()
        self._compressor.close()
        self._fh.close()


class Collector:
    """Owns the model, hooks, and per-token record emission + sanity counters."""

    def __init__(self, model, tok, writer: TraceWriter, model_tag: str):
        self.model = model
        self.tok = tok
        self.writer = writer
        self.model_tag = model_tag
        self.k = getattr(model.config, "num_experts_per_tok", 8)
        self.n_experts = getattr(model.config, "num_experts", None)

        self.gate_modules = find_gate_linears(model)
        self.n_layers = len(self.gate_modules)
        if self.n_experts is None:
            gm = self.gate_modules[0][1]
            # nn.Linear -> out_features; OlmoeTopKRouter (and similar custom
            # router modules) -> weight shape (num_experts, hidden_dim)
            self.n_experts = getattr(gm, "out_features", None) or getattr(gm, "num_experts", None) or gm.weight.shape[0]

        # sanity counters (Pilot.md §3.4)
        self.hook_fire_count = defaultdict(int)   # tok_key -> hook fires this token
        self.activ_per_layer = defaultdict(int)   # layer -> activation count
        self.entropies = []

        self.ctx = {"req_ids": [None], "domains": [None], "phase": "prefill", "tok": 0, "batch_id": 0}

        for i, (name, mod) in enumerate(self.gate_modules):
            mod.register_forward_hook(self._hook_factory(i))

    def _hook_factory(self, layer_idx):
        def hook(module, inputs, output):
            # OlmoeTopKRouter.forward returns (router_logits, router_scores,
            # router_indices) -- router_logits (element 0) is what we want,
            # already flattened to (tokens, n_experts). Older/other
            # architectures' gate is a plain nn.Linear returning a bare
            # tensor -- handle both.
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                # (batch, seq, n_experts) -> flatten batch-major, matching the
                # (batch*seq, n_experts) layout most MoE blocks route through.
                logits = logits.reshape(-1, logits.shape[-1])
            k = self.k
            m = max(2 * k, 8)
            topv, topi = torch.topk(logits, k=min(m, logits.shape[-1]), dim=-1)

            n_rows = logits.shape[0]
            req_ids = self.ctx["req_ids"]
            domains = self.ctx["domains"]
            phase = self.ctx["phase"]
            tok0 = self.ctx["tok"]
            batch_id = self.ctx["batch_id"]

            # prefill: n_rows = batch * seq_len (rows ordered batch-major, seq inner)
            # decode:  n_rows = batch (one token per sequence)
            if phase == "prefill":
                batch = len(req_ids)
                seq_len = n_rows // batch
                row = 0
                for b in range(batch):
                    for t in range(seq_len):
                        ent = softmax_entropy(logits[row])
                        self.writer.add({
                            "model": self.model_tag, "req": req_ids[b], "domain": domains[b],
                            "batch_id": batch_id,
                            "phase": "prefill", "layer": layer_idx, "tok": t,
                            "topk": topi[row, :k].tolist(),
                            "gate_w": [round(v, 4) for v in torch.softmax(topv[row], -1).tolist()][:k],
                            "router_entropy": round(ent, 4),
                        })
                        self.entropies.append(ent)
                        self.activ_per_layer[layer_idx] += 1
                        self.hook_fire_count[(req_ids[b], t)] += 1
                        row += 1
            else:
                for b in range(n_rows):
                    ent = softmax_entropy(logits[b])
                    self.writer.add({
                        "model": self.model_tag, "req": req_ids[b], "domain": domains[b],
                        "batch_id": batch_id,
                        "phase": "decode", "layer": layer_idx, "tok": tok0,
                        "topk": topi[b, :k].tolist(),
                        "gate_w": [round(v, 4) for v in torch.softmax(topv[b], -1).tolist()][:k],
                        "router_entropy": round(ent, 4),
                    })
                    self.entropies.append(ent)
                    self.activ_per_layer[layer_idx] += 1
                    self.hook_fire_count[(req_ids[b], tok0)] += 1
        return hook

    def assert_sanity(self):
        bad = [key for key, c in self.hook_fire_count.items() if c != self.n_layers]
        assert not bad, f"hook fire-count mismatch on {len(bad)} tokens (expected {self.n_layers}/tok); first: {bad[:5]}"
        counts = list(self.activ_per_layer.values())
        assert len(set(counts)) == 1, f"activation count differs across layers: {self.activ_per_layer}"
        import statistics
        mean_ent = statistics.mean(self.entropies) if self.entropies else 0.0
        max_ent = torch.log(torch.tensor(float(self.n_experts))).item()
        assert mean_ent > 0.05 * max_ent, (
            f"router entropy suspiciously low (mean={mean_ent:.3f}, max_possible={max_ent:.3f}) "
            "-- likely hooked the wrong module or logging argmax-of-garbage"
        )
        return {"n_layers": self.n_layers, "n_experts": self.n_experts,
                "mean_entropy": mean_ent, "max_entropy": max_ent,
                "activations_per_layer": counts[0] if counts else 0}


@torch.no_grad()
def run_request_single(model, tok, collector: Collector, req_id, domain, prompt, max_new, seed):
    torch.manual_seed(seed)
    collector.ctx.update(req_ids=[req_id], domains=[domain], phase="prefill", tok=0)
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


@torch.no_grad()
def run_batch_erosion(model, tok, collector: Collector, reqs, batch_size, max_new, seed):
    """Batch-8 erosion check (Pilot.md §2.3.6): same decode loop, but batched,
    so the hook sees the union of experts activated per layer per batch-step."""
    torch.manual_seed(seed)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    for chunk_idx, i in enumerate(range(0, len(reqs), batch_size)):
        chunk = reqs[i:i + batch_size]
        prompts = [r["prompt"] for r in chunk]
        req_ids = [r["req"] for r in chunk]
        domains = [r["domain"] for r in chunk]

        collector.ctx.update(req_ids=req_ids, domains=domains, phase="prefill", tok=0, batch_id=chunk_idx)
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model(**enc, use_cache=True)
        prefill_len = enc.input_ids.shape[1]
        collector.ctx.update(phase="decode", tok=prefill_len)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        for _ in range(max_new):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            probs = torch.softmax(out.logits[:, -1] / 0.7, dim=-1)
            next_id = torch.multinomial(probs, 1)
            collector.ctx["tok"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmoe", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--workloads", default="sharegpt,code")
    ap.add_argument("--n-sharegpt", type=int, default=150)
    ap.add_argument("--n-code", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/olmoe/b1")
    ap.add_argument("--device", default="auto",
                     help="'auto' lets accelerate split layers across all visible GPUs on this node")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--quant", default="none", choices=["none", "nf4"],
                     help="nf4 = 4-bit NF4 via bitsandbytes (Mixtral on a 32GB 2-GPU node; Pilot.md §1.6/§3.1)")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85,
                     help="fraction of each GPU's total VRAM accelerate is allowed to plan against "
                          "(headroom for KV cache/activations on top of quantized weights)")
    ap.add_argument("--cpu-offload-gb", type=int, default=32,
                     help="CPU RAM accelerate may use as overflow if weights don't fit in --gpu-mem-frac "
                          "of VRAM (nf4 only; needs llm_int8_enable_fp32_cpu_offload). "
                          "IMPORTANT: accelerate's device_map=\"auto\" planner sizes Mixtral MoE modules "
                          "using the *unquantized* footprint (~93GB) even under nf4, so raise this close "
                          "to actual free system RAM (check `free -h`) to keep the plan off disk.")
    ap.add_argument("--offload-folder", default="offload_cache",
                     help="disk fallback dir if GPU+CPU budget is still short (slow -- avoid if possible)")
    ap.add_argument("--mode", default="b1", choices=["b1", "b8"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only run this many prompts total (sanity smoke test)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = MODEL_REGISTRY[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    assert torch.cuda.is_available(), "CUDA not available -- check `nvidia-smi` / driver / torch install"
    tok = AutoTokenizer.from_pretrained(hf_id)

    if args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            # allows accelerate to spill any module that doesn't fit --gpu-mem-frac
            # of VRAM onto CPU (fp32) instead of hard-failing at load time --
            # NF4 weights alone are a tight fit in 32GB across 2x16GB GPUs with
            # accelerate's default (headroom-less) memory estimate.
            llm_int8_enable_fp32_cpu_offload=True,
        )
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory / 1e9 * args.gpu_mem_frac:.1f}GiB"
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = f"{args.cpu_offload_gb}GiB"
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        print(f"[collect] loading {hf_id} quant=nf4 device={args.device} max_memory={max_memory} "
              f"offload_folder={args.offload_folder}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map=args.device, max_memory=max_memory,
            offload_folder=args.offload_folder)
    else:
        print(f"[collect] loading {hf_id} dtype={args.dtype} device={args.device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map=args.device)
    model.eval()

    out_dir = Path(args.out)
    trace_path = out_dir / "traces.jsonl.zst"
    writer = TraceWriter(trace_path)
    collector = Collector(model, tok, writer, model_tag=args.model)
    print(f"[collect] gate modules found: {collector.n_layers} layers, "
          f"k={collector.k}, n_experts={collector.n_experts}", flush=True)

    workloads = args.workloads.split(",")
    all_prompts = []
    for w in workloads:
        n = args.n_sharegpt if w == "sharegpt" else args.n_code
        prompts = build_workload(w, n, seed=args.seed)
        all_prompts.extend(prompts)
    if args.smoke:
        all_prompts = all_prompts[: args.smoke]

    save_prompt_list(all_prompts, out_dir / "prompts_used.jsonl")
    print(f"[collect] {len(all_prompts)} prompts queued", flush=True)

    t0 = time.time()
    if args.mode == "b1":
        for i, r in enumerate(all_prompts):
            run_request_single(model, tok, collector, r["req"], r["domain"], r["prompt"],
                                args.max_new, seed=args.seed + i)
            if (i + 1) % 10 == 0 or i == len(all_prompts) - 1:
                elapsed = time.time() - t0
                print(f"[collect] {i+1}/{len(all_prompts)} requests, {elapsed:.0f}s elapsed", flush=True)
    else:
        run_batch_erosion(model, tok, collector, all_prompts, args.batch_size, args.max_new, seed=args.seed)
        print(f"[collect] batch erosion pass done, {time.time()-t0:.0f}s elapsed", flush=True)

    writer.close()

    report = collector.assert_sanity()
    report["n_records_written"] = writer.total_written
    report["elapsed_sec"] = time.time() - t0
    (out_dir / "collect_report.json").write_text(json.dumps(report, indent=2))
    print(f"[collect] DONE. sanity report: {report}", flush=True)


if __name__ == "__main__":
    main()
