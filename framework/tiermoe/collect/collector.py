"""Trace collector core: TraceWriter (buffered zstd JSONL) + Collector
(forward-hook registration, sanity assertions) + the manual prefill/decode
loops. Ported from elp_probe/src/collect.py. CUDA-required -- every entry
point checks tiermoe.hw.probe() first.

Manual decode loop (not model.generate) so the phase tag and token counter
stay exact -- the predictor needs this alignment. Hooking the gate Linear/
router module (not the whole MoE block) captures raw logits before top-k
dispatch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

MODEL_REGISTRY = {"olmoe": "allenai/OLMoE-1B-7B-0924", "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"}


class TraceWriter:
    def __init__(self, path: Path, flush_every: int = 20000):
        import zstandard as zstd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "wb")
        self._compressor = zstd.ZstdCompressor(level=9).stream_writer(self._fh)
        self.buf: list[dict] = []
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
        from tiermoe.collect.hooks import find_gate_linears

        self.model, self.tok, self.writer, self.model_tag = model, tok, writer, model_tag
        self.k = getattr(model.config, "num_experts_per_tok", 8)
        self.n_experts = getattr(model.config, "num_experts", None)
        self.gate_modules = find_gate_linears(model)
        self.n_layers = len(self.gate_modules)
        if self.n_experts is None:
            gm = self.gate_modules[0][1]
            self.n_experts = getattr(gm, "out_features", None) or getattr(gm, "num_experts", None) or gm.weight.shape[0]

        self.hook_fire_count: dict = defaultdict(int)
        self.activ_per_layer: dict = defaultdict(int)
        self.entropies: list[float] = []
        self.ctx = {"req_ids": [None], "domains": [None], "phase": "prefill", "tok": 0, "batch_id": 0}

        for i, (_name, mod) in enumerate(self.gate_modules):
            mod.register_forward_hook(self._hook_factory(i))

    def _hook_factory(self, layer_idx: int):
        import torch

        from tiermoe.collect.hooks import softmax_entropy

        def hook(module, inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            logits = raw.detach().float()
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.shape[-1])
            k = self.k
            m = max(2 * k, 8)
            topv, topi = torch.topk(logits, k=min(m, logits.shape[-1]), dim=-1)

            n_rows = logits.shape[0]
            req_ids, domains, phase = self.ctx["req_ids"], self.ctx["domains"], self.ctx["phase"]
            tok0, batch_id = self.ctx["tok"], self.ctx["batch_id"]

            if phase == "prefill":
                batch = len(req_ids)
                seq_len = n_rows // batch
                row = 0
                for b in range(batch):
                    for t in range(seq_len):
                        ent = softmax_entropy(logits[row])
                        self.writer.add({
                            "model": self.model_tag, "req": req_ids[b], "domain": domains[b], "batch_id": batch_id,
                            "phase": "prefill", "layer": layer_idx, "tok": t, "topk": topi[row, :k].tolist(),
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
                        "model": self.model_tag, "req": req_ids[b], "domain": domains[b], "batch_id": batch_id,
                        "phase": "decode", "layer": layer_idx, "tok": tok0, "topk": topi[b, :k].tolist(),
                        "gate_w": [round(v, 4) for v in torch.softmax(topv[b], -1).tolist()][:k],
                        "router_entropy": round(ent, 4),
                    })
                    self.entropies.append(ent)
                    self.activ_per_layer[layer_idx] += 1
                    self.hook_fire_count[(req_ids[b], tok0)] += 1
        return hook

    def assert_sanity(self) -> dict:
        import torch

        bad = [key for key, c in self.hook_fire_count.items() if c != self.n_layers]
        assert not bad, f"hook fire-count mismatch on {len(bad)} tokens (expected {self.n_layers}/tok); first: {bad[:5]}"
        counts = list(self.activ_per_layer.values())
        assert len(set(counts)) == 1, f"activation count differs across layers: {self.activ_per_layer}"
        import statistics
        mean_ent = statistics.mean(self.entropies) if self.entropies else 0.0
        max_ent = torch.log(torch.tensor(float(self.n_experts))).item()
        assert mean_ent > 0.05 * max_ent, (
            f"router entropy suspiciously low (mean={mean_ent:.3f}, max={max_ent:.3f}) -- "
            "likely hooked the wrong module or logging argmax-of-garbage"
        )
        return {"n_layers": self.n_layers, "n_experts": self.n_experts, "mean_entropy": mean_ent,
                "max_entropy": max_ent, "activations_per_layer": counts[0] if counts else 0}


def run_request_single(model, tok, collector: Collector, req_id, domain, prompt, max_new, seed):
    import torch

    with torch.no_grad():
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


def run_batch_erosion(model, tok, collector: Collector, reqs, batch_size, max_new, seed):
    import torch

    with torch.no_grad():
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


def main(argv=None):
    from tiermoe.hw import probe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="olmoe", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--workloads", default="sharegpt,code")
    ap.add_argument("--n-sharegpt", type=int, default=150)
    ap.add_argument("--n-code", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/olmoe/b1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--quant", default="none", choices=["none", "nf4"])
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--cpu-offload-gb", type=int, default=32)
    ap.add_argument("--offload-folder", default="offload_cache")
    ap.add_argument("--mode", default="b1", choices=["b1", "b8"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args(argv)

    caps = probe()
    if not caps.can_collect_traces:
        print(
            "tiermoe collect: refusing to run -- this machine cannot collect real traces "
            f"(CUDA available: {caps.cuda_available}). Run this on a CUDA box, or use the pilot's "
            "already-collected traces under results/{olmoe,mixtral}/b1/ for every other command.",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tiermoe.collect.prompts import build_workload, save_prompt_list

    hf_id = MODEL_REGISTRY[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(hf_id)

    if args.quant == "nf4":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                                         llm_int8_enable_fp32_cpu_offload=True)
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory / 1e9 * args.gpu_mem_frac:.1f}GiB"
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = f"{args.cpu_offload_gb}GiB"
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=bnb_config, device_map=args.device,
                                                       max_memory=max_memory, offload_folder=args.offload_folder)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map=args.device)
    model.eval()

    out_dir = Path(args.out)
    writer = TraceWriter(out_dir / "traces.jsonl.zst")
    collector = Collector(model, tok, writer, model_tag=args.model)
    print(f"[collect] gate modules: {collector.n_layers} layers, k={collector.k}, n_experts={collector.n_experts}")

    all_prompts = []
    for w in args.workloads.split(","):
        n = args.n_sharegpt if w == "sharegpt" else args.n_code
        all_prompts.extend(build_workload(w, n, seed=args.seed))
    if args.smoke:
        all_prompts = all_prompts[: args.smoke]
    save_prompt_list(all_prompts, out_dir / "prompts_used.jsonl")

    t0 = time.time()
    if args.mode == "b1":
        for i, r in enumerate(all_prompts):
            run_request_single(model, tok, collector, r["req"], r["domain"], r["prompt"], args.max_new, seed=args.seed + i)
            if (i + 1) % 10 == 0 or i == len(all_prompts) - 1:
                print(f"[collect] {i+1}/{len(all_prompts)} requests, {time.time()-t0:.0f}s elapsed")
    else:
        run_batch_erosion(model, tok, collector, all_prompts, args.batch_size, args.max_new, seed=args.seed)

    writer.close()
    report = collector.assert_sanity()
    report["n_records_written"] = writer.total_written
    report["elapsed_sec"] = time.time() - t0
    (out_dir / "collect_report.json").write_text(json.dumps(report, indent=2))
    print(f"[collect] DONE. {report}")


if __name__ == "__main__":
    main()
