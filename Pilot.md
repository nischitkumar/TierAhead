# Pilot Experiment: The Expert Locality & Predictability Probe (ELP-Probe)

**Purpose:** produce, in a single ~6–10 hour cluster run **today**, the numbers that (a) decide whether the router-guided HBM+CXL tiering direction is worth pursuing, and (b) go verbatim into the Aug 6 abstract as preliminary evidence.

**One-sentence version of the experiment:** trace the router decisions of one or two open MoE models over a few hundred real requests, then compute three quantities — *how concentrated* expert traffic is (Coverage@C), *how stable* the hot set is over time (churn), and *how predictable* the next layer's experts are from the current layer (Recall@m) — and combine them into a first-order projection of how much CXL fetch traffic a tiered system could hide.

---

## 1. Why this experiment, and why designed this way (read this section first)

### 1.1 Decision-theoretic reasoning: measure the riskiest *unknown*, not the known

The whole proposal is a chain of four claims:

1. MoE parameter footprints exceed HBM → **known**, arithmetic (87 GB Mixtral vs 80 GB H100).
2. CXL adds a byte-addressable tier at ~+170–250 ns and tens of GB/s → **known**, measured on real hardware by Sun et al. (ISCA'23) and published in Astera's Leo materials. Re-measuring this is impossible for us (no CXL silicon) and unnecessary.
3. **Expert traffic is skewed and stable enough that a small fast tier captures most of it** (H1), and **the router at layer *l* predicts layer *l+1* well enough to prefetch the rest** (H2) → **unknown for our exact models/workloads.** Published evidence is encouraging but indirect: fMoE and MoE-Infinity exploit activation patterns successfully, but report system-level speedups, not the raw locality/predictability statistics we need, and mostly on different models/workloads than ours.
4. If (3) holds, placement + prefetch hides tier            \vspace{2.1mm} latency → follows by construction and gets *quantified* later by the simulator.

Claims 1–2 need citations, not experiments. Claim 4 is downstream engineering. **Claim 3 is the only falsifiable load-bearing unknown, and it is fully measurable without any CXL hardware, without a simulator, and without policy code — just model forward passes and counting.** That is exactly what a pre-abstract pilot should attack: maximum decision information per GPU-hour, zero dependence on infrastructure we haven't built yet.

### 1.2 Why traces, not a latency benchmark

A tempting alternative pilot is "measure PCIe/NUMA transfer latency and show it's big." Rejected for three reasons: (i) it re-derives numbers already published with far better hardware than ours — zero novelty, and judges reading the abstract will know it; (ii) it produces a *problem* statement, not evidence *for our solution*; (iii) locality/predictability numbers are model-intrinsic — they will still be true on real Astera hardware, so they survive every later methodology change. Trace statistics are the most durable numbers we can put in an abstract.

### 1.3 Why these specific three metrics

They map one-to-one onto the three mechanisms of the proposed system:

| Metric | Mechanism it validates | Failure meaning |
| --- | --- | --- |
| Coverage@C (skew) | **Static placement** — top-C experts pinned in HBM absorb Coverage@C of traffic | If flat: placement useless, everything rides on prefetch |
| Hot-set churn (stability) | **Epoch migration** — a slowly-drifting hot set means cheap re-placement keeps working | If churny: static placement decays; dynamic policy required |
| Recall@m, 1-layer lookahead (predictability) | **Router-guided prefetch** — fetches for layer *l+1* issued during layer *l* compute | If ~random: prefetch degenerates to LRU caching; contribution shrinks |
| (Derived) Prefetch-coverage / hideable-traffic bound | Whole-system sanity: what fraction of non-resident expert bytes could be requested at least one layer early | The single number for the abstract |

Note the metrics are *policy-free*: we measure properties of the workload, not of our (not-yet-built) system. This is deliberate — a pilot that requires the system to exist isn't a pilot.

### 1.4 Why one-layer lookahead specifically

The prefetch window in a real system is one layer's compute time (attention + experts at layer *l*) — roughly 0.5–3 ms depending on model/hardware — versus a CXL fetch of a Mixtral-sized expert (~340 MB) needing ~5–11 ms at 32–64 GB/s. So one layer of lookahead hides *part* of a coarse-expert fetch and *all* of a fine-grained-expert fetch (OLMoE experts are ~7 MB). We therefore also compute Recall@m at lookahead depths 2–4 (deeper lookahead = longer overlap window = weaker prediction; the trade-off curve is itself an interesting preliminary figure). But depth-1 is the headline because it needs no chained prediction.

### 1.5 Why batch size 1 (and why we also record batch 8)

Per-token expert skew is diluted under batching: the *union* of experts selected across a batch approaches all experts as B grows (with 8 experts/layer top-2, B=8 nearly saturates; with 64 experts top-8, saturation is much slower). B=1 measures the intrinsic per-request structure that latency-sensitive and memory-constrained serving (the CXL-relevant regime) actually experiences; B=8 gives us the honest erosion number so the abstract cannot be accused of cherry-picking, and fine-grained models are expected to *keep* meaningful skew at B=8 — if confirmed, that's a bonus claim ("fine-grained MoEs are the natural CXL workload").

### 1.6 Why these models

- **OLMoE-1B-7B** (64 experts/layer, top-8, ~7 MB/expert FP16): fine-grained routing regime, fits any 24 GB GPU in FP16, fast — guarantees the pilot completes today even if nothing else works.
- **Mixtral-8x7B, 4-bit NF4** (8 experts/layer, top-2, ~340 MB/expert FP16): the canonical coarse-expert model where each fetch is huge — the strongest CXL narrative. 4-bit fits ~26 GB (one A100-40G / L40S / 2×T4). Routing decisions under NF4 are known to track FP16 closely; we spot-check agreement on 20 prompts anyway (§4.4) so the abstract number is defensible.
- One fine-grained + one coarse model spans the design space; two models make "we observed X" a pattern, not an anecdote. If cluster time is tight, **OLMoE alone is sufficient for the abstract** — run it first, Mixtral second.

### 1.7 Why a train/test split for the predictor (subtle but critical)

Recall@m computed on the same requests used to estimate the transition statistics is leakage and would inflate the abstract's central number. Split *by request* (not by token): fit transition matrices on 70% of requests, evaluate recall on the held-out 30%. Also evaluate cross-domain (fit on chat, test on code) — if recall transfers across domains, the predictor is learning model structure, not dataset quirks, which is a stronger claim and one sentence in the abstract.

### 1.8 What this experiment is *not*

Not a speedup measurement, not a simulator run, not a CXL emulation. Anyone reviewing the abstract who asks "did you actually make anything faster yet?" gets the honest answer: "No — the pilot establishes that the workload has the structure our system needs; the system and its evaluation are the project." That is the correct epistemic posture for an abstract and reads as rigor, not weakness.

---

## 2. Formal experiment specification

### 2.1 Factors and levels

| Factor | Levels | Notes |
| --- | --- | --- |
| Model | OLMoE-1B-7B (FP16); Mixtral-8x7B (NF4) | priority order |
| Workload | ShareGPT chat (150 req); HumanEval+MBPP code prompts (100 req) | ≥100 req/cell for stable tails |
| Phase | prefill, decode (tagged per token) | analyzed separately; decode is the money regime |
| Batch | 1 (primary); 8 (erosion check, OLMoE only) | §1.5 |
| Decode length | up to 256 new tokens, temperature 0.7, seed fixed | enough tokens for windowed churn stats |

Total forward-pass volume: ~250 requests × ≤256 tokens ≈ ≤64k decode steps per model — hours, not days.

### 2.2 Logged record (per token, per layer)

```json
{"model":"olmoe-1b-7b","req":"sharegpt-0031","domain":"chat","phase":"decode",
 "layer":9,"tok":57,"topk":[3,41,7,22,60,11,2,48],"gate_w":[...8 floats...],
 "router_entropy":2.41}
```

Logits beyond top-k are unnecessary for the pilot (keep top-2k for margin analysis); dropping them keeps traces small (<1 GB total, zstd).

### 2.3 Metrics — exact definitions

Let A = multiset of (layer, expert) activations in a phase; per layer ℓ, f_{ℓ,e} = activation count of expert e.

1. **Coverage@C** (per layer, then activation-weighted mean over layers): sort experts by f_{ℓ,e} descending on the *training* split; Coverage@C = Σ_{top C%} f / Σ f evaluated on the *test* split (an honest "deployable" skew number — using test-split counts on their own ranking would be oracle skew; report both, labeled).
   Report C ∈ {12.5, 25, 37.5, 50}% — these are candidate HBM-resident fractions.
2. **Gini coefficient / Zipf exponent** of {f_{ℓ,e}} — one-number skew summaries for the abstract.
3. **Hot-set churn:** windows of W=2,000 decode activations; hot set H_w = smallest expert set covering 80% of window activations; churn = 1 − |H_w ∩ H_{w+1}| / |H_w ∪ H_{w+1}| (Jaccard distance), report mean ± p95. Low churn (≲0.2) ⇒ epoch migration at coarse granularity suffices.
4. **Recall@m, lookahead d:** predictor = empirical conditional frequency table P(e′ selected at ℓ+d | S_ℓ) with additive smoothing, fit on train split; at test time rank experts of layer ℓ+d by Σ_{e∈S_ℓ} log P̂; Recall@m = |predicted-top-m ∩ actual top-k set| / k. Report m ∈ {k, 2k, 4k}, d ∈ {1,2,4}. Baselines that MUST appear next to it: (a) static-popularity top-m (no conditioning), (b) uniform random. The *gap over baseline (a)* is the evidence that the router signal adds information beyond global skew.
5. **Derived headline — Hideable-traffic bound (HTB):** assume HBM pins the training-split top-C experts; residual (non-resident) activations either were in the predictor's top-m for d=1 (⇒ prefetchable, latency overlappable) or not (⇒ demand stall).
   HTB(C,m) = Coverage@C + (1 − Coverage@C) · Recall@m|non-resident.
   Interpretation: fraction of expert activations that a tiered system never demand-stalls on, *before any simulator sophistication*. This is the abstract's central number. Compute Recall conditioned on non-resident experts specifically (prediction is harder exactly on the cold tail — do not let hot experts inflate it).
6. **Batch-8 erosion:** distinct experts per layer per batch-step / total experts, vs the B=1 value.

### 2.4 Pre-registered decision thresholds (write these down before running)

| Outcome | Condition (decode, B=1, test split) | Action |
| --- | --- | --- |
| **Strong go** | Coverage@25% ≥ 60% AND Recall@2k,d=1 ≥ 60% on non-resident ⇒ HTB ≥ ~0.85 | Abstract leads with HTB; full plan unchanged |
| **Go (placement-led)** | Coverage@25% ≥ 60% but Recall < 60% | Direction stands; prefetch demoted to "explored", placement + migration promoted; abstract cites Coverage + churn |
| **Go (prefetch-led)** | Coverage flat (<40%) but Recall@2k ≥ 60% | Rare but possible on balanced-routed models; story becomes pure predictive prefetch (closer to MoE-Infinity-on-CXL) |
| **Pivot** | Both < thresholds on both models | Workload has no exploitable structure at expert granularity → pivot the same tiering framework to **KV-cache spillover** (Astera's own published use case; Phase 1 machinery reusable ~80%) |

Thresholds are not arbitrary: Coverage@25% ≥ 60% means a 4× capacity reduction still serves a strong majority of traffic from HBM; Recall ≥ 60% at d=1 is the regime where prefetch demonstrably beat LRU in fMoE-class systems (which reported +36% hit-rate gains from far weaker signals than direct router conditioning).

---

## 3. Implementation (copy-paste starting points)

### 3.1 Environment

```bash
# cluster login node
python -m venv ~/elp && source ~/elp/bin/activate
pip install "torch>=2.3" transformers accelerate datasets zstandard pandas numpy \
            matplotlib bitsandbytes sentencepiece  # bitsandbytes only needed for Mixtral NF4
hf auth login   # OLMoE + Mixtral are gated-free but need auth for Mixtral
```

### 3.2 Trace collector (core ~60 lines; the whole pilot's code is <300 lines)

```python
import json, time, zstandard as zstd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"          # or "mistralai/Mixtral-8x7B-Instruct-v0.1"
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto",
    # for Mixtral add: load_in_4bit=True, bnb_4bit_quant_type="nf4"
)
model.eval()

records, ctx = [], {"req": None, "domain": None, "phase": "prefill", "tok": 0}

def hook_factory(layer_idx):
    def hook(module, inputs, output):
        # HF MoE gates return router logits; for OLMoE/Mixtral the gate module is
        # model.layers[i].mlp.gate (a Linear: hidden -> n_experts). Hook the Linear.
        logits = output.detach().float()                      # [tokens, n_experts]
        k = getattr(model.config, "num_experts_per_tok", 2)
        topv, topi = torch.topk(logits, k=max(2*k, 8), dim=-1)
        for t in range(logits.shape[0]):
            records.append({
                "req": ctx["req"], "domain": ctx["domain"], "phase": ctx["phase"],
                "layer": layer_idx, "tok": ctx["tok"] + t,
                "topk": topi[t, :k].tolist(),
                "logits_topm": list(zip(topi[t].tolist(), [round(v,3) for v in topv[t].tolist()])),
            })
    return hook

for i, layer in enumerate(model.model.layers):
    gate = layer.mlp.gate            # VERIFY module path with print(model) first —
    gate.register_forward_hook(hook_factory(i))   # names differ slightly across models

@torch.no_grad()
def run_request(req_id, domain, prompt, max_new=256):
    ctx.update(req=req_id, domain=domain, phase="prefill", tok=0)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    ctx["phase"] = "prefill"
    out = model(**ids, use_cache=True)            # prefill pass (hooks fire once for all tokens)
    ctx.update(phase="decode", tok=ids.input_ids.shape[1])
    past, next_id = out.past_key_values, out.logits[:, -1:].argmax(-1)
    for _ in range(max_new):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = torch.multinomial(torch.softmax(out.logits[:, -1] / 0.7, -1), 1)
        ctx["tok"] += 1
        if next_id.item() == tok.eos_token_id: break
```

Design notes (the *why* of the code): manual decode loop instead of `model.generate` so the prefill/decode phase tag and token counter stay exact; hooking the gate **Linear** (not the MoE block) gets raw logits before top-k dispatch, which the predictor needs; storing top-2k logits (not full vocab of experts) keeps traces tiny while enabling margin/entropy analysis later.

Prompts: `datasets.load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered")` — take first human turn of 150 conversations, dedup, length 50–500 chars; `openai_humaneval` + `mbpp` for 100 code prompts. Fixed seed, log the prompt list into the results directory (reproducibility line for the report).

### 3.3 Analysis (deterministic, runs on your Mac in minutes on the downloaded traces)

Order of operations, each ~20 lines of pandas:

1. Load JSONL.zst → DataFrame; split requests 70/30 train/test, stratified by domain.
2. Coverage@C per layer on train-ranking/test-counts; activation-weighted aggregate; plot CDF (this figure goes in the final report unchanged).
3. Gini + Zipf fit (linregress on log-rank/log-freq).
4. Windowed churn (W=2,000, 80% mass hot sets, Jaccard).
5. Transition tables: for d in {1,2,4}: joint counts of (S_ℓ → e′ at ℓ+d) with +1 smoothing → Recall@{k,2k,4k} on test split, overall AND conditioned on non-resident experts, next to static-popularity and random baselines.
6. HTB(C=25%, m=2k) with a small table over C×m.
7. Batch-8 rerun of coverage (OLMoE only, same prompts packed).
8. Emit `pilot_summary.json` — every abstract-bound number in one machine-readable file.

### 3.4 Sanity checks before believing any number

- Hook fires exactly n_layers times per token (assert count).
- Σ activations per layer = tokens × k.
- Router entropy distribution non-degenerate (a bug that logs argmax-of-garbage shows up as near-zero entropy).
- NF4 fidelity spot-check (Mixtral only): 20 prompts run in NF4 and FP16 (FP16 via CPU offload, slow but fine for 20), top-2 agreement rate per token — report it; literature suggests ≥95%+; if lower, caveat in abstract footnote.
- One fully-deterministic rerun of 5 requests → identical traces (seed discipline).

---

## 4. Cluster execution plan (today)

### 4.1 Resource request

| Job | GPU | Wall time | Est. actual |
| --- | --- | --- | --- |
| OLMoE traces (250 req, B=1) | 1× ≥24 GB (A5000/L40S/A100 — anything) | 4 h | ~1.5–2.5 h |
| OLMoE batch-8 erosion | same allocation | +30 min | ~20 min |
| Mixtral NF4 traces (250 req) | 1× 40 GB (A100-40G/L40S 48G) or 2× 24 GB | 6 h | ~3–4 h |
| NF4 fidelity check | same | +1 h | ~40 min |

SLURM sketch:

```bash
#!/bin/bash
#SBATCH -J elp-olmoe  -p gpu  --gres=gpu:1  --mem=48G  -c 8  -t 04:00:00
source ~/elp/bin/activate
python collect.py --model olmoe --workloads sharegpt,code --n 250 --max-new 256 \
                  --out /scratch/$USER/elp/olmoe/
```

Storage: <2 GB total. Copy `*.jsonl.zst` to your Mac; run analysis locally (keeps the cluster job dumb and restart-safe — collection and analysis decoupled on purpose: if analysis has a bug at 11 pm, you fix it without re-queuing).

### 4.2 Timeline for an Aug 6 abstract (T = now)

- T+0–1 h: env setup, `print(model)` to verify gate module paths, smoke test 3 prompts.
- T+1 h: submit OLMoE job. While it runs: write the analysis notebook against the smoke traces.
- T+3–4 h: OLMoE traces done → run analysis → **first look at Coverage/Recall/HTB; the go/no-go decision happens here.**
- T+4 h: submit Mixtral job (nice-to-have for the abstract; abstract does not block on it).
- T+5–6 h: insert numbers into abstract (§5 templates), submit abstract.
- Overnight: Mixtral finishes; numbers go into the Aug 12 shortlisting-stage material and the final report regardless.

**Minimal viable pilot if everything goes wrong:** OLMoE, ShareGPT only, 100 requests, Coverage + Recall(d=1) only — ~2 h end-to-end, still yields an HTB number.

---

## 5. Turning the numbers into abstract sentences (templates)

Strong-go template:
> *Preliminary characterization on OLMoE-1B-7B and Mixtral-8x7B over 500 real chat and code requests shows that the top 25% of experts serve **{X}%** of decode-time activations, and that a lightweight cross-layer statistical predictor recalls **{Y}%** of the next layer's experts one layer ahead — together bounding **{Z}%** of expert traffic as servable from a small HBM-resident set or prefetchable from CXL before use. These measurements motivate router-guided tiering as [...]*

Placement-led template (weak recall):
> *...the top 25% of experts serve {X}% of activations with a hot set whose windowed churn is only {J} — indicating that a small, slowly-migrating HBM-resident expert set can absorb the majority of traffic, with CXL providing capacity for the cold tail.*

Pivot template (both weak — still a publishable observation):
> *...we find expert traffic in modern fine-grained MoEs to be substantially balanced (Coverage@25% = {X}%), implying capacity-tier value lies not in expert placement but in KV-cache spillover; we therefore target [...]* — note this failure mode still yields an abstract, which is precisely why the pilot is safe to run the day before the deadline.

Rules: every number in the abstract must come from `pilot_summary.json` (no rounding up, no B=1 numbers presented without saying B=1); name the models and request counts; use "bound"/"suggests", not "achieves" — no system exists yet.

---

## 6. Threats to validity (pre-empt the shortlisting reviewers)

| Objection | Our answer (baked into the design) |
| --- | --- |
| "Skew is a dataset artifact" | Two domains + cross-domain predictor transfer test (§1.7); Zipf fit reported per domain |
| "Batching kills this" | Batch-8 erosion measured and reported, not hidden; framing: low-batch/latency-sensitive + fine-grained MoEs are the CXL regime |
| "4-bit changes routing" | Fidelity spot-check with reported agreement rate |
| "Predictor leakage" | Request-level train/test split, stated explicitly |
| "Load-balancing losses make experts uniform *by training design*" | True at the *global* average — which is exactly why we measure *conditional/temporal* structure (windowed hot sets, transitions), where balance regularizers do not operate; this distinction is worth one sentence in the report |
| "n=250 requests" | Bootstrap CIs on Coverage and Recall (percentile, 1,000 resamples) — cheap, do it |

## 7. If cluster access falls through today

OLMoE-1B-7B FP16 runs on a 24 GB+ Apple-Silicon Mac via MPS (slow decode, ~irrelevant for tracing) — cut to 100 requests, ShareGPT only; or Kaggle 2×T4 free tier runs the full OLMoE grid in one session. The abstract ships either way. Mixtral then lands during Aug 7–12 on Kaggle and enters the shortlisting material.

## 8. What happens to this work afterwards (nothing is throwaway)

The collector *is* Phase-1 component [1] of tierMoE; the transition tables *are* the `next-layer-topk` policy's input; the analysis figures *are* the report's characterization section; `pilot_summary.json` seeds the simulator's workload priors. The pilot is not a detour — it is Phase 1 started early, scoped to the decision-critical subset.
