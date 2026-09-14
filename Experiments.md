# tierMoE Experiment Roadmap (post-pilot, v2)

**Status:** ELP-Probe complete → verdict **GO, prefetch-led**. This document is the full experiment program from Aug 14 → Sept 25, with per-experiment rationale, method, deliverables, exit criteria, effort, and learning resources.

Read §0 first. It contains a finding that reorders the entire project.

---

## §0. The reframing: prefetch hides latency, it does not create bandwidth

The pilot answered "is expert traffic predictable?" (yes, ~55–62% recall at m=2k, d=1, beating the popularity baseline by 14–25 pp). But a back-of-envelope bandwidth calculation exposes a second, harder question the pilot did not ask, and it determines whether the recall number is even useful.

**Expert bytes moved per decode token, if non-resident (FP16):**

| Model | Expert size | Bytes/layer (top-k) | Bytes/token (all layers) | Time/layer @32 GB/s | Time/token @32 GB/s |
|---|---|---|---|---|---|
| Mixtral-8x7B (8 exp, top-2, 32 L) | 352 MB | 705 MB | 22.5 GB | 22.0 ms | 705 ms |
| Mixtral, 50% resident | — | 352 MB | 11.3 GB | 11.0 ms | 352 ms |
| OLMoE-1B-7B (64 exp, top-8, 16 L) | 12.6 MB | 101 MB | 1.61 GB | 3.15 ms | 50 ms |
| OLMoE, 50% resident | — | 50 MB | 0.81 GB | 1.57 ms | 25 ms |

Compare the per-layer transfer time against per-layer *compute* time (batch 1 decode: tens to low hundreds of µs for these models). **Transfer time exceeds compute time by one to two orders of magnitude in every cell.** That is the definition of a bandwidth-bound regime, and in a bandwidth-bound regime prefetching is worthless: you cannot overlap 22 ms of transfer behind 0.2 ms of compute no matter how accurate the predictor is. Prefetch converts *stalls* into *overlap*; it does not reduce bytes moved.

This does not kill the project. It sharpens it into something far more defensible and far more interesting to Astera:

> **The thesis (v2):** whether CXL is a viable capacity tier for MoE inference is determined by *arithmetic intensity per expert byte*, which is set by expert granularity and batch size — not by prediction accuracy. We map that regime boundary quantitatively (the **MoE Tiering Roofline**), show that fine-grained MoEs at moderate batch land on the viable side of it, and build the two mechanisms that exploit that regime: **router-guided prefetch** (hides latency once you are inside the compute-bound side) and **confidence-gated precision fallback** (reduces bytes when you are not).

Three reasons this framing wins the competition:

1. **It explains the pilot's contradictory results instead of hiding them.** OLMoE has skew but weaker recall; Mixtral has no skew but stronger recall. The roofline resolves it: each model anchors one side of the regime boundary. Every "confusing" number becomes a data point on a curve.
2. **It is an architecture-recommendation result, which is exactly what a memory-controller company wants.** The output is "for model class X at batch B, provision Y GB/s per accelerator and Z% HBM residency" — a provisioning rule for Leo-class devices, not a student benchmark.
3. **Batching is the hidden lever, and it cuts both ways** — more batch = more compute per byte (helps hiding) but wider expert union (hurts locality). There is an optimum. Finding it is a real, publishable contribution, and no prior MoE-offloading paper frames it as a roofline.

Everything below is organized around establishing, then exploiting, that boundary.

---

## §0.1 Experiment dependency graph

```
 E1 Roofline model ──┬──► E2 Batching sweet spot ──┐
        ▲            │                             │
 E3 Calibration ─────┘                             ├──► E6 Simulator + policy bake-off ──► E9 Pooling/TCO
        │                                          │              │
        └──► E4 Predictor + confidence ──► E5 Lookahead depth ─────┘              └──► E12 Ablations
                        │                                          │
                        └──────────► E7 Precision-aware prefetch ◄─┘
                                                                   │
 E10 Hardware validation (PCIe / QEMU / NUMA) ─────────────────────┤
 E11 Model generalization (DeepSeek, Qwen, GPT-OSS) ───────────────┘
                                                                   ▼
                                                    E13 Dashboard + report + demo
```

Critical path: **E1 → E3 → E6 → E13**. Everything else is upside; cut in reverse order of the priority tags below.

Priority tags: **[P0]** = project fails without it · **[P1]** = needed for first place · **[P2]** = strong differentiator, cut if time-pressed · **[P3]** = stretch/paper-only.

---

# E1 — The MoE Tiering Roofline **[P0]**

**Why this experiment, and why first.** Every downstream design decision (how much to prefetch, whether precision fallback is needed, which model class to target, what link width to recommend) depends on knowing which side of the compute/bandwidth boundary a configuration sits on. It is also cheap: it is an analytical model plus trace-derived byte counts, no GPU required. Running it first prevents you from spending three weeks tuning a predictor for a regime where prediction cannot help.

**Method.**
1. Define, per (model, batch B, residency r, precision p, link BW): 
   `bytes_per_layer(B,r,p) = E_union(B) · size_expert(p) · (1 − r)` where `E_union(B)` is the measured expected number of *distinct* experts activated per layer across a batch of B tokens (from your existing traces — you already have this machinery from the batch-8 erosion check).
2. `t_transfer = bytes_per_layer / BW_eff`; `t_compute = ` measured per-layer compute at batch B (from E3; use a FLOP-based estimate until E3 lands).
3. **Regime ratio** ρ = t_transfer / t_compute. ρ < 1 ⇒ latency-bound, prefetch can fully hide; ρ ≈ 1 ⇒ the interesting frontier; ρ ≫ 1 ⇒ bandwidth-bound, only byte reduction helps.
4. Sweep and plot ρ = 1 contours in the (B, r) plane, one panel per model, one line per link config (CXL 2.0 ×8 ≈ 32 GB/s, ×16 ≈ 64 GB/s, CXL 3.x / multi-link ≈ 128 GB/s), for FP16 and NF4.
5. Overlay the *achievable* operating points given measured Coverage@C from the pilot — i.e., which points are reachable with a real residency policy rather than an assumed one.

**Deliverables.**
- `tiermoe/roofline/` module + `figures/roofline_{model}.pdf` — **the report's Figure 1**.
- A one-page "provisioning table": for each model class, minimum GB/s per accelerator and minimum residency % to keep ρ<1 at batch {1,8,32}.
- `roofline_summary.json` consumed by E6 and the dashboard.

**Exit criteria.** You can state, in one sentence with numbers, the condition under which router-guided prefetch is worth anything. Expected shape of the answer: *"fine-grained MoE (expert ≤ ~30 MB) at batch ≥ 8 with ≥50% residency on a ≥64 GB/s link"*.

**Effort.** 8–12 h. No GPU.

**Resources.**
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model*, CACM 2009 — the original model you are adapting. Read §2–3.
- Horace He, *Making Deep Learning Go Brrrr From First Principles* (blog, horace.io) — the clearest treatment of memory-bound vs compute-bound for transformer inference; read before writing a line of E1 code.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), SOSP 2023 — for how decode-phase arithmetic intensity behaves with batch.
- Video: MIT 6.5940 *EfficientML.ai* (Song Han), lectures on LLM inference/serving — search "MIT 6.5940 efficient AI lecture LLM serving"; the roofline/arithmetic-intensity treatment there maps directly onto this experiment.
- Video: CXL Consortium webinar series on YouTube ("Introduction to Compute Express Link", CXL 2.0/3.x sessions) — for the link-bandwidth numbers you will cite.

---

# E2 — Batching sweet spot: locality erosion vs arithmetic intensity **[P1]**

**Why.** E1 says batching helps hiding (more compute per byte). The pilot's batch-8 erosion check says batching hurts locality (expert union widens: OLMoE went 12.5% → 47% of experts touched per step). These oppose each other, so there is an optimum, and its position depends on expert count — the same architectural parameter that drove the pilot's skew inversion. Establishing this curve turns two awkward pilot observations into one clean law.

**Method.**
1. From traces, compute `E_union(B)` for B ∈ {1,2,4,8,16,32,64} by resampling tokens into synthetic batches (you do not need to re-run the models — sample B token-steps from distinct requests, take the union of selected experts per layer). Do this per model.
2. Compute effective **bytes per token** = `E_union(B) · size_expert · (1−r) / B` — the key quantity, because batching amortizes a fetched expert across all tokens in the batch that need it.
3. Combine with the E1 compute model → ρ(B). Find B* minimizing time-per-token subject to a p99 TPOT SLO.
4. Repeat with the *predictor* in the loop: recall degrades when you must predict a union of experts rather than top-k for one token — measure Recall@m for batched targets (this is a genuinely new metric; no prior work reports it).

**Deliverables.** `figures/batch_sweetspot_{model}.pdf` (bytes/token and ρ vs B, with B* marked); a table of B* per model per link; a short subsection in the report titled "Batching is a memory-tiering knob, not just a throughput knob."

**Exit criteria.** B* identified for both models with CIs; the claim "fine-grained MoEs have a wide viable batch window, coarse MoEs have none" is either confirmed with numbers or refuted.

**Effort.** 8 h, trace analysis only, no GPU.

**Resources.**
- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024 — chunked prefill / batching-vs-latency framing you should cite.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — continuous batching, the mechanism your B* assumes.
- Rajbhandari et al., *DeepSpeed-MoE*, ICML 2022 — expert-parallel batching behavior.
- Video: Stanford CS336 *Language Modeling from Scratch* MoE lecture (2025) — search "CS336 mixture of experts lecture"; good grounding in routing/load-balancing loss, which is the mechanism behind your skew-inversion finding.

---

# E3 — Hardware calibration ladder **[P0]**

**Why.** Every simulated number in the final report must trace back to a measurement or a citation. This experiment produces the measured half. It is also your defense against the obvious judge question: "you don't have CXL hardware, so why should I believe any of this?"

**Method (one GPU session, ~4 h).**
1. **Per-layer / per-expert compute**: CUDA events around expert FFN GEMMs at batch {1,8,32} for both models; report mean ± p95. On the RTX 4500 Ada for OLMoE, the 4060 Ti box for Mixtral NF4.
2. **Host→device transfer**: pinned-memory H2D bandwidth and fixed-overhead latency vs transfer size (1 MB → 512 MB), fitted as `t = α + size/β`. This gives you a *measured* two-parameter transfer model — the same functional form you apply to CXL with literature-derived α, β.
3. **End-to-end offloading ground truth**: a minimal HF runner that keeps a fraction r of experts on GPU and streams the rest from pinned host memory over PCIe, with policies `none` / `static-hot` / `next-layer-topk`. Measure real TPOT. This is not CXL, but it is the same physical layer and the same policy question — and it validates the *ordering* your simulator will later predict.
4. **NF4 fidelity spot-check (open item #2 from the pilot)**: you cannot fit FP16 Mixtral, but you *can* run FP16 vs NF4 on a small MoE (OLMoE fits in FP16 on the 4500 Ada; quantize it to NF4 and measure top-k routing agreement). Report agreement on OLMoE and cite literature for Mixtral. This closes the caveat honestly rather than leaving it open.

**Deliverables.** `calib/{compute,transfer,e2e}.json`; `figures/transfer_model_fit.pdf`; a methodology table in the report listing every simulator parameter and whether it is *measured here*, *measured in cited work*, or *vendor-specified*. That table alone will out-rigor every other team.

**Exit criteria.** Simulator parameter file fully populated with provenance for each entry; PCIe policy ordering measured (expect `next-layer-topk` ≥ `static-hot` > `none`).

**Effort.** 10 h including debugging.

**Resources.**
- Sun et al., *Demystifying CXL Memory with True CXL-Ready Systems*, ISCA 2023 — the α/β numbers you will import for CXL.
- Ji et al., *Demystifying a CXL Type-2 Device*, MICRO 2024.
- NVIDIA developer blog on CUDA events + pinned memory ("How to Implement Performance Metrics in CUDA C/C++", "How to Optimize Data Transfers in CUDA C/C++") — exact methodology for step 2.
- Eliseev & Mazur, arXiv:2312.17238 — their offloading runner is the reference implementation for step 3.
- Dettmers et al., *QLoRA* (NF4), arXiv:2305.14314 — for the quantization fidelity discussion.

---

# E4 — Predictor upgrade and confidence calibration **[P1]**

**Why.** The pilot's frequency-table predictor gets 55–62% recall at m=2k. Two things must improve for the system to work: (a) higher recall at *lower* m (because m is a bandwidth multiplier — at m=2k you fetch twice the bytes you need, which is exactly what the roofline says you cannot afford), and (b) **calibrated confidence**, because E7's precision-gating decides *how many bytes* to spend on each candidate based on how sure the predictor is. Confidence is not optional garnish; it is the input to the mechanism that saves the coarse-grained regime.

**Method.**
1. **Predictor v2**: small MLP taking the layer-ℓ gate input hidden state (or the top-2k logit vector, which you already log) → layer ℓ+1 expert scores. ~0.5–2 M params, trains in minutes on CPU/MPS. Train per model on the 70% split; evaluate on the held-out 30% exactly as the pilot did.
2. **Ablate the input**: (a) selected expert IDs only [= pilot baseline], (b) + gate weights, (c) + router entropy, (d) + hidden state. Report recall@m for each — this tells you *where the predictive signal lives*, which is a genuine scientific result and cheap to obtain.
3. **Recall–bandwidth curve**: sweep m from k to 4k (cap at n_experts — the pilot correctly flagged that m=4k is degenerate for Mixtral, so cap and note it). Plot recall vs *bytes fetched per layer*. The right operating point is chosen on this curve by E1's budget, not by maximizing recall.
4. **Calibration**: reliability diagram + Expected Calibration Error for the predictor's per-expert probability. Apply temperature scaling if ECE is poor.
5. **Cross-domain transfer**: fit on chat, test on code, and vice versa (the pilot planned this; make sure it lands). Transfer ⇒ the predictor learns model structure, not dataset quirks.

**Deliverables.** `tiermoe/policy/predictor_v2.py`; `figures/recall_vs_bytes.pdf` (the operating-point figure); `figures/reliability.pdf`; ablation table; `predictor_report.json`.

**Exit criteria.** Either recall@m=k improved ≥8 pp over the frequency table (⇒ MLP earns its place), or it did not (⇒ report the negative result, keep the frequency table, and note that the router's *selected set* already carries nearly all the available signal — also a clean finding).

**Effort.** 12–15 h.

**Resources.**
- Hwang et al., *Pre-gated MoE*, ISCA 2024, arXiv:2308.12066 — the "predict next layer's experts" mechanism, done via model modification; you are doing the training-free version, so you must contrast explicitly.
- Du et al., *SiDA-MoE*, MLSys 2024, arXiv:2310.18859 — hash/predictor-based expert prediction; closest methodological relative of your predictor.
- Xue et al., *MoE-Infinity*, arXiv:2401.14361 — Expert Activation Matrix tracing.
- Yu et al., *fMoE*, arXiv:2502.05370 — iteration-level expert maps; your recall numbers should be discussed against their +36% hit-rate claim.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 — temperature scaling, reliability diagrams, ECE.
- Video: Umar Jamil, "Mixtral / Mixture of Experts explained" (YouTube) — fastest way to get a teammate up to speed on router internals before they touch this code.

---

# E5 — Lookahead depth and pipelined prefetch **[P1]**

**Why.** One layer of lookahead buys you one layer of compute time to hide a fetch. E1 shows that for coarse experts that window is 50–100× too small. The only way to extend it is deeper lookahead: predict layer ℓ+d and start the fetch d layers early. But prediction accuracy decays with d, and chained prediction compounds error. The depth-vs-accuracy curve tells you the maximum prefetch horizon, and therefore the maximum expert size the system can serve.

**Method.**
1. Direct predictors for d ∈ {1,2,4,8} (fit separate conditional tables/MLPs per d — do *not* chain, chaining compounds error; measure chained as a comparison).
2. Recall@m vs d, per model, on non-resident experts (matching the pilot's definition so numbers are comparable).
3. Convert to a **feasible-fetch-size curve**: given recall ≥ threshold at depth d, prefetch window = d × t_layer ⇒ max bytes fetchable at BW. Overlay actual expert sizes for the four target models. This figure directly answers "which MoE architectures can CXL serve?"
4. **Pipelining/queue depth**: simulate (in E6) a prefetch queue holding up to Q outstanding requests; measure queue occupancy and link utilization. Over-deep prefetch causes bandwidth contention with demand fetches — quantify the crossover.

**Deliverables.** `figures/recall_vs_depth.pdf`; `figures/feasible_fetch_size.pdf`; the prefetch-horizon subsection of the report.

**Exit criteria.** A stated maximum useful depth per model, plus the resulting maximum expert size. Expected: OLMoE-class fully served at d=1–2; Mixtral-class requires d≥8 *and* byte reduction (motivating E7).

**Effort.** 10 h (reuses E4 machinery).

**Resources.**
- Same predictor stack as E4, plus: Tang et al., *HOBBIT*, arXiv:2411.01433 (mixed-precision fallback under miss — read now, it is the direct ancestor of E7).
- Klotski, arXiv:2502.06888 — pipeline-bubble scheduling for MoE offload; the queue-depth reasoning is analogous.
- Patterson & Hennessy, *Computer Architecture: A Quantitative Approach*, ch. on memory hierarchy prefetching — for correctly framing prefetch accuracy/coverage/timeliness (use those three standard terms in the report; judges from a hardware company will notice).

---

# E6 — Simulator and policy bake-off **[P0]**

**Why.** This produces the headline result. Everything before it is inputs; everything after it is interpretation.

**Method.** Discrete-event simulator (Python + SimPy or hand-rolled), driven by the real traces, parameterized by E3's measured constants and literature CXL constants, with an explicit **shared-bandwidth link model** (demand fetches and prefetches contend; this is what makes the simulation non-trivial and what most teams will omit).

Policies to compare, all on identical traces:

| Policy | What it isolates |
|---|---|
| `none` (demand fetch) | lower bound |
| `lru` | classic caching baseline (Eliseev & Mazur) |
| `static-C` (fixed residency, pilot's Coverage@C) | placement-only — deliberately *not* an optimizer (pilot showed the optimizer isn't worth it) |
| `prefetch-freq` (pilot predictor, d=1) | the pilot's mechanism |
| `prefetch-v2` (E4 predictor, E5 depth) | the contribution |
| `prefetch-v2 + precision-gate` (E7) | the full system |
| `oracle` | upper bound; report "% of oracle gap closed" |

Sweep: residency r ∈ {0, 12.5, 25, 37.5, 50, 75}%, BW ∈ {16, 32, 64, 128} GB/s, added latency ∈ {150, 250, 400} ns, batch ∈ {1, 8, 32}, both models.

Metrics: TPOT mean/p50/p99, demand-stall rate (the pilot's headline metric — keep it as the through-line), bytes/token, link utilization, prefetch accuracy/coverage/timeliness, expert hit rate.

**Validation gates (run these as unit tests):** oracle ≥ every policy; zero-latency+infinite-BW CXL ⇒ parity with HBM-only; monotone improvement in r; simulated PCIe configuration reproduces E3's *measured* ordering within ±15% on relative deltas.

**Deliverables.** `tiermoe/sim/`; `results/sweep.parquet`; **`figures/headline_stall_vs_residency.pdf`**; `figures/policy_bakeoff.pdf`; validation table comparing simulated vs measured PCIe runs.

**Exit criteria.** A defensible statement of the form: *"at residency r on a BW GB/s link at batch B, router-guided prefetch reduces demand stalls from X% to Y% and TPOT by Z%, closing W% of the oracle gap."*

**Effort.** 30–35 h. The single largest engineering item; start it in parallel with E4.

**Resources.**
- Li et al., *DRAMSim3*, IEEE CAL 2020 + repo docs (umd-memsys/DRAMsim3) — drive it with your fetch stream to derive effective BW under realistic access patterns.
- Maruf et al., *TPP*, ASPLOS 2023; Lee et al., *Memtis*, SOSP 2023 — the reactive page-tiering policies your semantic policy is beating; implement a page-tiering-like baseline if time permits (very strong ablation).
- Li et al., *Pond*, ASPLOS 2023 — methodology for emulation-backed CXL evaluation.
- SimPy docs (discrete-event modeling in Python) — 1 h read, saves 10 h of event-loop bugs.
- Video: gem5 bootcamp lectures on YouTube (ARM/UC Davis) — watch only the memory-system sessions; useful vocabulary even though you are not using gem5.

---

# E7 — Confidence-gated precision-aware prefetch **[P1, the novel contribution]**

**Why.** This is the mechanism that rescues the bandwidth-bound regime, and it follows directly from your own pilot data. Recall at m=2k is ~60%: you fetch 2k experts to catch 60% of the k you need. That is a 2× byte cost for a partial hit rate — unaffordable per E1. The fix: **do not fetch every candidate at full precision.** Fetch the high-confidence candidates in FP16/BF16 and the low-confidence ones in NF4/INT8 (4–8× fewer bytes). If a low-confidence candidate turns out to be needed, you already have a usable low-precision copy — degraded quality instead of a multi-millisecond stall. If it was not needed, you wasted a quarter of the bytes you otherwise would have.

This converts prediction error from a *latency* penalty into a *quality* penalty, and quality penalties are measurable, tunable, and often negligible. No published system does confidence-gated precision selection for expert prefetch over a byte-addressable tier; HOBBIT does mixed-precision on *miss*, reactively. Yours is predictive and calibrated (which is why E4's calibration work is a prerequisite, not a nicety).

**Method.**
1. Store each expert in the CXL tier at two precisions (FP16 + NF4; total capacity cost +25%, which E1's capacity model must account for — state this honestly).
2. Policy: for predicted candidate e with calibrated probability p, fetch FP16 if p ≥ τ_hi, NF4 if τ_lo ≤ p < τ_hi, skip if p < τ_lo. Sweep (τ_lo, τ_hi).
3. **Quality evaluation** — the part that makes this credible: run real inference where low-confidence experts are substituted with their NF4 versions according to the policy's decisions replayed from traces, and measure perplexity on WikiText-2 plus task accuracy on a small benchmark slice (e.g., 200 GSM8K or HumanEval items). Report ΔPPL and Δaccuracy vs bytes saved.
4. Pareto frontier: bytes/token vs quality vs TPOT. Identify knee points.

**Deliverables.** `tiermoe/policy/precision_gate.py`; `figures/pareto_bytes_quality_latency.pdf`; a quality table (ΔPPL, Δacc) at each operating point; the report's contribution section.

**Exit criteria.** A configuration exists where bytes/token drop ≥30% with ΔPPL ≤ ~0.1 and no material accuracy loss. If quality degrades unacceptably, report the frontier anyway — a negative result with a Pareto curve is still a result, and the framing ("we bounded the achievable trade-off") holds.

**Effort.** 18–22 h (GPU needed for step 3).

**Resources.**
- Tang et al., *HOBBIT*, arXiv:2411.01433 — mandatory read; your delta is predictive + calibrated + tier-aware.
- Dettmers et al., *QLoRA / NF4*, arXiv:2305.14314; Frantar et al., *GPTQ*, arXiv:2210.17323; Lin et al., *AWQ*, MLSys 2024 — quantization background for the quality argument.
- Frantar & Alistarh, *Sparse/quantized MoE* work and *QMoE*, arXiv:2310.16795 — compressing MoE weights specifically; very relevant citation.
- Guo et al., ICML 2017 (calibration) — again, τ thresholds are only meaningful on calibrated probabilities.
- Video: MIT 6.5940 lectures on quantization (Song Han) — the clearest visual explanation of NF4/INT4 trade-offs if a teammate needs the background.

---

# E8 — KV-cache co-tenancy on the CXL link **[P2]**

**Why.** Astera's own published CXL-for-inference story is about KV cache, not experts. A section showing that your framework handles both — and, crucially, that they *contend* for the same link — speaks directly to their product roadmap and is the most internship-relevant part of the report. It also future-proofs the work against the objection "expert offload is niche."

**Method.**
1. Model KV-cache bytes per token per request (standard formula: 2 · L · n_kv_heads · d_head · precision · context_len) for realistic context lengths (4k, 32k, 128k).
2. Simulate a serving mix where long-context KV spills to CXL while experts are also being prefetched from CXL: shared-link contention, priority policies (KV-first vs expert-first vs weighted).
3. Report TTFT (KV-sensitive) and TPOT (expert-sensitive) under each priority policy; find the scheduling rule that Pareto-dominates.

**Deliverables.** `figures/kv_expert_contention.pdf`; a "link scheduling recommendation" paragraph aimed at controller designers.

**Exit criteria.** A stated priority rule with quantified TTFT/TPOT trade-off.

**Effort.** 10 h (simulator reuse).

**Resources.**
- Kwon et al., *vLLM/PagedAttention*, SOSP 2023 — KV cache mechanics.
- Tang et al. (ByteDance), *Exploring CXL-based KV Cache Storage for LLM Serving*, NeurIPS 2024 ML-for-Systems workshop — real ASIC-CXL + GPU inference server; your closest KV precedent.
- *TraCT*, arXiv:2512.18194 — rack-scale CXL shared-memory KV cache.
- Samsung, *CXL memory pooling for KV cache with vLLM + LMCache* white paper (2026).
- **Astera Labs, "Breaking Through the Memory Wall: CXL for RAG and KV Cache"** — read it twice; quote its framing back at them in the presentation.
- Video: LMCache / vLLM office-hours talks on YouTube covering KV offloading.

---

# E9 — Pooling and TCO **[P1]**

**Why.** Deliverable explicitly requested by the competition brief ("CXL memory expansion and pooling study"), and it is where the argument becomes a business case — which is what wins a company-judged competition.

**Method.** Follow Pond's stranding logic: fleet of N hosts serving a mix of MoE models with time-varying demand (derive per-model memory footprints from your traces and E1's residency requirements); compare per-host provisioning at p99 vs pooled provisioning at fleet p99; sweep HBM:DDR:CXL cost ratios 2–5× to show conclusions are robust to price assumptions. Headline metric: **tokens/sec per $ of memory at iso-p99-TPOT**, plus stranded-capacity reduction %.

**Deliverables.** `tiermoe/tco/` notebook; `figures/tco_curves.pdf`; `figures/stranding.pdf`; the report's economics section and the presentation's slide 10.

**Exit criteria.** A crossover point identified: the model size / traffic profile beyond which HBM+CXL beats HBM-only on tokens/sec/$.

**Effort.** 10 h.

**Resources.**
- Li et al., *Pond*, ASPLOS 2023, arXiv:2203.00241 — the template; replicate its stranding methodology.
- *Amplifying Effective CXL Memory Bandwidth for LLM Inference*, arXiv:2509.03377 — cost-ratio grounding.
- Meta/Microsoft CXL deployment talks at OCP Summit (YouTube) — search "OCP Summit CXL memory expansion deployment"; good source of real fleet-level framing and quotable industry context.

---

# E10 — Hardware validation without CXL silicon **[P1]**

**Why.** Your single biggest credibility risk is "no CXL hardware." Three artifacts neutralize it.

**Method.**
1. **PCIe ground truth (from E3.3)** — real offloading runs, policy ordering measured on real silicon over the same physical layer CXL uses.
2. **QEMU CXL Type-3 stack demo** — boot Linux 6.x with an emulated CXL device; `cxl create-region`, DAX/kmem bind, memory appears as a NUMA node; place expert tensors there with `numactl --membind` and run OLMoE end-to-end. Functional, not timing-accurate — **say so on the slide**. Purpose: proves you understand the production Linux CXL stack.
3. **NUMA-remote emulation** *(stretch, needs a dual-socket box — the DaSH Lab ask)* — the Pond/TPP-standard CXL stand-in. If it lands, it upgrades the validation section from "PCIe analogue" to "latency-accurate analogue."

**Deliverables.** `demo/qemu-cxl/` scripts + recorded video; `results/hw_validation/` measured-vs-simulated table; the "Methodology and Validity" section of the report.

**Exit criteria.** Simulator reproduces measured PCIe policy ordering; QEMU demo runs unattended from a script.

**Effort.** 12 h (QEMU setup is fiddly; budget a full day).

**Resources.**
- Linux kernel docs: `Documentation/driver-api/cxl/`; `ndctl`/`cxl-cli` man pages and the pmem.io CXL guides.
- QEMU docs on CXL emulation (`docs/system/devices/cxl.rst`).
- Li et al., *Pond* §evaluation and Maruf et al., *TPP* §methodology — cite these as precedent for NUMA-based emulation so it reads as standard practice, not improvisation.
- Video: SNIA SDC / CXL Consortium talks on the Linux CXL software stack (search "SNIA SDC CXL Linux software stack"); also Dan Williams' kernel talks on CXL/DAX.

---

# E11 — Model generalization **[P2]**

**Why.** The pilot's most striking scientific claim — *skew inverts with granularity: fine-grained MoEs are more skewed than coarse ones, because the load-balancing loss binds much harder when there are only 8 slots* — currently rests on n=2 models. Two more models turn an anecdote into a trend, and the shared-expert architectures (DeepSeek, Qwen) are a genuinely different third regime worth measuring.

**Method.** Re-run the ELP-Probe unchanged on: DeepSeek-MoE-16B (64 routed + 2 shared, top-6), Qwen1.5-MoE-A2.7B (60 routed + 4 shared, top-4), and optionally a recent large-vocabulary-of-experts model if one fits your GPUs. Additional analysis specific to shared experts: shared experts are *always* active ⇒ they belong permanently in HBM ⇒ measure what fraction of expert bytes they represent (a free, immediately actionable placement rule).

**Deliverables.** Extension of Table 1 to 4 models; `figures/gini_vs_n_experts.pdf` — a scatter of Gini vs experts-per-layer across all models with a fitted trend. **This single figure is the paper-worthy result** and takes one afternoon to produce once traces exist.

**Exit criteria.** Trend confirmed or refuted across ≥4 models.

**Effort.** 8 h GPU + 4 h analysis.

**Resources.**
- DeepSeek-AI, *DeepSeekMoE*, arXiv:2401.06066 (shared-expert isolation, fine-grained segmentation — directly explains your inversion finding).
- Muennighoff et al., *OLMoE*, arXiv:2409.02060 (their §analysis of expert specialization/routing saturation is a strong citation for your skew result).
- Fedus et al., *Switch Transformer*, JMLR 2022; Shazeer et al., *Outrageously Large Neural Networks*, ICLR 2017 — origin of the load-balancing auxiliary loss whose behavior you are empirically characterizing. **Cite these when you explain the inversion — it shows you know why, not just that.**
- Qwen team, Qwen1.5-MoE technical blog.

---

# E12 — Ablations, sensitivity, reproducibility **[P1]**

**Why.** The difference between a good student project and a submission that reads like a conference paper is the ablation section. It is also cheap once E6 exists.

**Checklist.**
- Sensitivity to CXL latency (150/250/400 ns) and BW — show which conclusions are robust.
- Sensitivity to cost ratios (E9) — already covered, cross-reference.
- Predictor ablation (E4.2) — where the signal lives.
- Policy component ablation: placement-only, prefetch-only, both, both+precision.
- Seed/sampling sensitivity: rerun the whole pipeline with a second request sample and second seed (addresses pilot open item #3 directly).
- Bootstrap CIs on every headline number (already established in the pilot — keep the practice).
- Artifact: `make reproduce` regenerating every figure from committed traces; HuggingFace dataset upload of the traces; README with a 90-second pitch.

**Deliverables.** `docs/ablations.md`, `tests/`, artifact README, HF dataset link in the report.

**Effort.** 12 h.

**Resources.** ACM Artifact Review and Badging guidelines (v1.1) — structure the artifact section to match; it signals research maturity to any judge with a systems background.

---

# E13 — Dashboard, report, presentation **[P0]**

**Why.** The brief explicitly requires an interactive demo and a final report; the presentation is where first place is actually decided.

**Dashboard (Streamlit + Plotly), four tabs.**
1. *Routing explorer* — animated expert heatmap over token time (from real traces). The screenshot people remember.
2. *Roofline explorer* — sliders for expert size, batch, residency, link BW; the ρ=1 boundary moves live and the current configuration lights up red/green. **This is the demo centerpiece: it makes your central intellectual contribution tangible in five seconds.**
3. *Policy explorer* — pre-computed sweep grid; TPOT/stall/bytes gauges per policy.
4. *Fleet & TCO* — pooling curves with cost-ratio slider.

**Report** (IEEE two-column, 12–16 pp): Abstract · Intro · Background/Related Work (explicit delta vs arXiv:2512.04476, HOBBIT, fMoE, Pre-gated MoE) · **Characterization (pilot + E11, including the skew-inversion result)** · **The MoE Tiering Roofline (E1, E2)** · System Design (E4, E5, E7) · Methodology & Validity (E3, E10 — the provenance table) · Evaluation (E6) · Economics (E9) · Ablations (E12) · Architecture Recommendations for Leo-class Deployments · Limitations · Conclusion.

**Slides (≤12):** 1 Title · 2 MoE memory wall · 3 Why CXL is the right tier · 4 **The reframe: prefetch hides latency, it doesn't create bandwidth** (this slide is your differentiator — lead with the insight nobody else will have) · 5 Roofline figure · 6 Characterization + skew inversion · 7 System design · 8 Headline result · 9 Precision-gating Pareto · 10 Validation (PCIe + QEMU) · 11 Economics · 12 Recommendations for Astera + live demo.

**Effort.** 30 h across the final three weeks.

**Resources.**
- Simon Peyton Jones, *How to Give a Great Research Talk* (video + slides, Microsoft Research) — watch it once before building slides; it will measurably improve the presentation.
- Streamlit + Plotly docs.
- IEEEtran LaTeX template; Zotero/BibTeX for the ~25 references.

---

## §14. Revised schedule (Aug 14 → Sept 25)

| Window | Primary (you) | Secondary (teammates, or your cut-list if solo) |
|---|---|---|
| Aug 14–18 | **E1 roofline**, E3 calibration session | E2 batching curves |
| Aug 19–25 | **E6 simulator core**, E4 predictor v2 | E11 extra models (GPU time), E5 depth study |
| Aug 26–Sept 1 | E6 sweeps + validation gates, **E7 precision gating** | E8 KV co-tenancy, E9 TCO |
| Sept 2–8 | E7 quality runs, E10 QEMU + PCIe validation | Dashboard build |
| Sept 9–12 | E12 ablations, **report freeze Sept 12** | Figures polish |
| Sept 13–15 | Buffer, **final submission Sept 15** | — |
| Sept 16–25 | Slides, demo script, 25-question Q&A bank, 2 rehearsals | Backup demo video |

**If you fall behind, cut in this order:** E8 → E11 → E5 (keep d=1 only) → E9 (reduce to analytical appendix) → E7 (demote to "future work" — but this costs you the novelty crown, so protect it above E9).

---

## §15. Reading order for the team (do not read all 25 papers)

**Week 1, everyone (≈4 h):** Horace He's "Go Brrrr" blog → Roofline (CACM'09) §2–3 → Astera Leo blog → Sun et al. ISCA'23 (skim figures).
**Week 1, you additionally:** Pond (ASPLOS'23) full → HOBBIT → Pre-gated MoE.
**Week 2:** fMoE → MoE-Infinity → DeepSeekMoE §routing → TPP (skim).
**Week 3:** ByteDance CXL-KV → TraCT → arXiv:2512.04476 (write your differentiation paragraph immediately after reading this one, while it's fresh).
**Videos, one per week, background listening:** CXL Consortium intro webinar → MIT 6.5940 LLM-serving lecture → CS336 MoE lecture → SPJ's research-talk video (before slide-building).

---

## §16. The three sentences that win this

Rehearse until automatic:

1. *"We measured, on real traces from two MoE architectures, that expert traffic is predictable one layer ahead — 55 to 62 percent recall on the experts that aren't already resident, beating a popularity baseline by 14 to 25 points."*
2. *"But we also found that prefetching hides latency without creating bandwidth, so we built a roofline that says exactly when CXL tiering works: expert granularity and batch size, not prediction accuracy, decide it."*
3. *"On the wrong side of that line we spend fewer bytes instead of more time — low-confidence experts are fetched in four-bit, so a misprediction costs a little quality instead of a ten-millisecond stall."*
