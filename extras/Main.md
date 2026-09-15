# Router-Guided Memory Tiering: Cost-Efficient MoE Inference on HBM + CXL Memory Systems

**Project master plan — Nebula@BITS Goa 2026 (Astera Labs), Software Track: "CXL-Based Memory Optimization for MoE Models"**

| Milestone | Date |
| --- | --- |
| ~~Abstract submission~~ | ~~**Aug 6, 2026**~~ |
| ~~Abstract shortlisting~~ | ~~Aug 12, 2026~~ |
| Final submission | **Sept 15, 2026** |
| Final shortlisting | Sept 19, 2026 |
| Final presentation | **Sept 25, 2026** |

> **VERSION 2 — revised Aug 14, 2026, after the ELP-Probe pilot.** Changes from v1 are marked **[v2]**. Headline changes: (a) the system is now **prefetch-primary**; the cost-constrained placement optimizer is demoted to a fixed-residency policy; (b) a new central contribution, the **MoE Tiering Roofline**, was added after a bandwidth calculation showed prefetch alone cannot rescue coarse-grained models; (c) **confidence-gated precision fallback** is the new novelty item; (d) hypotheses H1/H2 are replaced by measured results. Detailed experiment specs live in the companion file `experiment_roadmap.md`.

**Goal:** 1st place + convert visibility into an Astera Labs internship. The strategic differentiators: (1) a real algorithmic contribution (router-guided placement + prefetch), not just a parameter sweep; (2) evidence at three levels (real traces → calibrated simulation → hardware-grounded validation); (3) explicit alignment with Astera's Leo CXL Smart Memory Controller product story (they have publicly positioned Leo as the answer to the LLM memory wall / KV-cache problem).

---

## 1. Problem Statement

### 1.1 Short version (abstract-ready, trim to word limit)

> Mixture-of-Experts (MoE) models decouple parameter count from per-token compute, but their full parameter footprint (87 GB for Mixtral-8x7B in FP16; >280 GB for Mixtral-8x22B) far exceeds GPU HBM capacity, forcing expert offloading that turns inference from memory-bound to I/O-bound. CXL memory expansion offers a byte-addressable capacity tier with load/store semantics and pooling capability, but it is not obvious which MoE workloads it can actually serve.
>
> We traced router decisions for 250 real chat and code requests through two MoE architectures spanning the granularity spectrum — OLMoE-1B-7B (64 experts/layer, top-8) and Mixtral-8x7B (8 experts/layer, top-2) — with a request-level train/test split. Two findings reshape the design. First, **static expert placement does not generalize**: the top 25% of experts serve 52.2% of held-out decode activations in OLMoE but only 28.7% in Mixtral (Gini 0.378 vs 0.066), inverting the intuition that coarser experts are more skewed — an effect we attribute to the load-balancing auxiliary loss binding far more tightly over 8 slots than over 64. Second, **router-guided prefetch does generalize**: a training-free conditional predictor recalls 55.3% (OLMoE) and 61.8% (Mixtral) of non-resident next-layer experts one layer ahead, beating a static-popularity baseline by 14–25 points and bounding 72.8–78.7% of expert traffic as non-stalling.
>
> We therefore propose **TierAhead**, a prefetch-primary tiering framework for HBM+CXL MoE inference. Its core contribution is the **MoE Tiering Roofline**: because prefetching hides latency but does not reduce bytes moved, whether CXL is viable is set by arithmetic intensity per expert byte — a function of expert granularity and batch size — not by prediction accuracy. We map that boundary analytically and in a trace-driven simulator calibrated against measured GPU compute and PCIe transfer times plus published CXL timings, and we introduce **confidence-gated precision fallback**, which fetches low-confidence predicted experts at 4-bit so that mispredictions cost bounded quality rather than multi-millisecond stalls. We evaluate expert and KV-cache co-tenancy on a shared link, model multi-tenant CXL pooling, and report tokens/sec/$ and stranding reduction, concluding with provisioning recommendations for CXL memory-controller deployments.

### 1.2 Extended framing (for the report introduction)

Three converging trends create the problem:

1. **MoE is winning.** Frontier open models (Mixtral, DeepSeek-V2/V3, Qwen-MoE, OLMoE, Llama-4-class systems) use sparse expert routing: total parameters grow 5–10× faster than *activated* parameters. Compute per token stays cheap; **memory capacity becomes the binding constraint**.
2. **HBM cannot scale capacity.** HBM is packaging- and thermal-limited to low hundreds of GB per accelerator and costs ≥3× more per GB than DDR5. Provisioning HBM for *total* (not active) parameters is economically irrational when most experts are cold most of the time.
3. **CXL provides exactly the missing tier.** CXL.mem attaches DDR-class capacity behind a PCIe-physical link with cacheline load/store semantics (no explicit DMA programming), at ~170–250 ns added latency and tens of GB/s per link — worse than local DDR, dramatically better than SSD, and *poolable* across hosts (CXL 2.0 switching), which host DRAM is not.

**The gap [v2]:** existing expert-offloading systems (MoE-Infinity, fMoE, HOBBIT, Mixtral-offloading, Klotski, FloE) assume a two-level GPU↔host-DRAM hierarchy with explicit PCIe copies. Existing CXL tiering systems (TPP, Pond, Memtis) assume opaque page-granularity workloads and reactive hotness tracking. Two things are missing from both literatures:

1. **Semantic predictability is unexploited on a byte-addressable tier.** MoE routers make future memory accesses semantically predictable — the gate at layer *l* tells you which multi-hundred-MB objects will be needed a layer from now. Our pilot confirms this holds across architectures (55–62% recall on non-resident experts, +14–25 pp over popularity). Page-level tiering can never make this decision; application-level tiering can.
2. **Nobody has stated the viability condition.** Prefetching converts stalls into overlap; it does not reduce bytes moved. When per-layer transfer time exceeds per-layer compute time — which, per §3.1, is the *default* for coarse-grained MoE at low batch — no predictor of any accuracy helps. The literature reports speedups without characterizing where the mechanism stops working. We supply that boundary as a roofline, and a byte-reduction mechanism for the far side of it.

That pair is the thesis of this project.

### 1.3 Research questions **[v2 — reordered around the roofline]**

- **RQ1 (Regime):** Under what combinations of expert granularity, batch size, residency fraction, and link bandwidth is CXL-tiered MoE inference compute-bound (prefetch can help) versus bandwidth-bound (only byte reduction helps)? → E1, E2
- **RQ2 (Prediction):** How much of the next layer's expert set is predictable from the current layer's router state, how does accuracy decay with lookahead depth, and where does the predictive signal live (selected IDs vs gate weights vs hidden state)? → E4, E5. *Partially answered by the pilot: 55.3% / 61.8% at m=2k, d=1.*
- **RQ3 (Mechanism):** How much demand-stall reduction and TPOT improvement does router-guided prefetch deliver inside the viable regime, and what fraction of the oracle gap does it close? → E6
- **RQ4 (Byte reduction):** Can confidence-gated precision selection trade bounded quality for bandwidth well enough to extend the viable regime to coarse-grained models? → E7
- **RQ5 (Economics):** At what model scale and traffic profile does HBM+CXL beat HBM-only on tokens/sec/$, and how much stranded capacity does pooling reclaim? → E9
- **RQ6 (Co-tenancy):** How should a shared CXL link be scheduled between expert prefetch and KV-cache spillover? → E8

**Retired from v1:** *"what expert-to-tier assignment minimizes stall time, and how far is greedy from the ILP optimum?"* — the pilot showed placement headroom is small and architecture-dependent (Coverage@50% only reaches 74.1% / 54.5%), so an exact optimizer cannot buy much. Fixed-residency policy replaces it; the retirement is itself reported as a finding.

## 2. Background and Related Work (know this cold for Q&A)

### 2.1 MoE inference and expert offloading

| Work | Venue/ID | Key idea | Our relationship to it |
| --- | --- | --- | --- |
| Mixtral of Experts (Jiang et al.) | arXiv:2401.04088 | 8 experts/layer, top-2 routing; 47B total / 13B active | Primary workload |
| Fast MoE inference with offloading (Eliseev & Mazur) | arXiv:2312.17238 | LRU expert cache + speculative loading on consumer GPUs | Baseline cache policy |
| MoE-Infinity (Xue et al.) | arXiv:2401.14361 | Activation-aware prefetch from request-level Expert Activation Matrix | Adapt its tracing idea; we target CXL not DRAM-copy |
| Pre-gated MoE (Hwang et al.) | ISCA 2024, arXiv:2308.12066 | Architectural change: gate at layer l selects experts for l+1 | Motivates cross-layer predictability; we stay off-the-shelf |
| fMoE (Yu et al.) | arXiv:2502.05370 | Iteration-level expert maps + semantic hints; −47% latency, +36% hit rate | SOTA fine-grained offloading; policy baseline to compare against |
| HOBBIT (Tang et al.) | arXiv:2411.01433 | Mixed-precision expert loading (low-precision fallback on cache miss) | Orthogonal; cite as combinable |
| Klotski (arXiv:2502.06888), FloE (arXiv:2505.05950) | — | Pipeline-bubble reduction; on-the-fly compression | Evidence that expert I/O is the bottleneck |
| SiDA-MoE (Du et al.) | MLSys 2024, arXiv:2310.18859 | Hash/predictor-based expert selection ahead of time | Predictor design reference |
| Context-aware MoE on CXL-NDP (Fan et al.) | arXiv:2512.04476 | **Closest prior work.** Executes cold experts in-place on CXL near-data processors | **Differentiation anchor**: we target *commodity* CXL expanders (what Astera ships today), pure placement/prefetch policy + pooling economics, no custom NDP hardware |

### 2.2 CXL and tiered memory systems

| Work | Venue | Key idea | What we take |
| --- | --- | --- | --- |
| Pond (Li et al.) | ASPLOS 2023, arXiv:2203.00241 | CXL pooling for clouds; latency ~ NUMA-remote; ML-driven allocation | Pooling/stranding economics model; NUMA-emulation methodology |
| TPP (Maruf et al.) | ASPLOS 2023, arXiv:2206.02878 | OS-transparent page placement for CXL tiers | Reactive page-tiering baseline our semantic policy beats |
| Demystifying CXL Memory (Sun et al.) | ISCA 2023 | Measurements on true CXL hardware: latency, bandwidth, tail behavior | Authoritative timing parameters for the simulator |
| Demystifying CXL Type-2 (Ji et al.) | MICRO 2024 | Type-2 device co-computing characterization | Background for Q&A |
| Memtis (Lee et al.) | SOSP 2023 | Hotness-histogram-based tiering | Alternative baseline policy |
| Exploring CXL-based KV cache (Tang et al., ByteDance) | NeurIPS 2024 MLSys wksp | First real ASIC-CXL + GPU inference server for KV cache | Validates CXL-for-inference premise; KV extension reference |
| TraCT (arXiv:2512.18194) | 2025 | Rack-scale CXL shared-memory KV cache on Dynamo; −9.8× TTFT | Pooling-for-inference evidence |
| Samsung CXL pooling white paper (2026) | industry | vLLM + LMCache KV offload to pooled CXL | Industry validation slide |
| **Astera Labs Leo blog: "Breaking Through the Memory Wall"** | asteralabs.com | Leo controllers for RAG/KV-cache tiers; CXL-CPU ≈ CPU-GPU transfer performance | **Cite explicitly. Frame project as policy layer for Leo-class devices.** |

### 2.3 Simulation & emulation tooling

- **DRAMSim3** (Li et al., CAL 2020, github.com/umd-memsys/DRAMsim3) — cycle-accurate DDR timing; C++ with Python bindings possible; satisfies the organizers' tool list.
- **gem5 CXL forks** — exist but full-system simulation of an LLM is intractable in this timeline; we use gem5 only if time permits for a microbenchmark appendix. (Decision recorded in §9 Risk Register.)
- **QEMU ≥ 7.1 CXL Type-3 emulation** — functional (not timing-accurate) emulation of CXL memory devices; lets us demo the *real Linux CXL software stack* (cxl-cli, DAX, kmem driver, weighted interleave) with zero CXL hardware.
- **NUMA-remote emulation** — pin memory to a remote socket via `numactl`; adds ~100–140 ns, the standard CXL stand-in used by Pond/TPP. Requires a dual-socket Linux box (stretch goal — ask DaSH Lab).

---

## 3. Results so far, and the hypotheses that remain **[v2 — fully rewritten]**

### 3.1 The bandwidth reframe (drives everything below)

Expert bytes moved per decode token, if non-resident, FP16:

| Model | Expert size | Bytes/layer (top-k) | Bytes/token | Transfer/layer @32 GB/s | Per-layer compute (batch 1) |
| --- | --- | --- | --- | --- | --- |
| Mixtral-8x7B (8 exp, top-2, 32 L) | 352 MB | 705 MB | 22.5 GB | **22.0 ms** | ~0.1–0.3 ms |
| Mixtral, 50% resident | — | 352 MB | 11.3 GB | **11.0 ms** | ~0.1–0.3 ms |
| OLMoE-1B-7B (64 exp, top-8, 16 L) | 12.6 MB | 101 MB | 1.61 GB | **3.15 ms** | ~0.05–0.15 ms |
| OLMoE, 50% resident | — | 50 MB | 0.81 GB | **1.57 ms** | ~0.05–0.15 ms |

Transfer exceeds compute by 1–2 orders of magnitude at batch 1 in every cell. **Prefetch hides latency; it does not create bandwidth.** Hence the v2 thesis: viability is set by arithmetic intensity per expert byte (granularity × batch), and the project's job is to map that boundary and supply mechanisms on both sides of it. Compute figures above are estimates pending E3 measurement.

### 3.2 Confirmed by the pilot (no longer hypotheses)

| ID | Statement | Evidence | Status |
| --- | --- | --- | --- |
| **R1** | Static placement does not generalize across MoE architectures | Coverage@25% = 52.2% (OLMoE) vs 28.7% (Mixtral), held-out, bootstrap CIs [48.2–55.9] and [27.7–30.0] | **Confirmed** — optimizer demoted |
| **R2** | Router-guided one-layer-ahead prefetch generalizes | Recall@2k, d=1, non-resident = 55.3% / 61.8%; +14–25 pp over static-popularity baseline | **Confirmed** — promoted to centerpiece |
| **R3** | Skew inverts with granularity: fine-grained MoEs are *more* skewed than coarse ones | Gini 0.378 (64 exp) vs 0.066 (8 exp); Zipf 0.70 vs 0.17 | **Confirmed on n=2**, to be extended to n=4 in E11 |
| **R4** | Combined hideable-traffic bound | HTB(C=25%, m=2k) = 0.787 / 0.728; implied demand-stall 21.3% / 27.3% | **Confirmed as a bound** — the simulator (E6) must now show what a real policy achieves against it |

Artifacts to avoid quoting (flagged in the pilot write-up, repeated here so they never leak into the report): Mixtral's churn number is a pigeonhole artifact of a 7-of-8 hot set; all `m=4k` rows for Mixtral are saturated at 1.0 because 4k = n_experts.

### 3.3 Hypotheses still open

- **H-A (Regime boundary):** there exists a (granularity, batch, residency, BW) region where per-layer transfer ≤ per-layer compute, and fine-grained MoEs at moderate batch fall inside it. → *Figure: roofline with ρ=1 contours (E1).* **This is the new Figure 1 of the report.**
- **H-B (Batch optimum):** batching improves arithmetic intensity faster than it erodes expert locality, up to a model-dependent B\*, and B\* is larger for fine-grained models. → *Figure: bytes/token and ρ vs B with B\* marked (E2).*
- **H-C (Depth):** prediction accuracy decays gracefully enough with lookahead depth that a d-layer horizon can cover fetches d× larger. → *Figure: recall vs depth, feasible fetch size (E5).*
- **H-D (Precision gating):** fetching low-confidence predicted experts at 4-bit reduces bytes/token ≥30% for ΔPPL ≤ ~0.1. → *Figure: bytes–quality–latency Pareto (E7).* **This is the novelty claim.**
- **H-E (Economics):** beyond a model-size threshold, HBM+CXL dominates HBM-only on tokens/sec/$; pooling reclaims double-digit % stranded capacity. → *Figure: TCO curves (E9).*

Any of H-A–H-E failing is reportable as a bounded negative result, as the pilot's placement finding already was.

## 4. Framework Architecture (what you are actually building)

The deliverable is a Python framework called **`TierAhead`**, with eight components **[v2: placement optimizer demoted to [3]; roofline [0] and precision gate [4b] added]**. Everything except the optional hardware harnesses runs on a laptop.

```
                        ┌───────────────────────────────────────────┐
                        │                 TierAhead  (v2)                │
  HuggingFace MoE ────► │ [1] Trace Collector  ──► traces/*.jsonl.zst │
  models (CUDA)         │        │                                    │
                        │        ▼                                    │
                        │ [2] Workload Analyzer ──► stats/, figures/  │
                        │        │                                    │
                        │        ├──► [0] ROOFLINE MODEL  ◄── calib/  │  ◄── new in v2:
                        │        │        (regime map, B*)           │      decides which
                        │        ▼                                    │      mechanism applies
                        │ [3] Fixed-residency policy (was: ILP)       │
                        │        │                                    │
                        │        ▼                                    │
                        │ [4] Prefetch Engine (predictor + depth)     │  ◄── centerpiece
                        │        │                                    │
                        │        ▼                                    │
                        │ [4b] Confidence-gated precision fallback    │  ◄── new in v2:
                        │        │                                    │      novelty claim
                        │        ▼                                    │
                        │ [5] Tiered-Memory Simulator (DES, shared-BW │
                        │     link, DRAMSim3-calibrated)              │
                        │        ├──► [6] Pooling & TCO + KV co-tenancy│
                        │        └──► [7] Dashboard (Streamlit)        │
                        │                                             │
                        │ [V1] QEMU CXL software-stack demo           │
                        │ [V2] NUMA-remote emulation (optional)       │
                        │ [V3] PCIe offloading ground truth           │
                        └───────────────────────────────────────────┘
```

### 4.0 [0] Roofline Model **[v2, new — see E1]**

Analytical model mapping (model, batch B, residency r, precision p, link BW) → regime ratio ρ = t_transfer / t_compute per layer, with ρ = 1 contours in the (B, r) plane. Consumed by every downstream component: the prefetch engine only engages where ρ < 1; the precision gate engages where ρ ≥ 1. Outputs the provisioning table that becomes the report's architecture-recommendations section.

### 4.1 [1] Trace Collector

**What:** PyTorch forward hooks registered on every MoE gate module; logs, per token per layer: router logits (top-k + entropy), selected expert IDs, gate weights, token position, request ID, timestamp. Zero model modification.

**Trace schema (JSONL, zstd-compressed):**

```json
{"model":"mixtral-8x7b","req":"sharegpt-00421","phase":"decode",
 "layer":17,"tok":128,"topk_experts":[3,6],"gate_w":[0.61,0.39],
 "logits_top4":[[3,2.11],[6,1.87],[0,0.42],[5,0.31]],"t_ns":183456}
```

**Target models** (chosen to span routing regimes):

| Model | Total / active params | Experts per layer / top-k | FP16 weights | Why |
| --- | --- | --- | --- | --- |
| OLMoE-1B-7B | 7B / 1B | 64 / top-8 | ~14 GB | Fine-grained routing; runs on a Mac |
| Qwen1.5-MoE-A2.7B | 14.3B / 2.7B | 60 + 4 shared / top-4 | ~28 GB | Shared-expert architecture |
| DeepSeek-MoE-16B | 16.4B / 2.8B | 64 + 2 shared / top-6 | ~33 GB | Fine-grained + shared experts |
| Mixtral-8x7B | 46.7B / 12.9B | 8 / top-2 | ~87 GB | The canonical case; coarse experts (~340 MB each) — strongest CXL story |

**Workloads:** ShareGPT (chat), HumanEval/MBPP prompts (code), LongBench subsets (long-context), plus a synthetic uniform-random control. ≥200 requests per (model, workload) cell; log both prefill and decode phases separately (their locality differs — this is a point most teams will miss).

**Expected output:** ~5–20 GB of compressed traces. This dataset is itself a citable artifact — publish it on HuggingFace Datasets and put the link in the report (professional touch; costs nothing).

### 4.2 [2] Workload Analyzer

Pure pandas/NumPy/matplotlib. Computes, per (model, workload, phase):

1. Expert popularity distribution + Gini coefficient / Zipf fit (H1)
2. Temporal locality: reuse distance distribution over expert accesses; hot-set (top-p mass) churn across sliding windows
3. Cross-layer transition statistics: P(expert set at l+1 | expert set at l) — mutual information per layer pair (H2 feasibility before building the predictor)
4. Per-request expert footprint: how many *distinct* experts does one request touch (drives per-request pinning vs global policy)
5. Prefill-vs-decode contrast
6. Batch-aggregation effect: at batch size B, union of selected experts approaches all experts — quantify where per-token skew stops helping (this bounds the applicability regime honestly; judges will ask)

**Exit criterion:** H1 and H2 confirmed/refuted with numbers, and the figures for the report's characterization section done.

### 4.3 [3] Fixed-Residency Placement **[v2 — demoted from ILP optimizer]**

**What changed and why.** v1 specified a multiple-knapsack ILP over (expert, tier). The pilot showed the marginal value of solving it exactly is small: Coverage@50% tops out at 74.1% (OLMoE) / 54.5% (Mixtral), so even a perfect solver leaves a large residual that only prefetch can address, and for Mixtral the popularity signal is nearly flat (Gini 0.066) so any ranking is close to arbitrary. Engineering time moves to [4] and [4b].

**What remains (≈1 day of work):** a per-layer fixed-residency policy with C ∈ {12.5, 25, 37.5, 50, 75}%, ranked by training-split frequency, plus two free rules the pilot and E11 justify:

- **Shared experts always resident.** DeepSeek/Qwen-style shared experts are active on every token — pinning them is unconditionally correct and costs one line.
- **Epoch re-ranking** every N tokens, with the migration-cost term retained. Justified by OLMoE's churn of 0.447 (high enough that a once-and-forever ranking decays).

The ILP survives only as a one-paragraph appendix comparing greedy vs optimal on one configuration, to demonstrate the headroom is small — i.e. the *negative* result is the deliverable.

### 4.4 [4] Prefetch Engine — the centerpiece **[v2: promoted]**

Pluggable policies, all evaluated on identical traces (see E4, E5):

| Policy | Description | Role | Pilot status |
| --- | --- | --- | --- |
| `none` | Fetch on demand at use time | Lower bound | — |
| `lru-cache` | HBM as LRU cache of experts | Classic baseline | — |
| `static-C` | Fixed residency, no runtime motion | Placement-only ablation | Coverage 52.2% / 28.7% |
| `popularity-prefetch` | Prefetch globally popular experts, unconditioned | **Critical baseline** — isolates the value of conditioning | 52.8% / 54.4% (full-set) |
| `prefetch-freq` | Conditional frequency table, d=1, m=2k | Pilot mechanism | **55.3% / 61.8%** non-resident |
| `prefetch-v2` | MLP predictor, tuned (m, d), calibrated | The contribution | E4/E5 |
| `prefetch-v2 + precision-gate` | §4.4b | Full system | E7 |
| `oracle` | Perfect future knowledge from trace | Upper bound; report % of oracle gap closed | — |

**Design constraints the pilot imposed.** (i) *m is a bandwidth multiplier, not a free knob* — m=2k means fetching twice the bytes you need, which §3.1 says is often unaffordable; the operating point is chosen on the recall-vs-bytes curve under the roofline budget, never by maximizing recall. (ii) *Cap m at n_experts* — the pilot's m=4k rows saturate trivially for Mixtral; the code must cap and the report must note it. (iii) *Report accuracy, coverage, and timeliness separately* — standard prefetcher vocabulary; a hardware-company judge will expect it. (iv) *Mispredicted prefetches contend for link bandwidth* — the simulator models this explicitly, with a throttle knob.

### 4.4b [4b] Confidence-Gated Precision Fallback **[v2, new — the novelty claim; see E7]**

Store each expert in CXL at two precisions (FP16 + NF4, +25% capacity). For a predicted candidate with calibrated probability p: fetch FP16 if p ≥ τ_hi, NF4 if τ_lo ≤ p < τ_hi, skip if p < τ_lo. A misprediction then costs bounded quality (a low-precision expert was already resident) instead of a multi-millisecond demand stall.

Why this is the right response to the pilot: recall of ~60% at m=2k means the predictor is *right often enough to be useful but wrong often enough to waste bandwidth*. Precision gating converts that error rate from a latency tax into a quality tax, and quality taxes are measurable and tunable. Prerequisite: calibrated probabilities (E4.4) — uncalibrated confidences make τ meaningless. Closest prior work is HOBBIT, which does mixed precision *reactively on miss*; ours is predictive and calibrated, and targets a byte-addressable tier rather than PCIe copies. Quality is validated by real inference runs (ΔPPL on WikiText-2, Δaccuracy on a benchmark slice), not asserted.

### 4.5 [5] Tiered-Memory Simulator

Discrete-event simulator (Python + SimPy, or hand-rolled event loop; port hot loop to C++/pybind11 only if profiling demands it).

**Simulated timeline per decode step:** for each layer: attention compute (fixed cost from calibration) → router → expert residency check → per selected expert: if resident in HBM, compute (calibrated per-expert GEMM time); else stall for fetch (size / effective_tier_BW + tier latency), with in-flight prefetches completing asynchronously and contending for link bandwidth. Track: TPOT distribution (mean, p50, p99), TTFT, expert hit rate, bytes moved/token, stall-time breakdown, link utilization.

**Tier parameter table (defaults; every number sourced in the report):**

| Tier | Capacity (default sweep) | BW | Added latency | Source |
| --- | --- | --- | --- | --- |
| HBM3 (H100 SXM) | 80 GB (sweep 24–96) | 3.35 TB/s | ~0 (reference) | NVIDIA datasheet |
| Local DDR5 (8-ch) | 512 GB | ~307 GB/s | ~100 ns | JEDEC/vendor |
| CXL 2.0 ×8 (Leo-class) | 512 GB–2 TB | ~28–32 GB/s/dir usable | +170–250 ns | Sun et al. ISCA'23; Astera Leo product brief |
| PCIe 4.0 ×16 (GPU link) | — | ~25 GB/s effective | ~µs-scale per transfer | FloE measurements |

**DRAMSim3 integration:** drive DRAMSim3 with the simulator's fetch-request stream for the DDR-backed CXL device to derive *effective* bandwidth under the real access pattern (large sequential expert reads → near-peak; validates the analytical BW assumption and checks the organizers' tool box).

**Calibration & validation (three levels — this is the credibility ladder):**

1. Per-expert GPU compute times measured once on a real GPU (Kaggle T4 / Colab) via CUDA events; MPS timings on Mac as sanity check.
2. PCIe transfer ground truth: measure host→GPU expert-tensor copy times on Kaggle; check against FloE's published ~15 ms / 300 MB on PCIe 4.0.
3. If dual-socket access materializes: run real Mixtral offloading with expert tensors pinned to the remote NUMA node (`numactl --membind`) under `none` vs `static-hot` vs `next-layer-topk` policies; compare measured vs simulated deltas. Target: simulator within ±15% on *relative* policy ordering (absolute error matters less than preserving the ranking — say this explicitly).

### 4.6 [6] Pooling & TCO Model

Analytical model (NumPy notebook), following Pond's stranding logic applied to an MoE serving fleet:

- Fleet of N hosts, each serving heterogeneous MoE models with time-varying demand; per-host memory demand distribution derived from trace-based footprints.
- Unpooled: each host provisions for its own p99 demand → stranded GB = provisioned − used.
- Pooled (CXL 2.0 switch, pool of P GB shared by k hosts): provision for the *pooled* p99. Output: stranded-capacity reduction %, and cost curves.
- Cost inputs (state assumptions explicitly, cite where possible): HBM ≥3×
 DDR5 $/GB (Amplifying-CXL-BW paper, arXiv:2509.03377); CXL device = DDR5 + controller amortization; sweep the ratio 2–5× so conclusions are robust to price uncertainty.
- Headline metric: **tokens/sec/$ of memory** for HBM-only vs HBM+CXL vs HBM+pooled-CXL at iso-throughput SLO (p99 TPOT ≤ threshold).

### 4.7 [7] Dashboard

Streamlit + Plotly app, three tabs:

1. **Workload explorer** — pick model/workload, see live expert-heatmap animation over token time (visually stunning; this is the screenshot everyone remembers).
2. **Design-space explorer** — sliders: HBM GB, CXL BW/latency, prefetch policy, batch size → simulator (pre-computed grid + interpolation for instant response; run live for custom points) → TPOT/p99/hit-rate/cost gauges.
3. **Fleet/TCO** — pooling model curves with cost-ratio slider.

Pre-compute the sweep grid so the demo never blocks. Record a backup screen-capture video in case of demo-day failure.

### 4.8 [V1] QEMU CXL software-stack demo (no CXL hardware needed)

QEMU ≥ 7.1 emulates CXL Type-3 memory devices. Boot a Linux 6.x guest with a `cxl-type3` device; inside the guest: `cxl create-region`, bind via DAX/kmem so the CXL memory appears as a NUMA node; demonstrate `numactl` weighted interleave and allocate expert tensors into the CXL node from PyTorch (via `numactl --membind` wrapping a CPU inference of OLMoE). **Functional, not timing-accurate — say so on the slide.** Purpose: proves the team understands the *real* Linux CXL stack end-to-end (region creation, dax, memory-tier sysfs), which almost no student team will show. 30–60 s of the final demo.

---

## 5. Execution Plan **[v2 — Phases 0–1 complete; see `experiment_roadmap.md` for per-experiment specs]**

**Completed.** Phase 0 (abstract, repo, hardware access) and the decision-critical subset of Phase 1: collector built and hardened, 250 requests × 2 models traced with a request-level 70/30 split, analyzer producing Coverage/Gini/Zipf/churn/Recall/HTB with bootstrap CIs, pre-registered thresholds evaluated → **GO, prefetch-led**. Known engineering gotcha now baked into the repo: `transformers` 5.x fuses Mixtral experts into a 3D-parameter `MixtralExperts` module that `bitsandbytes` silently declines to quantize; pin `transformers==4.46.3`.

**Remaining program — experiments E1–E13, specified in `experiment_roadmap.md`.** Summary schedule:

| Window | Primary (you) | Secondary (teammates / cut-list if solo) | Gate |
| --- | --- | --- | --- |
| Aug 14–18 | **E1 roofline**, E3 calibration session | E2 batching curves | Regime boundary stated with numbers |
| Aug 19–25 | **E6 simulator core**, E4 predictor v2 | E11 extra models, E5 depth study | Simulator passes validation gates |
| Aug 26–Sept 1 | E6 sweeps, **E7 precision gating** | E8 KV co-tenancy, E9 TCO | Headline figure exists |
| Sept 2–8 | E7 quality runs, E10 QEMU + PCIe validation | Dashboard build | Sim reproduces measured PCIe ordering ±15% |
| Sept 9–12 | E12 ablations, **report freeze Sept 12** | Figures polish | Every figure regenerable by `make figures` |
| Sept 13–15 | Buffer, **final submission Sept 15** | — | Submitted |
| Sept 16–25 | Slides, demo script, Q&A bank, 2 rehearsals | Backup demo video | Rehearsed under time |

**Priority order if behind:** protect E1 → E6 → E3 → E7 → E4 → E13. Cut in reverse: E8 first, then E11, then E5 (keep d=1), then E9 (analytical appendix only). E7 is cut only as a last resort — it is the novelty crown.

**Deliverable coverage against the competition brief** (unchanged, all still met):

| Brief item | Where it is now satisfied |
| --- | --- |
| HBM + CXL memory architecture model | E1 roofline + E6 simulator tier models |
| Transformer & MoE workload analysis | Pilot (done) + E2 + E11 |
| Bandwidth, latency, scalability evaluation for CXL | E1, E5, E6 sweeps |
| Memory tiering & expert placement strategy | §4.3 fixed-residency + §4.4 prefetch (with the *negative* placement result reported) |
| CXL memory expansion & pooling study | E9 (+ E8 co-tenancy) |
| Simulation framework: HBM-only vs HBM+CXL | E6 |
| Interactive demo / dashboard | E13 (roofline explorer is the centerpiece tab) |
| Final report with architecture recommendations | E13 + E1's provisioning table |

## 6. Metrics Dictionary (use these terms consistently everywhere)

| Metric | Definition |
| --- | --- |
| TTFT | Time to first token (prefill-dominated) |
| TPOT | Time per output token, decode steady state; report mean/p50/p99 |
| Expert hit rate | Fraction of expert activations served from HBM without stall |
| Stall time % | Σ stall / Σ step time, decomposed by cause (demand fetch, prefetch contention) |
| Bytes/token | Data moved across the CXL link per generated token |
| Oracle gap closed | (policy − none) / (oracle − none) on TPOT |
| tokens/sec/$mem | Throughput per dollar of memory BOM at iso-p99-TPOT SLO |
| Stranding reduction | (unpooled provisioned − pooled provisioned) / unpooled, at iso-p99 fleet demand |
| **Regime ratio ρ** *[v2]* | per-layer transfer time / per-layer compute time. ρ<1 latency-bound (prefetch helps), ρ≥1 bandwidth-bound (only byte reduction helps) |
| **Coverage@C** *[v2]* | fraction of held-out activations served by the top-C% experts ranked on the training split (deployable, not oracle, skew) |
| **Recall@m, depth d** *[v2]* | fraction of actual layer-ℓ+d experts appearing in the predictor's top-m, restricted to non-resident experts; cap m at n_experts |
| **Prefetch accuracy / coverage / timeliness** *[v2]* | standard prefetcher triple: fraction of prefetches used; fraction of needed fetches prefetched; fraction arriving before use |
| **HTB** *[v2]* | Coverage@C + (1−Coverage@C)·Recall@m — a *bound*, not an achieved result; the simulator reports what a real policy attains against it |
| **ΔPPL / Δacc** *[v2]* | quality cost of precision-gated fetching, vs FP16-everywhere |

---

## 7. Final Assembly

### 7.1 Repository structure

```
tierahead/
├── README.md            # 90-second pitch + reproduce-everything instructions
├── pyproject.toml       # locked deps (uv)
├── tierahead/
│   ├── collect/         # hooks, runners (cuda / mps / cpu)
│   ├── analyze/         # characterization
│   ├── place/           # greedy, ILP, migration
│   ├── policy/          # none | lru | static | next_layer | mlp | oracle
│   ├── sim/             # DES core, tier models, link contention
│   ├── dramsim/         # DRAMSim3 bridge (optional build)
│   └── tco/             # pooling & cost model
├── traces/  calib/  results/  figures/
├── dashboard/           # streamlit app + precomputed grid
├── demo/qemu-cxl/       # scripts + README + recording
├── report/              # LaTeX (IEEEtran)
├── slides/
├── tests/
└── Makefile             # make traces|sweep|figures|report
```

### 7.2 Final report outline (IEEE two-column, 10–14 pp — write it at the standard you review at)

1. Abstract · 2. Introduction (memory wall → MoE sparsity → CXL tier → **the prefetch-vs-bandwidth reframe** → thesis) · 3. Background & Related Work (§2 condensed; explicit delta vs arXiv:2512.04476, HOBBIT, fMoE, Pre-gated MoE) · 4. **MoE Routing Characterization** (pilot results R1–R4, incl. the skew-inversion finding and why load-balancing loss explains it; E11 extension) · 5. **The MoE Tiering Roofline** (E1, E2 — the paper's conceptual contribution) · 6. TierAhead Design (fixed residency, prefetch engine, precision gating) · 7. Methodology & Validity (calibration ladder, parameter-provenance table, PCIe/QEMU validation, limitations stated plainly) · 8. Evaluation (RQ-by-RQ; headline stall-vs-residency figure; policy bake-off; Pareto) · 9. KV-Cache Co-Tenancy (E8) · 10. CXL Pooling & TCO (E9) · 11. Ablations & Sensitivity (E12) · 12. **Architecture Recommendations for CXL Memory-Controller Deployments** (provisioning table: GB/s and residency % per model class and batch; where near-data processing would and would not help) — *this section is the internship pitch in writing* · 13. Limitations & Future Work · 14. Conclusion · References.

### 7.3 Slide deck (≤12)

1 Title/team → 2 The MoE memory wall (87 GB vs 80 GB, one picture) → 3 Why CXL is the right tier (and why host-DRAM copy isn't) → 4 Insight A: *the router tells you the future* (pilot numbers: 55–62% recall, +14–25 pp over popularity) → 5 **Insight B: prefetch hides latency, it doesn't create bandwidth** — the 22 ms vs 0.2 ms slide; *this is the slide no other team will have* → 6 The roofline + where each model class lands → 7 Characterization incl. skew inversion → 8 TierAhead design (residency + prefetch + precision gate) → 9 Headline result (stall rate & TPOT vs residency, oracle gap closed) → 10 Validation (PCIe measured ordering + QEMU CXL stack) → 11 Economics + **live demo (roofline explorer)** → 12 Provisioning recommendations for CXL controller deployments + "what we'd do with real Astera hardware" (the internship slide).

### 7.4 Demo script (3 min)

0:00 tab 1 — expert heatmap animating on a real ShareGPT trace ("watch how few experts light up — and note this is the *fine-grained* model; the coarse one lights up almost uniformly") → 0:40 **tab 2, roofline explorer** — drag expert size from 12 MB to 352 MB and watch the operating point cross the ρ=1 line into the red; then raise batch and watch it cross back ("this is the whole design space in one picture") → 1:30 tab 3 — drop residency to 25%, flip `none` → `prefetch-v2` → `+precision-gate`, stall-rate gauge falls at each step → 2:10 30-s QEMU clip — real Linux `cxl create-region`, expert tensors bound to the CXL NUMA node → 2:35 tab 4 — cost slider, "same throughput, X% less memory spend" → close.

---

## 8. Hardware Feasibility: what runs where (no CXL access assumed)

**[v2] Known hardware:** RTX 4500 Ada (24 GB) — used for OLMoE FP16; 2× RTX 4060 Ti (16 GB each, 32 GB pooled) — used for Mixtral NF4. Both suffice for E3 calibration, E7 quality runs, and E11 (DeepSeek-MoE-16B NF4 ≈ 9 GB, Qwen1.5-MoE NF4 ≈ 8 GB both fit comfortably on the 4500 Ada). No further GPU procurement is needed for the whole program.

### 8.1 Runs fully on your Mac (Apple Silicon, macOS)

- **All framework code:** **roofline model (E1)**, batching analysis (E2), simulator (E6), residency policy, predictor training and evaluation (E4, E5), precision-gate policy logic, TCO model (E9), KV co-tenancy model (E8), dashboard, figures, report, tests. The full sweep grid is CPU-only and overnight-able on a laptop. **E1 and E2 — the two highest-priority experiments — need no GPU at all**, so they can start immediately regardless of cluster queue.
- **DRAMSim3:** plain CMake C++; builds on macOS/clang.
- **MLP predictor training:** minutes on CPU/MPS.
- **Trace collection for small MoE:** OLMoE-1B-7B in FP16 (~14 GB weights) runs via PyTorch **MPS** on a 24 GB+ Mac (slow decode is fine — traces don't need speed). On 16 GB Macs use `device_map` with disk offload (very slow but workable for tens of prompts) or move this to Colab. Qwen1.5-MoE/DeepSeek-MoE-16B (~28–33 GB FP16) need a 48–64 GB Mac or the cloud path. Note: **bitsandbytes 4-bit does not support MPS** — quantized runs happen on CUDA only; llama.cpp/Metal runs Mixtral GGUF fast but does not expose router logits without source modification, so do not plan traces through it.
- **QEMU CXL demo:** possible on a Mac via x86_64 emulation (UTM/QEMU, no KVM → slow boot but the demo is functional). Smoother path: any cheap x86 Linux VM (§8.3).

### 8.2 Needs CUDA (all free-tier accessible)

| Task | Where | Notes |
| --- | --- | --- |
| Mixtral-8x7B traces | ✅ **done** (2× 4060 Ti, NF4, `transformers==4.46.3`) | — |
| OLMoE traces | ✅ **done** (RTX 4500 Ada, FP16) | — |
| **E3 calibration**: per-expert GEMM times, layer compute at batch {1,8,32}, pinned H2D bandwidth/latency fit | 4500 Ada + 4060 Ti box | one session, ~4 h; **highest-value GPU work remaining** |
| **E3.4 NF4 routing-fidelity check** on OLMoE (FP16 vs NF4 top-k agreement) | 4500 Ada (OLMoE fits FP16 in 24 GB) | closes pilot open item #2 |
| **E7 quality runs** (ΔPPL WikiText-2, Δacc on a benchmark slice, precision-gated substitution) | 4500 Ada | ~6 h |
| **E10 V3** end-to-end PCIe offloading ground truth, 3 policies | 4060 Ti box (small VRAM = realistic offloading pressure, which is an advantage here) | the validation section |
| **E11** DeepSeek-MoE-16B, Qwen1.5-MoE traces (NF4) | 4500 Ada | ~8 h |
| Mixtral batch-8 erosion (pilot open item #1) | 4060 Ti box | already queued; feeds E2 |

### 8.3 Needs x86 Linux (not macOS, not CUDA)

- **QEMU CXL Type-3 (V1):** any x86_64 Linux with KVM — a lab desktop, or a small cloud VM for a few dollars total. Kernel 6.x guest + `ndctl/cxl-cli`.
- **NUMA-remote emulation (V2):** dual-socket bare metal only — DaSH Lab request; **treat as stretch**, the plan stands without it (credibility comes from the V3 PCIe runs + calibration ladder instead).

### 8.4 The "no CXL hardware" narrative for judges (rehearse this)

"No student team has CXL silicon, so we built a three-level evidence ladder: (1) **real traces** from real MoE models define the workload; (2) a **simulator calibrated against measured GPU compute and PCIe transfer times** and DRAMSim3 device timing, using CXL latency/bandwidth from ISCA'23 measurements of true CXL hardware and Astera's Leo specifications, evaluates the design space; (3) **real offloading runs over PCIe** — the same physical layer CXL uses — validate that our policy ordering holds on hardware, and a **QEMU-emulated CXL device running the production Linux CXL stack** proves software feasibility end-to-end. Every number in our report is either measured or carries a citation." — Delivered confidently, this turns your biggest weakness into a methodology strength.

**[v2] Add one sentence to the end of that narrative:** *"And because our central claim is a roofline — a ratio of transfer time to compute time — it is robust to the exact CXL latency: we sweep 150 to 400 ns and the regime boundary barely moves, because bandwidth, not latency, is what decides it."* This pre-empts the sharpest possible attack on a simulation-based submission: it says your conclusion does not depend on the parameter you could not measure.

---

## 9. Risk Register **[v2]**

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| **Bandwidth wall makes the whole idea look infeasible for Mixtral-class models** | high (it is real) | Do not hide it — *lead* with it. The roofline turns the problem into the contribution; precision gating (E7) is the mechanism that pushes the boundary. A team that quantifies where their own idea stops working outranks one that doesn't notice |
| Precision gating degrades quality unacceptably | med | Report the Pareto frontier as the result; the bound is publishable either way |
| Predictor v2 fails to beat the frequency table | med | Negative result + "the selected-expert set already carries nearly all the signal" is a clean finding; frequency table ships |
| gem5 rabbit hole | high if attempted | Pre-decided: trace-driven DES + DRAMSim3 only |
| Simulator disbelieved (no CXL hardware) | high | Three-artifact defense (E10) + parameter-provenance table; rehearse the §8.4 narrative |
| Batching erodes skew → scope attack in Q&A | certain to be asked | E2 answers it *quantitatively* and turns it into the B\* result; pre-empt on slide 6 rather than defending in Q&A |
| “You only tested 2 models” | certain | E11 extends to 4; until then, state n=2 explicitly and note the CIs |
| NF4 fidelity unmeasured for Mixtral (pilot open item #2) | resolved-ish | E3.4 measures FP16-vs-NF4 routing agreement on OLMoE (which fits), cites literature for Mixtral, and caveats it |
| Single seed / n=250 (pilot open item #3) | med | E12 reruns the pipeline with a second sample and seed; bootstrap CIs already reported |
| Solo execution | — | Cut order in §5; protect E1 → E6 → E3 → E7 |
| Scooped vibe from arXiv:2512.04476 | — | Differentiation is now *stronger*: they add NDP hardware, we characterize the regime for commodity expanders and reduce bytes in software. Cite them first, on your own terms |

## 10. Effort Budget

**[v2]** ~185–215 person-hours remaining across 6 weeks. Per-experiment estimates: E1 8–12 h · E2 8 h · E3 10 h · E4 12–15 h · E5 10 h · **E6 30–35 h** · E7 18–22 h · E8 10 h · E9 10 h · E10 12 h · E11 12 h · E12 12 h · E13 30 h. Front-load E1/E3/E6; the last 10 days are polish only. Note that demoting the ILP optimizer (v1 §4.3) freed roughly 15 h, which is exactly what E7 costs — the pivot is effort-neutral.

## 11. References (for the report bibliography)

1. Jiang et al., *Mixtral of Experts*, arXiv:2401.04088.
2. Eliseev & Mazur, *Fast Inference of MoE LMs with Offloading*, arXiv:2312.17238.
3. Xue et al., *MoE-Infinity*, arXiv:2401.14361.
4. Hwang et al., *Pre-gated MoE*, ISCA 2024, arXiv:2308.12066.
5. Yu et al., *fMoE: Fine-Grained Expert Offloading*, arXiv:2502.05370.
6. Tang et al., *HOBBIT: Mixed-Precision Expert Offloading*, arXiv:2411.01433.
7. *Klotski*, arXiv:2502.06888; *FloE*, arXiv:2505.05950.
8. Du et al., *SiDA-MoE*, MLSys 2024, arXiv:2310.18859.
9. Fan et al., *Context-Aware MoE Inference on CXL-Enabled GPU-NDP Systems*, arXiv:2512.04476.
10. Li et al., *Pond: CXL-Based Memory Pooling for Cloud Platforms*, ASPLOS 2023, arXiv:2203.00241.
11. Maruf et al., *TPP: Transparent Page Placement for CXL Tiered Memory*, ASPLOS 2023, arXiv:2206.02878.
12. Sun et al., *Demystifying CXL Memory with True CXL-Ready Systems*, ISCA 2023.
13. Ji et al., *Demystifying a CXL Type-2 Device*, MICRO 2024.
14. Lee et al., *Memtis*, SOSP 2023.
15. Tang et al. (ByteDance), *Exploring CXL-based KV Cache Storage for LLM Serving*, NeurIPS 2024 ML-for-Systems workshop.
16. *TraCT: Disaggregated LLM Serving with CXL Shared-Memory KV Cache*, arXiv:2512.18194.
17. *Amplifying Effective CXL Memory Bandwidth for LLM Inference*, arXiv:2509.03377.
18. Samsung Semiconductor, *Breaking AI Memory Limits with CXL Memory Pooling* (white paper, 2026).
19. Astera Labs, *Breaking Through the Memory Wall: How CXL Transforms RAG and KV Cache Performance* (Leo blog).
20. Li et al., *DRAMSim3*, IEEE CAL 2020.
21. CXL Consortium, *CXL 2.0 / 3.x Specifications*.
21a. Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model*, CACM 2009. **[v2 — the model §5 of the report adapts]**
21b. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention (vLLM)*, SOSP 2023. **[v2]**
21c. Agrawal et al., *Sarathi-Serve*, OSDI 2024; Yu et al., *Orca*, OSDI 2022. **[v2 — batching]**
21d. Dettmers et al., *QLoRA/NF4*, arXiv:2305.14314; Frantar et al., *GPTQ*, arXiv:2210.17323; Frantar & Alistarh, *QMoE*, arXiv:2310.16795. **[v2 — precision gating]**
21e. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. **[v2 — τ thresholds require calibrated probabilities]**
21f. Shazeer et al., *Outrageously Large Neural Networks*, ICLR 2017; Fedus et al., *Switch Transformer*, JMLR 2022. **[v2 — load-balancing loss, the mechanism behind the skew-inversion result]**
22. Muennighoff et al., *OLMoE*, arXiv:2409.02060; DeepSeek-AI, *DeepSeekMoE*, arXiv:2401.06066; Qwen team, *Qwen1.5-MoE* blog.

---

## 12. Changelog: v1 → v2 (post-pilot) **[v2]**

| Area | v1 | v2 | Driver |
| --- | --- | --- | --- |
| System shape | Placement optimizer + prefetch, co-equal | **Prefetch-primary**; fixed-residency placement | Coverage@25% 52.2% / 28.7% — placement does not generalize |
| Central contribution | Router-guided tiering | **MoE Tiering Roofline** + router-guided prefetch + precision gating | Bandwidth calc: 22 ms transfer vs 0.2 ms compute per layer |
| Placement ILP | Full multiple-knapsack formulation | One-paragraph appendix showing headroom is small | Coverage@50% caps at 74.1% / 54.5% |
| Novelty item | Router-guided prefetch | **Confidence-gated precision fallback** | Recall ~60% at m=2k is a 2× byte cost — unaffordable per roofline |
| H1 (skew) | Assumed; coarse models predicted *more* skewed | **Refuted and inverted**; now reported as finding R3 with a load-balancing-loss explanation | Gini 0.378 (64 exp) vs 0.066 (8 exp) |
| H2 (predictability) | Hypothesis | **Confirmed** (R2), cross-architecture | 55.3% / 61.8% non-resident recall |
| Model roster | 4 models, all required | 2 done; 2 more are E11 upside | Core decision no longer depends on them |
| Batch analysis | One erosion check | **E2: batching as a tiering knob, with B\*** | Batch is the lever that moves ρ |
| New experiments | — | E1 roofline, E2 batch sweet spot, E7 precision gating, E8 KV co-tenancy | Follow from the reframe |
| Presentation | Insight = "router tells you the future" | Two insights; **slide 5 is the bandwidth reframe** | It is the thing no other team will say |
