# TierAhead — Project Status & Deep Dive

Let me build this up from first principles, since the pilot result only makes sense once you see the shape of the underlying problem.

## 1. The basics: why does this project exist at all?

**Mixture-of-Experts (MoE) models** are a trick for making huge models cheap to *run*. Instead of one giant neural network that processes every token, you build many smaller sub-networks ("experts") per layer, and a small "router" network looks at each token and picks only a handful of experts (say, top-2 out of 8, or top-8 out of 64) to actually process it. Compute per token stays small. But here's the catch: **all the experts still have to exist somewhere in memory**, because you don't know in advance which ones a given token will need. Mixtral-8x7B has 87GB of total parameters in FP16, but an H100 GPU only has ~80GB of ultra-fast HBM. So the model's *storage footprint* has outgrown the *fast memory* even though its *compute* per token stayed cheap.

**CXL (Compute Express Link)** is a hardware answer to "I need more memory than fits on my accelerator, but SSD is too slow." It's a link (running over the PCIe physical wire) that lets you attach ordinary DDR5 memory to a system with normal load/store instructions — no manual DMA copying — at roughly +170–250ns extra latency and tens of GB/s of bandwidth. It's slower than local HBM/DDR but vastly faster than disk, cheaper per GB than HBM, and — uniquely — it can be *pooled* across multiple machines. Astera Labs (the competition sponsor) sells the controller chips (Leo) that make this work, and they market it specifically as the fix for the "LLM memory wall."

**The project's actual bet:** if you treat CXL as a second, larger, slower memory tier underneath HBM, and you're smart about *which* expert weights live where and *when* you fetch them, you can serve models way bigger than HBM would normally allow — without constantly stalling the GPU waiting for data. "Smart" here specifically means exploiting the fact that the MoE router's own decisions are a signal: the gate at layer *l* tells you something about what's likely to be needed at layer *l+1*, before you actually need it.

That system is called **TierAhead**, and it's your entry for the Astera Labs "CXL-Based Memory Optimization for MoE Models" track at Nebula@BITS Goa 2026.

## 2. Where the project stands right now (Aug 31, 2026)

- Abstract phase: done, shortlisted.
- **Pilot experiment (ELP-Probe): complete.** Verdict: **GO, prefetch-led.**
- The plan was substantially rewritten (v1 → v2) on Aug 14 as a direct result of the pilot.
- Per the v2 schedule, you're currently sitting in the **Aug 26–Sept 1 window**, which is supposed to be "E6 simulator sweeps + E7 precision gating (the novelty item)" as primary, with E8/E9 as secondary. Final report freeze is Sept 12, submission Sept 15, live presentation Sept 25.
- I don't see artifacts in these files confirming E1 (roofline) or E6 (simulator) are actually finished yet — only that the pilot's numbers exist and the plan assumes E1/E3 were tackled Aug 14–18. Worth checking your own repo against the exit criteria in `Experiments.md` §E1/E3/E6 to see if you're actually on schedule, since that's the critical path (E1 → E3 → E6 → E13).

## 3. What the pilot (ELP-Probe) actually tested, and why

Before building any of the eight TierAhead components, you needed to answer one falsifiable question: **is MoE expert traffic structured enough to be worth tiering at all?** Everything else — the placement optimizer, the prefetcher, the simulator — is engineering that only pays off if the answer is yes. So the pilot deliberately measured *properties of the workload*, not performance of any system you'd built yet (because no system existed yet).

Two real models were traced — **OLMoE-1B-7B** (64 experts/layer, top-8: the "fine-grained" end) and **Mixtral-8x7B** (8 experts/layer, top-2: the "coarse-grained" end) — over 250 real chat and code requests, logging every router decision at every layer. Requests were split 70/30 train/test so nothing was measured on data the statistics were fit to (avoiding leakage).

Three quantities were computed, each mapping to one candidate mechanism:

| What was measured | What it tells you | Which mechanism it validates |
| --- | --- | --- |
| **Coverage@25%** — if you permanently pin the busiest 25% of experts in fast memory, what % of *unseen* traffic hits them? | Is expert usage skewed enough that "put the popular ones in HBM and forget it" works? | Static placement |
| **Hot-set churn** | Does that popular set stay stable over time, or drift? | How often you'd need to re-pin |
| **Recall@m, d=1** — using only the router's current-layer decision, how often does a simple predictor correctly guess next-layer's experts? | Is the *next* step predictable enough to start fetching early? | Router-guided prefetch |

## 4. The results, and what they mean in plain terms

| | OLMoE (fine-grained) | Mixtral (coarse-grained) |
| --- | --- | --- |
| Coverage@25% (static placement) | 52.2% | 28.7% |
| Skew (Gini) | 0.378 (real skew) | 0.066 (almost flat) |
| Recall@2k, d=1 (prefetch) | 55.3% | 61.8% |
| Verdict | borderline / PIVOT on its own | clears the GO-prefetch-led bar |

Two findings fell out of this, and they pull in different directions:

**Finding A — static placement doesn't generalize.** "Just cache the popular experts in HBM" works okay-ish for OLMoE but is genuinely weak for Mixtral — only 29% of traffic hits your pinned top-quarter. Worse, it was backwards from the original hypothesis: the team expected the *coarse* model (fewer, bigger experts) to be *more* skewed, since intuitively fewer slots should mean more specialization. The opposite happened. The likely mechanism: MoE training uses a load-balancing loss that pushes every expert toward an equal share of tokens. With only 8 experts, that constraint is easy to enforce almost exactly — there's no room for any expert to drift far from 1/8. With 64 experts, the same aggregate constraint (average ≈ 1/64) leaves plenty of room for individual experts to specialize while still balancing on average. So *fewer, bigger experts get squeezed toward uniform; more, smaller experts get room to specialize* — the reverse of the naive intuition. This is a genuinely citable result on its own.

**Finding B — router-guided prefetch does generalize.** Both models comfortably beat a "just guess the globally popular experts" baseline by 14–25 percentage points, and both land around 55–62% recall for predicting next-layer experts from current-layer router state — on the *held-out, non-resident* set, which is the hard case. This held on both ends of the granularity spectrum, which is the thing that actually justifies building a system around it.

**Consequence for the plan (this is the v1→v2 pivot):** the original design treated a cost-optimized placement algorithm (an ILP solver deciding exactly which experts go where) as co-equal with prefetching. The pilot data says that's not where the leverage is — even a *perfect* placement solver tops out around 74%/54% coverage at 50% residency, so there isn't much headroom to buy with exact optimization. That whole component got demoted to a one-line fixed-residency policy, freeing up real engineering time, and **prefetch became the system's centerpiece.**

## 5. Deep dive: why this is more than "prefetching works" — the roofline

Here's where it gets genuinely interesting, and this is the part I'd flag as the actual differentiator. This deep dive follows the algorithmic → hardware → co-design structure since that's exactly the shape of the reasoning that produced it.

**a) Algorithmic bound.** Recall@2k=55–62% sounds like a win. But look at what "m=2k" costs: fetching the top-2k candidate experts to catch top-k actual ones means you're pulling **twice the bytes you actually need** for a partial hit rate. That's a real cost, not free insurance.

**b) Hardware constraint.** Do the bandwidth arithmetic per layer. Mixtral's experts are ~352MB each; top-2 routing moves ~705MB/layer if non-resident. At a realistic CXL 2.0 ×8 link (~32GB/s), that's **~22ms just to transfer one layer's experts**. Compute for that same layer, at batch 1, is roughly **0.1–0.3ms**. That's a 70–200× mismatch. One layer of lookahead buys you roughly one layer's worth of compute time to hide a fetch behind — call it fractions of a millisecond. You cannot hide 22ms of transfer behind 0.2ms of compute, no matter how good your predictor is. OLMoE is much friendlier (experts are ~12.6MB, transfer ≈3.15ms/layer vs ≈0.1ms compute) but the mismatch is still an order of magnitude.

**c) Co-design trade-off — the roofline.** This is the key reframe: **prefetching hides *latency*, it does not reduce *bytes moved*.** A perfect predictor issues the fetch earlier, but the fetch is still the same size and the link is still the same speed. So whether CXL tiering can work at all is *not* set by how accurate your predictor is — it's set by a ratio:

ρ = (bytes to transfer per layer) / (compute time available per layer)

If ρ < 1, you're latency-bound — prefetch fully hides the fetch and the system works great. If ρ ≥ 1, you're bandwidth-bound — no amount of prediction accuracy helps, because you literally cannot push that many bytes through the pipe in the time you have. This is a direct application of the classic **Roofline model** from computer architecture (Williams, Waterman, Patterson 2009), just adapted to expert-bytes-per-layer instead of FLOPs-per-byte.

This single insight does three things at once, which is why it's the "game-changing" piece rather than an incremental result:

1. **It resolves the pilot's apparent contradiction instead of hand-waving it.** OLMoE has real skew but middling recall margin; Mixtral has near-zero skew but the strongest recall. Under the roofline, that's not two unrelated facts — it's two points on the same curve, one sitting closer to the compute-bound side, one sitting deep in the bandwidth-bound side. Every "weird" pilot number becomes an expected data point instead of a caveat to explain away.

2. **It's the exact deliverable a memory-controller vendor wants.** Astera doesn't want "our predictor got 61% recall." They want "for expert size ≤30MB at batch ≥8 with ≥50% residency on a ≥64GB/s link, tiering works — below that line, you need byte reduction, not smarter prediction." That's a **provisioning rule**, i.e. exactly the kind of statement that goes on a spec sheet or a sales deck for Leo-class controllers. It reframes a student benchmark into an architecture-recommendation result.

3. **It directly motivates the new novelty mechanism: confidence-gated precision fallback.** Once you know you're on the bandwidth-bound side of the line (coarse experts, low batch — Mixtral's actual regime), the only lever left is *reducing bytes*, not hiding latency better. So: fetch high-confidence predicted experts at full FP16, fetch low-confidence ones at 4-bit (NF4, ~4× smaller), and skip the rest. A misprediction now costs a bounded, *measurable* quality hit (ΔPPL) instead of a multi-millisecond hard stall. This is a genuinely different failure mode than anything in prior work — HOBBIT does mixed-precision reactively, only after a cache miss already happened; this does it predictively and pre-calibrated, which is only possible because the roofline told you *when* you need it. No prior MoE-offloading paper (MoE-Infinity, fMoE, HOBBIT, Klotski) states this viability boundary explicitly — they report speedups without characterizing where the mechanism stops working. Stating that boundary, and building the mechanism for the far side of it, is the actual scientific contribution of the project now, not "we made prefetch."

**Edge case worth flagging explicitly** (this is where the co-design trade-off gets tested): under multi-tenant CXL pooling or KV-cache co-tenancy on the same link (E8), prefetch traffic and demand fetches — and now also KV spillover — all contend for the same finite bandwidth. A misprediction under precision-gating doesn't just cost quality anymore; a *burst* of low-confidence NF4 fetches can itself saturate the link and starve a demand fetch that's on the critical path. That's the kind of interaction the simulator's shared-bandwidth link model (E6) is specifically built to catch, and it's worth stress-testing once E6 exists — a policy that looks good in isolation can still create a contention failure mode once several tenants share one Leo-class link.

**Bottom line:** the pilot didn't just validate an assumption — it forced a real scientific pivot, from "which mechanism wins" to "when does either mechanism even apply," which is a stronger, more defensible, more Astera-relevant claim than the original plan had. That reframe (slide 5 in your deck) is explicitly called out in your own docs as "the slide no other team will have," and the pilot data is exactly what earned it.
