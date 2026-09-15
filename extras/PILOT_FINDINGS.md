# ELP-Probe Pilot: What We Found and What It Means for TierAhead

Companion doc to `Pilot.md` (the experiment design) and the raw output in `results/`. This file is the human-readable "so what" layer: what we measured, what it means, what's shaky about it, and how it should change the plan going forward. If you only read one section, read "The one-sentence version" and "What this changes about the system."

---

## 1. The one-sentence version

Static HBM placement of experts is weak and inconsistent across the two MoE architectures we tested, but router-guided one-layer-ahead prefetch is real, beats baselines by a meaningful margin, and holds up on both a fine-grained (OLMoE, 64 experts/layer) and a coarse-grained (Mixtral, 8 experts/layer) model. That means the system worth building is a **prefetch-primary** one, not the placement-and-prefetch co-equal design originally sketched in the abstract.

---

## 2. What we actually did

Per `Pilot.md`'s design: for each model, we hooked the router's gate module in every decoder layer, ran 250 real requests (150 ShareGPT chat prompts, 100 HumanEval/MBPP code prompts) through the model one token at a time, and logged which experts got selected at every layer for every decode step. We split requests 70/30 into train and test *before* computing anything, so every number below is measured on held-out requests the statistics weren't fit on. That split matters: without it, the coverage and recall numbers would be inflated by leakage and wouldn't mean anything for a deployed system.

Two models were traced:

- **OLMoE-1B-7B**, FP16, on a single RTX 4500 Ada (24GB). 64 experts per layer, top-8 routing. This is the fine-grained end of the MoE design space.
- **Mixtral-8x7B**, NF4 quantized, on a 2x RTX 4060 Ti (16GB each, 32GB pooled on one node) box. 8 experts per layer, top-2 routing. This is the coarse-grained end, and it's also the model the abstract's core memory-footprint argument (87GB FP16, 340MB/expert) is actually about.

Getting Mixtral running took real debugging, worth noting because it affects how much to trust the numbers: current `transformers` (5.x) refactored Mixtral's experts into a fused `MixtralExperts` module holding raw 3D `nn.Parameter` tensors instead of per-expert `nn.Linear` layers. `bitsandbytes`' `load_in_4bit` only auto-quantizes `nn.Linear` submodules, so against the fused module it was quantizing nothing, silently leaving the ~93GB fp16 footprint untouched no matter what memory budget we gave it. We pinned `transformers==4.46.3`, which still has the classic per-expert `nn.Linear` implementation, and NF4 quantization actually applied after that, bringing the model down to a real ~26GB. This is now baked into `run_mixtral.sh`. The sanity numbers from `collect_report.json` (32 layers, 8 experts, non-degenerate entropy, hook fired the expected number of times per token) confirm the fix worked and the traces are real, not corrupted.

---

## 3. The numbers, side by side

| Metric | OLMoE-1B-7B (64 experts, top-8) | Mixtral-8x7B (8 experts, top-2) |
|---|---|---|
| Coverage@25% (deployable, held-out) | **52.2%** | **28.7%** |
| Coverage@25%, 95% bootstrap CI | 48.2% – 55.9% | 27.7% – 30.0% |
| Coverage@50% | 74.1% | 54.5% |
| Mean Gini coefficient (skew) | 0.378 | 0.066 |
| Mean Zipf exponent | 0.70 | 0.17 |
| Recall@2k, d=1, non-resident | **55.3%** | **61.8%** |
| ...vs. static-popularity baseline (full-set, same denominator) | 77.8% vs 52.8% | 68.7% vs 54.4% |
| ...vs. random baseline | 25.0% | 50.0% (small n_experts inflates this floor) |
| Mean hot-set churn (Jaccard, W=2000) | 0.447 | 0.248 (partly a floor artifact, see §5) |
| HTB(C=25%, m=2k) | 0.787 | 0.728 |
| Implied demand-stall rate | 21.3% | 27.3% |
| Verdict against pre-registered thresholds | **PIVOT** | **GO_PREFETCH_LED** |

All numbers are from `results/pilot_summary.json` (OLMoE) and `results/mixtral_summary.json` (Mixtral), computed by `elp_probe/src/analyze.py` on the held-out test split.

---

## 4. What each number means, in plain terms

**Coverage@25%** asks: if you permanently pinned the busiest 25% of experts per layer in fast memory (based on training-set traffic), what fraction of a *different, unseen* set of requests' decode-time expert activations would actually hit that resident set? For OLMoE, 52% of traffic hits the top quarter of experts. For Mixtral, only 29% does. A system built purely on "pin the popular experts and hope" would leave nearly half of OLMoE's traffic and almost three quarters of Mixtral's traffic as cold-tier hits.

**Gini and Zipf exponent** are single-number skew summaries. Gini of 0 means every expert is hit equally often; Gini of 1 means one expert absorbs everything. OLMoE's 0.378 is moderate, real skew. Mixtral's 0.066 is close to flat, meaning its router activates its 8 experts almost uniformly.

**Recall@2k, d=1, non-resident** is the prefetch number. It asks: if you build a simple statistical predictor (a lookup table of "given the experts active at layer L, which experts are likely at layer L+1", fit only on training requests), and you let that predictor nominate 2x the normal number of candidate experts (2k instead of k) for layer L+1, how often does it actually catch the experts that turn out to be active there, restricted specifically to experts that *aren't* already pinned in the resident set (the hard case, since resident experts don't need prefetching)? OLMoE hits 55%, Mixtral hits 62%. Both are compared against two baselines: a naive "always guess the globally most popular experts" baseline (which the conditioned predictor beats by 14 to 25 percentage points on the matched full-set metric, meaning the router's current state genuinely carries predictive information, not just global popularity) and a random-guessing baseline (which the predictor beats by 2.2x on OLMoE; Mixtral's random baseline is already 50% just because it only has 8 experts total, so the margin there is smaller in relative terms but still real in absolute terms).

**Hot-set churn** measures how much the "80%-of-traffic" set of experts changes from one window of 2000 activations to the next, as a Jaccard distance (0 = identical set, 1 = completely different set). Lower is better for a system that wants to pin experts once and leave them pinned for a while. OLMoE's churn of 0.447 is fairly high, meaning a static placement would need fairly frequent re-pinning to stay accurate. Mixtral's 0.248 looks better on its face, but see the caveat below, this number is partly an artifact of Mixtral only having 8 experts total.

**HTB (Hideable Traffic Bound)** is the combined headline number: assuming you pin the top-C% of experts *and* prefetch using the top-m predictions one layer ahead, what fraction of all decode-time expert activations would never cause a demand stall (a cold, unprefetched fetch that blocks compute)? OLMoE bounds this at 78.7%, Mixtral at 72.8%. The flip side, the demand-stall rate, is 21.3% and 27.3% respectively. This is the number that would go into the abstract as "how much of the I/O problem does this actually solve."

---

## 5. Two things in this data that are artifacts, not findings

It matters to call these out explicitly so they don't get misquoted later.

**Mixtral's low churn number is partly mechanical, not a genuine stability finding.** With only 8 experts per layer and near-uniform activation (Gini 0.066), an "80%-of-traffic" hot set needs roughly 7 out of 8 experts to reach that threshold. When your hot set is already 7/8 of the total universe, two consecutive windows' hot sets are forced to overlap heavily just by pigeonhole counting, regardless of whether the underlying traffic pattern is actually stable over time. Don't cite "Mixtral's placement is more stable than OLMoE's" as a finding; the churn metric loses most of its meaning once the hot set approaches the full expert count.

**The `m=4k` rows in the recall table are trivially 1.0 for Mixtral and shouldn't be cited.** For Mixtral, k=2, so `m=4k` means the candidate budget equals `4*2=8`, which is the *entire* expert count for that layer. Asking "does the predictor's top-8 guess out of 8 total experts contain the actual answer" is guaranteed yes, always. That's a saturation artifact of the m-multiplier grid being defined relative to k without capping at n_experts, not a real result. Only the `mk` and `m2k` rows carry signal for Mixtral. This also slightly inflates the C×m HTB grid cells that use `m4k`, those show HTB=1.0 for every C level, which is likewise not meaningful and shouldn't be quoted.

---

## 6. The finding that actually matters: the hypothesis about "coarse models are more skewed" was backwards

`Pilot.md` §1.6 predicted that Mixtral, with only 8 experts per layer and top-2 routing, would show *stronger* skew than OLMoE's 64-expert top-8 routing, on the reasoning that a coarser model is "far less balanced by construction." The measured result is the opposite: OLMoE is meaningfully more skewed (Gini 0.378) than Mixtral (Gini 0.066).

The likely mechanical reason: MoE training uses a load-balancing auxiliary loss that pushes every expert toward getting a roughly equal share of tokens on average. With only 8 experts, that constraint is easy to satisfy almost exactly, there just isn't much room for a handful of experts to specialize away from the 1/8 baseline without the loss noticing and correcting it. With 64 experts, the same constraint only requires the *average* to be near 1/64, which leaves plenty of room for structured variation across individual experts while still satisfying the aggregate balance target. Fewer, bigger "slots" get squeezed toward uniform; more, smaller "slots" have room to specialize.

This is a genuinely interesting and citable empirical result on its own, and it's worth a sentence in the abstract or the paper, because it contradicts the naive intuition that "bigger, coarser experts must mean more skewed routing."

---

## 7. What generalizes and what doesn't

| Mechanism | OLMoE | Mixtral | Generalizes? |
|---|---|---|---|
| Static placement (Coverage@25%) | 52% | 29% | **No.** Opposite direction, inconsistent magnitude. |
| Predictive prefetch (Recall@2k, nonresident) | 55% | 62% | **Yes.** Both clear or nearly clear the 60% bar, both beat baselines by a real margin. |

This is the central takeaway. If the plan had been "prove static placement works and build a placement optimizer around it," the Mixtral number would have been a serious problem, a placement optimizer's entire value proposition weakens fast once coverage drops to 29%. But the plan doesn't have to rest on placement. The prefetch signal, which is what claim (3) in the original abstract was actually betting on, held up on both ends of the fine-grained/coarse-grained spectrum. That's the part of the hypothesis that survived contact with real data.

---

## 8. Decision against the pre-registered thresholds

From `Pilot.md` §2.4, written down before any data was collected:

- **Strong go**: Coverage@25% >= 60% AND Recall@2k,d=1 >= 60% on non-resident. Neither model hit this.
- **Go (placement-led)**: Coverage@25% >= 60% but Recall < 60%. Neither model hit this either.
- **Go (prefetch-led)**: Coverage flat (<40%) but Recall@2k >= 60%. **Mixtral hits this exactly.**
- **Pivot**: both below threshold on both models. **OLMoE lands here** (Coverage 52% is between the flat and strong thresholds, Recall 55% is just under 60%, so it doesn't cleanly qualify for any "go" bucket under the strict rule, even though it's close).

Because the pre-registered table only defines pivot as "both metrics weak on *both* models," and Mixtral clearly clears a go bucket, the overall project-level call is **go, prefetch-led**, not pivot. OLMoE's individual PIVOT label is a strict reading of one model in isolation; taken together with Mixtral, the cross-model picture is "placement doesn't reliably work, prefetch does," which is itself a clean, gateable design decision, just not the one originally assumed.

---

## 9. What this changes about the proposed system

The original abstract described two co-equal mechanisms: (2) a cost-constrained optimizer for placing experts across HBM/DDR/CXL, and (3) router-guided prefetch. The data says these are not equally load-bearing.

**Scale down:** the placement optimizer. Solving a precise cost-constrained placement problem is only worth the engineering effort if the *marginal* gain from getting placement exactly right is large. With Coverage@25% sitting at 29% for Mixtral and topping out around 74-77% even at 50% residency for both models, there just isn't a lot of headroom being left on the table by a naive fixed-fraction policy versus an optimally solved one. Replace the optimizer with a simple fixed-C per-layer resident fraction (the C values already measured: 12.5/25/37.5/50%) and spend the saved engineering time elsewhere.

**Scale up / keep as the centerpiece:** the prefetch scheduler. This is the mechanism with real, cross-model evidence behind it. The system's core contribution should be: given the router's decision at layer L, issue CXL fetches for the predicted layer L+1 experts during layer L's compute, so the fetch latency overlaps with compute instead of stalling it. The predictor design already validated in this pilot (a conditional frequency table, additive smoothing, ranked by summed log-probability) is a reasonable starting point for that scheduler's prediction logic.

**Keep as planned, downstream:** the DRAMSim3-calibrated simulator (4) and multi-tenant CXL pooling analysis. These don't need to change shape, but their inputs should now be "prefetch-primary with lightweight fixed placement" rather than "jointly optimized placement and prefetch."

**Reconsider, don't drop:** extending characterization to DeepSeek-MoE-16B and Qwen1.5-MoE (part of claim 1). Two models already show a real architecture-dependent split (fine-grained vs. coarse-grained skew reverses expectations). A third and fourth model would help establish whether the "prefetch generalizes, placement doesn't" pattern is a two-model coincidence or a real trend, and DeepSeek-MoE's shared-plus-routed-expert design is architecturally different enough from both OLMoE and Mixtral that it could reveal a third regime. This is valuable if hardware access allows it, but the core go/no-go decision for the project no longer depends on it, since the two-model result Already clears one of the pre-registered "go" buckets.

---

## 10. Open items before this goes further

1. **Batch-8 erosion check for Mixtral.** Not yet run at the time of writing (`batch8_erosion: null` in `mixtral_summary.json`). OLMoE's version showed real erosion (12.5% to 47% of experts touched per step at batch 8). Worth confirming whether Mixtral, with its already-flat routing, erodes similarly or differently before finalizing the "low-batch, latency-sensitive serving is the CXL-relevant regime" framing. Command is already queued in the `run_mixtral.sh` terminal output.
2. **NF4 fidelity spot-check skipped.** No box available here clears the ~87GB FP16 Mixtral footprint even with CPU offload. Per `Pilot.md` §3.4 this should be caveated in any written material ("literature suggests >=95% NF4/FP16 routing agreement") rather than asserted as measured.
3. **Single seed, n=250 requests per model.** Bootstrap CIs are in the summary JSONs and are reasonably tight (e.g. OLMoE Coverage@25% CI is 48.2-55.9%, about 8 points wide), but this is still one seed's worth of request sampling. Worth keeping in mind as a limitation if these numbers go into anything more formal than an abstract.
4. **The abstract's four-model characterization claim (1) is currently two-thirds unfulfilled.** DeepSeek-MoE-16B and Qwen1.5-MoE haven't been traced. The rewritten abstract frames this as "where hardware access allows" rather than a firm commitment, which matches the current state honestly.

---

## 11. Where everything lives

- `elp_probe/` - the collector, analyzer, and run scripts (`setup.sh`, `run_pilot.sh`, `run_mixtral.sh`).
- `results/pilot_summary.json`, `results/olmoe/` - OLMoE traces and computed metrics.
- `results/mixtral_summary.json`, `results/mixtral/` - Mixtral traces and computed metrics.
- `results/figures/coverage_cdf.png`, `results/figures_mixtral/coverage_cdf.png` - Coverage@C plots per model.
- `Pilot.md` - the original experiment design and pre-registered decision thresholds this whole analysis is measured against.
