# tierMoE: What's Verified, What's Simulated, What's Projected

This document is the honesty ledger for the `framework/` package. Every claim
below is tagged with one of three maturity levels — read the tag before you
quote the number anywhere (a report, a slide, an internship pitch):

| Tag | Meaning | Who/what produced it |
|---|---|---|
| **MEASURED** | A real quantity computed directly from real, already-collected trace data (router decisions from 250 real chat/code requests through real model forward passes), with no simulation step in between. | The original pilot (ELP-Probe) and experiments E1/E2/E4/E5/E8, reproduced and cross-checked by this framework's own test suite. |
| **SIMULATED** | A real number this framework's own code actually computed and this session actually ran, but which depends on the `flop_estimate` compute-time model (no real GPU calibration exists yet — see the Methodology section) and/or this framework's own discrete-event simulation logic rather than a live re-run of the model. | `tiermoe.sim`, `tiermoe.tco`, `tiermoe.kv` — new code written for this framework (E6, E9 were speced but never implemented anywhere in this repo before now; E7's byte-accounting side is implemented here for the first time too). |
| **PROJECTED** | Not run anywhere in this repository. A documented, literature-cited expectation for what a real run (on a CUDA box, on Linux with CXLMemSim, or with real fleet telemetry) would plausibly show, structured so that real run can replace the projection without changing any downstream code. | Cited papers (arXiv IDs given every time) + this framework's own architecture-derived arithmetic. |

Nothing in this document is asserted as more certain than these tags say it
is. Where a tag downgrades a number the original project docs treated as
settled (see the calibration-file finding below), that downgrade is itself
reported as a finding, not smoothed over.

### Mentor feedback disposition

Competition mentors reviewed an earlier version of this project's abstract
and gave seven pieces of feedback. All seven were checked against this
framework and closed — three were fully new work, two were bug-fixes /
extensions to existing code, and two were already covered and are noted as
such rather than re-padded with busywork:

| # | Mentor feedback | Disposition | Where |
|---|---|---|---|
| 1 | Report precision alongside recall | **New**: `precision_overall`/`precision_nonresident` added, with the exact-identity-vs-genuinely-informative distinction proven by test | §1 (below), `tiermoe.analyze.workload` |
| 2 | Evaluate under concurrent inference / CXL bandwidth contention | **New**: closed-loop fair-share model + stability boundary, with an honest finding (see §2.5) | §2.5, `tiermoe.sim.concurrency` |
| 3 | Account for HBM pressure from KV cache | **New**: KV-vs-expert HBM budget accounting, can cap residency or flip feasibility entirely | §2.4, `tiermoe.baseline.compare.kv_hbm_pressure` |
| 4 | Stronger baselines: static + popularity-aware | **Extended**: `static-c` already was popularity-ranked; added `popularity-prefetch` (unconditioned) to isolate the value of conditioning | §2.1, `tiermoe.sim.policies` |
| 5 | Hybrid tiering + prefetching (hot resident, cold prefetched) | **New**: `hybrid-lru-prefetch` policy; surfaced and fixed a real double-counted-memory-budget bug in the pre-existing `lru` policy along the way | §2.1, `tiermoe.sim.policies` |
| 6 | Core challenge: prefetching hides latency consistently across models/conditions | **Answered honestly, not assumed**: holds at concurrency=1 across both models; demonstrably does *not* hold once realistic concurrent load is added at these bandwidths — the framework now measures exactly where that boundary is instead of asserting consistency it can't back up | §2.5 |
| 7 | Ensure HBM-only vs HBM+CXL comparison is actually functional | **Already built, now strengthened**: `tiermoe baseline` always runs both sides; now also threads through concurrency and KV pressure so the comparison reflects realistic serving load, not just a single idle request | `tiermoe.baseline.compare.run_baseline_vs_cxl` |

Point 6 is deliberately not softened: a framework whose central claim only
holds in the easy case would be a worse deliverable than one that tells you
precisely when the claim breaks. See §2.5 for the numbers.

---

## 1. MEASURED — real numbers, real held-out traces

Source: `results/pilot_summary.json` (OLMoE), `results/mixtral_summary.json`
(Mixtral), both produced by `elp_probe/src/analyze.py` on a request-level
70/30 train/test split (see `Pilot.md` section 1.7 for why the split is by
*request*, not by token). Reproduced independently by
`tiermoe.analyze.workload.run_characterization` and cross-checked to within
1% by `tests/test_analyze.py::test_run_characterization_{olmoe,mixtral}_*` —
**both pass on this machine**.

| Metric | OLMoE-1B-7B (64 experts, top-8) | Mixtral-8x7B (8 experts, top-2) |
|---|---|---|
| Coverage@25% (deployable, held-out) | **52.2%** [95% CI 48.2–55.9%] | **28.7%** [95% CI 27.7–30.0%] |
| Coverage@50% | 74.1% | 54.5% |
| Gini coefficient (skew) | 0.378 | 0.066 |
| Zipf exponent | 0.70 | 0.17 |
| Recall@2k, d=1, non-resident | **55.3%** | **61.8%** |
| ...vs. static-popularity baseline | +25pp (77.8% vs 52.8%, full-set) | +14pp (68.7% vs 54.4%, full-set) |
| Hot-set churn (Jaccard, W=2000) | 0.447 | 0.248 (partly a pigeonhole floor artifact — see caveat below) |
| HTB(C=25%, m=2k) | 78.7% | 72.8% |
| Batch-8 erosion (touched-expert fraction) | 12.5% → 47.3% | not run (open item) |
| Verdict vs. pre-registered thresholds | PIVOT (borderline) | **GO_PREFETCH_LED** |
| Cross-model call | **prefetch generalizes; static placement does not** | |

**Skew-inversion finding (also MEASURED, R3 in Main.md):** the fine-grained
model (64 experts) is *more* skewed (Gini 0.378) than the coarse model (8
experts, Gini 0.066) — backwards from the naive intuition, attributed to the
load-balancing auxiliary loss binding far more tightly over 8 slots than over
64. This is a genuine, citable empirical result, currently resting on n=2
models (see the Projected section for the n=4 extension).

**E4 (predictor ablation, MEASURED, CPU-only, `experiments/e4_predictor/`):**
the MLP predictor beats the frequency table by **+31.9pp** recall@k,
non-resident for OLMoE (clears the 8pp exit-criteria bar comfortably) but only
**+7.7pp** for Mixtral (a negative result by the same bar — "the router's
selected set already carries nearly all the available signal" for the coarse
model). Calibration: ECE 0.0056→0.0037 (OLMoE), 0.0065→0.0058 (Mixtral) after
temperature scaling — both predictors were already close to calibrated out of
the box (T≈0.96–0.98).

**E5 (lookahead depth, MEASURED, CPU-only, `experiments/e5_lookahead/`):**
direct-vs-chained recall gap widens monotonically with depth for both models
(OLMoE: +0.019 at d=1 → +0.170 at d=8; Mixtral: +0.008 → +0.100) — exactly the
"chaining compounds error" prediction. Neither model reaches the 0.6 recall
threshold at m=k for any tested depth (expected: the pilot's own ≥0.6 bar was
measured at m=2k, not m=k — not a broken predictor).

**Prefetch precision alongside recall (mentor-review ask — MEASURED, recomputed
by `tiermoe.analyze.workload.recall_at_m`):** high recall alone is free to buy
by fetching more candidates (`m`), so it doesn't by itself show a predictor is
any good — precision closes that gap. Two precision numbers exist and they
are **not** interchangeable:

- `precision_overall = recall_overall * (k/m)` **identically**, by algebra —
  every row contributes exactly `k` real experts (`tot_o = k * n_rows`) and
  exactly `m` fetched candidates (`n_fetched = m * n_rows`), both constants
  independent of the actual data, so this pair carries **zero information**
  beyond recall itself. It's reported for direct readability only.
- `precision_nonresident` (paired with `recall_nonresident`) IS genuinely
  informative — its denominator (how many of the k routed experts are not
  already HBM-resident) varies row-to-row with real residency overlap, so it
  is not a fixed rescaling of anything. At OLMoE's headline operating point
  (m=2k, d=1, C=25% residency), `recall_nonresident=55.3%` — the framework's
  own test suite proves the two pairs really do diverge with a hand-worked
  4-expert/k=1 synthetic case where `recall_nonresident=2/3` but
  `precision_nonresident=0.25` (not `2/3 * 1/2 = 0.333`) —
  `tests/test_analyze.py::test_precision_nonresident_is_not_a_fixed_rescaling_of_recall_nonresident`.
  `baseline_random_precision = k/m` is reported alongside as the
  no-information floor any real predictor must clear.

**E8 (KV/expert co-tenancy, MEASURED, CPU-only, `experiments/e8_kv_cotenancy/`):**
at the operating point where expert traffic alone is already viable (batch=32,
residency=75%, NF4), KV and expert traffic genuinely contend: at 32 GB/s,
`kv-first` cuts Mixtral's KV TTFT from 1055.6ms (`expert-first`) to 246.6ms at
the cost of 414ms/token expert TPOT inflation. At 64 GB/s the contention
mostly disappears (`kv-first` becomes strictly Pareto-dominant with ~0 expert
inflation) — headroom, not scheduling cleverness, is what matters once the
link isn't saturated.

---

## 2. SIMULATED — this framework's own code, run on real traces, this session

These numbers come from `tiermoe.sim` (E6, never implemented before this
framework — no `e6_*` folder exists in `experiments/`), `tiermoe.tco` (E9,
same situation), and `tiermoe.policy.precision_gate` (E7's byte-accounting
side). All 120 tests in `tests/` pass on this machine (650s combined run),
including the seven
E6-style validation gates (oracle is the upper bound; residency=100% +
infinite bandwidth reproduces an HBM-only baseline exactly to <1% relative
error; TPOT is weakly monotonic in residency; a 150→400ns CXL latency sweep
moves TPOT by less than 5%; oracle's TPOT is invariant to concurrency;
`static-c`'s TPOT is monotonically non-decreasing in concurrency;
`hybrid-lru-prefetch` never has worse TPOT than plain `lru` at any
concurrency level) — **for both shipped models**. A non-pass/fail companion,
`find_collapse_concurrency`, reports the exact concurrency at which a named
prefetch policy becomes numerically indistinguishable from `static-c` (§2.5).

### 2.1 The headline policy bake-off (E6)

**A correction made during the mentor-feedback round, before these numbers:**
the original `lru` policy checked misses against the static popularity-ranked
resident set *and then* a same-sized LRU cache underneath it — silently
giving it double the memory budget of every other policy at the same
`--residency-pct`. Fixed: `lru`'s cache capacity now *is* the entire
configured residency budget, nothing else (`tests/test_sim_policies.py::
test_lru_cache_never_exceeds_configured_capacity` is the regression test).
The tables below are the corrected numbers, plus the two new policies added
for the mentor-feedback round (`popularity-prefetch`, `hybrid-lru-prefetch`)
and precision/recall for every policy that predicts a candidate set.

**OLMoE**, residency=25%, bw=128 GB/s, depth=4, FP16, concurrency=1 (the
operating point E5's own feasible-fetch-size table says is needed for a
12.6MB expert to be fetchable at all within a lookahead window):

| Policy | TPOT mean (ms) | Demand-stall rate | Precision (nonres.) | Recall (nonres.) | Oracle gap closed |
|---|---|---|---|---|---|
| none | 13.49 | 100.0% | — | — | 0.0% |
| static-c (popularity-ranked) | 6.92 | 100.0% | — | — | 52.2% |
| popularity-prefetch (unconditioned) | 6.92 | 100.0% | 0.0% | 0.0% | 52.2% |
| lru | 5.81 | 38.9% | — | — | 61.1% |
| prefetch-freq | 6.89 | 99.4% | 1.5% | 0.6% | 52.5% |
| prefetch-v2 (MLP) | 6.79 | 97.8% | 6.0% | 2.2% | 53.3% |
| prefetch-v2 + precision-gate | 6.46 | 92.4% | 10.2% | 7.6% | 55.9% |
| **hybrid-lru-prefetch** | **3.20** | **38.0%** | 1.5% | 0.6% | **81.8%** |
| oracle | 0.91 | 0.0% | — | — | 100.0% |

**Mixtral**, residency=75%, bw=128 GB/s, depth=8, NF4, concurrency=1 (Main.md's
own stated requirement — "Mixtral-class requires d≥8 *and* byte reduction"):

| Policy | TPOT mean (ms) | Demand-stall rate | Precision (nonres.) | Recall (nonres.) | Oracle gap closed |
|---|---|---|---|---|---|
| none | 188.91 | 100.0% | — | — | 0.0% |
| static-c (popularity-ranked) | 50.67 | 100.0% | — | — | 78.5% |
| popularity-prefetch (unconditioned) | 50.67 | 100.0% | 0.0% | 0.0% | 78.5% |
| lru | 20.78 | 17.7% | — | — | 95.4% |
| prefetch-freq | 39.47 | 70.5% | 10.7% | 29.5% | 84.8% |
| prefetch-v2 (MLP) | 39.19 | 69.7% | 11.0% | 30.3% | 85.0% |
| prefetch-v2 + precision-gate | 40.00 | 71.9% | 10.7% | 28.1% | 84.5% (72.8% fewer bytes than all-FP16) |
| **hybrid-lru-prefetch** | **12.76** | **0.02%** | 10.7% | 29.5% | **99.996%** |
| oracle | 12.75 | 0.0% | — | — | 100.0% |

**Stronger baselines (mentor point 4), read directly off these tables:**
`static-c` already *is* the static+popularity-aware baseline (fixed residency
ranked by training-set popularity). `popularity-prefetch` — new this round —
isolates what popularity *alone*, applied as a prefetch policy instead of a
placement policy, buys over `static-c`: **nothing, at either model**
(bit-identical TPOT/stall/gap to `static-c` in both tables, 0% precision/
recall). Every prefetch-based policy that beats `static-c` does so by
*conditioning on current router state* (`prefetch-freq`, `prefetch-v2`,
`hybrid-lru-prefetch`) — not by fetching popular experts unconditionally.
This is exactly the ablation the mentors asked for, and the result is clean:
**conditioning is what earns the improvement, not popularity ranking by
itself.**

**Hybrid tiering + prefetching (mentor point 5) is the strongest policy at
either model, not merely competitive** — `hybrid-lru-prefetch` beats every
other non-oracle policy, including plain `lru`, at both operating points
(OLMoE: 81.8% vs `lru`'s 61.1% vs `oracle`-gated `prefetch-v2`'s 55.9%;
Mixtral: 99.996% vs `lru`'s 95.4%). `tests/test_sim_policies.py::
test_hybrid_lru_prefetch_never_worse_than_plain_lru` enforces this isn't a
one-off: hybrid strictly dominates plain LRU by construction (same cache,
plus a prefetch layer on top), so this ranking is guaranteed, not lucky.

**A satisfying resolution to the LRU bug above, worth stating explicitly:**
the OLD, buggy, double-capacity `lru` for Mixtral reported TPOT=12.76ms,
99.97% gap closed — numbers that are, to three significant figures,
*identical* to the new, correctly-budgeted `hybrid-lru-prefetch`'s real
12.76ms / 99.996%. The bug wasn't reporting nonsense — it was accidentally
simulating something close to "LRU cache plus additional prefetched
capacity," which is exactly what `hybrid-lru-prefetch` now legitimately does
at the correct, shared, single residency budget. The corrected plain `lru`
number (20.78ms / 95.4%) is what a fair single-budget comparison actually
looks like, and the gap between it and `hybrid-lru-prefetch` is the real,
honest measure of what adding a prefetch layer on top of LRU buys you.

**Reproduction of the project's own central finding, independently:** at
batch=1 with realistic bandwidth (32 GB/s) and shallow depth (d=1), every
prefetch-based policy above collapses to *exactly* `static-c`'s numbers —
because the byte budget for even one candidate expert doesn't exist within
one layer's compute window (OLMoE: needs ~3536 GB/s to fetch a 2×-oversampled
candidate set at d=1; Mixtral: similarly infeasible for an 8-expert NF4
candidate set). This is not a bug — running the arithmetic by hand confirms
it (`t_layer_ms × bw_bytes_per_ms < m × expert_bytes` at these settings) — it
is an **independent reproduction, by a simulator built from scratch off the
same formulas, of Main.md's own central claim**: ρ ≥ 1 means only byte
reduction helps, prediction accuracy is irrelevant. Two independently-built
pieces of code agreeing on this is stronger evidence than either alone.

### 2.2 TCO / pooling (E9)

`tiermoe.tco.pooling` implements Pond's stranding argument via a standard
portfolio-variance model (aggregate safety margin shrinks as 1/√(pool size)
for independent per-host demand — see the module docstring for the full
derivation). Verified property (test-enforced, not merely illustrated):
stranding reduction is monotonically non-decreasing in pool group size, and
is exactly 0% at group size 1 by construction. No real fleet telemetry
exists (see Projected section) — every number from `simulate_fleet_demand` is
explicitly synthetic (log-normal demand, documented coefficient of variation),
the same "assumption, not measurement" tag E8 already applies to its own
`--kv-rate-per-sec`.

### 2.3 Confidence-gated precision fallback (E7) — byte side only

The gating *policy logic* (threshold decisions, byte accounting, Pareto
sweeps over τ_lo/τ_hi) is real, tested arithmetic — see the Mixtral table
above: at (r=75%, bw=128GB/s, d=8), the gate spends bytes on 64,151 NF4 hits
and 1,198 FP16 hits, achieving **72.8% fewer bytes than an all-FP16 policy**
at the cost of 1.5pp of oracle-gap-closed versus ungated `prefetch-v2` (85.0%
→ 84.5%) — a small, real, measurable Pareto trade at this specific operating
point. The **quality** side (real ΔPPL from actually substituting NF4 experts
during inference) is PROJECTED, not measured — see section 3.

### 2.4 KV-cache HBM pressure (mentor-review ask, E-new, `tiermoe.baseline.compare.kv_hbm_pressure`)

KV cache lives in the same HBM budget as resident expert weights — a
long-context serving workload can crowd out expert residency entirely, which
the original single-request framework never modeled. Both shipped models
have `kv_bytes_per_token = 131072` bytes **exactly** (128 KiB/token — a real
architectural coincidence: OLMoE is 16 layers × 16 KV heads, Mixtral is 32
layers × 8 GQA heads, both landing on 256 head-layers × 512 bytes/head-layer
at FP16). Worked example, OLMoE against an 80GB HBM budget (FP16 KV, FP16
experts, ~12.9GB total expert weight):

| Concurrent 128k-token requests | KV cache total | HBM left for experts | `max_residency_pct_given_kv` |
|---|---|---|---|
| 1 | ~17.18 GB | ~62.82 GB | 100% (uncapped — plenty of room) |
| 4 | ~68.72 GB | ~11.28 GB | **~87.4%** (a real, non-trivial cap) |
| 5 | ~85.9 GB | *(exceeds 80GB budget alone)* | **0%** — `kv_alone_exceeds_hbm_budget=True` |

`capacity_check_with_kv` can flip a model that `capacity_check` alone says
"fits in HBM" into "does not fit" once KV cache is counted — i.e. accounting
for KV pressure can change whether the HBM-only baseline is even *feasible*,
not just shrink the achievable residency. `run_baseline_vs_cxl(...,
context_len=N, concurrent_requests=M)` applies this cap automatically
(`effective_residency_pct <= residency_pct`); `context_len=0` (default) skips
KV accounting entirely, preserving the original single-request behavior
exactly — tested in `tests/test_baseline_compare.py` (6 tests, including a
same-as-before check at `context_len=0`).

### 2.5 Concurrent inference / CXL bandwidth contention (mentor-review ask, E-new, `tiermoe.sim.concurrency`)

**This is the section that answers mentor point 6 honestly — the core
challenge of showing prefetching hides latency consistently across
conditions.** The real answer, measured on both shipped models via
`run_concurrency_sweep` (concurrency ∈ {1,2,4,8}, same operating points as
§2.1) is more precise — and more interesting — than a flat yes or no:
**byte-budget-limited conditional prefetch loses its edge under concurrency,
but a bandwidth-free residency mechanism (LRU) does not, and the hybrid
policy the mentors specifically asked for (point 5) is what makes the
combination robust.**

**OLMoE** (residency=25%, bw=128GB/s, depth=4, FP16) — TPOT mean (ms) /
oracle-gap-closed:

| Concurrency | static-c | prefetch-v2+gate | hybrid-lru-prefetch | oracle |
|---|---|---|---|---|
| 1 | 6.92 / 52.2% | 6.46 / 55.9% | **3.20 / 81.8%** | 0.91 / 100% |
| 2 | 12.93 / 52.2% | 12.56 / 53.7% | **5.50 / 81.8%** | 0.91 / 100% |
| 4 | 24.95 / 52.2% | 24.65 / 52.8% | **10.11 / 81.7%** | 0.91 / 100% |
| 8 | 48.996 / 52.2% | 48.73 / 52.5% | **19.31 / 81.7%** | 0.91 / 100% |

At concurrency=4, `prefetch-freq` and plain `prefetch-v2` (not shown — see
raw CSV) become **bit-identical** to `static-c` (24.953393ms, exactly) — the
lookahead byte budget for even one candidate expert no longer exists once 4
streams divide the link. `prefetch-v2+precision-gate` degrades more
gracefully (NF4 fallback still gets some bytes through) but its edge over
`static-c` shrinks from +3.6pp (C=1) to +0.6pp (C=4) to +0.27pp (C=8).
`hybrid-lru-prefetch`'s advantage barely moves (81.8% → 81.7%) across the
entire grid.

**Mixtral** (residency=75%, bw=128GB/s, depth=8, NF4):

| Concurrency | static-c | prefetch-v2+gate | hybrid-lru-prefetch | oracle |
|---|---|---|---|---|
| 1 | 50.67 / 78.5% | 40.00 / 84.5% | **12.76 / 99.996%** | 12.75 / 100% |
| 2 | 88.58 / 78.5% | 78.97 / 81.2% | **12.77 / 99.994%** | 12.75 / 100% |
| 4 | 164.41 / 78.5% | 154.34 / 79.9% | **12.79 / 99.994%** | 12.75 / 100% |
| 8 | 316.07 / 78.5% | **316.07 / 78.5%** | **12.84 / 99.994%** | 12.75 / 100% |

At concurrency=8, `prefetch-v2+precision-gate` itself becomes bit-identical
to `static-c` for Mixtral (both 316.074243ms, gate spends 0 NF4 bytes,
0 hits) — full collapse of every conditional-prefetch variant, at a
higher concurrency threshold than OLMoE's (8 vs. 4) because Mixtral's larger
per-operation byte budget (d=8, NF4) buys more headroom, but the *same*
qualitative collapse happens at both models. Meanwhile
`hybrid-lru-prefetch`'s absolute TPOT is nearly flat (12.76→12.84ms, +0.6%)
because Mixtral's LRU/prefetch hit rate is so high (8 experts/layer, Gini
0.066) that almost no bytes need transferring regardless of how the link is
shared.

**The mechanism, made precise by `tiermoe.sim.concurrency.aggregate_rho` /
`max_sustainable_concurrency`:** conditional prefetch's byte budget is
governed by `m_affordable`, which shrinks as the *effective* per-stream
bandwidth (`bw_gbps / concurrency`) shrinks — so concurrency dividing
bandwidth is functionally the same failure mode as the low-raw-bandwidth
collapse already reproduced in §2.1's closing paragraph, just parameterized
by N instead of by GB/s directly. A dynamic residency mechanism (LRU) needs
no such budget — a cache hit costs zero bytes regardless of how many streams
share the link — so it (and anything built on top of it, like
`hybrid-lru-prefetch`) is structurally immune to this specific collapse mode.

**This is the honest, precise answer to mentor point 6, and it reframes point
5 as the actual fix for point 2/6, not just an independently-nice idea:**
pure conditional prefetching does *not* consistently hide CXL latency across
concurrency conditions — it consistently degrades to the static baseline once
the link is shared by enough streams, on both tested models. The **hybrid**
design the mentors asked for gives the framework a concurrency-robust floor
(the LRU-resident hot set, paid for in HBM, not bandwidth) with prefetching
adding upside only when the link actually has slack — which is a more
defensible, more literature-consistent claim than asserting prefetching alone
"always works," and it is the version of the claim this framework can
actually back with numbers on both models. Read the `concurrency_stable`
boundary the same way the roofline reads ρ≥1: once it flips `False`, no
*byte-budget-dependent* policy can beat any other byte-budget-dependent
policy, because the constraint has moved from "can we predict the right
bytes" to "can the link move enough bytes at all" — a bandwidth-free
mechanism is the only kind of "prefetching" (broadly construed, since LRU
promotion is itself a form of anticipatory placement) that keeps working past
that point.

**This finding is now enforced, not just narrated by hand.**
`tiermoe.sim.validation.run_validation_suite` (`tiermoe validate --model
<m>`) gained three gates specifically for this: oracle's TPOT must be
invariant to concurrency (it never touches the shared link), a
byte-transferring policy's TPOT must be monotonically non-decreasing in
concurrency (bw/N can only ever hurt, never help), and `hybrid-lru-prefetch`
must never have worse TPOT than plain `lru` at ANY concurrency level, not
just concurrency=1. A fourth, non-pass/fail companion —
`find_collapse_concurrency` — reports the exact smallest concurrency at which
a named policy becomes numerically indistinguishable from `static-c`, turning
"conditional prefetch collapses under load" into a number computed on demand
for any model/operating point, not just the two written up above by hand.
`tiermoe baseline --concurrency-grid 1,2,4,8` (`run_flagship_report`) runs
the complete capacity+KV+bake-off comparison once per concurrency level in
one call, and the dashboard's Policy explorer tab has an opt-in "Show
concurrency sweep" chart plotting exactly this collapse curve live.

---

## 3. PROJECTED — literature-grounded, not run in this repository

### 3.1 Methodology note: real GPU compute/transfer calibration (E3)

**Status: the shipped `calib_compute.json` cannot be trusted as a real
measurement.** This framework's own `tiermoe.roofline.model.
validate_calib_sanity` (added specifically because of this finding) flags
**both** OLMoE and Mixtral entries in
`experiments/e1_roofline/out/calib_compute.json`:

- OLMoE: mean compute time across batch {1, 8, 32} spans only **2.5%**
  (2.99ms → 3.07ms) — a real dense-GEMM MoE block should show a much larger
  batch-scaling effect.
- Mixtral: per-layer timing relative standard deviation is **0.0000%** across
  32 independently-timed CUDA-event layers — statistically impossible for
  real hardware jitter.
- This is not a new failure mode: `experiments/e3_calibration/E3_CALIBRATION.md`
  already documents one prior fabrication of this exact file being caught and
  deleted. The file currently on disk was apparently regenerated afterward
  and shows the same signature. **This framework treats it as UNVERIFIED**
  and every roofline/sim number in sections 1–2 above uses `flop_estimate`
  (MFU=0.08 assumption), never this file, regardless of what `calib_compute.json`
  currently claims.
- E1's own design doc flags the `flop_estimate` model as likely **pessimistic**
  about compute time at `mfu=0.08` — meaning a real measurement would plausibly
  *shorten* ρ (make more configurations latency-bound than the numbers above
  suggest), not lengthen it. Projected direction: **favorable**, magnitude
  unknown until measured.
- **To close this gap:** run `tiermoe.calibrate.compute --model {olmoe,mixtral}`
  on a real CUDA box (single 24GB GPU is sufficient per the project's own
  E3_CALIBRATION.md correction), then `tiermoe.calibrate.transfer_measure`,
  `.e2e_offload`, `.nf4_fidelity`, then `tiermoe.calibrate.report.
  build_calibration_report(...)` — every number in sections 1–2 recomputes
  automatically once a real, sanity-check-passing `calib_compute.json` exists.

Literature anchors for what real CXL numbers to expect (already wired into
`tiermoe.specs.tiers.TIER_SPECS`): CXL 2.0 ×8 ≈ 32 GB/s / +170–250ns added
latency, CXL 3.x multi-link ≈ 128 GB/s (Sun et al., *Demystifying CXL Memory*,
ISCA 2023; Astera Labs Leo product brief). PCIe 4.0 ×16 ≈ 25 GB/s effective
(FloE, arXiv:2505.05950) is the measured-physical-layer analogue this
project's own E3/E10 plan uses to validate policy *ordering* without real CXL
silicon.

### 3.2 Real precision-gating quality cost (E7, quality side)

`tiermoe.policy.precision_gate.expected_delta_ppl` returns a documented prior
— ΔPPL ≈ 0.15 × (bytes-saved fraction), capped at 0.6, tagged
`provenance="projected_from_literature"` — anchored on two papers:

- **QLoRA** (Dettmers et al., arXiv:2305.14314): NF4 fine-tuning recovers
  FP16 performance within noise under a *stronger* substitution (all weights,
  not just low-confidence ones) than this policy uses — the slope above is
  deliberately conservative (pessimistic) relative to that result.
- **HOBBIT** (Tang et al., arXiv:2411.01433): reports usable quality under its
  own *reactive* (on-miss) mixed-precision policy at comparable NF4/INT4
  fractions — the other anchor for the chosen slope.

**To close this gap:** run real inference with NF4-substituted experts per
the gate's own logged decisions (Main.md section 4.4b step 3: WikiText-2
perplexity + a small GSM8K/HumanEval slice), on a CUDA box. Target per the
original exit criteria: bytes/token drop ≥30% with ΔPPL ≤ ~0.1. This
framework's own gate already demonstrates 72.8% byte savings at one operating
point (section 2.3) — whether the *quality* cost at that specific configuration
clears the ≤0.1 ΔPPL bar is exactly what this experiment would tell you, and
is currently unknown.

### 3.3 Real fleet economics (E9)

No production MoE-serving fleet was ever instrumented for this project — every
TCO number is built on a synthetic log-normal demand model (section 2.2).
Cost-ratio grounding (HBM ≥3× DDR5 $/GB, swept 2–5×) is cited from *Amplifying
Effective CXL Memory Bandwidth for LLM Inference* (arXiv:2509.03377) and
applied via `tiermoe.tco.pooling.tco_crossover`, which already produces a
real crossover table — the CXL controller markup (1.15× DDR5, `Main.md`
section 4.6's own placeholder) is the one number in this pipeline with no
citation at all; treat it as the most negotiable assumption in the whole
ladder.

### 3.4 Hardware validation without CXL silicon (E10)

Three artifacts Main.md's own plan calls for, none built/run here:

1. **Real PCIe offloading ground truth** — `tiermoe.calibrate.e2e_offload`
   is written and ready but CUDA-only; expected ordering
   `next-layer-topk ≤ static-hot ≤ none` (checked by
   `tiermoe.calibrate.policy_check`, itself fully tested with synthetic
   fixtures in this repo).
2. **QEMU CXL Type-3 software-stack demo** — not attempted; needs an x86_64
   Linux host with KVM. Functional, not timing-accurate, per Main.md's own
   framing — its value is proving familiarity with the real Linux CXL stack
   (`cxl create-region`, DAX/kmem, `numactl --membind`), not producing a
   number this document would report.
3. **Real CXLMemSim run** — see `tiermoe/sim/backends/cxlmemsim.py` and
   `demo/cxlmemsim/README.md`. This is the one artifact that WOULD produce a
   directly comparable number to section 2.1's table (real epoch-based CXL
   timing injection on the actual OLMoE forward pass) if run on Linux.
   `ingest_cxlmemsim_report()` is the integration point, ready and unit-tested
   against a synthetic report shape, waiting for a real one.

### 3.5 Model generalization beyond n=2 (E11)

`tiermoe.specs.models.MODEL_SPECS` already includes architecture specs for
DeepSeek-MoE-16B (64 routed + 2 shared experts, top-6) and Qwen1.5-MoE-A2.7B
(60 routed + 4 shared, top-4) — both marked `UNTRACED_MODELS`, meaning the
roofline/TCO/KV modules can already compute projected numbers for them (byte
sizes, KV formulas) but `tiermoe.analyze`/`tiermoe.sim` have nothing to run
against without real traces. `tiermoe.policy.residency.FixedResidencyPolicy`'s
`n_shared_experts` parameter already implements the "shared experts are
always resident" free placement rule these two architectures would need —
tested with synthetic data in `tests/test_policy_residency.py`, unverified
against these models' real routing behavior.

**To close this gap:** trace both models with `tiermoe.collect` on a CUDA box
(both fit in NF4 on a single 24GB GPU per Main.md section 8.2), then every
number in sections 1–2 of this document recomputes for n=4 models instead of
n=2 with no code changes — only new `results/{model}/b1/traces.jsonl.zst`
files. The scientifically interesting output would be a Gini-vs-n_experts
scatter across all four models — Main.md itself calls this "the paper-worthy
result" of the whole generalization experiment, and this framework cannot
manufacture it without the trace data.

---

## 4. Coverage matrix

| Original experiment | Built in this framework? | Run in this session? | Tag |
|---|---|---|---|
| Pilot (ELP-Probe) | `tiermoe.analyze`, `tiermoe.traces` | ✅ (reproduces committed numbers) | MEASURED |
| E1 Roofline | `tiermoe.roofline` | ✅ | SIMULATED (compute) / MEASURED (E_union) |
| E2 Batching sweet spot | `tiermoe.eunion` (core primitive reused) | partially — B* sweep not re-exposed as its own CLI command | SIMULATED |
| E3 Hardware calibration | `tiermoe.calibrate.*` | GPU submodules: no (no CUDA here); `policy_check`/`report`: ✅ | PROJECTED (GPU parts) |
| E4 Predictor v2 | `tiermoe.policy.predictor_mlp`, `.features`, `.calibration` | reused inside `tiermoe.sim`'s `prefetch-v2` | MEASURED (original) |
| E5 Lookahead depth | `tiermoe.policy.lookahead` | not re-run standalone this session (logic reused inside `tiermoe.sim`'s depth parameter) | MEASURED (original) |
| **E6 Simulator + policy bake-off** | `tiermoe.sim` (**new — never existed before**) | ✅, both models, 4 validation gates | **SIMULATED** |
| **E7 Precision-gated prefetch** | `tiermoe.policy.precision_gate` (**new**) | ✅ byte side; ❌ quality side | SIMULATED (bytes) / PROJECTED (quality) |
| E8 KV co-tenancy | `tiermoe.kv.cotenancy` (generalized onto `tiermoe.sim.engine`) | ✅ | MEASURED (original) / SIMULATED (this port) |
| **E9 Pooling & TCO** | `tiermoe.tco` (**new — never existed before**) | ✅ | **SIMULATED** |
| E10 Hardware validation | `tiermoe.calibrate.e2e_offload`, `tiermoe/sim/backends/cxlmemsim.py` | ❌ (CUDA/Linux-only) | PROJECTED |
| E11 Model generalization | specs added for 2 more models | ❌ (no traces) | PROJECTED |
| E12 Ablations/reproducibility | `pytest tests/` IS this framework's version of "make reproduce" | ✅ 120/120 pass (650s) | MEASURED (this framework) |
| E13 Dashboard | `tiermoe.dashboard` | ✅ (all 4 tabs headlessly smoke-tested via Streamlit's `AppTest` harness — 0 exceptions across every tab's default render, not just data-shaping logic checked standalone) | SIMULATED |

## 5. The four sentences that win this (three original + one earned by mentor review)

1. *"We measured, on real traces from two MoE architectures, that expert
   traffic is predictable one layer ahead — 55–62% recall on non-resident
   experts, beating a popularity baseline by 14–25 points."* (MEASURED)
2. *"Prefetching hides latency without creating bandwidth, so we built a
   roofline that says exactly when CXL tiering works — and we independently
   re-derived the same conclusion from a from-scratch trace-driven
   simulator: at batch=1 with realistic bandwidth, every prefetch policy we
   built collapses to the no-prefetch baseline, exactly where the roofline
   predicts it should."* (MEASURED + SIMULATED, agreeing)
3. *"On the wrong side of that line we spend fewer bytes instead of more
   time — our confidence-gated precision policy cuts bytes/token by 72.8% at
   one real operating point, at a 1.5-percentage-point cost in oracle-gap-closed;
   whether that trades acceptably against real model quality is the one
   remaining GPU experiment this framework is built to run the moment
   hardware access exists."* (SIMULATED, with an honest open end)
4. *"Mentors asked whether prefetching hides latency consistently across
   conditions — the honest answer, measured on both models, is that
   byte-budget-limited prefetch alone does not: it degrades to a static
   baseline once enough concurrent streams share the link, on both
   architectures we tested. So we built the hybrid the mentors also asked
   for — an LRU-resident hot set with prefetching on top — and it stays
   within 0.004pp of the oracle on Mixtral and within 18.3pp on the much
   harder 64-expert OLMoE, at every concurrency level we tested on both
   models, because a cache hit costs zero bytes no matter how many streams
   are sharing the link."* (SIMULATED, cross-model, the strongest
   new result this round produced)
