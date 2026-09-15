# E8 — KV-Cache Co-Tenancy on the CXL Link

Source spec: `Experiments.md` §"E8 — KV-cache co-tenancy on the CXL link [P2]". This file
is the from-first-principles explanation of what this experiment is, why it exists, and
exactly how `kv_link_sim.py` implements it. **Results go in `RESULTS.md`, auto-generated
by `kv_link_sim.py` on every run — don't hand-edit it.**

This experiment was built and run entirely on a laptop (no CUDA, no Mistral/OLMoE model
weights, no network). Everything it needs — a formula derived from public model configs,
and E_union(B) resampled from the pilot's already-collected traces — is either pure
architecture math or already sitting on disk.

---

## 1. Why this experiment exists at all

Every other experiment in this program (E1 through E7) is about one traffic class:
**expert weights** moving from a capacity tier (CXL) to where compute happens (HBM/GPU).
E1 established the central fact of the whole project — that this traffic is so large
relative to link bandwidth that it's usually *bandwidth-bound*, not latency-bound, which
is why precision-gating (E7) matters more than pure prediction accuracy.

But experts aren't the only thing that might live on a CXL tier. **KV cache** — the
running memory of every token's attention keys and values, for every in-flight request —
is the *other* thing that grows without bound and doesn't fit in HBM once context length
gets long. In fact, Astera's own public material about CXL for LLM inference (see
Experiments.md E8's resource list) is framed almost entirely around KV cache, not
experts. A TierAhead report that only ever talks about experts is answering half of the
question a memory-controller company actually cares about.

The interesting part isn't "KV cache can also live on CXL" (obvious, and already covered
by prior work like vLLM/LMCache's CXL offload). The interesting part is: **if you're
already streaming expert weights over that same physical link to serve MoE decode, and
now a new long-context request shows up and needs its entire KV cache written to the same
link before it can produce a token, those two things fight over the same wire.** That's
co-tenancy, and nobody in the roofline (E1) or the prediction work (E4/E5/E7) accounted
for it — they all implicitly assumed the link belonged to expert traffic alone.

---

## 2. First principles: what is a KV cache, and why does it need bytes at all?

### 2.1 The mechanism (why caching exists)

A transformer generates tokens one at a time (autoregressive decoding). At each new
token, every attention layer needs to attend over **every previous token's** key (K) and
value (V) vectors — that's the entire point of attention: "how relevant is each past
token to the one I'm generating now?" Recomputing K and V for the whole prefix at every
single new token would mean the cost of generating token *t* grows with the *square* of
sequence length (attention is already O(context²) per forward pass; redoing it from
scratch every step makes total generation cost O(context³)). So instead, every serving
system **caches** each token's K and V vectors the first time they're computed, and reuses
them for every subsequent decode step. That cache is the "KV cache." It is pure memory,
not weights — it grows every single token, per request, and never shrinks until the
request finishes.

### 2.2 The byte-cost derivation (why the formula is what it is)

For one transformer layer, at one token position, the K and V vectors together have
shape `(num_key_value_heads, head_dim)` each (attention splits the model's hidden
dimension into multiple heads; **KV heads** may be fewer than **query heads** if the model
uses grouped-query attention (GQA) — Mixtral does this, OLMoE doesn't, see §4 below). So
per token, per layer:

```
elements per token per layer = 2 (K and V) * num_key_value_heads * head_dim
```

Summing over every layer in the model (the cache holds one K/V pair *per layer*, not just
the last one — every layer's attention needs its own history):

```
elements per token, whole model = 2 * n_layers * num_key_value_heads * head_dim
```

Multiply by the byte width of one element (2 bytes for fp16, 1 for fp8/int8 — see
`common.model_specs.KV_BYTES_PER_ELEMENT`) to get **bytes per token**:

```
kv_bytes_per_token = 2 * n_layers * num_key_value_heads * head_dim * bytes_per_element
```

This is `ModelSpec.kv_bytes_per_token()`. A request holding `context_len` tokens of
history has used exactly `kv_bytes_per_token * context_len` bytes so far —
`ModelSpec.kv_total_bytes()`, and this product is *exactly* Experiments.md E8 step 1's
formula (`2 * L * n_kv_heads * d_head * precision * context_len`) — the doc states it as
one combined expression; this code just factors out the per-token constant because that
factoring is what makes "total bytes at context length C" a one-line multiplication
instead of restating the whole formula for every C.

### 2.3 A worked example, so the formula isn't just symbols

Mixtral: `n_layers=32`, `num_key_value_heads=8` (GQA — see §4), `head_dim=128`, fp16 (2
bytes):

```
kv_bytes_per_token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KiB/token
```

At a 128k-token context (a real, current-generation context length), one request's KV
cache is `131,072 * 131,072 ≈ 17.2 * 10^9` bytes ≈ **17 GB** — for ONE request. This is
the number that makes "KV cache doesn't fit in HBM" a real, not hypothetical, problem:
serving even a handful of concurrent long-context requests can exceed a single
accelerator's entire HBM capacity on KV cache alone, before a single expert weight is
loaded. That's the physical reason a KV cache would ever need to spill to CXL in the
first place, and it's why Astera's messaging leads with this scenario.

### 2.4 The OLMoE/Mixtral coincidence (and why it's real, not a bug)

Run the same formula for OLMoE (`n_layers=16`, `num_key_value_heads=16` — no GQA, see
§4 — `head_dim=128`, fp16):

```
kv_bytes_per_token = 2 * 16 * 16 * 128 * 2 = 131,072 bytes -- IDENTICAL to Mixtral
```

This isn't a copy-paste error (there's a regression test locking it in,
`test_olmoe_and_mixtral_kv_bytes_per_token_coincidence`): `n_layers * num_key_value_heads`
happens to be 256 for both models (`16*16` and `32*8`), and both share `head_dim=128`.
Two architecturally very different models (GQA vs plain multi-head attention, 16 vs 32
layers) land on the same per-token KV cost purely by arithmetic coincidence. It's a good
illustration of why you verify formulas against real numbers instead of trusting
intuition about "the bigger model must cost more" — and why `model_specs.py`'s
`num_key_value_heads`/`head_dim` values are cited against the *live* HF `config.json` for
each model (fetched and cross-checked 2026-09-13), not guessed from parameter counts.

---

## 3. Why the two traffic classes actually fight

A physical link (CXL, PCIe, whatever the substrate) moves bytes at some bandwidth `BW`
(GB/s). That number is a hard ceiling on the *sum* of every byte stream sharing that link
at any instant — not per traffic class, per link. If expert-prefetch traffic is already
using, say, 90% of the link's capacity to keep decode fed, a new KV-cache write for a
freshly-arrived long-context request has at most 10% of the link left, or must wait for
the expert stream to yield, depending on how the memory controller (or its software
stack) chooses to arbitrate. That arbitration choice is the entire subject of this
experiment.

Two metrics matter, and they're sensitive to opposite things:

- **TTFT (time to first token)**: how long a *new* request waits before it can start
  generating. For a long-context request whose KV cache must first be written to (or
  streamed from) CXL, TTFT is dominated by how fast that KV transfer completes — which
  depends entirely on how much link bandwidth it's given.
- **TPOT (time per output token)**: how fast an *already-running* decode keeps producing
  tokens. In this project's world, TPOT is dominated by how fast non-resident expert
  weights arrive — which depends on how much link bandwidth *expert* traffic is given.

Give the link entirely to one class and you help it at the direct expense of the other.
That's the trade-off Experiments.md E8 step 3 asks for a "priority rule" to navigate.

---

## 4. GQA vs plain multi-head attention (why num_key_value_heads exists as a separate field)

Standard multi-head attention gives every attention head its own K and V projections: if
there are `H` heads, there are `H` independent KV heads, and `num_key_value_heads == H`.
Grouped-query attention (GQA) is a memory-saving trick: keep `H` *query* heads (so the
model's expressive capacity is unchanged) but share K/V projections across groups of
query heads, so there are fewer *distinct* KV heads than query heads
(`num_key_value_heads < num_attention_heads`). Since the KV **cache** only ever stores K
and V (queries are never cached — a query is only needed for the single token being
generated right now, not for history), GQA directly and proportionally shrinks the KV
cache: Mixtral's 8 KV heads out of 32 query heads is a 4x KV-cache reduction versus if it
used plain MHA at the same model size. OLMoE doesn't use GQA at all
(`num_key_value_heads == num_attention_heads == 16`) — this is exactly the axis that
makes the OLMoE/Mixtral coincidence in §2.4 interesting: GQA and layer count are pulling
in different directions but landing on the same number.

---

## 5. What "simulating a serving mix" actually means here

Experiments.md E8 step 2 asks to "simulate a serving mix where long-context KV spills to
CXL while experts are also being prefetched from CXL." Concretely, that means building a
model of:

1. **Two independent streams of "jobs"** (a job = some number of bytes that must cross
   the link), each with its own arrival pattern and job size.
2. **One shared resource** (the link, at some bandwidth `BW`) that can only serve some
   subset of pending jobs at any instant, splitting its capacity according to a
   **scheduling policy**.
3. **Metrics** computed from how long jobs actually take to cross the link, compared to
   how long they'd take with the whole link to themselves.

This is a **discrete-event queueing simulation** — the same category of model used to
reason about network routers, disk schedulers, and (per Experiments.md's own citations)
memory-tiering systems like TPP and Memtis. `kv_link_sim.simulate_link` implements an
**exact event-driven fluid simulation**: instead of stepping through time in fixed
increments (which would either waste computation on quiet intervals or risk missing fast
events), it jumps directly from one event (an arrival, or a job finishing) to the next,
computing exactly how many bytes were transferred in between via `rate * elapsed_time`.
Because the rate a job receives only ever changes at one of those event boundaries, this
introduces **zero discretization error** — a smaller time-step wouldn't produce a more
accurate answer, because there's no approximation to refine in the first place.

### 5.1 The two traffic streams, concretely

- **Expert traffic**: `compute_expert_traffic_params` reuses E1's own machinery
  (`bytes_per_layer`, `get_t_compute_ms`) to ask "how many non-resident expert bytes does
  one decode token need, across every layer, and how much compute time does generating
  one token take?" It packages that into **one job per decode token** (aggregating all
  layers into a single job, and using the token's total compute time as the arrival
  period) — a deliberate coarsening of E1's per-layer model; see §7.1 for why.
- **KV traffic**: modeled as a **Poisson arrival process** (the standard way to model
  "requests show up at random times, independently of each other," used throughout
  queueing theory) of new long-context requests, at a configurable rate
  (`--kv-rate-per-sec`, an explicit **assumption** — Experiments.md doesn't pin a number,
  and this codebase doesn't have real serving-trace data to fit one from). Each arriving
  request's job size is `ModelSpec.kv_total_bytes(context_len)` for a context length drawn
  from `--context-lens` (default 4k/32k/128k, per Experiments.md's own "realistic context
  lengths").

### 5.2 Scheduling policies, unified as one weighted split

Experiments.md names three policies: "KV-first," "expert-first," and "weighted."
`make_rate_fn` implements all three as **one function**: whenever both classes have a job
queued, KV gets a fraction `w` of the link and expert gets `1-w`. `expert-first` is just
`w=0`; `kv-first` is just `w=1`; everything in between is the general weighted case. This
isn't a simplification that loses anything — it's recognizing that the two "strict
priority" policies are the boundary cases of the same underlying mechanism, which is why
the code represents them that way instead of as three separate branches of logic.
Whichever class is the *only* one with something queued always gets the full link — a
scheduler doesn't leave bandwidth idle just because only one class currently needs it.

### 5.3 Reading the two output metrics

- **TTFT (this experiment's version)**: the time from a KV job's arrival to its
  completion. This is explicitly **not** full end-to-end TTFT (which also includes
  prefill compute, request queueing before the KV write even starts, etc.) — it's the
  *link-contention component* of TTFT, which is the only part this simulation can speak
  to. `RESULTS.md` says this every time so it's never silently conflated with the real
  number a serving system would report.
- **TPOT inflation**: for each expert job, `latency - ideal`, where `ideal = bytes / BW`
  (how long the transfer would take with the whole link to itself). This isolates the
  *extra* delay contention adds on top of whatever E1 already predicts for that
  (batch, residency, precision, bandwidth) operating point — it does not redo E1's own
  compute/transfer-overlap analysis, it adds one more term on top of it.

### 5.4 The Pareto frontier (why it's the right way to compare policies)

Once you have (TTFT, TPOT-inflation) for every policy, there usually isn't one strictly
best answer — reducing one tends to increase the other. A policy is **Pareto-optimal** if
no other policy improves *both* metrics simultaneously; the set of all such policies is
the **Pareto frontier**, and any policy off that frontier is just a worse choice than
something else on it, full stop. `pareto_labels` implements this exactly (checking every
pair for strict dominance), and `PARETO_ROUND_DECIMALS` rounds both metrics to
microsecond precision before comparing — without that, two policies that are both
"effectively zero" TPOT inflation could still differ by ~10⁻¹² ms of pure floating-point
residue from the event-driven arithmetic, and the dominance check would crown one of them
a "winner" over a difference no real system could ever observe. That was an actual
artifact caught by inspecting a real run's output (see §7.3) — not a hypothetical.

---

## 6. What's actually implemented

| Experiments.md E8 step | File / function | Notes |
|---|---|---|
| 1. KV bytes/token formula, swept over realistic context lengths | `common.model_specs.ModelSpec.kv_bytes_per_token` / `.kv_total_bytes`, swept via `--context-lens` (default 4k/32k/131k) | Derived from architecture (§2.2), cross-checked against live HF `config.json` (§2.4), not hardcoded from the doc's prose. |
| 2. Simulate serving mix, shared-link contention, priority policies | `kv_link_sim.simulate_link` (the discrete-event fluid simulator, §5), `make_rate_fn` (policy unification, §5.2) | Self-contained; does not depend on E6 (doesn't exist yet — see §7.2). |
| 3. Report TTFT/TPOT, find the Pareto-dominant rule | `run_sweep`, `add_pareto_column`, `pareto_labels`, `recommend_policy` | See §5.4 and §7.4 for why "the" recommendation is conditional on an SLO rather than one unconditional answer. |

---

## 7. Design choices, honestly labeled (things a reviewer should know before trusting a number)

### 7.1 Aggregating all layers into one expert "job" per token

E1's own model is per-layer: a fetch decision is made, and a byte cost paid, at every
individual layer. A fully faithful co-tenancy simulation would interleave 16-32
per-layer expert events with KV events at a comparable timescale. In practice, per-layer
expert arrivals happen roughly every 0.01-0.4 ms (depending on model/batch), while KV
arrivals (new long-context requests) happen on the order of hundreds of milliseconds to
seconds apart. Simulating both at native per-layer granularity for long enough to collect
a statistically meaningful number of KV completions would mean tens of millions of
expert-layer events per run — intractable in pure Python (no SimPy dependency was added;
see §7.2) within a reasonable wall-clock budget on a laptop. Coarsening to **one job per
decode token** (all layers' bytes and all layers' compute time summed together) keeps the
two streams' timescales close enough to simulate jointly for tens of seconds of simulated
time in well under a minute of wall-clock, at the cost of losing *within-token* per-layer
interleaving detail. This is a legitimate first-cut simplification, not a hidden one —
`compute_expert_traffic_params`'s docstring states it explicitly.

### 7.2 Why this doesn't reuse E6 (even though Experiments.md's own dependency graph says it should)

Experiments.md's phrasing for E8 is "simulator reuse," implicitly assuming E6 (the full
policy bake-off simulator, with its own shared-bandwidth link model) already exists. As
of this experiment, only E1 and E2 have been built — E6 hasn't. Rather than block E8 on a
much larger piece of infrastructure, this experiment implements its own minimal
discrete-event link simulator, purpose-built for exactly the two-traffic-class question
E8 asks. If/when E6 is built, the honest path is to fold E8's KV traffic class into E6's
link model (which Experiments.md's own E6 section already flags needs "an explicit
shared-bandwidth link model") rather than maintain two separate simulators long-term.

### 7.3 Bugs the tests actually caught (kept here, not just in a commit message, because they change how much to trust a number)

- **Unbounded horizon at near-zero bandwidth.** The first version of `simulate_link` had
  no check that an event's timestamp stayed within `sim_ms` — at a pathologically low
  bandwidth, "time until this job finishes" is a valid (if enormous) number, and the loop
  would happily jump there and record a "completion" far outside the window it was
  supposedly measuring. `test_zero_link_bandwidth_never_completes_anything_within_horizon`
  caught this. Fixed by right-censoring: any job still in flight when the next event would
  cross `sim_ms` is simply excluded from the stats, not force-completed.
- **A single-seed phase-lock artifact that looked like a real (and backwards) finding.**
  An early single-seed run showed `weighted:0.5` producing *more* expert TPOT inflation
  than `weighted:0.8` — i.e., giving KV *more* priority looked like it hurt expert traffic
  *less*, which would have been a genuinely interesting and reportable non-monotonicity if
  real. It wasn't: the perfectly-periodic expert arrival stream and a single realization
  of the random KV arrival process can phase-lock in a way that's an artifact of that one
  seed's random draw, not a property of the system. Pooling raw completion records across
  `--n-seeds` (default 5) independent seeds before computing any percentile — not
  averaging per-seed percentiles, which would be statistically the wrong operation —
  restored the expected monotonic relationship (more KV priority → more expert
  inflation). `pooled_simulate`'s docstring keeps this story attached to the code, not
  just to a commit message, because it's exactly the kind of number a first read of a
  single-seed run would have reported confidently and wrongly.
- **A scalarization that always gave the same answer regardless of the data.** The first
  "pick one policy" rule minimized TPOT inflation first, using TTFT only as a tie-break.
  `expert-first` is *structurally* guaranteed to hit the lowest possible inflation (it
  never yields the link, by construction) — so that rule picked `expert-first` in every
  single case, independent of how catastrophic its TTFT got (over 8 *seconds* p50 in one
  swept point). A recommendation that never changes isn't a recommendation; it's a
  hardcoded answer wearing a scoring function as a costume.
  `test_recommend_policy_does_not_trivially_always_pick_expert_first` locks in that the
  final design (§7.4) doesn't have this failure mode.

### 7.4 Why there's no single unconditional "best policy"

`recommend_policy` follows the exact pattern E2 already uses for its own B* selection: an
optional SLO constraint (`--ttft-slo-ms`, mirroring E2's `--slo-ms`) turns the question
into "minimize the metric you actually care about protecting (TPOT), subject to a hard
limit on the other one (TTFT)" — which is how a real deployment would actually pose the
question, because it has a concrete TTFT budget from its own product requirements. Without
an SLO, collapsing a genuine two-metric trade-off into one scalar number requires an
arbitrary weighting between "a slower first token" and "a slower decode" that this
experiment has no principled basis for choosing (see the two rejected scalarizations in
§7.3) — so the default behavior instead reports both Pareto extremes (best-for-TTFT,
best-for-TPOT) and lets the reader apply their own weighting.

### 7.5 The default operating point matters — and was picked deliberately

The very first run of this experiment used `batch=1, residency=25%, expert precision=fp16`
— and produced expert jobs so large relative to the link (>1 GB every ~1 ms) that the
expert queue backlog grew without bound over the whole 30-second window on its own,
*before any KV traffic was added*. That's E1's own bandwidth-bound regime (ρ≫1) taken to
an extreme, and it makes the co-tenancy question moot: if the link can't keep up with
expert traffic alone, adding KV traffic doesn't reveal a new trade-off, it just makes an
already-broken regime worse in an uninteresting, numerically extreme way (TPOT inflation
values in the thousands of milliseconds per token). The defaults were changed to
`batch=32, residency=75%, expert precision=nf4` specifically because E1's own
`roofline_summary.json` (`headline_sentences`) found this to be the operating point where
"router-guided prefetch is worth something" (ρ<1) for *both* models within this script's
default bandwidth grid — i.e., a regime where expert traffic alone is viable, so any
contention effect visible in the results is actually attributable to KV traffic, not to
an already-saturated link. Pass `--expert-precision fp16 --batch 1` deliberately if you
want to see the degenerate saturated case instead.

### 7.6 KV precision options (fp16/fp8, not NF4)

`common.model_specs.KV_BYTES_PER_ELEMENT` is deliberately a separate table from
`NF4_BYTE_FACTOR`. NF4 (used for expert weights throughout this project, and in E7) is a
*weight*-quantization scheme: blockwise 4-bit codes plus a quantized per-block scale
factor, designed for values that are set once and never change. KV-cache entries are
*activations* computed fresh per token — nobody quantizes them with NF4 in practice. Real
systems that shrink KV-cache precision (vLLM's FP8 KV cache is the standard example) use
fp8 or int8, which is why that's what this experiment offers instead.

---

## 8. What this simulator does NOT model (limitations, stated up front)

- **No prefill compute.** TTFT here is purely the KV link-transfer component. Real TTFT
  also includes running the prefill forward pass itself, which this experiment doesn't
  simulate.
- **No admission control or backpressure.** KV jobs arrive and queue indefinitely; a real
  system might reject or delay admission under sustained overload rather than let a queue
  grow arbitrarily.
- **A single link, single-server-per-class model.** Real CXL fabrics may have multiple
  channels/links; this model treats "the link" as one shared pipe, consistent with
  Experiments.md's own framing ("shared-bandwidth link model" in E6).
- **`--kv-rate-per-sec` is an assumption, not a measurement.** Experiments.md doesn't pin
  a number, and this codebase has no real multi-tenant serving trace to fit one from.
  `RESULTS.md`'s analysis-notes checklist calls this out explicitly as something to
  sensitivity-test before quoting a single number in the final report.
- **One representative (batch, residency, precision) operating point per run.** A full
  characterization would sweep these the way E1 does; this experiment sweeps bandwidth
  and policy, and picks one point on the other axes (see §7.5) rather than the full
  Cartesian product, to keep runtime and output size manageable for a first cut.

---

## 9. How to run it

```bash
# from the repo root -- creates experiments/.venv-mac on first run (numpy/pandas/
# matplotlib inherited from system Python via --system-site-packages; only
# zstandard + pytest get pip-installed fresh)
bash experiments/e8_kv_cotenancy/run_e8.sh

# with an explicit TTFT SLO (collapses the Pareto report to one recommendation):
python3 experiments/e8_kv_cotenancy/kv_link_sim.py --ttft-slo-ms 300

# sweep the full link-bandwidth grid instead of the default {32,64} GB/s subset:
BW_GRID=16,32,64,128 bash experiments/e8_kv_cotenancy/run_e8.sh
```

No GPU, no model download, no `SKIP_CALIB`/`SKIP_DOWNLOAD` flags needed (unlike
`run_e1.sh`) — this is the one experiment in the program designed to run unmodified on a
laptop.

## 10. Exit criteria (Experiments.md's own wording)

> "A stated priority rule with quantified TTFT/TPOT trade-off."

`RESULTS.md`'s "Recommendation" section satisfies this two ways: an unconditional report
of the Pareto frontier and its two extremes (always produced), and — once a real TTFT
budget is known — a single stated rule via `--ttft-slo-ms` (§7.4). The headline numbers to
quote in the report are the two Pareto-extreme rows at the bandwidth tier closest to
whatever link width the report is recommending for Leo-class deployments (cross-reference
E1's provisioning table for that number).

## 11. What to look at after running

1. `RESULTS.md`'s policy sweep table — at the lower swept bandwidth, is the TTFT spread
   between `expert-first` and `kv-first` dramatic (seconds, not milliseconds)? That's the
   headline "these two traffic classes genuinely fight" result.
2. `out/figures/kv_expert_contention_{model}.pdf` — at the higher bandwidth, do the points
   collapse toward the y-axis (TPOT inflation ≈ 0 for every policy)? That's the "once the
   link has headroom, prioritizing KV is nearly free" result — check whether it holds for
   both models or only one.
3. Cross-reference against `e1_roofline/RESULTS.md`: is the (batch, residency, precision)
   point used here (§7.5) actually on the viable side of E1's own roofline? If a future
   run changes those defaults into the bandwidth-bound regime, expect degenerate,
   E1-already-explains-this numbers rather than a genuine co-tenancy finding.
