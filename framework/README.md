# tierMoE

**Router-guided memory tiering for cost-efficient MoE inference on HBM + CXL memory systems.**

tierMoE is the production framework for the *Nebula@BITS Goa 2026* (Astera Labs)
"CXL-Based Memory Optimization for MoE Models" project. Mixture-of-Experts (MoE)
models decouple parameter count from per-token compute — Mixtral-8x7B needs 87 GB
of FP16 expert weights but an H100 has 80 GB of HBM — which forces expert
offloading and turns inference from memory-bound into I/O-bound. This framework
answers two questions with numbers instead of guesses:

1. **When does a CXL capacity tier actually help?** (the *MoE Tiering Roofline*:
   ρ = transfer time / compute time — prefetching hides latency, it does not
   create bandwidth, so viability is set by expert granularity and batch size,
   not by prediction accuracy).
2. **What do you do about it?** — router-guided prefetch where ρ < 1, and
   confidence-gated precision fallback (fetch low-confidence candidates at
   4-bit instead of not at all) where ρ ≥ 1.

Everything here is either a **real number from an already-collected trace**, a
**real simulation this framework actually runs**, or an **explicitly labeled
projection from a cited paper** — never a guess presented as a measurement. See
[`PREDICTED.md`](PREDICTED.md) for the full ledger of which is which.

## Why this repository doesn't need a GPU to be useful

This machine has no CUDA. That is the normal, fully-supported case for almost
everything here — the original research program already proved this: E1
(roofline), E2 (batching), E4 (predictor), E5 (lookahead), and E8 (KV
co-tenancy) were all run for real on a laptop with no CUDA at all, against
already-collected traces. This framework inherits and generalizes that
discipline. Run `tiermoe doctor` to see exactly what this means on your
machine:

```
$ tiermoe doctor
platform:            Darwin (arm64)
CUDA available:      False
MPS available:       True
torch installed:     True
Linux perf available:False
CXLMemSim binary:    not found on PATH

capability            -> strategy on this machine
------------------------------------------------------------
trace collection      -> UNAVAILABLE here -- use the pilot committed traces under results/, or run tiermoe collect on a CUDA box
GPU compute calib     -> flop_estimate fallback (roofline/sim tag every number compute_provenance=flop_estimate)
MLP predictor         -> CPU/MPS -- runs here, no CUDA needed
DES / analytical sim  -> ALWAYS available (pure Python, this machine included)
CXLMemSim backend     -> UNAVAILABLE here (needs Linux + perf + CXLMemSim built from source) -- sim falls back to the des/analytical backend, tagged accordingly
```

Only two things genuinely require a CUDA box: **collecting new router-decision
traces** (`tiermoe.collect`) and **real per-layer GPU compute-time
calibration** (`tiermoe.calibrate.compute` / `.transfer_measure` /
`.e2e_offload` / `.nf4_fidelity`). Everything else — characterization, the
roofline, every policy in the simulator, KV co-tenancy, pooling/TCO, and the
dashboard — runs natively on a laptop, using the traces this repository
already ships under `../results/{olmoe,mixtral}/b1/`.

## Architecture

Eight components, matching the original project's own "Framework
Architecture" design one-to-one, plus the flagship baseline comparison:

```
                    normal  (HBM-only, no CXL tier at all)
                        │
tiermoe.baseline.compare──── the flagship "is this even feasible, and what
                        │     does tiering cost/save" entry point
                        ▼
  results/*/b1/  ──►  [1] tiermoe.collect        (CUDA-required)
  traces.jsonl.zst          │
                             ▼
                     [2] tiermoe.analyze          Coverage@C / Gini / churn /
                             │                     Recall@m,d / HTB
                             ▼
                     [0] tiermoe.roofline         rho = t_transfer / t_compute
                             │
              ┌──────────────┼───────────────────────┐
              ▼              ▼                       ▼
   [3] tiermoe.policy.  [4] tiermoe.policy.   [4b] tiermoe.policy.
       residency            predictor_freq /        precision_gate
       (fixed-C)            predictor_mlp
              │              │                       │
              └──────────────┴───────────┬───────────┘
                                          ▼
                            [5] tiermoe.sim  (policy bake-off + N-class DES)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
             [8] tiermoe.kv        [6] tiermoe.tco    tiermoe.sim.backends
             (KV co-tenancy)       (pooling & TCO)     analytical / des / cxlmemsim
                                                                  │
                                                       [7] tiermoe.dashboard
```

| # | Package | What it does | Needs GPU? |
|---|---|---|---|
| 0 | `tiermoe.roofline` | ρ = t_transfer/t_compute regime model, provisioning table | No |
| 1 | `tiermoe.collect` | Forward-hook router-decision trace collector | **Yes** |
| 2 | `tiermoe.analyze` | Coverage@C, Gini/Zipf, churn, Recall@m,d, HTB, verdict | No |
| 3 | `tiermoe.policy.residency` | Fixed-residency placement, shared-expert pinning | No |
| 4 | `tiermoe.policy.predictor_freq` / `predictor_mlp` | Prefetch engine (frequency table + trainable MLP) | No (CPU/MPS) |
| 4b | `tiermoe.policy.precision_gate` | Confidence-gated FP16/NF4/skip fallback | No |
| 5 | `tiermoe.sim` | Trace-driven policy bake-off + general N-class fluid DES | No |
| 6 | `tiermoe.tco` | Pond-style pooling & stranding, tokens/sec/$ crossover | No |
| 7 | `tiermoe.dashboard` | Streamlit + Plotly interactive explorer, 4 tabs | No |
| 8 | `tiermoe.kv` | KV-cache/expert link co-tenancy simulation | No |
| — | `tiermoe.baseline` | **Flagship**: normal (HBM-only) vs CXL-tiered, side by side | No |
| — | `tiermoe.calibrate` | Hardware calibration ladder (compute/transfer/e2e/NF4) | **Yes** (except `policy_check`/`report`) |

## Install

```bash
cd framework
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[predictor,dashboard,parquet,dev]"
```

Or simply `make setup`. Extras are separated deliberately (`pyproject.toml`)
so a minimal install (`pip install -e .`) needs only numpy/pandas/scipy/
matplotlib/zstandard/pyyaml — no torch, no streamlit — for anyone who only
wants the roofline/analyze/policy-residency/precision-gate/TCO modules.

By default the framework auto-detects the data root (this repository's parent
directory, which holds `results/` and `experiments/`) by walking upward from
wherever it's installed. Override with `TIERMOE_DATA_ROOT=/path/to/Nebula/Code`
or `--data-root` on any CLI command if you've moved `framework/` elsewhere.

## Quickstart

```bash
# 1. What can run on this machine?
tiermoe doctor

# 2. Prove it: run the full test suite against the real committed traces.
pytest tests/ -v          # ~7 min (mostly Mixtral's 71k-row CPU MLP training)

# 3. The flagship comparison: normal (HBM-only) vs CXL-tiered.
tiermoe baseline --model olmoe   --residency-pct 25 --bw-gbps 128 --precision fp16 --depth 4
tiermoe baseline --model mixtral --residency-pct 75 --bw-gbps 128 --precision nf4  --depth 8

# 4. Everything else in one shot.
bash scripts/run_end_to_end.sh

# 5. Interactive dashboard (needs the `dashboard` extra).
tiermoe dashboard
```

`make baseline`, `make sim`, `make roofline`, `make validate`, `make
dashboard` are shortcuts for the equivalent `tiermoe ...` invocations.

### Why `--residency-pct 25 --bw-gbps 128 --depth 4` for OLMoE and `--residency-pct 75 --bw-gbps 128 --depth 8` for Mixtral?

These are not arbitrary — they are the operating points the original roofline
and lookahead-depth experiments (E1, E5) already established as necessary for
each model's expert granularity. Pick a point outside these regimes (e.g.
batch=1, `--depth 1`, `--bw-gbps 32`) and every prefetch-based policy
collapses to identical numbers as `static-c`, because the predicted candidate
set genuinely cannot arrive in time — that is not a bug, it is the project's
own central finding (ρ ≥ 1 ⇒ only byte reduction helps) reproduced by an
independently-built simulator. `tiermoe sim --model <m> --depth <d>` lets you
explore this yourself; watch `demand_stall_rate` collapse to `1.0` for every
non-`lru`/`oracle` policy the moment the candidate set stops fitting the
lookahead window.

## Component usage

### `tiermoe roofline` — the regime model

```bash
tiermoe roofline --model olmoe
tiermoe roofline --model mixtral --calib /path/to/calib_compute.json --out sweep.csv
```

Prints the auto-derived headline sentence per (model, precision) — e.g. *"olmoe
(nf4): router-guided prefetch is worth something (rho<1) once residency >= 75%
on a >=32 GB/s link at batch >= 32"* — and, if a calibration file is supplied,
runs it through `tiermoe.roofline.model.validate_calib_sanity` first (see
[PREDICTED.md](PREDICTED.md)'s Methodology section for why that check exists
and what it already caught in this repository).

### `tiermoe analyze` — workload characterization

```bash
tiermoe analyze --model olmoe --out olmoe_characterization.json
```

Reproduces Coverage@C, Gini/Zipf skew, hot-set churn, Recall@m,d, and the
go/no-go verdict against the pre-registered thresholds — the exact numbers in
`PILOT_FINDINGS.md`, recomputed from the same committed traces through this
framework's own (ported, not reimplemented-from-scratch) pipeline.

### `tiermoe sim` — the policy bake-off

```bash
tiermoe sim --model olmoe --residency-pct 25 --bw-gbps 128 --depth 4 \
  --precision fp16 --out olmoe_bakeoff.csv

# Sweep concurrent decode streams instead of a single run:
tiermoe sim --model olmoe --residency-pct 25 --bw-gbps 128 --depth 4 \
  --precision fp16 --concurrency-grid 1,2,4,8,16
```

Walks the held-out test-split trace layer by layer and evaluates nine
policies on identical data:

| Policy | What it does |
|---|---|
| `none` | No tiering at all — every non-resident access stalls. HBM-only floor. |
| `static-c` | Fixed residency set, ranked by **training-set popularity** — the "static + popularity-aware baseline" the mentors asked for. |
| `popularity-prefetch` | Prefetches the top-m globally most-popular non-resident experts each layer, **unconditioned on current router state** — isolates how much value conditioning (below) actually adds over popularity alone. |
| `lru` | Dynamic recency-based cache, capacity = the configured residency budget (the *only* residency mechanism for this policy — see the note below on a bug this had earlier). |
| `hybrid-lru-prefetch` | **The hybrid tiering+prefetching strategy**: LRU-resident hot set, and on an LRU miss, predicts/prefetches using the same conditional model `prefetch-freq` uses, inserting the fetched expert into the LRU so future accesses can hit it. |
| `prefetch-freq` | Conditional next-layer prediction via a frequency table (`predictor_freq`). |
| `prefetch-v2` | Same, via a trainable MLP (needs the `predictor` extra). |
| `prefetch-v2+precision-gate` | `prefetch-v2` plus confidence-gated NF4/FP16 fallback on low-confidence candidates. |
| `oracle` | Perfect foreknowledge — the upper bound every other policy is judged against. |

Reports TPOT mean/p50/p99, demand-stall rate, bytes/token,
`oracle_gap_closed_pct`, and — for every policy that actually predicts a
candidate set (`popularity-prefetch`, `hybrid-lru-prefetch`, `prefetch-freq`,
`prefetch-v2`, `prefetch-v2+precision-gate`) — **`prefetch_precision` and
`prefetch_recall`** (mentor ask: report precision alongside recall, since high
recall alone is free to buy by fetching more candidates at a precision cost).
Note: the *overall* precision/recall pair is an exact algebraic rescaling of
each other by the constant `k/m` (both denominators are data-independent) —
the genuinely-informative comparison is the *non-resident* pair, whose
denominator varies with real residency overlap; see
`tiermoe.analyze.workload.recall_at_m`'s docstring for the full derivation and
a hand-worked proof.

**A bug fixed during the mentor-feedback round, worth knowing about if you
touch this code:** `lru` originally checked misses against the static
residency set *and then* a same-sized LRU cache underneath — silently giving
it double the memory budget of every other policy at the same
`--residency-pct`. Fixed: `lru`'s cache capacity now *is* the entire
configured residency budget, nothing else. If you see old numbers anywhere
claiming `lru` closes ~100% of the oracle gap at the *same* nominal residency
as `static-c`, they're from before this fix and are stale.

`tiermoe validate --model <m>` runs the seven validation gates this bake-off
must satisfy: the original four (oracle is always the upper bound; residency
100% + infinite bandwidth reproduces an HBM-only baseline exactly; more
residency never hurts; CXL added-latency sensitivity barely moves the result)
plus three added for the concurrency work (oracle's TPOT is invariant to
concurrency; `static-c`'s TPOT is monotonically non-decreasing in
concurrency, per the closed-loop bw/N model; `hybrid-lru-prefetch` never has
worse TPOT than plain `lru` at any concurrency level, not just
concurrency=1). It also reports a non-pass/fail `concurrency_collapse_analysis`:
the smallest concurrency at which a byte-budget-limited prefetch policy
becomes numerically indistinguishable from `static-c` — the same finding
PREDICTED.md §2.5 documents by hand for two operating points, now computed
for any trace on demand (`tiermoe.sim.validation.find_collapse_concurrency`).

#### Concurrent inference (mentor ask: "evaluate under concurrent inference where CXL bandwidth contention becomes a real factor")

`--concurrency N` models N decode sessions sharing one physical link. Because
a real decode session is **closed-loop / self-throttling** (it cannot ask for
layer *l+1*'s experts before layer *l*'s fetch has arrived), the textbook-correct
model here is egalitarian fair-share: each session realizes `bw_gbps /
concurrency`. This is what `--concurrency` divides by. `tiermoe.sim.concurrency`
provides `max_sustainable_concurrency(...)`: the concurrency level at which the
link — not any per-stream compute — becomes the *persistent* bottleneck for
every stream. Below that level, the fair-share number is a **conservative**
estimate (real TPOT is plausibly better); at or above it, fair-share is the
**accurate** theoretical answer, because persistent contention is now the
norm the theory assumes. When `concurrency > 1`, every policy's bake-off row
carries an `extra["concurrency_stable"]` flag reflecting exactly this
boundary. The module docstring in `tiermoe/sim/concurrency.py` also documents
a *second*, different, open-loop question ("how many new sessions/sec can
this link admit before a startup backlog diverges") answered separately by
`verify_stability_boundary_with_des` — deliberately not used to validate the
closed-loop fair-share model (an earlier draft made exactly that mistake; the
docstring explains why it doesn't apply and is worth reading before changing
either function).

### `tiermoe baseline` — the flagship comparison

```bash
tiermoe baseline --model mixtral --precision fp16 --hbm-budget-gb 80

# With concurrent decode streams AND KV-cache HBM pressure from long-context requests:
tiermoe baseline --model olmoe --residency-pct 25 --bw-gbps 128 --precision fp16 --depth 4 \
  --concurrency 4 --concurrent-requests 4 --context-len 131072

# One call, every concurrency level at once (tiermoe.baseline.compare.run_flagship_report):
tiermoe baseline --model olmoe --residency-pct 25 --bw-gbps 128 --precision fp16 --depth 4 \
  --concurrency-grid 1,2,4,8 --concurrent-requests 4 --context-len 131072 --out olmoe_flagship.csv
```

`--concurrency-grid` runs the complete normal-vs-CXL comparison (capacity
check, KV pressure, policy bake-off) once per concurrency level and stacks
every level's `policies_df` into one table — the single-call version of what
otherwise needs one manual `tiermoe baseline` invocation per concurrency
level to see the full picture.

Checks capacity first (`does the model even fit in HBM alone?`) and only then
reports the tiering trade-off. For Mixtral FP16 the answer is *no* — 87 GB
against an 80 GB H100 — so the headline correctly says CXL tiering "isn't an
optimization here, it's the only way to serve this model," rather than quoting
a TPOT-overhead number as if this were a normal trade-off. This **is** the
"ensure HBM-only vs HBM+CXL comparison is actually done end-to-end" mentor
ask: `run_baseline_vs_cxl` always runs both sides — an HBM-only feasibility
check and, where relevant, the CXL-tiered policy bake-off — side by side in
one call.

#### KV-cache HBM pressure (mentor ask: "account for HBM pressure from KV cache competing with prefetched experts")

`--context-len` (tokens of KV-cache history per request) and
`--concurrent-requests` (how many such requests are live at once) turn on KV
accounting via `tiermoe.baseline.compare.kv_hbm_pressure`; leaving
`--context-len 0` (the default) skips it entirely, preserving the original
single-request, experts-only behavior exactly. When enabled, the requested
`--residency-pct` is capped by however much HBM the KV cache hasn't already
claimed (`effective_residency_pct` in the output, alongside the uncapped
`residency_pct` you asked for), and if KV cache alone exceeds the whole
`--hbm-budget-gb`, the headline says so explicitly (residency ceiling drops to
0% — every expert access becomes a CXL fetch, no exceptions). Both OLMoE and
Mixtral have `kv_bytes_per_token = 131072` bytes exactly (128 KiB/token — a
real architectural coincidence: OLMoE's 16 layers × 16 KV heads and Mixtral's
32 layers × 8 GQA heads both land on 256), so at a 128k-token context that's
~17.18 GB *per request* — e.g. against an 80 GB budget, 4 concurrent
128k-context OLMoE requests already cap achievable residency at ~87.4%, and 5
concurrent requests alone exceed the entire budget before a single expert
weight is loaded. `--kv-precision {fp16,fp8}` controls the KV-side precision
assumption independent of `--precision` (which is the expert-weight
precision).

### `tiermoe kv` — KV-cache / expert co-tenancy

```bash
tiermoe kv --models olmoe,mixtral --bw-grid 32,64 --policies expert-first,kv-first,weighted:0.5
```

### `tiermoe tco` — pooling & TCO

```bash
tiermoe tco --n-hosts 128 --group-sizes 1,8,32 --cv 0.4
```

### `tiermoe report` — a shareable Markdown summary

```bash
tiermoe report --model olmoe --residency-pct 25 --bw-gbps 128 --depth 4 --out olmoe_report.md
```

Combines characterization, roofline headlines (with calibration-sanity
warnings inline if any), and the policy bake-off table into one Markdown
file. Re-running it preserves any hand-written `## Analysis` section below
the generated content instead of overwriting it — the same trick the
original per-experiment `RESULTS.md` files used.

### `tiermoe dashboard`

Four tabs: **Routing explorer** (animated per-request expert heatmap from a
real trace), **Roofline explorer** (live ρ=1 boundary with sliders — the demo
centerpiece), **Policy explorer** (live bake-off gauges, precision-vs-recall
chart, and an opt-in "Show concurrency sweep" checkbox plotting TPOT vs
concurrency per policy on a log-log scale — the same collapse curve
PREDICTED.md §2.5 documents by hand, explorable live), **Fleet & TCO**
(pooling curves with a cost-ratio slider). Needs `pip install
tiermoe[dashboard]`. Smoke-tested headlessly via Streamlit's `AppTest`
harness (`from streamlit.testing.v1 import AppTest`) — 0 exceptions across
every tab's default render, no browser needed.

### `tiermoe collect` / `tiermoe.calibrate.*` — CUDA-required

Every one of these refuses immediately with an actionable message on a
non-CUDA machine (via `tiermoe.hw.probe()`), rather than failing deep inside a
`torch.cuda` call:

```bash
# On a CUDA box, from the repo root (not framework/):
python3 -m tiermoe.collect.collector --model olmoe --out results/olmoe/b1
python3 -m tiermoe.calibrate.compute --model olmoe --out calib_compute.json
python3 -m tiermoe.calibrate.compute --model mixtral --quant nf4 --out calib_compute.json
python3 -m tiermoe.calibrate.transfer_measure --out calib_transfer.json
python3 -m tiermoe.calibrate.e2e_offload --residency-pcts 25,50 --out calib_e2e.json
python3 -m tiermoe.calibrate.nf4_fidelity --n-prompts 20 --out calib_nf4_fidelity.json
```

Then assemble the unified provenance table (no GPU needed for this step):

```python
from tiermoe.calibrate.report import build_calibration_report
report = build_calibration_report(compute_path=..., transfer_path=..., e2e_path=..., nf4_path=...)
```

**Before trusting a `calib_compute.json` you didn't produce yourself**, run it
through `tiermoe.roofline.model.validate_calib_sanity` — this repository's own
shipped `experiments/e1_roofline/out/calib_compute.json` fails that check for
both models (near-zero timing variance across batch sizes and across layers),
matching a fabrication incident this project already documented once before.
See [PREDICTED.md](PREDICTED.md) for the full story.

## The three simulation backends

`tiermoe.sim.backends.select_backend("auto"|"analytical"|"des"|"cxlmemsim")`:

- **`analytical`** — pure ρ-ratio math. Instant, coarsest fidelity.
- **`des`** (default, always available) — this framework's own trace-driven
  discrete-event policy bake-off. Every number in `PREDICTED.md`'s Verified
  section that isn't from the original pilot/E1–E8 experiments comes from
  this backend.
- **`cxlmemsim`** — adapter to the real
  [SlugLab/CXLMemSim](https://github.com/SlugLab/CXLMemSim) tool
  (arXiv:2303.06153), which traces a real running process's memory accesses
  via Linux `perf_event_open` and injects real epoch-based CXL timing delays.
  Needs Linux + perf + a from-source build — **unavailable on this machine**,
  and `select_backend` raises a clear, actionable error rather than silently
  falling back when you explicitly ask for it. See
  [`demo/cxlmemsim/README.md`](demo/cxlmemsim/README.md) for exactly how to
  run this on a Linux box and feed its output back into this framework.

This is "our baseline being the normal and our one for CXL simulation using
CXLMemSim" made concrete: `tiermoe.baseline.compare.run_baseline_vs_cxl(...,
backend="cxlmemsim")` is the same call as `backend="des"`, just routed to a
different, higher-fidelity (and platform-restricted) engine.

## Testing

```bash
pytest tests/ -v
```

120 tests, all passing on this machine (650s / ~11 min full run, mostly
Mixtral's 71k-row CPU MLP training and the DES-based concurrency validation
gates), covering:
architecture-spec cross-checks, the E_union resampling math, the roofline
(including a regression test that reproduces the exact "flat batch-scaling"
fabrication signature this repo's own calibration file exhibits), the full
characterization pipeline against real committed traces (values cross-checked
against `PILOT_FINDINGS.md` to 1%, including the precision/recall identity
proofs added for the mentor-feedback round), fixed-residency and
precision-gate policy logic, the generalized N-class DES engine (including
the right-censoring and multi-seed-pooling fixes carried over from the
original KV co-tenancy work, plus phase-offset arrivals), the full 9-policy
bake-off and all seven E6-style validation gates (four original plus three
concurrency gates) on both shipped models (including the `lru`
double-capacity regression test, the new
`popularity-prefetch`/`hybrid-lru-prefetch` policies, and concurrency
tagging), the closed-loop-vs-open-loop concurrency boundary
(`tiermoe.sim.concurrency`), KV-cache HBM pressure accounting, KV/expert
co-tenancy, TCO pooling, the end-to-end baseline comparison, the flagship
report's single-call concurrency sweep, and the pure-math halves of the
calibration ladder. No test requires network access,
a GPU, or Linux-only tooling.

**Performance note:** `run_policy_bakeoff` used to reload and re-split the
trace file from disk on every single call — harmless for a one-off `tiermoe
sim` invocation, but every concurrency-grid sweep and validation gate calls
it once per grid point, so this was pure waste stacking up. It's now
memoized (`tiermoe.sim.policies._load_and_split_decode`, keyed by
`(trace_dir, seed)`); adding the three concurrency validation gates made this
worth fixing — `tests/test_sim_validation.py` alone dropped from 838s to
418s (~2x) after the fix, with byte-for-byte identical results.

## Configuration

`configs/tiers.yaml` and `configs/models.yaml` document the same constants
`tiermoe/specs/{tiers,models}.py` define in code (the code is the source of
truth the framework actually imports; the YAML files are for anyone auditing
assumptions without reading Python, and as a template for wiring in your own
`--tiers-config`/`--models-config` override later).

## Project structure

```
framework/
├── README.md            <- you are here
├── PREDICTED.md          <- verified vs. simulated vs. projected, with numbers
├── pyproject.toml
├── Makefile
├── configs/               tiers.yaml, models.yaml
├── tiermoe/                the installable package (see table above)
├── tests/                  120 tests, run against real committed traces
├── scripts/
│   └── run_end_to_end.sh   doctor -> tests -> roofline -> validate -> baseline
└── demo/cxlmemsim/         how to run the real CXLMemSim backend (Linux-only)
```

## What this framework does NOT claim

- It does not claim to have run on real CXL hardware. Nobody in this
  competition has any (see `PREDICTED.md`'s Methodology section for the
  full evidence-ladder argument this inherits from the original research
  program).
- It does not claim `calib_compute.json`'s current numbers are trustworthy
  GPU measurements — see above and `PREDICTED.md`.
- `tiermoe.policy.precision_gate.expected_delta_ppl` is an explicitly
  `provenance="projected_from_literature"` prior, not a measured quality
  curve — real substitution + perplexity evaluation needs a CUDA box.
- The per-token DES bake-off's `batch` parameter changes per-layer compute
  time (via the roofline model) but does not widen the candidate/miss set
  the way real batched serving would (that effect is the separate,
  already-real E_union(B)/roofline batching analysis) — see
  `tiermoe/sim/policies.py`'s module docstring.
- It does not claim conditional prefetching alone hides CXL latency
  consistently across concurrent load — measured on both models, it doesn't:
  `prefetch-freq`/`prefetch-v2` degrade to `static-c`'s numbers (bit-identical
  at high enough concurrency) once enough decode streams share the link. The
  claim this framework actually backs is narrower and, we think, more useful:
  the **hybrid** design (`hybrid-lru-prefetch`) stays near-oracle at every
  concurrency level tested on both models, because its LRU-residency
  component needs no bandwidth budget at all. See `PREDICTED.md` §2.5 for the
  full numbers.
