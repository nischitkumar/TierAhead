# TierAhead Experiments: E1 (Roofline), E2 (Batching Sweet Spot), E3 (Hardware Calibration), E4 (Predictor v2), E5 (Lookahead Depth), E8 (KV Co-Tenancy)

Implements `Experiments.md`'s E1, E2, E3, E4, E5, and E8. Each experiment folder has:

- `E{1,2,3,4,5,8}_*.md` — design rationale: why the experiment exists, what's implemented,
  what it assumes, caveats found while building/testing it. Static; doesn't change
  on re-run.
- `RESULTS.md` — **auto-generated** by the experiment's script on every run. Don't
  hand-edit; it gets overwritten. A first-cut baseline (from the pilot's already-
  collected traces, `flop_estimate` compute) is already committed.
- `<script>.py` — the analysis code (CLI, argparse).
- `run_e{1,2,3,4,5,8}.sh` — the shell entry point for that experiment.
- `tests/` — pytest unit tests, no GPU/network required.
- `out/` — generated data + figures (`roofline_summary.json`, `batching_summary.json`,
  `calibration_report.json`, `predictor_report.json`, `lookahead_report.json`,
  `kv_contention_summary.json`, `figures/*.pdf`, etc.).

**E3 is GPU-only/server-only**, unlike E8 below — it measures real PCIe transfer
bandwidth, real offload-policy latency, and real NF4 routing fidelity, none of which
can be faked from a laptop. Its pure-math/analysis pieces (`transfer_fit.py`,
`policy_check.py`, `build_calibration_report.py`) DO run here with real tests and a
real (honestly-partial) report — see `e3_calibration/E3_CALIBRATION.md` for exactly
what was and wasn't run before pushing to the server, and its "single-GPU correction"
note (the actual box is one 24GB GPU, not the two-box setup Experiments.md's prose
for E3 step 1 assumes).

**E8 is different from E1/E2: it runs on a laptop, not the GPU server.** No model
weights, no GPU, no Mistral/Mixtral/OLMoE install — it's a discrete-event queueing
simulation over an architecture-derived formula plus the pilot's already-collected
traces. See `e8_kv_cotenancy/E8_KV_COTENANCY.md` and run it with
`bash experiments/e8_kv_cotenancy/run_e8.sh` (creates a local
`experiments/.venv-mac` on first run — system numpy/pandas/matplotlib plus a
pip-installed zstandard+pytest).

**E4 and E5 also run on a laptop, not (only) the GPU server** — Experiments.md says so
itself ("trains in minutes on CPU/MPS"). Both were run for real on a laptop with no CUDA
at all, against the pilot's real traces, with real (not fixture) committed
`RESULTS.md`/`out/` baselines:
- **E4** (`e4_predictor/`): a small per-expert-sigmoid MLP predictor, ablated over which
  input features matter (expert IDs / +gate weights / +router entropy / +hidden state —
  the last needs a GPU-only `collect_hidden_states.py` pass and is skipped cleanly when
  absent), a recall-vs-bytes curve, ECE/temperature-scaling calibration, and
  cross-domain (chat/code) transfer. Real bug found and fixed while building this: the
  first full run showed adding *more* input features catastrophically collapsing
  recall — traced to a feature-scaling bug (gate weights and router entropy sit on a
  ~10-60x smaller numeric scale than the plain one-hot IDs, so the same fixed
  epoch/LR budget hadn't converged), not an actual finding. Fixed with
  train-statistics-only standardization; see `E4_PREDICTOR.md` for the full story and a
  regression test that locks the fix in.
- **E5** (`e5_lookahead/`): reuses E4's `features.py`/`mlp.py` directly. Direct
  (one predictor per lookahead depth d) vs. naively-chained (one d=1 predictor composed
  d times) recall, a feasible-fetch-size curve (E1's compute-time model × this
  experiment's recall-vs-depth), and a closed-form bounded prefetch-queue simulation.
  Found, on a real run, that the direct-vs-chained recall gap widens monotonically with
  depth for both models — exactly Experiments.md's own "chaining compounds error"
  prediction — see `E5_LOOKAHEAD.md`.
- Run with `bash experiments/e4_predictor/run_e4.sh` and `bash experiments/e5_lookahead/run_e5.sh`
  — both create/reuse the same `experiments/.venv-mac`, adding a `torch` install to it on
  first use (CPU/MPS wheel, no CUDA).

`common/` holds shared model constants (`model_specs.py`) and the `E_union(B)`
resampling logic both experiments use (`eunion.py`), plus a thin adapter onto
`elp_probe/src/analyze.py` (`trace_io.py`) so nothing here duplicates the pilot's
trace-loading/predictor-fitting code.

## TL;DR: how to run this on the server

Copy the **whole repo** (not just `experiments/`) to the GPU server (single 24GB-VRAM
GPU, e.g. an RTX 4500 Ada — see `e3_calibration/E3_CALIBRATION.md` for why this
corrects an earlier dual-4060Ti assumption in `Experiments.md`'s own prose) —
the scripts read the pilot's already-collected traces from `results/` and reuse
`elp_probe/src/`. Then:

```bash
bash experiments/run_all_experiments.sh
```

That's it — it runs the unit tests as a gate, downloads both models, runs a real GPU
compute calibration, then E1 and E2, and prints where every result file landed.
Takes roughly: env setup (a few min, mostly torch install) + model download (Mixtral
is ~93GB fp16 on disk, so this depends entirely on your link — the box has
"almost unlimited internet," per the brief) + ~10-20 min GPU calibration + under a
minute for the analysis itself (E1/E2 are pure trace analysis, no GPU needed there).

See each experiment's design doc for what to look at afterward, and in particular
**E2_BATCHING.md's sensitivity note** before quoting its claim-check verdict.

## Why no GPU is needed for most of this

Experiments.md marks both E1 and E2 "no GPU" — they're trace analysis and an
analytical/FLOP-based model, not something requiring re-running the models. The one
GPU-touching piece is `e1_roofline/measure_compute.py`, an *optional* (but
recommended) step that replaces the FLOP/MFU compute-time guess with a real
measurement on this box's actual GPUs. Every number downstream carries an explicit
`compute_provenance` tag (`measured_here` vs `flop_estimate`) so it's always clear
which one you're looking at.

## What's already validated (before you run anything)

- 39 unit tests (`pytest experiments/ -v`), covering the arithmetic, the
  compute-provenance fallback logic, the E_union resampling (checked against
  brute-force enumeration), B* selection, and the batched-recall metric (checked
  against a deterministic bijective transition table and a synthetic-noise case) —
  all pure Python/numpy/pandas, no GPU or network.
- Both full pipelines (E1 and E2, including the slower batched-recall step) run
  end-to-end against the pilot's real committed traces, with no crashes, and their
  output is what's currently committed in each `RESULTS.md`/`out/`.
- Two real bugs were caught and fixed by that end-to-end run before being handed
  off (not by unit tests alone): a `None`/`NaN` display bug in the provisioning
  table, and a throughput-vs-latency conflation bug in E2's B* selection that would
  have silently made the SLO constraint a no-op. Both have regression tests now —
  see `E1_ROOFLINE.md` and `E2_BATCHING.md` for the details.
