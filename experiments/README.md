# tierMoE Experiments: E1 (Roofline), E2 (Batching Sweet Spot), E8 (KV Co-Tenancy)

Implements `Experiments.md`'s E1, E2, and E8. Each experiment folder has:

- `E{1,2,8}_*.md` — design rationale: why the experiment exists, what's implemented,
  what it assumes, caveats found while building/testing it. Static; doesn't change
  on re-run.
- `RESULTS.md` — **auto-generated** by the experiment's script on every run. Don't
  hand-edit; it gets overwritten. A first-cut baseline (from the pilot's already-
  collected traces, `flop_estimate` compute) is already committed.
- `<script>.py` — the analysis code (CLI, argparse).
- `run_e{1,2,8}.sh` — the shell entry point for that experiment.
- `tests/` — pytest unit tests, no GPU/network required.
- `out/` — generated data + figures (`roofline_summary.json`, `batching_summary.json`,
  `kv_contention_summary.json`, `figures/*.pdf`, etc.).

**E8 is different from E1/E2: it runs on a laptop, not the GPU server.** No model
weights, no GPU, no Mistral/Mixtral/OLMoE install — it's a discrete-event queueing
simulation over an architecture-derived formula plus the pilot's already-collected
traces. See `e8_kv_cotenancy/E8_KV_COTENANCY.md` and run it with
`bash experiments/e8_kv_cotenancy/run_e8.sh` (creates a local
`experiments/.venv-mac` on first run — system numpy/pandas/matplotlib plus a
pip-installed zstandard+pytest).

`common/` holds shared model constants (`model_specs.py`) and the `E_union(B)`
resampling logic both experiments use (`eunion.py`), plus a thin adapter onto
`elp_probe/src/analyze.py` (`trace_io.py`) so nothing here duplicates the pilot's
trace-loading/predictor-fitting code.

## TL;DR: how to run this on the server

Copy the **whole repo** (not just `experiments/`) to the 2x RTX 4060 Ti server —
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
