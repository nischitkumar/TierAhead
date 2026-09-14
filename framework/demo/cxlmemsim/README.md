# Running tierMoE under real CXLMemSim (Linux-only)

This directory has no scripts to run **on this machine** -- CXLMemSim
requires Linux + `perf_event_open` + a from-source build, none of which
exist here (a no-CUDA Apple Silicon Mac). See
`tiermoe/sim/backends/cxlmemsim.py`'s module docstring for the full
technical explanation of what CXLMemSim is and why it can't run here.

## What CXLMemSim is

[SlugLab/CXLMemSim](https://github.com/SlugLab/CXLMemSim) (Yang et al.,
*"CXLMemSim: A pure software simulated CXL.mem for performance
characterization"*, arXiv:2303.06153) traces real memory accesses of an
**unmodified running process** via Linux kernel probes and hardware
performance counters, and injects real epoch-based timing delays to emulate
CXL.mem latency/bandwidth -- orders of magnitude faster than a cycle-accurate
simulator (e.g. gem5) for real workloads, and needs no CXL silicon.

This is qualitatively different from this framework's own `analytical` and
`des` backends (`tiermoe.sim.backends`), which both reuse ONE precomputed
per-layer compute-time number and replay already-collected router-decision
traces. CXLMemSim would instead run OLMoE's actual forward pass, live, under
real memory-access interposition -- a fourth rung on Main.md's credibility
ladder that neither backend built into this repository can claim.

## How to actually run it (on an x86_64 Linux box with KVM)

1. Clone the repo and follow **its own current build instructions** (CMake +
   its patched QEMU submodule) -- do not treat any command below as
   guaranteed-correct; check the upstream README for the version you clone,
   since this is an actively developed research tool.
   ```
   git clone https://github.com/SlugLab/CXLMemSim
   ```
2. Build/boot the guest per its docs (QEMU CXL Type-3 device model + guest
   driver/runtime shim).
3. Inside the guest, install this framework and run OLMoE's inference path
   on CPU (fits fp16 without a GPU, per this repo's README.md):
   ```
   pip install -e '.[gpu]'   # transformers/accelerate/etc, CPU is fine here
   python3 -c "
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained('allenai/OLMoE-1B-7B-0924')
   model = AutoModelForCausalLM.from_pretrained('allenai/OLMoE-1B-7B-0924')
   # ... decode loop, per tiermoe/collect/collector.py's run_request_single ...
   "
   ```
   under CXLMemSim's epoch-based interposition, pointing its memory-mapped
   region at wherever the expert weight tensors are allocated.
4. Feed the resulting per-epoch latency/bandwidth report back into this
   framework:
   ```python
   from tiermoe.sim.backends.cxlmemsim import ingest_cxlmemsim_report
   result = ingest_cxlmemsim_report("cxlmemsim_report.json", model_tag="olmoe",
                                     residency_pct=25.0, bw_gbps=32.0, precision="fp16")
   ```
   This produces the SAME `PolicyResult` shape the `des` backend does, so
   `tiermoe.report` / the dashboard never need a CXLMemSim-specific code
   path -- only the `policy="cxlmemsim-real"` / `compute_provenance=
   "measured_here_cxlmemsim"` tags distinguish it in output tables.

## Status

Not run. `ingest_cxlmemsim_report`'s exact expected JSON schema is a
best-effort mapping onto this framework's own `PolicyResult` fields and may
need adjusting once someone actually runs CXLMemSim end-to-end -- that
function is the ONE place such a schema change would need to happen.
