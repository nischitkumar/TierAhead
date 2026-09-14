"""CXLMemSim backend adapter -- SlugLab/CXLMemSim
(https://github.com/SlugLab/CXLMemSim, Yiwei Yang et al., "CXLMemSim: A pure
software simulated CXL.mem for performance characterization", arXiv:2303.06153).

WHAT IT IS (per the paper/repo, not independently re-verified by this
framework -- this machine cannot run it, see below): a real, open-source
CXL.mem simulator that traces memory allocations/accesses via Linux kernel
probes and hardware performance counters (`perf_event_open`), divides
execution into epochs, and injects timing delays to emulate CXL.mem
latency/bandwidth characteristics on an UNMODIFIED real application binary --
orders of magnitude faster than a cycle-accurate simulator (e.g. gem5) for
real workloads. Integration shape: guest application -> a guest driver/
runtime shim -> a QEMU CXL device model -> the CXLMemSim server + host
backends (host memory, shared memory, RDMA/TCP, or host GPU).

WHY THIS MATTERS FOR tierMoE: it is the one CXL-simulation tool in this
space that (a) runs a REAL unmodified process under real epoch-based timing
injection rather than only replaying pre-collected traces analytically, and
(b) needs no CXL silicon -- Main.md section 8.4's own credibility ladder
("real traces -> calibrated simulator -> hardware-grounded validation") gets
a fourth rung from it: a real Linux CXL.mem emulation of the actual OLMoE/
Mixtral inference process, timing-plausible in a way this framework's own
analytical/DES backends (which reuse ONE precomputed t_compute number per
layer, not real per-instruction memory-access timing) cannot claim to be.

WHY IT DOES NOT RUN HERE: `perf_event_open` is Linux-only, typically needs
`CAP_PERFMON`/root or `/proc/sys/kernel/perf_event_paranoid <= 1`, and
CXLMemSim itself must be built from source (CMake, a patched QEMU). This
machine (see tiermoe.hw.probe()) is a no-CUDA Apple Silicon Mac -- Darwin
has no `perf_event_open` at all, full stop. This is EXACTLY the same
"written and reasoned through carefully, but never executed on this specific
box" status this repo already gives measure_transfer.py / e2e_offload_runner.py
/ nf4_fidelity.py for GPU access, applied here to CXL/Linux access instead --
see E3_CALIBRATION.md for that precedent and its own honesty framing.

HOW TO ACTUALLY RUN THIS (on an x86_64 Linux box with KVM):
  1. Clone https://github.com/SlugLab/CXLMemSim and follow its own build
     instructions (CMake + its patched QEMU submodule) -- do not guess at
     flags from this docstring; check that repo's current README, since a
     CLI surface for an actively-developed research tool can change.
  2. Build/boot the guest per its docs (QEMU CXL device model + guest driver
     shim).
  3. Run tiermoe's OWN CPU-only inference entry point (see
     tiermoe.collect / a plain `transformers` decode loop for OLMoE, which
     fits fp16 on CPU per Main.md section 8.1) as the guest workload under
     CXLMemSim's epoch-based interposition, pointing its memory-mapped
     region at wherever the expert weight tensors live.
  4. Feed the resulting per-epoch latency/bandwidth measurements back into
     this framework via ``ingest_cxlmemsim_report`` below, which maps its
     output onto the SAME ``tiermoe.sim.policies.PolicyResult`` shape the
     "des" backend produces, so a report/dashboard consumer never needs to
     know which backend a number came from -- only its `backend` /
     `provenance` field.

This module's ``run()`` therefore does NOT invoke a guessed CLI -- it raises
a clear, actionable error naming exactly what is missing, and
``ingest_cxlmemsim_report`` is the integration point a Linux box would call
after following the steps above.
"""
from __future__ import annotations

import json
from pathlib import Path

from tiermoe.hw import probe
from tiermoe.sim.policies import PolicyResult

CXLMEMSIM_REPO = "https://github.com/SlugLab/CXLMemSim"
CXLMEMSIM_PAPER = "Yang et al., CXLMemSim, arXiv:2303.06153"


def is_available() -> bool:
    return probe().can_run_cxlmemsim_backend


def run(*args, **kwargs):
    caps = probe()
    raise RuntimeError(
        "tiermoe.sim.backends.cxlmemsim.run() cannot execute on this machine: "
        f"platform={caps.platform}/{caps.machine}, linux_perf_available={caps.linux_perf_available}, "
        f"cxlmemsim_binary={caps.cxlmemsim_binary}. CXLMemSim ({CXLMEMSIM_REPO}, {CXLMEMSIM_PAPER}) "
        "requires Linux + perf_event_open + a from-source build (see this module's docstring for the "
        "exact steps). Use backend='des' (tiermoe.sim.backends.des) for the always-available equivalent, "
        "or run this framework's collect/inference workload under a real CXLMemSim build on a Linux box "
        "and call ingest_cxlmemsim_report() with its output."
    )


def ingest_cxlmemsim_report(report_path: Path, model_tag: str, residency_pct: float, bw_gbps: float,
                             precision: str) -> PolicyResult:
    """Maps a real CXLMemSim run's output JSON onto this framework's
    PolicyResult shape. Expects (at minimum) `{"tpot_ms_mean": ..., "tpot_ms_p50":
    ..., "tpot_ms_p99": ..., "demand_stall_rate": ..., "bytes_per_token_mean":
    ..., "n_decode_tokens": ...}` -- adapt the field names here to whatever
    CXLMemSim's actual report schema turns out to be once someone runs it for
    real; this function is the ONE place that mapping lives, so every
    downstream consumer (report/markdown.py, dashboard/app.py) never needs a
    CXLMemSim-specific code path of its own.
    """
    data = json.loads(Path(report_path).read_text())
    return PolicyResult(
        policy="cxlmemsim-real", model=model_tag, residency_pct=residency_pct, bw_gbps=bw_gbps,
        precision=precision, batch=data.get("batch", 1), n_test_reqs=data.get("n_test_reqs", 0),
        n_decode_tokens=data.get("n_decode_tokens", 0), tpot_ms_mean=data["tpot_ms_mean"],
        tpot_ms_p50=data.get("tpot_ms_p50", data["tpot_ms_mean"]),
        tpot_ms_p99=data.get("tpot_ms_p99", data["tpot_ms_mean"]),
        demand_stall_rate=data.get("demand_stall_rate", float("nan")),
        bytes_per_token_mean=data.get("bytes_per_token_mean", float("nan")),
        compute_provenance="measured_here_cxlmemsim", extra={"source_report": str(report_path)},
    )
