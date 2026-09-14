"""Pluggable simulation backends -- "our baseline being the normal and our
one for CXL simulation using CXLMemSim and what not" (the framework's
explicit brief). Three backends, selected by name or auto-detected:

  "analytical"  -- pure roofline ratio math (tiermoe.roofline). Instant,
                   always available, coarsest fidelity.
  "des"         -- this framework's own trace-driven discrete-event policy
                   bake-off (tiermoe.sim.policies / tiermoe.sim.engine).
                   Always available (pure Python), the default, and what
                   every number in PREDICTED.md's Verified section is
                   computed with.
  "cxlmemsim"   -- adapter to the real SlugLab/CXLMemSim tool
                   (arXiv:2303.06153): traces real memory accesses via Linux
                   perf_event_open and injects real epoch-based timing
                   delays. Only available on Linux with perf access and the
                   CXLMemSim binary built from source -- see
                   backends/cxlmemsim.py's module docstring for the exact
                   install/run recipe. UNAVAILABLE on this machine (Darwin,
                   no perf) -- ``select_backend`` falls back to "des" and
                   says so explicitly rather than silently substituting.
"""
from __future__ import annotations

from tiermoe.hw import probe


def select_backend(requested: str = "auto") -> tuple[str, str]:
    """Returns (backend_name, note). `requested="auto"` picks the highest-
    fidelity backend actually available on this machine; an explicit request
    for an unavailable backend raises rather than silently downgrading."""
    caps = probe()
    if requested == "cxlmemsim":
        if not caps.can_run_cxlmemsim_backend:
            raise RuntimeError(
                "backend='cxlmemsim' requested but unavailable on this machine "
                f"(Linux perf available: {caps.linux_perf_available}, binary found: "
                f"{caps.cxlmemsim_binary}). See tiermoe/sim/backends/cxlmemsim.py "
                "for the install recipe, or use backend='des' (always available)."
            )
        return "cxlmemsim", "real CXLMemSim epoch-based timing injection"
    if requested in ("des", "analytical"):
        return requested, f"explicitly requested backend={requested!r}"
    if requested != "auto":
        raise ValueError(f"unknown backend {requested!r}, expected one of 'auto','analytical','des','cxlmemsim'")

    if caps.can_run_cxlmemsim_backend:
        return "cxlmemsim", "auto-selected: highest fidelity backend available on this machine"
    return "des", ("auto-selected: 'des' (cxlmemsim unavailable -- needs Linux + perf + a from-source "
                   "CXLMemSim build; this machine is " + caps.platform + "/" + caps.machine + ")")
