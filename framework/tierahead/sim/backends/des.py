"""DES backend: this framework's own trace-driven policy bake-off
(tierahead.sim.policies). Always available, pure Python. This is what
tierahead.baseline.compare uses for the "normal vs CXL-tiered" headline
comparison and what produces every number in PREDICTED.md's Verified
section.
"""
from __future__ import annotations

from pathlib import Path

from tierahead.sim.policies import run_policy_bakeoff


def run(model_tag: str, trace_dir: Path, residency_pct: float, bw_gbps: float, precision: str,
        policies: list[str] | None = None, batch: int = 1, calib: dict | None = None,
        added_latency_ns: float = 0.0, depth: int = 1, m_mult: int = 2, concurrency: int = 1, seed: int = 0):
    df = run_policy_bakeoff(model_tag, trace_dir, policies=policies, residency_pct=residency_pct,
                             bw_gbps=bw_gbps, precision=precision, batch=batch, calib=calib,
                             added_latency_ns=added_latency_ns, depth=depth, m_mult=m_mult,
                             concurrency=concurrency, seed=seed)
    df.insert(0, "backend", "des")
    return df
