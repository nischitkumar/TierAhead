"""N-class shared-bandwidth discrete-event fluid simulator.

Generalizes experiments/e8_kv_cotenancy/kv_link_sim.py's ``simulate_link``
(originally hardcoded to exactly two traffic classes, expert-prefetch and
KV-spillover) to an arbitrary number of named ``TrafficClass`` streams
sharing one physical link -- the consolidation E8's own design doc called
for once E6 existed ("fold E8's KV traffic class into E6's link model...
rather than maintain two separate simulators long-term"). Every fixed bug
E8 found the hard way is preserved here, because the underlying event-driven
mechanics are unchanged:

  - right-censoring at the simulation horizon (a job still in flight when
    the window ends is excluded from stats, never force-completed -- avoids
    the "unbounded horizon at near-zero bandwidth" bug).
  - multi-seed pooling before computing percentiles (avoids single-seed
    phase-lock artifacts between periodic and Poisson streams).
  - microsecond-rounded Pareto comparisons (avoids floating-point-residue
    "winners").

This is an EXACT event-driven fluid simulation, not a fixed-timestep
approximation: the rate any class receives only changes at an arrival or a
completion, so integrating bytes-served linearly between those instants
introduces zero discretization error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

COMPLETION_EPS_BYTES = 1e-3
ARRIVAL_EPS_MS = 1e-9
MAX_EVENTS_SAFETY = 5_000_000
PARETO_ROUND_DECIMALS = 6


@dataclass
class TrafficClass:
    name: str
    weight: float = 1.0
    period_ms: float | None = None
    job_bytes: float | None = None
    phase_ms: float = 0.0  # offsets a periodic class's first arrival -- lets N otherwise-identical
                           # periodic classes (e.g. N concurrent decode streams) avoid lockstep arrivals
    arrival_rate_per_sec: float | None = None
    job_sampler: Callable[[np.random.Generator], float] | None = None

    def is_periodic(self) -> bool:
        return self.period_ms is not None


@dataclass
class _Job:
    arrival_ms: float
    total_bytes: float
    remaining_bytes: float


def _gen_arrivals(cls: TrafficClass, sim_ms: float, rng: np.random.Generator):
    if cls.is_periodic():
        if cls.period_ms <= 0:
            return [], []
        start = cls.period_ms + (cls.phase_ms % cls.period_ms)
        arrivals = list(np.arange(start, sim_ms, cls.period_ms))
        sizes = [cls.job_bytes] * len(arrivals)
        return arrivals, sizes
    rate = cls.arrival_rate_per_sec or 0.0
    if rate <= 0:
        return [], []
    arrivals, t = [], 0.0
    while True:
        t += rng.exponential(1000.0 / rate)
        if t > sim_ms:
            break
        arrivals.append(t)
    sizes = [cls.job_sampler(rng) for _ in arrivals]
    return arrivals, sizes


def _rate_alloc(present_names: list[str], weights: dict[str, float], bw_bytes_per_ms: float) -> dict[str, float]:
    """Splits bandwidth among classes with a non-empty queue, proportional to
    weight. A lone present class always gets the FULL link (a shared link
    doesn't waste bandwidth just because only one class needs it), even if
    its own weight is 0 -- generalizes the original two-class
    expert-first/kv-first/weighted semantics exactly (weight=0 means "yields
    whenever contended," not "never gets bandwidth at all")."""
    if not present_names:
        return {}
    if len(present_names) == 1:
        return {present_names[0]: bw_bytes_per_ms}
    total_w = sum(weights[n] for n in present_names)
    if total_w <= 0:
        share = bw_bytes_per_ms / len(present_names)
        return {n: share for n in present_names}
    return {n: bw_bytes_per_ms * weights[n] / total_w for n in present_names}


def simulate_link(*, classes: list[TrafficClass], bw_gbps: float, sim_ms: float, seed: int,
                   warmup_ms: float = 0.0) -> dict[str, list[tuple[float, float, float]]]:
    """Returns {class_name: [(arrival_ms, latency_ms, ideal_solo_ms), ...]}.
    ideal_solo_ms = total_bytes / bw_bytes_per_ms, i.e. how long the transfer
    would take with the whole link to itself -- subtracting it from latency
    isolates contention-caused inflation."""
    rng = np.random.default_rng(seed)
    bw_bytes_per_ms = bw_gbps * 1e6  # GB/s = 1e9 B/s = 1e6 B/ms
    weights = {c.name: c.weight for c in classes}

    arrivals_by_class: dict[str, list[float]] = {}
    sizes_by_class: dict[str, list[float]] = {}
    for c in classes:
        arr, sz = _gen_arrivals(c, sim_ms, rng)
        arrivals_by_class[c.name] = arr
        sizes_by_class[c.name] = sz

    queues: dict[str, list[_Job]] = {c.name: [] for c in classes}
    idx: dict[str, int] = {c.name: 0 for c in classes}
    done: dict[str, list[tuple[float, float, float]]] = {c.name: [] for c in classes}
    t = 0.0
    n_events = 0

    while True:
        n_events += 1
        if n_events > MAX_EVENTS_SAFETY:
            raise RuntimeError("simulate_link: exceeded MAX_EVENTS_SAFETY -- check period/rate configuration")

        present = [name for name, q in queues.items() if q]
        rates = _rate_alloc(present, weights, bw_bytes_per_ms)

        candidates = []
        for name in present:
            head = queues[name][0]
            r = rates.get(name, 0.0)
            if r > 0:
                candidates.append(t + head.remaining_bytes / r)
        for name, c in zip((c.name for c in classes), classes):
            if idx[name] < len(arrivals_by_class[name]):
                candidates.append(arrivals_by_class[name][idx[name]])
        if not candidates:
            break

        t_next = min(candidates)
        if t_next > sim_ms:
            break  # right-censor: exclude in-flight jobs from stats, don't force-complete them
        dt = max(0.0, t_next - t)
        for name in present:
            r = rates.get(name, 0.0)
            if r > 0:
                queues[name][0].remaining_bytes -= r * dt
        t = t_next

        for name in list(present):
            head = queues[name][0]
            if head.remaining_bytes <= COMPLETION_EPS_BYTES:
                queues[name].pop(0)
                if head.arrival_ms >= warmup_ms:
                    done[name].append((head.arrival_ms, t - head.arrival_ms, head.total_bytes / bw_bytes_per_ms))

        for c in classes:
            name = c.name
            arr = arrivals_by_class[name]
            while idx[name] < len(arr) and arr[idx[name]] <= t + ARRIVAL_EPS_MS:
                size = sizes_by_class[name][idx[name]]
                queues[name].append(_Job(arr[idx[name]], size, size))
                idx[name] += 1

    return done


def pooled_simulate(*, n_seeds: int, base_seed: int, **kwargs) -> dict[str, list[tuple[float, float, float]]]:
    """Pools raw completion records across n_seeds independent runs BEFORE
    computing any percentile -- a single seed can phase-lock a periodic
    stream against a Poisson one and produce a non-representative result
    (the exact artifact E8's own development caught: giving one class MORE
    priority appeared to produce LESS contention for the other, which
    reversed once pooled)."""
    pooled: dict[str, list] = {}
    for i in range(n_seeds):
        res = simulate_link(seed=base_seed + i, **kwargs)
        for name, records in res.items():
            pooled.setdefault(name, []).extend(records)
    return pooled


def pctile_stats(records: list[tuple[float, float, float]]) -> dict:
    if not records:
        return {"n": 0, "latency_mean_ms": float("nan"), "latency_p50_ms": float("nan"),
                "latency_p95_ms": float("nan"), "inflation_mean_ms": float("nan"),
                "inflation_p50_ms": float("nan"), "inflation_p95_ms": float("nan")}
    lat = np.array([r[1] for r in records])
    ideal = np.array([r[2] for r in records])
    infl = lat - ideal
    return {
        "n": len(records), "latency_mean_ms": float(lat.mean()), "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)), "inflation_mean_ms": float(infl.mean()),
        "inflation_p50_ms": float(np.percentile(infl, 50)), "inflation_p95_ms": float(np.percentile(infl, 95)),
    }


def pareto_labels(points: list[tuple[float, float, str]]) -> list[str]:
    """points: (x, y, label), lower-is-better on both axes."""
    out = []
    for i, (x1, y1, l1) in enumerate(points):
        if any(j != i and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1)
               for j, (x2, y2, _l2) in enumerate(points)):
            continue
        out.append(l1)
    return out
