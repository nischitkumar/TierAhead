"""Concurrent-inference bandwidth contention (mentor-review ask: "evaluate
under concurrent inference where CXL bandwidth contention becomes a real
factor"). Read this module's history before trusting a concurrency number --
an earlier draft of this file drew the wrong conclusion from a real test,
and correcting that is itself the useful result documented here.

THE TWO DIFFERENT QUESTIONS THIS MODULE COULD ANSWER, AND WHY THEY NEED
DIFFERENT MODELS:

  1. "N decode sessions are ALREADY active and self-throttling -- each one
     literally cannot request layer l+1's experts until layer l's fetch has
     arrived, so a session never 'gets ahead' of its own link usage." This
     is a CLOSED queueing network (a fixed population of N customers cycling
     through one shared resource). The standard, textbook result for a
     closed network under persistent contention is egalitarian processor
     sharing: each customer realizes roughly 1/N of the resource. This is
     exactly what `tiermoe.sim.policies.simulate_policy`'s `concurrency`
     parameter models via `bw_gbps / concurrency` -- and it is the RIGHT
     model for this question, not a hand-wave. It is most ACCURATE exactly
     when contention is persistent (every stream is link-bound often enough
     that N-way sharing is the norm, not the exception), and becomes
     increasingly CONSERVATIVE (predicts worse TPOT than reality) the more
     intermittent real contention actually is -- i.e. it never understates
     the problem, which is the safe direction for a capacity-planning number.

  2. "NEW decode sessions arrive over time (not a fixed pool of N), each
     occupying the link with its own periodic per-layer demand for its
     lifetime -- what arrival rate can this link admit before a backlog of
     not-yet-started sessions grows without bound?" This is a genuinely
     OPEN queueing question (admission control), and IS correctly validated
     by an open-loop discrete-event simulation -- which is what this
     module's `verify_stability_boundary_with_des` actually builds (N
     phase-staggered periodic `TrafficClass` streams via
     `tiermoe.sim.engine`, an open-loop periodic-arrival model).

THE MISTAKE AN EARLIER VERSION OF THIS MODULE MADE: it used answer (2)'s
open-loop DES to "validate" (and then, wrongly, to abandon) answer (1)'s
closed-loop bw/N model. That comparison doesn't apply -- an open-loop
periodic emitter that keeps generating new demand every fixed period
REGARDLESS of whether its previous job was served is not a model of a
self-throttling decode session at all (a real session cannot ask for layer
5's experts before layer 4's arrived), so of course it diverges under heavy
load in a way closed, self-throttling sessions structurally cannot (a
closed network's population is bounded, so its queue length is too). The
"catastrophic divergence" that draft found was a real, correctly-computed
property of the WRONG model for question (1), not evidence against bw/N.

WHAT THIS MODULE ACTUALLY PROVIDES, CORRECTLY SCOPED:
  - `aggregate_rho` / `max_sustainable_concurrency`: for question (1)'s
    closed-loop reading, this is the concurrency level at which the LINK
    (not per-stream compute) becomes the binding constraint for every
    stream -- i.e. where bw/N stops being a conservative estimate and starts
    being the ACCURATE one (persistent contention is now the norm). Below
    it, real TPOT is likely better than bw/N predicts; at or above it, bw/N
    is the theoretically expected answer, not just a proxy.
  - `verify_stability_boundary_with_des`: answers question (2) directly and
    correctly (open-loop is the right model there) -- how many NEW sessions
    per unit time this link can admit before a startup backlog diverges.
    Genuinely validated by DES (see tests/test_sim_concurrency.py); NOT a
    check on the bw/N model, which it was never the right tool to check.

`simulate_policy`'s `extra["concurrency_stable"]` reflects question (1)'s
boundary: `False` means the link is (or is expected to be) the persistent
bottleneck at this concurrency, which is exactly where its bw/N-based
`tpot_ms_mean` is most trustworthy, not least.
"""
from __future__ import annotations

from tiermoe.sim.engine import TrafficClass, pctile_stats, pooled_simulate


def max_sustainable_concurrency(bytes_per_stream_per_ms: float, bw_gbps: float) -> float:
    """Continuous (non-integer) concurrency level N at which N streams'
    combined WORST-CASE demand first equals link bandwidth (aggregate_rho =
    N * bytes_per_stream_per_ms / bw_bytes_per_ms = 1). For the closed-loop
    reading (N self-throttling decode sessions, see module docstring
    question 1): below this point, streams are not always simultaneously
    link-bound, so real TPOT is plausibly better than the bw/N estimate;
    at or above it, every stream IS persistently link-bound, which is
    exactly the regime bw/N's classical fair-share result describes
    accurately."""
    bw_bytes_per_ms = bw_gbps * 1e6
    if bytes_per_stream_per_ms <= 0:
        return float("inf")
    return bw_bytes_per_ms / bytes_per_stream_per_ms


def aggregate_rho(bytes_per_stream_per_ms: float, bw_gbps: float, concurrency: int) -> float:
    bw_bytes_per_ms = bw_gbps * 1e6
    if bw_bytes_per_ms <= 0:
        return float("inf")
    return (concurrency * bytes_per_stream_per_ms) / bw_bytes_per_ms


def verify_stability_boundary_with_des(period_ms: float, job_bytes: float, bw_gbps: float, concurrency: int,
                                        sim_ms: float = 5000.0) -> dict:
    """Answers module docstring question (2), NOT question (1): treats each
    of the `concurrency` streams as an OPEN-LOOP periodic emitter (phases
    evenly staggered via `TrafficClass.phase_ms`) that keeps generating new
    demand on a fixed clock regardless of whether its previous job was
    served -- the right model for "how many independent, always-emitting
    traffic sources can this link admit," e.g. new sessions arriving faster
    than the system drains its startup backlog. It is NOT a validation of
    `simulate_policy`'s closed-loop bw/N concurrency model (an earlier
    version of this module used it as exactly that, incorrectly -- see the
    module docstring's "THE MISTAKE" section). Reports the DES-measured mean
    latency alongside the ideal (uncontended) solo latency and the
    aggregate_rho open-loop stability indicator.
    """
    bytes_per_stream_per_ms = job_bytes / period_ms
    agg_rho = aggregate_rho(bytes_per_stream_per_ms, bw_gbps, concurrency)
    ideal_solo_ms = job_bytes / (bw_gbps * 1e6)

    classes = [
        TrafficClass(name=f"req{i}", weight=1.0, period_ms=period_ms, job_bytes=job_bytes,
                     phase_ms=i * period_ms / concurrency)
        for i in range(concurrency)
    ]
    res = pooled_simulate(n_seeds=1, base_seed=0, classes=classes, bw_gbps=bw_gbps, sim_ms=sim_ms,
                           warmup_ms=period_ms * concurrency * 2)
    all_records = [r for recs in res.values() for r in recs]
    des_stats = pctile_stats(all_records)

    return {
        "concurrency": concurrency, "bw_gbps": bw_gbps, "period_ms": period_ms, "job_bytes": job_bytes,
        "aggregate_rho": agg_rho, "predicted_stable": agg_rho < 1.0,
        "ideal_solo_latency_ms": ideal_solo_ms,
        "des_measured_latency_ms": des_stats["latency_mean_ms"], "des_n_completions": des_stats["n"],
        "matches_solo_within_5pct": (
            abs(des_stats["latency_mean_ms"] - ideal_solo_ms) / ideal_solo_ms < 0.05
            if des_stats["n"] > 0 and ideal_solo_ms > 0 else False
        ),
    }


def sweep_stability_boundary(period_ms: float, job_bytes: float, bw_gbps: float,
                              concurrency_grid: list[int], **kwargs) -> list[dict]:
    return [verify_stability_boundary_with_des(period_ms, job_bytes, bw_gbps, c, **kwargs) for c in concurrency_grid]
