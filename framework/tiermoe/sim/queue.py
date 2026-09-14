"""Closed-form bounded prefetch-queue occupancy/utilization/drop-rate.
Ported from experiments/e5_lookahead/queue_sim.py.

Deterministic arrivals (period_ms) + deterministic service (service_ms) admit
an exact closed-form solution -- no Monte Carlo needed. See
tiermoe.sim.engine for the stochastic (Poisson-arrival) multi-class case,
which does need event-driven simulation.
"""
from __future__ import annotations


def simulate_bounded_queue(period_ms: float, service_ms: float, capacity_q: int) -> dict:
    """rho = service_ms/period_ms. rho<=1: occupancy=utilization=rho,
    drop_rate=0 (server keeps up). rho>1: queue saturates at capacity_q,
    utilization=1.0, steady-state drop_rate=1-1/rho regardless of
    capacity_q -- a deeper buffer only delays the fill-up transient, it
    does not fix an overloaded link (Main.md's own headline finding from
    this exact module, reproduced verbatim here)."""
    if period_ms <= 0 or service_ms < 0 or capacity_q < 1:
        raise ValueError("period_ms>0, service_ms>=0, capacity_q>=1 required")
    rho = service_ms / period_ms
    if rho <= 1.0:
        return {"rho": rho, "mean_occupancy": rho, "link_utilization": rho, "drop_rate": 0.0}
    return {"rho": rho, "mean_occupancy": float(capacity_q), "link_utilization": 1.0, "drop_rate": 1.0 - 1.0 / rho}


def sweep_queue_depth(period_ms: float, service_ms: float, q_grid: list[int]) -> dict[int, dict]:
    return {q: simulate_bounded_queue(period_ms, service_ms, q) for q in q_grid}
