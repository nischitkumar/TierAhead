"""Pipelining / queue-depth simulation (Experiments.md E5 step 4): "simulate
a prefetch queue holding up to Q outstanding requests; measure queue
occupancy and link utilization. Over-deep prefetch causes bandwidth
contention with demand fetches -- quantify the crossover."

This is a bounded-buffer M/D/1-with-drop system: prefetch requests arrive
deterministically (one per layer-step, period = t_layer_ms), each needs
service time = bytes/BW to transfer, and the buffer holds at most Q
outstanding requests -- an arrival that finds the buffer full is DROPPED
(that layer falls back to demand-fetch instead of getting prefetched; a
bounded real prefetch engine would do exactly this rather than block
indefinitely or grow the buffer without limit).

Deterministic arrivals + deterministic service time means occupancy and
utilization can be computed in closed form rather than needing a stochastic
event-driven simulator (see the module-level derivation below) -- so this is
implemented as exact arithmetic, not Monte Carlo, and is correspondingly
fast and exact rather than approximate.
"""


def simulate_bounded_queue(period_ms: float, service_ms: float, capacity_q: int):
    """Exact steady-state occupancy/utilization/drop-rate for a
    deterministic-arrival (period_ms), deterministic-service (service_ms),
    finite-capacity (capacity_q) single-server queue.

    Derivation: let rho = service_ms / period_ms (the load factor -- also
    exactly E1's "rho" if this were the compute/transfer ratio at bw budgeted
    to prefetch, but expressed as a queueing load factor here). If rho <= 1,
    each arrival's service completes before the next one arrives (deterministic
    non-overlapping service), so the queue never holds more than 1 job at a
    time and nothing is ever dropped regardless of capacity_q: occupancy =
    rho (fraction of time the server is busy on the single in-service job),
    utilization = rho, drop_rate = 0. If rho > 1, service can't keep up with
    arrivals at all -- the queue fills to capacity_q and STAYS there in
    steady state (every slot is perpetually occupied, since jobs enter faster
    than they can leave): occupancy = capacity_q, utilization = 1.0 (server
    never idles), and the steady-state fraction of arrivals dropped is
    exactly 1 - 1/rho (only 1 job in every rho arrivals can possibly be
    served; the rest must be dropped once the buffer's full transient has
    passed) -- capacity_q affects only how long that fill-up transient lasts,
    not the steady-state drop rate, which is exactly why "over-deep prefetch
    causes bandwidth contention" per Experiments.md's own framing: making Q
    bigger doesn't fix an overloaded link, it only delays the point at which
    drops become visible.
    """
    if period_ms <= 0 or service_ms < 0 or capacity_q < 1:
        raise ValueError("period_ms>0, service_ms>=0, capacity_q>=1 required")
    rho = service_ms / period_ms
    if rho <= 1.0:
        return {"rho": rho, "mean_occupancy": rho, "link_utilization": rho, "drop_rate": 0.0}
    return {"rho": rho, "mean_occupancy": float(capacity_q), "link_utilization": 1.0,
            "drop_rate": 1.0 - 1.0 / rho}


def sweep_queue_depth(period_ms: float, service_ms: float, q_grid: list[int]):
    return {q: simulate_bounded_queue(period_ms, service_ms, q) for q in q_grid}
