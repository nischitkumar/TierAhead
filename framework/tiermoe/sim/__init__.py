"""[5] Tiered-Memory Simulator (Main.md section 4.5 / Experiments.md E6).

Experiments.md speced E6 as "the single largest engineering item... produces
the headline result" but no e6_* folder exists anywhere in experiments/ --
it was never built. This package is that build. It does not start from
nothing: experiments/e8_kv_cotenancy/kv_link_sim.py already implements and
tests a real discrete-event fluid link simulator with shared-bandwidth
contention (the exact primitive E6's own spec calls for: "an explicit
shared-bandwidth link model... what makes the simulation non-trivial and
what most teams will omit"). ``tiermoe.sim.engine`` generalizes that proven
engine from two hardcoded traffic classes (expert, kv) to N named classes --
which is precisely the consolidation E8's own design doc recommended
("If/when E6 is built, the honest path is to fold E8's KV traffic class into
E6's link model... rather than maintain two separate simulators long-term").

Two complementary layers, matching the fidelity split Main.md's own program
already uses between E5 (per-request feasible-fetch curve) and E8 (link
contention queueing):

  tiermoe.sim.policies  -- per-layer, trace-driven policy bake-off (none,
                           static-C, lru, prefetch-freq, prefetch-v2,
                           prefetch-v2+precision-gate, oracle) walked over
                           REAL held-out decode traces. This is what answers
                           "how much does demand-stall/TPOT improve." No
                           bandwidth-contention modeling at this layer
                           (single request in flight) -- see note below.

  tiermoe.sim.engine    -- the N-class fluid DES for when MULTIPLE traffic
                           streams share one physical link (expert prefetch
                           vs KV spillover vs multiple concurrent policies'
                           own prefetch bursts) and contention is the actual
                           question, as in tiermoe.kv.cotenancy.

``tiermoe.sim.backends`` picks which of the above (or the real CXLMemSim
tool, when available) actually answers a `run(config)` call -- see
backends/__init__.py for the selection logic and hw.py for why CXLMemSim is
unavailable on this machine.
"""
from .policies import simulate_policy, run_policy_bakeoff, PolicyResult
from .validation import run_validation_suite

__all__ = ["simulate_policy", "run_policy_bakeoff", "PolicyResult", "run_validation_suite"]
