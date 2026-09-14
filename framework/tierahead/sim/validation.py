"""E6 validation gates (Experiments.md E6: "Validation gates (run these as
unit tests)"), implemented as reusable functions rather than only inline
pytest asserts, so `tierahead doctor` / `tierahead sim --validate` can run them
against any trace directory, not just the ones tests/ happens to fixture.

Four gates, exactly as specced:
  1. oracle <= every policy on TPOT (oracle is the upper bound / best case).
  2. zero-latency + infinite-BW CXL ⇒ parity with HBM-only (no stall at all).
  3. monotone improvement in residency r (more residency never hurts TPOT).
  4. CXL latency sensitivity: sweeping added latency 150/250/400 ns barely
     moves TPOT, because bandwidth -- not latency -- dominates in the
     bandwidth-bound regime (Main.md section 8.4's own pre-emptive claim).

Three more gates, added for the mentor-feedback round's concurrency work
(PREDICTED.md Sec 2.5), extending the same "run these as unit tests" spirit
to the closed-loop concurrency model instead of leaving it as a narrative
finding only:
  5. oracle's TPOT is invariant to concurrency (it never demand-fetches, so
     sharing the link cannot touch it -- a violation would mean concurrency
     scaling leaked into the wrong code path).
  6. bw/N (closed-loop fair-share) means TPOT can only stay flat or get worse
     as concurrency grows, never improve, for any policy that ever transfers
     bytes -- checked for `static-c` by default.
  7. `hybrid-lru-prefetch` never has worse TPOT than plain `lru` at ANY
     concurrency level, not just concurrency=1 -- it strictly extends lru's
     cache with a prefetch layer on top, so this must hold everywhere.

Plus `find_collapse_concurrency`, a *quantifying* (not pass/fail) companion:
the smallest concurrency at which a byte-budget-limited prefetch policy
becomes numerically indistinguishable from `static-c` -- turning PREDICTED.md
Sec 2.5's narrative finding ("conditional prefetch collapses under
concurrency") into a number `run_validation_suite` reports every time, on any
trace, not just the two operating points written up by hand.

These are executed for real against the pilot's committed traces in
tests/test_sim_validation.py -- see PREDICTED.md's Verified section for the
actual pass/fail this repo's traces produce.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tierahead.sim.policies import _load_and_split_decode, run_concurrency_sweep, simulate_policy, run_policy_bakeoff
from tierahead.specs import MODEL_SPECS
from tierahead.specs.tiers import CXL_LATENCY_NS_GRID


def _split(trace_dir: Path, seed: int = 0):
    # Delegates to the memoized loader in tierahead.sim.policies -- this
    # module's four original gates (each calling simulate_policy directly on
    # a single policy) share the same cache as run_policy_bakeoff/
    # run_concurrency_sweep, instead of each re-reading and re-splitting the
    # trace file from scratch.
    return _load_and_split_decode(Path(trace_dir), seed)


def gate_oracle_is_upper_bound(model_tag: str, trace_dir: Path, residency_pct=25.0, bw_gbps=32.0,
                                precision="fp16", seed=0) -> dict:
    df = run_policy_bakeoff(model_tag, trace_dir, residency_pct=residency_pct, bw_gbps=bw_gbps,
                             precision=precision, seed=seed)
    if "oracle" not in set(df["policy"]):
        return {"passed": False, "detail": "oracle policy result missing"}
    oracle_tpot = df.loc[df.policy == "oracle", "tpot_ms_mean"].iloc[0]
    worse = df[df["tpot_ms_mean"] < oracle_tpot - 1e-9]
    passed = len(worse) == 0
    return {"passed": passed, "oracle_tpot_ms": float(oracle_tpot),
            "violations": worse[["policy", "tpot_ms_mean"]].to_dict("records")}


def gate_zero_latency_infinite_bw_parity(model_tag: str, trace_dir: Path, seed=0, rel_tol=0.01) -> dict:
    """At residency=100% (equivalent to an infinite-bandwidth, zero-latency
    link -- nothing ever needs a real fetch) TPOT should equal pure compute
    time (n_layers * t_layer_ms), matching an HBM-only system with no tiering
    at all -- this framework's explicit "baseline being normal" comparison
    point."""
    train, test = _split(trace_dir, seed=seed)
    spec = MODEL_SPECS[model_tag]
    result = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="static-c",
                              residency_pct=100.0, bw_gbps=1e9, precision="fp16", seed=seed)
    if result is None:
        return {"passed": False, "detail": "simulate_policy returned None"}
    from tierahead.roofline.model import get_t_compute_ms
    t_layer_ms, _prov = get_t_compute_ms(spec, batch=1, calib=None)
    expected_hbm_only_tpot = spec.n_layers * t_layer_ms
    rel_err = abs(result.tpot_ms_mean - expected_hbm_only_tpot) / expected_hbm_only_tpot
    return {"passed": rel_err <= rel_tol, "measured_tpot_ms": result.tpot_ms_mean,
            "expected_hbm_only_tpot_ms": expected_hbm_only_tpot, "rel_err": rel_err}


def gate_monotone_in_residency(model_tag: str, trace_dir: Path, bw_gbps=32.0, precision="fp16",
                                residency_grid=(0.0, 12.5, 25.0, 37.5, 50.0, 75.0), seed=0) -> dict:
    train, test = _split(trace_dir, seed=seed)
    spec = MODEL_SPECS[model_tag]
    tpots = []
    for r in residency_grid:
        res = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="static-c",
                               residency_pct=r, bw_gbps=bw_gbps, precision=precision, seed=seed)
        tpots.append(res.tpot_ms_mean if res else float("nan"))
    diffs = np.diff(tpots)
    passed = bool(np.all(diffs <= 1e-6))
    return {"passed": passed, "residency_grid": list(residency_grid), "tpot_ms_by_residency": tpots}


def gate_latency_barely_moves_regime(model_tag: str, trace_dir: Path, residency_pct=25.0, bw_gbps=32.0,
                                      precision="fp16", seed=0, max_rel_spread=0.05) -> dict:
    """Sweeps CXL_LATENCY_NS_GRID (150/250/400ns) and checks TPOT spread is
    small relative to the bandwidth-bound transfer time -- Main.md section
    8.4's claim that the conclusion "does not depend on the parameter you
    could not measure" (latency), because bandwidth dominates."""
    train, test = _split(trace_dir, seed=seed)
    spec = MODEL_SPECS[model_tag]
    tpots = []
    for lat_ns in CXL_LATENCY_NS_GRID:
        res = simulate_policy(decode_train=train, decode_test=test, spec=spec, policy="static-c",
                               residency_pct=residency_pct, bw_gbps=bw_gbps, precision=precision, seed=seed,
                               added_latency_ns=lat_ns)
        tpots.append(res.tpot_ms_mean if res else float("nan"))
    spread = (max(tpots) - min(tpots)) / min(tpots) if min(tpots) > 0 else float("nan")
    return {"passed": spread <= max_rel_spread, "latency_ns_grid": list(CXL_LATENCY_NS_GRID),
            "tpot_ms_by_latency": tpots, "rel_spread": spread}


def gate_oracle_invariant_to_concurrency(model_tag: str, trace_dir: Path, residency_pct=25.0, bw_gbps=32.0,
                                          precision="fp16", concurrency_grid=(1, 2, 4, 8), seed=0) -> dict:
    """Oracle has perfect foreknowledge -- everything it needs is already
    resident when needed, so it never actually contends for the shared link.
    Its TPOT must therefore be exactly the same no matter how many streams
    are (nominally) sharing bandwidth it never uses."""
    df = run_concurrency_sweep(model_tag, trace_dir, list(concurrency_grid), residency_pct=residency_pct,
                                bw_gbps=bw_gbps, precision=precision, policies=["oracle"], seed=seed)
    tpots = df.sort_values("concurrency")["tpot_ms_mean"].tolist()
    spread = (max(tpots) - min(tpots)) / min(tpots) if tpots and min(tpots) > 0 else float("nan")
    return {"passed": bool(spread <= 1e-6), "concurrency_grid": list(concurrency_grid),
            "oracle_tpot_ms_by_concurrency": tpots, "rel_spread": spread}


def gate_tpot_monotone_nondecreasing_in_concurrency(model_tag: str, trace_dir: Path, residency_pct=25.0,
                                                      bw_gbps=32.0, precision="fp16", policy="static-c",
                                                      concurrency_grid=(1, 2, 4, 8), seed=0) -> dict:
    """Closed-loop fair-share (bw/N) can only ever hold a byte-transferring
    policy's TPOT flat or make it worse as more streams share the link, never
    better -- a violation would mean the concurrency division is wired
    backwards or applied to the wrong term."""
    df = run_concurrency_sweep(model_tag, trace_dir, list(concurrency_grid), residency_pct=residency_pct,
                                bw_gbps=bw_gbps, precision=precision, policies=[policy], seed=seed)
    tpots = df.sort_values("concurrency")["tpot_ms_mean"].tolist()
    diffs = np.diff(tpots)
    return {"passed": bool(np.all(diffs >= -1e-6)), "policy": policy, "concurrency_grid": list(concurrency_grid),
            "tpot_ms_by_concurrency": tpots}


def gate_hybrid_never_worse_than_lru_under_concurrency(model_tag: str, trace_dir: Path, residency_pct=25.0,
                                                        bw_gbps=32.0, precision="fp16",
                                                        concurrency_grid=(1, 2, 4, 8), seed=0) -> dict:
    """`hybrid-lru-prefetch` strictly extends plain `lru` (same cache, plus a
    prefetch layer on top of it), so across EVERY concurrency level -- not
    just the concurrency=1 case tests/test_sim_policies.py already checks --
    it must never have worse TPOT. This is what turns "the hybrid design
    mentors asked for is robust" (PREDICTED.md Sec 2.5) into an enforced
    invariant instead of a one-off observation."""
    df = run_concurrency_sweep(model_tag, trace_dir, list(concurrency_grid), residency_pct=residency_pct,
                                bw_gbps=bw_gbps, precision=precision, policies=["lru", "hybrid-lru-prefetch"],
                                seed=seed)
    violations = []
    for c in concurrency_grid:
        sub = df[df.concurrency == c]
        lru_tpot = sub.loc[sub.policy == "lru", "tpot_ms_mean"]
        hybrid_tpot = sub.loc[sub.policy == "hybrid-lru-prefetch", "tpot_ms_mean"]
        if len(lru_tpot) and len(hybrid_tpot) and float(hybrid_tpot.iloc[0]) > float(lru_tpot.iloc[0]) + 1e-9:
            violations.append({"concurrency": c, "lru_tpot_ms": float(lru_tpot.iloc[0]),
                                "hybrid_tpot_ms": float(hybrid_tpot.iloc[0])})
    return {"passed": len(violations) == 0, "concurrency_grid": list(concurrency_grid), "violations": violations}


def find_collapse_concurrency(model_tag: str, trace_dir: Path, residency_pct=25.0, bw_gbps=32.0, precision="fp16",
                               depth=1, concurrency_grid=(1, 2, 4, 8, 16, 32), seed=0,
                               policy="prefetch-freq", rel_tol=1e-6) -> dict:
    """NOT a pass/fail gate -- a quantifying companion. Finds the smallest
    concurrency in `concurrency_grid` at which `policy`'s TPOT becomes
    numerically indistinguishable from `static-c`'s (the byte-budget-limited
    prefetch collapse PREDICTED.md Sec 2.5 documents by hand for two specific
    operating points). Returns `collapse_concurrency=None` if it never
    collapses within the given grid -- e.g. this is expected to return None
    for `policy="hybrid-lru-prefetch"`, which is the point of Sec 2.5's
    finding."""
    df = run_concurrency_sweep(model_tag, trace_dir, list(concurrency_grid), residency_pct=residency_pct,
                                bw_gbps=bw_gbps, precision=precision, depth=depth,
                                policies=["static-c", policy], seed=seed)
    rows, collapse_at = [], None
    for c in concurrency_grid:
        sub = df[df.concurrency == c]
        static_tpot = float(sub.loc[sub.policy == "static-c", "tpot_ms_mean"].iloc[0])
        policy_tpot = float(sub.loc[sub.policy == policy, "tpot_ms_mean"].iloc[0])
        collapsed = abs(policy_tpot - static_tpot) <= rel_tol * max(1.0, static_tpot)
        rows.append({"concurrency": c, "static_tpot_ms": static_tpot, "policy_tpot_ms": policy_tpot,
                     "collapsed": collapsed})
        if collapsed and collapse_at is None:
            collapse_at = c
    return {"policy": policy, "collapse_concurrency": collapse_at, "rows": rows}


def run_validation_suite(model_tag: str, trace_dir: Path, seed: int = 0) -> dict:
    gates = {
        "oracle_is_upper_bound": gate_oracle_is_upper_bound(model_tag, trace_dir, seed=seed),
        "zero_latency_infinite_bw_parity": gate_zero_latency_infinite_bw_parity(model_tag, trace_dir, seed=seed),
        "monotone_in_residency": gate_monotone_in_residency(model_tag, trace_dir, seed=seed),
        "latency_barely_moves_regime": gate_latency_barely_moves_regime(model_tag, trace_dir, seed=seed),
        "oracle_invariant_to_concurrency": gate_oracle_invariant_to_concurrency(model_tag, trace_dir, seed=seed),
        "tpot_monotone_in_concurrency": gate_tpot_monotone_nondecreasing_in_concurrency(model_tag, trace_dir, seed=seed),
        "hybrid_never_worse_than_lru_under_concurrency": gate_hybrid_never_worse_than_lru_under_concurrency(
            model_tag, trace_dir, seed=seed),
    }
    gates["all_passed"] = all(g["passed"] for g in gates.values())
    gates["concurrency_collapse_analysis"] = find_collapse_concurrency(model_tag, trace_dir, seed=seed)
    return gates
