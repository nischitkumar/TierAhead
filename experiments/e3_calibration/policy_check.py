"""Validates the PCIe offload-policy ORDERING E3 step 3 measures. Pure
logic, no GPU required -- `e2e_offload_runner.py` is the GPU half that
produces the tpot_ms_by_policy dict this function checks.

Experiments.md's exit criteria phrases the expected ordering as "PCIe policy
ordering measured (expect next-layer-topk >= static-hot > none)", which
reads naturally under a higher-is-better metric (e.g. throughput = 1/TPOT).
This experiment measures decode-step LATENCY (TPOT) directly, where LOWER is
better, so the equivalent statement in these units -- and what this function
actually checks -- is: next-layer-topk <= static-hot <= none. next-layer-
topk should be fastest (it prefetches ahead of need), none should be
slowest (every non-resident expert is a synchronous on-demand fetch),
static-hot sits in between (some experts never need fetching at all, but
whichever do are still fetched on demand, no prefetch).
"""
POLICY_ORDER = ["next-layer-topk", "static-hot", "none"]  # best (lowest TPOT) first


def check_policy_ordering(tpot_ms_by_policy: dict, tolerance_pct: float = 2.0):
    """Returns (passed: bool, message: str). Raises KeyError if a required
    policy is missing (a silent mis-ranking on incomplete data would be
    worse than a loud, immediate failure here).

    tolerance_pct allows `better` to exceed `worse` by up to that fraction of
    `worse` before it counts as a real ordering violation -- real
    measurements have run-to-run noise, and a difference inside the noise
    floor isn't evidence the ordering is actually wrong."""
    missing = [p for p in POLICY_ORDER if p not in tpot_ms_by_policy]
    if missing:
        raise KeyError(f"missing required polic{'y' if len(missing) == 1 else 'ies'} "
                        f"in tpot_ms_by_policy: {missing}")

    violations = []
    for better, worse in zip(POLICY_ORDER, POLICY_ORDER[1:]):
        t_better, t_worse = tpot_ms_by_policy[better], tpot_ms_by_policy[worse]
        allowed = t_worse * (1.0 + tolerance_pct / 100.0)
        if t_better > allowed:
            violations.append(
                f"{better} ({t_better:.4f} ms) should be <= {worse} ({t_worse:.4f} ms) "
                f"+ {tolerance_pct:g}% tolerance ({allowed:.4f} ms), but isn't"
            )

    if violations:
        return False, "FAIL: " + "; ".join(violations)
    ordered = " <= ".join(f"{p}={tpot_ms_by_policy[p]:.4f}ms" for p in POLICY_ORDER)
    return True, f"PASS: {ordered} (tolerance {tolerance_pct:g}%)"
