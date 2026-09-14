"""Validates the PCIe offload-policy ORDERING (Main.md's E3 step 3 exit
criteria). Pure logic, no GPU -- e2e_offload.py is the GPU half that
produces the tpot_ms_by_policy dict this checks.

Expected ordering under a lower-is-better latency (TPOT) metric:
next-layer-topk <= static-hot <= none. next-layer-topk should be fastest
(prefetches ahead of need); none should be slowest (every non-resident
expert is a synchronous on-demand fetch); static-hot sits in between.
"""
from __future__ import annotations

POLICY_ORDER = ["next-layer-topk", "static-hot", "none"]  # best (lowest TPOT) first


def check_policy_ordering(tpot_ms_by_policy: dict, tolerance_pct: float = 2.0) -> tuple[bool, str]:
    missing = [p for p in POLICY_ORDER if p not in tpot_ms_by_policy]
    if missing:
        raise KeyError(f"missing required polic{'y' if len(missing) == 1 else 'ies'} in tpot_ms_by_policy: {missing}")

    violations = []
    for better, worse in zip(POLICY_ORDER, POLICY_ORDER[1:]):
        t_better, t_worse = tpot_ms_by_policy[better], tpot_ms_by_policy[worse]
        allowed = t_worse * (1.0 + tolerance_pct / 100.0)
        if t_better > allowed:
            violations.append(f"{better} ({t_better:.4f}ms) should be <= {worse} ({t_worse:.4f}ms) "
                               f"+ {tolerance_pct:g}% tolerance ({allowed:.4f}ms), but isn't")
    if violations:
        return False, "FAIL: " + "; ".join(violations)
    ordered = " <= ".join(f"{p}={tpot_ms_by_policy[p]:.4f}ms" for p in POLICY_ORDER)
    return True, f"PASS: {ordered} (tolerance {tolerance_pct:g}%)"
