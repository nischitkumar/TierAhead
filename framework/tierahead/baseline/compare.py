"""The flagship end-to-end entry point: **baseline ("normal", HBM-only, no
CXL tier) vs CXL-tiered TierAhead**, on identical held-out traces.

"Normal" here means exactly what the MoE memory wall (Main.md section 1.2)
describes: every expert weight must fit in HBM, full stop -- no capacity
tier, so no tiering-induced stall is possible, but ALSO no relief if the
model doesn't physically fit. This module reports both halves honestly: (1)
does the model even fit in a given HBM budget without CXL at all
(``capacity_check``), and (2) for models that fit either way, what does
CXL-tiering cost in TPOT/stall-rate for how much HBM it frees up
(``run_baseline_vs_cxl``) -- run through ``tierahead.sim.backends`` so the
"des" (always available) and "cxlmemsim" (real epoch-based emulation, Linux-
only) backends are interchangeable from this one call site.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tierahead.sim.backends import select_backend
from tierahead.sim.backends.des import run as run_des
from tierahead.specs import MODEL_SPECS
from tierahead.traces.io import resolve_trace_dir


def capacity_check(model_tag: str, precision: str = "fp16", hbm_budget_gb: float = 80.0) -> dict:
    """The memory-wall arithmetic Main.md leads with: total expert-parameter
    footprint vs a given HBM budget (default 80GB, H100 SXM). No CXL tier
    changes this number -- it is what "normal" cannot get around.

    This is the expert-weights-only view. It does NOT account for KV cache,
    which also lives in HBM and competes for the same budget -- see
    `kv_hbm_pressure` / `capacity_check_with_kv` for that (mentor-review
    addition: "accounting for HBM pressure from KV cache competing with
    prefetched experts"). Kept separate (not folded in here with a
    default-off KV size) so existing callers that only care about expert
    weights get a stable, simple answer.
    """
    spec = MODEL_SPECS[model_tag]
    total_expert_bytes = spec.n_layers * spec.n_experts * spec.expert_bytes(precision)
    total_gb = total_expert_bytes / 1e9
    return {
        "model": model_tag, "precision": precision, "hbm_budget_gb": hbm_budget_gb,
        "total_expert_weight_gb": total_gb, "fits_in_hbm_alone": total_gb <= hbm_budget_gb,
        "overflow_gb": max(0.0, total_gb - hbm_budget_gb),
    }


def kv_hbm_pressure(model_tag: str, hbm_budget_gb: float, concurrent_requests: int, context_len: int,
                     kv_precision: str = "fp16", expert_precision: str = "fp16") -> dict:
    """How much of the HBM budget KV cache reserves for `concurrent_requests`
    simultaneous requests each holding `context_len` tokens of history, and
    what residency percentage that leaves available for resident experts.

    KV cache is activations (recomputed fresh, not weights) and in a "normal"
    (no-CXL) system it has nowhere else to live but HBM, right alongside
    expert weights -- so it directly reduces how many experts can be pinned
    resident, independent of whatever --residency-pct was requested. This
    was previously unmodeled: `run_baseline_vs_cxl` treated the full HBM
    budget as available for expert residency, silently assuming zero
    concurrent long-context load.
    """
    spec = MODEL_SPECS[model_tag]
    kv_gb_per_request = spec.kv_total_bytes(context_len, kv_precision) / 1e9
    kv_gb_total = kv_gb_per_request * max(1, concurrent_requests)
    total_expert_gb = spec.n_layers * spec.n_experts * spec.expert_bytes(expert_precision) / 1e9
    hbm_left_for_experts_gb = max(0.0, hbm_budget_gb - kv_gb_total)
    max_residency_pct_given_kv = (
        100.0 * min(1.0, hbm_left_for_experts_gb / total_expert_gb) if total_expert_gb > 0 else 0.0
    )
    return {
        "model": model_tag, "hbm_budget_gb": hbm_budget_gb, "concurrent_requests": concurrent_requests,
        "context_len": context_len, "kv_precision": kv_precision,
        "kv_gb_per_request": kv_gb_per_request, "kv_gb_total": kv_gb_total,
        "total_expert_weight_gb": total_expert_gb, "hbm_left_for_experts_gb": hbm_left_for_experts_gb,
        "max_residency_pct_given_kv": max_residency_pct_given_kv,
        "kv_alone_exceeds_hbm_budget": kv_gb_total > hbm_budget_gb,
    }


def capacity_check_with_kv(model_tag: str, precision: str = "fp16", hbm_budget_gb: float = 80.0,
                           concurrent_requests: int = 1, context_len: int = 4096,
                           kv_precision: str = "fp16") -> dict:
    """capacity_check(), extended: does expert weights + KV cache for
    `concurrent_requests` x `context_len` fit in HBM together, with NO CXL
    tier at all (the true "normal" system, where both weights and KV cache
    have nowhere else to live)."""
    base = capacity_check(model_tag, precision, hbm_budget_gb)
    kv = kv_hbm_pressure(model_tag, hbm_budget_gb, concurrent_requests, context_len, kv_precision, precision)
    total_needed_gb = base["total_expert_weight_gb"] + kv["kv_gb_total"]
    return {
        **base, "kv_gb_total": kv["kv_gb_total"], "total_needed_with_kv_gb": total_needed_gb,
        "fits_in_hbm_with_kv": total_needed_gb <= hbm_budget_gb,
        "overflow_with_kv_gb": max(0.0, total_needed_gb - hbm_budget_gb),
        "max_residency_pct_given_kv": kv["max_residency_pct_given_kv"],
    }


def run_baseline_vs_cxl(model_tag: str, residency_pct: float = 25.0, bw_gbps: float = 32.0,
                          precision: str = "fp16", policies: list[str] | None = None, batch: int = 1,
                          calib: dict | None = None, added_latency_ns: float = 0.0, backend: str = "auto",
                          depth: int = 1, m_mult: int = 2, concurrency: int = 1,
                          concurrent_requests: int = 1, context_len: int = 0, kv_precision: str = "fp16",
                          hbm_budget_gb: float = 80.0, seed: int = 0, data_root: Path | None = None) -> dict:
    """Returns a dict with:
      - capacity: capacity_check() output (does "normal" even work at all)
      - kv_pressure: kv_hbm_pressure() output if context_len > 0, else None
                     (KV cache's HBM footprint, and the residency ceiling it
                     imposes -- see kv_hbm_pressure's docstring). Default
                     context_len=0 skips this and behaves exactly as before
                     (single-request, no long-context KV pressure modeled).
      - baseline_tpot_ms: HBM-only TPOT (residency=100%, no stalls possible)
      - policies_df: DataFrame from the selected backend's policy bake-off
                     at the CONFIGURED (residency_pct, bw_gbps) CXL operating
                     point -- capped by kv_pressure's residency ceiling if
                     that's tighter than the requested residency_pct -- with
                     an extra `tpot_overhead_vs_baseline_pct` column added
                     here (backend-agnostic). `concurrency` (N simultaneous
                     decode streams sharing the link, fair-share
                     approximation -- see tierahead.sim.policies' module
                     docstring) is threaded straight through.
      - hbm_bytes_freed_gb: how much HBM the CXL tier frees up at this
                     residency level, vs keeping everything resident.
      - headline: one auto-generated sentence, matching this repo's own
                     `headline_sentences` convention elsewhere.
    """
    spec = MODEL_SPECS[model_tag]
    trace_dir = resolve_trace_dir(spec.trace_dir_default, data_root)
    backend_name, backend_note = select_backend(backend)

    capacity = capacity_check(model_tag, precision, hbm_budget_gb)

    kv_pressure = None
    effective_residency_pct = residency_pct
    if context_len > 0:
        kv_pressure = kv_hbm_pressure(model_tag, hbm_budget_gb, concurrent_requests, context_len,
                                       kv_precision, precision)
        effective_residency_pct = min(residency_pct, kv_pressure["max_residency_pct_given_kv"])

    if backend_name == "cxlmemsim":
        from tierahead.sim.backends import cxlmemsim
        cxlmemsim.run()  # raises with an actionable message -- see that module
    # backend_name == "des" (the only backend that can actually execute here)
    baseline_df = run_des(model_tag, trace_dir, residency_pct=100.0, bw_gbps=bw_gbps, precision=precision,
                            policies=["static-c"], batch=batch, calib=calib, seed=seed)
    baseline_tpot_ms = float(baseline_df["tpot_ms_mean"].iloc[0]) if len(baseline_df) else float("nan")

    policies_df = run_des(model_tag, trace_dir, residency_pct=effective_residency_pct, bw_gbps=bw_gbps,
                            precision=precision, policies=policies, batch=batch, calib=calib,
                            added_latency_ns=added_latency_ns, depth=depth, m_mult=m_mult,
                            concurrency=concurrency, seed=seed)
    if len(policies_df):
        policies_df["tpot_overhead_vs_baseline_pct"] = (
            100.0 * (policies_df["tpot_ms_mean"] - baseline_tpot_ms) / baseline_tpot_ms
        )

    total_expert_bytes = spec.n_layers * spec.n_experts * spec.expert_bytes(precision)
    hbm_bytes_freed_gb = total_expert_bytes * (1 - effective_residency_pct / 100.0) / 1e9

    best_row = None
    if len(policies_df):
        non_baseline = policies_df[~policies_df.policy.isin(["none", "oracle"])]
        if len(non_baseline):
            best_row = non_baseline.loc[non_baseline["tpot_ms_mean"].idxmin()]

    kv_clause = ""
    if kv_pressure is not None and effective_residency_pct < residency_pct - 1e-9:
        kv_clause = (
            f" (KV cache for {concurrent_requests} concurrent request(s) at {context_len} tokens reserves "
            f"{kv_pressure['kv_gb_total']:.2f} GB of HBM, capping residency at {effective_residency_pct:.1f}% "
            f"instead of the requested {residency_pct:.1f}%)"
        )

    if not capacity["fits_in_hbm_alone"]:
        headline = (
            f"{model_tag} ({precision}) needs {capacity['total_expert_weight_gb']:.1f} GB of expert weights, "
            f"which does NOT fit in a {hbm_budget_gb:.0f} GB HBM-only 'normal' baseline at all "
            f"(overflow {capacity['overflow_gb']:.1f} GB) -- CXL tiering isn't an optimization here, it's "
            f"the only way to serve this model on this accelerator."
        )
    elif kv_pressure is not None and kv_pressure["kv_alone_exceeds_hbm_budget"]:
        headline = (
            f"{model_tag} ({precision}) expert weights alone fit in {hbm_budget_gb:.0f} GB HBM, but KV cache for "
            f"{concurrent_requests} concurrent request(s) at {context_len} tokens needs "
            f"{kv_pressure['kv_gb_total']:.2f} GB on its own -- exceeds the ENTIRE HBM budget before a single "
            f"expert weight is placed. Zero residency is achievable; every expert access is a CXL demand fetch."
        )
    elif best_row is not None:
        headline = (
            f"{model_tag} ({precision}) fits HBM-only at {capacity['total_expert_weight_gb']:.1f} GB, but tiering "
            f"{100 - effective_residency_pct:.1f}% of experts to a {bw_gbps:.0f} GB/s CXL link frees "
            f"{hbm_bytes_freed_gb:.2f} GB of HBM at a {best_row['tpot_overhead_vs_baseline_pct']:.1f}% TPOT cost "
            f"under '{best_row['policy']}' (backend={backend_name}, concurrency={concurrency}x){kv_clause}."
        )
    else:
        headline = f"{model_tag}: no non-baseline policy result available (backend={backend_name})."

    return {
        "model": model_tag, "backend": backend_name, "backend_note": backend_note,
        "capacity": capacity, "kv_pressure": kv_pressure, "baseline_tpot_ms": baseline_tpot_ms,
        "residency_pct": residency_pct, "effective_residency_pct": effective_residency_pct,
        "bw_gbps": bw_gbps, "precision": precision, "concurrency": concurrency,
        "hbm_bytes_freed_gb": hbm_bytes_freed_gb, "policies_df": policies_df, "headline": headline,
    }


def run_flagship_report(model_tag: str, residency_pct: float = 25.0, bw_gbps: float = 32.0,
                         precision: str = "fp16", policies: list[str] | None = None, batch: int = 1,
                         calib: dict | None = None, added_latency_ns: float = 0.0, backend: str = "auto",
                         depth: int = 1, m_mult: int = 2, concurrency_grid: list[int] | None = None,
                         concurrent_requests: int = 1, context_len: int = 0, kv_precision: str = "fp16",
                         hbm_budget_gb: float = 80.0, seed: int = 0, data_root: Path | None = None) -> dict:
    """The single-call version of "normal (HBM-only) vs HBM+CXL, at N
    concurrent requests, with M concurrent long-context sessions' worth of KV
    pressure eating into the HBM budget" (mentor point 7, fully realized).

    Before this, seeing the complete picture meant calling
    `run_baseline_vs_cxl` once per concurrency level by hand, threading
    `concurrency`/`concurrent_requests`/`context_len` through separately each
    time. This wraps that loop: runs `run_baseline_vs_cxl` once per level in
    `concurrency_grid` (default `[1]`, i.e. identical to a single
    `run_baseline_vs_cxl` call -- this function is a strict superset, not a
    different code path), and returns:
      - capacity / kv_pressure: identical at every concurrency level (neither
        depends on it), taken from the first run.
      - by_concurrency: {concurrency: full run_baseline_vs_cxl() result dict}
        for anyone who wants one level's full detail.
      - combined_policies_df: every level's `policies_df` stacked into one
        DataFrame (each row already carries its own `concurrency` column via
        `PolicyResult`), ready to plot TPOT/oracle-gap-closed vs concurrency
        per policy in one shot -- see `tierahead.dashboard.app`'s Policy
        explorer tab for exactly this use.
      - headlines: {concurrency: headline sentence} from each level's own
        `run_baseline_vs_cxl` call, unmodified -- this function does not
        synthesize its own narrative on top, to avoid baking one write-up's
        framing into library code that other call sites also use.
    """
    grid = list(concurrency_grid) if concurrency_grid else [1]
    by_concurrency = {
        c: run_baseline_vs_cxl(model_tag, residency_pct=residency_pct, bw_gbps=bw_gbps, precision=precision,
                                policies=policies, batch=batch, calib=calib, added_latency_ns=added_latency_ns,
                                backend=backend, depth=depth, m_mult=m_mult, concurrency=c,
                                concurrent_requests=concurrent_requests, context_len=context_len,
                                kv_precision=kv_precision, hbm_budget_gb=hbm_budget_gb, seed=seed,
                                data_root=data_root)
        for c in grid
    }
    first = by_concurrency[grid[0]]
    frames = [r["policies_df"] for r in by_concurrency.values() if len(r["policies_df"])]
    combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "model": model_tag, "backend": first["backend"], "concurrency_grid": grid,
        "capacity": first["capacity"], "kv_pressure": first["kv_pressure"],
        "residency_pct": residency_pct, "bw_gbps": bw_gbps, "precision": precision,
        "by_concurrency": by_concurrency, "combined_policies_df": combined_df,
        "headlines": {c: r["headline"] for c, r in by_concurrency.items()},
    }
