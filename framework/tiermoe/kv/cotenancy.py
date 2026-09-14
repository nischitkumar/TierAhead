"""[8] KV-cache co-tenancy on the shared CXL link (Main.md's E8 scope).

Ported from experiments/e8_kv_cotenancy/kv_link_sim.py, but now built on top
of ``tiermoe.sim.engine``'s general N-class fluid DES instead of its own
bespoke two-class simulator -- exactly the consolidation E8's own design doc
recommended once a general engine existed. Every finding that module's real
run produced (KV and expert traffic genuinely contend; the recommended
policy is bandwidth-dependent, not universal) carries over unchanged because
the underlying event-driven mechanics are identical, just generalized.

No GPU, no model weights: expert traffic reuses tiermoe.roofline's
bytes_per_layer/get_t_compute_ms over already-collected traces; KV traffic
is a Poisson arrival process of ModelSpec.kv_total_bytes(context_len) jobs
(context length drawn from the configured mix). See
experiments/e8_kv_cotenancy/E8_KV_COTENANCY.md for the full first-principles
derivation of the KV-bytes formula and the co-tenancy question this answers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tiermoe.roofline.model import bytes_per_layer, build_e_union_lookup, get_t_compute_ms
from tiermoe.sim.engine import TrafficClass, pareto_labels, pctile_stats, pooled_simulate
from tiermoe.specs import MODEL_SPECS
from tiermoe.traces.io import resolve_trace_dir

PARETO_ROUND_DECIMALS = 6


def _kv_weight_for_policy(policy: str) -> float:
    if policy == "expert-first":
        return 0.0
    if policy == "kv-first":
        return 1.0
    if policy.startswith("weighted:"):
        w = float(policy.split(":", 1)[1])
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"weighted policy fraction must be in [0,1], got {w}")
        return w
    raise ValueError(f"unknown policy {policy!r}, expected 'expert-first', 'kv-first', or 'weighted:<w>'")


def compute_expert_traffic_params(spec, trace_dir: Path, batch: int, residency_pct: float, precision: str,
                                   calib, n_trials: int, seed: int) -> dict:
    """One periodic job per decode token: `expert_job_bytes` (total
    non-resident bytes across ALL layers) every `expert_period_ms` (that
    token's total compute time across all layers). Coarsens the per-layer
    roofline model into one job per token so the KV stream's much coarser
    timescale (new requests, not new layers) can be simulated jointly
    without an intractable event count -- the same simplification E8
    documented and justified."""
    mean_by_batch, _ = build_e_union_lookup(spec.tag, trace_dir, [batch], n_trials=n_trials, seed=seed)
    e_union_val = mean_by_batch[batch]
    bpl = bytes_per_layer(spec, batch, residency_pct, precision, e_union_val)
    t_compute_layer_ms, provenance = get_t_compute_ms(spec, batch, calib)
    return {
        "e_union_val": e_union_val, "expert_job_bytes": bpl * spec.n_layers,
        "expert_period_ms": t_compute_layer_ms * spec.n_layers, "compute_provenance": provenance,
    }


def make_kv_job_sampler(spec, context_lens, weights, kv_precision):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    def sampler(rng):
        c = rng.choice(context_lens, p=weights)
        return spec.kv_total_bytes(int(c), kv_precision)

    return sampler


def pareto_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pareto_optimal"] = False
    for (model, bw), sub in df.groupby(["model", "bw_gbps"]):
        pts = [(round(r.ttft_p50_ms, PARETO_ROUND_DECIMALS), round(r.tpot_inflation_p50_ms, PARETO_ROUND_DECIMALS), idx)
               for idx, r in sub.iterrows()]
        winners = set(pareto_labels(pts))
        df.loc[list(winners), "pareto_optimal"] = True
    return df


# Backwards-compatible alias (the name used by earlier drafts / dashboard code).
add_pareto_column = pareto_column


def run_kv_expert_sweep(models: list[str], bw_grid: list[float], policies: list[str], *, batch: int = 32,
                         residency_pct: float = 75.0, expert_precision: str = "nf4", kv_precision: str = "fp16",
                         context_lens=(4096, 32768, 131072), context_weights=None, kv_rate_per_sec: float = 2.0,
                         sim_seconds: float = 30.0, seed: int = 0, n_seeds: int = 5, n_trials: int = 500,
                         calib: dict | None = None, data_root: Path | None = None) -> pd.DataFrame:
    """Default operating point (batch=32, residency=75%, expert precision
    nf4) matches E1's own finding for where 'router-guided prefetch is worth
    something' for BOTH shipped models -- so any contention effect visible
    here is attributable to KV traffic, not to an already link-saturated
    expert stream (E8_KV_COTENANCY.md section 7.5's own reasoning, carried
    forward unchanged)."""
    context_weights = context_weights or [1.0] * len(context_lens)
    rows = []
    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        trace_dir = resolve_trace_dir(spec.trace_dir_default, data_root)
        traffic = compute_expert_traffic_params(spec, trace_dir, batch, residency_pct, expert_precision,
                                                  calib, n_trials, seed)
        kv_sampler = make_kv_job_sampler(spec, list(context_lens), context_weights, kv_precision)

        for bw in bw_grid:
            iso_expert = pooled_simulate(
                n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, sim_ms=sim_seconds * 1000.0,
                classes=[TrafficClass("expert", weight=1.0, period_ms=traffic["expert_period_ms"],
                                       job_bytes=traffic["expert_job_bytes"])])
            iso_kv = pooled_simulate(
                n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, sim_ms=sim_seconds * 1000.0,
                classes=[TrafficClass("kv", weight=1.0, arrival_rate_per_sec=kv_rate_per_sec, job_sampler=kv_sampler)])

            for policy in policies:
                w_kv = _kv_weight_for_policy(policy)
                classes = [
                    TrafficClass("expert", weight=1.0 - w_kv, period_ms=traffic["expert_period_ms"],
                                 job_bytes=traffic["expert_job_bytes"]),
                    TrafficClass("kv", weight=w_kv, arrival_rate_per_sec=kv_rate_per_sec, job_sampler=kv_sampler),
                ]
                res = pooled_simulate(n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, sim_ms=sim_seconds * 1000.0,
                                       classes=classes)
                e_stats = pctile_stats(res.get("expert", []))
                k_stats = pctile_stats(res.get("kv", []))
                rows.append({
                    "model": model_tag, "bw_gbps": bw, "policy": policy, "w_kv": w_kv,
                    "ttft_p50_ms": k_stats["latency_p50_ms"], "ttft_p95_ms": k_stats["latency_p95_ms"],
                    "ttft_mean_ms": k_stats["latency_mean_ms"], "n_kv_completions": k_stats["n"],
                    "tpot_inflation_p50_ms": e_stats["inflation_p50_ms"],
                    "tpot_inflation_p95_ms": e_stats["inflation_p95_ms"],
                    "tpot_inflation_mean_ms": e_stats["inflation_mean_ms"],
                    "n_expert_completions": e_stats["n"],
                    "expert_isolated_p50_ms": iso_expert and pctile_stats(iso_expert.get("expert", []))["inflation_p50_ms"],
                    "kv_isolated_p50_ms": iso_kv and pctile_stats(iso_kv.get("kv", []))["latency_p50_ms"],
                })
    df = pd.DataFrame(rows)
    return pareto_column(df) if len(df) else df


def recommend_policy(df: pd.DataFrame, ttft_slo_ms: float | None = None) -> dict:
    """Mirrors E2's B* pattern: argmin TPOT inflation subject to a TTFT SLO,
    or (no SLO given) report both Pareto extremes rather than an arbitrary
    scalarization -- see the original module's rejected-scalarization notes
    in experiments/e8_kv_cotenancy/E8_KV_COTENANCY.md section 7.3/7.4 for why
    a single "always pick the lowest-inflation policy" rule is a degenerate,
    non-informative recommendation."""
    out = {}
    for (model, bw), sub in df.groupby(["model", "bw_gbps"]):
        pareto = sub[sub.pareto_optimal]
        if len(pareto) == 0:
            continue
        key = f"{model}_{bw}gbps"
        best_ttft = pareto.loc[pareto["ttft_p50_ms"].idxmin()]
        best_tpot = pareto.loc[pareto["tpot_inflation_p50_ms"].idxmin()]
        if ttft_slo_ms is None:
            out[key] = {"pareto_policies": pareto["policy"].tolist(), "recommended_policy": None,
                        "best_for_ttft": {"policy": best_ttft["policy"], "ttft_p50_ms": float(best_ttft["ttft_p50_ms"])},
                        "best_for_tpot": {"policy": best_tpot["policy"], "tpot_inflation_p50_ms": float(best_tpot["tpot_inflation_p50_ms"])}}
            continue
        feasible = pareto[pareto["ttft_p50_ms"] <= ttft_slo_ms]
        if len(feasible) == 0:
            out[key] = {"recommended_policy": best_ttft["policy"], "note": f"no policy meets {ttft_slo_ms}ms TTFT SLO"}
        else:
            best = feasible.loc[feasible["tpot_inflation_p50_ms"].idxmin()]
            out[key] = {"recommended_policy": best["policy"], "ttft_p50_ms": float(best["ttft_p50_ms"]),
                        "tpot_inflation_p50_ms": float(best["tpot_inflation_p50_ms"])}
    return out
