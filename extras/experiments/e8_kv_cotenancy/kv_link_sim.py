#!/usr/bin/env python3
"""E8 -- KV-cache co-tenancy on the CXL link (Experiments.md E8, [P2]).

Astera's own published CXL-for-inference story is about the KV cache, not
experts. This experiment shows the TierAhead framework handles both traffic
classes, and -- the part that makes it more than a footnote -- that they
CONTEND for the same physical link when a long-context request's KV cache
has to live on CXL (doesn't fit in HBM) at the same time expert weights are
being prefetched from CXL for ongoing decode.

Method (Experiments.md E8 steps 1-3):
  1. KV-cache bytes per token, per request: 2*L*n_kv_heads*d_head*precision
     (see common.model_specs.ModelSpec.kv_bytes_per_token/kv_total_bytes).
     Swept over "realistic context lengths" 4k/32k/128k.
  2. Simulate a serving mix: one continuous stream of periodic expert-prefetch
     jobs (paced by decode's per-token compute time, reusing E1's roofline
     model for job size and cadence) sharing a single CXL link with a Poisson
     stream of KV-cache transfer jobs (one per new long-context request),
     under three scheduling policies: strict kv-first, strict expert-first,
     and a weighted bandwidth split.
  3. Report TTFT (the KV stream's completion latency -- the link-contention
     component of it, not full end-to-end TTFT) and TPOT inflation (the
     extra time added to expert fetches by KV traffic stealing bandwidth),
     find the policy that isn't Pareto-dominated by another.

No GPU, no model weights, no Mistral/Mixtral install required: this is a
pure discrete-event queueing simulation over already-collected pilot traces
(for E_union(B), reused from E1) plus an architecture-derived KV formula.
Runs anywhere Python/numpy/pandas/matplotlib run, including a laptop.

Scope note vs Experiments.md's phrasing ("simulator reuse"): the doc assumes
E6 (the full policy-bakeoff simulator) already exists and E8 just adds a KV
traffic class to it. E6 hasn't been built yet (only E1 and E2 have, as of
this writing) -- so E8 instead reuses E1's roofline primitives directly
(bytes_per_layer, get_t_compute_ms) and implements its own minimal
discrete-event link simulator, self-contained. See E8_KV_COTENANCY.md
"Relationship to E6" for the honest accounting of what that does and
doesn't cover.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                  # experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1_roofline"))  # e1_roofline/

from common.model_specs import (  # noqa: E402
    MODEL_SPECS, LINK_BW_GBPS_GRID, CONTEXT_LEN_GRID_E8, KV_PRECISION_GRID,
)
import roofline as rl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POLICIES = ["expert-first", "kv-first", "weighted:0.2", "weighted:0.5", "weighted:0.8"]
COMPLETION_EPS_BYTES = 1e-3   # float-remaining-bytes tolerance for "job done"
ARRIVAL_EPS_MS = 1e-9          # float time tolerance for "arrival due now"
MAX_EVENTS_SAFETY = 5_000_000  # guards against a misconfigured infinite loop


# ---------- job / policy primitives ----------

@dataclass
class Job:
    arrival_ms: float
    total_bytes: float
    remaining_bytes: float


def make_rate_fn(policy: str):
    """Returns (rate_fn, kind, w_kv) where rate_fn(expert_present, kv_present,
    bw_bytes_per_ms) -> (r_expert, r_kv), both in bytes/ms. Unifies all three
    Experiments.md E8 policies as one weighted split: expert-first == weight 0
    (KV gets nothing while expert is queued), kv-first == weight 1, and an
    explicit `weighted:w` gives KV a fixed w fraction of the link whenever
    both classes have something queued. Either class gets the FULL link
    whenever it is the only one with something queued -- a shared link
    doesn't waste bandwidth just because only one class needs it right now.
    """
    if policy == "expert-first":
        w_kv, kind = 0.0, "expert-first"
    elif policy == "kv-first":
        w_kv, kind = 1.0, "kv-first"
    elif policy.startswith("weighted:"):
        w_kv = float(policy.split(":", 1)[1])
        if not (0.0 <= w_kv <= 1.0):
            raise ValueError(f"weighted policy fraction must be in [0,1], got {w_kv}")
        kind = "weighted"
    else:
        raise ValueError(f"unknown policy {policy!r}, expected 'expert-first', 'kv-first', or 'weighted:<w>'")

    def rate_fn(expert_present: bool, kv_present: bool, bw: float):
        if expert_present and kv_present:
            return bw * (1.0 - w_kv), bw * w_kv
        if expert_present:
            return bw, 0.0
        if kv_present:
            return 0.0, bw
        return 0.0, 0.0

    return rate_fn, kind, (w_kv if kind == "weighted" else None)


# ---------- the discrete-event fluid simulator ----------

def simulate_link(*, bw_gbps: float, policy: str,
                   expert_period_ms: float, expert_job_bytes: float,
                   kv_rate_per_sec: float, kv_job_sampler, sim_ms: float,
                   seed: int, warmup_ms: float = 0.0):
    """One run of the shared-link simulation. `kv_job_sampler(rng) -> bytes`
    draws one KV job's size (a request's full KV-cache footprint at its
    sampled context length). Expert jobs arrive deterministically, one per
    `expert_period_ms` (one per decode token, paced by compute), each of
    fixed size `expert_job_bytes` (the token's total non-resident expert
    bytes across all layers, per E1's bytes_per_layer model). Both classes
    are FIFO within themselves; only the head-of-line job of each class is
    "in service" at any instant, receiving that class's policy-allocated
    share of link bandwidth (see make_rate_fn) -- other queued same-class
    jobs simply wait their turn, as they would behind one real fetch engine.

    This is an EXACT event-driven fluid simulation (not a fixed-timestep
    approximation): the link rate a job receives only ever changes at an
    arrival or a completion, so stepping exactly to the next such instant and
    integrating bytes-served linearly over that interval introduces no
    discretization error.

    `warmup_ms` (default 0) lets a caller discard early jobs to avoid
    startup-transient bias in percentile stats; unused by default because at
    these arrival rates the transient is a tiny fraction of sim_ms anyway
    (see the design doc's "what this simulator does NOT model" section).

    Returns dict with 'expert' and 'kv' completion records: each a list of
    (arrival_ms, latency_ms, ideal_solo_ms) tuples.
    """
    rng = np.random.default_rng(seed)
    bw_bytes_per_ms = bw_gbps * 1e6  # GB/s = 1e9 B/s = 1e6 B/ms
    rate_fn, _, _ = make_rate_fn(policy)

    # Pre-generate arrival times so the event loop only ever has to look at
    # "what's next", never regenerate randomness mid-loop.
    expert_arrivals = list(np.arange(expert_period_ms, sim_ms, expert_period_ms)) if expert_period_ms > 0 else []

    kv_arrivals = []
    if kv_rate_per_sec > 0:
        t = 0.0
        while True:
            t += rng.exponential(1000.0 / kv_rate_per_sec)  # ms
            if t > sim_ms:
                break
            kv_arrivals.append(t)
    kv_job_sizes = [kv_job_sampler(rng) for _ in kv_arrivals]

    expert_q: list[Job] = []
    kv_q: list[Job] = []
    ei, ki = 0, 0
    t = 0.0
    expert_done, kv_done = [], []
    n_events = 0

    while True:
        n_events += 1
        if n_events > MAX_EVENTS_SAFETY:
            raise RuntimeError("simulate_link: exceeded MAX_EVENTS_SAFETY -- check period/rate configuration")

        e_head = expert_q[0] if expert_q else None
        k_head = kv_q[0] if kv_q else None
        r_e, r_k = rate_fn(e_head is not None, k_head is not None, bw_bytes_per_ms)

        candidates = []
        if e_head is not None and r_e > 0:
            candidates.append(t + e_head.remaining_bytes / r_e)
        if k_head is not None and r_k > 0:
            candidates.append(t + k_head.remaining_bytes / r_k)
        if ei < len(expert_arrivals):
            candidates.append(expert_arrivals[ei])
        if ki < len(kv_arrivals):
            candidates.append(kv_arrivals[ki])
        if not candidates:
            break  # nothing queued, nothing left to arrive -- done

        t_next = min(candidates)
        if t_next > sim_ms:
            # Right-censor: don't advance past the simulation horizon just to
            # let a straggler job "complete" -- at near-zero bandwidth this
            # candidate can be astronomically far in the future (bytes/rate
            # with rate->0), and jumping there would both blow the wall-clock
            # budget and silently count a job as "done" outside the window it
            # was measured in. Any job still in flight at sim_ms is simply
            # excluded from the completion stats (standard finite-horizon
            # right-censoring), not force-completed.
            break
        dt = max(0.0, t_next - t)
        if e_head is not None and r_e > 0:
            e_head.remaining_bytes -= r_e * dt
        if k_head is not None and r_k > 0:
            k_head.remaining_bytes -= r_k * dt
        t = t_next

        if e_head is not None and e_head.remaining_bytes <= COMPLETION_EPS_BYTES:
            expert_q.pop(0)
            if e_head.arrival_ms >= warmup_ms:
                expert_done.append((e_head.arrival_ms, t - e_head.arrival_ms, e_head.total_bytes / bw_bytes_per_ms))
        if k_head is not None and k_head.remaining_bytes <= COMPLETION_EPS_BYTES:
            kv_q.pop(0)
            if k_head.arrival_ms >= warmup_ms:
                kv_done.append((k_head.arrival_ms, t - k_head.arrival_ms, k_head.total_bytes / bw_bytes_per_ms))

        while ei < len(expert_arrivals) and expert_arrivals[ei] <= t + ARRIVAL_EPS_MS:
            expert_q.append(Job(expert_arrivals[ei], expert_job_bytes, expert_job_bytes))
            ei += 1
        while ki < len(kv_arrivals) and kv_arrivals[ki] <= t + ARRIVAL_EPS_MS:
            size = kv_job_sizes[ki]
            kv_q.append(Job(kv_arrivals[ki], size, size))
            ki += 1

    return {
        "expert": expert_done, "kv": kv_done,
        "bw_bytes_per_ms": bw_bytes_per_ms,
        "n_expert_arrivals": len(expert_arrivals), "n_kv_arrivals": len(kv_arrivals),
    }


# ---------- stats / pareto ----------

def _pctile_stats(records):
    """records: list of (arrival_ms, latency_ms, ideal_ms). Returns latency
    and inflation (latency - ideal) percentile stats, or NaNs if empty."""
    if not records:
        return {"n": 0, "latency_mean_ms": float("nan"), "latency_p50_ms": float("nan"),
                "latency_p95_ms": float("nan"), "inflation_mean_ms": float("nan"),
                "inflation_p50_ms": float("nan"), "inflation_p95_ms": float("nan")}
    lat = np.array([r[1] for r in records])
    ideal = np.array([r[2] for r in records])
    infl = lat - ideal
    return {
        "n": len(records),
        "latency_mean_ms": float(lat.mean()), "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "inflation_mean_ms": float(infl.mean()), "inflation_p50_ms": float(np.percentile(infl, 50)),
        "inflation_p95_ms": float(np.percentile(infl, 95)),
    }


def pareto_labels(points: list[tuple[float, float, str]]) -> list[str]:
    """points: (x, y, label), lower-is-better on both axes. Returns labels of
    the non-dominated (Pareto-optimal) points."""
    out = []
    for i, (x1, y1, l1) in enumerate(points):
        if any(j != i and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1)
               for j, (x2, y2, _l2) in enumerate(points)):
            continue
        out.append(l1)
    return out


# ---------- per-model setup (reuses E1's roofline primitives) ----------

def compute_expert_traffic_params(spec, b1_dir: Path, batch: int, residency_pct: float,
                                   precision: str, calib, n_trials: int, seed: int):
    """One periodic-job description of decode's expert-fetch demand: a job of
    `expert_job_bytes` (total non-resident expert bytes across ALL layers for
    one decode token) arriving every `expert_period_ms` (that token's total
    compute time across all layers). This coarsens E1's per-layer rho model
    into one job per token -- a deliberate simplification so the KV stream's
    much coarser timescale (new requests, not new layers) can be simulated
    jointly without an intractable event count; see the design doc."""
    e_union_mean, _ = rl.build_e_union_lookup(spec.tag, b1_dir, [batch], n_trials=n_trials, seed=seed)
    e_union_val = e_union_mean[batch]
    bytes_per_layer = rl.bytes_per_layer(spec, batch, residency_pct, precision, e_union_val)
    t_compute_layer_ms, provenance = rl.get_t_compute_ms(spec, batch, calib)
    return {
        "e_union_val": e_union_val,
        "expert_job_bytes": bytes_per_layer * spec.n_layers,
        "expert_period_ms": t_compute_layer_ms * spec.n_layers,
        "compute_provenance": provenance,
    }


def make_kv_job_sampler(spec, context_lens, weights, kv_precision):
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    def sampler(rng):
        c = rng.choice(context_lens, p=weights)
        return spec.kv_total_bytes(int(c), kv_precision)

    return sampler


# ---------- sweep ----------

def pooled_simulate(*, n_seeds: int, base_seed: int, **kwargs):
    """Runs simulate_link across `n_seeds` independent seeds and pools the
    raw completion records before computing any statistic. Necessary because
    expert arrivals are perfectly periodic while KV arrivals are Poisson: a
    single seed can land in a "phase lock" where the periodic stream happens
    to systematically dodge (or hit) the random one, producing single-run
    artifacts -- e.g. an EARLIER version of this sweep showed expert TPOT
    inflation at weighted:0.5 exceeding weighted:0.8 (giving KV MORE priority
    produced LESS expert inflation), which reversed once pooled across seeds.
    Pooling raw records (not averaging per-seed percentiles) is the
    statistically correct way to get a percentile estimate from multiple
    independent runs of the same underlying process."""
    pooled = {"expert": [], "kv": []}
    for i in range(n_seeds):
        res = simulate_link(seed=base_seed + i, **kwargs)
        pooled["expert"].extend(res["expert"])
        pooled["kv"].extend(res["kv"])
        pooled["bw_bytes_per_ms"] = res["bw_bytes_per_ms"]
    return pooled


def run_sweep(models, bw_grid, policies, *, batch, residency_pct, expert_precision,
              kv_precision, context_lens, context_weights, kv_rate_per_sec, sim_ms,
              seed, n_seeds, n_trials, calib, b1_overrides):
    rows = []
    isolated_rows = []
    per_model_traffic = {}

    for model_tag in models:
        spec = MODEL_SPECS[model_tag]
        b1_dir = Path(b1_overrides.get(model_tag, REPO_ROOT / spec.trace_dir_default))
        if not (b1_dir / "traces.jsonl.zst").exists():
            print(f"[e8] ERROR: no traces at {b1_dir}/traces.jsonl.zst", flush=True)
            sys.exit(1)
        traffic = compute_expert_traffic_params(spec, b1_dir, batch, residency_pct,
                                                 expert_precision, calib, n_trials, seed)
        per_model_traffic[model_tag] = traffic
        kv_sampler = make_kv_job_sampler(spec, context_lens, context_weights, kv_precision)
        print(f"[e8] {model_tag}: expert job = {traffic['expert_job_bytes']/1e6:.2f} MB every "
              f"{traffic['expert_period_ms']:.4f} ms ({traffic['compute_provenance']}); "
              f"KV bytes/token(fp16) = {spec.kv_bytes_per_token('fp16')/1e3:.2f} KB", flush=True)

        for bw in bw_grid:
            # Isolated baselines: same seed sequence so job-size draws are
            # comparable across policies; contention is what's being
            # measured, not sampling noise. Pooled across n_seeds -- see
            # pooled_simulate's docstring for why single-seed is unreliable.
            iso_expert = pooled_simulate(
                n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, policy="expert-first",
                expert_period_ms=traffic["expert_period_ms"], expert_job_bytes=traffic["expert_job_bytes"],
                kv_rate_per_sec=0.0, kv_job_sampler=kv_sampler, sim_ms=sim_ms)
            iso_kv = pooled_simulate(
                n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, policy="kv-first",
                expert_period_ms=0.0, expert_job_bytes=0.0,
                kv_rate_per_sec=kv_rate_per_sec, kv_job_sampler=kv_sampler, sim_ms=sim_ms)
            isolated_rows.append({
                "model": model_tag, "bw_gbps": bw,
                "expert_isolated": _pctile_stats(iso_expert["expert"]),
                "kv_isolated": _pctile_stats(iso_kv["kv"]),
            })

            for policy in policies:
                res = pooled_simulate(
                    n_seeds=n_seeds, base_seed=seed, bw_gbps=bw, policy=policy,
                    expert_period_ms=traffic["expert_period_ms"], expert_job_bytes=traffic["expert_job_bytes"],
                    kv_rate_per_sec=kv_rate_per_sec, kv_job_sampler=kv_sampler, sim_ms=sim_ms)
                e_stats = _pctile_stats(res["expert"])
                k_stats = _pctile_stats(res["kv"])
                _, kind, w_kv = make_rate_fn(policy)
                rows.append({
                    "model": model_tag, "bw_gbps": bw, "policy": policy, "policy_kind": kind,
                    "w_kv": w_kv,
                    "ttft_p50_ms": k_stats["latency_p50_ms"], "ttft_p95_ms": k_stats["latency_p95_ms"],
                    "ttft_mean_ms": k_stats["latency_mean_ms"], "n_kv_completions": k_stats["n"],
                    "tpot_inflation_p50_ms": e_stats["inflation_p50_ms"],
                    "tpot_inflation_p95_ms": e_stats["inflation_p95_ms"],
                    "tpot_inflation_mean_ms": e_stats["inflation_mean_ms"],
                    "n_expert_completions": e_stats["n"],
                })
    df = pd.DataFrame(rows)
    return df, isolated_rows, per_model_traffic


PARETO_ROUND_DECIMALS = 6  # microsecond precision on ms-scale metrics


def add_pareto_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rounds both metrics to microsecond precision before computing
    dominance. Without this, two policies that are both "effectively zero"
    TPOT inflation (e.g. 2e-13 ms vs 1.6e-12 ms -- sub-attosecond-scale
    floating-point residue from the event-driven arithmetic, not a real
    effect) can produce a spurious Pareto winner: whichever one happens to
    round to a smaller float wins purely on numerical noise, even though no
    real system could ever observe or exploit a ~1e-12 ms difference."""
    df = df.copy()
    df["pareto_optimal"] = False
    for (model, bw), sub in df.groupby(["model", "bw_gbps"]):
        pts = [(round(r.ttft_p50_ms, PARETO_ROUND_DECIMALS),
                round(r.tpot_inflation_p50_ms, PARETO_ROUND_DECIMALS), idx) for idx, r in sub.iterrows()]
        winners = set(pareto_labels(pts))
        df.loc[list(winners), "pareto_optimal"] = True
    return df


def recommend_policy(df: pd.DataFrame, ttft_slo_ms: float | None) -> dict:
    """Per (model,bw): report the Pareto-optimal set, and pick a single
    recommendation the same way E2 picks B* -- as the argmin of the metric
    you actually want to protect (TPOT inflation, i.e. decode throughput),
    SUBJECT TO an SLO constraint on the other metric (TTFT), rather than via
    an arbitrary scalarization of both into one score.

    Why not a scalarization (tried two, both broke): a product score
    (ttft*inflation) degenerates when inflation rounds to ~0 for several
    candidates, picking among them almost by floating-point noise instead of
    preferring the lowest TTFT. A pure lexicographic "minimize inflation
    first" is worse: `expert-first` is *structurally* guaranteed to hit the
    exact minimum possible inflation (0, by construction -- it never yields
    the link), so it always wins outright regardless of how catastrophic its
    TTFT gets, making the "recommendation" always the same policy independent
    of the data. An SLO constraint is the only one of the three that can
    actually flip its answer as the operating point changes.

    `ttft_slo_ms=None` (default): no SLO given, so there's no principled way
    to collapse the tradeoff to one point -- report the two Pareto extremes
    (best-TTFT and best-TPOT) instead of a single arbitrary "recommended"
    policy, exactly the honest-uncertainty stance E1/E2 take when a doc-level
    assumption isn't pinned down.
    """
    out = {}
    for (model, bw), sub in df.groupby(["model", "bw_gbps"]):
        pareto = sub[sub.pareto_optimal]
        if len(pareto) == 0:
            continue
        key = f"{model}_{bw}gbps"
        best_ttft = pareto.loc[pareto["ttft_p50_ms"].idxmin()]
        best_tpot = pareto.loc[pareto["tpot_inflation_p50_ms"].idxmin()]

        if ttft_slo_ms is None:
            out[key] = {
                "model": model, "bw_gbps": bw,
                "pareto_policies": pareto["policy"].tolist(),
                "recommended_policy": None,
                "note": "no --ttft-slo-ms given -- reporting the two Pareto extremes instead of "
                        "collapsing to one policy; pick based on your deployment's actual TTFT budget.",
                "best_for_ttft": {"policy": best_ttft["policy"], "ttft_p50_ms": float(best_ttft["ttft_p50_ms"]),
                                   "tpot_inflation_p50_ms": float(best_ttft["tpot_inflation_p50_ms"])},
                "best_for_tpot": {"policy": best_tpot["policy"], "ttft_p50_ms": float(best_tpot["ttft_p50_ms"]),
                                   "tpot_inflation_p50_ms": float(best_tpot["tpot_inflation_p50_ms"])},
            }
            continue

        feasible = pareto[pareto["ttft_p50_ms"] <= ttft_slo_ms]
        if len(feasible) == 0:
            out[key] = {
                "model": model, "bw_gbps": bw,
                "pareto_policies": pareto["policy"].tolist(),
                "recommended_policy": best_ttft["policy"],
                "recommended_ttft_p50_ms": float(best_ttft["ttft_p50_ms"]),
                "recommended_tpot_inflation_p50_ms": float(best_ttft["tpot_inflation_p50_ms"]),
                "note": f"NO policy meets the {ttft_slo_ms} ms TTFT SLO -- falling back to the "
                        f"best-achievable-TTFT policy instead (still {best_ttft['ttft_p50_ms']:.1f} ms).",
            }
        else:
            best = feasible.loc[feasible["tpot_inflation_p50_ms"].idxmin()]
            out[key] = {
                "model": model, "bw_gbps": bw,
                "pareto_policies": pareto["policy"].tolist(),
                "recommended_policy": best["policy"],
                "recommended_ttft_p50_ms": float(best["ttft_p50_ms"]),
                "recommended_tpot_inflation_p50_ms": float(best["tpot_inflation_p50_ms"]),
                "note": f"argmin TPOT inflation among policies meeting the {ttft_slo_ms} ms TTFT SLO.",
            }
    return out


# ---------- plotting ----------

def plot_contention(df: pd.DataFrame, isolated_rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    iso_by_key = {(r["model"], r["bw_gbps"]): r for r in isolated_rows}

    for model in sorted(df["model"].unique()):
        bws = sorted(df[df.model == model]["bw_gbps"].unique())
        fig, axes = plt.subplots(1, len(bws), figsize=(4.5 * len(bws), 4.5), squeeze=False)
        for ax, bw in zip(axes[0], bws):
            sub = df[(df.model == model) & (df.bw_gbps == bw)]
            for _, row in sub.iterrows():
                marker = "*" if row["pareto_optimal"] else "o"
                size = 160 if row["pareto_optimal"] else 60
                ax.scatter(row["ttft_p50_ms"], row["tpot_inflation_p50_ms"], marker=marker, s=size,
                           label=row["policy"])
                ax.annotate(row["policy"], (row["ttft_p50_ms"], row["tpot_inflation_p50_ms"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
            iso = iso_by_key.get((model, bw))
            if iso:
                ax.axvline(iso["kv_isolated"]["latency_p50_ms"], color="gray", linestyle=":",
                           label="KV isolated (no expert traffic)")
                ax.axhline(0.0, color="black", linestyle="-", linewidth=0.5)
            ax.set_xlabel("KV transfer latency p50 (ms) -- TTFT-sensitive")
            ax.set_ylabel("expert-fetch TPOT inflation p50 (ms/token)")
            ax.set_title(f"{model}, {bw} GB/s")
        fig.suptitle(f"{model}: link-contention Pareto frontier (* = Pareto-optimal)")
        fig.tight_layout()
        path = out_dir / f"kv_expert_contention_{model}.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="olmoe,mixtral")
    ap.add_argument("--b1-dirs", default="")
    ap.add_argument("--calib", default="", help="optional calib_compute.json from E1's measure_compute.py")
    ap.add_argument("--bw-grid", default="32,64", help=f"comma list of GB/s; full grid is {LINK_BW_GBPS_GRID}")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--batch", type=int, default=32,
                     help="decode batch for the expert traffic model. Default 32 matches E1's own "
                          "'router-guided prefetch is worth something' headline operating point "
                          "(batch>=32) -- see --residency-pct/--expert-precision for why.")
    ap.add_argument("--residency-pct", type=float, default=75.0,
                     help="Default 75%% matches E1's headline viable-regime residency for both models.")
    ap.add_argument("--expert-precision", default="nf4", choices=["fp16", "nf4"],
                     help="Default nf4: at batch=32/residency=75%%, E1 found nf4 reaches rho<1 for BOTH "
                          "models within this script's default --bw-grid (olmoe needs >=32GB/s, mixtral "
                          ">=16GB/s); fp16 needs >=128GB/s for olmoe, which would make expert traffic "
                          "alone already link-saturated (unbounded queue growth) at this script's default "
                          "bandwidths -- an uninteresting, degenerate co-tenancy scenario. Pass --expert-"
                          "precision fp16 deliberately if you want to see that saturated-regime case.")
    ap.add_argument("--kv-precision", default="fp16", choices=KV_PRECISION_GRID)
    ap.add_argument("--context-lens", default=",".join(str(c) for c in CONTEXT_LEN_GRID_E8))
    ap.add_argument("--context-weights", default="", help="comma list matching --context-lens; default uniform")
    ap.add_argument("--kv-rate-per-sec", type=float, default=2.0,
                     help="Poisson arrival rate of new long-context requests (ASSUMPTION -- "
                          "Experiments.md doesn't pin a number; see design doc)")
    ap.add_argument("--sim-seconds", type=float, default=30.0)
    ap.add_argument("--n-seeds", type=int, default=5,
                     help="run each (model,bw,policy) point across this many seeds and pool the raw "
                          "completion records before computing percentiles -- smooths out phase-lock "
                          "artifacts between the periodic expert stream and Poisson KV arrivals "
                          "(see pooled_simulate's docstring). Cheap: total sim cost scales linearly.")
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--ttft-slo-ms", type=float, default=None,
                     help="optional TTFT p50 SLO (ms); if given, recommends the min-TPOT-inflation "
                          "policy among those meeting it (mirrors E2's --slo-ms pattern). Unset = report "
                          "both Pareto extremes instead of one arbitrary pick.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="experiments/e8_kv_cotenancy/out")
    args = ap.parse_args()

    models = args.models.split(",")
    bw_grid = [float(x) for x in args.bw_grid.split(",")]
    policies = args.policies.split(",")
    context_lens = [int(x) for x in args.context_lens.split(",")]
    context_weights = ([float(x) for x in args.context_weights.split(",")] if args.context_weights
                        else [1.0] * len(context_lens))

    b1_overrides = {}
    for pair in args.b1_dirs.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            b1_overrides[k] = v

    calib = None
    if args.calib and Path(args.calib).exists():
        calib = json.loads(Path(args.calib).read_text())
        print(f"[e8] loaded compute calibration from {args.calib}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[e8] sweeping models={models} bw_grid={bw_grid} policies={policies} "
          f"sim_seconds={args.sim_seconds}...", flush=True)
    df, isolated_rows, per_model_traffic = run_sweep(
        models, bw_grid, policies, batch=args.batch, residency_pct=args.residency_pct,
        expert_precision=args.expert_precision, kv_precision=args.kv_precision,
        context_lens=context_lens, context_weights=context_weights,
        kv_rate_per_sec=args.kv_rate_per_sec, sim_ms=args.sim_seconds * 1000.0,
        seed=args.seed, n_seeds=args.n_seeds, n_trials=args.n_trials, calib=calib,
        b1_overrides=b1_overrides,
    )
    df = add_pareto_column(df)
    df.to_csv(out_dir / "kv_contention_sweep.csv", index=False)

    recs = recommend_policy(df, args.ttft_slo_ms)
    for k, v in recs.items():
        if v["recommended_policy"] is None:
            print(f"[e8] {k}: no SLO given -- best-TTFT={v['best_for_ttft']['policy']} "
                  f"({v['best_for_ttft']['ttft_p50_ms']:.1f}ms), "
                  f"best-TPOT={v['best_for_tpot']['policy']} "
                  f"({v['best_for_tpot']['tpot_inflation_p50_ms']:.5f}ms/tok)", flush=True)
        else:
            print(f"[e8] RECOMMENDATION {k}: {v['recommended_policy']} "
                  f"(TTFT p50={v['recommended_ttft_p50_ms']:.3f}ms, "
                  f"TPOT inflation p50={v['recommended_tpot_inflation_p50_ms']:.5f}ms/token) -- {v['note']}",
                  flush=True)

    print("[e8] plotting Pareto scatter (kv_expert_contention.pdf)...", flush=True)
    fig_paths = plot_contention(df, isolated_rows, out_dir / "figures")

    kv_formula_check = {
        model_tag: {
            "kv_bytes_per_token_fp16": MODEL_SPECS[model_tag].kv_bytes_per_token("fp16"),
            "kv_bytes_per_token_fp8": MODEL_SPECS[model_tag].kv_bytes_per_token("fp8"),
            "kv_total_bytes_by_context_len_fp16": {
                c: MODEL_SPECS[model_tag].kv_total_bytes(c, "fp16") for c in context_lens
            },
        }
        for model_tag in models
    }

    summary = {
        "models": models, "bw_grid": bw_grid, "policies": policies,
        "batch": args.batch, "residency_pct": args.residency_pct,
        "expert_precision": args.expert_precision, "kv_precision": args.kv_precision,
        "context_lens": context_lens, "context_weights": context_weights,
        "kv_rate_per_sec": args.kv_rate_per_sec, "sim_seconds": args.sim_seconds,
        "seed": args.seed, "n_seeds": args.n_seeds, "ttft_slo_ms": args.ttft_slo_ms,
        "per_model_traffic": per_model_traffic,
        "kv_formula_check": kv_formula_check,
        "sweep": df.to_dict(orient="records"),
        "isolated_baselines": isolated_rows,
        "recommendations": recs,
        "figures": fig_paths,
    }
    summary_path = out_dir / "kv_contention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[e8] wrote {summary_path}", flush=True)

    print("[e8] rendering RESULTS.md...", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_results_md import render  # noqa: E402
    render(summary_path, Path(__file__).resolve().parent / "RESULTS.md")


if __name__ == "__main__":
    main()
