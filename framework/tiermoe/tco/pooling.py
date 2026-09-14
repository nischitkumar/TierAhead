"""[6] Pooling & TCO model (Main.md section 4.6 / Experiments.md E9) --
follows Pond's (Li et al., ASPLOS 2023, arXiv:2203.00241) stranding logic
applied to an MoE-serving fleet. Never implemented anywhere in this repo's
experiments/ folder (no e9_* directory exists) -- this module is that build.

HONESTY NOTE: this repo has no real multi-tenant fleet telemetry (no
production serving cluster was ever instrumented -- the pilot traced single-
request decode locality, not fleet-level memory demand). Every number this
module produces from ``simulate_fleet_demand`` is therefore explicitly a
MODELED/synthetic per-host demand distribution (log-normal, a standard choice
for resource-demand modeling), not a measurement -- exactly the same
"assumption, not measurement" honesty tag experiments/e8_kv_cotenancy already
applies to its own ``--kv-rate-per-sec``. Swap in real per-host memory
telemetry (`observed_demands_gb`) the moment it exists and every downstream
number (stranding %, tokens/sec/$) recomputes from real data with no code
change.
"""
from __future__ import annotations

import numpy as np


def simulate_fleet_demand(n_hosts: int, mean_gb: float, cv: float = 0.35, seed: int = 0) -> np.ndarray:
    """Per-host MEAN (time-averaged) memory demand -- log-normal across hosts
    (a standard, conservative choice: heavier right tail than Gaussian,
    always positive). `cv` here describes cross-host heterogeneity in the
    mean; ``stranding_analysis``'s own `temporal_cv` parameter separately
    describes each host's OWN demand variability over time, which is the
    quantity that actually drives provisioning (a host with a perfectly flat
    load needs no safety margin at all, regardless of how it compares to its
    neighbors). Neither number is measured -- see this module's docstring."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(mean_gb) - sigma ** 2 / 2
    return rng.lognormal(mean=mu, sigma=sigma, size=n_hosts)


def stranding_analysis(mean_demands_gb: np.ndarray, pool_group_sizes: list[int], temporal_cv: float = 0.35,
                        p: float = 99.0) -> list[dict]:
    """Pond's stranding argument, made precise with a standard portfolio-
    variance model: each host's demand over time has mean `mean_i` and
    std `mean_i * temporal_cv` (independent across hosts -- a modeling
    assumption, not a measurement). A host (or a pool of `group_size` hosts
    sharing one CXL-pooled budget) must provision for its p-th percentile of
    AGGREGATE demand: `provisioned = mean + z_p * std`, `z_p` the standard
    normal quantile (2.326 for p99, 1.645 for p95).

    For a group of g independent hosts, aggregate mean scales as g*mean_i
    but aggregate std scales only as sqrt(g)*mean_i*temporal_cv (variances,
    not stds, add for independent variables) -- so the SAFETY MARGIN per
    unit of mean demand shrinks as 1/sqrt(g). This is the actual mechanism
    behind Pond's stranding reduction: pooling doesn't change how much
    memory is USED on average, only how much extra headroom must be
    provisioned to cover uncorrelated peaks, and that headroom shrinks with
    the square root of pool size. group_size=1 recovers the unpooled case
    exactly by construction (sqrt(1)=1), so `stranding_reduction_pct` at
    group_size=1 is always 0% -- a built-in sanity check, not a coincidence.
    """
    z = 2.326 if p >= 99 else 1.645
    n_hosts = len(mean_demands_gb)
    total_mean = float(np.sum(mean_demands_gb))
    per_host_std = mean_demands_gb * temporal_cv
    unpooled_provisioned = float(np.sum(mean_demands_gb + z * per_host_std))

    rows = []
    for group_size in pool_group_sizes:
        if group_size > n_hosts or group_size < 1:
            continue
        pooled_total = 0.0
        for start in range(0, n_hosts, group_size):
            group_means = mean_demands_gb[start:start + group_size]
            group_stds = per_host_std[start:start + group_size]
            group_mean = float(group_means.sum())
            group_std = float(np.sqrt(np.sum(group_stds ** 2)))  # independent variances add
            pooled_total += group_mean + z * group_std
        stranded_unpooled = unpooled_provisioned - total_mean
        stranded_pooled = pooled_total - total_mean
        rows.append({
            "group_size": group_size, "unpooled_provisioned_gb": unpooled_provisioned,
            "pooled_provisioned_gb": pooled_total, "mean_used_gb": total_mean,
            "stranded_unpooled_gb": stranded_unpooled, "stranded_pooled_gb": stranded_pooled,
            "stranding_reduction_pct": 100.0 * (stranded_unpooled - stranded_pooled) / stranded_unpooled
            if stranded_unpooled > 0 else 0.0,
        })
    return rows


def tokens_per_sec_per_dollar(tokens_per_sec: float, gb_hbm: float, gb_cxl: float, cost_per_gb_hbm: float,
                                cost_per_gb_cxl: float) -> float:
    total_cost = gb_hbm * cost_per_gb_hbm + gb_cxl * cost_per_gb_cxl
    return tokens_per_sec / total_cost if total_cost > 0 else float("inf")


def tco_crossover(tokens_per_sec_hbm_only: float, tokens_per_sec_hbm_plus_cxl: float, gb_hbm_only: float,
                    gb_hbm_hybrid: float, gb_cxl_hybrid: float, cost_ratio_grid=(2.0, 3.0, 4.0, 5.0),
                    cost_per_gb_ddr5: float = 1.0) -> list[dict]:
    """Sweeps HBM:DDR5(CXL) cost ratio 2-5x (Main.md section 4.6: "sweep the
    ratio 2-5x so conclusions are robust to price uncertainty") and reports
    tokens/sec/$ for HBM-only vs HBM+CXL at each ratio, plus whether HBM+CXL
    wins at that ratio -- the "crossover" Main.md's exit criteria asks for.
    `cost_per_gb_ddr5` is a normalized unit (=1.0); HBM cost = ratio *
    cost_per_gb_ddr5; CXL device cost = DDR5 + controller amortization,
    modeled here as 1.15x DDR5 (a documented placeholder -- Main.md itself
    only says "DDR5 + controller amortization" without pinning a number;
    override via the `cxl_controller_markup` parameter once a real BOM
    estimate exists)."""
    rows = []
    cxl_controller_markup = 1.15
    for ratio in cost_ratio_grid:
        cost_hbm = ratio * cost_per_gb_ddr5
        cost_cxl = cost_per_gb_ddr5 * cxl_controller_markup
        tpspd_hbm_only = tokens_per_sec_per_dollar(tokens_per_sec_hbm_only, gb_hbm_only, 0.0, cost_hbm, cost_cxl)
        tpspd_hybrid = tokens_per_sec_per_dollar(tokens_per_sec_hbm_plus_cxl, gb_hbm_hybrid, gb_cxl_hybrid, cost_hbm, cost_cxl)
        rows.append({
            "hbm_ddr5_cost_ratio": ratio, "cost_per_gb_hbm": cost_hbm, "cost_per_gb_cxl": cost_cxl,
            "tokens_per_sec_per_dollar_hbm_only": tpspd_hbm_only,
            "tokens_per_sec_per_dollar_hbm_plus_cxl": tpspd_hybrid,
            "hbm_plus_cxl_wins": tpspd_hybrid > tpspd_hbm_only,
            "advantage_pct": 100.0 * (tpspd_hybrid - tpspd_hbm_only) / tpspd_hbm_only if tpspd_hbm_only > 0 else float("nan"),
        })
    return rows
