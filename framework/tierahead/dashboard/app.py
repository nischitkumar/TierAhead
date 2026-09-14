"""[7] TierAhead interactive dashboard (Main.md sections 4.7 / 7.3's demo
script). Four tabs, matching the E13 dashboard spec exactly:

  1. Routing explorer  -- animated expert-activation heatmap over token time,
                           from a REAL committed trace.
  2. Roofline explorer -- sliders for expert size / batch / residency / link
                           BW; the rho=1 boundary is drawn live and the
                           current operating point lights up red/green. This
                           is the demo centerpiece per Main.md 7.4.
  3. Policy explorer   -- live policy bake-off (tierahead.sim.policies) on the
                           selected model/operating point: TPOT / stall-rate
                           / bytes-per-token per policy.
  4. Fleet & TCO       -- pooling curves with a cost-ratio slider.

Run with `tierahead dashboard` (wraps `streamlit run` on this file) or
directly: `streamlit run tierahead/dashboard/app.py`. Needs the `dashboard`
extra (`pip install tierahead[dashboard]`). No GPU required -- every number on
every tab is computed from already-collected traces or the analytical/TCO
models, all CPU-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tierahead.hw import probe
from tierahead.roofline.model import bytes_per_layer, get_t_compute_ms, rho as rho_fn, run_roofline
from tierahead.specs import MODEL_SPECS
from tierahead.specs.tiers import LINK_BW_GBPS_GRID
from tierahead.tco.pooling import simulate_fleet_demand, stranding_analysis, tco_crossover
from tierahead.traces.io import find_data_root, load_decode_df

st.set_page_config(page_title="TierAhead Explorer", layout="wide")

DATA_ROOT = find_data_root()
CAPS = probe()


@st.cache_data
def _load_decode(model_tag: str):
    spec = MODEL_SPECS[model_tag]
    return load_decode_df(DATA_ROOT / spec.trace_dir_default)


@st.cache_data
def _run_roofline_cached(model_tag: str):
    return run_roofline(models=[model_tag], data_root=DATA_ROOT, n_trials=200)


@st.cache_data
def _run_bakeoff_cached(model_tag: str, residency_pct: float, bw_gbps: float, precision: str, depth: int,
                          concurrency: int):
    from tierahead.sim.policies import run_policy_bakeoff
    spec = MODEL_SPECS[model_tag]
    return run_policy_bakeoff(model_tag, DATA_ROOT / spec.trace_dir_default, residency_pct=residency_pct,
                                bw_gbps=bw_gbps, precision=precision, depth=depth, concurrency=concurrency,
                                policies=["none", "static-c", "popularity-prefetch", "lru", "hybrid-lru-prefetch",
                                          "prefetch-freq", "prefetch-v2", "prefetch-v2+precision-gate", "oracle"])


@st.cache_data
def _run_concurrency_sweep_cached(model_tag: str, residency_pct: float, bw_gbps: float, precision: str,
                                    depth: int, concurrency_grid: tuple):
    from tierahead.sim.policies import run_concurrency_sweep
    spec = MODEL_SPECS[model_tag]
    return run_concurrency_sweep(model_tag, DATA_ROOT / spec.trace_dir_default, list(concurrency_grid),
                                   residency_pct=residency_pct, bw_gbps=bw_gbps, precision=precision, depth=depth,
                                   policies=["static-c", "lru", "hybrid-lru-prefetch",
                                             "prefetch-v2+precision-gate", "oracle"])


st.title("TierAhead -- Router-Guided Memory Tiering for HBM + CXL MoE Inference")
st.caption(
    f"Backend capability: {'CUDA' if CAPS.cuda_available else 'CPU/analytical (no CUDA on this machine)'} · "
    f"data root: `{DATA_ROOT}`"
)

tab1, tab2, tab3, tab4 = st.tabs(["Routing explorer", "Roofline explorer", "Policy explorer", "Fleet & TCO"])

with tab1:
    st.subheader("Expert activation heatmap over token time (real trace)")
    model_tag = st.selectbox("Model", list(MODEL_SPECS.keys())[:2], key="routing_model")
    decode_df = _load_decode(model_tag)
    reqs = sorted(decode_df["req"].unique())
    req = st.selectbox("Request", reqs[:50], key="routing_req")
    sub = decode_df[decode_df.req == req].sort_values(["layer", "tok"])
    n_layers = int(sub["layer"].max()) + 1
    n_toks = int(sub["tok"].max()) + 1
    spec = MODEL_SPECS[model_tag]
    grid = np.zeros((n_layers, spec.n_experts))
    for _, row in sub.iterrows():
        for e in row["topk"]:
            grid[int(row["layer"]), int(e)] += 1
    fig = go.Figure(data=go.Heatmap(z=grid, colorscale="Viridis"))
    fig.update_layout(xaxis_title="expert id", yaxis_title="layer", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{model_tag}: {spec.n_experts} experts/layer, top-{spec.k}. Notice how few experts light up "
        f"for the fine-grained model vs. the coarse one -- exactly the skew-inversion finding "
        f"(Gini {'0.378' if model_tag=='olmoe' else '0.066'}) that motivates fixed-residency placement."
    )

with tab2:
    st.subheader("The MoE Tiering Roofline -- rho = t_transfer / t_compute")
    col1, col2 = st.columns([1, 2])
    with col1:
        model_tag2 = st.selectbox("Model", list(MODEL_SPECS.keys())[:2], key="roofline_model")
        precision2 = st.selectbox("Precision", ["fp16", "nf4"], key="roofline_precision")
        batch2 = st.select_slider("Batch", options=[1, 8, 32], value=1, key="roofline_batch")
        residency2 = st.slider("Residency %", 0.0, 75.0, 25.0, step=12.5, key="roofline_residency")
        bw2 = st.select_slider("Link BW (GB/s)", options=LINK_BW_GBPS_GRID, value=32, key="roofline_bw")

    result = _run_roofline_cached(model_tag2)
    df = result["sweep_df"]
    df = df[(df.precision == precision2)]
    spec2 = MODEL_SPECS[model_tag2]

    with col2:
        pivot = df[df.batch == batch2].pivot(index="residency_pct", columns="bw_gbps", values="rho")
        fig2 = go.Figure(data=go.Contour(
            z=np.log10(np.clip(pivot.to_numpy(), 1e-3, 1e3)), x=pivot.columns, y=pivot.index,
            contours=dict(start=-1, end=1, size=0.2), colorscale="RdBu_r", reversescale=True,
        ))
        current = df[(df.batch == batch2) & (df.residency_pct == residency2) & (df.bw_gbps == bw2)]
        cur_rho = float(current["rho"].iloc[0]) if len(current) else float("nan")
        color = "lime" if cur_rho < 1 else "red"
        fig2.add_trace(go.Scatter(x=[bw2], y=[residency2], mode="markers",
                                   marker=dict(color=color, size=18, line=dict(color="black", width=2)),
                                   name="current operating point"))
        fig2.update_layout(xaxis_title="link BW (GB/s)", yaxis_title="residency (%)", height=450,
                            title=f"log10(rho) -- {'LATENCY-BOUND (prefetch helps)' if cur_rho < 1 else 'BANDWIDTH-BOUND (byte reduction needed)'}")
        st.plotly_chart(fig2, use_container_width=True)

    st.metric("rho at this operating point", f"{cur_rho:.3g}",
               delta="latency-bound" if cur_rho < 1 else "bandwidth-bound", delta_color="normal" if cur_rho < 1 else "inverse")
    for w in result["calib_sanity_warnings"].get(model_tag2, []):
        st.warning(f"Calibration sanity warning: {w}")

with tab3:
    st.subheader("Policy bake-off (trace-driven, held-out test split)")
    model_tag3 = st.selectbox("Model", list(MODEL_SPECS.keys())[:2], key="policy_model")
    residency3 = st.slider("Residency %", 0.0, 75.0, 25.0, step=12.5, key="policy_residency")
    bw3 = st.select_slider("Link BW (GB/s)", options=LINK_BW_GBPS_GRID, value=32, key="policy_bw")
    precision3 = st.selectbox("Precision", ["fp16", "nf4"], key="policy_precision")
    depth3 = st.select_slider("Lookahead depth d", options=[1, 2, 4, 8], value=1, key="policy_depth")
    concurrency3 = st.select_slider("Concurrent decode streams", options=[1, 2, 4, 8, 16, 32], value=1,
                                      key="policy_concurrency")

    bakeoff_df = _run_bakeoff_cached(model_tag3, residency3, bw3, precision3, depth3, concurrency3)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig3 = go.Figure(go.Bar(x=bakeoff_df["policy"], y=bakeoff_df["tpot_ms_mean"]))
        fig3.update_layout(title="TPOT mean (ms)", height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        fig4 = go.Figure(go.Bar(x=bakeoff_df["policy"], y=bakeoff_df["demand_stall_rate"] * 100))
        fig4.update_layout(title="Demand-stall rate (%)", height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)
    with c3:
        fig5 = go.Figure(go.Bar(x=bakeoff_df["policy"], y=bakeoff_df["bytes_per_token_mean"] / 1e6))
        fig5.update_layout(title="Bytes/token (MB)", height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig5, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        prec_df = bakeoff_df.dropna(subset=["prefetch_precision"])
        if len(prec_df):
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(x=prec_df["policy"], y=prec_df["prefetch_precision"], name="precision"))
            fig6.add_trace(go.Bar(x=prec_df["policy"], y=prec_df["prefetch_recall"], name="recall"))
            fig6.update_layout(title="Prefetch precision vs recall (high recall isn't free)", barmode="group",
                                height=350, xaxis_tickangle=-30)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No prefetch-eligible policy in this run reported precision/recall.")
    with c5:
        link_bound = [row["policy"] for _, row in bakeoff_df.iterrows()
                      if isinstance(row.get("extra"), dict) and row["extra"].get("concurrency_stable") is False]
        if concurrency3 > 1:
            if link_bound:
                st.info(f"At concurrency={concurrency3}x, the LINK (not per-stream compute) is now the "
                         f"persistent bottleneck for: {', '.join(link_bound)}. The TPOT numbers shown are the "
                         f"theoretically-expected fair-share values for this regime, not an optimistic proxy -- "
                         f"see tierahead.sim.concurrency.")
            else:
                st.info(f"At concurrency={concurrency3}x, streams aren't yet persistently link-bound for any "
                         f"policy shown -- the TPOT numbers above are a conservative (worse-than-reality) "
                         f"estimate; real contention is plausibly milder.")

    st.dataframe(bakeoff_df[["policy", "tpot_ms_mean", "demand_stall_rate", "prefetch_precision", "prefetch_recall",
                              "bytes_per_token_mean", "oracle_gap_closed_pct"]])

    show_sweep = st.checkbox("Show concurrency sweep (TPOT vs concurrency per policy)", value=False,
                              key="policy_show_sweep")
    if show_sweep:
        sweep_df = _run_concurrency_sweep_cached(model_tag3, residency3, bw3, precision3, depth3, (1, 2, 4, 8))
        fig7 = go.Figure()
        for pol in sweep_df["policy"].unique():
            sub = sweep_df[sweep_df.policy == pol].sort_values("concurrency")
            fig7.add_trace(go.Scatter(x=sub["concurrency"], y=sub["tpot_ms_mean"], mode="lines+markers", name=pol))
        fig7.update_layout(title="TPOT vs concurrency per policy (log-log)", xaxis_title="concurrency",
                            yaxis_title="TPOT mean (ms)", xaxis_type="log", yaxis_type="log", height=420)
        st.plotly_chart(fig7, use_container_width=True)
        st.caption(
            "Byte-budget-limited conditional prefetch (prefetch-v2+precision-gate) converges toward "
            "static-c's line as concurrency grows -- the lookahead byte budget vanishes once enough "
            "streams divide the link (same m_affordable-hits-zero mechanism as the low-bandwidth "
            "collapse, just triggered by concurrency instead). hybrid-lru-prefetch stays near-flat "
            "because its LRU-residency component spends no bandwidth at all, so it doesn't need a "
            "byte budget to keep working. oracle is exactly flat by construction (validated by "
            "tierahead.sim.validation.gate_oracle_invariant_to_concurrency). See PREDICTED.md Sec 2.5 "
            "for the full cross-model writeup."
        )

with tab4:
    st.subheader("Fleet pooling & TCO")
    n_hosts = st.slider("Fleet size (hosts)", 8, 256, 64, key="tco_hosts")
    cv = st.slider("Demand coefficient of variation", 0.1, 0.8, 0.35, key="tco_cv")
    group_size = st.select_slider("CXL pool group size", options=[1, 2, 4, 8, 16, 32], value=8, key="tco_group")
    demands = simulate_fleet_demand(n_hosts, mean_gb=200.0, cv=cv, seed=0)
    stranding_rows = stranding_analysis(demands, sorted({1, group_size, n_hosts}), temporal_cv=cv)
    st.dataframe(pd.DataFrame(stranding_rows))

    st.markdown("**Tokens/sec/$ crossover vs HBM:DDR5 cost ratio**")
    crossover = tco_crossover(1000.0, 850.0, 80.0, 20.0, 90.0)
    cdf = pd.DataFrame(crossover)
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=cdf["hbm_ddr5_cost_ratio"], y=cdf["tokens_per_sec_per_dollar_hbm_only"], name="HBM-only"))
    fig6.add_trace(go.Scatter(x=cdf["hbm_ddr5_cost_ratio"], y=cdf["tokens_per_sec_per_dollar_hbm_plus_cxl"], name="HBM+CXL"))
    fig6.update_layout(xaxis_title="HBM:DDR5 cost ratio", yaxis_title="tokens/sec/$", height=400)
    st.plotly_chart(fig6, use_container_width=True)
