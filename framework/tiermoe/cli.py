"""Unified `tiermoe` CLI. `tiermoe <subcommand> --help` for details on any one.

Subcommands map directly onto Main.md's eight framework components:
  doctor     -- tiermoe.hw capability report (what runs here, what needs a GPU/Linux box)
  roofline   -- [0] regime sweep
  analyze    -- [2] workload characterization
  sim        -- [5] policy bake-off (none/lru/static-c/prefetch-freq/prefetch-v2/+gate/oracle)
  kv         -- [8] KV/expert link co-tenancy sweep
  tco        -- [6] pooling & TCO model
  baseline   -- the flagship "normal (HBM-only) vs CXL-tiered" comparison
  dashboard  -- [7] launch the Streamlit explorer
  validate   -- run the E6 validation gates against a trace directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common_trace_args(p: argparse.ArgumentParser):
    p.add_argument("--model", default="olmoe", choices=["olmoe", "mixtral"])
    p.add_argument("--data-root", default=None, help="repo root holding results/ and experiments/ (default: auto-detected)")
    p.add_argument("--calib", default=None, help="optional calib_compute.json path")


def cmd_doctor(_args):
    from tiermoe.hw import probe
    caps = probe()
    print(caps.describe())
    for note in caps.notes:
        print(f"\nnote: {note}")


def cmd_roofline(args):
    from tiermoe.roofline.model import run_roofline
    data_root = Path(args.data_root) if args.data_root else None
    calib_path = Path(args.calib) if args.calib else None
    result = run_roofline(models=[args.model], calib_path=calib_path, data_root=data_root)
    for k, v in result["calib_sanity_warnings"].items():
        print(f"WARNING calib sanity ({k}):", file=sys.stderr)
        for w in v:
            print(f"  - {w}", file=sys.stderr)
    for k, v in result["headline_sentences"].items():
        print(f"[{k}] {v}")
    if args.out:
        result["sweep_df"].to_csv(args.out, index=False)
        print(f"wrote sweep table to {args.out}")


def cmd_analyze(args):
    from tiermoe.analyze.workload import run_characterization
    from tiermoe.traces.io import resolve_trace_dir
    data_root = Path(args.data_root) if args.data_root else None
    from tiermoe.specs import MODEL_SPECS
    trace_dir = resolve_trace_dir(MODEL_SPECS[args.model].trace_dir_default, data_root)
    result = run_characterization(trace_dir, seed=args.seed)
    print(json.dumps(result["headline"], indent=2))
    print(json.dumps(result["verdict"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote full report to {args.out}")


def cmd_sim(args):
    from tiermoe.sim.policies import run_concurrency_sweep, run_policy_bakeoff
    from tiermoe.traces.io import resolve_trace_dir
    from tiermoe.specs import MODEL_SPECS
    data_root = Path(args.data_root) if args.data_root else None
    trace_dir = resolve_trace_dir(MODEL_SPECS[args.model].trace_dir_default, data_root)
    calib = json.loads(Path(args.calib).read_text()) if args.calib and Path(args.calib).exists() else None
    if args.concurrency_grid:
        grid = [int(x) for x in args.concurrency_grid.split(",")]
        df = run_concurrency_sweep(args.model, trace_dir, grid, residency_pct=args.residency_pct,
                                     bw_gbps=args.bw_gbps, precision=args.precision, depth=args.depth,
                                     calib=calib, seed=args.seed)
    else:
        df = run_policy_bakeoff(args.model, trace_dir, residency_pct=args.residency_pct, bw_gbps=args.bw_gbps,
                                  precision=args.precision, calib=calib, added_latency_ns=args.added_latency_ns,
                                  depth=args.depth, m_mult=args.m_mult, concurrency=args.concurrency, seed=args.seed)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


def cmd_validate(args):
    from tiermoe.sim.validation import run_validation_suite
    from tiermoe.traces.io import resolve_trace_dir
    from tiermoe.specs import MODEL_SPECS
    data_root = Path(args.data_root) if args.data_root else None
    trace_dir = resolve_trace_dir(MODEL_SPECS[args.model].trace_dir_default, data_root)
    result = run_validation_suite(args.model, trace_dir, seed=args.seed)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["all_passed"] else 1)


def cmd_kv(args):
    from tiermoe.kv.cotenancy import run_kv_expert_sweep, recommend_policy
    data_root = Path(args.data_root) if args.data_root else None
    df = run_kv_expert_sweep(args.models.split(","), [float(x) for x in args.bw_grid.split(",")],
                               args.policies.split(","), batch=args.batch, residency_pct=args.residency_pct,
                               expert_precision=args.expert_precision, kv_rate_per_sec=args.kv_rate_per_sec,
                               sim_seconds=args.sim_seconds, seed=args.seed, data_root=data_root)
    print(df.to_string(index=False))
    recs = recommend_policy(df, args.ttft_slo_ms)
    print(json.dumps(recs, indent=2, default=str))
    if args.out:
        df.to_csv(args.out, index=False)


def cmd_tco(args):
    from tiermoe.tco.pooling import simulate_fleet_demand, stranding_analysis, tco_crossover
    demands = simulate_fleet_demand(args.n_hosts, args.mean_gb, cv=args.cv, seed=args.seed)
    stranding = stranding_analysis(demands, [int(x) for x in args.group_sizes.split(",")], temporal_cv=args.cv)
    print(json.dumps(stranding, indent=2, default=str))
    crossover = tco_crossover(args.tokens_per_sec_hbm_only, args.tokens_per_sec_hybrid, args.gb_hbm_only,
                                args.gb_hbm_hybrid, args.gb_cxl_hybrid)
    print(json.dumps(crossover, indent=2, default=str))


def cmd_baseline(args):
    from tiermoe.baseline.compare import run_baseline_vs_cxl, run_flagship_report
    data_root = Path(args.data_root) if args.data_root else None
    calib = json.loads(Path(args.calib).read_text()) if args.calib and Path(args.calib).exists() else None
    if args.concurrency_grid:
        grid = [int(x) for x in args.concurrency_grid.split(",")]
        result = run_flagship_report(args.model, residency_pct=args.residency_pct, bw_gbps=args.bw_gbps,
                                       precision=args.precision, calib=calib, backend=args.backend,
                                       depth=args.depth, m_mult=args.m_mult, concurrency_grid=grid,
                                       concurrent_requests=args.concurrent_requests, context_len=args.context_len,
                                       kv_precision=args.kv_precision,
                                       hbm_budget_gb=args.hbm_budget_gb, seed=args.seed, data_root=data_root)
        if result["kv_pressure"] is not None:
            print(json.dumps(result["kv_pressure"], indent=2, default=str))
        for c, headline in result["headlines"].items():
            print(f"[concurrency={c}] {headline}")
        print(result["combined_policies_df"].to_string(index=False))
        if args.out:
            result["combined_policies_df"].to_csv(args.out, index=False)
        return
    result = run_baseline_vs_cxl(args.model, residency_pct=args.residency_pct, bw_gbps=args.bw_gbps,
                                   precision=args.precision, calib=calib, backend=args.backend,
                                   depth=args.depth, m_mult=args.m_mult, concurrency=args.concurrency,
                                   concurrent_requests=args.concurrent_requests, context_len=args.context_len,
                                   kv_precision=args.kv_precision,
                                   hbm_budget_gb=args.hbm_budget_gb, seed=args.seed, data_root=data_root)
    print(result["headline"])
    if result["kv_pressure"] is not None:
        print(json.dumps(result["kv_pressure"], indent=2, default=str))
    print(result["policies_df"].to_string(index=False))
    if args.out:
        result["policies_df"].to_csv(args.out, index=False)


def cmd_report(args):
    from tiermoe.analyze.workload import run_characterization
    from tiermoe.report.markdown import df_to_markdown_table, render_report_md
    from tiermoe.roofline.model import run_roofline
    from tiermoe.sim.policies import run_policy_bakeoff
    from tiermoe.specs import MODEL_SPECS
    from tiermoe.traces.io import resolve_trace_dir

    data_root = Path(args.data_root) if args.data_root else None
    trace_dir = resolve_trace_dir(MODEL_SPECS[args.model].trace_dir_default, data_root)

    char = run_characterization(trace_dir, seed=args.seed)
    char_body = (
        f"- Coverage@25% (deployable): **{char['headline']['coverage_25pct_deployable']:.3f}**\n"
        f"- Recall@2k, d=1, non-resident: **{char['headline']['recall_2k_d1_nonresident']:.3f}**\n"
        f"- HTB(C=25%, m=2k): **{char['headline']['htb_c25_m2k']:.3f}**\n"
        f"- Verdict: **{char['verdict']['label']}** -- {char['verdict']['reason']}\n"
    )

    roof = run_roofline(models=[args.model], data_root=data_root)
    roof_body = "\n".join(f"- {v}" for v in roof["headline_sentences"].values())
    for w in roof["calib_sanity_warnings"].get(args.model, []):
        roof_body += f"\n- **WARNING (calibration sanity):** {w}"

    sim_df = run_policy_bakeoff(args.model, trace_dir, residency_pct=args.residency_pct, bw_gbps=args.bw_gbps,
                                  precision=args.precision, depth=args.depth, m_mult=args.m_mult, seed=args.seed)
    sim_body = (
        f"Operating point: residency={args.residency_pct}%, bw={args.bw_gbps}GB/s, "
        f"precision={args.precision}, depth={args.depth}.\n\n"
        + df_to_markdown_table(sim_df[["policy", "tpot_ms_mean", "demand_stall_rate", "prefetch_precision",
                                        "prefetch_recall", "bytes_per_token_mean", "oracle_gap_closed_pct"]])
    )

    out_path = Path(args.out or f"{args.model}_report.md")
    render_report_md(
        title=f"tierMoE report: {args.model}",
        sections=[("Characterization (MEASURED)", char_body), ("Roofline (SIMULATED, flop_estimate)", roof_body),
                  ("Policy bake-off (SIMULATED)", sim_body)],
        out_path=out_path,
        default_analysis_lines=["- [ ] Does this match PREDICTED.md's Verified/Simulated sections?",
                                  "- [ ] Any calibration sanity warnings above that need a real GPU re-run?"],
    )
    print(f"wrote {out_path}")


def cmd_dashboard(args):
    import subprocess
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true" if args.headless else "false"]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="tiermoe", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="hardware capability report")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("roofline", help="[0] regime sweep")
    _add_common_trace_args(p)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_roofline)

    p = sub.add_parser("analyze", help="[2] workload characterization")
    _add_common_trace_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("sim", help="[5] policy bake-off")
    _add_common_trace_args(p)
    p.add_argument("--residency-pct", type=float, default=25.0)
    p.add_argument("--bw-gbps", type=float, default=32.0)
    p.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    p.add_argument("--added-latency-ns", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=1, help="prefetch lookahead depth d")
    p.add_argument("--m-mult", type=int, default=2, help="candidate-set size multiplier (m = m_mult * k)")
    p.add_argument("--concurrency", type=int, default=1,
                    help="N concurrent decode streams sharing the link (fair-share proxy -- see "
                         "tiermoe.sim.concurrency for its validated limits)")
    p.add_argument("--concurrency-grid", default=None,
                    help="comma list, e.g. 1,2,4,8 -- if given, sweeps concurrency instead of a single run "
                         "(ignores --concurrency)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("validate", help="run E6 validation gates")
    _add_common_trace_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("kv", help="[8] KV/expert co-tenancy sweep")
    p.add_argument("--models", default="olmoe,mixtral")
    p.add_argument("--data-root", default=None)
    p.add_argument("--bw-grid", default="32,64")
    p.add_argument("--policies", default="expert-first,kv-first,weighted:0.5")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--residency-pct", type=float, default=75.0)
    p.add_argument("--expert-precision", default="nf4", choices=["fp16", "nf4"])
    p.add_argument("--kv-rate-per-sec", type=float, default=2.0)
    p.add_argument("--sim-seconds", type=float, default=30.0)
    p.add_argument("--ttft-slo-ms", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_kv)

    p = sub.add_parser("tco", help="[6] pooling & TCO model")
    p.add_argument("--n-hosts", type=int, default=100)
    p.add_argument("--mean-gb", type=float, default=200.0)
    p.add_argument("--cv", type=float, default=0.35)
    p.add_argument("--group-sizes", default="1,4,8,16")
    p.add_argument("--tokens-per-sec-hbm-only", type=float, default=1000.0)
    p.add_argument("--tokens-per-sec-hybrid", type=float, default=850.0)
    p.add_argument("--gb-hbm-only", type=float, default=80.0)
    p.add_argument("--gb-hbm-hybrid", type=float, default=20.0)
    p.add_argument("--gb-cxl-hybrid", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_tco)

    p = sub.add_parser("baseline", help="flagship: normal (HBM-only) vs CXL-tiered")
    _add_common_trace_args(p)
    p.add_argument("--residency-pct", type=float, default=25.0)
    p.add_argument("--bw-gbps", type=float, default=32.0)
    p.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    p.add_argument("--backend", default="auto", choices=["auto", "des", "analytical", "cxlmemsim"])
    p.add_argument("--depth", type=int, default=1, help="prefetch lookahead depth d")
    p.add_argument("--m-mult", type=int, default=2, help="candidate-set size multiplier (m = m_mult * k)")
    p.add_argument("--concurrency", type=int, default=1, help="N concurrent decode streams sharing the link")
    p.add_argument("--concurrency-grid", default=None,
                    help="comma list, e.g. 1,2,4,8 -- if given, runs the flagship comparison once per "
                         "concurrency level (tiermoe.baseline.compare.run_flagship_report) and stacks the "
                         "results instead of a single run (ignores --concurrency)")
    p.add_argument("--concurrent-requests", type=int, default=1,
                    help="concurrent long-context requests for KV-cache HBM pressure accounting")
    p.add_argument("--context-len", type=int, default=0,
                    help="tokens of KV-cache history per concurrent request (0 = skip KV pressure accounting "
                         "entirely, preserving the old single-request behavior)")
    p.add_argument("--kv-precision", default="fp16", choices=["fp16", "fp8"])
    p.add_argument("--hbm-budget-gb", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("report", help="render a combined Markdown report (characterization + roofline + bake-off)")
    _add_common_trace_args(p)
    p.add_argument("--residency-pct", type=float, default=25.0)
    p.add_argument("--bw-gbps", type=float, default=32.0)
    p.add_argument("--precision", default="fp16", choices=["fp16", "nf4"])
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--m-mult", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="[7] launch the Streamlit explorer")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=cmd_dashboard)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
