#!/usr/bin/env python3
"""E3 -- assembles the unified calibration report (Experiments.md E3
deliverable: "a methodology table in the report listing every simulator
parameter and whether it is measured here, measured in cited work, or
vendor-specified"). Pure Python, no GPU/network required -- this is the
analysis half of E3 that runs anywhere; measure_transfer.py,
e2e_offload_runner.py, and nf4_fidelity.py are the GPU halves that produce
the calib_*.json files this script reads.

Never fabricates a placeholder number: any calibration file that doesn't
exist yet (or is present but empty/corrupt/malformed) is reported as
MISSING with a clear instruction for how to produce it, not silently
skipped or filled in with a guess. This is the same discipline E1's
`compute_provenance` tagging already uses, applied to the full ladder.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_fit as tf  # noqa: E402
import policy_check as pc  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_TRANSFER_PATH = HERE / "out" / "calib_transfer.json"
DEFAULT_E2E_PATH = HERE / "out" / "calib_e2e.json"
DEFAULT_NF4_PATH = HERE / "out" / "calib_nf4_fidelity.json"
DEFAULT_COMPUTE_PATH = HERE.parent / "e1_roofline" / "out" / "calib_compute.json"


def _load_json_safe(path: Path):
    """Returns (data, status_suffix). data is None if the file doesn't exist
    or fails to parse -- callers must not assume a non-None return means
    "fully valid", only "valid JSON, worth trying to use"."""
    path = Path(path)
    if not path.exists():
        return None, f"MISSING (run the GPU script to produce {path})"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError:
        return None, f"MISSING (unreadable/corrupt JSON at {path} -- rerun the GPU script)"


def build_report(*, transfer_path: Path, e2e_path: Path, nf4_path: Path, compute_path: Path,
                  policy_tolerance_pct: float = 2.0) -> dict:
    provenance_table = []
    transfer_fit_result = None
    policy_ordering_check = None
    nf4_result = None

    # ---- compute (E1's calib_compute.json, one row per model found) ----
    compute_data, compute_err = _load_json_safe(compute_path)
    if compute_data is None:
        provenance_table.append({
            "parameter": "compute", "status": compute_err or "MISSING",
            "detail": "run e1_roofline/measure_compute.py, or point --compute-calib-path at its output",
        })
    else:
        for model_tag, entry in compute_data.items():
            prov = entry.get("provenance", "unknown")
            gpu = ", ".join(entry.get("gpu", [])) or "unknown GPU"
            provenance_table.append({
                "parameter": f"compute ({model_tag})",
                "status": f"{prov} (on {gpu})",
                "detail": f"n_layers={entry.get('n_layers')}, quant={entry.get('quant')}",
            })

    # ---- transfer (this experiment's own measure_transfer.py) ----
    transfer_data, transfer_err = _load_json_safe(transfer_path)
    if transfer_data is None:
        provenance_table.append({
            "parameter": "transfer (alpha/beta)", "status": transfer_err or "MISSING",
            "detail": "run measure_transfer.py",
        })
    else:
        try:
            fit = tf.fit_from_calib_json(transfer_data)
            transfer_fit_result = {
                "alpha_ms": fit.alpha_ms, "beta_bytes_per_ms": fit.beta_bytes_per_ms,
                "r_squared": fit.r_squared, "n_points": fit.n_points,
            }
            gpu = ", ".join(transfer_data.get("gpu", [])) or "unknown GPU"
            provenance_table.append({
                "parameter": "transfer (alpha/beta)",
                "status": f"measured_here (on {gpu})",
                "detail": f"alpha={fit.alpha_ms:.4f}ms, beta={fit.beta_bytes_per_ms/1e6:.1f} MB/ms "
                          f"(~{fit.beta_bytes_per_ms/1e3:.0f} GB/s asymptotic), R^2={fit.r_squared:.4f}",
            })
        except (ValueError, KeyError) as e:
            provenance_table.append({
                "parameter": "transfer (alpha/beta)",
                "status": f"MISSING (data present but fit failed: {e})",
                "detail": str(transfer_path),
            })

    # ---- e2e offload policy ordering (this experiment's e2e_offload_runner.py) ----
    e2e_data, e2e_err = _load_json_safe(e2e_path)
    if e2e_data is None:
        provenance_table.append({
            "parameter": "e2e offload policy ordering", "status": e2e_err or "MISSING",
            "detail": "run e2e_offload_runner.py",
        })
    else:
        policy_ordering_check = {}
        levels = e2e_data.get("residency_levels", {})
        for level, level_data in levels.items():
            tpot_by_policy = level_data.get("tpot_ms_by_policy", {})
            try:
                policy_ordering_check[level] = pc.check_policy_ordering(tpot_by_policy, policy_tolerance_pct)
            except KeyError as e:
                policy_ordering_check[level] = (False, f"FAIL: {e}")
        all_passed = all(v[0] for v in policy_ordering_check.values()) if policy_ordering_check else False
        provenance_table.append({
            "parameter": "e2e offload policy ordering",
            "status": f"measured_here ({'ALL PASS' if all_passed else 'SEE DETAIL'})",
            "detail": "; ".join(f"r={lvl}%: {msg}" for lvl, (_ok, msg) in policy_ordering_check.items()),
        })

    # ---- NF4 routing fidelity (this experiment's nf4_fidelity.py, OLMoE only) ----
    nf4_data, nf4_err = _load_json_safe(nf4_path)
    if nf4_data is None:
        provenance_table.append({
            "parameter": "nf4 routing fidelity (olmoe)", "status": nf4_err or "MISSING",
            "detail": "run nf4_fidelity.py",
        })
    else:
        nf4_result = nf4_data
        mean_j = nf4_data.get("mean_jaccard_similarity")
        n_prompts = nf4_data.get("n_prompts")
        provenance_table.append({
            "parameter": "nf4 routing fidelity (olmoe)",
            "status": f"measured_here (mean top-k agreement={mean_j:.3f} over {n_prompts} prompts)"
                      if mean_j is not None else "measured_here (see detail)",
            "detail": f"per-layer Jaccard overlap fp16 vs nf4 top-k, n_prompts={n_prompts}",
        })

    return {
        "provenance_table": provenance_table,
        "transfer_fit": transfer_fit_result,
        "policy_ordering_check": policy_ordering_check,
        "nf4_fidelity": nf4_result,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transfer-path", default=str(DEFAULT_TRANSFER_PATH))
    ap.add_argument("--e2e-path", default=str(DEFAULT_E2E_PATH))
    ap.add_argument("--nf4-path", default=str(DEFAULT_NF4_PATH))
    ap.add_argument("--compute-calib-path", default=str(DEFAULT_COMPUTE_PATH))
    ap.add_argument("--policy-tolerance-pct", type=float, default=2.0)
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_calibration_report] assembling provenance table from:", flush=True)
    print(f"  compute:  {args.compute_calib_path}", flush=True)
    print(f"  transfer: {args.transfer_path}", flush=True)
    print(f"  e2e:      {args.e2e_path}", flush=True)
    print(f"  nf4:      {args.nf4_path}", flush=True)

    report = build_report(
        transfer_path=Path(args.transfer_path), e2e_path=Path(args.e2e_path),
        nf4_path=Path(args.nf4_path), compute_path=Path(args.compute_calib_path),
        policy_tolerance_pct=args.policy_tolerance_pct,
    )

    for row in report["provenance_table"]:
        print(f"[build_calibration_report]   {row['parameter']}: {row['status']}", flush=True)

    fig_paths = []
    if report["transfer_fit"] is not None:
        try:
            fig_paths = _plot_transfer_fit(Path(args.transfer_path), report["transfer_fit"], out_dir / "figures")
        except Exception as e:  # matplotlib import/env issues shouldn't crash the report itself
            print(f"[build_calibration_report] WARNING: transfer-fit plot skipped: {e}", flush=True)

    report["figures"] = fig_paths
    report_path = out_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[build_calibration_report] wrote {report_path}", flush=True)

    print("[build_calibration_report] rendering RESULTS.md...", flush=True)
    sys.path.insert(0, str(HERE))
    from render_results_md import render  # noqa: E402
    render(report_path, HERE / "RESULTS.md")


def _plot_transfer_fit(transfer_path: Path, fit: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(transfer_path.read_text())
    by_size = data["by_size_bytes"]
    sizes = np.array([float(k) for k in by_size])
    times = np.array([v["mean_ms"] for v in by_size.values()])
    order = np.argsort(sizes)
    sizes, times = sizes[order], times[order]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sizes / 1e6, times, label="measured", color="tab:blue", zorder=3)
    fit_line = fit["alpha_ms"] + sizes / fit["beta_bytes_per_ms"]
    ax.plot(sizes / 1e6, fit_line, "--", color="tab:red",
            label=f"fit: t={fit['alpha_ms']:.3f}+size/{fit['beta_bytes_per_ms']/1e6:.1f}MB/ms "
                  f"(R^2={fit['r_squared']:.3f})")
    ax.set_xlabel("transfer size (MB)")
    ax.set_ylabel("H2D transfer time (ms)")
    ax.set_xscale("log")
    ax.set_title("E3: pinned H2D transfer-time model")
    ax.legend(fontsize=8)
    path = out_dir / "transfer_model_fit.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


if __name__ == "__main__":
    main()
