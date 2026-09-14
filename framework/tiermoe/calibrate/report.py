"""Assembles the unified calibration provenance table (Main.md's E3
deliverable: "a methodology table ... listing every simulator parameter and
whether it is measured here, measured in cited work, or vendor-specified").
Pure Python, no GPU -- reads whatever calib_*.json files exist and NEVER
fabricates a placeholder for a missing one; a missing file is reported as
MISSING with an actionable instruction, same discipline as
tiermoe.roofline's compute_provenance tagging.
"""
from __future__ import annotations

import json
from pathlib import Path

from tiermoe.calibrate.policy_check import check_policy_ordering
from tiermoe.calibrate.transfer_fit import fit_from_calib_json
from tiermoe.roofline.model import validate_calib_sanity


def _load_json_safe(path: Path):
    path = Path(path)
    if not path.exists():
        return None, f"MISSING (run the GPU script to produce {path})"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError:
        return None, f"MISSING (unreadable/corrupt JSON at {path} -- rerun the GPU script)"


def build_calibration_report(*, compute_path: Path, transfer_path: Path, e2e_path: Path, nf4_path: Path,
                              policy_tolerance_pct: float = 2.0) -> dict:
    provenance_table = []
    transfer_fit_result = policy_ordering_check = nf4_result = None

    compute_data, compute_err = _load_json_safe(compute_path)
    if compute_data is None:
        provenance_table.append({"parameter": "compute", "status": compute_err,
                                  "detail": "run tiermoe.calibrate.compute on a CUDA box"})
    else:
        sanity_warnings = validate_calib_sanity(compute_data)
        for model_tag, entry in compute_data.items():
            status = f"{entry.get('provenance', 'unknown')} (on {', '.join(entry.get('gpu', [])) or 'unknown GPU'})"
            if model_tag in sanity_warnings:
                status = f"SUSPECT ({status}) -- " + "; ".join(sanity_warnings[model_tag])
            provenance_table.append({"parameter": f"compute ({model_tag})", "status": status,
                                      "detail": f"n_layers={entry.get('n_layers')}, quant={entry.get('quant')}"})

    transfer_data, transfer_err = _load_json_safe(transfer_path)
    if transfer_data is None:
        provenance_table.append({"parameter": "transfer (alpha/beta)", "status": transfer_err,
                                  "detail": "run tiermoe.calibrate.transfer_measure on a CUDA box"})
    else:
        try:
            fit = fit_from_calib_json(transfer_data)
            transfer_fit_result = {"alpha_ms": fit.alpha_ms, "beta_bytes_per_ms": fit.beta_bytes_per_ms,
                                    "r_squared": fit.r_squared, "n_points": fit.n_points}
            gpu = ", ".join(transfer_data.get("gpu", [])) or "unknown GPU"
            provenance_table.append({
                "parameter": "transfer (alpha/beta)", "status": f"measured_here (on {gpu})",
                "detail": f"alpha={fit.alpha_ms:.4f}ms, beta={fit.beta_bytes_per_ms/1e6:.1f}MB/ms "
                          f"(~{fit.beta_bytes_per_ms/1e3:.0f}GB/s), R^2={fit.r_squared:.4f}",
            })
        except (ValueError, KeyError) as e:
            provenance_table.append({"parameter": "transfer (alpha/beta)",
                                      "status": f"MISSING (data present but fit failed: {e})", "detail": str(transfer_path)})

    e2e_data, e2e_err = _load_json_safe(e2e_path)
    if e2e_data is None:
        provenance_table.append({"parameter": "e2e offload policy ordering", "status": e2e_err,
                                  "detail": "run tiermoe.calibrate.e2e_offload on a CUDA box"})
    else:
        policy_ordering_check = {}
        for level, level_data in e2e_data.get("residency_levels", {}).items():
            try:
                policy_ordering_check[level] = check_policy_ordering(
                    level_data.get("tpot_ms_by_policy", {}), policy_tolerance_pct)
            except KeyError as e:
                policy_ordering_check[level] = (False, f"FAIL: {e}")
        all_passed = all(v[0] for v in policy_ordering_check.values()) if policy_ordering_check else False
        provenance_table.append({
            "parameter": "e2e offload policy ordering",
            "status": f"measured_here ({'ALL PASS' if all_passed else 'SEE DETAIL'})",
            "detail": "; ".join(f"r={lvl}%: {msg}" for lvl, (_ok, msg) in policy_ordering_check.items()),
        })

    nf4_data, nf4_err = _load_json_safe(nf4_path)
    if nf4_data is None:
        provenance_table.append({"parameter": "nf4 routing fidelity (olmoe)", "status": nf4_err,
                                  "detail": "run tiermoe.calibrate.nf4_fidelity on a CUDA box"})
    else:
        nf4_result = nf4_data
        mean_j = nf4_data.get("mean_jaccard_similarity")
        provenance_table.append({
            "parameter": "nf4 routing fidelity (olmoe)",
            "status": f"measured_here (mean top-k agreement={mean_j:.3f} over {nf4_data.get('n_prompts')} prompts)"
                      if mean_j is not None else "measured_here (see detail)",
            "detail": f"per-layer Jaccard overlap fp16 vs nf4 top-k, n_prompts={nf4_data.get('n_prompts')}",
        })

    n_missing = sum(1 for row in provenance_table if str(row["status"]).startswith("MISSING"))
    return {
        "provenance_table": provenance_table, "n_missing": n_missing,
        "transfer_fit": transfer_fit_result, "policy_ordering_check": policy_ordering_check, "nf4_fidelity": nf4_result,
    }
