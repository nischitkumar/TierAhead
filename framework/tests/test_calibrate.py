import json

import numpy as np
import pytest

from tierahead.calibrate.policy_check import check_policy_ordering
from tierahead.calibrate.report import build_calibration_report
from tierahead.calibrate.transfer_fit import fit_affine_transfer_curve, fit_from_calib_json


def test_fit_affine_transfer_curve_recovers_known_alpha_beta():
    alpha_true, beta_true = 0.05, 2_000_000.0  # ms, bytes/ms
    sizes = np.array([1e6, 2e6, 4e6, 8e6, 16e6, 32e6])
    times = alpha_true + sizes / beta_true
    fit = fit_affine_transfer_curve(sizes, times)
    assert abs(fit.alpha_ms - alpha_true) < 1e-6
    assert abs(fit.beta_bytes_per_ms - beta_true) / beta_true < 1e-6
    assert fit.r_squared > 0.999


def test_fit_affine_transfer_curve_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_affine_transfer_curve([1e6], [1.0])


def test_fit_affine_transfer_curve_rejects_identical_sizes():
    with pytest.raises(ValueError):
        fit_affine_transfer_curve([1e6, 1e6, 1e6], [1.0, 1.1, 0.9])


def test_fit_from_calib_json():
    calib = {"by_size_bytes": {"1000000": {"mean_ms": 0.55}, "2000000": {"mean_ms": 1.05}, "4000000": {"mean_ms": 2.05}}}
    fit = fit_from_calib_json(calib)
    assert fit.beta_bytes_per_ms > 0


def test_check_policy_ordering_pass():
    passed, msg = check_policy_ordering({"next-layer-topk": 1.0, "static-hot": 1.5, "none": 2.0})
    assert passed and msg.startswith("PASS")


def test_check_policy_ordering_fail():
    passed, msg = check_policy_ordering({"next-layer-topk": 3.0, "static-hot": 1.5, "none": 2.0})
    assert not passed and msg.startswith("FAIL")


def test_check_policy_ordering_missing_key_raises():
    with pytest.raises(KeyError):
        check_policy_ordering({"next-layer-topk": 1.0, "none": 2.0})


def test_build_calibration_report_all_missing(tmp_path):
    report = build_calibration_report(
        compute_path=tmp_path / "c.json", transfer_path=tmp_path / "t.json",
        e2e_path=tmp_path / "e.json", nf4_path=tmp_path / "n.json",
    )
    assert report["n_missing"] == 4
    assert all(str(row["status"]).startswith("MISSING") for row in report["provenance_table"])


def test_build_calibration_report_partial(tmp_path):
    compute_path = tmp_path / "c.json"
    compute_path.write_text(json.dumps({"olmoe": {"provenance": "measured_here", "gpu": ["Test GPU"],
                                                     "n_layers": 16, "quant": "none",
                                                     "by_batch": {"1": {"mean_ms": 1.0}, "8": {"mean_ms": 4.5}}}}))
    report = build_calibration_report(
        compute_path=compute_path, transfer_path=tmp_path / "t.json",
        e2e_path=tmp_path / "e.json", nf4_path=tmp_path / "n.json",
    )
    assert report["n_missing"] == 3
    compute_row = next(r for r in report["provenance_table"] if "compute" in r["parameter"])
    assert "measured_here" in compute_row["status"]


def test_build_calibration_report_flags_suspect_compute(tmp_path):
    compute_path = tmp_path / "c.json"
    compute_path.write_text(json.dumps({"olmoe": {"provenance": "measured_here", "gpu": ["Test GPU"],
                                                     "n_layers": 16, "quant": "none",
                                                     "by_batch": {"1": {"mean_ms": 2.99}, "8": {"mean_ms": 3.03}}}}))
    report = build_calibration_report(
        compute_path=compute_path, transfer_path=tmp_path / "t.json",
        e2e_path=tmp_path / "e.json", nf4_path=tmp_path / "n.json",
    )
    compute_row = next(r for r in report["provenance_table"] if "compute" in r["parameter"])
    assert "SUSPECT" in compute_row["status"]
