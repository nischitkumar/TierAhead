"""Unit tests for E3's pure-Python analysis logic (transfer curve fitting,
policy-ordering validation, calibration report assembly). No GPU, no
network, no trace files required. Run with:
    pytest experiments/e3_calibration/tests/test_e3_calibration.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # e3_calibration/

import transfer_fit as tf  # noqa: E402
import policy_check as pc  # noqa: E402
import build_calibration_report as bcr  # noqa: E402


# ---------- transfer_fit ----------

def test_fit_recovers_known_alpha_beta_noiseless():
    true_alpha, true_beta = 0.05, 20_000.0  # ms, bytes/ms (=20 GB/s)
    sizes = [1e6, 2e6, 4e6, 8e6, 16e6, 32e6]
    times = [true_alpha + s / true_beta for s in sizes]
    fit = tf.fit_affine_transfer_curve(sizes, times)
    assert fit.alpha_ms == pytest.approx(true_alpha, abs=1e-9)
    assert fit.beta_bytes_per_ms == pytest.approx(true_beta, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.n_points == len(sizes)


def test_fit_recovers_known_alpha_beta_with_noise():
    rng = np.random.default_rng(0)
    true_alpha, true_beta = 0.1, 15_000.0
    sizes = [1e6 * m for m in [1, 2, 4, 8, 16, 32, 64, 128]]
    times = [true_alpha + s / true_beta + rng.normal(0, 0.01) for s in sizes]
    fit = tf.fit_affine_transfer_curve(sizes, times)
    assert fit.alpha_ms == pytest.approx(true_alpha, abs=0.05)
    assert fit.beta_bytes_per_ms == pytest.approx(true_beta, rel=0.1)
    assert fit.r_squared > 0.99


def test_fit_rejects_fewer_than_two_points():
    with pytest.raises(ValueError, match=">=2"):
        tf.fit_affine_transfer_curve([1e6], [0.5])


def test_fit_rejects_all_identical_sizes():
    with pytest.raises(ValueError, match="identical"):
        tf.fit_affine_transfer_curve([1e6, 1e6, 1e6], [0.5, 0.6, 0.4])


def test_fit_rejects_non_positive_slope():
    # times DECREASING with size -- physically nonsensical for a transfer, slope < 0
    sizes = [1e6, 2e6, 4e6, 8e6]
    times = [1.0, 0.8, 0.6, 0.4]
    with pytest.raises(ValueError, match="non-positive"):
        tf.fit_affine_transfer_curve(sizes, times)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        tf.fit_affine_transfer_curve([1e6, 2e6, 3e6], [0.1, 0.2])


def test_fit_from_calib_json_shape():
    calib = {
        "by_size_bytes": {
            "1000000": {"mean_ms": 0.15},
            "2000000": {"mean_ms": 0.20},
            "4000000": {"mean_ms": 0.30},
            "8000000": {"mean_ms": 0.50},
        }
    }
    fit = tf.fit_from_calib_json(calib)
    assert fit.n_points == 4
    assert fit.beta_bytes_per_ms > 0


# ---------- policy_check ----------

def test_policy_ordering_passes_when_correctly_ordered():
    passed, msg = pc.check_policy_ordering({"next-layer-topk": 1.0, "static-hot": 2.0, "none": 3.0})
    assert passed
    assert "PASS" in msg


def test_policy_ordering_fails_when_next_layer_topk_slower_than_static_hot():
    passed, msg = pc.check_policy_ordering({"next-layer-topk": 5.0, "static-hot": 2.0, "none": 3.0})
    assert not passed
    assert "FAIL" in msg
    assert "next-layer-topk" in msg


def test_policy_ordering_fails_when_static_hot_slower_than_none():
    passed, msg = pc.check_policy_ordering({"next-layer-topk": 1.0, "static-hot": 5.0, "none": 3.0})
    assert not passed
    assert "static-hot" in msg


def test_policy_ordering_within_tolerance_still_passes():
    # static-hot is 1% slower than next-layer-topk -- inside the default 2% tolerance
    passed, msg = pc.check_policy_ordering({"next-layer-topk": 2.0, "static-hot": 2.02, "none": 3.0})
    assert passed


def test_policy_ordering_beyond_tolerance_fails():
    # next-layer-topk (2.2) exceeds static-hot (2.0) by 10% -- beyond the 2% tolerance,
    # and in the wrong direction (next-layer-topk should be <= static-hot, not >).
    passed, msg = pc.check_policy_ordering(
        {"next-layer-topk": 2.2, "static-hot": 2.0, "none": 3.0}, tolerance_pct=2.0)
    assert not passed


def test_policy_ordering_raises_on_missing_policy():
    with pytest.raises(KeyError, match="static-hot"):
        pc.check_policy_ordering({"next-layer-topk": 1.0, "none": 3.0})


def test_policy_ordering_raises_on_all_missing():
    with pytest.raises(KeyError):
        pc.check_policy_ordering({})


def test_policy_ordering_all_equal_passes_within_tolerance():
    passed, msg = pc.check_policy_ordering({"next-layer-topk": 2.0, "static-hot": 2.0, "none": 2.0})
    assert passed


# ---------- build_calibration_report: missing-file handling ----------

def test_report_all_missing_when_no_files_exist(tmp_path):
    report = bcr.build_report(
        transfer_path=tmp_path / "nope_transfer.json",
        e2e_path=tmp_path / "nope_e2e.json",
        nf4_path=tmp_path / "nope_nf4.json",
        compute_path=tmp_path / "nope_compute.json",
    )
    for entry in report["provenance_table"]:
        assert entry["status"].startswith("MISSING")
    assert report["policy_ordering_check"] is None
    assert report["transfer_fit"] is None


def test_report_mixed_missing_and_present(tmp_path):
    """Regression scenario for this exact repo state: E1's real compute
    calibration exists, but E3's own transfer/e2e/nf4 files don't yet --
    the report must handle a PARTIAL set correctly, not treat 'some files
    exist' as 'all files exist' or vice versa."""
    compute_path = tmp_path / "calib_compute.json"
    compute_path.write_text(json.dumps({
        "olmoe": {"provenance": "measured_here", "gpu": ["NVIDIA RTX 4500 Ada Generation"],
                   "by_batch": {"1": {"mean_ms": 3.0}}}
    }))
    report = bcr.build_report(
        transfer_path=tmp_path / "missing_transfer.json",
        e2e_path=tmp_path / "missing_e2e.json",
        nf4_path=tmp_path / "missing_nf4.json",
        compute_path=compute_path,
    )
    statuses = {e["parameter"]: e["status"] for e in report["provenance_table"]}
    assert statuses["compute (olmoe)"].startswith("measured_here")
    assert statuses["transfer (alpha/beta)"].startswith("MISSING")
    assert statuses["e2e offload policy ordering"].startswith("MISSING")
    assert statuses["nf4 routing fidelity (olmoe)"].startswith("MISSING")


def test_report_includes_policy_check_when_e2e_present(tmp_path):
    e2e_path = tmp_path / "calib_e2e.json"
    e2e_path.write_text(json.dumps({
        "provenance": "measured_here",
        "residency_levels": {
            "25": {"tpot_ms_by_policy": {"next-layer-topk": 10.0, "static-hot": 15.0, "none": 20.0}}
        },
    }))
    report = bcr.build_report(
        transfer_path=tmp_path / "x.json", e2e_path=e2e_path,
        nf4_path=tmp_path / "y.json", compute_path=tmp_path / "z.json",
    )
    assert report["policy_ordering_check"] is not None
    assert report["policy_ordering_check"]["25"][0] is True


def test_report_includes_transfer_fit_when_transfer_present(tmp_path):
    transfer_path = tmp_path / "calib_transfer.json"
    transfer_path.write_text(json.dumps({
        "provenance": "measured_here",
        "by_size_bytes": {
            "1000000": {"mean_ms": 0.1}, "8000000": {"mean_ms": 0.3}, "64000000": {"mean_ms": 1.5},
        },
    }))
    report = bcr.build_report(
        transfer_path=transfer_path, e2e_path=tmp_path / "x.json",
        nf4_path=tmp_path / "y.json", compute_path=tmp_path / "z.json",
    )
    assert report["transfer_fit"] is not None
    assert report["transfer_fit"]["beta_bytes_per_ms"] > 0


def test_report_never_crashes_on_malformed_json(tmp_path):
    bad = tmp_path / "calib_e2e.json"
    bad.write_text("{not valid json")
    report = bcr.build_report(
        transfer_path=tmp_path / "x.json", e2e_path=bad,
        nf4_path=tmp_path / "y.json", compute_path=tmp_path / "z.json",
    )
    statuses = {e["parameter"]: e["status"] for e in report["provenance_table"]}
    assert "MISSING" in statuses["e2e offload policy ordering"] or "unreadable" in statuses["e2e offload policy ordering"].lower()
