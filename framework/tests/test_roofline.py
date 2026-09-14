from tiermoe.roofline.model import (
    bytes_per_layer, get_t_compute_ms, required_bw_gbps, rho, run_roofline,
    t_compute_flop_ms, t_transfer_ms, validate_calib_sanity,
)
from tiermoe.specs import MODEL_SPECS


def test_rho_less_than_one_is_latency_bound():
    assert rho(t_transfer_val_ms=1.0, t_compute_val_ms=10.0) < 1.0
    assert rho(t_transfer_val_ms=10.0, t_compute_val_ms=1.0) >= 1.0


def test_t_compute_flop_scales_with_batch():
    spec = MODEL_SPECS["olmoe"]
    t1 = t_compute_flop_ms(spec, batch=1)
    t8 = t_compute_flop_ms(spec, batch=8)
    assert t8 > t1
    assert abs(t8 / t1 - 8.0) < 1e-6  # this analytical model is exactly linear in batch by construction


def test_get_t_compute_ms_falls_back_to_flop_estimate_without_calib():
    spec = MODEL_SPECS["olmoe"]
    t_ms, provenance = get_t_compute_ms(spec, batch=1, calib=None)
    assert provenance == "flop_estimate"
    assert t_ms > 0


def test_get_t_compute_ms_uses_calib_when_present():
    spec = MODEL_SPECS["mixtral"]
    calib = {"mixtral": {"by_batch": {"1": {"mean_ms": 0.5}}}}
    t_ms, provenance = get_t_compute_ms(spec, batch=1, calib=calib)
    assert provenance == "measured_here"
    assert t_ms == 0.5


def test_required_bw_gbps_round_trips_with_t_transfer():
    bw_needed = required_bw_gbps(bytes_val=1e9, t_compute_val_ms=10.0, target_rho=1.0)
    back = t_transfer_ms(1e9, bw_needed)
    assert abs(back - 10.0) < 1e-6


def test_bytes_per_layer_zero_at_full_residency():
    spec = MODEL_SPECS["olmoe"]
    assert bytes_per_layer(spec, batch=1, residency_pct=100.0, precision="fp16", e_union_val=8.0) == 0.0


def test_validate_calib_sanity_flags_flat_batch_scaling():
    """This is the exact shape this framework's own due-diligence review
    found in the repo's shipped calib_compute.json -- see roofline/model.py's
    docstring and PREDICTED.md's Methodology section."""
    suspicious = {"olmoe": {"by_batch": {
        "1": {"mean_ms": 2.99, "per_layer_mean_ms": {str(i): 2.99 + 0.001 * i for i in range(16)}},
        "8": {"mean_ms": 3.03, "per_layer_mean_ms": {str(i): 3.03 + 0.001 * i for i in range(16)}},
    }}}
    warnings = validate_calib_sanity(suspicious)
    assert "olmoe" in warnings and len(warnings["olmoe"]) >= 1


def test_validate_calib_sanity_accepts_plausible_scaling():
    plausible = {"olmoe": {"by_batch": {
        "1": {"mean_ms": 1.0, "per_layer_mean_ms": {str(i): 1.0 + 0.3 * ((i * 37) % 7) for i in range(16)}},
        "8": {"mean_ms": 4.5, "per_layer_mean_ms": {str(i): 4.5 + 0.3 * ((i * 53) % 7) for i in range(16)}},
        "32": {"mean_ms": 15.0, "per_layer_mean_ms": {str(i): 15.0 + 0.3 * ((i * 61) % 7) for i in range(16)}},
    }}}
    warnings = validate_calib_sanity(plausible)
    assert warnings == {}


def test_run_roofline_on_real_committed_traces(data_root):
    result = run_roofline(models=["olmoe", "mixtral"], data_root=data_root, n_trials=50)
    assert set(result["e_union_by_batch"].keys()) == {"olmoe", "mixtral"}
    assert len(result["sweep_df"]) > 0
    assert all(prov == "flop_estimate" for prov in result["compute_provenance"].values())
    # E_union(B=1) must equal k exactly (batch=1 -> the union of one row IS that row).
    assert result["e_union_by_batch"]["olmoe"][1] == MODEL_SPECS["olmoe"].k
    assert result["e_union_by_batch"]["mixtral"][1] == MODEL_SPECS["mixtral"].k
