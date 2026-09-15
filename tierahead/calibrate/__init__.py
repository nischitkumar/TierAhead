"""Hardware calibration ladder (Main.md's E3 scope). Every simulated number
in this framework either traces back to a measurement or a citation -- this
package produces the measured half.

  compute.py          -- GPU-required: per-layer/per-expert compute time.
  transfer_measure.py -- GPU-required: real pinned H2D transfer bandwidth.
  transfer_fit.py      -- pure math: fits t = alpha + size/beta to the above.
  e2e_offload.py       -- GPU-required: real offload-policy ordering ground truth.
  nf4_fidelity.py       -- GPU-required: FP16 vs NF4 routing-agreement spot-check.
  policy_check.py       -- pure logic: validates the e2e ordering.
  report.py             -- pure: assembles the unified provenance table,
                            never fabricating a placeholder for a missing file.
"""
from .policy_check import check_policy_ordering
from .report import build_calibration_report
from .transfer_fit import TransferFit, fit_affine_transfer_curve, fit_from_calib_json

__all__ = [
    "check_policy_ordering", "build_calibration_report",
    "TransferFit", "fit_affine_transfer_curve", "fit_from_calib_json",
]
