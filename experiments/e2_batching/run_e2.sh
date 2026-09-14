#!/usr/bin/env bash
# E2 -- Batching sweet spot. Run on the target GPU box after run_e1.sh (reuses
# its calib_compute.json if present). See E2_BATCHING.md for the design
# rationale, including the throughput-vs-latency and batched-recall-saturation
# caveats -- read those before quoting a single run's numbers.
#
# No GPU needed for this script itself -- it only reads already-collected
# traces + (optionally) E1's calibration file.
#
# Env overrides: VENV_DIR, SEED, N_TRIALS (E_union resampling), RECALL_N_TRIALS
# (step 4, the slow part -- lower this for a quick check), RESIDENCY_PCT,
# BW_GBPS, PRECISION, SLO_MS (see E2_BATCHING.md's sensitivity note -- try more
# than one operating point before trusting the claim-check verdict).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

SEED="${SEED:-0}"
N_TRIALS="${N_TRIALS:-500}"
RECALL_N_TRIALS="${RECALL_N_TRIALS:-200}"
RESIDENCY_PCT="${RESIDENCY_PCT:-25}"
BW_GBPS="${BW_GBPS:-64}"
PRECISION="${PRECISION:-fp16}"
SLO_MS="${SLO_MS:-100}"

cd "$REPO_ROOT"
mkdir -p "$HERE/out"

CALIB_PATH="$REPO_ROOT/experiments/e1_roofline/out/calib_compute.json"
CALIB_ARG=()
if [ -f "$CALIB_PATH" ]; then
  echo "=== using E1's compute calibration: $CALIB_PATH ==="
  CALIB_ARG=(--calib "$CALIB_PATH")
else
  echo "=== no calib_compute.json found (run e1_roofline/run_e1.sh first for measured compute) --"
  echo "    falling back to flop_estimate ==="
fi

echo "=== batching sweep: olmoe + mixtral, r=${RESIDENCY_PCT}% bw=${BW_GBPS}GB/s ${PRECISION} SLO=${SLO_MS}ms ==="
python3 "$HERE/batching_sweep.py" --models olmoe,mixtral \
  --residency-pct "$RESIDENCY_PCT" --bw-gbps "$BW_GBPS" --precision "$PRECISION" --slo-ms "$SLO_MS" \
  --n-trials "$N_TRIALS" --recall-n-trials "$RECALL_N_TRIALS" --seed "$SEED" \
  --out-dir "$HERE/out" "${CALIB_ARG[@]}"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/batching_summary.json"
echo
echo "NOTE: the claim-check verdict is sensitive to (RESIDENCY_PCT, BW_GBPS) --"
echo "      see E2_BATCHING.md before quoting it from a single run."
