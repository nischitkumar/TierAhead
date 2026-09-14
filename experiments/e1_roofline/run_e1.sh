#!/usr/bin/env bash
# E1 -- The MoE Tiering Roofline. Run on the target GPU box (any single GPU
# with >=~16GB or a multi-GPU pool -- hardware-agnostic, see
# measure_compute.py). See E1_ROOFLINE.md for the full design rationale.
#
# Needs: the venv from elp_probe/setup.sh, and OLMoE/Mixtral weights already
# downloaded (../download_models.sh) -- though Mixtral's calibration (--quant
# nf4) doesn't actually load those weights; see measure_compute.py's "Why no
# full-model load for NF4" docstring. Reuses the pilot's already-collected
# traces at results/{olmoe,mixtral}/b1/ -- no new trace collection here.
#
# Env overrides: VENV_DIR, SEED, N_TRIALS, SKIP_CALIB=1 (skip the GPU
# calibration step and use the flop_estimate fallback).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

SEED="${SEED:-0}"
N_TRIALS="${N_TRIALS:-500}"
SKIP_CALIB="${SKIP_CALIB:-0}"

cd "$REPO_ROOT"
CALIB_PATH="$HERE/out/calib_compute.json"
mkdir -p "$HERE/out"

if [ "$SKIP_CALIB" != "1" ]; then
  echo "=== [1/3] GPU compute calibration: olmoe (fp16, single GPU) ==="
  python3 "$HERE/measure_compute.py" --model olmoe --out "$CALIB_PATH"

  echo "=== [2/3] GPU compute calibration: mixtral (standalone NF4 MoE block, single GPU) ==="
  python3 "$HERE/measure_compute.py" --model mixtral --quant nf4 --out "$CALIB_PATH"
else
  echo "=== [1-2/3] SKIP_CALIB=1 -- skipping GPU calibration, roofline will use flop_estimate ==="
fi

echo "=== [3/3] roofline sweep + figures + RESULTS.md (no GPU needed for this part) ==="
CALIB_ARG=()
if [ -f "$CALIB_PATH" ]; then
  CALIB_ARG=(--calib "$CALIB_PATH")
fi
python3 "$HERE/roofline.py" --models olmoe,mixtral --n-trials "$N_TRIALS" --seed "$SEED" \
  --out-dir "$HERE/out" "${CALIB_ARG[@]}"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/roofline_summary.json"
