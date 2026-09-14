#!/usr/bin/env bash
# E5 -- Lookahead depth and pipelined prefetch. Runs on a LAPTOP or the GPU
# server, identical command either way -- reuses E4's predictor machinery
# (no GPU needed for training these small per-layer MLPs). No model
# weights, no download.
#
# Env overrides: VENV_DIR, EPOCHS, HIDDEN_DIM, SEED, D_VALUES, CALIB
# (path to a real calib_compute.json from E1/E3, once one exists on the
# server -- replaces the flop_estimate fallback for the feasible-fetch-size
# and queue-sim numbers), SKIP_TESTS=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/experiments/.venv-mac}"

if [ ! -d "$VENV_DIR" ]; then
  echo "=== creating local venv at $VENV_DIR (--system-site-packages) ==="
  python3 -m venv --system-site-packages "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --quiet torch zstandard pytest
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python3 -c "import torch" 2>/dev/null || pip install --quiet torch
fi

EPOCHS="${EPOCHS:-60}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
SEED="${SEED:-0}"
D_VALUES="${D_VALUES:-1,2,4,8}"
CALIB="${CALIB:-}"
SKIP_TESTS="${SKIP_TESTS:-0}"

cd "$REPO_ROOT"

if [ "$SKIP_TESTS" != "1" ]; then
  echo "=== [1/2] unit tests (no GPU/network) ==="
  python3 -m pytest "$HERE/tests" -v
  echo
else
  echo "=== [1/2] SKIP_TESTS=1 -- skipping unit tests ==="
fi

CALIB_ARG=()
if [ -n "$CALIB" ]; then
  CALIB_ARG=(--calib "$CALIB")
fi

echo "=== [2/2] direct vs chained lookahead + feasible-fetch-size + queue sim ==="
python3 "$HERE/lookahead.py" --models olmoe,mixtral --d-values "$D_VALUES" --epochs "$EPOCHS" \
  --hidden-dim "$HIDDEN_DIM" --seed "$SEED" --out-dir "$HERE/out" "${CALIB_ARG[@]}"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/lookahead_report.json"
