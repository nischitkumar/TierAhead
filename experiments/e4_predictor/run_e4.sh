#!/usr/bin/env bash
# E4 -- Predictor upgrade and confidence calibration. Runs on a LAPTOP or the
# GPU server -- identical command either way, since training these small
# per-layer MLPs needs no GPU (Experiments.md E4 itself: "trains in minutes
# on CPU/MPS"). No model weights, no download.
#
# On this repo's laptop dev venv (experiments/.venv-mac), creates it if
# missing (numpy/pandas/matplotlib inherited via --system-site-packages,
# torch + zstandard + pytest pip-installed). On the GPU server, just reuse
# the existing $HOME/elp venv (already has torch/pandas/numpy from
# elp_probe/setup.sh) via VENV_DIR=$HOME/elp.
#
# Env overrides: VENV_DIR, EPOCHS, HIDDEN_DIM, SEED, HIDDEN_STATE_FILE
# (comma list model=path, from collect_hidden_states.py -- see design doc),
# SKIP_TESTS=1.
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
HIDDEN_STATE_FILE="${HIDDEN_STATE_FILE:-}"
SKIP_TESTS="${SKIP_TESTS:-0}"

cd "$REPO_ROOT"

if [ "$SKIP_TESTS" != "1" ]; then
  echo "=== [1/2] unit tests (no GPU/network) ==="
  python3 -m pytest "$HERE/tests" -v
  echo
else
  echo "=== [1/2] SKIP_TESTS=1 -- skipping unit tests ==="
fi

HS_ARG=()
if [ -n "$HIDDEN_STATE_FILE" ]; then
  HS_ARG=(--hidden-state-file "$HIDDEN_STATE_FILE")
fi

echo "=== [2/2] predictor ablation + recall-vs-bytes + calibration + cross-domain transfer ==="
python3 "$HERE/train_eval.py" --models olmoe,mixtral --epochs "$EPOCHS" --hidden-dim "$HIDDEN_DIM" \
  --seed "$SEED" --out-dir "$HERE/out" "${HS_ARG[@]}"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/predictor_report.json"
