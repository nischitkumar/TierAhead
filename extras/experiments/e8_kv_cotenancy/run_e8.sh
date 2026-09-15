#!/usr/bin/env bash
# E8 -- KV-cache co-tenancy on the CXL link. Unlike run_e1.sh/run_e2.sh, this
# is written to run on a LAPTOP, not the GPU server: no model weights, no
# GPU, no download. It only needs numpy/pandas/matplotlib (+ zstandard to
# read the pilot's .zst traces, + pytest for the test gate).
#
# Uses experiments/.venv-mac (created with --system-site-packages so it
# reuses whatever numpy/pandas/matplotlib the system Python already has,
# and only needs to pip-install zstandard+pytest into the venv itself). If
# that venv doesn't exist yet, this script creates it.
#
# Env overrides: VENV_DIR, SEED, SIM_SECONDS, KV_RATE, BW_GRID, POLICIES,
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
  pip install --quiet zstandard pytest
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

SEED="${SEED:-0}"
SIM_SECONDS="${SIM_SECONDS:-30}"
KV_RATE="${KV_RATE:-2.0}"
BW_GRID="${BW_GRID:-32,64}"
POLICIES="${POLICIES:-expert-first,kv-first,weighted:0.2,weighted:0.5,weighted:0.8}"
SKIP_TESTS="${SKIP_TESTS:-0}"

cd "$REPO_ROOT"

if [ "$SKIP_TESTS" != "1" ]; then
  echo "=== [1/2] unit tests (no GPU/network) ==="
  python3 -m pytest "$HERE/tests" -v
  echo
else
  echo "=== [1/2] SKIP_TESTS=1 -- skipping unit tests ==="
fi

echo "=== [2/2] KV/expert link-contention sweep + figures + RESULTS.md ==="
python3 "$HERE/kv_link_sim.py" --models olmoe,mixtral --bw-grid "$BW_GRID" \
  --policies "$POLICIES" --kv-rate-per-sec "$KV_RATE" --sim-seconds "$SIM_SECONDS" \
  --seed "$SEED" --out-dir "$HERE/out"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/kv_contention_summary.json"
