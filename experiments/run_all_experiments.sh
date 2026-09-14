#!/usr/bin/env bash
# Full E1+E2 pipeline for the remote GPU server (13th-gen i7, 64GB RAM,
# 2x RTX 4060 Ti 16GB). Run this top-to-bottom on the server; nothing here
# should run on a laptop/dev machine.
#
#   0. venv setup (elp_probe/setup.sh, idempotent)          -- no GPU/network
#   1. unit tests (pytest, ~1s, no GPU/network)              -- gate: stop on failure
#   2. model download (download_models.sh)                   -- network, no GPU
#   3. E1: GPU compute calibration + roofline sweep           -- GPU for calib only
#   4. E2: batching sweep (reuses E1's calibration)            -- no GPU
#   5. print where the results landed
#
# Env overrides (all optional): VENV_DIR, SEED, N_TRIALS, RECALL_N_TRIALS,
# RESIDENCY_PCT, BW_GBPS, PRECISION, SLO_MS, CPU_OFFLOAD_GB, GPU_MEM_FRAC,
# SKIP_DOWNLOAD=1, SKIP_CALIB=1, SKIP_TESTS=1.
#
# Usage:
#   bash experiments/run_all_experiments.sh
#   SLO_MS=50 RECALL_N_TRIALS=100 bash experiments/run_all_experiments.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CALIB="${SKIP_CALIB:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"

echo "############################################################"
echo "# tierMoE E1+E2 pipeline"
echo "# repo:  $REPO_ROOT"
echo "# venv:  $VENV_DIR"
echo "############################################################"
echo

echo "=== [0/5] environment setup ==="
if [ ! -d "$VENV_DIR" ]; then
  bash "$REPO_ROOT/elp_probe/setup.sh"
else
  echo "venv already exists at $VENV_DIR -- skipping elp_probe/setup.sh "
  echo "(delete $VENV_DIR first to force a clean rebuild)"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet -r "$HERE/requirements.txt"
echo

if [ "$SKIP_TESTS" != "1" ]; then
  echo "=== [1/5] unit tests (no GPU/network -- gate before spending GPU time) ==="
  python3 -m pytest "$HERE/e1_roofline/tests" "$HERE/e2_batching/tests" -v
  echo
else
  echo "=== [1/5] SKIP_TESTS=1 -- skipping unit tests ==="
  echo
fi

if [ "$SKIP_DOWNLOAD" != "1" ]; then
  echo "=== [2/5] model download ==="
  bash "$HERE/download_models.sh"
  echo
else
  echo "=== [2/5] SKIP_DOWNLOAD=1 -- skipping model download (assuming already cached) ==="
  echo
fi

echo "=== [3/5] E1: MoE Tiering Roofline (GPU compute calibration + sweep) ==="
SKIP_CALIB="$SKIP_CALIB" bash "$HERE/e1_roofline/run_e1.sh"
echo

echo "=== [4/5] E2: Batching Sweet Spot ==="
bash "$HERE/e2_batching/run_e2.sh"
echo

echo "=== [5/5] done ==="
echo "E1 design doc + results: $HERE/e1_roofline/E1_ROOFLINE.md , $HERE/e1_roofline/RESULTS.md"
echo "E2 design doc + results: $HERE/e2_batching/E2_BATCHING.md , $HERE/e2_batching/RESULTS.md"
echo "Figures:                 $HERE/e1_roofline/out/figures/ , $HERE/e2_batching/out/figures/"
echo
echo "Read E2_BATCHING.md's sensitivity note before quoting the claim-check verdict --"
echo "consider re-running e2_batching/run_e2.sh with a different RESIDENCY_PCT/BW_GBPS."
