#!/usr/bin/env bash
# E3 -- Hardware calibration ladder. Run on the target GPU box (single
# 24GB-VRAM GPU -- see E3_CALIBRATION.md's "single-GPU correction" section
# before assuming Experiments.md's dual-4060Ti prose applies literally).
#
# This script covers E3's steps 2-4 (transfer, e2e offload, NF4 fidelity).
# Step 1 (compute) is E1's measure_compute.py, already built and already run
# on this exact box (see ../e1_roofline/out/calib_compute.json) -- run it
# first (or just point --compute-calib-path at wherever its output lives) to
# complete the full ladder:
#   python3 ../e1_roofline/measure_compute.py --model olmoe
#   python3 ../e1_roofline/measure_compute.py --model mixtral --quant nf4
#
# Needs: the venv from elp_probe/setup.sh, and OLMoE's weights already
# downloaded (../download_models.sh) -- Mixtral's weights are NOT needed by
# anything in this script (nf4_fidelity.py and e2e_offload_runner.py are
# both OLMoE-only, by design -- see E3_CALIBRATION.md). Reuses the pilot's
# already-collected traces at results/olmoe/b1/ for the static-hot ranking
# -- no fresh warmup pass.
#
# Env overrides: VENV_DIR, RESIDENCY_PCTS, N_PROMPTS, MAX_NEW, SEED,
# POLICY_TOLERANCE_PCT, SKIP_TRANSFER=1, SKIP_E2E=1, SKIP_NF4=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

RESIDENCY_PCTS="${RESIDENCY_PCTS:-25,50}"
N_PROMPTS="${N_PROMPTS:-10}"
MAX_NEW="${MAX_NEW:-48}"
SEED="${SEED:-0}"
POLICY_TOLERANCE_PCT="${POLICY_TOLERANCE_PCT:-2.0}"
SKIP_TRANSFER="${SKIP_TRANSFER:-0}"
SKIP_E2E="${SKIP_E2E:-0}"
SKIP_NF4="${SKIP_NF4:-0}"

cd "$REPO_ROOT"
mkdir -p "$HERE/out"

if [ "$SKIP_TRANSFER" != "1" ]; then
  echo "=== [1/4] pinned H2D transfer bandwidth sweep ==="
  python3 "$HERE/measure_transfer.py" --out "$HERE/out/calib_transfer.json"
else
  echo "=== [1/4] SKIP_TRANSFER=1 -- skipping ==="
fi
echo

if [ "$SKIP_E2E" != "1" ]; then
  echo "=== [2/4] end-to-end offload policy comparison (OLMoE, r=${RESIDENCY_PCTS}%) ==="
  python3 "$HERE/e2e_offload_runner.py" --residency-pcts "$RESIDENCY_PCTS" \
    --n-prompts "$N_PROMPTS" --max-new "$MAX_NEW" --seed "$SEED" --out "$HERE/out/calib_e2e.json"
else
  echo "=== [2/4] SKIP_E2E=1 -- skipping ==="
fi
echo

if [ "$SKIP_NF4" != "1" ]; then
  echo "=== [3/4] NF4 routing-fidelity spot-check (OLMoE fp16 vs nf4) ==="
  python3 "$HERE/nf4_fidelity.py" --n-prompts 20 --seed "$SEED" --out "$HERE/out/calib_nf4_fidelity.json"
else
  echo "=== [3/4] SKIP_NF4=1 -- skipping ==="
fi
echo

echo "=== [4/4] assembling provenance table + RESULTS.md (no GPU needed for this part) ==="
python3 "$HERE/build_calibration_report.py" \
  --transfer-path "$HERE/out/calib_transfer.json" \
  --e2e-path "$HERE/out/calib_e2e.json" \
  --nf4-path "$HERE/out/calib_nf4_fidelity.json" \
  --compute-calib-path "$REPO_ROOT/experiments/e1_roofline/out/calib_compute.json" \
  --policy-tolerance-pct "$POLICY_TOLERANCE_PCT" \
  --out-dir "$HERE/out"

echo
echo "Done."
echo "Results:  $HERE/RESULTS.md"
echo "Figures:  $HERE/out/figures/"
echo "Raw data: $HERE/out/calibration_report.json"
