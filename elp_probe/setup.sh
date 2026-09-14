#!/usr/bin/env bash
# ELP-Probe setup: one-shot env prep for the target box
# (i7-14700F, 28 threads, 62 GB RAM, 1x RTX 4500 Ada 24GB, driver 580.173.02 / CUDA 13.0).
#
# This script installs the env only. It does NOT download the model —
# that happens on first run of collect.py (HF cache), by design, per request.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"

echo "[setup] venv -> $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

echo "[setup] installing CUDA torch (cu121 wheel; driver 580/CUDA13 is backward compatible)"
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "[setup] installing project requirements"
pip install -r "$HERE/requirements.txt"

echo "[setup] verifying CUDA visible to torch"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('GPU:', torch.cuda.get_device_name(0), '| VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1))"

echo "[setup] creating data dirs"
mkdir -p "$HERE/results/olmoe/b1" "$HERE/results/olmoe/b8" "$HERE/results/prompts" "$HERE/results/figures"

echo
echo "[setup] done. Next steps:"
echo "  1) source $VENV_DIR/bin/activate"
echo "  2) huggingface-cli login          # only needed if HF account required for download"
echo "  3) bash run_pilot.sh              # downloads OLMoE-1B-7B on first run, then runs the full pilot"
