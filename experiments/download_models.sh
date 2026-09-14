#!/usr/bin/env bash
# Downloads both models' weights into the HF cache on THIS machine. Run this
# on the remote GPU server only -- never on a machine without GPU/large disk,
# and never trigger a model download from a laptop-side session.
#
# OLMoE-1B-7B is ungated (public). Mixtral-8x7B-Instruct is GATED: you must
# (1) accept the license at
#     https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
# while logged in with the HF account you'll authenticate with below, and
# (2) run `hf auth login` (or export HF_TOKEN) before this script,
# or the Mixtral download will fail with a 403.
#
# Uses the `hf` CLI -- `huggingface-cli` is deprecated and no longer works on
# current huggingface_hub. Needs only the venv + huggingface_hub<1.0 (see the
# pin note further down for why the ceiling matters) -- does NOT need
# torch/transformers loaded, so this step is fast and light even before the
# rest of the env is fully set up.
#
# Expected total: ~14GB (OLMoE) + ~93GB (Mixtral safetensors) = ~107GB.
# Verified via the HF API (2026-09) that the Mixtral repo ALSO ships 8
# `consolidated.NN.pt` files -- Mistral's own native checkpoint format,
# a full duplicate of the same weights the 19 `.safetensors` shards already
# hold (transformers/bitsandbytes never touch the .pt files; only the
# safetensors shards get loaded). Downloading both would silently double the
# pull to ~186GB for nothing, so the Mixtral download below explicitly
# excludes `*.pt`. OLMoE's repo has no such duplication (safetensors only).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"

if [ ! -d "$VENV_DIR" ]; then
  echo "[download_models] no venv at $VENV_DIR -- running elp_probe/setup.sh first"
  bash "$REPO_ROOT/elp_probe/setup.sh"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Only install huggingface_hub if `hf` is missing -- do NOT blindly upgrade.
# This venv is SHARED with everything else in experiments/ and elp_probe/,
# including transformers==4.46.3 (pinned elsewhere for the Mixtral NF4 bug --
# see e1_roofline/measure_compute.py), which requires huggingface-hub<1.0.
# An earlier version of this script ran a bare `pip install --upgrade
# huggingface_hub`, which pulled the newest release (>=1.0) and broke
# transformers venv-wide -- even for OLMoE, which never touches bitsandbytes
# -- with `ImportError: huggingface-hub>=0.23.2,<1.0 is required`. `hf` has
# been available (with `huggingface-cli` deprecated-but-present) well before
# huggingface_hub's 1.0 release, so pin the ceiling if an install is needed.
if ! command -v hf >/dev/null; then
  echo "[download_models] 'hf' not found -- installing huggingface_hub<1.0 (see comment above for why <1.0)"
  pip install --quiet "huggingface_hub<1.0"
fi
command -v hf >/dev/null || { echo "[download_models] 'hf' still not on PATH -- check the venv activated correctly" >&2; exit 1; }
# Defensive: if some other package already dragged in huggingface_hub>=1.0
# before this script ran, pull it back under the ceiling instead of leaving
# transformers broken venv-wide.
python3 -c "
import sys
from importlib.metadata import version
from packaging.version import Version
v = Version(version('huggingface_hub'))
sys.exit(0 if v < Version('1.0') else 1)
" || pip install --quiet "huggingface_hub<1.0"

echo "=== [1/2] downloading allenai/OLMoE-1B-7B-0924 (ungated, ~14GB fp16) ==="
hf download allenai/OLMoE-1B-7B-0924

echo
echo "=== [2/2] downloading mistralai/Mixtral-8x7B-Instruct-v0.1 (GATED, ~93GB, .pt duplicates excluded) ==="
echo "    If this fails with a 403/401: run 'hf auth login' and accept the license at"
echo "    https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 first."
hf download mistralai/Mixtral-8x7B-Instruct-v0.1 --exclude "*.pt"

echo
echo "[download_models] done. Cached under \${HF_HOME:-~/.cache/huggingface}."
df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || true
