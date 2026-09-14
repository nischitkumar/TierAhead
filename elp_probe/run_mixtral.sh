#!/usr/bin/env bash
# ELP-Probe Mixtral run (Pilot.md §4.1, §1.6). Run this on one of the 2x
# RTX 4060 Ti 16GB boxes (32GB pooled on one node) -- NOT across machines.
# NF4 quant -> ~26GB weights, fits with headroom for KV cache/activations.
# device_map="auto" splits layers across both GPUs on this node automatically.
#
# Needs: `huggingface-cli login` + accepted license at
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 (gated repo).
#
# Pinned transformers==4.46.3: current transformers (>=5.x) refactored
# Mixtral's experts into a fused MixtralExperts module holding raw
# nn.Parameter 3D tensors instead of per-expert nn.Linear (same refactor we
# hit on OLMoE's router). bitsandbytes' load_in_4bit only auto-quantizes
# nn.Linear submodules -- against the fused module it quantizes nothing, so
# the ~93GB fp16 footprint stays full-precision and no memory-budget flag
# can make that fit. 4.46.3's Mixtral still uses classic per-expert
# nn.Linear (w1/w2/w3), which bnb quantizes correctly to ~26GB. This pin is
# local to this venv -- does not touch the OLMoE box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "=== [pin] transformers==4.46.3 (pre fused-experts refactor, needed for bnb NF4 to actually quantize Mixtral) ==="
pip install --quiet "transformers==4.46.3"

SEED="${SEED:-0}"
N_SHAREGPT="${N_SHAREGPT:-150}"
N_CODE="${N_CODE:-100}"
MAX_NEW="${MAX_NEW:-256}"
# Shared box (anjuna2.dashlab.in) -- `free -h` "available" is the real budget,
# other tenants' RSS (cockroach/keycloak/influxdb/etc) is not yours to reclaim.
# Keep this comfortably under whatever `free -h` shows as available right now.
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-20}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.85}"

B1_DIR="$HERE/results/mixtral/b1"

cd "$HERE"
mkdir -p "$B1_DIR"

echo "=== [0/2] smoke test: 3 prompts, verify hooks + sanity asserts (NF4 load can take a few min) ==="
python3 src/collect.py --model mixtral --quant nf4 --workloads sharegpt --n-sharegpt 3 --n-code 0 \
  --max-new 32 --seed "$SEED" --out "$B1_DIR/_smoke" --smoke 3 \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" --gpu-mem-frac "$GPU_MEM_FRAC"

echo "=== [1/2] full trace collection: B=1, sharegpt($N_SHAREGPT) + code($N_CODE), max_new=$MAX_NEW ==="
python3 src/collect.py --model mixtral --quant nf4 --workloads sharegpt,code \
  --n-sharegpt "$N_SHAREGPT" --n-code "$N_CODE" --max-new "$MAX_NEW" \
  --seed "$SEED" --out "$B1_DIR" --mode b1 \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" --gpu-mem-frac "$GPU_MEM_FRAC"

echo "=== [2/2] analysis ==="
python3 src/analyze.py --b1-dir "$B1_DIR" --b8-dir "$HERE/results/mixtral/b8" \
  --out "$HERE/results/mixtral_summary.json" --fig-dir "$HERE/results/figures_mixtral" --seed "$SEED"

python3 - <<'PY'
import json
s = json.load(open("results/mixtral_summary.json"))
h = s["headline"]; v = s["verdict"]
print(f"Coverage@25% (deployable): {h['coverage_25pct_deployable']:.3f}")
print(f"Recall@2k,d=1|nonresident: {h['recall_2k_d1_nonresident']:.3f}")
print(f"HTB(C=25%,m=2k):           {h['htb_c25_m2k']:.3f}")
print(f"VERDICT: {v['label']} -- {v['reason']}")
PY

echo
echo "No batch-8 traces yet -- run this to add the erosion check (§2.3.6), then re-run analyze.py:"
echo "  python3 src/collect.py --model mixtral --quant nf4 --workloads sharegpt,code \\"
echo "    --n-sharegpt 32 --n-code 32 --max-new 64 --seed $SEED \\"
echo "    --out $HERE/results/mixtral/b8 --mode b8 --batch-size 8"
echo
echo "No FP16 fidelity spot-check yet (Pilot.md §3.4/§4.1): NF4 vs FP16 top-2 agreement on 20 prompts."
echo "FP16 Mixtral needs ~87GB; no single box here clears that comfortably even with CPU offload"
echo "(24GB box: 24GB VRAM + 62GB RAM = ~86GB, no headroom for activations/KV cache)."
echo "Recommend skipping and caveating the abstract per Pilot.md §3.4 ('literature suggests >=95% agreement')"
echo "rather than risking an OOM crash a few hours before the deadline."
