#!/usr/bin/env bash
# ELP-Probe execution (Pilot.md §4). One shot: smoke test -> full collection
# -> batch-8 erosion -> analysis -> go/no-go verdict.
#
# GPU target: 1x RTX 4500 Ada 24GB. OLMoE-1B-7B FP16 only (Mixtral out of scope
# here per current run). Model downloads on first use via HF cache -- nothing
# pre-downloaded by setup.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/elp}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

SEED="${SEED:-0}"
N_SHAREGPT="${N_SHAREGPT:-150}"
N_CODE="${N_CODE:-100}"
MAX_NEW="${MAX_NEW:-256}"
B8_N="${B8_N:-64}"          # requests for batch-8 erosion check, must be multiple of batch size
B8_BATCH="${B8_BATCH:-8}"
B8_MAX_NEW="${B8_MAX_NEW:-64}"

B1_DIR="$HERE/results/olmoe/b1"
B8_DIR="$HERE/results/olmoe/b8"

cd "$HERE"

echo "=== [0/4] smoke test: 3 prompts, verify hooks + sanity asserts ==="
python3 src/collect.py --model olmoe --workloads sharegpt --n-sharegpt 3 --n-code 0 \
  --max-new 32 --seed "$SEED" --out "$B1_DIR/_smoke" --smoke 3

echo "=== [1/4] full trace collection: B=1, sharegpt($N_SHAREGPT) + code($N_CODE), max_new=$MAX_NEW ==="
python3 src/collect.py --model olmoe --workloads sharegpt,code \
  --n-sharegpt "$N_SHAREGPT" --n-code "$N_CODE" --max-new "$MAX_NEW" \
  --seed "$SEED" --out "$B1_DIR" --mode b1

echo "=== [2/4] batch-8 erosion check: $B8_N requests, batch=$B8_BATCH, max_new=$B8_MAX_NEW ==="
python3 src/collect.py --model olmoe --workloads sharegpt,code \
  --n-sharegpt $((B8_N / 2)) --n-code $((B8_N / 2)) --max-new "$B8_MAX_NEW" \
  --seed "$SEED" --out "$B8_DIR" --mode b8 --batch-size "$B8_BATCH"

echo "=== [3/4] analysis: Coverage/Gini/Zipf/churn/Recall/HTB/erosion -> pilot_summary.json ==="
python3 src/analyze.py --b1-dir "$B1_DIR" --b8-dir "$B8_DIR" \
  --out "$HERE/results/pilot_summary.json" --fig-dir "$HERE/results/figures" --seed "$SEED"

echo "=== [4/4] verdict ==="
python3 - <<'PY'
import json
s = json.load(open("results/pilot_summary.json"))
h = s["headline"]; v = s["verdict"]
print(f"Coverage@25% (deployable): {h['coverage_25pct_deployable']:.3f}")
print(f"Recall@2k,d=1|nonresident: {h['recall_2k_d1_nonresident']:.3f}")
print(f"HTB(C=25%,m=2k):           {h['htb_c25_m2k']:.3f}")
print(f"VERDICT: {v['label']} -- {v['reason']}")
PY

echo
echo "Done. Numbers: $HERE/results/pilot_summary.json"
echo "Figure:        $HERE/results/figures/coverage_cdf.png"
