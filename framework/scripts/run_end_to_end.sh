#!/usr/bin/env bash
# One-shot end-to-end run: hardware doctor -> unit+integration tests ->
# roofline -> validation gates -> baseline (normal) vs CXL-tiered comparison
# for both shipped models. No GPU needed -- everything here runs on the
# pilot's already-committed traces.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ ! -d .venv ]; then
  echo "=== [0/6] no .venv found -- running setup ==="
  make setup
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== [1/6] hardware capability report ==="
python3 -m tiermoe.cli doctor
echo

echo "=== [2/6] unit + integration tests (real committed traces) ==="
python3 -m pytest tests/ -q
echo

echo "=== [3/6] roofline (both models) ==="
python3 -m tiermoe.cli roofline --model olmoe
python3 -m tiermoe.cli roofline --model mixtral
echo

echo "=== [4/6] E6 validation gates ==="
python3 -m tiermoe.cli validate --model olmoe
python3 -m tiermoe.cli validate --model mixtral
echo

echo "=== [5/6] baseline (normal, HBM-only) vs CXL-tiered -- OLMoE ==="
python3 -m tiermoe.cli baseline --model olmoe --residency-pct 25 --bw-gbps 32 --precision fp16
echo

echo "=== [6/6] baseline (normal, HBM-only) vs CXL-tiered -- Mixtral (NF4) ==="
python3 -m tiermoe.cli baseline --model mixtral --residency-pct 25 --bw-gbps 32 --precision nf4
echo

echo "Done. See README.md for the dashboard (tiermoe dashboard) and PREDICTED.md"
echo "for how these numbers extend to what hasn't been run yet."
