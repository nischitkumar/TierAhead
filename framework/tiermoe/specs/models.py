"""Model architecture constants.

Ported from experiments/common/model_specs.py (the pilot's own validated
source of truth) rather than re-derived, so the framework's numbers cannot
silently drift from what the pilot/E1-E5/E8 traces were actually measured
against. n_layers/n_experts/k are the pilot's own measured values
(results/{olmoe,mixtral}/b1/collect_report.json); hidden_size/
expert_intermediate_size/num_key_value_heads/head_dim are cross-checked
against each model's live HF config.json (verified 2026-09-13, per
experiments/e8_kv_cotenancy/E8_KV_COTENANCY.md section 2.4).

expert_bytes(precision) is derived from architecture (3-matrix SwiGLU FFN),
not hardcoded -- see tests/test_specs.py::test_expert_bytes_matches_documented_table
for the cross-check against Main.md section 0's own bandwidth table
(Mixtral 352 MB/expert, OLMoE 12.6 MB/expert, both fp16).
"""
from __future__ import annotations

from dataclasses import dataclass

NF4_BITS_PER_PARAM = 4.127  # QLoRA, Dettmers et al. arXiv:2305.14314 (double-quant overhead included)
FP16_BITS_PER_PARAM = 16.0
NF4_BYTE_FACTOR = NF4_BITS_PER_PARAM / FP16_BITS_PER_PARAM  # ~0.258

# KV-cache element widths (bytes/element). Distinct from NF4_BYTE_FACTOR:
# NF4 is a *weight*-quantization scheme (blockwise codes + quantized scale
# factors) never applied to KV-cache activations in practice; real systems
# that shrink KV precision use fp8/int8 (e.g. vLLM's FP8 KV cache).
KV_BYTES_PER_ELEMENT = {"fp16": 2.0, "fp8": 1.0, "int8": 1.0}


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    hf_id: str
    n_layers: int
    n_experts: int
    k: int
    hidden_size: int
    expert_intermediate_size: int
    gated: bool  # requires HF license acceptance + `hf auth login`
    trace_dir_default: str  # relative to repo root
    num_key_value_heads: int
    head_dim: int
    n_shared_experts: int = 0  # DeepSeek/Qwen-style shared experts (always resident, see policy.residency)

    @property
    def params_per_expert(self) -> int:
        # SwiGLU FFN: gate_proj, up_proj (hidden->intermediate), down_proj (intermediate->hidden)
        return 3 * self.hidden_size * self.expert_intermediate_size

    def expert_bytes(self, precision: str) -> float:
        if precision == "fp16":
            return self.params_per_expert * (FP16_BITS_PER_PARAM / 8.0)
        if precision == "nf4":
            return self.params_per_expert * (NF4_BITS_PER_PARAM / 8.0)
        raise ValueError(f"unknown weight precision {precision!r}, expected 'fp16' or 'nf4'")

    def kv_bytes_per_token(self, precision: str = "fp16") -> float:
        """Bytes to store one token's K+V entries across every layer, for
        one request: 2 (K and V) * n_layers * num_key_value_heads * head_dim
        * bytes_per_element. See tiermoe.kv.cotenancy for the derivation."""
        if precision not in KV_BYTES_PER_ELEMENT:
            raise ValueError(f"unknown KV precision {precision!r}, expected one of {list(KV_BYTES_PER_ELEMENT)}")
        return 2.0 * self.n_layers * self.num_key_value_heads * self.head_dim * KV_BYTES_PER_ELEMENT[precision]

    def kv_total_bytes(self, context_len: int, precision: str = "fp16") -> float:
        return self.kv_bytes_per_token(precision) * context_len


MODEL_SPECS: dict[str, ModelSpec] = {
    "olmoe": ModelSpec(
        tag="olmoe",
        hf_id="allenai/OLMoE-1B-7B-0924",
        n_layers=16, n_experts=64, k=8,
        hidden_size=2048, expert_intermediate_size=1024,
        gated=False,
        trace_dir_default="results/olmoe/b1",
        num_key_value_heads=16, head_dim=128,
    ),
    "mixtral": ModelSpec(
        tag="mixtral",
        hf_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        n_layers=32, n_experts=8, k=2,
        hidden_size=4096, expert_intermediate_size=14336,
        gated=True,
        trace_dir_default="results/mixtral/b1",
        num_key_value_heads=8, head_dim=128,
    ),
    # Not yet traced in this repo (Main.md section 8.2 lists these as E11
    # upside, GPU-time permitting). Specs are from each model's published HF
    # config so the roofline/TCO/KV modules can still project numbers for
    # them (clearly tagged "projected, no trace data" -- see PREDICTED.md)
    # without waiting on real trace collection.
    "deepseek_moe_16b": ModelSpec(
        tag="deepseek_moe_16b",
        hf_id="deepseek-ai/deepseek-moe-16b-base",
        n_layers=28, n_experts=64, k=6,
        hidden_size=2048, expert_intermediate_size=1408,
        gated=False,
        trace_dir_default="results/deepseek_moe_16b/b1",
        num_key_value_heads=16, head_dim=128,
        n_shared_experts=2,
    ),
    "qwen1_5_moe_a2_7b": ModelSpec(
        tag="qwen1_5_moe_a2_7b",
        hf_id="Qwen/Qwen1.5-MoE-A2.7B",
        n_layers=24, n_experts=60, k=4,
        hidden_size=2048, expert_intermediate_size=1408,
        gated=False,
        trace_dir_default="results/qwen1_5_moe_a2_7b/b1",
        num_key_value_heads=16, head_dim=128,
        n_shared_experts=4,
    ),
}

# Every model in this dict has NOT had traces collected in this repo yet.
# tiermoe.baseline/analyze/roofline all check this before claiming a
# "measured" provenance tag for these tags.
UNTRACED_MODELS = frozenset({"deepseek_moe_16b", "qwen1_5_moe_a2_7b"})

# Cross-check target for tests/test_specs.py -- Main.md section 0's own table.
EXPERT_BYTES_FP16_TABLE_MB = {"olmoe": 12.6, "mixtral": 352.0}

RESIDENCY_PCT_GRID = [0, 12.5, 25, 37.5, 50, 75]
BATCH_GRID_E1 = [1, 8, 32]
BATCH_GRID_FINE = [1, 2, 4, 8, 16, 32, 64]
PRECISION_GRID = ["fp16", "nf4"]
CONTEXT_LEN_GRID = [4096, 32768, 131072]
KV_PRECISION_GRID = ["fp16", "fp8"]
