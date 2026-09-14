"""Model architecture constants shared by E1 (roofline) and E2 (batching sweep).

Values are cross-checked against Experiments.md's own bandwidth table and the
pilot's collect_report.json / pilot_summary.json / mixtral_summary.json
(n_layers, n_experts, k are exactly the pilot's measured values, not guesses).

expert_size_bytes["fp16"] matches Experiments.md's table (§0) verbatim:
  Mixtral: 352 MB/expert.  OLMoE: 12.6 MB/expert.
These are also reproducible from architecture: an expert is a 3-matrix SwiGLU
FFN (gate/up/down), so params/expert = 3 * hidden_size * expert_intermediate_size,
bytes = params * 2 (fp16). See test_roofline.py::test_expert_bytes_matches_table.

NF4 byte factor: QLoRA (Dettmers et al., arXiv:2305.14314) reports ~4.127
bits/param after double quantization (4-bit weights + quantized quant-constants),
vs 16 bits/param for fp16 -> factor = 4.127/16. This is an approximation
(actual overhead depends on block size); flagged as such wherever it's used.
"""
from dataclasses import dataclass, field

NF4_BITS_PER_PARAM = 4.127  # QLoRA arXiv:2305.14314, double-quant overhead included
FP16_BITS_PER_PARAM = 16.0
NF4_BYTE_FACTOR = NF4_BITS_PER_PARAM / FP16_BITS_PER_PARAM  # ~0.258

# KV-cache element widths (E8). Distinct from NF4_BYTE_FACTOR above -- NF4 is a
# *weight*-quantization scheme (blockwise, with a quantized-quant-constant
# overhead) and is not how anyone stores KV-cache activations. KV-cache
# precision reduction in real systems is fp16 -> fp8/int8 (e.g. vLLM's FP8 KV
# cache), so E8 models those two options instead of reusing the NF4 factor.
KV_BYTES_PER_ELEMENT = {"fp16": 2.0, "fp8": 1.0}


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
    num_key_value_heads: int  # E8: attention KV heads (post-GQA count, not query heads)
    head_dim: int  # E8: per-head dimension (hidden_size / num_attention_heads)

    @property
    def params_per_expert(self) -> int:
        # SwiGLU FFN: gate_proj, up_proj (hidden->intermediate), down_proj (intermediate->hidden)
        return 3 * self.hidden_size * self.expert_intermediate_size

    def expert_bytes(self, precision: str) -> float:
        if precision == "fp16":
            return self.params_per_expert * (FP16_BITS_PER_PARAM / 8.0)
        if precision == "nf4":
            return self.params_per_expert * (NF4_BITS_PER_PARAM / 8.0)
        raise ValueError(f"unknown precision {precision!r}, expected 'fp16' or 'nf4'")

    def kv_bytes_per_token(self, precision: str = "fp16") -> float:
        """E8: bytes needed to store ONE token's K+V entries across every
        layer, for one request. Experiments.md E8 step 1's formula
        (`2 * L * n_kv_heads * d_head * precision * context_len`) is this
        quantity times context_len -- see kv_total_bytes(). Derivation: a
        transformer layer's KV cache holds, per token, one K vector and one V
        vector per KV head, each of length head_dim; summed over n_kv_heads
        and doubled for K+V, then summed over all n_layers.
        """
        if precision not in KV_BYTES_PER_ELEMENT:
            raise ValueError(f"unknown KV precision {precision!r}, expected one of {list(KV_BYTES_PER_ELEMENT)}")
        return 2.0 * self.n_layers * self.num_key_value_heads * self.head_dim * KV_BYTES_PER_ELEMENT[precision]

    def kv_total_bytes(self, context_len: int, precision: str = "fp16") -> float:
        """Total KV-cache footprint for one request holding `context_len`
        tokens of history -- Experiments.md E8 step 1's formula exactly."""
        return self.kv_bytes_per_token(precision) * context_len


MODEL_SPECS: dict[str, ModelSpec] = {
    # num_key_value_heads/head_dim verified 2026-09-13 against the live
    # config.json on the HF hub (not guessed): Mixtral uses GQA (32 query
    # heads, 8 KV heads); OLMoE uses plain MHA (16 query heads == 16 KV
    # heads). Both have head_dim = hidden_size / num_attention_heads = 128.
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
}

# Reported in Experiments.md §0's bandwidth table -- used as a cross-check in
# tests, not as the source of truth (ModelSpec.expert_bytes computes it from
# architecture so a spec typo can't silently diverge from the documented number).
EXPERT_BYTES_FP16_TABLE_MB = {"olmoe": 12.6, "mixtral": 352.0}

# Link configs from Experiments.md §0 (E1) / E6's sweep grid, unioned.
LINK_BW_GBPS_GRID = [16, 32, 64, 128]
LINK_BW_LABELS = {16: "sub-CXL2.0x8", 32: "CXL2.0 x8", 64: "CXL2.0 x16 / CXL3.x x8", 128: "CXL3.x multi-link"}

# Residency sweep from the pilot's C-levels (12.5/25/37.5/50) plus the
# endpoints 0% and 75% called out in E6's sweep.
RESIDENCY_PCT_GRID = [0, 12.5, 25, 37.5, 50, 75]

# E8's "realistic context lengths" per Experiments.md E8 step 1.
CONTEXT_LEN_GRID_E8 = [4096, 32768, 131072]
KV_PRECISION_GRID = ["fp16", "fp8"]

# E1's batch points; E2 sweeps the finer grid separately.
BATCH_GRID_E1 = [1, 8, 32]
BATCH_GRID_E2 = [1, 2, 4, 8, 16, 32, 64]

PRECISION_GRID = ["fp16", "nf4"]

# RTX 4060 Ti 16GB dense FP16 tensor-core peak (vendor spec, ~22.1 TFLOPS FP16
# with FP32 accumulate; not sparsity-doubled). Used only for the flop-estimate
# fallback path -- see roofline.py's provenance tagging.
RTX_4060TI_PEAK_FP16_FLOPS = 22.1e12

# Default assumed decode-batch MFU (model FLOPs utilization) for the flop
# estimate. Batch-1/8/32 decode is heavily memory-bound on the compute side
# too (small GEMMs, kernel-launch overhead), so realistic MFU is low --
# this is a documented ASSUMPTION, not a measurement; measure_compute.py's
# calibration is what should override it whenever available.
DEFAULT_DECODE_MFU = 0.08
