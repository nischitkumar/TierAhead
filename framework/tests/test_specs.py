from tierahead.specs.models import EXPERT_BYTES_FP16_TABLE_MB, MODEL_SPECS


def test_expert_bytes_matches_documented_table():
    for tag, mb in EXPERT_BYTES_FP16_TABLE_MB.items():
        spec = MODEL_SPECS[tag]
        computed_mb = spec.expert_bytes("fp16") / 1e6
        assert abs(computed_mb - mb) / mb < 0.02, f"{tag}: computed {computed_mb:.2f}MB vs documented {mb}MB"


def test_nf4_smaller_than_fp16():
    for spec in MODEL_SPECS.values():
        assert spec.expert_bytes("nf4") < spec.expert_bytes("fp16")


def test_kv_bytes_per_token_positive():
    for spec in MODEL_SPECS.values():
        assert spec.kv_bytes_per_token("fp16") > 0
        assert spec.kv_bytes_per_token("fp8") == spec.kv_bytes_per_token("fp16") / 2


def test_olmoe_mixtral_kv_bytes_per_token_coincidence():
    """Documented architectural coincidence (E8_KV_COTENANCY.md section 2.4):
    16 layers * 16 KV heads == 32 layers * 8 KV heads (GQA) == 256, so both
    ship models land on the identical per-token KV cost despite very
    different architectures."""
    olmoe, mixtral = MODEL_SPECS["olmoe"], MODEL_SPECS["mixtral"]
    assert olmoe.kv_bytes_per_token("fp16") == mixtral.kv_bytes_per_token("fp16")


def test_unknown_precision_raises():
    spec = MODEL_SPECS["olmoe"]
    try:
        spec.expert_bytes("int8")
        assert False, "should have raised"
    except ValueError:
        pass
