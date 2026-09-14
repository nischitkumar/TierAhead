import pytest

from tierahead.specs import MODEL_SPECS
from tierahead.traces.io import find_data_root, resolve_trace_dir


@pytest.fixture(scope="session")
def data_root():
    root = find_data_root()
    assert (root / "results").is_dir(), (
        f"expected {root}/results to exist (the pilot's committed traces) -- "
        f"is the framework/ directory still inside the Nebula/Code repo root?"
    )
    return root


@pytest.fixture(scope="session")
def olmoe_trace_dir(data_root):
    d = resolve_trace_dir(MODEL_SPECS["olmoe"].trace_dir_default, data_root)
    assert (d / "traces.jsonl.zst").exists(), f"no OLMoE traces at {d}"
    return d


@pytest.fixture(scope="session")
def mixtral_trace_dir(data_root):
    d = resolve_trace_dir(MODEL_SPECS["mixtral"].trace_dir_default, data_root)
    assert (d / "traces.jsonl.zst").exists(), f"no Mixtral traces at {d}"
    return d
