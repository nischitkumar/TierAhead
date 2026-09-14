"""Thin adapter onto elp_probe/src/analyze.py so E1/E2 reuse the pilot's exact
trace-loading, train/test split, and transition-table code instead of
re-implementing (and risking silent drift from) it.

No GPU or network required -- these traces were already collected by the
pilot's collect.py and live under results/{model}/b1/traces.jsonl.zst.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ELP_SRC = REPO_ROOT / "elp_probe" / "src"
if str(ELP_SRC) not in sys.path:
    sys.path.insert(0, str(ELP_SRC))

from analyze import (  # noqa: E402
    load_traces,
    load_req_order,
    train_test_split_reqs,
    explode_topk,
    expert_counts,
    build_transition_tables,
    bootstrap_ci,
)

__all__ = [
    "REPO_ROOT",
    "load_traces",
    "load_req_order",
    "train_test_split_reqs",
    "explode_topk",
    "expert_counts",
    "build_transition_tables",
    "bootstrap_ci",
]


def load_decode_df(b1_dir: Path):
    """Load a b1 trace dir's decode-phase rows only (what E1/E2 both need)."""
    b1_dir = Path(b1_dir)
    df = load_traces(b1_dir / "traces.jsonl.zst")
    return df[df.phase == "decode"].reset_index(drop=True)
