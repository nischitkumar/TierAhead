"""Trace loading and the request-level train/test split.

Ported from elp_probe/src/analyze.py (the pilot's own loading code) rather
than reimplemented, so nothing here can silently diverge from how the
pilot's committed results/*/b1/traces.jsonl.zst were actually consumed to
produce the numbers in PREDICTED.md's "Verified" section. Pure pandas/numpy
-- no GPU, no torch import at module scope, so importing tierahead.traces never
drags in torch (matters for tierahead.analyze / tierahead.roofline / tierahead.sim,
none of which need a predictor to be installed).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd


def find_data_root(start: Path | None = None) -> Path:
    """Walks upward from `start` (default: this file's location) looking for
    a directory containing both `results/` and `experiments/` -- i.e. the
    Nebula/Code repo root that ships the pilot's committed traces. Falls back
    to the TIERAHEAD_DATA_ROOT env var, then to the current working directory,
    rather than raising -- callers (tests, CLI) decide what to do with a root
    that turns out not to actually hold traces (a clear FileNotFoundError at
    the point of use is more useful than a guess failing silently here)."""
    if "TIERAHEAD_DATA_ROOT" in os.environ:
        return Path(os.environ["TIERAHEAD_DATA_ROOT"]).resolve()
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "results").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    return Path.cwd()


def resolve_trace_dir(trace_dir_default: str, data_root: Path | None = None) -> Path:
    root = data_root or find_data_root()
    return root / trace_dir_default


def load_traces(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no trace file at {path}. Real trace collection needs a CUDA GPU "
            f"(see tierahead.hw.probe().can_collect_traces and README.md's "
            f"'Trace collection' section) -- this repo ships the pilot's "
            f"already-collected traces under results/{{olmoe,mixtral}}/b1/, "
            f"point --data-root at the repo root if this path looks wrong."
        )
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        rows = [json.loads(line) for line in text if line.strip()]
    df = pd.DataFrame(rows)
    df["topk"] = df["topk"].apply(tuple)
    return df


def load_decode_df(trace_dir: Path) -> pd.DataFrame:
    """Convenience: load a b1 trace dir and return decode-phase rows only
    (what roofline/eunion/policy all need -- prefill locality differs and is
    analyzed separately, per Main.md section 4.1)."""
    df = load_traces(Path(trace_dir) / "traces.jsonl.zst")
    return df[df.phase == "decode"].reset_index(drop=True)


def load_req_order(prompts_path: Path) -> dict:
    order = {}
    with open(prompts_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            order[json.loads(line)["req"]] = i
    return order


def train_test_split_reqs(df: pd.DataFrame, seed: int = 0, test_frac: float = 0.3):
    """Split BY REQUEST (not by token) so the same conversation never leaks
    across train/test -- Pilot.md section 1.7's leakage-avoidance design,
    stratified by domain (chat/code) so both splits keep the same domain
    mix. This is the exact split every headline recall/coverage number in
    PREDICTED.md is computed under."""
    rng = np.random.default_rng(seed)
    train_reqs, test_reqs = set(), set()
    meta = df[["req", "domain"]].drop_duplicates()
    for _domain, sub in meta.groupby("domain"):
        reqs = sub["req"].to_numpy().copy()
        rng.shuffle(reqs)
        n_test = max(1, int(round(len(reqs) * test_frac)))
        test_reqs.update(reqs[:n_test])
        train_reqs.update(reqs[n_test:])
    return train_reqs, test_reqs


def explode_topk(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (req, layer, tok, expert) activation."""
    e = df[["req", "layer", "tok", "topk"]].explode("topk").rename(columns={"topk": "expert"})
    e["expert"] = e["expert"].astype(int)
    return e


def expert_counts(df_exploded: pd.DataFrame) -> pd.DataFrame:
    return df_exploded.groupby(["layer", "expert"]).size().rename("count").reset_index()
