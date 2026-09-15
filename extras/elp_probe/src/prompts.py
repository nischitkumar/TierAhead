"""Prompt-set builders for ELP-Probe (Pilot.md §2.1, §3.2 prompts note).

ShareGPT: first human turn, dedup, length 50-500 chars, seeded sample.
Code: HumanEval + MBPP prompts, split roughly evenly to reach n.

Every call also writes the exact selected prompt list to disk (reproducibility
line, Pilot.md §3.2 / §7.1 threats table).
"""
import hashlib
import json
import random
from pathlib import Path


def _dedup_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()


def build_sharegpt_prompts(n: int, seed: int = 0):
    from datasets import load_dataset

    ds = None
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    except Exception:
        # some forks ship a raw json file rather than a loadable dataset script
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
            repo_type="dataset",
        )
        ds = load_dataset("json", data_files=path, split="train")

    seen, candidates = set(), []
    for row in ds:
        convs = row.get("conversations") or row.get("conversation") or []
        first_human = next(
            (c.get("value", "") for c in convs if c.get("from") in ("human", "user")), None
        )
        if not first_human:
            continue
        text = first_human.strip()
        if not (50 <= len(text) <= 500):
            continue
        key = _dedup_key(text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(text)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} ShareGPT prompts survived filtering, need {n}")
    chosen = candidates[:n]
    return [{"req": f"sharegpt-{i:04d}", "domain": "chat", "prompt": p} for i, p in enumerate(chosen)]


def build_code_prompts(n: int, seed: int = 0):
    from datasets import load_dataset

    he = load_dataset("openai/openai_humaneval", split="test")
    mbpp = load_dataset("google-research-datasets/mbpp", split="test")

    he_prompts = [r["prompt"] for r in he]
    mbpp_prompts = [r["text"] for r in mbpp]

    rng = random.Random(seed)
    rng.shuffle(he_prompts)
    rng.shuffle(mbpp_prompts)

    n_he = min(len(he_prompts), n // 2)
    n_mbpp = min(len(mbpp_prompts), n - n_he)
    n_he = min(len(he_prompts), n - n_mbpp)  # rebalance if one set short

    pool = [("humaneval", p) for p in he_prompts[:n_he]] + [("mbpp", p) for p in mbpp_prompts[:n_mbpp]]
    rng.shuffle(pool)
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} code prompts available, need {n}")
    pool = pool[:n]
    return [
        {"req": f"code-{i:04d}", "domain": "code", "prompt": p, "source": src}
        for i, (src, p) in enumerate(pool)
    ]


def build_workload(name: str, n: int, seed: int = 0):
    if name == "sharegpt":
        return build_sharegpt_prompts(n, seed)
    if name == "code":
        return build_code_prompts(n, seed)
    raise ValueError(f"unknown workload {name!r}")


def save_prompt_list(prompts, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
