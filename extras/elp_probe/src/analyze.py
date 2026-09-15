"""ELP-Probe analysis pipeline (Pilot.md §3.3, §2.3, §2.4).

Deterministic, runs on decoded traces only (no model/GPU needed here).
Order of operations mirrors Pilot.md §3.3 steps 1-8:
  1. load + train/test split (by request, stratified by domain)
  2. Coverage@C (deployable + oracle)
  3. Gini / Zipf
  4. windowed hot-set churn
  5. Recall@m,d transition tables (+ static-popularity & random baselines)
  6. HTB(C,m) grid
  7. batch-8 erosion
  8. pilot_summary.json + go/no-go verdict (§2.4) + CDF figure
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd
from scipy.stats import linregress

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import gini  # noqa: E402


# ---------- loading ----------

def load_traces(path: Path) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        rows = [json.loads(line) for line in text if line.strip()]
    df = pd.DataFrame(rows)
    df["topk"] = df["topk"].apply(tuple)
    return df


def load_req_order(prompts_path: Path):
    order = {}
    with open(prompts_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            order[json.loads(line)["req"]] = i
    return order


def train_test_split_reqs(df: pd.DataFrame, seed: int = 0, test_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    train_reqs, test_reqs = set(), set()
    meta = df[["req", "domain"]].drop_duplicates()
    for domain, sub in meta.groupby("domain"):
        reqs = sub["req"].to_numpy().copy()
        rng.shuffle(reqs)
        n_test = max(1, int(round(len(reqs) * test_frac)))
        test_reqs.update(reqs[:n_test])
        train_reqs.update(reqs[n_test:])
    return train_reqs, test_reqs


# ---------- 2. Coverage@C ----------

def expert_counts(df_exploded: pd.DataFrame) -> pd.DataFrame:
    """df_exploded has one row per (layer, expert) activation. Returns counts per (layer, expert)."""
    return df_exploded.groupby(["layer", "expert"]).size().rename("count").reset_index()


def explode_topk(df: pd.DataFrame) -> pd.DataFrame:
    e = df[["req", "layer", "tok", "topk"]].explode("topk").rename(columns={"topk": "expert"})
    e["expert"] = e["expert"].astype(int)
    return e


def coverage_at_c(train_counts: pd.DataFrame, test_counts: pd.DataFrame, n_experts: int, c_levels):
    """Returns dict: {c_level: {'deployable': x, 'oracle': y}} activation-weighted mean over layers."""
    results = {c: {"deployable": [], "oracle": [], "weight": []} for c in c_levels}
    layers = sorted(set(train_counts.layer) | set(test_counts.layer))
    for layer in layers:
        tr = train_counts[train_counts.layer == layer].set_index("expert")["count"]
        te = test_counts[test_counts.layer == layer].set_index("expert")["count"]
        te_total = te.sum()
        if te_total == 0:
            continue
        tr_rank = tr.reindex(range(n_experts), fill_value=0).sort_values(ascending=False)
        te_rank = te.reindex(range(n_experts), fill_value=0).sort_values(ascending=False)
        te_by_train_rank_order = te.reindex(tr_rank.index, fill_value=0)
        te_oracle_sorted = te_rank  # already sorted desc by test's own counts

        for c in c_levels:
            top_n = max(1, int(np.ceil(n_experts * c / 100.0)))
            deployable = te_by_train_rank_order.iloc[:top_n].sum() / te_total
            oracle = te_oracle_sorted.iloc[:top_n].sum() / te_total
            results[c]["deployable"].append(deployable)
            results[c]["oracle"].append(oracle)
            results[c]["weight"].append(te_total)

    out = {}
    for c in c_levels:
        w = np.array(results[c]["weight"], dtype=np.float64)
        if w.sum() == 0:
            out[c] = {"deployable": 0.0, "oracle": 0.0}
            continue
        dep = np.average(results[c]["deployable"], weights=w)
        ora = np.average(results[c]["oracle"], weights=w)
        out[c] = {"deployable": float(dep), "oracle": float(ora)}
    return out


# ---------- 3. Gini / Zipf ----------

def gini_zipf_per_layer(test_counts: pd.DataFrame):
    out = {}
    for layer, sub in test_counts.groupby("layer"):
        freqs = sub["count"].to_numpy()
        freqs_sorted = np.sort(freqs)[::-1]
        freqs_sorted = freqs_sorted[freqs_sorted > 0]
        g = gini(freqs_sorted)
        if len(freqs_sorted) >= 3:
            rank = np.arange(1, len(freqs_sorted) + 1)
            slope, *_ = linregress(np.log(rank), np.log(freqs_sorted))
            zipf_exp = -slope
        else:
            zipf_exp = float("nan")
        out[int(layer)] = {"gini": float(g), "zipf_exponent": float(zipf_exp)}
    ginis = [v["gini"] for v in out.values()]
    zipfs = [v["zipf_exponent"] for v in out.values() if not np.isnan(v["zipf_exponent"])]
    summary = {"mean_gini": float(np.mean(ginis)) if ginis else 0.0,
               "mean_zipf_exponent": float(np.mean(zipfs)) if zipfs else float("nan")}
    return out, summary


# ---------- 4. windowed churn ----------

def hot_set(counter: dict, mass: float = 0.8):
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    total = sum(c for _, c in items)
    if total == 0:
        return set()
    cum, hs = 0, set()
    for e, c in items:
        hs.add(e)
        cum += c
        if cum / total >= mass:
            break
    return hs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def windowed_churn(df: pd.DataFrame, req_order: dict, window: int = 2000, mass: float = 0.8):
    """Per layer: chronological decode-activation stream (req_order, then tok),
    non-overlapping windows of `window` expert activations, Jaccard churn between
    consecutive windows' 80%-mass hot sets."""
    per_layer = {}
    decode = df[df.phase == "decode"].copy()
    decode["order"] = decode["req"].map(req_order)
    decode = decode.dropna(subset=["order"])
    for layer, sub in decode.groupby("layer"):
        sub = sub.sort_values(["order", "tok"])
        experts_stream = np.concatenate(sub["topk"].apply(np.array).to_numpy())
        n = len(experts_stream)
        n_windows = n // window
        if n_windows < 2:
            continue
        hot_sets = []
        for w in range(n_windows):
            chunk = experts_stream[w * window:(w + 1) * window]
            vals, counts = np.unique(chunk, return_counts=True)
            hot_sets.append(hot_set(dict(zip(vals.tolist(), counts.tolist())), mass))
        churns = [jaccard(hot_sets[i], hot_sets[i + 1]) for i in range(len(hot_sets) - 1)]
        if churns:
            per_layer[int(layer)] = {
                "mean_churn": float(np.mean(churns)),
                "p95_churn": float(np.percentile(churns, 95)),
                "n_windows": n_windows,
            }
    all_means = [v["mean_churn"] for v in per_layer.values()]
    summary = {"mean_churn": float(np.mean(all_means)) if all_means else float("nan"),
               "p95_churn": float(np.percentile(all_means, 95)) if all_means else float("nan")}
    return per_layer, summary


# ---------- 5. Recall@m,d transition tables ----------

def build_transition_tables(df: pd.DataFrame, train_reqs: set, n_experts: int, n_layers: int,
                             d_values, alpha: float = 1.0):
    """P_hat[d][l] = (n_experts, n_experts) array, P_hat[d][l][e, e'] = P(e' active at l+d | e active at l)."""
    decode = df[df.phase == "decode"]
    train = decode[decode.req.isin(train_reqs)]
    tables = {d: {} for d in d_values}

    pivot = train.pivot_table(index=["req", "tok"], columns="layer", values="topk", aggfunc="first")

    for d in d_values:
        for l in range(n_layers - d):
            if l not in pivot.columns or (l + d) not in pivot.columns:
                continue
            pairs = pivot[[l, l + d]].dropna()
            if len(pairs) == 0:
                continue
            k = len(pairs[l].iloc[0])
            s_l_arr = np.stack(pairs[l].apply(np.array).to_numpy())    # (N, k)
            s_ld_arr = np.stack(pairs[l + d].apply(np.array).to_numpy())  # (N, k)

            co = np.full((n_experts, n_experts), alpha, dtype=np.float64)
            e_count = np.full(n_experts, alpha * n_experts, dtype=np.float64)

            # vectorized outer-product co-occurrence: for each row, all k*k (e, e') pairs
            e_idx = np.repeat(s_l_arr, k, axis=1).reshape(-1)   # e repeated k times per row
            ep_idx = np.tile(s_ld_arr, (1, k)).reshape(-1)      # e' tiled k times per row
            np.add.at(co, (e_idx, ep_idx), 1)
            np.add.at(e_count, s_l_arr.reshape(-1), 1)

            p_hat = co / e_count[:, None]  # P(e' | e)
            tables[d][l] = p_hat
    return tables


def recall_at_m(df: pd.DataFrame, test_reqs: set, tables: dict, train_counts: pd.DataFrame,
                 n_experts: int, k: int, n_layers: int, d_values, m_mults, c_for_nonresident: float = 25.0):
    decode = df[df.phase == "decode"]
    test = decode[decode.req.isin(test_reqs)]
    pivot = test.pivot_table(index=["req", "tok"], columns="layer", values="topk", aggfunc="first")

    # resident set per layer under C = c_for_nonresident, from train ranking
    resident = {}
    for layer, sub in train_counts.groupby("layer"):
        cnt = sub.set_index("expert")["count"].reindex(range(n_experts), fill_value=0)
        top_n = max(1, int(np.ceil(n_experts * c_for_nonresident / 100.0)))
        resident[int(layer)] = set(cnt.sort_values(ascending=False).index[:top_n].tolist())

    # global (unconditioned) test-split popularity per layer, for the static-popularity baseline
    pop_rank = {}
    for layer, sub in train_counts.groupby("layer"):
        cnt = sub.set_index("expert")["count"].reindex(range(n_experts), fill_value=0)
        pop_rank[int(layer)] = cnt.sort_values(ascending=False).index.to_numpy()

    ms = [(mult * k, name) for mult, name in m_mults]
    results = {}
    for d in d_values:
        # accumulators keyed by m_name
        acc = {name: {"hit_o": 0, "tot_o": 0, "hit_n": 0, "tot_n": 0, "hit_p": 0, "tot_p": 0}
               for _, name in ms}

        for l in range(n_layers - d):
            if l not in tables[d] or l not in pivot.columns or (l + d) not in pivot.columns:
                continue
            log_p = np.log(tables[d][l])
            pairs = pivot[[l, l + d]].dropna()
            res_l_d = resident.get(l + d, set())
            pop_order = pop_rank.get(l + d, np.arange(n_experts))
            pop_top_by_m = {name: set(pop_order[:m].tolist()) for m, name in ms}

            for s_l, s_ld in zip(pairs[l], pairs[l + d]):
                scores = log_p[list(s_l), :].sum(axis=0)
                order = np.argsort(-scores)  # full ranking, computed once
                actual = set(s_ld)
                nonres_actual = actual - res_l_d

                for m, name in ms:
                    top_m_idx = set(order[:m].tolist())
                    a = acc[name]
                    a["hit_o"] += len(top_m_idx & actual)
                    a["tot_o"] += len(actual)
                    a["hit_n"] += len(top_m_idx & nonres_actual)
                    a["tot_n"] += len(nonres_actual)
                    a["hit_p"] += len(pop_top_by_m[name] & actual)
                    a["tot_p"] += len(actual)

        for m, name in ms:
            a = acc[name]
            results[f"d{d}_m{name}"] = {
                "d": d, "m": m,
                "recall_overall": a["hit_o"] / a["tot_o"] if a["tot_o"] else float("nan"),
                "recall_nonresident": a["hit_n"] / a["tot_n"] if a["tot_n"] else float("nan"),
                "baseline_static_popularity": a["hit_p"] / a["tot_p"] if a["tot_p"] else float("nan"),
                "baseline_random": m / n_experts,
            }
    return results


# ---------- 6. HTB grid ----------

def htb_grid(coverage: dict, recall_nonres_by_m: dict, c_levels, m_defs):
    grid = {}
    for c in c_levels:
        cov = coverage[c]["deployable"]
        for m_name in m_defs:
            key = f"d1_m{m_name}"
            if key not in recall_nonres_by_m:
                continue
            r = recall_nonres_by_m[key]["recall_nonresident"]
            if np.isnan(r):
                r = 0.0
            htb = cov + (1 - cov) * r
            grid[f"C{c}_m{m_name}"] = {"coverage": cov, "recall_nonresident_d1": r, "htb": htb}
    return grid


# ---------- 7. batch-8 erosion ----------

def batch8_erosion(b8_df: pd.DataFrame, n_experts: int, k: int):
    decode = b8_df[b8_df.phase == "decode"]
    per_step = decode.groupby(["layer", "batch_id", "tok"])["topk"].apply(
        lambda col: len(set().union(*col.apply(set)))
    )
    mean_distinct = per_step.groupby("layer").mean()
    b1_baseline = k / n_experts
    out = {
        "mean_distinct_experts_per_step": {int(l): float(v) for l, v in mean_distinct.items()},
        "mean_fraction_of_experts_b8": float((mean_distinct / n_experts).mean()),
        "fraction_of_experts_b1": float(b1_baseline),
    }
    return out


# ---------- 8. bootstrap CI ----------

def bootstrap_ci(values_per_req, n_resamples=1000, seed=0):
    """values_per_req: list of (hit, total) pairs at request granularity. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    hits = np.array([h for h, t in values_per_req])
    tots = np.array([t for h, t in values_per_req])
    n = len(hits)
    if n == 0 or tots.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = hits.sum() / tots.sum()
    boots = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        h, t = hits[idx].sum(), tots[idx].sum()
        boots.append(h / t if t else 0.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def coverage_bootstrap(test_counts: pd.DataFrame, ranking_index_per_layer, n_experts: int, c: float, req_layer_counts):
    """req_layer_counts: dict req -> DataFrame-like per-(layer,expert) counts, for resampling by request."""
    pairs = []
    for req, counts in req_layer_counts.items():
        hit, tot = 0, 0
        for layer, sub in counts.groupby("layer"):
            if layer not in ranking_index_per_layer:
                continue
            top_set = ranking_index_per_layer[layer]
            c_sub = sub.set_index("expert")["count"]
            hit += c_sub[c_sub.index.isin(top_set)].sum()
            tot += c_sub.sum()
        pairs.append((hit, tot))
    return pairs


def verdict(coverage25_deployable, recall_2k_d1_nonresident, htb):
    if coverage25_deployable >= 0.60 and recall_2k_d1_nonresident >= 0.60:
        return "STRONG_GO", f"Coverage@25%={coverage25_deployable:.2f} & Recall@2k,d=1|nonres={recall_2k_d1_nonresident:.2f} both >=0.60 -> HTB~{htb:.2f}"
    if coverage25_deployable >= 0.60 and recall_2k_d1_nonresident < 0.60:
        return "GO_PLACEMENT_LED", f"Coverage@25%={coverage25_deployable:.2f}>=0.60 but Recall={recall_2k_d1_nonresident:.2f}<0.60"
    if coverage25_deployable < 0.40 and recall_2k_d1_nonresident >= 0.60:
        return "GO_PREFETCH_LED", f"Coverage@25%={coverage25_deployable:.2f}<0.40 but Recall={recall_2k_d1_nonresident:.2f}>=0.60"
    return "PIVOT", f"Coverage@25%={coverage25_deployable:.2f}, Recall={recall_2k_d1_nonresident:.2f} -- neither threshold met"


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-dir", default="results/olmoe/b1")
    ap.add_argument("--b8-dir", default="results/olmoe/b8")
    ap.add_argument("--out", default="results/pilot_summary.json")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--hotset-mass", type=float, default=0.8)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--c-levels", default="12.5,25,37.5,50")
    ap.add_argument("--d-values", default="1,2,4")
    args = ap.parse_args()

    b1_dir = Path(args.b1_dir)
    c_levels = [float(x) for x in args.c_levels.split(",")]
    d_values = [int(x) for x in args.d_values.split(",")]

    print("[analyze] loading traces...", flush=True)
    df = load_traces(b1_dir / "traces.jsonl.zst")
    req_order = load_req_order(b1_dir / "prompts_used.jsonl")
    n_layers = int(df["layer"].max()) + 1
    n_experts = int(pd.Series([e for tk in df["topk"].head(2000) for e in tk]).max()) + 1
    k = len(df["topk"].iloc[0])
    print(f"[analyze] n_layers={n_layers} k={k} n_experts(observed>=)={n_experts} rows={len(df)}", flush=True)

    train_reqs, test_reqs = train_test_split_reqs(df, seed=args.seed)
    print(f"[analyze] train reqs={len(train_reqs)} test reqs={len(test_reqs)}", flush=True)

    decode = df[df.phase == "decode"]
    exploded = explode_topk(decode)
    n_experts = int(max(n_experts, exploded["expert"].max() + 1))

    train_exp = exploded[exploded.req.isin(train_reqs)]
    test_exp = exploded[exploded.req.isin(test_reqs)]
    train_counts = expert_counts(train_exp)
    test_counts = expert_counts(test_exp)

    print("[analyze] Coverage@C ...", flush=True)
    coverage = coverage_at_c(train_counts, test_counts, n_experts, c_levels)

    print("[analyze] Gini / Zipf ...", flush=True)
    gz_per_layer, gz_summary = gini_zipf_per_layer(test_counts)

    print("[analyze] windowed churn ...", flush=True)
    churn_per_layer, churn_summary = windowed_churn(df, req_order, window=args.window, mass=args.hotset_mass)

    print("[analyze] fitting transition tables ...", flush=True)
    tables = build_transition_tables(df, train_reqs, n_experts, n_layers, d_values)

    print("[analyze] Recall@m,d ...", flush=True)
    m_mults = [(1, "k"), (2, "2k"), (4, "4k")]
    recall = recall_at_m(df, test_reqs, tables, train_counts, n_experts, k, n_layers, d_values, m_mults)

    print("[analyze] HTB grid ...", flush=True)
    htb = htb_grid(coverage, recall, c_levels, [name for _, name in m_mults])

    headline_cov = coverage[25.0]["deployable"] if 25.0 in coverage else list(coverage.values())[0]["deployable"]
    headline_recall_key = "d1_m2k"
    headline_recall = recall.get(headline_recall_key, {}).get("recall_nonresident", float("nan"))
    htb_entry = htb.get("C25.0_m2k") or (next(iter(htb.values())) if htb else {})
    headline_htb = htb_entry.get("htb", float("nan"))

    v_label, v_reason = verdict(headline_cov, headline_recall if not np.isnan(headline_recall) else 0.0, headline_htb if not np.isnan(headline_htb) else 0.0)
    print(f"[analyze] VERDICT: {v_label} -- {v_reason}", flush=True)

    b8_path = Path(args.b8_dir) / "traces.jsonl.zst"
    erosion = None
    if b8_path.exists():
        print("[analyze] batch-8 erosion ...", flush=True)
        b8_df = load_traces(b8_path)
        erosion = batch8_erosion(b8_df, n_experts, k)
    else:
        print(f"[analyze] no batch-8 traces at {b8_path}, skipping erosion", flush=True)

    print("[analyze] bootstrap CIs ...", flush=True)
    top_n_25 = max(1, int(np.ceil(n_experts * 25.0 / 100.0)))
    ranking_index_per_layer = {}
    for layer, sub in train_counts.groupby("layer"):
        top_experts = sub.set_index("expert")["count"].sort_values(ascending=False).index[:top_n_25]
        ranking_index_per_layer[int(layer)] = set(top_experts.tolist())
    req_layer_counts = {req: sub.groupby(["layer", "expert"]).size().rename("count").reset_index()
                         for req, sub in test_exp.groupby("req")}
    cov_pairs = coverage_bootstrap(test_counts, ranking_index_per_layer, n_experts, 25.0, req_layer_counts)
    cov_point, cov_lo, cov_hi = bootstrap_ci(cov_pairs, n_resamples=args.bootstrap, seed=args.seed)

    summary = {
        "model": "olmoe-1b-7b",
        "n_layers": n_layers, "n_experts": n_experts, "k": k,
        "n_train_reqs": len(train_reqs), "n_test_reqs": len(test_reqs),
        "coverage_at_c": coverage,
        "coverage_25pct_bootstrap_ci95": {"point": cov_point, "lo": cov_lo, "hi": cov_hi},
        "gini_zipf": {"per_layer": gz_per_layer, "summary": gz_summary},
        "churn": {"per_layer": churn_per_layer, "summary": churn_summary,
                  "window": args.window, "hotset_mass": args.hotset_mass},
        "recall": recall,
        "htb_grid": htb,
        "batch8_erosion": erosion,
        "headline": {
            "coverage_25pct_deployable": headline_cov,
            "recall_2k_d1_nonresident": headline_recall,
            "htb_c25_m2k": headline_htb,
        },
        "verdict": {"label": v_label, "reason": v_reason},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[analyze] wrote {out_path}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        cs = sorted(coverage.keys())
        dep = [coverage[c]["deployable"] for c in cs]
        ora = [coverage[c]["oracle"] for c in cs]
        plt.figure(figsize=(5, 4))
        plt.plot(cs, dep, marker="o", label="deployable (train-ranked)")
        plt.plot(cs, ora, marker="o", linestyle="--", label="oracle (test-ranked)")
        plt.xlabel("Top-C% of experts (HBM-resident fraction)")
        plt.ylabel("Coverage of decode-time activations")
        plt.title("OLMoE-1B-7B: Coverage@C")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "coverage_cdf.png", dpi=150)
        print(f"[analyze] wrote {fig_dir / 'coverage_cdf.png'}", flush=True)
    except Exception as e:
        print(f"[analyze] plotting skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
