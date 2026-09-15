"""Figure generation for E5: recall_vs_depth.pdf and feasible_fetch_size.pdf
(Experiments.md E5 deliverables, named exactly)."""
from pathlib import Path


def plot_recall_vs_depth(model_tag, direct_by_d, chained_by_d, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = sorted(direct_by_d)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ds, [direct_by_d[d] for d in ds], marker="o", label="direct (one model per d)")
    ax.plot(ds, [chained_by_d.get(d, float("nan")) for d in ds], marker="s", linestyle="--",
            label="chained (d=1 model composed d times)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("lookahead depth d")
    ax.set_ylabel("Recall@k, nonresident")
    ax.set_title(f"{model_tag}: direct vs chained lookahead")
    ax.legend()
    path = out_dir / f"recall_vs_depth_{model_tag}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]


def plot_feasible_fetch_size(model_tag, feasible_rows, direct_recall_by_d, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    bws = sorted(set(r["bw_gbps"] for r in feasible_rows))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for bw in bws:
        rows = sorted([r for r in feasible_rows if r["bw_gbps"] == bw], key=lambda r: r["d"])
        ax.plot([r["d"] for r in rows], [r["bytes_fetchable"] / 1e6 for r in rows], marker="o",
                label=f"{bw:g} GB/s")
    if feasible_rows:
        expert_bytes_mb = feasible_rows[0]["expert_bytes"] / 1e6
        ax.axhline(expert_bytes_mb, color="black", linestyle=":", label="1 expert (actual size)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("lookahead depth d")
    ax.set_ylabel("max fetchable bytes in window (MB)")
    ax.set_title(f"{model_tag}: feasible fetch size vs depth")
    ax.legend(fontsize=8)
    path = out_dir / f"feasible_fetch_size_{model_tag}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
