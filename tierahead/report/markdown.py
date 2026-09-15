"""Shared Markdown-rendering helpers, generalizing the
render_results_md.py-per-experiment pattern (and the ANALYSIS-section-
preservation trick from experiments/common/results_md.py) into one place
`tierahead report` uses for every subcommand's output, instead of each command
reimplementing its own table formatter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ANALYSIS_HEADING = "## Analysis"


def df_to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "")
    return formatted.to_markdown(index=False)


def preserve_analysis_section(out_path: Path, default_lines: list[str]) -> list[str]:
    """Lets a hand-written '## Analysis' section survive re-runs -- generated
    sections above it always reflect the latest data, but human commentary
    below it should not be silently discarded on the next `tierahead report`."""
    out_path = Path(out_path)
    if out_path.exists():
        old_text = out_path.read_text()
        idx = old_text.find(ANALYSIS_HEADING)
        if idx != -1:
            existing = old_text[idx:].rstrip("\n")
            return existing.split("\n")
    return [ANALYSIS_HEADING, ""] + default_lines


def render_report_md(title: str, sections: list[tuple[str, str]], out_path: Path,
                      default_analysis_lines: list[str] | None = None) -> Path:
    """sections: list of (heading, body_markdown). Appends a preserved (or
    default) Analysis section at the end."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines += [f"## {heading}", "", body, ""]
    analysis = preserve_analysis_section(out_path, default_analysis_lines or ["_(fill in after reviewing)_"])
    lines += analysis
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
