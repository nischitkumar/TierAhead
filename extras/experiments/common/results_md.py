"""Lets RESULTS.md's hand-written Analysis section survive re-runs.

roofline.py/batching_sweep.py fully overwrite RESULTS.md on every run (by
design -- the data sections above must always reflect the latest
roofline_summary.json/batching_summary.json, not a stale prior run). But once
someone (Claude or the user) has actually filled in the "## Analysis" section
with real conclusions, blowing it away on the next re-run would be a data-loss
trap. This reads whatever's already under the "## Analysis" heading in the
existing file (if any) and carries it forward verbatim into the new one,
instead of resetting it to the blank placeholder checklist every time.
"""
from pathlib import Path

ANALYSIS_HEADING = "## Analysis"


def preserve_analysis_section(out_path: Path, default_lines: list[str]) -> list[str]:
    """Returns the lines to use for the Analysis section: whatever already
    exists under ANALYSIS_HEADING in out_path if present, else default_lines.
    default_lines should NOT include the heading itself -- this returns the
    heading plus body either way."""
    out_path = Path(out_path)
    if out_path.exists():
        old_text = out_path.read_text()
        idx = old_text.find(ANALYSIS_HEADING)
        if idx != -1:
            existing = old_text[idx:].rstrip("\n")
            return existing.split("\n")
    return [ANALYSIS_HEADING, ""] + default_lines
