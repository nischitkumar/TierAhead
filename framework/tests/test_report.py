import pandas as pd

from tiermoe.report.markdown import df_to_markdown_table, preserve_analysis_section, render_report_md


def test_df_to_markdown_table_empty():
    assert df_to_markdown_table(pd.DataFrame()) == "_(no rows)_"


def test_df_to_markdown_table_formats_floats():
    df = pd.DataFrame({"a": [1.23456], "b": ["x"]})
    md = df_to_markdown_table(df)
    assert "1.2346" in md
    assert "x" in md


def test_preserve_analysis_section_defaults_when_no_file(tmp_path):
    lines = preserve_analysis_section(tmp_path / "nope.md", ["- default line"])
    assert lines == ["## Analysis", "", "- default line"]


def test_preserve_analysis_section_keeps_hand_written_content(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("# Title\n\n## Analysis\n\n- a real finding I wrote by hand\n")
    lines = preserve_analysis_section(p, ["- default"])
    assert "- a real finding I wrote by hand" in "\n".join(lines)
    assert "- default" not in "\n".join(lines)


def test_render_report_md_writes_file_and_preserves_analysis_on_rerun(tmp_path):
    out = tmp_path / "report.md"
    render_report_md("Test Report", [("Section A", "body A")], out, default_analysis_lines=["- todo"])
    text = out.read_text()
    assert "# Test Report" in text and "## Section A" in text and "body A" in text and "- todo" in text

    # simulate a human editing the Analysis section, then a re-run
    edited = text.replace("- todo", "- reviewed, looks correct")
    out.write_text(edited)
    render_report_md("Test Report", [("Section A", "body A2")], out, default_analysis_lines=["- todo"])
    text2 = out.read_text()
    assert "body A2" in text2  # data section updated
    assert "- reviewed, looks correct" in text2  # human analysis preserved
