"""Test the reproducible PrivaCI compliance-matrix macro/table generator."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_analyze_privaci_emits_macros_and_table(tmp_path):
    out = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "experiments" / "analyze_privaci.py"),
            "--results",
            str(REPO / "experiments" / "results"),
            "--out",
            str(out),
        ],
        check=True,
    )

    macros = (out / "macros_privaci.tex").read_text(encoding="utf-8")
    for token in (
        "privFullGemini",
        "privFullClaude",
        "privFullGrok",
        "privCacheGemini",
        "privNlGemini",
        "privArtFOneFullGemini",
    ):
        assert token in macros, f"missing macro {token}"
    # Gemini full amortisation is 2.057 in the committed matrix (3 sig figs -> 2.06).
    assert "2.06" in macros

    table = (out / "tab_privaci.tex").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in table
    for model in ("gemini-3.5-flash", "claude-sonnet-4.6", "grok-4.3"):
        assert model in table, f"missing model row {model}"
