"""Golden-file test: the Markdown report's content is pinned, not just its
non-crashing. A refactor that silently drops a section, renames a verdict or
breaks a table now fails loudly.

Regenerate after an INTENTIONAL report change:

    UPDATE_GOLDEN=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_golden_report.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import polars as pl

from auto_cleaner import CleanConfig, run_pipeline

GOLDEN = Path(__file__).parent / "golden" / "report_fixture.md"


def _fixture_frame() -> pl.DataFrame:
    # Deterministic, small, and dirty enough to exercise summary + impact:
    # nulls (imputation), an outlier (flagging), a categorical, a constant.
    return pl.DataFrame({
        "amount": [10.0, 11.0, 9.5, None, 10.5, 11.5, 9.0, 10.0, 250.0, None] * 3,
        "segment": (["retail", "corp", "retail", "corp", None] * 6),
        "touches": [1, 2, 1, 3, 2, 1, 2, 3, 1, 2] * 3,
    })


def _normalise(md: str) -> str:
    """Strip run-dependent noise: timestamps, absolute paths, versions."""
    lines = []
    for ln in md.splitlines():
        if "*Generated:*" in ln or "Generated:" in ln:
            continue
        ln = re.sub(r"`/[^`]*`", "`<path>`", ln)  # absolute source paths
        lines.append(ln.rstrip())
    return "\n".join(lines).strip() + "\n"


def test_markdown_report_matches_golden(tmp_path):
    src = tmp_path / "fixture.csv"
    _fixture_frame().write_csv(src)

    cfg = CleanConfig(verbose=False).with_overrides(
        make_charts=False, make_pdf=False, make_json=False, save_model=False,
        advanced=False, inference=False, modeling=False, fda=False,
        extended_stats=False, forecast=False, auto_specialize=False,
        random_seed=7,
    )
    run_pipeline(src, tmp_path / "clean.parquet", cfg,
                 title="Golden fixture report")
    got = _normalise((tmp_path / "clean_eda.md").read_text())

    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(got, encoding="utf-8")

    assert GOLDEN.exists(), (
        "golden file missing — run once with UPDATE_GOLDEN=1 to create it"
    )
    expected = GOLDEN.read_text()
    assert got == expected, (
        "report drifted from the golden file; if the change is intentional, "
        "regenerate with UPDATE_GOLDEN=1"
    )
