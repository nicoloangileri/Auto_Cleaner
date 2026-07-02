"""Excel (.xlsx) ingestion: detection, reading, sheet selection, mislabelling.

Data analysts live in Excel — a cleaner that cannot open a workbook forces a
manual CSV export, which defeats "point it at any messy dataset". Workbooks are
written with polars/xlsxwriter, read back through the public read_any path.
"""

from __future__ import annotations

import shutil

import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.ingest import read_any
from auto_cleaner.ingest.detect import detect_format, profile_source


@pytest.fixture()
def workbook(tmp_path):
    """A two-sheet workbook: 'anagrafica' (first) and 'vendite' (second)."""
    path = tmp_path / "book.xlsx"
    first = pl.DataFrame({"id": [1, 2, 3], "name": ["anna", "luca", "gaia"]})
    second = pl.DataFrame({"month": ["jan", "feb"], "revenue": [1200.5, 980.0]})
    with __import__("xlsxwriter").Workbook(str(path)) as wb:
        first.write_excel(wb, worksheet="anagrafica")
        second.write_excel(wb, worksheet="vendite")
    return path


def test_detect_format_by_extension_and_magic(workbook):
    assert detect_format(workbook) == "excel"
    # Mislabelled: same bytes under a .csv name must still be detected as Excel
    # (xlsx is a ZIP container; magic bytes beat the extension).
    disguised = workbook.with_name("data.csv")
    shutil.copyfile(workbook, disguised)
    assert detect_format(disguised) == "excel"
    assert profile_source(disguised).fmt == "excel"


def test_read_any_reads_first_sheet_and_warns_about_others(workbook):
    df, report = read_any(workbook, CleanConfig(verbose=False))
    assert df.shape == (3, 2)
    assert df["name"].to_list() == ["anna", "luca", "gaia"]
    assert report.metrics["excel_sheet"] == "anagrafica"
    assert any("sheets" in w for w in report.warnings)


def test_read_any_selects_sheet_via_config(workbook):
    cfg = CleanConfig(verbose=False).with_overrides(excel_sheet="vendite")
    df, report = read_any(workbook, cfg)
    assert df.columns == ["month", "revenue"]
    assert report.metrics["excel_sheet"] == "vendite"
    assert not any("sheets" in w for w in report.warnings)  # explicit choice: no warning


def test_read_any_unknown_sheet_fails_loudly(workbook):
    cfg = CleanConfig(verbose=False).with_overrides(excel_sheet="inesistente")
    with pytest.raises(ValueError, match="inesistente"):
        read_any(workbook, cfg)


def test_excel_flows_through_full_pipeline(workbook, tmp_path):
    from auto_cleaner import run_pipeline

    cfg = CleanConfig(verbose=False).with_overrides(
        make_charts=False, make_pdf=False, make_json=False,
        advanced=False, inference=False, modeling=False, fda=False,
        extended_stats=False, forecast=False, auto_specialize=False,
    )
    out = tmp_path / "clean.parquet"
    result = run_pipeline(workbook, out, cfg)
    assert out.exists()
    assert result.frame.height == 3
