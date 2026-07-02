"""Cleaning-impact measurement + executive summary.

The report must quantify how much imputation/outlier treatment distorted each
column (cells changed, mean shift in sd units, KS distance, verdict) and open
with a prose summary an analyst can act on without reading every table.
"""

from __future__ import annotations

import polars as pl

from auto_cleaner import CleanConfig, run_pipeline
from auto_cleaner.clean.impact import measure_impact
from auto_cleaner.eda.summary import build_summary
from auto_cleaner.reporting import StepReport


def _frame(values, name="x"):
    return pl.DataFrame({name: values})


def test_impact_negligible_for_tiny_faithful_fill():
    before = _frame([float(i) for i in range(1, 200)] + [None])
    after = before.with_columns(pl.col("x").fill_null(pl.col("x").median()))
    rep = StepReport(step="impute")
    impacts = measure_impact(before, after, rep)
    assert len(impacts) == 1
    imp = impacts[0]
    assert imp.cells_changed == 1
    assert imp.verdict == "negligible"
    assert not rep.warnings


def test_impact_material_when_a_third_of_column_is_imputed():
    base = [float(i) for i in range(1, 101)]
    before = _frame(base + [None] * 50)
    after = before.with_columns(pl.col("x").fill_null(0.0))  # brutal constant fill
    rep = StepReport(step="impute")
    impacts = measure_impact(before, after, rep)
    imp = impacts[0]
    assert imp.change_share > 0.30
    assert imp.verdict == "material"
    assert any("materially changed 'x'" in w for w in rep.warnings)
    assert rep.metrics["impact"][0]["column"] == "x"


def test_impact_handles_dropped_rows():
    before = _frame([1.0, 2.0, 3.0, 1000.0])
    after = before.filter(pl.col("x") < 100)
    rep = StepReport(step="outliers")
    impacts = measure_impact(before, after, rep)
    assert rep.metrics["rows_dropped"] == 1
    assert impacts and impacts[0].cells_changed == 1


def test_summary_surfaces_material_impact_and_checklist():
    rep = StepReport(step="impute")
    rep.metrics["impact"] = [{
        "column": "price", "cells_changed": 40, "change_share": 0.4,
        "mean_before": 10.0, "mean_after": 6.0, "std_before": 2.0,
        "std_after": 2.5, "mean_shift_sd": 2.0, "ks_stat": 0.4,
        "verdict": "material",
    }]

    class P:  # minimal profile stub
        quality_score = 55
        warnings = ["Feature 'name' is high-cardinality (311 levels)"]

    lines = build_summary(P(), [rep])
    text = "\n".join(lines)
    assert "materially changed `price`" in text
    assert "Review checklist" in text
    assert "55/100" in text
    assert "high-cardinality" in text


def test_report_contains_summary_and_impact_sections(tmp_path):
    df = pl.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 5.0] * 20,
        "b": ["x", "y", "x", None, "y"] * 20,
    })
    src = tmp_path / "in.csv"
    df.write_csv(src)
    cfg = CleanConfig(verbose=False).with_overrides(
        make_charts=False, make_pdf=False, make_json=False,
        advanced=False, inference=False, modeling=False, fda=False,
        extended_stats=False, forecast=False, auto_specialize=False,
    )
    out = tmp_path / "clean.parquet"
    run_pipeline(src, out, cfg)

    md = (tmp_path / "clean_eda.md").read_text()
    assert "## Executive Summary" in md
    assert "TL;DR" in md
    assert "Cleaning impact (impute)" in md
    assert "| a |" in md  # the impact table row for column 'a'

    html = (tmp_path / "clean_eda.html").read_text()
    assert "Executive Summary" in html
    assert "Cleaning impact (impute)" in html
