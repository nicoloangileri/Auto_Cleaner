"""Unit + end-to-end tests for auto_cleaner.

Run with:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from auto_cleaner import CleanConfig, run_pipeline
from auto_cleaner.clean import downcast, handle_outliers, impute_missing, standardize
from auto_cleaner.eda import profile_dataset
from auto_cleaner.ingest.detect import detect_delimiter, detect_format


# --------------------------------------------------------------------------- #
# Ingestion detection
# --------------------------------------------------------------------------- #
def test_detect_delimiter_semicolon():
    assert detect_delimiter("a;b;c\n1;2;3\n4;5;6") == ";"


def test_detect_delimiter_comma():
    assert detect_delimiter("a,b,c\n1,2,3\n4,5,6") == ","


def test_detect_delimiter_pipe():
    assert detect_delimiter("a|b|c\n1|2|3\n4|5|6") == "|"


# --------------------------------------------------------------------------- #
# Downcasting
# --------------------------------------------------------------------------- #
def test_downcast_picks_smallest_types():
    df = pl.DataFrame(
        {
            "small_uint": pl.Series([1, 2, 3, 250], dtype=pl.Int64),
            "signed": pl.Series([-5, 0, 7, 1], dtype=pl.Int64),
            "f": pl.Series([1.5, 2.25, 3.125, 4.0], dtype=pl.Float64),
        }
    )
    out, rep = downcast(df, CleanConfig())
    assert out["small_uint"].dtype == pl.UInt8
    assert out["signed"].dtype == pl.Int8
    assert out["f"].dtype == pl.Float32
    assert rep.metrics["memory_after"] <= rep.metrics["memory_before"]


def test_downcast_refuses_unsafe_float():
    # A magnitude beyond Float32's range must NOT be silently downcast (overflow).
    df = pl.DataFrame({"x": pl.Series([1.0, 1e39, 2.0], dtype=pl.Float64)})
    out, _ = downcast(df, CleanConfig())
    assert out["x"].dtype == pl.Float64

    # And tightening the tolerance blocks a precision-losing downcast.
    # 0.1/0.2/0.3 round-trip through Float32 with ~1e-8 relative error > 1e-9.
    df2 = pl.DataFrame({"y": pl.Series([0.1, 0.2, 0.3], dtype=pl.Float64)})
    out2, _ = downcast(df2, CleanConfig().with_overrides(float32_rel_tolerance=1e-9))
    assert out2["y"].dtype == pl.Float64


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #
def test_impute_numeric_mean_fills_all():
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    out, _ = impute_missing(df, CleanConfig().with_overrides(impute_numeric="mean"))
    assert out["x"].null_count() == 0
    assert out["x"][2] == pytest.approx((1 + 2 + 4) / 3)


def test_impute_categorical_mode():
    df = pl.DataFrame({"c": ["a", "a", "b", None]})
    out, _ = impute_missing(df, CleanConfig())
    assert out["c"].null_count() == 0
    assert out["c"][3] == "a"


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
def test_parse_currency_strings():
    df = pl.DataFrame({"price": ["$1,200.50", "$3,400.00", "$5,000.00", "$2,000.00"]})
    out, _ = standardize(df, CleanConfig())
    assert out["price"].dtype == pl.Float64
    assert out["price"][0] == pytest.approx(1200.50)


def test_parse_percent_strings():
    df = pl.DataFrame({"rate": ["45%", "50%", "55%", "60%"]})
    out, _ = standardize(df, CleanConfig())
    assert out["rate"][0] == pytest.approx(0.45)


def test_parse_iso_dates():
    df = pl.DataFrame({"d": ["2021-01-01", "2021-02-15", "2021-03-20", "2021-04-25"]})
    out, _ = standardize(df, CleanConfig())
    assert out["d"].dtype == pl.Date


def test_strip_whitespace_and_empty_to_null():
    df = pl.DataFrame({"s": ["  hello  ", "world", "   ", "x"]})
    out, _ = standardize(df, CleanConfig())
    assert out["s"][0] == "hello"
    assert out["s"][2] is None


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def _spiky_frame() -> pl.DataFrame:
    return pl.DataFrame({"v": [float(x) for x in range(1, 21)] + [1000.0]})


def test_outliers_flag_adds_column():
    out, rep = handle_outliers(_spiky_frame(), CleanConfig().with_overrides(outlier_action="flag"))
    assert "is_outlier" in out.columns
    assert out["is_outlier"].sum() >= 1
    assert bool(out["is_outlier"][-1]) is True


def test_outliers_cap_reduces_max():
    out, _ = handle_outliers(_spiky_frame(), CleanConfig().with_overrides(outlier_action="cap"))
    assert out["v"].max() < 1000.0


def test_outliers_drop_removes_rows():
    out, _ = handle_outliers(_spiky_frame(), CleanConfig().with_overrides(outlier_action="drop"))
    assert out.height == 20


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def test_profile_detects_collinearity():
    x = pl.Series("x", [float(i) for i in range(50)])
    df = pl.DataFrame({"x": x, "y": x * 2.0 + 1.0, "z": [float((i * 7) % 13) for i in range(50)]})
    profile = profile_dataset(df, CleanConfig())
    pairs = {frozenset((a, b)) for a, b, _ in profile.collinear_pairs}
    assert frozenset(("x", "y")) in pairs


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "id;price;when;cat\n"
        "1;$1,000;2021-01-01;A\n"
        "2;$2,000;2021-01-02;B\n"
        "3;;2021-01-03;A\n"
        "4;$4,000;2021-01-04;A\n"
    )
    out = tmp_path / "clean.parquet"
    result = run_pipeline(raw, out, CleanConfig().with_overrides(verbose=False))

    assert out.exists()
    # Every missing value was imputed.
    assert result.frame.select(pl.all().is_null().sum()).sum_horizontal().item() == 0
    # Datetime + currency parsing happened.
    assert result.frame["when"].dtype == pl.Date
    assert result.frame["price"].dtype in (pl.Float32, pl.Float64)
    # Reports were written.
    assert "html" in result.report_paths and "markdown" in result.report_paths
    for p in result.report_paths.values():
        assert Path(p).exists()


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def test_build_charts_and_offline_html():
    pytest.importorskip("plotly")
    from auto_cleaner.eda import build_charts, charts_to_html
    from auto_cleaner.eda.stats import profile_dataset

    df = pl.DataFrame(
        {
            "x": [float(i) for i in range(40)],
            "y": [float((i * 3) % 7) for i in range(40)],
            "cat": ["A", "B"] * 20,
        }
    )
    profile = profile_dataset(df, CleanConfig())
    charts = build_charts(df, profile, CleanConfig())
    assert len(charts) >= 3  # missingness + histograms + frequency + heatmap ...
    head, body = charts_to_html(charts)
    assert "Plotly" in head and "<div" in body  # offline plotly.js embedded once
