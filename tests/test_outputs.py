"""Tests for forecasting, drift, data-quality score and JSON export."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig, run_pipeline
from auto_cleaner.drift import compute_drift
from auto_cleaner.eda.stats import profile_dataset
from auto_cleaner.stats.forecast import forecast_series


def _ts(n: int = 60) -> pl.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    t = np.linspace(0.0, 6.28, n)
    return pl.DataFrame({
        "date": dates,
        "a": np.sin(t) + 0.05 * np.arange(n),
        "b": np.cos(t),
        "c": np.sin(2 * t),
    })


# --- Forecasting ------------------------------------------------------------ #
def test_forecast_produces_horizon():
    pytest.importorskip("statsmodels")
    cfg = CleanConfig()
    res = forecast_series(_ts(), "date", cfg)
    assert res
    assert len(res[0].forecast) == cfg.forecast_horizon
    assert len(res[0].lower) == cfg.forecast_horizon


# --- Data-quality score ----------------------------------------------------- #
def test_quality_score_bounds():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "a", "b"]})
    p = profile_dataset(df, CleanConfig())
    assert 0.0 <= p.quality_score <= 100.0
    assert "completeness" in p.quality_components


def test_quality_score_penalises_missing():
    full = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    holey = pl.DataFrame({"x": [1.0, None, None, 4.0]})
    assert profile_dataset(full, CleanConfig()).quality_score > profile_dataset(holey, CleanConfig()).quality_score


# --- Drift ------------------------------------------------------------------ #
def test_drift_detects_shift():
    rng = np.random.default_rng(0)
    a = pl.DataFrame({"x": rng.normal(0.0, 1.0, 300)})
    b = pl.DataFrame({"x": rng.normal(3.0, 1.0, 300)})  # mean-shifted
    rep = compute_drift(a, b, CleanConfig())
    by = {r.feature: r for r in rep.results}
    assert by["x"].level != "stable" and by["x"].psi > 0.25


def test_drift_stable_when_same():
    rng = np.random.default_rng(1)
    a = pl.DataFrame({"x": rng.normal(0.0, 1.0, 4000)})
    b = pl.DataFrame({"x": rng.normal(0.0, 1.0, 4000)})  # same distribution
    rep = compute_drift(a, b, CleanConfig())
    # Same distribution → PSI should stay below the "major drift" threshold.
    assert {r.feature: r for r in rep.results}["x"].psi < 0.25


# --- JSON export + manifest ------------------------------------------------- #
def test_json_results_export(tmp_path: Path):
    raw = tmp_path / "d.csv"
    raw.write_text("a,b\n1,2\n3,4\n5,6\n2,9\n7,1\n4,8\n")
    out = tmp_path / "clean.parquet"
    run_pipeline(raw, out, CleanConfig().with_overrides(verbose=False, make_charts=False, make_pdf=False))
    jpath = tmp_path / "clean_results.json"
    assert jpath.exists()
    data = json.loads(jpath.read_text())
    assert {"manifest", "summary", "columns"} <= set(data)
    assert data["manifest"]["library_versions"]["python"]
    assert 0 <= data["summary"]["quality_score"] <= 100
