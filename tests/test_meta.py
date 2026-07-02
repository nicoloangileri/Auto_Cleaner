"""Tests for auto-specialisation, inference, baseline modelling, FDA, netCDF."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.functional import run_fda
from auto_cleaner.inference import run_inference
from auto_cleaner.ingest import read_any
from auto_cleaner.modeling import run_baseline_models
from auto_cleaner.specialize import detect_specialization


# --- Auto-specialisation ---------------------------------------------------- #
def test_specialize_time_series():
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(50)]
    df = pl.DataFrame({
        "date": dates,
        "v": [float(i) for i in range(50)],
        "w": [float((i * 2) % 5) for i in range(50)],
        "z": [float(i % 7) for i in range(50)],
    })
    spec = detect_specialization(df, CleanConfig())
    assert spec.time_index == "date"
    assert "time-series" in [a for a, _ in spec.archetypes]
    assert "fda" in spec.auto_modules


def test_specialize_geospatial():
    df = pl.DataFrame({
        "latitude": [10.0, 20.0, 30.0, 40.0],
        "longitude": [1.0, 2.0, 3.0, 4.0],
        "val": [1.0, 2.0, 3.5, 4.0],
    })
    spec = detect_specialization(df, CleanConfig())
    assert spec.geo is not None and spec.geo["lat"] == "latitude"
    assert "geospatial" in spec.auto_modules


def test_specialize_text():
    comments = [f"user review {i} about product quality and overall experience here" for i in range(30)]
    df = pl.DataFrame({"comment": comments, "x": [float(i) for i in range(30)]})
    spec = detect_specialization(df, CleanConfig())
    assert "comment" in spec.text_columns
    assert "text" in spec.auto_modules


# --- Inference -------------------------------------------------------------- #
def _xy_frame(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, n)
    y = x * 2.0 + rng.normal(0.0, 0.3, n)
    grp = np.where(x > 0, "A", "B")
    return pl.DataFrame({"x": x, "y": y, "grp": list(grp)})


def test_inference_full():
    pytest.importorskip("scipy")
    pytest.importorskip("statsmodels")
    rep = run_inference(_xy_frame(), CleanConfig(), target="y")
    assert rep.cis                                    # bootstrap CIs computed
    assert rep.group_tests                            # grp drives a comparison
    assert any(c.significant for c in rep.corr_sig)   # x~y is significant
    assert rep.regression is not None and rep.regression.terms


# --- Baseline modelling ----------------------------------------------------- #
def test_modeling_regression():
    rep = run_baseline_models(_xy_frame(), "y", CleanConfig())
    assert rep is not None and rep.task == "regression"
    assert rep.scores and rep.best is not None


def test_modeling_classification():
    df = _xy_frame().with_columns((pl.col("x") > 0).cast(pl.Int8).alias("label"))
    rep = run_baseline_models(df, "label", CleanConfig())
    assert rep is not None and rep.task == "classification"
    assert rep.scores


# --- Functional data analysis ----------------------------------------------- #
def test_fda_runs_on_time_series():
    pytest.importorskip("scipy")
    n = 60
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    t = np.linspace(0.0, 6.28, n)
    df = pl.DataFrame({"date": dates, "a": np.sin(t), "b": np.cos(t), "c": np.sin(2 * t)})
    rep = run_fda(df, "date", CleanConfig())
    assert rep is not None and rep.n_curves == 3 and rep.n_points >= 7
    assert rep.n_modes_90 >= 1


# --- netCDF ingestion ------------------------------------------------------- #
def test_netcdf_ingestion(tmp_path: Path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.Dataset(
        {"temp": (("time", "loc"), np.random.default_rng(0).random((5, 3)))},
        coords={"time": range(5), "loc": range(3)},
    )
    nc = tmp_path / "grid.nc"
    ds.to_netcdf(str(nc))
    df, _ = read_any(nc, CleanConfig().with_overrides(verbose=False))
    assert df.height == 15 and "temp" in df.columns
