"""Tests for the advanced analysis layer, FITS ingestion and streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.analyze import (
    feature_relevance,
    normality_tests,
    pca_summary,
    run_advanced,
    suggest_transforms,
    vif_scores,
)
from auto_cleaner.ingest import read_any


def _frame(n: int = 300) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, n)
    skewed = rng.exponential(2.0, n)          # right-skewed, strictly positive
    y = x * 1.5 + rng.normal(0.0, 0.1, n)     # almost-perfectly collinear with x
    cat = np.where(x > 0, "hi", "lo")
    return pl.DataFrame({"x": x, "skewed": skewed, "y": y, "cat": list(cat)})


# --- Normality -------------------------------------------------------------- #
def test_normality_flags_nonnormal():
    pytest.importorskip("scipy")
    by = {r.feature: r for r in normality_tests(_frame(), CleanConfig())}
    assert by["skewed"].is_normal is False     # exponential is clearly non-normal
    assert by["x"].is_normal is True           # standard normal


# --- Transforms ------------------------------------------------------------- #
def test_suggest_transform_reduces_skew():
    pytest.importorskip("scipy")
    sugg = {s.feature: s for s in suggest_transforms(_frame(), CleanConfig())}
    assert "skewed" in sugg
    assert abs(sugg["skewed"].skew_after) < abs(sugg["skewed"].skew_before)


# --- Multicollinearity ------------------------------------------------------ #
def test_vif_detects_collinearity():
    vifs = {v.feature: v.vif for v in vif_scores(_frame(), CleanConfig())}
    assert vifs["x"] > 5 and vifs["y"] > 5     # x and y are collinear
    assert vifs["skewed"] < 5                  # independent feature


def test_pca_summary_returns_components():
    res = pca_summary(_frame(), CleanConfig())
    assert res is not None and res.n_components_90 >= 1
    assert abs(sum(res.explained_variance_ratio) - 1.0) < 1e-6


# --- Feature relevance ------------------------------------------------------ #
def test_feature_relevance_ranks_strongest_first():
    pytest.importorskip("sklearn")
    results, task = feature_relevance(_frame(), "y", CleanConfig())
    assert task == "regression"
    assert results[0].feature == "x"           # x is the strongest predictor of y


def test_run_advanced_aggregates_all():
    pytest.importorskip("scipy")
    adv = run_advanced(_frame(), CleanConfig(), target="y")
    assert adv.normality and adv.vif and adv.pca is not None
    assert adv.relevance and adv.task_type == "regression"


# --- FITS ingestion --------------------------------------------------------- #
def test_fits_ingestion(tmp_path: Path):
    pytest.importorskip("astropy")
    from astropy.table import Table

    fits_path = tmp_path / "t.fits"
    Table({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]}).write(str(fits_path), format="fits")
    df, _ = read_any(fits_path, CleanConfig().with_overrides(verbose=False))
    assert df.shape == (3, 2)
    assert set(df.columns) == {"a", "b"}


# --- Streaming ingestion ---------------------------------------------------- #
def test_streaming_ingestion(tmp_path: Path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
    df, _ = read_any(csv_path, CleanConfig().with_overrides(streaming=True, verbose=False))
    assert df.shape == (3, 2)
