"""Tests for the extended statistics suite + validation hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.stats import run_extended
from auto_cleaner.stats.associations import associations
from auto_cleaner.stats.distributions import fit_distributions
from auto_cleaner.stats.multivariate import multivariate_analysis
from auto_cleaner.stats.robust import robust_summary
from auto_cleaner.validate import (
    IngestionError,
    ValidationError,
    validate_frame,
    validate_source,
)


def _frame(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, n)
    y = x * 2.0 + rng.normal(0.0, 0.3, n)
    z = rng.exponential(2.0, n)  # positive & skewed
    cat = np.where(x > 0, "A", "B")
    return pl.DataFrame({"x": x, "y": y, "z": z, "cat": list(cat)})


# --- Robust stats ----------------------------------------------------------- #
def test_robust_means():
    pytest.importorskip("scipy")
    by = {r.feature: r for r in robust_summary(_frame(), CleanConfig())}
    assert by["z"].geometric_mean is not None      # positive data → geom mean defined
    assert by["x"].geometric_mean is None          # has negatives → undefined
    assert by["x"].mad is not None and by["x"].trimmed_mean_10 is not None


# --- Associations ----------------------------------------------------------- #
def test_associations_rank_correlation():
    pytest.importorskip("scipy")
    rep = associations(_frame(), CleanConfig())
    pairs = {frozenset((a, b)) for a, b, _, _ in rep.spearman}
    assert frozenset(("x", "y")) in pairs


# --- Distributions ---------------------------------------------------------- #
def test_distribution_fitting():
    pytest.importorskip("scipy")
    by = {d.feature: d for d in fit_distributions(_frame(), CleanConfig())}
    assert "z" in by and by["z"].best_distribution
    assert by["z"].aic == min(c[1] for c in by["z"].candidates)  # best == lowest AIC


# --- Multivariate ----------------------------------------------------------- #
def test_multivariate_clustering():
    res = multivariate_analysis(_frame(), CleanConfig())
    assert res is not None
    assert res.best_k is not None and res.mahalanobis_outliers is not None


# --- Orchestrator ----------------------------------------------------------- #
def test_run_extended_aggregates():
    pytest.importorskip("scipy")
    ext = run_extended(_frame(), CleanConfig(), spec=None, target="y")
    assert ext.robust and ext.associations is not None and ext.distributions


# --- Validation hardening --------------------------------------------------- #
def test_validate_empty_frame_raises():
    with pytest.raises(ValidationError):
        validate_frame(pl.DataFrame(), CleanConfig())


def test_validate_flags_all_null_column():
    df = pl.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    report = validate_frame(df, CleanConfig())
    assert any("entirely null" in i for i in report.issues)


def test_validate_source_missing_raises(tmp_path: Path):
    with pytest.raises(IngestionError):
        validate_source(tmp_path / "does_not_exist.csv")


def test_validate_source_empty_raises(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(IngestionError):
        validate_source(empty)
