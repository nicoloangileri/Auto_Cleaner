"""Tests for the opt-in modules: Optuna tuning and causal / A-B analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig
from auto_cleaner.causal import causal_analysis
from auto_cleaner.tuning import tune_model


# --- Tuning ----------------------------------------------------------------- #
def test_tuning_returns_result():
    pytest.importorskip("optuna")
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0.0, 1.0, n)
    df = pl.DataFrame({"x": x, "z": rng.normal(0.0, 1.0, n), "y": x * 2.0 + rng.normal(0.0, 0.3, n)})
    res = tune_model(df, "y", CleanConfig().with_overrides(tune_trials=8))
    assert res is not None and res.task == "regression"
    assert res.best_cv >= res.baseline_cv - 0.1   # tuning should not be much worse
    assert res.best_params  # found some configuration


# --- Causal / A-B ----------------------------------------------------------- #
def test_causal_recovers_numeric_effect():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0.0, 1.0, n)
    t = (rng.random(n) < 0.5).astype(int)        # randomised treatment
    y = 1.0 * t + 0.5 * x + rng.normal(0.0, 0.5, n)  # true effect ~ 1.0
    df = pl.DataFrame({"treat": t.tolist(), "x": x, "y": y})
    rep = causal_analysis(df, "treat", "y", CleanConfig())
    assert rep is not None and rep.outcome_kind == "numeric"
    assert 0.5 < rep.effect < 1.5               # recovers the ~1.0 effect
    assert rep.ci_low < rep.effect < rep.ci_high
    assert rep.ipw_ate is not None              # propensity model used the covariate


def test_causal_binary_outcome():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2)
    n = 600
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < (0.3 + 0.2 * t)).astype(int)  # +0.2 risk difference
    df = pl.DataFrame({"treat": t.tolist(), "y": y.tolist()})
    rep = causal_analysis(df, "treat", "y", CleanConfig())
    assert rep is not None and rep.outcome_kind == "binary"
    assert rep.effect > 0 and rep.effect_size_name == "relative risk"


def test_causal_requires_binary_treatment():
    df = pl.DataFrame({"treat": [0, 1, 2, 0, 1, 2], "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]})
    assert causal_analysis(df, "treat", "y", CleanConfig()) is None  # 3-level treatment → skipped
