"""Known-value tests for the extended-statistics periphery.

These modules emit numbers an analyst will read and repeat; each gets at least
one test with an analytically known answer, not just a smoke test. Heavy
optional dependencies are skipped cleanly when absent.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from auto_cleaner import CleanConfig

CFG = CleanConfig(verbose=False)


# --------------------------------------------------------------------------- #
# robust.py — location estimators
# --------------------------------------------------------------------------- #
def test_robust_estimators_resist_a_gross_outlier():
    from auto_cleaner.stats.robust import robust_summary

    values = [float(i) for i in range(1, 100)] + [10_000.0]  # one wild point
    df = pl.DataFrame({"x": values})
    res = {r.feature: r for r in robust_summary(df, CFG)}["x"]

    assert res.median == pytest.approx(50.5)
    # The outlier drags the arithmetic mean far above every robust estimator.
    assert res.arithmetic_mean > 149
    assert res.trimmed_mean_10 == pytest.approx(50.5, abs=1.0)
    assert res.winsorized_mean_10 == pytest.approx(res.trimmed_mean_10, abs=6.0)
    assert res.mad == pytest.approx(25.0, abs=1.0)  # MAD of 1..99 is 25


# --------------------------------------------------------------------------- #
# associations.py — rank & categorical association
# --------------------------------------------------------------------------- #
def test_spearman_is_one_for_monotone_nonlinear_relation():
    from auto_cleaner.stats.associations import associations

    x = list(range(1, 61))
    df = pl.DataFrame({"x": [float(v) for v in x], "y": [float(v) ** 3 for v in x]})
    rep = associations(df, CFG)
    rho = {(a, b): r for a, b, r, _p in rep.spearman}
    key = ("x", "y") if ("x", "y") in rho else ("y", "x")
    assert rho[key] == pytest.approx(1.0)


def test_cramers_v_detects_perfect_and_null_association():
    from auto_cleaner.stats.associations import associations

    levels = ["a", "b", "c"] * 30
    rng = np.random.default_rng(7)
    df = pl.DataFrame({
        "u": levels,
        "v": levels,                                    # identical → V ≈ 1
        "w": rng.permutation(np.array(levels)).tolist(),  # shuffled → V ≈ 0
    })
    rep = associations(df, CFG)
    v = {(a, b): val for a, b, val in rep.cramers_v}
    assert v.get(("u", "v"), v.get(("v", "u"))) == pytest.approx(1.0, abs=0.05)
    # The shuffled pair has V ≈ 0, below the 0.3 reporting threshold: it must
    # NOT be reported as an association.
    assert ("u", "w") not in v and ("w", "u") not in v


# --------------------------------------------------------------------------- #
# distributions.py — parametric fits
# --------------------------------------------------------------------------- #
def test_distribution_fitting_prefers_normal_for_gaussian_data():
    from auto_cleaner.stats.distributions import fit_distributions

    rng = np.random.default_rng(7)
    df = pl.DataFrame({"g": rng.normal(50.0, 5.0, 800)})
    fits = fit_distributions(df, CFG)
    assert fits, "expected at least one fitted feature"
    fit = fits[0]
    assert fit.best_distribution == "norm"
    assert np.isfinite(fit.aic)
    assert fit.ks_p > 0.01  # the winning fit should not be rejected


# --------------------------------------------------------------------------- #
# survival.py — Kaplan-Meier / Cox (needs lifelines)
# --------------------------------------------------------------------------- #
def test_survival_detects_columns_and_median():
    pytest.importorskip("lifelines")
    from auto_cleaner.stats.survival import survival_analysis

    rng = np.random.default_rng(7)
    n = 200
    duration = rng.exponential(12.0, n).round(1) + 0.1
    event = (rng.random(n) < 0.7).astype(int)
    df = pl.DataFrame({"tenure_months": duration, "churned_event": event})
    rep = survival_analysis(df, CFG)
    assert rep is not None
    assert rep.duration_col == "tenure_months"
    assert rep.n == n and rep.n_events == int(event.sum())
    assert rep.median_survival is not None and rep.median_survival > 0


# --------------------------------------------------------------------------- #
# bayesian.py — Bayes factors (needs pingouin)
# --------------------------------------------------------------------------- #
def test_bayes_factor_finds_obvious_group_difference():
    pytest.importorskip("pingouin")
    from auto_cleaner.stats.bayesian import bayesian_analysis

    rng = np.random.default_rng(7)
    df = pl.DataFrame({
        "group": ["A"] * 60 + ["B"] * 60,
        "value": np.concatenate([rng.normal(0, 1, 60), rng.normal(5, 1, 60)]),
    })
    rep = bayesian_analysis(df, CFG)
    assert rep.factors, "expected a Bayes factor for the binary group"
    assert max(f.bf10 for f in rep.factors) > 30  # very strong evidence


# --------------------------------------------------------------------------- #
# timeseries.py + forecast.py — diagnostics & projection (need statsmodels)
# --------------------------------------------------------------------------- #
def _monthly_series(n=72):
    t = np.arange(n, dtype=float)
    season = 10.0 * np.sin(2 * np.pi * t / 12)
    noise = np.random.default_rng(7).normal(0, 0.5, n)
    dates = [dt.date(2018, 1, 1) + dt.timedelta(days=30 * i) for i in range(n)]
    return pl.DataFrame({"ds": dates, "y": 100.0 + 0.5 * t + season + noise})


def test_timeseries_diagnostics_find_trend():
    pytest.importorskip("statsmodels")
    from auto_cleaner.stats.timeseries import timeseries_analysis

    res = timeseries_analysis(_monthly_series(), "ds", CFG)
    assert res, "expected diagnostics for the numeric series"
    r = res[0]
    assert r.n == 72
    assert r.mk_trend == "increasing"  # Mann-Kendall must see the +0.5/step drift


def test_forecast_projects_ahead_with_intervals():
    pytest.importorskip("statsmodels")
    from auto_cleaner.stats.forecast import forecast_series

    res = forecast_series(_monthly_series(), "ds", CFG)
    assert res, "expected a forecast for the numeric series"
    fc = res[0]
    assert len(fc.forecast) == CFG.forecast_horizon
    assert len(fc.lower) == len(fc.upper) == len(fc.forecast)
    # A rising series must not be forecast to collapse to zero.
    assert fc.forecast[0] > 50
    assert all(lo <= hi for lo, hi in zip(fc.lower, fc.upper))
