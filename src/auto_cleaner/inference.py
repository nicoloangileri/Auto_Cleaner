"""Formal statistical inference — with guardrails, not blind automation.

Provides bootstrap confidence intervals, automatically-chosen group-comparison
tests, multiple-testing-corrected correlation significance, and OLS/logit
regression (statsmodels). Every result is explicitly framed as **exploratory**:
the module attaches methodological caveats so a human treats these as hypotheses
to validate, never as final causal claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = [
    "CIResult", "GroupTestResult", "CorrSigResult", "RegressionResult",
    "InferenceReport", "run_inference",
]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_CAVEATS = (
    "p-values are exploratory and not corrected for the full garden of forking paths.",
    "Confidence intervals assume i.i.d. sampling; clustered/time-dependent data need block methods.",
    "Associations are not causal — confounders are not controlled here.",
)


@dataclass(slots=True)
class CIResult:
    feature: str
    statistic: str
    point: float
    lo: float
    hi: float


@dataclass(slots=True)
class GroupTestResult:
    value: str
    group: str
    test: str
    statistic: float
    p_value: float
    n_groups: int
    significant: bool
    effect_name: str | None = None
    effect_value: float | None = None


@dataclass(slots=True)
class CorrSigResult:
    a: str
    b: str
    r: float
    p_value: float
    p_adj: float
    significant: bool


@dataclass(slots=True)
class RegressionResult:
    kind: str                      # "OLS" | "Logit"
    target: str
    n: int
    r2: float | None
    terms: list[tuple[str, float, float, float, float, float]] = field(default_factory=list)
    # term = (name, coef, std_err, p_value, ci_lo, ci_hi)
    note: str | None = None


@dataclass(slots=True)
class InferenceReport:
    cis: list[CIResult] = field(default_factory=list)
    group_tests: list[GroupTestResult] = field(default_factory=list)
    corr_sig: list[CorrSigResult] = field(default_factory=list)
    regression: RegressionResult | None = None
    caveats: tuple[str, ...] = _CAVEATS


def _numeric_columns(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2
    ]


def _bootstrap_ci(values, statistic: str, n_boot: int, alpha: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    fn = np.median if statistic == "median" else np.mean
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = fn(x[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(x)), float(lo), float(hi)


def _is_normal(arr, seed: int) -> bool:
    import numpy as np
    from scipy import stats

    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if a.size < 8 or np.unique(a).size < 3:
        return False
    sample = a if a.size <= 5000 else np.random.default_rng(seed).choice(a, 5000, replace=False)
    try:
        return stats.shapiro(sample).pvalue > 0.05
    except Exception:  # noqa: BLE001
        return False


def _cohen_d(a, b):
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return None
    pooled = (((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)) ** 0.5
    return None if pooled == 0 else round(float((a.mean() - b.mean()) / pooled), 3)


def _cliffs_delta(a, b, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size > 1000:
        a = rng.choice(a, 1000, replace=False)
    if b.size > 1000:
        b = rng.choice(b, 1000, replace=False)
    return round(float(np.sign(a[:, None] - b[None, :]).mean()), 3)


def _eta_squared(groups):
    import numpy as np

    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    return None if ss_total == 0 else round(float(ss_between / ss_total), 3)


def _group_test(df: pl.DataFrame, value: str, group: str, seed: int) -> GroupTestResult | None:
    import numpy as np
    from scipy import stats

    sub = df.select([value, group]).drop_nulls()
    groups = [
        sub.filter(pl.col(group) == g).get_column(value).to_numpy().astype(float)
        for g in sub.get_column(group).unique().to_list()
    ]
    groups = [g[np.isfinite(g)] for g in groups if g.size >= 5]
    if len(groups) < 2:
        return None
    all_normal = all(_is_normal(g, seed) for g in groups)
    eff_name = eff_val = None
    if len(groups) == 2:
        a, b = groups[0], groups[1]
        if all_normal:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            name = "Welch t-test"
            eff_name, eff_val = "Cohen's d", _cohen_d(a, b)
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            name = "Mann-Whitney U"
            eff_name, eff_val = "Cliff's delta", _cliffs_delta(a, b, seed)
    else:
        if all_normal:
            stat, p = stats.f_oneway(*groups)
            name = "One-way ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            name = "Kruskal-Wallis"
        eff_name, eff_val = "eta^2", _eta_squared(groups)
    return GroupTestResult(
        value, group, name, float(stat), float(p), len(groups), bool(p < 0.05), eff_name, eff_val
    )


def _corr_significance(df: pl.DataFrame, numeric: list[str]) -> list[CorrSigResult]:
    import numpy as np
    from scipy import stats

    sub = df.select(numeric).drop_nulls()
    if sub.height < 5 or len(numeric) < 2:
        return []
    mat = sub.to_numpy().astype(float)
    pairs, pvals = [], []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            try:
                r, p = stats.pearsonr(mat[:, i], mat[:, j])
            except Exception:  # noqa: BLE001
                continue
            pairs.append((numeric[i], numeric[j], float(r)))
            pvals.append(float(p))
    if not pvals:
        return []
    try:
        from statsmodels.stats.multitest import multipletests

        reject, p_adj, *_ = multipletests(pvals, method="fdr_bh")
    except Exception:  # noqa: BLE001
        p_adj = pvals
        reject = [p < 0.05 for p in pvals]
    out = [
        CorrSigResult(a, b, r, p, float(pa), bool(rj))
        for (a, b, r), p, pa, rj in zip(pairs, pvals, p_adj, reject)
    ]
    return sorted(out, key=lambda x: x.p_adj)


def _regression(df: pl.DataFrame, target: str, exclude: set[str], seed: int) -> RegressionResult | None:
    import numpy as np

    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    y_series = df.get_column(target)
    features = _numeric_columns(df, exclude | {target})
    if not features:
        return None
    if len(features) >= df.height:
        return RegressionResult(
            "OLS", target, df.height, None, [],
            note="p >= n — regression skipped; use regularised models (Ridge/Lasso).",
        )

    sub = df.select(features + [target]).drop_nulls()
    if sub.height < len(features) + 5:
        return None
    X = sm.add_constant(sub.select(features).to_numpy().astype(float))
    y = sub.get_column(target)

    is_binary = y.n_unique() == 2
    try:
        if is_binary:
            if y.dtype == pl.Boolean:
                yb = y.cast(pl.Int64).to_numpy().astype(float)
            elif y.dtype in _NUMERIC:
                yb = y.to_numpy().astype(float)
            else:
                yb = y.cast(pl.Categorical).to_physical().to_numpy().astype(float)
            model = sm.Logit(yb, X).fit(disp=0, maxiter=100)
            kind, r2 = "Logit", float(getattr(model, "prsquared", float("nan")))
        else:
            yv = y.cast(pl.Float64).to_numpy()
            model = sm.OLS(yv, X).fit()
            kind, r2 = "OLS", float(model.rsquared)
    except Exception as exc:  # noqa: BLE001
        return RegressionResult("OLS", target, sub.height, None, [], note=f"regression failed: {exc}")

    names = ["const"] + features
    ci = model.conf_int()
    terms = []
    for i, name in enumerate(names):
        terms.append((
            name, float(model.params[i]), float(model.bse[i]), float(model.pvalues[i]),
            float(ci[i][0]), float(ci[i][1]),
        ))
    terms = [terms[0]] + sorted(terms[1:], key=lambda t: t[3])  # const first, then by p
    return RegressionResult(kind, target, sub.height, r2, terms)


def run_inference(
    df: pl.DataFrame,
    config: CleanConfig | None = None,
    *,
    target: str | None = None,
    id_columns: list[str] | None = None,
) -> InferenceReport:
    """Run the exploratory inference suite (CIs, group tests, correlations, regression)."""
    config = config or CleanConfig()
    target = target if target is not None else config.target
    exclude = set(id_columns or []) | {"is_outlier"}
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return InferenceReport()

    report = InferenceReport()
    numeric = _numeric_columns(df, exclude)

    # Bootstrap CIs (mean) for up to N numeric features
    for c in numeric[:30]:
        res = _bootstrap_ci(
            df.get_column(c).drop_nulls().to_list(), "mean", 2000, 0.05, config.random_seed
        )
        if res is not None:
            report.cis.append(CIResult(c, "mean", *res))

    # Correlation significance with BH correction
    report.corr_sig = _corr_significance(df, numeric[:30])

    # Group comparisons: prefer the target if categorical, else a suitable factor
    group_col = None
    if target and df.get_column(target).n_unique() <= 10 and df.get_column(target).dtype != pl.Float64:
        group_col = target
    else:
        for c, dt in zip(df.columns, df.dtypes):
            if c in exclude or c == target:
                continue
            if dt in (pl.Utf8, pl.Categorical, pl.Boolean) and 2 <= df.get_column(c).n_unique() <= 6:
                group_col = c
                break
    if group_col is not None:
        for value in [c for c in numeric if c != group_col][:6]:
            res = _group_test(df, value, group_col, config.random_seed)
            if res is not None:
                report.group_tests.append(res)

    # Regression (only if a target is provided)
    if target and target in df.columns:
        report.regression = _regression(df, target, exclude, config.random_seed)

    return report
