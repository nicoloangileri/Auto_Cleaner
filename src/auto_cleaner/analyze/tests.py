"""Normality & distribution diagnostics (research-grade).

Runs four complementary normality tests per numeric feature — Shapiro-Wilk,
D'Agostino-Pearson (K²), Jarque-Bera, and Anderson-Darling — so a feature is
judged on agreement rather than a single statistic. ``scipy`` is imported
lazily and every test is individually guarded, so odd columns never crash a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["NormalityResult", "normality_tests"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_MAX_SHAPIRO_N = 5000  # Shapiro-Wilk is unreliable / capped above this


@dataclass(slots=True)
class NormalityResult:
    """Per-feature normality verdict with all four test statistics."""

    feature: str
    n: int
    shapiro_p: float | None
    dagostino_p: float | None
    jarque_bera_p: float | None
    anderson_stat: float | None
    anderson_crit_5pct: float | None
    is_normal: bool


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC_DTYPES and df.get_column(c).n_unique() > 2
    ]


def normality_tests(df: pl.DataFrame, config: CleanConfig | None = None) -> list[NormalityResult]:
    """Return a :class:`NormalityResult` per eligible numeric feature."""
    config = config or CleanConfig()
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return []

    rng = np.random.default_rng(config.random_seed)
    results: list[NormalityResult] = []
    for c in _numeric_columns(df):
        x = df.get_column(c).drop_nulls().to_numpy().astype(float)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n < 8 or np.unique(x).size < 3:
            continue

        sample = x if n <= _MAX_SHAPIRO_N else rng.choice(x, _MAX_SHAPIRO_N, replace=False)

        def _safe(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None

        shapiro_p = _safe(lambda: float(stats.shapiro(sample).pvalue))
        dagostino_p = _safe(lambda: float(stats.normaltest(x).pvalue))
        jb_p = _safe(lambda: float(stats.jarque_bera(x).pvalue))

        def _anderson():
            try:  # SciPy >= 1.17 wants an explicit method (returns a pvalue)
                return stats.anderson(x, dist="norm", method="interpolate")
            except TypeError:  # older SciPy: no `method` parameter
                return stats.anderson(x, dist="norm")

        ad = _safe(_anderson)
        ad_stat = float(ad.statistic) if ad is not None else None
        ad_p = float(ad.pvalue) if ad is not None and hasattr(ad, "pvalue") else None
        ad_crit5 = (
            float(ad.critical_values[2])  # 5% level
            if ad is not None and hasattr(ad, "critical_values") else None
        )

        # Consensus: normal if the majority of available p-tests fail to reject.
        pvals = [p for p in (shapiro_p, dagostino_p, jb_p) if p is not None]
        votes_normal = sum(p > 0.05 for p in pvals)
        n_tests = len(pvals)
        if ad_p is not None:
            votes_normal += int(ad_p > 0.05)
            n_tests += 1
        elif ad_stat is not None and ad_crit5 is not None:
            votes_normal += int(ad_stat < ad_crit5)
            n_tests += 1
        is_normal = n_tests > 0 and votes_normal >= (n_tests / 2.0)

        results.append(
            NormalityResult(
                feature=c, n=n, shapiro_p=shapiro_p, dagostino_p=dagostino_p,
                jarque_bera_p=jb_p, anderson_stat=ad_stat, anderson_crit_5pct=ad_crit5,
                is_normal=is_normal,
            )
        )
    return results
