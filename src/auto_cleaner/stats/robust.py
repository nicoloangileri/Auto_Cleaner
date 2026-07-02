"""Robust & alternative location estimators per numeric feature.

Goes well beyond mean/median: geometric & harmonic means, 10%-trimmed and
winsorized means, the median absolute deviation (MAD), and a Huber M-estimator
of location — the estimators a statistician reaches for when data are skewed or
contaminated by outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["RobustResult", "robust_summary"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class RobustResult:
    feature: str
    arithmetic_mean: float | None
    geometric_mean: float | None
    harmonic_mean: float | None
    trimmed_mean_10: float | None
    winsorized_mean_10: float | None
    median: float | None
    mad: float | None
    huber_location: float | None


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC and df.get_column(c).n_unique() > 2
    ]


def robust_summary(df: pl.DataFrame, config: CleanConfig | None = None) -> list[RobustResult]:
    """Compute robust location estimators for each eligible numeric feature."""
    config = config or CleanConfig()
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return []

    out: list[RobustResult] = []
    for c in _numeric_columns(df)[:50]:
        x = df.get_column(c).drop_nulls().to_numpy().astype(float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            continue
        all_positive = bool((x > 0).all())

        def _safe(fn):
            try:
                return float(fn())
            except Exception:  # noqa: BLE001
                return None

        huber_loc = None
        try:
            from statsmodels.robust.scale import huber

            huber_loc = float(huber(x)[0])
        except Exception:  # noqa: BLE001
            pass

        out.append(
            RobustResult(
                feature=c,
                arithmetic_mean=float(np.mean(x)),
                geometric_mean=_safe(lambda: stats.gmean(x)) if all_positive else None,
                harmonic_mean=_safe(lambda: stats.hmean(x)) if all_positive else None,
                trimmed_mean_10=_safe(lambda: stats.trim_mean(x, 0.1)),
                winsorized_mean_10=_safe(lambda: stats.mstats.winsorize(x, limits=[0.1, 0.1]).mean()),
                median=float(np.median(x)),
                mad=_safe(lambda: stats.median_abs_deviation(x)),
                huber_location=huber_loc,
            )
        )
    return out
