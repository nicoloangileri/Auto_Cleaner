"""Outlier detection — univariate (IQR, Z-score) and multivariate (Isolation Forest).

Each selected method contributes to a row-level outlier mask. The configured
*action* then decides the fate of flagged values:

* ``flag`` — append a boolean ``is_outlier`` column, change nothing else;
* ``cap``  — winsorize numeric columns to their univariate bounds;
* ``drop`` — remove any row flagged by any selected method;
* ``none`` — detect & report only.

Isolation Forest is multivariate, so it can only *flag* or *drop* (never cap).
``scikit-learn`` is an optional dependency; if absent, IF degrades gracefully.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.logging_utils import log
from auto_cleaner.reporting import StepReport

__all__ = ["handle_outliers", "OUTLIER_FLAG_COLUMN"]

OUTLIER_FLAG_COLUMN = "is_outlier"

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def _candidate_columns(df: pl.DataFrame) -> list[str]:
    """Numeric columns excluding (near-)binary indicators to avoid false flags."""
    cols = []
    for c, dt in zip(df.columns, df.dtypes):
        if dt not in _NUMERIC_DTYPES:
            continue
        if df.get_column(c).n_unique() <= 2:
            continue
        cols.append(c)
    return cols


def _iqr_bounds(series: pl.Series, multiplier: float) -> tuple[float, float] | None:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    if q1 is None or q3 is None:
        return None
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def _zscore_bounds(series: pl.Series, threshold: float) -> tuple[float, float] | None:
    mu, sd = series.mean(), series.std()
    if mu is None or sd is None or sd == 0:
        return None
    return mu - threshold * sd, mu + threshold * sd


def _isolation_forest_mask(
    df: pl.DataFrame, cols: list[str], config: CleanConfig, report: StepReport
) -> np.ndarray | None:
    if len(cols) < 2:
        report.warn("Isolation Forest needs ≥2 numeric features; skipped")
        return None
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        report.warn("scikit-learn unavailable; Isolation Forest skipped")
        return None
    matrix = df.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    if np.isnan(matrix).any():  # safety net if called before imputation
        col_medians = np.nanmedian(matrix, axis=0)
        idx = np.where(np.isnan(matrix))
        matrix[idx] = np.take(col_medians, idx[1])
    model = IsolationForest(
        contamination=config.iforest_contamination,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    return model.fit_predict(matrix) == -1


def handle_outliers(
    df: pl.DataFrame, config: CleanConfig | None = None
) -> tuple[pl.DataFrame, StepReport]:
    """Detect and treat outliers per the configured methods and action."""
    config = config or CleanConfig()
    report = StepReport(step="outliers")
    report.measure("methods", list(config.outlier_methods))
    report.measure("action", config.outlier_action)

    cols = _candidate_columns(df)
    if not cols or config.outlier_action == "none" and not config.outlier_methods:
        report.act("No eligible numeric columns for outlier analysis")
        return df, report

    n = df.height
    univariate_mask = np.zeros(n, dtype=bool)
    clip_exprs: list[pl.Expr] = []
    per_column: dict[str, int] = {}
    use_iqr = "iqr" in config.outlier_methods
    use_z = "zscore" in config.outlier_methods

    for c in cols:
        s = df.get_column(c)
        arr = s.cast(pl.Float64).to_numpy()
        bounds: list[tuple[float, float]] = []
        if use_iqr and (b := _iqr_bounds(s, config.iqr_multiplier)) is not None:
            bounds.append(b)
        if use_z and (b := _zscore_bounds(s, config.zscore_threshold)) is not None:
            bounds.append(b)
        if not bounds:
            continue
        col_mask = np.zeros(n, dtype=bool)
        for lo, hi in bounds:
            col_mask |= (arr < lo) | (arr > hi)
        per_column[c] = int(col_mask.sum())
        univariate_mask |= col_mask
        if config.outlier_action == "cap":
            lo = max(b[0] for b in bounds)  # tightest lower
            hi = min(b[1] for b in bounds)  # tightest upper
            clip_exprs.append(pl.col(c).clip(lo, hi).alias(c))

    if_mask = None
    if "isolation_forest" in config.outlier_methods:
        if_mask = _isolation_forest_mask(df, cols, config, report)

    combined = univariate_mask.copy()
    if if_mask is not None:
        combined |= if_mask
        report.measure("isolation_forest_flagged", int(if_mask.sum()))

    report.measure("per_column_flagged", per_column)
    report.measure("rows_flagged_total", int(combined.sum()))

    if config.outlier_action == "cap":
        if clip_exprs:
            df = df.with_columns(clip_exprs)
            report.act(f"Capped (winsorized) {len(clip_exprs)} column(s) to robust bounds")
        if if_mask is not None:  # IF can't be capped — surface as a flag instead
            df = df.with_columns(pl.Series(OUTLIER_FLAG_COLUMN, if_mask))
    elif config.outlier_action == "drop":
        if combined.any():
            df = df.filter(pl.Series(~combined))
            report.act(f"Dropped {int(combined.sum())} outlier row(s)")
    elif config.outlier_action == "flag":
        df = df.with_columns(pl.Series(OUTLIER_FLAG_COLUMN, combined))
        report.act(f"Flagged {int(combined.sum())} row(s) in '{OUTLIER_FLAG_COLUMN}'")
    else:  # none
        report.act(f"Detection-only: {int(combined.sum())} row(s) would be flagged")

    for action in report.actions:
        log(action, "INFO", enabled=config.verbose)
    return df, report
