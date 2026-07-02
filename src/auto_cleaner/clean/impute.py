"""Missing-data imputation with context-aware strategy selection.

Decision logic
--------------
1. **Time-series** — if a *sorted* date/datetime column is present, numeric gaps
   are forward-then-backward filled (the correct default for ordered series).
2. **Numeric** — ``auto`` uses the **median** for skewed columns
   (``|skew| > skew_threshold``) and the **mean** for symmetric ones; ``knn``
   uses a lightweight :class:`sklearn.impute.KNNImputer` (guarded by a row cap).
3. **Categorical** — filled with the **mode** (default) or a constant token.
"""

from __future__ import annotations

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.logging_utils import log
from auto_cleaner.reporting import StepReport

__all__ = ["impute_missing"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC_DTYPES]


def _categorical_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Utf8, pl.Categorical)]


# A genuine time index is sorted *and* (near-)unique. This guards against panel
# data (e.g. many records sharing a coarse 'year') being mistaken for a series.
_TIME_INDEX_MIN_UNIQUE_RATIO = 0.90


def _find_sorted_time_index(df: pl.DataFrame, config: CleanConfig) -> str | None:
    """Return the name of a null-free, sorted, near-unique time column (or None)."""
    if not config.detect_timeseries:
        return None
    for c, dt in zip(df.columns, df.dtypes):
        if dt not in (pl.Date, pl.Datetime):
            continue
        s = df.get_column(c)
        if s.null_count() != 0 or s.len() < 3:
            continue
        if s.n_unique() / s.len() < _TIME_INDEX_MIN_UNIQUE_RATIO:
            continue  # too many repeats → panel/categorical date, not a series
        diffs = s.to_physical().diff().drop_nulls()
        if diffs.len() and bool((diffs >= 0).all()):
            return c
    return None


def _impute_timeseries(
    df: pl.DataFrame, num_cols: list[str], report: StepReport
) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward").alias(c)
            for c in num_cols
        ]
    )
    report.act(f"Time-series forward/backward fill applied to {len(num_cols)} numeric col(s)")
    return df


def _impute_numeric_statistical(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    exprs = []
    for c in _numeric_columns(df):
        n_missing = df.get_column(c).null_count()
        if n_missing == 0:
            continue
        if df.get_column(c).drop_nulls().len() == 0:
            report.warn(f"Column '{c}' is entirely null — left unimputed")
            continue
        strategy = config.impute_numeric
        if strategy == "auto":
            skew = df.select(pl.col(c).skew()).item()
            strategy = "median" if (skew is not None and abs(skew) > config.skew_threshold) else "mean"
        fill_value = (
            pl.col(c).median() if strategy == "median" else pl.col(c).mean()
        )
        exprs.append(pl.col(c).fill_null(fill_value).alias(c))
        report.act(f"Imputed '{c}': {n_missing} null(s) via {strategy}")
    return df.with_columns(exprs) if exprs else df


def _impute_numeric_knn(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    num_cols = [c for c in _numeric_columns(df) if df.get_column(c).null_count() > 0]
    if not num_cols:
        return df
    if df.height > config.knn_max_rows:
        report.warn(
            f"KNN skipped ({df.height:,} rows > cap {config.knn_max_rows:,}); "
            "fell back to statistical imputation"
        )
        return _impute_numeric_statistical(df, config.with_overrides(impute_numeric="auto"), report)
    try:
        from sklearn.impute import KNNImputer
    except ImportError:
        report.warn("scikit-learn unavailable; KNN fell back to statistical imputation")
        return _impute_numeric_statistical(df, config.with_overrides(impute_numeric="auto"), report)

    all_numeric = _numeric_columns(df)  # use the full numeric space as predictors
    matrix = df.select([pl.col(c).cast(pl.Float64) for c in all_numeric]).to_numpy()
    imputed = KNNImputer(n_neighbors=config.knn_neighbors).fit_transform(matrix)
    df = df.with_columns(
        [pl.Series(name=all_numeric[i], values=imputed[:, i]) for i in range(len(all_numeric))]
    )
    report.act(f"KNN-imputed numeric columns {num_cols} (k={config.knn_neighbors})")
    return df


def _impute_categorical(
    df: pl.DataFrame, config: CleanConfig, report: StepReport
) -> pl.DataFrame:
    if config.impute_categorical == "none":
        return df
    exprs = []
    for c in _categorical_columns(df):
        n_missing = df.get_column(c).null_count()
        if n_missing == 0:
            continue
        if config.impute_categorical == "constant":
            fill = config.categorical_fill_value
            method = f"constant '{fill}'"
        else:  # mode
            modes = df.get_column(c).drop_nulls().mode()
            # Ties are returned in arbitrary order: sort for a deterministic
            # fill (same input -> same output, run after run).
            fill = modes.sort()[0] if modes.len() else config.categorical_fill_value
            method = f"mode '{fill}'"
        exprs.append(pl.col(c).fill_null(fill).alias(c))
        report.act(f"Imputed '{c}': {n_missing} null(s) via {method}")
    return df.with_columns(exprs) if exprs else df


def impute_missing(
    df: pl.DataFrame, config: CleanConfig | None = None
) -> tuple[pl.DataFrame, StepReport]:
    """Fill missing values using the strategy stack described in the module docstring."""
    config = config or CleanConfig()
    report = StepReport(step="impute")
    report.measure("nulls_before", int(df.null_count().sum_horizontal().item()))

    ts_col = _find_sorted_time_index(df, config)
    if ts_col is not None:
        report.measure("time_index", ts_col)
        df = _impute_timeseries(df, _numeric_columns(df), report)

    if config.impute_numeric == "knn":
        df = _impute_numeric_knn(df, config, report)
    elif config.impute_numeric != "none":
        df = _impute_numeric_statistical(df, config, report)

    df = _impute_categorical(df, config, report)

    report.measure("nulls_after", int(df.null_count().sum_horizontal().item()))
    for action in report.actions:
        log(action, "INFO", enabled=config.verbose)
    return df, report
