"""Statistical profiling: per-column descriptors + dataset-level diagnostics.

Produces a :class:`DatasetProfile` containing everything the report renderer
needs: distribution shape (skewness, kurtosis), missingness, cardinality, a
Pearson correlation matrix, covariance, collinearity pairs, and a derived list
of data-health warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["ColumnProfile", "DatasetProfile", "profile_dataset"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class ColumnProfile:
    """Per-column statistical summary (numeric fields are ``None`` for text)."""

    name: str
    dtype: str
    kind: str  # "numeric" | "categorical" | "datetime" | "boolean" | "other"
    count: int
    nulls: int
    null_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    maximum: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    top_value: Any | None = None
    top_freq: int | None = None


@dataclass(slots=True)
class DatasetProfile:
    """Whole-dataset profile, ready to be rendered into a report."""

    n_rows: int
    n_cols: int
    memory_bytes: int
    duplicate_rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    corr_labels: list[str] = field(default_factory=list)
    corr_matrix: list[list[float]] = field(default_factory=list)
    cov_matrix: list[list[float]] = field(default_factory=list)
    collinear_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_components: dict[str, float] = field(default_factory=dict)


def _kind_of(dtype: pl.DataType) -> str:
    if dtype in _NUMERIC_DTYPES:
        return "numeric"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return "datetime"
    if dtype in (pl.Utf8, pl.Categorical):
        return "categorical"
    return "other"


def _profile_column(df: pl.DataFrame, name: str) -> ColumnProfile:
    s = df.get_column(name)
    dtype = s.dtype
    kind = _kind_of(dtype)
    n = s.len()
    nulls = s.null_count()
    prof = ColumnProfile(
        name=name,
        dtype=str(dtype),
        kind=kind,
        count=n - nulls,
        nulls=nulls,
        null_pct=round(nulls / n * 100.0, 3) if n else 0.0,
        n_unique=s.n_unique(),
    )
    if kind == "numeric":
        agg = df.select(
            pl.col(name).mean().alias("mean"),
            pl.col(name).std().alias("std"),
            pl.col(name).min().alias("min"),
            pl.col(name).quantile(0.25).alias("q25"),
            pl.col(name).median().alias("median"),
            pl.col(name).quantile(0.75).alias("q75"),
            pl.col(name).max().alias("max"),
            pl.col(name).skew().alias("skew"),
            pl.col(name).kurtosis().alias("kurt"),
        ).row(0, named=True)
        prof.mean, prof.std = agg["mean"], agg["std"]
        prof.minimum, prof.q25, prof.median = agg["min"], agg["q25"], agg["median"]
        prof.q75, prof.maximum = agg["q75"], agg["max"]
        prof.skewness, prof.kurtosis = agg["skew"], agg["kurt"]
    elif kind in ("categorical", "boolean"):
        vc = s.drop_nulls().value_counts(sort=True)
        if vc.height:
            prof.top_value = vc.row(0)[0]
            prof.top_freq = int(vc.row(0)[1])
    return prof


def _correlation(df: pl.DataFrame, numeric_cols: list[str]) -> tuple[
    list[list[float]], list[list[float]]
]:
    sub = df.select(numeric_cols).drop_nulls()
    if sub.height < 2 or sub.width < 2:
        return [], []
    m = sub.to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(m, rowvar=False)
        cov = np.cov(m, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return np.round(corr, 4).tolist(), np.round(cov, 4).tolist()


def _derive_warnings(profile: DatasetProfile, config: CleanConfig) -> list[str]:
    warns: list[str] = []
    for col in profile.columns:
        if col.null_pct / 100.0 > config.missing_warn_threshold:
            warns.append(f"Feature '{col.name}' has {col.null_pct:.1f}% missing values")
        if col.n_unique <= 1:
            warns.append(f"Feature '{col.name}' is constant (zero variance) — consider dropping")
        if col.kind == "categorical" and col.n_unique > config.high_cardinality_warn:
            warns.append(
                f"Feature '{col.name}' is high-cardinality ({col.n_unique} levels) — "
                "encoding may explode dimensionality"
            )
        if col.kind == "numeric" and col.skewness is not None and abs(col.skewness) > config.skew_warn_threshold:
            warns.append(
                f"Feature '{col.name}' is strongly skewed (skew={col.skewness:.2f}) — "
                "consider a log/Box-Cox transform"
            )
    for a, b, r in profile.collinear_pairs:
        warns.append(f"High collinearity between '{a}' and '{b}' (r={r:+.2f})")
    if profile.duplicate_rows:
        warns.append(f"{profile.duplicate_rows} duplicate row(s) detected")
    return warns


def profile_dataset(df: pl.DataFrame, config: CleanConfig | None = None) -> DatasetProfile:
    """Compute a full :class:`DatasetProfile` for ``df``."""
    config = config or CleanConfig()
    numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC_DTYPES]

    profile = DatasetProfile(
        n_rows=df.height,
        n_cols=df.width,
        memory_bytes=int(df.estimated_size()),
        duplicate_rows=int(df.is_duplicated().sum()),
        numeric_columns=numeric_cols,
    )
    profile.columns = [_profile_column(df, c) for c in df.columns]

    corr, cov = _correlation(df, numeric_cols)
    profile.corr_labels = numeric_cols
    profile.corr_matrix = corr
    profile.cov_matrix = cov
    if corr:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr[i][j]
                if abs(r) >= config.corr_threshold:
                    profile.collinear_pairs.append((numeric_cols[i], numeric_cols[j], r))

    profile.warnings = _derive_warnings(profile, config)

    # Composite data-quality score (0-100): weighted completeness, validity,
    # uniqueness and (multivariate) outlier-cleanliness.
    total_cells = profile.n_rows * max(profile.n_cols, 1)
    total_nulls = sum(c.nulls for c in profile.columns)
    completeness = 1 - (total_nulls / total_cells) if total_cells else 1.0
    validity = max(0.0, 1 - len(profile.warnings) / max(profile.n_cols, 1))
    uniqueness = 1 - (profile.duplicate_rows / profile.n_rows) if profile.n_rows else 1.0
    outlier_clean = 1.0
    if "is_outlier" in df.columns:
        outlier_clean = 1 - float(df.get_column("is_outlier").sum()) / max(profile.n_rows, 1)
    comps = {
        "completeness": completeness, "validity": validity,
        "uniqueness": uniqueness, "outlier_clean": outlier_clean,
    }
    profile.quality_components = {k: round(v * 100, 1) for k, v in comps.items()}
    profile.quality_score = round(
        100 * (0.4 * completeness + 0.25 * validity + 0.2 * uniqueness + 0.15 * outlier_clean), 1
    )
    return profile
