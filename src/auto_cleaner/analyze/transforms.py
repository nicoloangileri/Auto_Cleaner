"""Automatic distribution-normalising transforms for skewed features.

For each strongly-skewed numeric feature it fits the best power transform —
**Box-Cox** for strictly-positive data, **Yeo-Johnson** otherwise — reports the
skewness before/after, and (optionally) applies it, appending a transformed
``<feature>__boxcox`` / ``<feature>__yeojohnson`` column. Original columns are
never overwritten, so the cleaned dataset stays predictable.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.logging_utils import log
from auto_cleaner.reporting import StepReport

__all__ = ["TransformSuggestion", "suggest_transforms", "apply_transforms"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class TransformSuggestion:
    """A recommended power transform and the skewness improvement it yields."""

    feature: str
    method: str          # "box-cox" | "yeo-johnson"
    suffix: str          # column suffix if applied
    lam: float           # fitted lambda
    skew_before: float
    skew_after: float


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC_DTYPES and df.get_column(c).n_unique() > 2
    ]


def suggest_transforms(df: pl.DataFrame, config: CleanConfig | None = None) -> list[TransformSuggestion]:
    """Recommend a normalising transform for each strongly-skewed feature."""
    config = config or CleanConfig()
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return []

    out: list[TransformSuggestion] = []
    for c in _numeric_columns(df):
        x = df.get_column(c).drop_nulls().to_numpy().astype(float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            continue
        skew_before = float(stats.skew(x))
        if abs(skew_before) <= config.skew_threshold:
            continue
        try:
            if x.min() > 0:
                method, suffix = "box-cox", "boxcox"
                transformed, lam = stats.boxcox(x)
            else:
                method, suffix = "yeo-johnson", "yeojohnson"
                transformed, lam = stats.yeojohnson(x)
        except Exception:  # noqa: BLE001
            continue
        skew_after = float(stats.skew(transformed))
        if abs(skew_after) < abs(skew_before):
            out.append(
                TransformSuggestion(c, method, suffix, float(lam), skew_before, skew_after)
            )
    return out


def apply_transforms(
    df: pl.DataFrame,
    suggestions: list[TransformSuggestion],
    config: CleanConfig | None = None,
) -> tuple[pl.DataFrame, StepReport]:
    """Append transformed columns for each suggestion (original columns kept)."""
    config = config or CleanConfig()
    report = StepReport(step="transforms")
    if not suggestions:
        return df, report
    import numpy as np
    from scipy import stats

    new_cols = []
    for s in suggestions:
        full = df.get_column(s.feature).cast(pl.Float64).to_numpy().astype(float)
        mask = np.isfinite(full)
        out = full.copy()
        if s.method == "box-cox":
            out[mask] = stats.boxcox(full[mask], s.lam)
        else:
            out[mask] = stats.yeojohnson(full[mask], s.lam)
        name = f"{s.feature}__{s.suffix}"
        new_cols.append(pl.Series(name, out))
        report.act(f"Applied {s.method} to '{s.feature}' → '{name}' (skew {s.skew_before:+.2f} → {s.skew_after:+.2f})")
    df = df.with_columns(new_cols)
    for a in report.actions:
        log(a, "INFO", enabled=config.verbose)
    return df, report
