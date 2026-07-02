"""Memory optimisation via safe, lossless dtype downcasting.

* **Integers** are narrowed to the smallest signed/unsigned width that fits the
  observed ``[min, max]`` range.
* **Float64** columns become **Float32** only when the round-trip relative error
  stays under ``float32_rel_tolerance`` *and* the magnitude fits Float32 — i.e.
  the downcast is provably safe, never a silent precision grenade.
* **Low-cardinality strings** are converted to ``Categorical`` (dictionary
  encoding) for a large memory win on repetitive text.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from auto_cleaner.config import CleanConfig
from auto_cleaner.logging_utils import human_bytes, log
from auto_cleaner.reporting import StepReport

__all__ = ["downcast"]

_FLOAT32_MAX = 3.4028235e38

# Ordered smallest → largest; unsigned tried first when min >= 0.
_UINT_LADDER: tuple[tuple[pl.DataType, int], ...] = (
    (pl.UInt8, 255),
    (pl.UInt16, 65_535),
    (pl.UInt32, 4_294_967_295),
)
_INT_LADDER: tuple[tuple[pl.DataType, int, int], ...] = (
    (pl.Int8, -128, 127),
    (pl.Int16, -32_768, 32_767),
    (pl.Int32, -2_147_483_648, 2_147_483_647),
)

_INT_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def _smallest_int_dtype(lo: int, hi: int) -> pl.DataType:
    """Pick the narrowest integer dtype covering ``[lo, hi]``."""
    if lo >= 0:
        for dtype, umax in _UINT_LADDER:
            if hi <= umax:
                return dtype
    for dtype, imin, imax in _INT_LADDER:
        if lo >= imin and hi <= imax:
            return dtype
    return pl.Int64


def _float32_is_safe(series: pl.Series, tol: float) -> bool:
    """True when Float64 → Float32 round-trips within relative tolerance."""
    nn = series.drop_nulls()
    if nn.len() == 0:
        return True
    a = nn.to_numpy().astype(np.float64)
    if not np.all(np.isfinite(a)):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return True
    if np.abs(a).max() > _FLOAT32_MAX:
        return False
    b = a.astype(np.float32).astype(np.float64)
    denom = np.where(np.abs(a) > 0.0, np.abs(a), 1.0)
    return float(np.max(np.abs(a - b) / denom)) <= tol


def downcast(df: pl.DataFrame, config: CleanConfig | None = None) -> tuple[pl.DataFrame, StepReport]:
    """Downcast numeric/text columns in place-of-type, reporting the memory saved."""
    config = config or CleanConfig()
    report = StepReport(step="downcast")
    before = int(df.estimated_size())
    report.measure("memory_before", before)

    if not config.downcast:
        report.act("Downcasting disabled by config")
        report.measure("memory_after", before)
        return df, report

    exprs: list[pl.Expr] = []
    changes: dict[str, str] = {}

    for c, dt in zip(df.columns, df.dtypes):
        s = df.get_column(c)
        if dt in _INT_DTYPES:
            lo, hi = s.min(), s.max()
            if lo is None or hi is None:
                continue
            target = _smallest_int_dtype(int(lo), int(hi))
            if target != dt:
                exprs.append(pl.col(c).cast(target))
                changes[c] = f"{dt} → {target}"
        elif dt == pl.Float64 and config.downcast_floats:
            if _float32_is_safe(s, config.float32_rel_tolerance):
                exprs.append(pl.col(c).cast(pl.Float32))
                changes[c] = "Float64 → Float32"
        elif dt == pl.Utf8:
            nu = s.n_unique()
            if 0 < nu <= 1024 and (nu / max(s.len(), 1)) <= 0.5:
                exprs.append(pl.col(c).cast(pl.Categorical))
                changes[c] = "Utf8 → Categorical"

    if exprs:
        df = df.with_columns(exprs)

    after = int(df.estimated_size())
    report.measure("memory_after", after)
    report.measure("dtype_changes", changes)
    saved = before - after
    pct = (saved / before * 100.0) if before else 0.0
    report.act(
        f"Downcast {len(changes)} column(s): "
        f"{human_bytes(before)} → {human_bytes(after)} (saved {pct:.1f}%)"
    )
    log(report.actions[-1], "OK", enabled=config.verbose)
    return df, report
