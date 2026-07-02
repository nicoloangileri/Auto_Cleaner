"""Functional Data Analysis (FDA): treat measurements as smooth curves.

When the specialisation engine finds a genuine time index, each numeric feature
is viewed as a function of that index. This module aggregates onto the time
grid, applies Savitzky-Golay smoothing, and runs **functional PCA** (the
dominant modes of variation across the curves), reporting how many functional
components capture 90% of the variance.

This is an automated, transparent first pass — for rigorous FDA (basis
expansions, registration/warping, derivatives) use a dedicated library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["FDAReport", "run_fda"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class FDAReport:
    time_index: str
    n_curves: int
    n_points: int
    smoothing: str
    variance_ratio: list[float] = field(default_factory=list)
    n_modes_90: int = 0
    note: str = (
        "Each numeric feature treated as a function of the time index; "
        "results are an automated first pass, not a substitute for dedicated FDA."
    )


def run_fda(
    df: pl.DataFrame,
    time_index: str,
    config: CleanConfig | None = None,
    *,
    id_columns: list[str] | None = None,
) -> FDAReport | None:
    """Smooth + functional-PCA the numeric curves indexed by ``time_index``."""
    config = config or CleanConfig()
    try:
        import numpy as np
        from scipy.signal import savgol_filter
    except ImportError:
        return None

    exclude = set(id_columns or []) | {"is_outlier", time_index}
    numeric = [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2
    ]
    if len(numeric) < 3:
        return None

    # Aggregate onto the (sorted, unique) time grid.
    grid = (
        df.group_by(time_index)
        .agg([pl.col(c).mean().alias(c) for c in numeric])
        .sort(time_index)
        .drop_nulls()
    )
    n_points = grid.height
    if n_points < 7:
        return None

    curves = grid.select(numeric).to_numpy().astype(float).T  # (n_curves, n_points)
    # Standardise each curve so modes reflect shape, not scale.
    mean = curves.mean(axis=1, keepdims=True)
    std = curves.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    curves = (curves - mean) / std

    # Savitzky-Golay smoothing (window must be odd and <= n_points).
    smoothing = "none"
    window = min(n_points if n_points % 2 == 1 else n_points - 1, 11)
    if window >= 5:
        try:
            curves = np.vstack([savgol_filter(row, window, polyorder=2) for row in curves])
            smoothing = f"Savitzky-Golay (window={window}, order=2)"
        except Exception:  # noqa: BLE001
            smoothing = "none"

    # Functional PCA: SVD of the centred curve matrix.
    centred = curves - curves.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(centred, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    var = singular**2
    if var.sum() == 0:
        return None
    ratio = (var / var.sum())
    cum = np.cumsum(ratio)
    return FDAReport(
        time_index=time_index,
        n_curves=len(numeric),
        n_points=n_points,
        smoothing=smoothing,
        variance_ratio=[round(float(v), 4) for v in ratio[:8]],
        n_modes_90=int(np.searchsorted(cum, 0.90) + 1),
    )
