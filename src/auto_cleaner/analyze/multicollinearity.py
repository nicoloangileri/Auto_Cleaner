"""Multicollinearity diagnostics: Variance Inflation Factors + PCA.

* **VIF** is read straight off the diagonal of the inverted correlation matrix
  (``VIF_i = (R⁻¹)_ii``) — exact, fast, and dependency-light. VIF > 5 is
  moderate, > 10 is severe multicollinearity.
* **PCA** reports the explained-variance spectrum and how many components are
  needed to retain 90% / 95% of variance — a direct read on redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["VIFResult", "PCAResult", "vif_scores", "pca_summary"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class VIFResult:
    feature: str
    vif: float


@dataclass(slots=True)
class PCAResult:
    explained_variance_ratio: list[float] = field(default_factory=list)
    cumulative: list[float] = field(default_factory=list)
    n_components_90: int = 0
    n_components_95: int = 0


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC_DTYPES and df.get_column(c).n_unique() > 2
    ]


def vif_scores(df: pl.DataFrame, config: CleanConfig | None = None) -> list[VIFResult]:
    """Variance Inflation Factor per numeric feature (descending)."""
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return []
    try:
        import numpy as np
    except ImportError:
        return []
    sub = df.select(cols).drop_nulls()
    if sub.height < len(cols) + 2:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(sub.to_numpy().astype(float), rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        try:
            inv = np.linalg.inv(corr)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(corr)  # singular → pseudo-inverse
    results = [VIFResult(c, float(abs(inv[i, i]))) for i, c in enumerate(cols)]
    return sorted(results, key=lambda r: -r.vif)


def pca_summary(df: pl.DataFrame, config: CleanConfig | None = None) -> PCAResult | None:
    """Explained-variance spectrum + components needed for 90% / 95% variance."""
    cols = _numeric_columns(df)
    if len(cols) < 2:
        return None
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None
    sub = df.select(cols).drop_nulls()
    if sub.height < len(cols) + 2:
        return None
    x = StandardScaler().fit_transform(sub.to_numpy().astype(float))
    pca = PCA().fit(x)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    return PCAResult(
        explained_variance_ratio=[round(float(v), 4) for v in evr],
        cumulative=[round(float(v), 4) for v in cum],
        n_components_90=int(np.searchsorted(cum, 0.90) + 1),
        n_components_95=int(np.searchsorted(cum, 0.95) + 1),
    )
