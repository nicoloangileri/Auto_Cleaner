"""Parametric distribution fitting + model selection.

For each numeric feature, fits a panel of candidate distributions (normal,
log-normal, gamma, Weibull, exponential, Pareto), scores them by AIC/BIC and a
Kolmogorov-Smirnov goodness-of-fit test, and reports the best-fitting law. Tells
you *which distribution your data actually follow* — not just whether they are
"normal".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["DistributionFit", "fit_distributions"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
# (scipy name, requires strictly-positive data)
_CANDIDATES = (
    ("norm", False),
    ("lognorm", True),
    ("gamma", True),
    ("weibull_min", True),
    ("expon", True),
    ("pareto", True),
)


@dataclass(slots=True)
class DistributionFit:
    feature: str
    best_distribution: str
    aic: float
    bic: float
    ks_p: float
    candidates: list[tuple[str, float, float]] = field(default_factory=list)  # name, aic, ks_p


def fit_distributions(df: pl.DataFrame, config: CleanConfig | None = None) -> list[DistributionFit]:
    """Fit and rank candidate distributions for each numeric feature."""
    config = config or CleanConfig()
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return []

    out: list[DistributionFit] = []
    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and df.get_column(c).n_unique() > 5]
    for c in numeric[:30]:
        x = df.get_column(c).drop_nulls().to_numpy().astype(float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 20:
            continue
        positive = bool((x > 0).all())
        scored: list[tuple[str, float, float, float]] = []  # name, aic, bic, ks_p
        for name, needs_pos in _CANDIDATES:
            if needs_pos and not positive:
                continue
            dist = getattr(stats, name)
            try:
                params = dist.fit(x)
                ll = float(np.sum(dist.logpdf(x, *params)))
                if not np.isfinite(ll):
                    continue
                k = len(params)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n) - 2 * ll
                ks_p = float(stats.kstest(x, dist.cdf, args=params).pvalue)
                scored.append((name, round(aic, 1), round(bic, 1), round(ks_p, 4)))
            except Exception:  # noqa: BLE001
                continue
        if not scored:
            continue
        scored.sort(key=lambda s: s[1])  # lowest AIC wins
        best = scored[0]
        out.append(
            DistributionFit(
                feature=c, best_distribution=best[0], aic=best[1], bic=best[2], ks_p=best[3],
                candidates=[(s[0], s[1], s[3]) for s in scored],
            )
        )
    return out
