"""Association measures beyond Pearson.

* **Spearman** & **Kendall** rank correlations (monotonic, robust to outliers),
* **partial correlation** (association after controlling for all other numerics),
* **Cramer's V** for categorical-categorical association,
* the **correlation ratio (eta)** for categorical -> numeric association.

Only the strongest/most-significant relationships are surfaced, capped for
readability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["AssociationsReport", "associations"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class AssociationsReport:
    spearman: list[tuple[str, str, float, float]] = field(default_factory=list)   # a,b,rho,p
    kendall: list[tuple[str, str, float, float]] = field(default_factory=list)    # a,b,tau,p
    partial: list[tuple[str, str, float]] = field(default_factory=list)           # a,b,partial r
    cramers_v: list[tuple[str, str, float]] = field(default_factory=list)         # a,b,V
    eta: list[tuple[str, str, float]] = field(default_factory=list)               # cat,num,eta


def _numeric_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and df.get_column(c).n_unique() > 2]


def _categorical_columns(df: pl.DataFrame, max_levels: int = 30) -> list[str]:
    out = []
    for c, dt in zip(df.columns, df.dtypes):
        if dt in (pl.Utf8, pl.Categorical, pl.Boolean) and 2 <= df.get_column(c).n_unique() <= max_levels:
            out.append(c)
    return out


def _cramers_v(a, b) -> float:
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    table = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = max(min(kcorr - 1, rcorr - 1), 1e-12)
    return float((phi2corr / denom) ** 0.5)


def _correlation_ratio(categories, values) -> float:
    import numpy as np

    cats = np.asarray(categories)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    cats, y = cats[mask], y[mask]
    if y.size < 3:
        return 0.0
    grand = y.mean()
    ss_between = 0.0
    for g in np.unique(cats):
        gy = y[cats == g]
        ss_between += gy.size * (gy.mean() - grand) ** 2
    ss_total = ((y - grand) ** 2).sum()
    return float((ss_between / ss_total) ** 0.5) if ss_total > 0 else 0.0


def associations(df: pl.DataFrame, config: CleanConfig | None = None) -> AssociationsReport:
    """Compute rank/partial correlations and categorical association measures."""
    config = config or CleanConfig()
    rep = AssociationsReport()
    try:
        import numpy as np
        from scipy import stats
    except ImportError:
        return rep

    numeric = _numeric_columns(df)
    if len(numeric) >= 2:
        sub = df.select(numeric).drop_nulls()
        if sub.height >= 5:
            mat = sub.to_numpy().astype(float)
            for i in range(len(numeric)):
                for j in range(i + 1, len(numeric)):
                    try:
                        rho, p_s = stats.spearmanr(mat[:, i], mat[:, j])
                        tau, p_k = stats.kendalltau(mat[:, i], mat[:, j])
                    except Exception:  # noqa: BLE001
                        continue
                    if abs(rho) >= 0.5:
                        rep.spearman.append((numeric[i], numeric[j], round(float(rho), 3), float(p_s)))
                    if abs(tau) >= 0.4:
                        rep.kendall.append((numeric[i], numeric[j], round(float(tau), 3), float(p_k)))
            rep.spearman.sort(key=lambda x: -abs(x[2]))
            rep.kendall.sort(key=lambda x: -abs(x[2]))

            # Partial correlations (control for all other numerics) via pingouin.
            if 2 <= len(numeric) <= 20:
                try:
                    import pingouin as pg

                    pc = pg.pcorr(sub.to_pandas())
                    cols = list(pc.columns)
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            r = float(pc.iloc[i, j])
                            if abs(r) >= 0.3:
                                rep.partial.append((cols[i], cols[j], round(r, 3)))
                    rep.partial.sort(key=lambda x: -abs(x[2]))
                except Exception:  # noqa: BLE001
                    pass

    # Categorical associations.
    cats = _categorical_columns(df)
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            sub = df.select([cats[i], cats[j]]).drop_nulls()
            if sub.height >= 5:
                v = _cramers_v(sub.get_column(cats[i]).to_list(), sub.get_column(cats[j]).to_list())
                if v >= 0.3:
                    rep.cramers_v.append((cats[i], cats[j], round(v, 3)))
    rep.cramers_v.sort(key=lambda x: -x[2])

    # Correlation ratio: categorical -> numeric.
    for cat in cats[:10]:
        for num in numeric[:15]:
            sub = df.select([cat, num]).drop_nulls()
            if sub.height >= 5:
                e = _correlation_ratio(sub.get_column(cat).to_list(), sub.get_column(num).to_numpy())
                if e >= 0.4:
                    rep.eta.append((cat, num, round(e, 3)))
    rep.eta.sort(key=lambda x: -x[2])
    return rep
