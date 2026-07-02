"""Lightweight Bayesian comparison: Bayes factors for group differences.

For a binary grouping, computes the Bayes factor (BF10) of a difference in means
via :mod:`pingouin`, with a plain-language interpretation (Jeffreys' scale).
Complements the frequentist p-values with evidence *for* the null as well as the
alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["BayesFactor", "BayesianReport", "bayesian_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class BayesFactor:
    value: str
    group: str
    bf10: float
    interpretation: str


@dataclass(slots=True)
class BayesianReport:
    factors: list[BayesFactor] = field(default_factory=list)


def _interpret(bf10: float) -> str:
    if bf10 >= 100:
        return "extreme evidence for a difference"
    if bf10 >= 30:
        return "very strong evidence for a difference"
    if bf10 >= 10:
        return "strong evidence for a difference"
    if bf10 >= 3:
        return "moderate evidence for a difference"
    if bf10 > 1:
        return "anecdotal evidence for a difference"
    if bf10 == 0:
        return "n/a"
    if bf10 > 1 / 3:
        return "anecdotal evidence for no difference"
    return "evidence favours no difference"


def bayesian_analysis(
    df: pl.DataFrame, config: CleanConfig | None = None, *,
    target: str | None = None, id_columns: list[str] | None = None,
) -> BayesianReport:
    """Bayes factors for numeric differences across a binary grouping."""
    config = config or CleanConfig()
    target = target if target is not None else config.target
    exclude = set(id_columns or []) | {"is_outlier"}
    rep = BayesianReport()
    try:
        import pingouin as pg
    except ImportError:
        return rep

    # Pick a binary grouping: the target if binary, else a 2-level factor.
    group = None
    if target and target in df.columns and df.get_column(target).n_unique() == 2:
        group = target
    else:
        for c, dt in zip(df.columns, df.dtypes):
            if c in exclude or c == target:
                continue
            if dt in (pl.Utf8, pl.Categorical, pl.Boolean) and df.get_column(c).n_unique() == 2:
                group = c
                break
    if group is None:
        return rep

    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and c != group and df.get_column(c).n_unique() > 2]
    levels = df.get_column(group).unique().drop_nulls().to_list()
    if len(levels) != 2:
        return rep
    for value in numeric[:8]:
        sub = df.select([value, group]).drop_nulls()
        a = sub.filter(pl.col(group) == levels[0]).get_column(value).to_list()
        b = sub.filter(pl.col(group) == levels[1]).get_column(value).to_list()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            res = pg.ttest(a, b)
            bf10 = float(str(res["BF10"].iloc[0]))
        except Exception:  # noqa: BLE001
            continue
        rep.factors.append(BayesFactor(value, group, round(bf10, 3), _interpret(bf10)))
    rep.factors.sort(key=lambda f: -f.bf10)
    return rep
