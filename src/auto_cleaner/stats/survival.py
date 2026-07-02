"""Survival / time-to-event analysis (auto-detected).

If the data contain a duration column and a binary event/status column, fits a
Kaplan-Meier estimator (median survival) and a Cox proportional-hazards model
(concordance + hazard ratios) via :mod:`lifelines`. Only runs when both columns
are confidently detected — otherwise silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["SurvivalReport", "survival_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_DURATION_HINTS = ("time", "duration", "tenure", "t", "lifetime", "survival", "age_at", "days", "months")
_EVENT_HINTS = ("event", "status", "dead", "death", "churn", "observed", "failed", "failure")


@dataclass(slots=True)
class SurvivalReport:
    duration_col: str
    event_col: str
    n: int
    n_events: int
    median_survival: float | None = None
    cox_concordance: float | None = None
    cox_terms: list[tuple[str, float, float, float]] = field(default_factory=list)  # name, coef, p, hazard ratio


def _find_columns(df: pl.DataFrame) -> tuple[str | None, str | None]:
    duration = event = None
    for c, dt in zip(df.columns, df.dtypes):
        lo = c.lower()
        if duration is None and dt in _NUMERIC and any(h in lo for h in _DURATION_HINTS):
            s = df.get_column(c)
            if (s.min() or 0) >= 0 and s.n_unique() > 5:
                duration = c
        if event is None and any(h in lo for h in _EVENT_HINTS):
            s = df.get_column(c)
            if s.n_unique() == 2:
                event = c
    return duration, event


def survival_analysis(
    df: pl.DataFrame, config: CleanConfig | None = None, *, id_columns: list[str] | None = None
) -> SurvivalReport | None:
    """Kaplan-Meier + Cox PH when a duration and event column are detected."""
    config = config or CleanConfig()
    duration, event = _find_columns(df)
    if duration is None or event is None:
        return None
    try:
        import numpy as np
        from lifelines import CoxPHFitter, KaplanMeierFitter
    except ImportError:
        return None

    exclude = set(id_columns or []) | {"is_outlier", duration, event}
    sub = df.select([duration, event]).drop_nulls()
    if sub.height < 20:
        return None
    pdf = sub.to_pandas()
    # Normalise event to 0/1.
    ev = pdf[event]
    pdf[event] = (ev == sorted(ev.unique())[-1]).astype(int)

    rep = SurvivalReport(
        duration_col=duration, event_col=event, n=int(len(pdf)), n_events=int(pdf[event].sum())
    )
    try:
        kmf = KaplanMeierFitter().fit(pdf[duration], pdf[event])
        rep.median_survival = None if kmf.median_survival_time_ != kmf.median_survival_time_ else float(kmf.median_survival_time_)
    except Exception:  # noqa: BLE001
        pass

    try:
        covars = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2][:8]
        if covars:
            cph_df = df.select([duration, event] + covars).drop_nulls().to_pandas()
            cph_df[event] = (cph_df[event] == sorted(cph_df[event].unique())[-1]).astype(int)
            cph = CoxPHFitter(penalizer=0.1).fit(cph_df, duration_col=duration, event_col=event)
            rep.cox_concordance = round(float(cph.concordance_index_), 3)
            summ = cph.summary
            for name in summ.index:
                rep.cox_terms.append((
                    str(name), round(float(summ.loc[name, "coef"]), 3),
                    float(summ.loc[name, "p"]), round(float(summ.loc[name, "exp(coef)"]), 3),
                ))
            rep.cox_terms.sort(key=lambda t: t[2])
    except Exception:  # noqa: BLE001
        pass
    return rep
