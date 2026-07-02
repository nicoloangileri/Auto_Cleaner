"""Survey / questionnaire methodology: reliability + design weighting.

* **Cronbach's alpha** measures internal-consistency reliability across
  Likert-like ordinal items (with a 95% CI).
* **Design-weighted means** apply a detected sampling-weight column so estimates
  reflect the survey design rather than the raw sample.

Auto-runs when the specialisation engine flags a survey/questionnaire archetype
or a weight column is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["SurveyReport", "survey_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_WEIGHT_HINTS = ("weight", "wt", "wgt", "sampling_weight", "pweight", "finalwt")


@dataclass(slots=True)
class SurveyReport:
    cronbach_alpha: float | None = None
    cronbach_ci: tuple[float, float] | None = None
    cronbach_items: list[str] = field(default_factory=list)
    weight_column: str | None = None
    weighted_means: list[tuple[str, float, float]] = field(default_factory=list)  # col, weighted, unweighted


def survey_analysis(
    df: pl.DataFrame, config: CleanConfig | None = None, *, id_columns: list[str] | None = None
) -> SurveyReport | None:
    """Compute Cronbach's alpha over Likert items + design-weighted means."""
    config = config or CleanConfig()
    exclude = set(id_columns or []) | {"is_outlier"}
    rep = SurveyReport()
    produced = False

    # Likert-like items: small-range non-negative integers.
    likert = [
        c for c, dt in zip(df.columns, df.dtypes)
        if dt in _NUMERIC and c not in exclude
        and df.get_column(c).n_unique() <= 7 and (df.get_column(c).min() or 0) >= 0
        and df.get_column(c).n_unique() > 1
    ]
    if len(likert) >= 3:
        try:
            import pingouin as pg

            items = df.select(likert).drop_nulls().to_pandas()
            alpha, ci = pg.cronbach_alpha(data=items)
            rep.cronbach_alpha = round(float(alpha), 3)
            rep.cronbach_ci = (round(float(ci[0]), 3), round(float(ci[1]), 3))
            rep.cronbach_items = likert
            produced = True
        except Exception:  # noqa: BLE001
            pass

    # Design weighting.
    weight = next((c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and any(h == c.lower() or c.lower().endswith(h) for h in _WEIGHT_HINTS)), None)
    if weight is not None:
        rep.weight_column = weight
        numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and c != weight and df.get_column(c).n_unique() > 2]
        for c in numeric[:15]:
            sub = df.select([c, weight]).drop_nulls()
            w = sub.get_column(weight).to_numpy()
            v = sub.get_column(c).to_numpy()
            if w.sum() > 0:
                import numpy as np

                rep.weighted_means.append((c, round(float(np.average(v, weights=w)), 4), round(float(v.mean()), 4)))
        produced = True

    return rep if produced else None
