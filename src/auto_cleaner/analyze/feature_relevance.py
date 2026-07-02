"""Target-aware feature relevance ranking.

Given a target column, ranks every feature by predictive relevance using three
complementary signals: **mutual information** (captures non-linear dependence),
the **ANOVA / regression F-statistic** (with p-value), and **Pearson
correlation** with the target (numeric→numeric only). The task type
(classification vs regression) is inferred from the target automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["RelevanceResult", "feature_relevance"]

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_CATEGORICAL_DTYPES = (pl.Utf8, pl.Categorical, pl.Boolean)
_ARTIFACT_COLUMNS = frozenset({"is_outlier"})  # pipeline-generated flags, not real features


def _int_codes(series: pl.Series):
    """Integer codes for a categorical/boolean/integer series.

    Boolean and integer dtypes can't cast straight to Categorical in polars, so
    they are handled explicitly (integers are already class codes).
    """
    if series.dtype == pl.Boolean:
        return series.cast(pl.Int8).fill_null(-1).to_numpy().astype(float)
    if series.dtype in _NUMERIC_DTYPES:
        return series.fill_null(-1).to_numpy().astype(float)
    return series.cast(pl.Categorical).to_physical().fill_null(-1).to_numpy().astype(float)


@dataclass(slots=True)
class RelevanceResult:
    feature: str
    mutual_info: float
    f_stat: float | None
    p_value: float | None
    target_corr: float | None
    rank: int


def _infer_task(series: pl.Series) -> str:
    """Classification for categorical / low-cardinality integer targets; else regression."""
    if series.dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
        return "classification"
    if series.dtype in _NUMERIC_DTYPES and series.dtype not in (pl.Float32, pl.Float64):
        return "classification" if series.n_unique() <= 20 else "regression"
    return "regression"


def feature_relevance(
    df: pl.DataFrame, target: str, config: CleanConfig | None = None
) -> tuple[list[RelevanceResult], str | None]:
    """Rank features by relevance to ``target``. Returns ``(results, task_type)``."""
    config = config or CleanConfig()
    if target not in df.columns:
        return [], None
    try:
        import numpy as np
        from sklearn.feature_selection import (
            f_classif, f_regression, mutual_info_classif, mutual_info_regression,
        )
    except ImportError:
        return [], None

    task = _infer_task(df.get_column(target))
    frame = df.drop_nulls(subset=[target])

    feature_cols: list[str] = []
    discrete_mask: list[bool] = []
    columns_data: list = []
    for c in frame.columns:
        if c == target or c in _ARTIFACT_COLUMNS:
            continue
        dt = frame.get_column(c).dtype
        s = frame.get_column(c)
        if dt in _NUMERIC_DTYPES:
            columns_data.append(s.cast(pl.Float64).fill_null(s.median()).to_numpy())
            discrete_mask.append(False)
        elif dt in _CATEGORICAL_DTYPES:
            if dt != pl.Boolean and s.n_unique() > 50:  # high-cardinality text → skip (noise)
                continue
            columns_data.append(_int_codes(s))
            discrete_mask.append(True)
        else:
            continue
        feature_cols.append(c)

    if not feature_cols:
        return [], task

    import numpy as np
    x = np.column_stack(columns_data)

    y_series = frame.get_column(target)
    if task == "classification":
        y = _int_codes(y_series)
        mi = mutual_info_classif(x, y, discrete_features=np.array(discrete_mask), random_state=config.random_seed)
        try:
            f_stat, p_val = f_classif(x, y)
        except Exception:  # noqa: BLE001
            f_stat = p_val = [None] * len(feature_cols)
        y_num = None
    else:
        y = y_series.cast(pl.Float64).to_numpy()
        mi = mutual_info_regression(x, y, discrete_features=np.array(discrete_mask), random_state=config.random_seed)
        try:
            f_stat, p_val = f_regression(x, y)
        except Exception:  # noqa: BLE001
            f_stat = p_val = [None] * len(feature_cols)
        y_num = y

    results: list[RelevanceResult] = []
    for i, c in enumerate(feature_cols):
        corr = None
        if y_num is not None and not discrete_mask[i]:
            with np.errstate(invalid="ignore"):
                cc = np.corrcoef(x[:, i], y_num)[0, 1]
            corr = None if np.isnan(cc) else round(float(cc), 4)
        results.append(
            RelevanceResult(
                feature=c,
                mutual_info=round(float(mi[i]), 5),
                f_stat=(None if f_stat[i] is None else round(float(f_stat[i]), 4)),
                p_value=(None if p_val[i] is None else float(f"{float(p_val[i]):.3g}")),
                target_corr=corr,
                rank=0,
            )
        )
    results.sort(key=lambda r: -r.mutual_info)
    for rank, r in enumerate(results, start=1):
        r.rank = rank
    return results, task
