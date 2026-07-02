"""Model persistence — train an exportable model and score new data later.

Turns the analysis tool into something *operational*: fit a robust model on the
cleaned data, serialise it together with its encoding (categorical level maps +
numeric medians) into a single ``.joblib`` bundle, and reload it to score fresh
data via the :mod:`auto_cleaner.predict` CLI. A pragmatic bridge to production —
not a full serving stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["train_exportable", "save_bundle", "load_bundle", "predict_frame"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_BUNDLE_VERSION = 1


def _infer_task(series: pl.Series) -> str:
    if series.dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
        return "classification"
    if series.dtype in _NUMERIC and series.dtype not in (pl.Float32, pl.Float64):
        return "classification" if series.n_unique() <= 20 else "regression"
    return "regression"


def _feature_columns(df: pl.DataFrame, target: str, exclude: set[str]) -> list[str]:
    cols = []
    for c, dt in zip(df.columns, df.dtypes):
        if c == target or c in exclude:
            continue
        if dt in _NUMERIC:
            cols.append(c)
        elif dt in (pl.Utf8, pl.Categorical, pl.Boolean) and df.get_column(c).n_unique() <= 50:
            cols.append(c)
    return cols


def _encode(df: pl.DataFrame, features: list[str], cat_maps: dict, medians: dict):
    import numpy as np

    columns = []
    for c in features:
        s = df.get_column(c)
        if c in cat_maps:
            mapping = cat_maps[c]
            codes = [mapping.get(str(v), -1) for v in s.to_list()]
            columns.append(np.asarray(codes, dtype=float))
        else:
            arr = s.cast(pl.Float64).to_numpy().astype(float)
            med = medians.get(c, 0.0)
            arr = np.where(np.isfinite(arr), arr, med)
            columns.append(arr)
    return np.column_stack(columns) if columns else np.empty((df.height, 0))


def train_exportable(
    df: pl.DataFrame, target: str, config: CleanConfig | None = None,
    *, id_columns: list[str] | None = None,
) -> dict[str, Any] | None:
    """Fit a robust gradient-boosting model and return a serialisable bundle."""
    config = config or CleanConfig()
    if target not in df.columns:
        return None
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    except ImportError:
        return None

    exclude = set(id_columns or []) | {"is_outlier"}
    frame = df.drop_nulls(subset=[target])
    task = _infer_task(frame.get_column(target))
    features = _feature_columns(frame, target, exclude)
    if not features or frame.height < 20:
        return None

    cat_maps: dict[str, dict[str, int]] = {}
    medians: dict[str, float] = {}
    for c in features:
        dt = frame.get_column(c).dtype
        if dt in (pl.Utf8, pl.Categorical, pl.Boolean):
            levels = [str(v) for v in frame.get_column(c).drop_nulls().unique().to_list()]
            cat_maps[c] = {lvl: i for i, lvl in enumerate(sorted(levels))}
        else:
            medians[c] = float(frame.get_column(c).median() or 0.0)

    x = _encode(frame, features, cat_maps, medians)
    if task == "classification":
        y_series = frame.get_column(target)
        target_levels = [str(v) for v in y_series.drop_nulls().unique().to_list()]
        target_map = {lvl: i for i, lvl in enumerate(sorted(target_levels))}
        y = np.asarray([target_map[str(v)] for v in y_series.to_list()], dtype=int)
        model = HistGradientBoostingClassifier(random_state=config.random_seed).fit(x, y)
        inverse_target = {i: lvl for lvl, i in target_map.items()}
    else:
        y = frame.get_column(target).cast(pl.Float64).to_numpy()
        model = HistGradientBoostingRegressor(random_state=config.random_seed).fit(x, y)
        inverse_target = None

    return {
        "version": _BUNDLE_VERSION,
        "estimator": model,
        "task": task,
        "target": target,
        "features": features,
        "cat_maps": cat_maps,
        "medians": medians,
        "inverse_target": inverse_target,
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> str:
    """Serialise a model bundle to ``path`` (joblib)."""
    from joblib import dump

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, p)
    return str(p)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a model bundle saved by :func:`save_bundle`."""
    from joblib import load

    return load(Path(path))


def predict_frame(bundle: dict[str, Any], df: pl.DataFrame) -> pl.Series:
    """Score ``df`` with a loaded bundle, returning a prediction Series."""
    x = _encode(df, bundle["features"], bundle["cat_maps"], bundle["medians"])
    preds = bundle["estimator"].predict(x)
    if bundle["task"] == "classification" and bundle.get("inverse_target"):
        inv = bundle["inverse_target"]
        preds = [inv.get(int(p), p) for p in preds]
    return pl.Series(f"{bundle['target']}_prediction", list(preds))
