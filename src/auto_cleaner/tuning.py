"""Hyper-parameter tuning with Optuna (opt-in).

Tunes a gradient-boosting model by cross-validation and reports the improvement
over the untuned baseline plus the best hyper-parameters. Off by default
(``CleanConfig.tune`` / ``--tune``) because tuning costs time and, like all
optimisation on CV, can over-fit the validation folds if over-used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["TuneResult", "tune_model"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass(slots=True)
class TuneResult:
    task: str
    target: str
    metric: str
    n_trials: int
    baseline_cv: float
    best_cv: float
    improvement: float
    best_params: dict = field(default_factory=dict)


def _infer_task(series: pl.Series) -> str:
    if series.dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
        return "classification"
    if series.dtype in _NUMERIC and series.dtype not in (pl.Float32, pl.Float64):
        return "classification" if series.n_unique() <= 20 else "regression"
    return "regression"


def _label_codes(series: pl.Series):
    import numpy as np

    if series.dtype == pl.Boolean:
        return series.cast(pl.Int64).to_numpy()
    if series.dtype in _NUMERIC:
        return series.to_numpy()
    return series.cast(pl.Categorical).to_physical().to_numpy()


def _encode_features(frame: pl.DataFrame, target: str, exclude: set[str]):
    import numpy as np

    cols = []
    for c, dt in zip(frame.columns, frame.dtypes):
        if c == target or c in exclude:
            continue
        s = frame.get_column(c)
        if dt in _NUMERIC:
            cols.append(s.cast(pl.Float64).fill_null(s.median()).to_numpy())
        elif dt in (pl.Utf8, pl.Categorical, pl.Boolean):
            if s.n_unique() > 50:
                continue
            cols.append(_label_codes(s.cast(pl.Categorical) if dt != pl.Boolean else s).astype(float))
    return np.column_stack(cols) if cols else None


def tune_model(
    df: pl.DataFrame, target: str, config: CleanConfig | None = None,
    *, id_columns: list[str] | None = None,
) -> TuneResult | None:
    """Optuna-tune gradient boosting; return the CV improvement over baseline."""
    config = config or CleanConfig()
    if target not in df.columns:
        return None
    try:
        import numpy as np
        import optuna
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    exclude = set(id_columns or []) | {"is_outlier"}
    frame = df.drop_nulls(subset=[target])
    task = _infer_task(frame.get_column(target))
    x = _encode_features(frame, target, exclude)
    if x is None or x.shape[0] < 30:
        return None
    y = _label_codes(frame.get_column(target)) if task == "classification" else frame.get_column(target).cast(pl.Float64).to_numpy()

    n_splits = min(5, max(3, x.shape[0] // 10))
    scoring = "r2" if task == "regression" else "f1_macro"
    estimator_cls = GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier
    seed = config.random_seed

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        )
        est = estimator_cls(random_state=seed, **params)
        return float(np.mean(cross_val_score(est, x, y, cv=n_splits, scoring=scoring)))

    baseline = float(np.mean(cross_val_score(estimator_cls(random_state=seed), x, y, cv=n_splits, scoring=scoring)))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=config.tune_trials, show_progress_bar=False)

    return TuneResult(
        task=task, target=target, metric=scoring, n_trials=config.tune_trials,
        baseline_cv=round(baseline, 4), best_cv=round(study.best_value, 4),
        improvement=round(study.best_value - baseline, 4), best_params=dict(study.best_params),
    )
