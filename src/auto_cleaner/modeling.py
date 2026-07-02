"""Baseline AutoML — cross-validated *benchmarks*, not a finished model.

Given a target, this detects the task (classification vs regression), runs a
panel of sensible baselines (dummy, linear/logistic, random forest, gradient
boosting) under k-fold cross-validation, reports the headline metrics, computes
permutation importance for the best model, and flags likely **data leakage**.
It is explicitly a starting benchmark for a human to build on — not a deployable
model (no hyper-parameter tuning, no nested CV, ordinal-encoded categoricals).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["ModelScore", "ModelReport", "run_baseline_models"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_CAVEATS = (
    "Default hyper-parameters and ordinal-encoded categoricals — tune + one-hot for production.",
    "Cross-validated scores reduce optimism but do not replace a locked held-out test set.",
    "High importance on an ID-like or near-perfect feature usually signals leakage, not signal.",
)


@dataclass(slots=True)
class ModelScore:
    name: str
    metrics: dict[str, float]


@dataclass(slots=True)
class ModelReport:
    task: str
    target: str
    n: int
    scores: list[ModelScore] = field(default_factory=list)
    best: str | None = None
    importances: list[tuple[str, float]] = field(default_factory=list)
    shap_importances: list[tuple[str, float]] = field(default_factory=list)
    leakage: list[str] = field(default_factory=list)
    note: str | None = None
    caveats: tuple[str, ...] = _CAVEATS


def _infer_task(series: pl.Series) -> str:
    if series.dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
        return "classification"
    if series.dtype in _NUMERIC and series.dtype not in (pl.Float32, pl.Float64):
        return "classification" if series.n_unique() <= 20 else "regression"
    return "regression"


def _encode(df: pl.DataFrame, target: str, exclude: set[str]):
    """Build (X, feature_names, discrete_mask) with ordinal-coded categoricals."""
    import numpy as np

    cols, names, discrete = [], [], []
    for c, dt in zip(df.columns, df.dtypes):
        if c == target or c in exclude:
            continue
        s = df.get_column(c)
        if dt in _NUMERIC:
            cols.append(s.cast(pl.Float64).fill_null(s.median()).to_numpy())
            discrete.append(False)
        elif dt == pl.Boolean:
            cols.append(s.cast(pl.Int8).fill_null(-1).to_numpy().astype(float))
            discrete.append(True)
        elif dt in (pl.Utf8, pl.Categorical):
            if s.n_unique() > 50:
                continue
            cols.append(s.cast(pl.Categorical).to_physical().fill_null(-1).to_numpy().astype(float))
            discrete.append(True)
        else:
            continue
        names.append(c)
    if not cols:
        return None, [], []
    return np.column_stack(cols), names, discrete


def run_baseline_models(
    df: pl.DataFrame,
    target: str,
    config: CleanConfig | None = None,
    *,
    id_columns: list[str] | None = None,
) -> ModelReport | None:
    """Cross-validated baseline benchmark for ``target``."""
    config = config or CleanConfig()
    if target not in df.columns:
        return None
    try:
        import numpy as np
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            RandomForestClassifier, RandomForestRegressor,
        )
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    exclude = set(id_columns or []) | {"is_outlier"}
    frame = df.drop_nulls(subset=[target])
    task = _infer_task(frame.get_column(target))
    X, names, _ = _encode(frame, target, exclude)
    if X is None or X.shape[0] < 20:
        return ModelReport(task, target, 0, note="Too few complete rows for modelling.")

    seed = config.random_seed
    report = ModelReport(task=task, target=target, n=int(X.shape[0]))
    n_splits = min(5, max(2, X.shape[0] // 10))

    # Leakage heuristic: a feature almost perfectly aligned with the target.
    if task == "regression":
        y = frame.get_column(target).cast(pl.Float64).to_numpy()
        with np.errstate(invalid="ignore"):
            for i, nm in enumerate(names):
                if np.nanstd(X[:, i]) > 0:
                    r = np.corrcoef(X[:, i], y)[0, 1]
                    if abs(r) > 0.98:
                        report.leakage.append(f"'{nm}' correlates |r|={abs(r):.3f} with target")
        models = {
            "Dummy (mean)": DummyRegressor(),
            "Linear": make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=seed),
        }
        scoring = {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}
        primary = "R2"
    else:
        yt = frame.get_column(target)
        if yt.dtype == pl.Boolean:
            y = yt.cast(pl.Int64).to_numpy()
        elif yt.dtype in _NUMERIC:           # integer class labels — already codes
            y = yt.to_numpy()
        else:
            y = yt.cast(pl.Categorical).to_physical().to_numpy()
        n_classes = len(np.unique(y))
        models = {
            "Dummy (prior)": DummyClassifier(strategy="most_frequent"),
            "Logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        }
        scoring = {"Accuracy": "accuracy", "F1_macro": "f1_macro"}
        if n_classes == 2:
            scoring["ROC_AUC"] = "roc_auc"
        primary = "F1_macro"

    best_name, best_score, best_estimator = None, -np.inf, None
    for name, est in models.items():
        try:
            cv = cross_validate(est, X, y, cv=n_splits, scoring=scoring, n_jobs=-1)
        except Exception as exc:  # noqa: BLE001
            report.scores.append(ModelScore(name, {"error": float("nan")}))
            continue
        metrics = {}
        for key, sk in scoring.items():
            vals = cv[f"test_{key}"]
            # invert negated error metrics back to positive
            metrics[key] = float(np.mean(np.abs(vals) if sk.startswith("neg_") else vals))
        report.scores.append(ModelScore(name, metrics))
        if "Dummy" not in name and metrics.get(primary, -np.inf) > best_score:
            best_name, best_score, best_estimator = name, metrics[primary], est

    report.best = best_name
    # Permutation importance for the best (non-dummy) model.
    if best_estimator is not None:
        try:
            best_estimator.fit(X, y)
            pi = permutation_importance(best_estimator, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
            order = np.argsort(pi.importances_mean)[::-1]
            report.importances = [
                (names[i], float(pi.importances_mean[i])) for i in order[:15]
            ]
        except Exception:  # noqa: BLE001
            pass

        # SHAP contributions for tree-based winners (fast TreeExplainer).
        if best_name and ("Forest" in best_name or "Boosting" in best_name):
            try:
                import shap

                sample = X if X.shape[0] <= 500 else X[np.random.default_rng(seed).choice(X.shape[0], 500, replace=False)]
                sv = np.array(shap.TreeExplainer(best_estimator).shap_values(sample))
                while sv.ndim > 2:           # collapse multiclass dimension
                    sv = np.abs(sv).mean(axis=0)
                mean_abs = np.abs(sv).mean(axis=0)
                order = np.argsort(mean_abs)[::-1]
                report.shap_importances = [(names[i], round(float(mean_abs[i]), 4)) for i in order[:15]]
            except Exception:  # noqa: BLE001
                pass
    return report
