"""Time-series forecasting (ARIMA vs Holt-Winters/ETS, auto-selected).

For each numeric series indexed by time, fits a small ARIMA order grid and an
exponential-smoothing model, picks the lower-AIC one, and projects ``horizon``
steps ahead with 95% prediction intervals. This is what answers "what might
happen next" — with the honest caveat that forecasts are model-based
extrapolations, not certainties.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["ForecastResult", "forecast_series"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_ARIMA_GRID = [(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1, 2)]


@dataclass(slots=True)
class ForecastResult:
    feature: str
    model: str
    aic: float
    horizon: int
    last_value: float
    forecast: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)


def _best_arima(y, seed: int):
    from statsmodels.tsa.arima.model import ARIMA

    best = (float("inf"), None, None)
    for order in _ARIMA_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(y, order=order).fit()
            if fit.aic < best[0]:
                best = (float(fit.aic), order, fit)
        except Exception:  # noqa: BLE001
            continue
    return best  # (aic, order, fitted)


def _ets(y, seasonal_periods: int | None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    kwargs = dict(trend="add", initialization_method="estimated")
    if seasonal_periods and seasonal_periods >= 2 and len(y) >= 2 * seasonal_periods:
        kwargs.update(seasonal="add", seasonal_periods=seasonal_periods)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(y, **kwargs).fit()


def forecast_series(
    df: pl.DataFrame, time_index: str, config: CleanConfig | None = None,
    *, horizon: int | None = None, id_columns: list[str] | None = None,
) -> list[ForecastResult]:
    """Forecast each numeric series ``horizon`` steps ahead with 95% intervals."""
    config = config or CleanConfig()
    horizon = horizon or config.forecast_horizon
    try:
        import numpy as np
    except ImportError:
        return []

    exclude = set(id_columns or []) | {"is_outlier", time_index}
    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2]
    if not numeric:
        return []

    grid = df.group_by(time_index).agg([pl.col(c).mean().alias(c) for c in numeric]).sort(time_index).drop_nulls()
    n = grid.height
    if n < 20:
        return []

    out: list[ForecastResult] = []
    for c in numeric[:10]:
        y = grid.get_column(c).to_numpy().astype(float)
        if not np.all(np.isfinite(y)) or np.std(y) == 0:
            continue

        aic_arima, order, arima_fit = _best_arima(y, config.random_seed)
        ets_fit, aic_ets = None, float("inf")
        try:
            ets_fit = _ets(y, None)
            aic_ets = float(ets_fit.aic)
        except Exception:  # noqa: BLE001
            pass

        try:
            if arima_fit is not None and aic_arima <= aic_ets:
                fc = arima_fit.get_forecast(steps=horizon)
                mean = np.asarray(fc.predicted_mean, dtype=float)
                ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
                lower, upper = ci[:, 0], ci[:, 1]
                model = f"ARIMA{order}"
                aic = aic_arima
            elif ets_fit is not None:
                mean = np.asarray(ets_fit.forecast(horizon), dtype=float)
                resid_sd = float(np.std(ets_fit.resid)) if hasattr(ets_fit, "resid") else 0.0
                lower, upper = mean - 1.96 * resid_sd, mean + 1.96 * resid_sd
                model, aic = "ETS(add trend)", aic_ets
            else:
                continue
        except Exception:  # noqa: BLE001
            continue

        out.append(
            ForecastResult(
                feature=c, model=model, aic=round(aic, 1), horizon=horizon,
                last_value=round(float(y[-1]), 4),
                forecast=[round(float(v), 4) for v in mean],
                lower=[round(float(v), 4) for v in lower],
                upper=[round(float(v), 4) for v in upper],
            )
        )
    return out
