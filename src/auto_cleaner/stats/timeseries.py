"""Time-series diagnostics for each numeric series indexed by time.

Per series: stationarity (ADF + KPSS), dominant autocorrelation lags, STL
seasonal/trend decomposition (the de-seasonalisation the climate/finance world
needs), Mann-Kendall trend significance, and a Ljung-Box white-noise test.
Triggered automatically when the specialisation engine finds a time index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from auto_cleaner.config import CleanConfig

__all__ = ["TimeSeriesResult", "timeseries_analysis"]

_NUMERIC = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
_CANDIDATE_PERIODS = (7, 12, 24, 30, 52, 365)


@dataclass(slots=True)
class TimeSeriesResult:
    feature: str
    n: int
    adf_p: float | None = None
    kpss_p: float | None = None
    stationary: bool | None = None
    top_acf_lags: list[int] = field(default_factory=list)
    seasonal_period: int | None = None
    seasonal_strength: float | None = None
    trend_strength: float | None = None
    mk_trend: str | None = None
    mk_p: float | None = None
    ljung_box_p: float | None = None


def timeseries_analysis(
    df: pl.DataFrame, time_index: str, config: CleanConfig | None = None,
    *, id_columns: list[str] | None = None,
) -> list[TimeSeriesResult]:
    """Run the per-series time-series diagnostic suite."""
    config = config or CleanConfig()
    try:
        import numpy as np
    except ImportError:
        return []

    exclude = set(id_columns or []) | {"is_outlier", time_index}
    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt in _NUMERIC and c not in exclude and df.get_column(c).n_unique() > 2]
    if not numeric:
        return []

    grid = (
        df.group_by(time_index).agg([pl.col(c).mean().alias(c) for c in numeric])
        .sort(time_index).drop_nulls()
    )
    if grid.height < 20:
        return []

    out: list[TimeSeriesResult] = []
    for c in numeric[:20]:
        x = grid.get_column(c).to_numpy().astype(float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 20 or np.std(x) == 0:
            continue
        res = TimeSeriesResult(feature=c, n=int(n))

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            res.adf_p = float(adfuller(x, autolag="AIC")[1])
            try:
                res.kpss_p = float(kpss(x, nlags="auto")[1])
            except Exception:  # noqa: BLE001
                res.kpss_p = None
            # Stationary if ADF rejects unit root and KPSS does not reject stationarity.
            adf_stat = res.adf_p is not None and res.adf_p < 0.05
            kpss_stat = res.kpss_p is None or res.kpss_p > 0.05
            res.stationary = bool(adf_stat and kpss_stat)
        except Exception:  # noqa: BLE001
            pass

        try:
            from statsmodels.tsa.stattools import acf

            max_lag = min(40, n // 2 - 1)
            if max_lag >= 1:
                acf_vals = acf(x, nlags=max_lag, fft=True)
                lags = [int(i) for i in range(1, max_lag + 1) if abs(acf_vals[i]) > 0.3]
                res.top_acf_lags = sorted(lags, key=lambda i: -abs(acf_vals[i]))[:5]
        except Exception:  # noqa: BLE001
            pass

        # STL decomposition with an auto-chosen seasonal period.
        try:
            from statsmodels.tsa.seasonal import STL
            from statsmodels.tsa.stattools import acf as _acf

            cand = [p for p in _CANDIDATE_PERIODS if 2 * p <= n]
            if cand:
                acf_full = _acf(x, nlags=max(cand), fft=True)
                period = max(cand, key=lambda p: acf_full[p] if p < len(acf_full) else 0)
                if (period < len(acf_full)) and acf_full[period] > 0.2:
                    stl = STL(x, period=period, robust=True).fit()
                    var_resid = np.var(stl.resid)
                    res.seasonal_period = int(period)
                    res.seasonal_strength = round(float(max(0.0, 1 - var_resid / max(np.var(stl.seasonal + stl.resid), 1e-12))), 3)
                    res.trend_strength = round(float(max(0.0, 1 - var_resid / max(np.var(stl.trend + stl.resid), 1e-12))), 3)
        except Exception:  # noqa: BLE001
            pass

        try:
            import pymannkendall as mk

            t = mk.original_test(x)
            res.mk_trend = str(t.trend)
            res.mk_p = float(t.p)
        except Exception:  # noqa: BLE001
            pass

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb = acorr_ljungbox(x, lags=[min(10, n // 2 - 1)], return_df=True)
            res.ljung_box_p = float(lb["lb_pvalue"].iloc[-1])
        except Exception:  # noqa: BLE001
            pass

        out.append(res)
    return out
