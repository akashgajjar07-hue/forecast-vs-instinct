"""Baseline forecasters. All follow the harness signature

    model_fn(view, origin, horizon, **params) -> DataFrame[series_id, horizon, forecast]

and use ONLY rows of `view` with date <= origin (they filter it themselves, so they are
leak-free even if handed an unmasked panel). Closed days have sales NaN and are skipped.

- naive            last observed sales at or before T, for every h
- seasonal_naive   same weekday last week: sales at T+h-7*ceil(h/7); if that day is
                   closed/NaN, step back further weeks on the same weekday
- moving_average   mean of the last `window` calendar days <= T, NaN (closed) days ignored
- holt_winters     statsmodels ExponentialSmoothing, additive weekly seasonality (period 7),
                   fitted per series per origin on the trailing `fit_days` days. NaN days in the
                   fitting window are filled with the same-weekday value of the previous week
                   (fitting only). Fit failures / non-finite forecasts fall back to seasonal naive.
"""
from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# Fallback log for holt_winters, appended in the parent process: one dict per fallback
# {origin, series_id, reason}. The runner reads and resets it.
HW_FALLBACKS: list[dict] = []


def _history_wide(view: pd.DataFrame, origin) -> pd.DataFrame:
    """dates x series_id matrix of sales for date <= origin, on a full daily calendar."""
    origin = pd.Timestamp(origin)
    h = view.loc[view["date"] <= origin, ["date", "series_id", "sales"]]
    w = h.pivot(index="date", columns="series_id", values="sales").sort_index()
    full = pd.date_range(w.index.min(), origin, freq="D")
    return w.reindex(full)


def _long(values: dict[str, np.ndarray], horizon: int) -> pd.DataFrame:
    """{series_id: array(len horizon)} -> long frame."""
    sids = list(values)
    arr = np.vstack([np.asarray(values[s], dtype=float) for s in sids])
    return pd.DataFrame({
        "series_id": np.repeat(sids, horizon),
        "horizon": np.tile(np.arange(1, horizon + 1), len(sids)),
        "forecast": arr.ravel(),
    })


def _flat(level: pd.Series, horizon: int) -> pd.DataFrame:
    return _long({s: np.full(horizon, v) for s, v in level.items()}, horizon)


def naive(view, origin, horizon, **_):
    w = _history_wide(view, origin)
    return _flat(w.ffill().iloc[-1], horizon)


def _seasonal_naive_matrix(w: pd.DataFrame, horizon: int) -> np.ndarray:
    """(n_series, horizon) seasonal-naive forecasts from a full-calendar wide history."""
    vals = w.to_numpy()  # rows = dates ending at T
    n = len(vals)
    last_obs = w.ffill().iloc[-1].to_numpy()
    out = np.full((vals.shape[1], horizon), np.nan)
    for h in range(1, horizon + 1):
        lag = 7 * math.ceil(h / 7)
        idx = n - 1 + h - lag  # row index of T+h-lag
        col = np.full(vals.shape[1], np.nan)
        while idx >= 0 and np.isnan(col).any():
            col = np.where(np.isnan(col), vals[idx], col)
            idx -= 7
        out[:, h - 1] = np.where(np.isnan(col), last_obs, col)  # no same-weekday history at all
    return out


def seasonal_naive(view, origin, horizon, **_):
    w = _history_wide(view, origin)
    m = _seasonal_naive_matrix(w, horizon)
    return _long(dict(zip(w.columns, m)), horizon)


def moving_average(view, origin, horizon, window: int = 28, **_):
    w = _history_wide(view, origin)
    lvl = w.iloc[-window:].mean(skipna=True)
    lvl = lvl.fillna(w.ffill().iloc[-1])  # whole window closed (never in practice)
    return _flat(lvl, horizon)


# ---------------------------------------------------------------- Holt-Winters
def _fill_same_weekday(y: np.ndarray) -> np.ndarray:
    """Fill NaN with the value 7 days earlier (repeat); leading NaNs with the value 7 days later."""
    y = y.copy()
    for i in range(7, len(y)):
        if np.isnan(y[i]):
            y[i] = y[i - 7]
    for i in range(len(y) - 8, -1, -1):
        if np.isnan(y[i]):
            y[i] = y[i + 7]
    return y


def _fit_one(args):
    """Worker: returns (forecast array or None, reason)."""
    y, horizon, trend = args
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        y = _fill_same_weekday(y)
        if np.isnan(y).any() or len(y) < 28:
            return None, "insufficient_history"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ExponentialSmoothing(
                y, trend=trend, damped_trend=trend is not None, seasonal="add",
                seasonal_periods=7, initialization_method="estimated",
            ).fit()
            f = np.asarray(m.forecast(horizon), dtype=float)
        if not np.isfinite(f).all():
            return None, "non_finite_forecast"
        return f, None
    except Exception as e:  # noqa: BLE001 - any fit failure -> fallback
        return None, f"fit_error:{type(e).__name__}"


_POOL: ProcessPoolExecutor | None = None


def _pool(n_jobs: int) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None or _POOL._max_workers != n_jobs:  # noqa: SLF001
        if _POOL is not None:
            _POOL.shutdown()
        _POOL = ProcessPoolExecutor(max_workers=n_jobs)
    return _POOL


def shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None


def holt_winters(view, origin, horizon, trend: str | None = None, fit_days: int = 364,
                 n_jobs: int = 1, _fit=None, **_):
    """trend: None (level + weekly seasonality) or "add" (additive damped trend).
    `_fit` lets tests inject a failing fitter to exercise the fallback."""
    w = _history_wide(view, origin)
    tail = w.iloc[-fit_days:]
    jobs = [(tail[s].to_numpy(dtype=float), horizon, trend) for s in w.columns]
    fit = _fit or _fit_one
    if n_jobs > 1 and _fit is None:
        results = list(_pool(n_jobs).map(fit, jobs, chunksize=8))
    else:
        results = [fit(j) for j in jobs]
    snaive = _seasonal_naive_matrix(w, horizon)
    out = {}
    for i, (s, (f, reason)) in enumerate(zip(w.columns, results)):
        if f is None:
            HW_FALLBACKS.append({"origin": str(pd.Timestamp(origin).date()), "series_id": s, "reason": reason})
            f = snaive[i]
        out[s] = f
    return _long(out, horizon)


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "holt_winters": holt_winters,
}
