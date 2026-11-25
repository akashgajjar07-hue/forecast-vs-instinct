"""Forecast accuracy metrics.

Conventions (used everywhere in the project, documented in METHODOLOGY.md):
  WAPE = sum|A - F| / sum|A| * 100          (lower is better)
  Bias = sum(F - A) / sum(A) * 100          (positive = over-forecast)
  MAE  = mean|A - F|
  RMSE = sqrt(mean((A - F)^2))
Zero denominators never raise or return inf: the metric is NaN and the
returned `MetricResult.flag` explains why.
Note: actuals are *observed sales*, not unconstrained demand.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

ZERO_ACTUALS = "zero_actuals"
EMPTY = "empty"


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def clip_forecast(f) -> np.ndarray:
    """Sales cannot be negative: clip predictions at 0 (NaN preserved)."""
    f = _arr(f)
    return np.where(np.isnan(f), np.nan, np.maximum(f, 0.0))


@dataclass
class MetricResult:
    wape: float
    bias: float
    mae: float
    rmse: float
    n: int
    sum_actual: float
    sum_abs_error: float
    flag: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(a, f):
    a, f = _arr(a), _arr(f)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {f.shape}")
    m = ~(np.isnan(a) | np.isnan(f))
    return a[m], f[m]


def wape(a, f) -> float:
    a, f = _clean(a, f)
    den = np.abs(a).sum()
    return float("nan") if len(a) == 0 or den == 0 else float(np.abs(a - f).sum() / den * 100)


def bias(a, f) -> float:
    a, f = _clean(a, f)
    den = a.sum()
    return float("nan") if len(a) == 0 or den == 0 else float((f - a).sum() / den * 100)


def mae(a, f) -> float:
    a, f = _clean(a, f)
    return float("nan") if len(a) == 0 else float(np.abs(a - f).mean())


def rmse(a, f) -> float:
    a, f = _clean(a, f)
    return float("nan") if len(a) == 0 else float(np.sqrt(((a - f) ** 2).mean()))


def all_metrics(a, f) -> MetricResult:
    ac, fc = _clean(a, f)
    flag = EMPTY if len(ac) == 0 else (ZERO_ACTUALS if np.abs(ac).sum() == 0 else None)
    return MetricResult(
        wape=wape(a, f), bias=bias(a, f), mae=mae(a, f), rmse=rmse(a, f), n=int(len(ac)),
        sum_actual=float(ac.sum()), sum_abs_error=float(np.abs(ac - fc).sum()), flag=flag,
    )


def coverage(a, lo, hi) -> float:
    """Share (%) of actuals inside [lo, hi] (inclusive)."""
    a, lo, hi = _arr(a), _arr(lo), _arr(hi)
    m = ~(np.isnan(a) | np.isnan(lo) | np.isnan(hi))
    if m.sum() == 0:
        return float("nan")
    return float(((a[m] >= lo[m]) & (a[m] <= hi[m])).mean() * 100)


def mean_width(lo, hi) -> float:
    lo, hi = _arr(lo), _arr(hi)
    m = ~(np.isnan(lo) | np.isnan(hi))
    return float("nan") if m.sum() == 0 else float((hi[m] - lo[m]).mean())


def grouped_metrics(df: pd.DataFrame, by, actual="actual", forecast="forecast") -> pd.DataFrame:
    """Metrics per group. WAPE/bias are ratio-of-sums within the group (not mean of ratios)."""
    rows = []
    for key, g in df.groupby(by, observed=True, dropna=False):
        r = all_metrics(g[actual], g[forecast]).to_dict()
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip([by] if isinstance(by, str) else by, key)), **r})
    return pd.DataFrame(rows)


def bucket_totals(df: pd.DataFrame, keys, horizon="horizon", actual="actual", forecast="forecast") -> pd.DataFrame:
    """Week-1 (h 1-7) / week-2 (h 8-14) bucket totals per series-origin, as planners are measured."""
    d = df.assign(bucket=np.where(df[horizon] <= 7, "week1", "week2"))
    return d.groupby(list(keys) + ["bucket"], observed=True)[[actual, forecast]].sum(min_count=1).reset_index()
