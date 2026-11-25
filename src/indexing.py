"""Index scale for everything published per series (Kaggle rule 7B: raw competition data is not redistributed).

Index base (never published):
  series base  = mean observed daily sales of the series over the training period before the first selection origin,
                 MODEL_START .. first selection origin - 1 day (2015-06-01 .. 2017-02-06), open days only
                 (closed == 0 and sales observed).
  chain base   = mean of the 96-series daily total over the same period, on days when all 96 series were open.
  promo base   = mean daily planned/observed promoted-item count of the series on open days from MODEL_START up to and
                 including the forecast origin (None when it is 0).
Index = value / base x 100, so 100 = an average training-period day. A 14-day quantity (total of 14 daily values) is
published as its mean daily value over the 14 days / base x 100.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

SCALE_LABEL = "index (series training-period mean = 100)"
CHAIN_SCALE_LABEL = "index (chain-total training-period mean = 100)"
PROMO_SCALE_LABEL = "promo index (series mean promoted-item count up to the origin = 100)"
AXIS_LABEL = "Index (training-period mean = 100)"


def base_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    cal = json.loads((C.CONFIG / "split_calendar.json").read_text())
    first_sel = pd.Timestamp(cal["stages"]["selection"][0])
    return pd.Timestamp(C.MODEL_START), first_sel - pd.Timedelta(days=1)


def _open(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(panel["closed"] == 0) & panel["sales"].notna()]


def series_base(panel: pd.DataFrame) -> pd.Series:
    """Per-series index base (mean daily sales, open days, base window). Index: series_id."""
    lo, hi = base_window()
    p = _open(panel[(panel["date"] >= lo) & (panel["date"] <= hi)])
    b = p.groupby("series_id")["sales"].mean()
    assert (b > 0).all(), "a series has a non-positive index base"
    return b


def chain_base(panel: pd.DataFrame) -> float:
    """Mean daily total of all series over the base window, days when every series was open."""
    lo, hi = base_window()
    p = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]
    n_series = p["series_id"].nunique()
    g = p.assign(ok=((p["closed"] == 0) & p["sales"].notna()).astype(int)).groupby("date")
    tot = g["sales"].sum(min_count=1)
    full = g["ok"].sum() == n_series
    return float(tot[full].mean())


def promo_base(panel: pd.DataFrame, origin) -> pd.Series:
    """Per-series mean promoted-item count on open days MODEL_START..origin (NaN where the mean is 0)."""
    o = pd.Timestamp(origin)
    p = panel[(panel["date"] >= pd.Timestamp(C.MODEL_START)) & (panel["date"] <= o) & (panel["closed"] == 0)]
    m = p.groupby("series_id")["onpromotion"].mean()
    return m.where(m > 0)


def to_index(values, base):
    """value / base x 100 (works on scalars, arrays and aligned Series)."""
    if isinstance(values, (list, tuple)):
        values = np.asarray(values, dtype=float)
    return values / base * 100.0
