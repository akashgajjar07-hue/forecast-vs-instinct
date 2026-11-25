"""Rolling-origin backtest harness - the leakage boundary.

Every model is a function  model_fn(view, origin, horizon, **params) -> DataFrame
with columns [series_id, horizon, forecast]. It never sees the raw panel: it gets a
*masked view* in which
  - sales after the origin are NaN (structurally impossible to peek), and
  - for promo variant "B" (no future promo info) onpromotion after the origin is NaN.
Variant "A" (known promo plan) keeps the promo count on target dates, as a planner
with an agreed promo calendar would have.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import config as C

BACKTEST_COLUMNS = [
    "model", "variant", "stage", "origin", "series_id", "store_nbr", "family", "horizon",
    "target_date", "forecast", "actual", "closed", "onpromotion", "is_holiday",
]
PROMO_VARIANTS = {"A": "known promo plan (onpromotion at T+h used)",
                  "B": "no future promo info (promo history up to T only)"}


def masked_view(panel: pd.DataFrame, origin, variant: str = "A") -> pd.DataFrame:
    origin = pd.Timestamp(origin)
    v = panel.copy()
    after = v["date"] > origin
    v.loc[after, "sales"] = np.nan
    v.loc[after, "closed"] = 0  # closure status after T is unknown at T (known calendar closures handled by hard flags)
    if variant == "B":
        v.loc[after, "onpromotion"] = np.nan
    elif variant != "A":
        raise ValueError(variant)
    return v


def target_frame(panel: pd.DataFrame, origin, horizon: int = C.HORIZON) -> pd.DataFrame:
    """All (series, h) target rows for an origin, with actuals from the UNMASKED panel."""
    origin = pd.Timestamp(origin)
    t = panel[(panel["date"] > origin) & (panel["date"] <= origin + pd.Timedelta(days=horizon))]
    t = t[["series_id", "store_nbr", "family", "date", "sales", "closed", "onpromotion", "is_holiday"]].rename(
        columns={"date": "target_date", "sales": "actual"})
    t["horizon"] = (t["target_date"] - origin).dt.days
    return t


def run_origin(model_name: str, model_fn: Callable, panel: pd.DataFrame, origin, stage: str,
               variant: str = "A", horizon: int = C.HORIZON, **params) -> pd.DataFrame:
    origin = pd.Timestamp(origin)
    view = masked_view(panel, origin, variant)
    fc = model_fn(view, origin, horizon, **params)
    missing = {"series_id", "horizon", "forecast"} - set(fc.columns)
    if missing:
        raise ValueError(f"{model_name} output missing {missing}")
    tgt = target_frame(panel, origin, horizon)
    out = tgt.merge(fc[["series_id", "horizon", "forecast"]], on=["series_id", "horizon"], how="left")
    n_exp = panel["series_id"].nunique() * horizon
    if len(out) != n_exp or out["forecast"].isna().any():
        raise ValueError(f"{model_name} @ {origin.date()}: expected {n_exp} non-null forecasts, got "
                         f"{len(out)} rows / {int(out['forecast'].isna().sum())} NaN")
    out["forecast"] = np.maximum(out["forecast"].astype(float), 0.0)  # sales cannot be negative
    out["model"], out["variant"], out["stage"], out["origin"] = model_name, variant, stage, origin
    return out[BACKTEST_COLUMNS]


def run_backtest(model_name: str, model_fn: Callable, panel: pd.DataFrame, origins, variant: str = "A",
                 **params) -> pd.DataFrame:
    """origins: iterable of (stage, Timestamp)."""
    return pd.concat([run_origin(model_name, model_fn, panel, o, s, variant, **params) for s, o in origins],
                     ignore_index=True)


def evaluable(bt: pd.DataFrame) -> pd.DataFrame:
    """Identical evaluation population for every model: open target days with an observed actual."""
    return bt[(bt["closed"] == 0) & bt["actual"].notna()]
