"""Empirical (split-conformal style) prediction intervals.

For the champion system on CALIBRATION backtests we compute *scaled residuals*
    r = (actual - forecast) / scale_T
where scale_T = mean sales over the 28 days up to and including origin T (known at T,
closed days ignored). Scaling lets one family-level quantile serve big and small stores.
Per family x horizon bucket (1-7, 8-14) we take the 10th and 90th percentiles of r:
    P10 = max(0, F + q10 * scale_T),   P90 = max(0, F + q90 * scale_T)
Target: 80 % of actuals inside [P10, P90]. This is an empirical target, not a guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .backtest import evaluable

LO, HI = (1 - C.INTERVAL_LEVEL) / 2, 1 - (1 - C.INTERVAL_LEVEL) / 2
SCALE_DAYS = 28


def bucket(h) -> np.ndarray:
    return np.where(np.asarray(h) <= 7, "1-7", "8-14")


def origin_scales(panel: pd.DataFrame, origins) -> pd.DataFrame:
    """scale_T per (series, origin): mean of sales in (T-27 .. T), NaN-aware. Uses data <= T only."""
    out = []
    for T in pd.to_datetime(list(origins)):
        w = panel[(panel["date"] > T - pd.Timedelta(days=SCALE_DAYS)) & (panel["date"] <= T)]
        s = w.groupby("series_id")["sales"].mean().rename("scale").reset_index()
        s["origin"] = T
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s["scale"] = s["scale"].clip(lower=1e-6)
    return s


def attach_scale(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    sc = origin_scales(panel, rows["origin"].unique())
    return rows.merge(sc, on=["series_id", "origin"], how="left")


def fit_quantiles(rows: pd.DataFrame, panel: pd.DataFrame, by=("family",)) -> pd.DataFrame:
    r = attach_scale(evaluable(rows), panel)
    r["bucket"] = bucket(r["horizon"])
    r["res"] = (r["actual"] - r["forecast"]) / r["scale"]
    q = r.groupby(list(by) + ["bucket"])["res"].quantile([LO, HI]).unstack()
    q.columns = ["q_lo", "q_hi"]
    q["n"] = r.groupby(list(by) + ["bucket"]).size()
    return q.reset_index()


def apply_intervals(rows: pd.DataFrame, panel: pd.DataFrame, q: pd.DataFrame, by=("family",)) -> pd.DataFrame:
    r = attach_scale(rows, panel) if "scale" not in rows else rows.copy()
    r["bucket"] = bucket(r["horizon"])
    r = r.merge(q[list(by) + ["bucket", "q_lo", "q_hi"]], on=list(by) + ["bucket"], how="left")
    r["p10"] = np.maximum(0.0, r["forecast"] + r["q_lo"] * r["scale"])
    r["p90"] = np.maximum(0.0, r["forecast"] + r["q_hi"] * r["scale"])
    r["p10"] = np.minimum(r["p10"], r["forecast"])  # keep the point forecast inside its band
    r["p90"] = np.maximum(r["p90"], r["forecast"])
    return r.drop(columns=["q_lo", "q_hi"])


# ---- chain-total band (sum of series P10s is NOT the P10 of the sum) ----
def total_rows(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows.groupby(["origin", "target_date", "horizon"]).agg(
        forecast=("forecast", "sum"), actual=("actual", lambda a: a.sum(min_count=len(a))), closed=("closed", "max"))
    return ok.reset_index()


def fit_total_quantiles(rows: pd.DataFrame) -> pd.DataFrame:
    t = total_rows(rows)
    t = t[(t["closed"] == 0) & t["actual"].notna()]
    t["bucket"] = bucket(t["horizon"])
    t["res"] = (t["actual"] - t["forecast"]) / t["forecast"].clip(lower=1e-6)  # relative residual of the total
    q = t.groupby("bucket")["res"].quantile([LO, HI]).unstack()
    q.columns = ["q_lo", "q_hi"]
    q["n"] = t.groupby("bucket").size()
    return q.reset_index()


def apply_total_band(t: pd.DataFrame, qt: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["bucket"] = bucket(t["horizon"])
    t = t.merge(qt[["bucket", "q_lo", "q_hi"]], on="bucket", how="left")
    t["p10"] = np.maximum(0.0, t["forecast"] * (1 + t["q_lo"]))
    t["p90"] = t["forecast"] * (1 + t["q_hi"])
    return t.drop(columns=["q_lo", "q_hi"])


# ---- 14-day window-total band per series (added after independent review) ----
# Summing 14 daily P10s/P90s is NOT an 80 % band for the 14-day total (it covered ~94 %).
# Same relative-residual conformal method as the chain-total band (fit_total_quantiles /
# apply_total_band), applied to each series' 14-day sum, with quantiles per family.
def window_totals(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (series, origin): 14-day forecast and actual sums. `complete` = every day open with an actual."""
    g = rows.groupby(["series_id", "family", "origin"])
    t = g.agg(forecast=("forecast", "sum"), actual=("actual", lambda a: a.sum(min_count=len(a))),
              n_days=("horizon", "size"), n_closed=("closed", "sum"),
              n_actual=("actual", lambda a: int(a.notna().sum()))).reset_index()
    t["complete"] = (t["n_days"] == C.HORIZON) & (t["n_closed"] == 0) & (t["n_actual"] == C.HORIZON)
    return t


def fit_window_quantiles(rows: pd.DataFrame, by=("family",)) -> pd.DataFrame:
    t = window_totals(rows)
    t = t[t["complete"]]
    t["res"] = (t["actual"] - t["forecast"]) / t["forecast"].clip(lower=1e-6)
    q = t.groupby(list(by))["res"].quantile([LO, HI]).unstack()
    q.columns = ["q_lo", "q_hi"]
    q["n"] = t.groupby(list(by)).size()
    return q.reset_index()


def apply_window_band(rows: pd.DataFrame, q: pd.DataFrame, by=("family",)) -> pd.DataFrame:
    t = window_totals(rows).merge(q[list(by) + ["q_lo", "q_hi"]], on=list(by), how="left")
    t["p10"] = np.maximum(0.0, t["forecast"] * (1 + t["q_lo"]))
    t["p90"] = t["forecast"] * (1 + t["q_hi"])
    return t.drop(columns=["q_lo", "q_hi"])
