"""Load raw Kaggle files and build the modelling panel.

Panel = one row per (store_nbr, family, date) on a FULL daily calendar from MODEL_START
to LAST_ACTUAL, with:
  sales        observed sales (NaN when the store is closed - NOT zero demand)
  closed       1 if the store was closed: date missing from train.csv (25 Dec) or the
               store sold nothing across all 33 families that day
  onpromotion  count of promoted items in the family/store/day (not a discount depth)
  calendar / holiday flags for the date, matched to the store's city and state
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw():
    train = pd.read_csv(C.RAW / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(C.RAW / "stores.csv")
    hol = pd.read_csv(C.RAW / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(C.RAW / "oil.csv", parse_dates=["date"])
    return train, stores, hol, oil


def store_day_totals(train: pd.DataFrame) -> pd.DataFrame:
    """Total sales per store-day over all families (used to detect closures)."""
    return train.groupby(["store_nbr", "date"], as_index=False)["sales"].sum().rename(columns={"sales": "store_total"})


def holiday_flags(hol: pd.DataFrame, stores: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per store-date holiday flags.

    Day-off types: Holiday (only if NOT transferred), Transfer (the day it moved to),
    Bridge (extra day off), Additional (extra days around a holiday).
    Work Day = a weekend day made a working day (compensates a Bridge).
    Event = national events (e.g. Mother's Day, Black Friday, earthquake aftermath, World Cup).
    Scope: National -> all stores; Regional -> locale_name == store state; Local -> == store city.
    """
    h = hol.copy()
    dayoff = ((h["type"] == "Holiday") & (~h["transferred"])) | h["type"].isin(["Transfer", "Bridge", "Additional"])
    h["is_holiday"] = dayoff.astype(int)
    h["is_workday"] = (h["type"] == "Work Day").astype(int)
    h["is_event"] = (h["type"] == "Event").astype(int)
    h["is_national_holiday"] = (dayoff & (h["locale"] == "National")).astype(int)
    rows = []
    for _, s in stores.iterrows():
        m = (h["locale"] == "National") | ((h["locale"] == "Regional") & (h["locale_name"] == s["state"])) | (
            (h["locale"] == "Local") & (h["locale_name"] == s["city"]))
        g = h[m].groupby("date")[["is_holiday", "is_workday", "is_event", "is_national_holiday"]].max()
        g = g.reindex(dates, fill_value=0)
        g["store_nbr"] = s["store_nbr"]
        rows.append(g.rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)


def is_payday(dates: pd.Series) -> pd.Series:
    """Ecuador public-sector wages are paid on the 15th and the last day of the month."""
    d = pd.to_datetime(dates)
    return ((d.dt.day == 15) | d.dt.is_month_end).astype(int)


def load_future_promo() -> pd.DataFrame:
    """test.csv: 2017-08-16..08-31 with the PLANNED promo counts but no sales.
    Used only as the known promo plan for the forward forecast - never for evaluation."""
    t = pd.read_csv(C.RAW / "test.csv", parse_dates=["date"])
    return t[["date", "store_nbr", "family", "onpromotion"]]


def build_panel(store_list, families=C.FAMILIES, start=C.MODEL_START, end=C.LAST_ACTUAL, raw=None,
                include_future: bool = True) -> pd.DataFrame:
    """include_future adds 2017-08-16..08-31 rows (is_future=1, sales NaN, promo from test.csv)."""
    train, stores, hol, _ = raw if raw is not None else load_raw()
    if include_future:
        fut = load_future_promo()
        train = pd.concat([train, fut.assign(sales=np.nan, id=-1)], ignore_index=True)
        end = fut["date"].max()
    dates = pd.date_range(start, end, freq="D")
    tot = store_day_totals(train[train["date"] <= C.LAST_ACTUAL])
    sub = train[train["store_nbr"].isin(store_list) & train["family"].isin(families)]
    idx = pd.MultiIndex.from_product([sorted(store_list), list(families), dates], names=["store_nbr", "family", "date"])
    p = sub.set_index(["store_nbr", "family", "date"])[["sales", "onpromotion"]].reindex(idx).reset_index()
    p = p.merge(tot, on=["store_nbr", "date"], how="left")
    p["is_future"] = (p["date"] > pd.Timestamp(C.LAST_ACTUAL)).astype(int)
    p["closed"] = ((p["store_total"].isna() | (p["store_total"] == 0)) & (p["is_future"] == 0)).astype(int)
    p.loc[p["closed"] == 1, "sales"] = np.nan  # closure != zero demand
    p["onpromotion"] = p["onpromotion"].fillna(0).astype(int)
    p = p.drop(columns="store_total")
    p = p.merge(holiday_flags(hol, stores, dates), on=["store_nbr", "date"], how="left")
    p = p.merge(stores[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")
    p["payday"] = is_payday(p["date"])
    p["quake"] = p["date"].between(C.QUAKE_START, C.QUAKE_END).astype(int)
    p["series_id"] = p["store_nbr"].astype(str) + "|" + p["family"]
    return p.sort_values(["series_id", "date"]).reset_index(drop=True)
