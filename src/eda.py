"""Demand Intelligence / EDA: runs the DuckDB queries in sql/ and assembles the published EDA outputs.

Every number in outputs/eda/*.json comes from one of the sql/*.sql files. The SQL files query named views
(panel, holidays, transactions, oil, train) that `connect()` registers, so the same SQL runs on the real
panel and on a tiny synthetic frame in tests.

Leakage guard: by default every query is limited to history up to EDA_CUTOFF = the last calibration target date
(2017-06-27). The holdout period (2017-06-28 .. 2017-08-15) is never touched by the EDA.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config as C

SQL_DIR = C.ROOT / "sql"
PANEL_PATH = C.PROCESSED / "panel.parquet"


def eda_cutoff() -> str:
    cal = json.loads((C.CONFIG / "split_calendar.json").read_text())
    return cal["target_windows"]["calibration"][1]


EDA_CUTOFF = eda_cutoff()

ANALOGUE_WINDOW = (
    "Holiday day vs same-weekday baseline: national day off (Holiday not transferred, Transfer, Bridge) "
    "compared with the mean of the same weekday 1 and 2 weeks before and after (t-14, t-7, t+7, t+14), "
    "baseline days excluding any holiday, Work Day, Event or earthquake-month day; >=2 valid baseline days per store; "
    "12 stores summed per holiday; family value = mean over holidays; history 2015-06-01..{cutoff}."
)


# ----------------------------------------------------------------------------------------------- connection
def connect(panel: pd.DataFrame | str | Path | None = None, holidays: pd.DataFrame | None = None,
            transactions: pd.DataFrame | None = None, oil: pd.DataFrame | None = None,
            raw_dir: Path = C.RAW) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with the views the SQL files expect.
    DataFrames override the default files (used by tests). `date` columns are cast to DATE."""
    con = duckdb.connect()

    def view(name: str, src, default_sql: str | None):
        if isinstance(src, pd.DataFrame):
            con.register(f"_{name}_df", src)
            con.execute(f"CREATE VIEW {name} AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM _{name}_df")
        elif default_sql is not None:
            con.execute(f"CREATE VIEW {name} AS {default_sql}")

    ppath = Path(panel) if isinstance(panel, (str, Path)) else PANEL_PATH
    pdef = f"SELECT * REPLACE (CAST(date AS DATE) AS date) FROM read_parquet('{ppath.as_posix()}')"
    view("panel", panel, pdef if (not isinstance(panel, pd.DataFrame) and ppath.exists()) else None)

    def csv(name):
        f = raw_dir / f"{name}.csv"
        return f"SELECT * FROM read_csv_auto('{f.as_posix()}')" if f.exists() else None

    view("holidays", holidays, csv("holidays_events"))
    view("transactions", transactions, csv("transactions"))
    view("oil", oil, csv("oil"))
    view("train", None, csv("train"))
    return con


def default_params() -> dict:
    return {"start": C.MODEL_START, "cutoff": EDA_CUTOFF, "quake_start": C.QUAKE_START, "quake_end": C.QUAKE_END,
            "offset": 0, "min_baseline": 2, "families": list(C.FAMILIES)}


def sql_text(name: str) -> str:
    path = SQL_DIR / (name if name.endswith(".sql") else f"{name}.sql")
    return path.read_text()


def run_sql(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame:
    """Run sql/<name>.sql, binding only the $params the file actually uses."""
    text = sql_text(name)
    used = set(re.findall(r"\$([A-Za-z_]\w*)", text))
    allp = {**default_params(), **params}
    missing = used - set(allp)
    if missing:
        raise ValueError(f"{name}: missing params {sorted(missing)}")
    return con.execute(text, {k: allp[k] for k in used}).df()


# ----------------------------------------------------------------------------------------------- pure helpers
def dow_profile(df: pd.DataFrame) -> dict[str, list[float]]:
    """02 output -> {family: [Mon..Sun index]} with mean exactly 1 (re-normalised after rounding check)."""
    out = {}
    for fam, g in df.sort_values("dow").groupby("family"):
        v = g.set_index("dow")["mean_sales"].reindex(range(1, 8)).to_numpy(float)
        out[fam] = list(v / v.mean())
    return out


def analogue_summary(rows: pd.DataFrame, families=C.FAMILIES, window: str | None = None) -> list[dict]:
    """Per-holiday rows from 10_ecuador_holiday_analogue.sql -> one item per family:
    change_pct = simple mean over holidays of the per-holiday % change; n_holidays = holidays used."""
    window = window or ANALOGUE_WINDOW.format(cutoff=EDA_CUTOFF)
    out = []
    for fam in families:
        g = rows[rows["family"] == fam].dropna(subset=["change_pct"])
        item = {"family": fam, "change_pct": float(g["change_pct"].mean()) if len(g) else None,
                "n_holidays": int(g["holiday_date"].nunique()), "window": window}
        if len(g):
            item.update({"median_pct": float(g["change_pct"].median()), "min_pct": float(g["change_pct"].min()),
                         "max_pct": float(g["change_pct"].max())})
        out.append(item)
    return out


def _r(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _records(df: pd.DataFrame, nd=4) -> list[dict]:
    recs = []
    for row in df.to_dict("records"):
        r = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
                r[k] = str(v)[:10]
            elif isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                r[k] = _r(v, nd)
            else:
                r[k] = v
        recs.append(r)
    return recs


# ----------------------------------------------------------------------------------------------- builders
def build_demand(con) -> dict:
    p = default_params()
    wk = run_sql(con, "01_weekly_family_totals")
    weeks = sorted(wk["week_start"].unique())
    piv = wk.pivot(index="week_start", columns="family", values="sales").reindex(weeks)
    by_family = {f: [None if pd.isna(v) else round(float(v), 3) for v in piv[f]] for f in C.FAMILIES}

    dow = dow_profile(run_sql(con, "02_day_of_week_profile"))
    dow = {f: [round(x, 6) for x in dow[f]] for f in C.FAMILIES}

    vol = run_sql(con, "03_volatility_by_series")
    volatility = [{"store_nbr": int(r.store_nbr), "family": r.family, "mean": round(r.mean, 3), "cv": round(r.cv, 4),
                   "cv_weekly": round(r.cv_weekly, 4), "zero_share": round(r.zero_share, 4),
                   "n_open_days": int(r.n_open_days)} for r in vol.itertuples()]

    pb = run_sql(con, "04_promo_bins")
    promo_bins = {f: [{"bin": r.bin, "n_days": int(r.n_days), "mean_sales": round(r.mean_sales, 3),
                       "median_sales": round(r.median_sales, 3), "mean_index": round(r.mean_index, 4),
                       "n_stores": int(r.n_stores)} for r in g.sort_values("bin_order").itertuples()]
                  for f, g in pb.groupby("family")}

    pf = run_sql(con, "05_promo_frequency")
    promo_frequency = [{"store_nbr": int(r.store_nbr), "family": r.family,
                        "promo_day_share": round(r.promo_day_share, 4), "mean_promo_count": round(r.mean_promo_count, 3),
                        "max_promo_count": int(r.max_promo_count), "corr_promo_sales": _r(r.corr_promo_sales)}
                       for r in pf.itertuples()]

    he = run_sql(con, "06_holiday_effect")
    holiday_effect = [{"family": r.family, "holiday_mean": round(r.holiday_mean, 3), "normal_mean": round(r.normal_mean, 3),
                       "change_pct": round(r.change_pct, 2), "index_change_pct": round(r.index_change_pct, 2),
                       "change_pct_excl_additional": round(r.change_pct_excl_additional, 2),
                       "n_holiday_days": int(r.n_holiday_days), "n_holiday_store_days": int(r.n_holiday_store_days),
                       "n_holiday_days_excl_additional": int(r.n_holiday_days_excl_additional)}
                      for r in he.itertuples()]

    notes = [
        "Real data from Corporacion Favorita (Ecuador), Kaggle 'Store Sales - Time Series Forecasting'. "
        "Scope: 12 stores x 8 families = 96 daily series.",
        f"All EDA uses history {p['start']}..{p['cutoff']} (up to the last calibration target date). "
        "The holdout period 2017-06-28..2017-08-15 is not used, so nothing here can leak into model evaluation.",
        "Sales are observed sales, not unconstrained demand: there is no stock-out data, so lost sales are invisible.",
        "Closed store-days (25 Dec and 1 Jan, chain-wide) are missing (null), not zero, and are excluded from means.",
        "weekly: complete Monday-Sunday weeks only; totals are summed over the 12 stores. Weeks containing 25 Dec or "
        "1 Jan include one closed day.",
        "dow_profile: index Mon..Sun = weekday mean of chain-level daily sales / mean of the 7 weekday means "
        "(mean = 1); national days off excluded.",
        "volatility: cv = coefficient of variation of daily sales on open days (includes the weekday pattern); "
        "cv_weekly uses complete-week totals.",
        "onpromotion is the NUMBER of promoted items in a family/store/day. There is no discount depth or price. "
        "Promotion figures are associations, not causal uplift; mean_index compares each day with the same "
        "store-family's average for that month.",
        "holiday_effect: national days off (Holiday not transferred, Transfer, Bridge, Additional) vs normal days "
        "(no day off of any locale matched to the store's city/state, no Work Day, no Event, outside the earthquake "
        "month). Raw means do not control for weekday mix; n_holiday_days = distinct dates. National 'Additional' "
        "days are mostly the Christmas run-up (21-26 Dec, 31 Dec), which dominates FROZEN FOODS; "
        "change_pct_excl_additional removes them. The weekday-matched holiday analogue is in ecuador_analogue.json.",
        "The 2016-04-16 earthquake month (2016-04-16..2016-05-15) is kept in the data but excluded from "
        "normal-day baselines.",
    ]
    return {"weekly": {"week_start": [str(w)[:10] for w in weeks], "by_family": by_family},
            "dow_profile": dow, "volatility": volatility, "promo_bins": promo_bins,
            "promo_frequency": promo_frequency, "holiday_effect": holiday_effect, "notes": notes}


def build_analogue(con) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    d0 = run_sql(con, "10_ecuador_holiday_analogue", offset=0)
    dm1 = run_sql(con, "10_ecuador_holiday_analogue", offset=-1)
    return analogue_summary(d0), d0, dm1


def build_extras(con, d0: pd.DataFrame, dm1: pd.DataFrame) -> dict:
    p = default_params()
    scope = json.loads((C.CONFIG / "scope.json").read_text())
    stores = [s["store_nbr"] for s in scope["stores"]]

    ov = run_sql(con, "13_data_overview")
    wk = run_sql(con, "01_weekly_family_totals")
    # trend: first 52 complete weeks vs last 52 complete weeks
    trend = []
    for f, g in wk.groupby("family"):
        g = g.sort_values("week_start")
        first, last = g["sales"].iloc[:52].sum(), g["sales"].iloc[-52:].sum()
        trend.append({"family": f, "first_52w_sales": round(first, 1), "last_52w_sales": round(last, 1),
                      "first_52w_start": str(g["week_start"].iloc[0])[:10],
                      "last_52w_start": str(g["week_start"].iloc[-52])[:10],
                      "change_pct": round(100 * (last / first - 1), 2)})
    tot = wk.groupby("week_start")["sales"].sum().sort_index()
    peak_week = {"week_start": str(tot.idxmax())[:10], "sales": round(float(tot.max()), 1),
                 "median_week_sales": round(float(tot.median()), 1)}

    eq = run_sql(con, "07_earthquake_period")
    eq_rows = []
    for (f, grp), g in eq.groupby(["family", "store_group"]):
        m = g.set_index("period")["mean_daily_sales_per_store"]
        ref = (m["pre"] + m["post"]) / 2
        eq_rows.append({"family": f, "store_group": grp, "pre": round(m["pre"], 3), "quake": round(m["quake"], 3),
                        "post": round(m["post"], 3), "quake_vs_pre_post_pct": round(100 * (m["quake"] / ref - 1), 2),
                        "quake_vs_pre_pct": round(100 * (m["quake"] / m["pre"] - 1), 2),
                        "post_vs_pre_pct": round(100 * (m["post"] / m["pre"] - 1), 2)})
    eqa = eq.groupby(["store_group", "period"])[["total_sales", "n_open_store_days"]].sum()
    eqa = (eqa["total_sales"] / eqa["n_open_store_days"] * 8).unstack("period")  # per store, all 8 families
    eq_all = [{"store_group": g, "pre": round(r["pre"], 2), "quake": round(r["quake"], 2), "post": round(r["post"], 2),
               "quake_vs_pre_post_pct": round(100 * (r["quake"] / ((r["pre"] + r["post"]) / 2) - 1), 2),
               "quake_vs_pre_pct": round(100 * (r["quake"] / r["pre"] - 1), 2),
               "post_vs_pre_pct": round(100 * (r["post"] / r["pre"] - 1), 2)}
              for g, r in eqa.iterrows()]

    pay = run_sql(con, "08_payday_effect")
    pay_fam = pay[pay["family"].notna()]
    pay_tbl = pay_fam.pivot(index="family", columns="day_class", values="mean_index")
    pay_n = pay_fam.pivot(index="family", columns="day_class", values="n_store_days")
    payday = [{"family": f, "payday_index": round(r["payday"], 4), "day_after_index": round(r["day_after"], 4),
               "other_index": round(r["other"], 4), "n_payday_store_days": int(pay_n.loc[f, "payday"])}
              for f, r in pay_tbl.iterrows()]
    dom = pay[pay["family"].isna() & pay["day_of_month"].notna()].sort_values("day_of_month")
    dom_profile = [{"day_of_month": int(r.day_of_month), "mean_index": round(r.mean_index, 4),
                    "n_store_days": int(r.n_store_days)} for r in dom.itertuples()]

    tx = run_sql(con, "09_transactions_vs_sales")
    tx_store = [{"store_nbr": int(r.store_nbr), "corr_daily": round(r.corr_daily, 4),
                 "sales_per_txn": round(r.sales_per_txn, 4), "mean_transactions": round(r.mean_transactions, 1),
                 "n_days": int(r.n_days), "n_missing_txn": int(r.n_missing_txn)}
                for r in tx[tx["store_nbr"].notna()].itertuples()]
    pooled = tx[tx["store_nbr"].isna()].iloc[0]

    gap = run_sql(con, "11_produce_recording_gap", stores=stores)
    gap_month = gap.pivot(index="month", columns="family", values="zero_share").sort_index()
    pre = gap[pd.to_datetime(gap["month"]) < pd.Timestamp(C.MODEL_START)]
    post = gap[pd.to_datetime(gap["month"]) >= pd.Timestamp(C.MODEL_START)]

    def wshare(df):
        g = df.assign(z=df["zero_share"] * df["n_open_store_days"]).groupby("family")[["z", "n_open_store_days"]].sum()
        return (g["z"] / g["n_open_store_days"]).round(4).to_dict()

    oil = run_sql(con, "12_oil_vs_sales_monthly")
    oil_valid = oil.dropna(subset=["oil_mean"])

    def hol_rows(df):
        return _records(df[["holiday_date", "description", "holiday_type", "target_date", "target_dow", "family",
                            "change_pct", "n_stores"]], 2)

    return {
        "window": {"start": p["start"], "cutoff": p["cutoff"],
                   "note": "All numbers use history up to the last calibration target date; holdout untouched. "
                           "The PRODUCE gap check also reads pre-window raw history (2013-2015) on purpose."},
        "overview": _records(ov),
        "trend_52w": trend,
        "peak_week_all_families": peak_week,
        "produce_recording_gap": {
            "zero_share_before_window": wshare(pre), "zero_share_in_window": wshare(post),
            "monthly": {"month": [str(m)[:10] for m in gap_month.index],
                        "by_family": {f: [_r(v) for v in gap_month[f]] for f in gap_month.columns}},
            "definition": "Share of open store-days (store sold something across all 33 families) with zero "
                          "recorded sales, 12 scope stores, raw train.csv.",
        },
        "earthquake": {"definition": "Mean daily sales per open store-day in the 30 days before 2016-04-16 (pre), "
                                     "2016-04-16..2016-05-15 (quake) and the 30 days after (post); change = quake vs "
                                     "mean of pre and post. Manabi = stores 53 (Manta) and 54 (El Carmen). Descriptive.",
                       "by_family": eq_rows, "all_families": eq_all},
        "payday": {"definition": "index = sales / same store-family, same month, same weekday mean on non-payday days; "
                                 "payday = 15th or last day of month; day_after = 16th or 1st; days off, Work Days, "
                                 "Events and the earthquake month excluded.",
                   "by_family": payday, "day_of_month_profile": dom_profile},
        "transactions": {"note": "EDA only: transactions are not a model feature (future values unknown at forecast time).",
                         "pooled_corr_daily": round(float(pooled["corr_daily"]), 4),
                         "pooled_n_days": int(pooled["n_days"]),
                         "median_store_corr": round(float(np.median([r["corr_daily"] for r in tx_store])), 4),
                         "by_store": tx_store},
        "weekday_promo_profile": _records(run_sql(con, "14_weekday_promo_profile"), 3),
        "oil": {"note": "Context only, not a model feature. Two trending series can correlate without a causal link.",
                "first_month": str(oil_valid["month"].iloc[0])[:10],
                "first_month_mean": round(float(oil_valid["oil_mean"].iloc[0]), 2),
                "last_month": str(oil_valid["month"].iloc[-1])[:10],
                "last_month_mean": round(float(oil_valid["oil_mean"].iloc[-1]), 2),
                "min_month": str(oil_valid.loc[oil_valid["oil_mean"].idxmin(), "month"])[:10],
                "min_month_mean": round(float(oil_valid["oil_mean"].min()), 2),
                "monthly": _records(oil, 3)},
        "holiday_analogue_detail": {
            "definition": ANALOGUE_WINDOW.format(cutoff=p["cutoff"]),
            "day_of": hol_rows(d0),
            "day_before_summary": [{k: v for k, v in it.items() if k != "window"}
                                   for it in analogue_summary(dm1, window="day before")],
            "day_before": hol_rows(dm1),
        },
    }
