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
from . import indexing as IX

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


# ----------------------------------------------------------------------------------------------- index helpers
# Owner rule (Kaggle 7B): per-series values only as % or on the index scale; aggregates over series are EQUAL-WEIGHTED
# means of the member series' index values, which carry no information about the private bases (src/indexing.py).
def index_frame(con, cutoff: str | None = None) -> pd.DataFrame:
    """Open, observed panel rows in [MODEL_START, cutoff] with idx = sales / series base x 100 and
    pidx = onpromotion / series promo base (up to cutoff) x 100."""
    cutoff = cutoff or default_params()["cutoff"]
    full = con.execute("SELECT * FROM panel").df()
    full["date"] = pd.to_datetime(full["date"])
    if "series_id" not in full:
        full["series_id"] = full["store_nbr"].astype(str) + "|" + full["family"]
    base = IX.series_base(full)
    pbase = IX.promo_base(full, cutoff)
    d = full[(full["date"] >= pd.Timestamp(C.MODEL_START)) & (full["date"] <= pd.Timestamp(cutoff))
             & (full["closed"] == 0) & full["sales"].notna()].copy()
    d["idx"] = d["sales"] / d["series_id"].map(base) * 100
    d["pidx"] = d["onpromotion"] / d["series_id"].map(pbase) * 100
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    return d


def family_weekly_index(d: pd.DataFrame, weeks) -> dict[str, list]:
    """Equal-weighted family weekly index: per series mean daily index over the week's open days, then the plain mean
    over the family's series."""
    sw = d.groupby(["family", "series_id", "week_start"])["idx"].mean()
    fw = sw.groupby(["family", "week_start"]).mean().unstack("family")
    fw = fw.reindex(pd.to_datetime([str(w)[:10] for w in weeks]))
    return {f: [None if pd.isna(v) else round(float(v), 2) for v in fw[f]] for f in C.FAMILIES if f in fw}


def equal_weighted_dow(d: pd.DataFrame) -> dict[str, list[float]]:
    """Per family: plain mean over its series of the series' weekday mean index (national days off excluded),
    normalised so the 7 values average 1."""
    q = d[d["is_national_holiday"] == 0]
    sd = q.groupby(["family", "series_id", q["date"].dt.dayofweek])["idx"].mean()
    fd = sd.groupby(level=[0, 2]).mean().unstack(1)
    return {f: list((fd.loc[f].reindex(range(7)).to_numpy(float) / fd.loc[f].mean())) for f in fd.index}


# ----------------------------------------------------------------------------------------------- builders
def build_demand(con) -> dict:
    p = default_params()
    wk = run_sql(con, "01_weekly_family_totals")
    weeks = sorted(wk["week_start"].unique())
    d = index_frame(con)
    family_index = family_weekly_index(d, weeks)

    dw = equal_weighted_dow(d)
    dow = {f: [round(x, 6) for x in dw[f]] for f in C.FAMILIES if f in dw}

    vol = run_sql(con, "03_volatility_by_series")
    # no per-series mean level: published series values are indexed to a private base (src/indexing.py)
    volatility = [{"store_nbr": int(r.store_nbr), "family": r.family, "cv": round(r.cv, 4),
                   "cv_weekly": round(r.cv_weekly, 4), "zero_share": round(r.zero_share, 4),
                   "n_open_days": int(r.n_open_days)} for r in vol.itertuples()]

    pb = run_sql(con, "04_promo_bins")
    # no unit levels: mean_index (each store-day vs its store-family's monthly mean) and counts only
    promo_bins = {f: [{"bin": r.bin, "n_days": int(r.n_days), "mean_index": round(r.mean_index, 4),
                       "n_stores": int(r.n_stores)} for r in g.sort_values("bin_order").itertuples()]
                  for f, g in pb.groupby("family")}

    pf = run_sql(con, "05_promo_frequency")
    promo_frequency = [{"store_nbr": int(r.store_nbr), "family": r.family,
                        "promo_day_share": round(r.promo_day_share, 4), "corr_promo_sales": _r(r.corr_promo_sales)}
                       for r in pf.itertuples()]

    he = run_sql(con, "06_holiday_effect")
    holiday_effect = [{"family": r.family, "change_pct": round(r.change_pct, 2), "index_change_pct": round(r.index_change_pct, 2),
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
        "weekly.family_index: equal-weighted family index per complete Monday-Sunday week = plain mean over the "
        "family's 12 store series of each series' weekly index (mean daily sales over the week's open days / the "
        "series' mean daily sales 2015-06-01..2017-02-06 x 100). 100 = an average training-period day. No units.",
        "dow_profile: per family, plain mean over its 12 series of each series' weekday mean index, normalised so "
        "Mon..Sun average 1; national days off excluded.",
        "volatility: cv = coefficient of variation of daily sales on open days (includes the weekday pattern); "
        "cv_weekly uses complete-week totals.",
        "onpromotion is the NUMBER of promoted items in a family/store/day. There is no discount depth or price. "
        "Promotion figures are associations, not causal uplift; mean_index compares each day with the same "
        "store-family's average for that month. Promoted-item counts and sales levels are not published (Kaggle rules).",
        "holiday_effect: national days off (Holiday not transferred, Transfer, Bridge, Additional) vs normal days "
        "(no day off of any locale matched to the store's city/state, no Work Day, no Event, outside the earthquake "
        "month). change_pct compares mean sales on holiday vs normal days (levels not published); it does not control for weekday mix; n_holiday_days = distinct dates. National 'Additional' "
        "days are mostly the Christmas run-up (21-26 Dec, 31 Dec), which dominates FROZEN FOODS; "
        "change_pct_excl_additional removes them. The weekday-matched holiday analogue is in ecuador_analogue.json.",
        "The 2016-04-16 earthquake month (2016-04-16..2016-05-15) is kept in the data but excluded from "
        "normal-day baselines.",
    ]
    return {"weekly": {"week_start": [str(w)[:10] for w in weeks], "family_index": family_index,
                       "definition": "equal-weighted family index (mean of the 12 store series' weekly index; "
                                     "series training-period mean = 100)"},
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

    # Owner rule: no unit levels and no volume-weighted aggregates; everything below is on the index scale and
    # equal-weighted over series (src/indexing.py), or a count / share / correlation.
    ov = run_sql(con, "13_data_overview").drop(columns=["total_sales", "volume_share"], errors="ignore")
    wk = run_sql(con, "01_weekly_family_totals")
    weeks = sorted(wk["week_start"].unique())
    d = index_frame(con)
    fwi = pd.DataFrame(family_weekly_index(d, weeks), index=[str(w)[:10] for w in weeks])
    # trend: first 52 complete weeks vs last 52 complete weeks of the equal-weighted family index
    trend = []
    for f in fwi.columns:
        g = fwi[f]
        first, last = g.iloc[:52].mean(), g.iloc[-52:].mean()
        trend.append({"family": f, "first_52w_mean_index": round(first, 2), "last_52w_mean_index": round(last, 2),
                      "first_52w_start": g.index[0], "last_52w_start": g.index[-52],
                      "change_pct": round(100 * (last / first - 1), 2)})
    allw = fwi.mean(axis=1)  # equal-weighted over families = over all 96 series (12 per family)
    peak_week = {"week_start": str(allw.idxmax())[:10], "mean_index": round(float(allw.max()), 2),
                 "median_week_mean_index": round(float(allw.median()), 2)}

    # earthquake: equal-weighted mean index per period (all history in the panel, not only up to the cutoff)
    qs, qe = pd.Timestamp(C.QUAKE_START), pd.Timestamp(C.QUAKE_END)
    dq = index_frame(con, cutoff=str((qe + pd.Timedelta(days=30)).date()))
    dq["period"] = np.select([(dq["date"] >= qs - pd.Timedelta(days=30)) & (dq["date"] < qs),
                              (dq["date"] >= qs) & (dq["date"] <= qe),
                              (dq["date"] > qe) & (dq["date"] <= qe + pd.Timedelta(days=30))],
                             ["pre", "quake", "post"], default="")
    dq = dq[dq["period"] != ""]
    dq["store_group"] = np.where(dq["state"] == "Manabi", "Manabi", "Other")
    sm = dq.groupby(["family", "store_group", "series_id", "period"])["idx"].mean()

    def eq_row(m):
        ref = (m["pre"] + m["post"]) / 2
        return {"pre": round(m["pre"], 2), "quake": round(m["quake"], 2), "post": round(m["post"], 2),
                "quake_vs_pre_post_pct": round(100 * (m["quake"] / ref - 1), 2),
                "quake_vs_pre_pct": round(100 * (m["quake"] / m["pre"] - 1), 2),
                "post_vs_pre_pct": round(100 * (m["post"] / m["pre"] - 1), 2)}
    fam_m = sm.groupby(["family", "store_group", "period"]).mean()
    eq_rows = [{"family": f, "store_group": g, **eq_row(fam_m.loc[(f, g)])}
               for f, g in sorted({(a, b) for a, b, _ in fam_m.index})]
    all_m = sm.groupby(["store_group", "period"]).mean()
    eq_all = [{"store_group": g, **eq_row(all_m.loc[g])} for g in sorted(all_m.index.get_level_values(0).unique())]

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
    mi = d.groupby([d["date"].dt.to_period("M").dt.to_timestamp(), "series_id"])["idx"].mean().groupby(level=0).mean()
    oil["month"] = pd.to_datetime(oil["month"])
    oil["mean_index"] = oil["month"].map(mi)
    oil = oil.drop(columns=["mean_daily_sales"])
    oil_valid = oil.dropna(subset=["oil_mean"])

    wpp = d.assign(dow=d["date"].dt.dayofweek + 1, last12=d["date"] > pd.Timestamp(p["cutoff"]) - pd.Timedelta(days=365))
    wpp_rows = []
    for (f, dw), g in wpp.groupby(["family", "dow"]):
        s_all = g.groupby("series_id")[["idx", "pidx"]].mean().mean()
        s_12 = g[g["last12"]].groupby("series_id")[["idx", "pidx"]].mean().mean()
        wpp_rows.append({"family": f, "dow": int(dw), "mean_promo_index": _r(s_all["pidx"], 2),
                         "mean_index": _r(s_all["idx"], 2), "mean_promo_index_last_12m": _r(s_12["pidx"], 2),
                         "mean_index_last_12m": _r(s_12["idx"], 2), "n_store_days": int(len(g))})

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
        "earthquake": {"definition": "Equal-weighted mean index (series training-period mean = 100) over the 30 days "
                                     "before 2016-04-16 (pre), 2016-04-16..2016-05-15 (quake) and the 30 days after "
                                     "(post): per series mean daily index, then the plain mean over the group's series; "
                                     "change = quake vs mean of pre and post. Manabi = stores 53 (Manta) and 54 (El "
                                     "Carmen). Descriptive.",
                       "by_family": eq_rows, "all_families": eq_all},
        "payday": {"definition": "index = sales / same store-family, same month, same weekday mean on non-payday days; "
                                 "payday = 15th or last day of month; day_after = 16th or 1st; days off, Work Days, "
                                 "Events and the earthquake month excluded.",
                   "by_family": payday, "day_of_month_profile": dom_profile},
        "transactions": {"note": "EDA only: transactions are not a model feature (future values unknown at forecast time). "
                                 "Per-store transaction levels are not published (correlations only).",
                         "pooled_corr_daily": round(float(pooled["corr_daily"]), 4),
                         "pooled_n_days": int(pooled["n_days"]),
                         "median_store_corr": round(float(np.median([r["corr_daily"] for r in tx_store])), 4),
                         "by_store": tx_store},
        "weekday_promo_profile": wpp_rows,
        "weekday_promo_profile_definition": "Per family x ISO weekday: plain mean over the family's series of each "
                                            "series' mean index (sales) and mean promo index (promoted-item count / "
                                            "the series' mean count up to the cutoff x 100).",
        "oil": {"note": "Context only, not a model feature. Two trending series can correlate without a causal link.",
                "first_month": str(oil_valid["month"].iloc[0])[:10],
                "first_month_mean": round(float(oil_valid["oil_mean"].iloc[0]), 2),
                "last_month": str(oil_valid["month"].iloc[-1])[:10],
                "last_month_mean": round(float(oil_valid["oil_mean"].iloc[-1]), 2),
                "min_month": str(oil_valid.loc[oil_valid["oil_mean"].idxmin(), "month"])[:10],
                "min_month_mean": round(float(oil_valid["oil_mean"].min()), 2),
                "monthly": _records(oil, 3),
                "monthly_note": "mean_index = equal-weighted mean daily sales index of the 96 series (no units)."},
        "holiday_analogue_detail": {
            "definition": ANALOGUE_WINDOW.format(cutoff=p["cutoff"]),
            # per-holiday-day rows (12-store sums per family and date) are not published: only the summaries
            "n_holiday_family_rows": int(len(d0)),
            "day_before_summary": [{k: v for k, v in it.items() if k != "window"}
                                   for it in analogue_summary(dm1, window="day before")],
        },
    }
