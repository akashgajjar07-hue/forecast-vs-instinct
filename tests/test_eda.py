"""EDA / Demand Intelligence tests. CI-safe tests use synthetic frames; @pytest.mark.data tests need the real panel
and outputs/eda/*.json (run scripts/04_build_eda.py first)."""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from src import eda
from src.schemas import SCHEMAS
from tests.conftest import make_synthetic_panel

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"
OUT = ROOT / "outputs" / "eda"
REQUIRED = ["01_weekly_family_totals", "02_day_of_week_profile", "03_volatility_by_series", "04_promo_bins",
            "05_promo_frequency", "06_holiday_effect", "07_earthquake_period", "08_payday_effect",
            "09_transactions_vs_sales", "10_ecuador_holiday_analogue"]
HOL_COLS = ["date", "type", "locale", "locale_name", "description", "transferred"]


# ------------------------------------------------------------------------------------------- CI-safe
def test_required_sql_files_exist():
    for name in REQUIRED:
        assert (SQL / f"{name}.sql").exists(), name


def test_every_sql_file_starts_with_business_question():
    files = sorted(SQL.glob("*.sql"))
    assert len(files) >= len(REQUIRED)
    for f in files:
        first = f.read_text().lstrip().splitlines()[0]
        assert first.startswith("-- Business question:"), f.name


def _empty_holidays():
    return pd.DataFrame({"date": pd.to_datetime([]), "type": pd.Series([], dtype=str),
                         "locale": pd.Series([], dtype=str), "locale_name": pd.Series([], dtype=str),
                         "description": pd.Series([], dtype=str), "transferred": pd.Series([], dtype=bool)})


def _tiny_analogue_frames():
    """1 store, 2 families, Jan-Feb 2016, sales 100 everywhere except the hand-set values below."""
    dates = pd.date_range("2016-01-01", "2016-02-29", freq="D")
    rows = []
    for fam in ["BEVERAGES", "DAIRY"]:
        rows.append(pd.DataFrame({"store_nbr": 1, "family": fam, "date": dates, "sales": 100.0}))
    p = pd.concat(rows, ignore_index=True)
    for c in ["closed", "is_future", "is_holiday", "is_workday", "is_event", "quake", "is_national_holiday"]:
        p[c] = 0

    def setv(fam, d, col, v):
        p.loc[(p.family == fam) & (p.date == pd.Timestamp(d)), col] = v

    # Holiday 1: Wed 2016-01-20. Baselines 01-06, 01-13, 01-27, 02-03.
    p.loc[p.date.isin(pd.to_datetime(["2016-01-20", "2016-01-29"])), ["is_holiday", "is_national_holiday"]] = 1
    setv("BEVERAGES", "2016-01-20", "sales", 150.0)
    setv("BEVERAGES", "2016-01-13", "sales", 80.0)
    setv("BEVERAGES", "2016-01-27", "sales", 120.0)   # baseline mean 100 -> +50 %
    setv("DAIRY", "2016-01-20", "sales", 90.0)
    p.loc[p.date == pd.Timestamp("2016-02-03"), "is_event"] = 1  # excluded baseline day ...
    setv("DAIRY", "2016-02-03", "sales", 500.0)                   # ... so this outlier must not count -> -10 %
    # Holiday 2: Fri 2016-01-29 (Transfer). Baselines 01-15, 01-22, 02-05, 02-12.
    setv("BEVERAGES", "2016-01-29", "sales", 110.0)               # +10 % ; DAIRY stays 100 -> 0 %
    # Transferred Holiday on 2016-02-10 must be ignored even with a huge value
    setv("BEVERAGES", "2016-02-10", "sales", 1000.0)
    hol = pd.DataFrame([
        ["2016-01-20", "Holiday", "National", "Ecuador", "Test holiday", False],
        ["2016-01-29", "Transfer", "National", "Ecuador", "Test transfer", False],
        ["2016-02-10", "Holiday", "National", "Ecuador", "Transferred away", True],
        ["2016-02-06", "Work Day", "National", "Ecuador", "Work day", False],
        ["2016-01-23", "Holiday", "Local", "Quito", "Local only", False],
    ], columns=HOL_COLS)
    hol["date"] = pd.to_datetime(hol["date"])
    return p, hol


def test_analogue_matches_hand_computation():
    p, hol = _tiny_analogue_frames()
    con = eda.connect(panel=p, holidays=hol)
    rows = eda.run_sql(con, "10_ecuador_holiday_analogue", start="2016-01-01", cutoff="2016-02-29", offset=0)
    got = {(str(r.holiday_date)[:10], r.family): r.change_pct for r in rows.itertuples()}
    assert set(got) == {("2016-01-20", "BEVERAGES"), ("2016-01-20", "DAIRY"),
                        ("2016-01-29", "BEVERAGES"), ("2016-01-29", "DAIRY")}
    assert got[("2016-01-20", "BEVERAGES")] == pytest.approx(50.0)
    assert got[("2016-01-20", "DAIRY")] == pytest.approx(-10.0)
    assert got[("2016-01-29", "BEVERAGES")] == pytest.approx(10.0)
    assert got[("2016-01-29", "DAIRY")] == pytest.approx(0.0)
    summ = {s["family"]: s for s in eda.analogue_summary(rows, families=["BEVERAGES", "DAIRY"], window="w")}
    assert summ["BEVERAGES"]["change_pct"] == pytest.approx(30.0)
    assert summ["DAIRY"]["change_pct"] == pytest.approx(-5.0)
    assert summ["BEVERAGES"]["n_holidays"] == 2
    item_schema = SCHEMAS["scenario.json"]["properties"]["ecuador_analogue"]
    jsonschema.validate(list(summ.values()), item_schema)


def test_analogue_cutoff_limits_baseline_days():
    """Cutoff 2016-02-01: baseline days after the cutoff are not used.
    01-20 keeps 01-06, 01-13, 01-27 (3 days) -> kept with min_baseline=3;
    01-29 keeps only 01-15, 01-22 (2 days) -> dropped."""
    p, hol = _tiny_analogue_frames()
    con = eda.connect(panel=p, holidays=hol)
    rows = eda.run_sql(con, "10_ecuador_holiday_analogue", start="2016-01-01", cutoff="2016-02-01", offset=0,
                       min_baseline=3)
    assert set(rows["holiday_date"].astype(str).str[:10]) == {"2016-01-20"}
    b = rows.set_index("family")
    assert b.loc["BEVERAGES", "change_pct"] == pytest.approx(100 * (150 / ((100 + 80 + 120) / 3) - 1))
    assert b.loc["DAIRY", "change_pct"] == pytest.approx(-10.0)


def test_dow_profile_mean_is_one_on_synthetic():
    p = make_synthetic_panel()
    con = eda.connect(panel=p, holidays=_empty_holidays())
    dow = eda.dow_profile(eda.run_sql(con, "02_day_of_week_profile", start="2016-01-01", cutoff="2016-12-31"))
    for fam, v in dow.items():
        assert len(v) == 7
        assert np.mean(v) == pytest.approx(1.0, abs=1e-12)
    # synthetic pattern has Saturday 1.4 vs Tuesday 0.9
    assert dow["BEVERAGES"][5] > dow["BEVERAGES"][1]


def test_core_queries_run_on_synthetic_panel():
    p = make_synthetic_panel()
    con = eda.connect(panel=p, holidays=_empty_holidays())
    kw = dict(start="2016-01-01", cutoff="2016-08-27")
    wk = eda.run_sql(con, "01_weekly_family_totals", **kw)
    assert (wk["n_dates"] == 7).all()
    # weekly totals reconcile with a pandas groupby on complete weeks
    q = p[(p.date >= "2016-01-04") & (p.date <= "2016-08-21")].copy()
    q["week_start"] = q["date"] - pd.to_timedelta(q["date"].dt.dayofweek, unit="D")
    exp = q.assign(week_start=q.week_start.astype("datetime64[ns]")).groupby(["family", "week_start"])["sales"].sum()
    got = wk.assign(week_start=pd.to_datetime(wk.week_start).astype("datetime64[ns]")).set_index(["family", "week_start"])["sales"]
    pd.testing.assert_series_equal(got.sort_index(), exp.sort_index(), check_names=False, atol=1e-6)
    vol = eda.run_sql(con, "03_volatility_by_series", **kw)
    assert len(vol) == 4 and (vol["zero_share"] == 0).all()
    pb = eda.run_sql(con, "04_promo_bins", **kw)
    assert pb.groupby("family")["n_days"].sum().eq(p[(p.date <= "2016-08-27") & (p.closed == 0)]
                                                   .groupby("family").size()).all()
    pf = eda.run_sql(con, "05_promo_frequency", **kw)
    assert pf["promo_day_share"].between(0, 1).all()
    he = eda.run_sql(con, "06_holiday_effect", **kw)
    assert set(he["family"]) == {"BEVERAGES", "PRODUCE"}


def test_run_sql_rejects_missing_params():
    p = make_synthetic_panel()
    con = eda.connect(panel=p, holidays=_empty_holidays())
    with pytest.raises(ValueError):
        eda.run_sql(con, "11_produce_recording_gap")  # needs $stores


# ------------------------------------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def demand():
    f = OUT / "demand.json"
    if not f.exists():
        pytest.skip("outputs/eda/demand.json not built")
    return json.loads(f.read_text())


@pytest.mark.data
def test_demand_json_validates_and_is_small(demand):
    jsonschema.validate(demand, SCHEMAS["demand.json"])
    assert (OUT / "demand.json").stat().st_size < 400_000
    assert set(demand["weekly"]["family_index"]) == set(eda.C.FAMILIES)
    assert len(demand["volatility"]) == 96 and len(demand["promo_frequency"]) == 96


@pytest.mark.data
def test_demand_dow_profile_mean_one(demand):
    for fam, v in demand["dow_profile"].items():
        assert len(v) == 7 and np.mean(v) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.data
def test_weekly_family_index_reconciles_with_panel(demand, real_panel):
    """Equal-weighted family weekly index recomputed independently: per series mean daily sales over the week's open
    days / series base (open-day mean 2015-06-01..2017-02-06) x 100, then the plain mean over the family's series."""
    p = real_panel[(real_panel.is_future == 0) & (real_panel.closed == 0) & real_panel.sales.notna()].copy()
    base = p[(p.date >= "2015-06-01") & (p.date <= "2017-02-06")].groupby("series_id")["sales"].mean()
    p["idx"] = p["sales"] / p["series_id"].map(base) * 100
    p["week_start"] = (p["date"] - pd.to_timedelta(p["date"].dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    weeks = demand["weekly"]["week_start"]
    assert pd.Timestamp(weeks[-1]) + pd.Timedelta(days=6) <= pd.Timestamp(eda.EDA_CUTOFF)  # no holdout data
    sw = p[p.week_start.isin(weeks)].groupby(["family", "series_id", "week_start"])["idx"].mean()
    exp = sw.groupby(["family", "week_start"]).mean()
    for fam, vals in demand["weekly"]["family_index"].items():
        e = exp.loc[fam].reindex(weeks).to_numpy()
        assert np.allclose(np.array(vals, float), e, rtol=0, atol=0.005 + 1e-9)  # JSON rounded to 2 dp
    assert "units" not in json.dumps(demand["weekly"]).replace("no units", "")


@pytest.mark.data
def test_analogue_output_validates():
    f = OUT / "ecuador_analogue.json"
    if not f.exists():
        pytest.skip("outputs/eda/ecuador_analogue.json not built")
    items = json.loads(f.read_text())
    jsonschema.validate(items, SCHEMAS["scenario.json"]["properties"]["ecuador_analogue"])
    assert {i["family"] for i in items} == set(eda.C.FAMILIES)
    assert all(i["n_holidays"] > 0 for i in items)
