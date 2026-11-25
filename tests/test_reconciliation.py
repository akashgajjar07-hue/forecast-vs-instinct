"""Reconcile the published dashboard JSON (docs/data) with the pipeline outputs.

Every number is recomputed here from outputs/ (parquet + frozen JSON) and the processed panel with independent code:
nothing from src.export is imported. Published values that are rounded (documented in docs/agent_notes/export.md) are
compared after applying the same rounding; everything else is compared at 1e-6.
Run after scripts/08_export_dashboard.py. Needs the Kaggle-derived files, so it is marked `data`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DD = ROOT / "docs" / "data"
TOL = 1e-6
FWD = pd.Timestamp("2017-08-15")

if not (DD / "overview.json").exists() or not (OUT / "holdout_results.json").exists():
    pytest.skip("docs/data or outputs not built", allow_module_level=True)


def J(name):
    return json.loads((DD / name).read_text())


def O(name):
    return json.loads((OUT / name).read_text())


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def r(x, nd):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


@pytest.fixture(scope="module")
def bt():
    d = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "backtests").glob("*.parquet"))], ignore_index=True)
    d["origin"] = pd.to_datetime(d["origin"])
    return d


@pytest.fixture(scope="module")
def ci():
    d = pd.read_parquet(OUT / "holdout" / "champion_intervals.parquet")
    d["origin"] = pd.to_datetime(d["origin"])
    return d


@pytest.fixture(scope="module")
def ts():
    d = pd.read_parquet(OUT / "holdout" / "trust_scores.parquet")
    d["origin"] = pd.to_datetime(d["origin"])
    return d


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet")


@pytest.fixture(scope="module")
def champ():
    return {f: c["model"] for f, c in O("selection.json")["champions"].items()}


def ok(d):
    return d[(d["closed"] == 0) & d["actual"].notna()]


def wape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return np.abs(a - f).sum() / np.abs(a).sum() * 100


def bias(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return (f - a).sum() / a.sum() * 100


def champ_rows(bt, champ, stage):
    b = bt[(bt["stage"] == stage) & (bt["variant"] == "A")]
    return b[[champ[f] == m for f, m in zip(b["family"], b["model"])]]


# ---------------------------------------------------------------- overview KPIs
def test_overview_kpis(bt, ci, ts, champ):
    k = J("overview.json")["kpis"]
    fwd = champ_rows(bt, champ, "forward")
    assert close(k["forecast_14d_total"]["value"], fwd["forecast"].sum(), 1e-4)
    assert close(k["forecast_14d_total"]["value"], ci[ci["stage"] == "forward"]["forecast"].sum())
    assert (k["forecast_14d_total"]["start"], k["forecast_14d_total"]["end"]) == ("2017-08-16", "2017-08-29")
    h = ok(champ_rows(bt, champ, "holdout"))
    assert close(k["holdout_wape"]["value"], wape(h["actual"], h["forecast"]))
    assert close(k["holdout_bias"]["value"], bias(h["actual"], h["forecast"]))
    latest = ts[ts["origin"] == FWD]
    assert k["review_exceptions"]["value"] == int((latest["band"] == "Review").sum())
    assert k["review_exceptions"]["of"] == 96
    hc = ok(ci[ci["stage"] == "holdout"])
    cov = ((hc["actual"] >= hc["p10"]) & (hc["actual"] <= hc["p90"])).mean() * 100
    assert close(k["interval_coverage"]["value"], cov) and k["interval_coverage"]["target"] == 80


def test_overview_outlook_and_lists(panel, ts, champ):
    ov = J("overview.json")
    ct = pd.read_parquet(OUT / "holdout" / "chain_total.parquet")
    f = ct[ct["stage"] == "forward"].sort_values("horizon")
    assert len(ov["outlook"]["forecast"]) == 14
    for row, (_, x) in zip(ov["outlook"]["forecast"], f.iterrows()):
        assert row["date"] == x["target_date"].strftime("%Y-%m-%d")
        for c in ["forecast", "p10", "p90"]:
            assert close(row[c], r(x[c], 1))
    hist = ov["outlook"]["history"]
    assert len(hist) == 56 and hist[-1]["date"] == "2017-08-15"
    for row in hist:
        day = panel[panel["date"] == pd.Timestamp(row["date"])]
        assert close(row["actual"], r(day["sales"].sum(), 1))
    latest = ts[ts["origin"] == FWD].sort_values("priority", ascending=False).head(8)
    assert [x["series_id"] for x in ov["top_exceptions"]] == list(latest["series_id"])
    for x, (_, t) in zip(ov["top_exceptions"], latest.iterrows()):
        assert close(x["trust_score"], r(t["score"], 2)) and x["band"] == t["band"]
    byf = pd.DataFrame(O("holdout_results.json")["by_family"]).set_index("family")
    models = ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"]
    assert ov["ml_wins"]["families_won"] == int((byf[models].idxmin(axis=1) == "lightgbm").sum())
    assert ov["ml_wins"]["champions"] == champ


def test_findings_numbers():
    hr = O("holdout_results.json")
    text = " ".join(J("overview.json")["findings"])
    lb = hr["leaderboard"]
    bb = min((x for x in lb if x["model"] != "lightgbm"), key=lambda x: x["wape"])
    a = next(x for x in lb if x["model"] == "lightgbm" and x["variant"] == "A")
    b = next(x for x in lb if x["model"] == "lightgbm" and x["variant"] == "B")
    for s in [f"{hr['champion_system']['wape']:.1f} %", f"{bb['wape']:.1f} %", f"{a['wape']:.2f} %", f"{b['wape']:.2f} %",
              f"{b['wape'] - a['wape']:+.2f} pp", f"{hr['trust_validation']['spearman']:.2f}",
              f"{hr['trust_validation']['error_ratio_review_vs_accept']:.2f}×"]:
        assert s in text, s


# ---------------------------------------------------------------- hero
def test_hero(ci):
    hr = O("holdout_results.json")
    meta = J("meta.json")
    bb = min((x for x in hr["leaderboard"] if x["model"] != "lightgbm"), key=lambda x: x["wape"])
    hc = ok(ci[ci["stage"] == "holdout"])
    cov = ((hc["actual"] >= hc["p10"]) & (hc["actual"] <= hc["p90"])).mean() * 100
    vals = [h["value"] for h in meta["hero"]]
    assert vals[0] == f"{hr['champion_system']['wape']:.1f} %"
    assert f"{bb['wape']:.1f} %" in meta["hero"][0]["detail"]
    assert vals[1] == f"{hr['trust_validation']['error_ratio_review_vs_accept']:.2f}×"
    assert vals[2] == f"{cov:.1f} %"
    assert meta["citation"].startswith("Alexis Cook, DanB, inversion, Ryan Holbrook (2021)")
    cal = json.loads((ROOT / "config" / "split_calendar.json").read_text())
    assert meta["split_calendar"] == {k: cal["stages"][k] for k in ["selection", "calibration", "holdout"]}


# ---------------------------------------------------------------- arena
def test_arena_holdout_overall(bt):
    rows = {(x["model"], x["variant"]): x for x in J("arena.json")["metrics"]
            if x["stage"] == "holdout" and x["slice_type"] == "overall"}
    h = ok(bt[bt["stage"] == "holdout"])
    assert len(rows) == 6
    for (m, v), g in h.groupby(["model", "variant"]):
        x = rows[(m, v)]
        a, f = g["actual"].to_numpy(float), g["forecast"].to_numpy(float)
        assert close(x["wape"], r(wape(a, f), 4)) and close(x["bias"], r(bias(a, f), 4))
        assert close(x["mae"], r(np.abs(a - f).mean(), 4)) and close(x["rmse"], r(np.sqrt(((a - f) ** 2).mean()), 4))
        assert x["n"] == len(g)
    lb = {(x["model"], x["variant"]): x for x in O("holdout_results.json")["leaderboard"]}
    for key, x in rows.items():
        assert close(x["wape"], r(lb[key]["wape"], 4))


def test_arena_week_bucket(bt):
    rows = {(x["model"], x["variant"], x["slice_value"]): x for x in J("arena.json")["metrics"]
            if x["stage"] == "holdout" and x["slice_type"] == "week_bucket"}
    g = ok(bt[(bt["stage"] == "holdout") & (bt["model"] == "lightgbm") & (bt["variant"] == "A")]).copy()
    g["wk"] = np.where(g["horizon"] <= 7, "week1", "week2")
    s = g.groupby(["series_id", "origin", "wk"])[["actual", "forecast"]].sum().reset_index()
    for wk, d in s.groupby("wk"):
        assert close(rows[("lightgbm", "A", wk)]["wape"], r(wape(d["actual"], d["forecast"]), 4))


# ---------------------------------------------------------------- intervals
def test_intervals(ci):
    iv = J("intervals.json")
    hc = ok(ci[ci["stage"] == "holdout"])
    inside = (hc["actual"] >= hc["p10"]) & (hc["actual"] <= hc["p90"])
    assert close(iv["holdout"]["coverage"], inside.mean() * 100)
    assert close(iv["holdout"]["mean_width"], (hc["p90"] - hc["p10"]).mean())
    for x in iv["holdout"]["by_family"]:
        m = hc["family"] == x["family"]
        assert close(x["coverage"], r(inside[m].mean() * 100, 4))
    assert close(iv["calibration_coverage"], O("intervals.json")["calibration_coverage_leave_one_origin_out"])
    assert close(iv["holdout"]["coverage"], O("holdout_results.json")["champion_system"]["coverage"])


# ---------------------------------------------------------------- trust
def test_trust(ts, ci):
    t = J("trust.json")
    hr = O("holdout_results.json")
    for k, v in hr["trust_validation"].items():
        if k in t["validation"]:
            assert t["validation"][k] == v, k
    hold = ts[ts["origin"] < FWD]
    assert hold["band"].value_counts().to_dict() == hr["trust_band_counts_holdout"]
    latest = ts[ts["origin"] == FWD]
    pub = pd.Series([x["band"] for x in t["latest"]]).value_counts().to_dict()
    assert pub == latest["band"].value_counts().to_dict()
    assert t["weights"] == O("trust_weights.json")["weights"]
    sums = ci[ci["stage"] == "forward"].groupby("series_id")[["forecast", "p10", "p90"]].sum()
    for x in t["latest"]:
        s = sums.loc[x["series_id"]]
        assert close(x["forecast_14d"], r(s["forecast"], 2)) and close(x["p10_14d"], r(s["p10"], 2))
        assert close(x["p90_14d"], r(s["p90"], 2))
    cands = {c["name"]: c["spearman"] for c in O("trust_calibration.json")["candidates"]}
    for c in t["calibration_choice"]["candidates"]:
        assert c["spearman"] == cands[c["name"]]


# ---------------------------------------------------------------- S&OP
def test_sop_totals_and_fva(bt, ci, ts, champ):
    sop = J("sop.json")
    fwd = ci[ci["stage"] == "forward"]
    assert len(sop["rows"]) == 96
    assert close(sum(x["stat_forecast"] for x in sop["rows"]), fwd["forecast"].sum(), 96 * 0.005 + TOL)
    sums = fwd.groupby("series_id")[["forecast", "p10", "p90"]].sum()
    for x in sop["rows"]:
        assert close(x["stat_forecast"], r(sums.loc[x["series_id"], "forecast"], 2))
        assert close(x["p10"], r(sums.loc[x["series_id"], "p10"], 2))
        assert close(x["p90"], r(sums.loc[x["series_id"], "p90"], 2))
    # FVA: independent re-implementation of the documented rule
    past = ok(pd.concat([champ_rows(bt, champ, "calibration"), champ_rows(bt, champ, "holdout")]))
    hold = ok(champ_rows(bt, champ, "holdout")).copy()
    sim = hold["forecast"].to_numpy(float).copy()
    for T in sorted(hold["origin"].unique()):
        done = past[past["origin"] + pd.Timedelta(days=14) <= T]
        for fam in done["family"].unique():
            d = done[done["family"] == fam]
            b = bias(d["actual"], d["forecast"])
            if abs(b) > 2:
                m = ((hold["origin"] == T) & (hold["family"] == fam)).to_numpy()
                sim[m] = sim[m] / (1 + b / 100)
    cw, sw = wape(hold["actual"], hold["forecast"]), wape(hold["actual"], sim)
    f = sop["fva"]
    assert close(f["champion_wape"], cw) and close(f["simulated_wape"], sw) and close(f["fva_pp"], cw - sw)
    assert f["stage"] == "holdout" and "not human-planner" in f["label"]


# ---------------------------------------------------------------- scenario
def test_scenario_baselines(ci):
    sc = J("scenario.json")
    tot = ci[ci["stage"] == "forward"].groupby("family")["forecast"].sum()
    for x in sc["families"]:
        assert close(x["baseline_14d"], tot[x["family"]])
    assert sc["assumption_label"] == "Editable demonstration assumptions, not measured UAE uplift"
    assert sc["ecuador_analogue"] == O("eda/ecuador_analogue.json")


# ---------------------------------------------------------------- series sample
@pytest.mark.parametrize("slug,sid", [("12_beverages", "12|BEVERAGES"), ("49_grocery-i", "49|GROCERY I"),
                                      ("27_bread-bakery", "27|BREAD/BAKERY"), ("53_frozen-foods", "53|FROZEN FOODS")])
def test_series_sample(slug, sid, bt, ci, panel):
    s = json.loads((DD / "series" / f"{slug}.json").read_text())
    assert s["series_id"] == sid
    p = panel[(panel["series_id"] == sid) & (panel["date"] >= "2017-01-01") & (panel["date"] <= "2017-08-15")].sort_values("date")
    exp = [None if c == 1 or pd.isna(v) else r(v, 1) for v, c in zip(p["sales"], p["closed"])]
    assert len(s["history"]["sales"]) == len(exp)
    assert all(close(a, b) for a, b in zip(s["history"]["sales"], exp))
    b = bt[bt["series_id"] == sid]
    for win in s["holdout"]:
        o = pd.Timestamp(win["origin"])
        c = ci[(ci["series_id"] == sid) & (ci["origin"] == o)].sort_values("horizon")
        for row, (_, x) in zip(win["rows"], c.iterrows()):
            assert close(row["actual"], r(x["actual"], 1)) and close(row["p10"], r(x["p10"], 1))
            assert close(row["p90"], r(x["p90"], 1))
            for m, v in row["models"].items():
                model, var = (m[:-2], "B") if m.endswith("_B") else (m, "A")
                y = b[(b["model"] == model) & (b["variant"] == var) & (b["origin"] == o) & (b["horizon"] == row["h"])]["forecast"]
                assert close(v, r(y.iloc[0], 1))
    lg = b[(b["model"] == "lightgbm") & (b["variant"] == "A") & (b["stage"] == "forward")].sort_values("horizon")
    w0 = next(x for x in s["whatif"]["rows"] if x["scale_pct"] == 0)
    assert all(close(a, r(v, 1)) for a, v in zip(w0["daily"], lg["forecast"]))
    assert close(w0["forecast_total"], r(lg["forecast"].sum(), 1))
    assert close(w0["promo_total"], r(lg["onpromotion"].sum(), 1))
    assert [x["onpromotion"] for x in s["forward"]["rows"]] == [r(v, 1) for v in lg["onpromotion"]]
