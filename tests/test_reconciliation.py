"""Reconcile the published dashboard JSON (docs/data) with the pipeline outputs.

Every number is recomputed here from outputs/ (parquet + frozen JSON) and the processed panel with independent code:
nothing from src.export is imported. Published values that are rounded (documented in docs/agent_notes/export.md) are
compared after applying the same rounding; everything else is compared at 1e-6.
Run after scripts/08_export_dashboard.py. Needs the Kaggle-derived files, so it is marked `data`.
Per-series values are published as an index; the index bases are recomputed here independently from the panel
(src.indexing is not imported either).
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
def bases(panel):
    """Independent index bases: open-day mean sales 2015-06-01 .. day before the first selection origin."""
    first = pd.Timestamp(json.loads((ROOT / "config" / "split_calendar.json").read_text())["stages"]["selection"][0])
    w = panel[(panel["date"] >= "2015-06-01") & (panel["date"] < first)]
    o = w[(w["closed"] == 0) & w["sales"].notna()]
    series = o.groupby("series_id")["sales"].mean()
    daily = w.groupby("date").agg(tot=("sales", "sum"), n=("sales", "count"), cl=("closed", "sum"))
    chain = float(daily.loc[(daily["n"] == 96) & (daily["cl"] == 0), "tot"].mean())
    pm = panel[(panel["date"] >= "2015-06-01") & (panel["date"] <= FWD) & (panel["closed"] == 0)]
    promo = pm.groupby("series_id")["onpromotion"].mean()
    return {"series": series, "chain": chain, "promo": promo}


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
def window_band_forward(ci):
    """Independent recomputation of the 14-day-total band: frozen per-family relative quantiles
    (outputs/window_intervals.json) applied to each series' 14-day champion forward sum."""
    q = pd.DataFrame(O("window_intervals.json")["quantiles"]).set_index("family")
    f = ci[ci["stage"] == "forward"]
    s = f.groupby(["series_id", "family"])["forecast"].sum().reset_index().set_index("series_id")
    s["p10"] = np.maximum(0.0, s["forecast"] * (1 + q.loc[s["family"], "q_lo"].to_numpy()))
    s["p90"] = s["forecast"] * (1 + q.loc[s["family"], "q_hi"].to_numpy())
    return s


def test_window_band_holdout_coverage(ci):
    """intervals.json window_band coverage recomputed from champion holdout rows + frozen quantiles."""
    wb = J("intervals.json")["window_band"]
    q = pd.DataFrame(O("window_intervals.json")["quantiles"]).set_index("family")
    h = ci[ci["stage"] == "holdout"]
    w = h.groupby(["series_id", "family", "origin"]).agg(F=("forecast", "sum"), A=("actual", "sum"),
                                                         n=("actual", "count")).reset_index()
    w = w[w["n"] == 14]
    lo = np.maximum(0.0, w["F"] * (1 + q.loc[w["family"], "q_lo"].to_numpy()))
    hi = w["F"] * (1 + q.loc[w["family"], "q_hi"].to_numpy())
    cov = float(((w["A"] >= lo) & (w["A"] <= hi)).mean() * 100)
    assert len(w) == wb["n_holdout_windows"] == 576
    assert close(cov, wb["holdout_coverage"])
    assert close(float(((hi - lo) / w["F"]).mean() * 100), wb["holdout_mean_width_pct"])


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


def test_overview_outlook_and_lists(panel, ci, ts, champ, bases):
    ov = J("overview.json")
    b = bases["series"]
    f = ci[ci["stage"] == "forward"].copy()
    f["idx"] = f["forecast"] / f["series_id"].map(b) * 100
    fm = f.groupby("target_date")["idx"].mean()
    assert len(ov["outlook"]["forecast"]) == 14
    for row, (d, v) in zip(ov["outlook"]["forecast"], fm.sort_index().items()):
        assert row["date"] == d.strftime("%Y-%m-%d") and close(row["mean_index"], r(v, 2))
        assert set(row) == {"date", "mean_index"}  # no chain band on the equal-weighted index
    hist = ov["outlook"]["history"]
    assert len(hist) == 56 and hist[-1]["date"] == "2017-08-15"
    for row in hist:
        day = panel[panel["date"] == pd.Timestamp(row["date"])]
        ok_ = (day["closed"] == 0) & day["sales"].notna()
        exp = r((day["sales"] / day["series_id"].map(b) * 100).mean(), 2) if ok_.sum() == 96 else None
        assert close(row["mean_index"], exp)
    ct = pd.read_parquet(OUT / "holdout" / "chain_total.parquet")
    hc = ct[(ct["stage"] == "holdout") & (ct["closed"] == 0) & ct["actual"].notna()]
    cov = ((hc["actual"] >= hc["p10"]) & (hc["actual"] <= hc["p90"])).mean() * 100
    assert close(ov["outlook"]["chain_band_holdout_coverage"], cov) and f"{cov:.1f} %" in ov["outlook"]["band_note"]
    lt = ts[ts["origin"] == FWD].sort_values("priority", ascending=False).reset_index(drop=True)
    rank = {sid: i + 1 for i, sid in enumerate(lt["series_id"])}
    latest = lt[lt["band"].isin(["Review", "Monitor"])].head(8)
    assert [x["series_id"] for x in ov["top_exceptions"]] == list(latest["series_id"])
    assert all(x["priority_rank"] == rank[x["series_id"]] for x in ov["top_exceptions"])
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
def test_arena_holdout_overall(bt, bases):
    rows = {(x["model"], x["variant"]): x for x in J("arena.json")["metrics"]
            if x["stage"] == "holdout" and x["slice_type"] == "overall"}
    h = ok(bt[bt["stage"] == "holdout"])
    assert len(rows) == 6
    for (m, v), g in h.groupby(["model", "variant"]):
        x = rows[(m, v)]
        a, f = g["actual"].to_numpy(float), g["forecast"].to_numpy(float)
        assert close(x["wape"], r(wape(a, f), 4)) and close(x["bias"], r(bias(a, f), 4))
        e = (a - f) / g["series_id"].map(bases["series"]).to_numpy(float) * 100  # index points
        assert close(x["mae"], r(np.abs(e).mean(), 4)) and close(x["rmse"], r(np.sqrt((e ** 2).mean()), 4))
        assert x["n"] == len(g)
    lb = {(x["model"], x["variant"]): x for x in O("holdout_results.json")["leaderboard"]}
    for key, x in rows.items():
        assert close(x["wape"], r(lb[key]["wape"], 4))


def test_arena_mae_index_every_slice(bt, bases):
    """MAE / RMSE of every stage and slice are on the index scale (|A − F| / series base × 100)."""
    ar = J("arena.json")
    assert "index" in ar["mae_unit"]
    rows = {(x["stage"], x["model"], x["variant"], x["slice_type"], x["slice_value"]): x for x in ar["metrics"]}
    ev = ok(bt).copy()
    ev["e"] = (ev["actual"] - ev["forecast"]) / ev["series_id"].map(bases["series"]) * 100
    ev["horizon_bucket"] = np.where(ev["horizon"] <= 7, "1-7", "8-14")
    for st, col in [("family", "family"), ("store", "store_nbr"), ("horizon", "horizon"), ("horizon_bucket", "horizon_bucket")]:
        for (stage, m, v, val), g in ev.groupby(["stage", "model", "variant", col]):
            if stage == "forward":
                continue
            x = rows[(stage, m, v, st, str(val))]
            assert close(x["mae"], r(g["e"].abs().mean(), 4)) and close(x["rmse"], r(np.sqrt((g["e"] ** 2).mean()), 4))
    g = ev[(ev["stage"] == "holdout") & (ev["model"] == "lightgbm") & (ev["variant"] == "A")].copy()
    g["wk"] = np.where(g["horizon"] <= 7, "week1", "week2")
    s = g.groupby(["series_id", "origin", "wk"])[["actual", "forecast"]].sum().reset_index()
    s["e"] = (s["actual"] - s["forecast"]) / s["series_id"].map(bases["series"]) * 100
    for wk, d in s.groupby("wk"):
        assert close(rows[("holdout", "lightgbm", "A", "week_bucket", wk)]["mae"], r(d["e"].abs().mean(), 4))


def test_arena_week_bucket(bt):
    rows = {(x["model"], x["variant"], x["slice_value"]): x for x in J("arena.json")["metrics"]
            if x["stage"] == "holdout" and x["slice_type"] == "week_bucket"}
    g = ok(bt[(bt["stage"] == "holdout") & (bt["model"] == "lightgbm") & (bt["variant"] == "A")]).copy()
    g["wk"] = np.where(g["horizon"] <= 7, "week1", "week2")
    s = g.groupby(["series_id", "origin", "wk"])[["actual", "forecast"]].sum().reset_index()
    for wk, d in s.groupby("wk"):
        assert close(rows[("lightgbm", "A", wk)]["wape"], r(wape(d["actual"], d["forecast"]), 4))


# ---------------------------------------------------------------- intervals
def test_intervals(ci, bases):
    iv = J("intervals.json")
    hc = ok(ci[ci["stage"] == "holdout"])
    inside = (hc["actual"] >= hc["p10"]) & (hc["actual"] <= hc["p90"])
    wi = (hc["p90"] - hc["p10"]) / hc["series_id"].map(bases["series"]) * 100
    assert close(iv["holdout"]["coverage"], inside.mean() * 100)
    assert close(iv["holdout"]["mean_width"], wi.mean())
    assert close(iv["holdout"]["mean_width_pct"], (hc["p90"] - hc["p10"]).sum() / hc["forecast"].sum() * 100)
    assert "index" in iv["width_unit"]
    for x in iv["holdout"]["by_family"]:
        m = hc["family"] == x["family"]
        assert close(x["coverage"], r(inside[m].mean() * 100, 4))
        assert close(x["mean_width"], r(wi[m].mean(), 4))
    for x in iv["holdout"]["by_horizon"]:
        m = hc["horizon"] == x["horizon"]
        assert close(x["mean_width"], r(wi[m].mean(), 4))
    assert close(iv["calibration_coverage"], O("intervals.json")["calibration_coverage_leave_one_origin_out"])
    assert close(iv["holdout"]["coverage"], O("holdout_results.json")["champion_system"]["coverage"])


# ---------------------------------------------------------------- trust
def test_trust(ts, ci, bases):
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
    sums = window_band_forward(ci)
    order = latest.sort_values("priority", ascending=False)["series_id"].tolist()
    assert [x["series_id"] for x in t["latest"]] == order
    for i, x in enumerate(t["latest"]):
        s = sums.loc[x["series_id"]]
        k = 100 / 14 / bases["series"][x["series_id"]]
        assert close(x["forecast_14d_index"], r(s["forecast"] * k, 2)) and close(x["p10_14d_index"], r(s["p10"] * k, 2))
        assert close(x["p90_14d_index"], r(s["p90"] * k, 2))
        assert x["priority_rank"] == i + 1
    cands = {c["name"]: c["spearman"] for c in O("trust_calibration.json")["candidates"]}
    for c in t["calibration_choice"]["candidates"]:
        assert c["spearman"] == cands[c["name"]]


# ---------------------------------------------------------------- S&OP
def test_sop_totals_and_fva(bt, ci, ts, champ, bases):
    sop = J("sop.json")
    fwd = ci[ci["stage"] == "forward"]
    assert len(sop["rows"]) == 96
    # index back to units with the independently computed bases reproduces the chain total
    back = sum(x["stat_index"] * bases["series"][x["series_id"]] * 14 / 100 for x in sop["rows"])
    assert abs(back / fwd["forecast"].sum() - 1) < 1e-4
    sums = window_band_forward(ci)
    for x in sop["rows"]:
        k = 100 / 14 / bases["series"][x["series_id"]]
        assert close(x["stat_index"], r(sums.loc[x["series_id"], "forecast"] * k, 2))
        assert close(x["p10_index"], r(sums.loc[x["series_id"], "p10"] * k, 2))
        assert close(x["p90_index"], r(sums.loc[x["series_id"], "p90"] * k, 2))
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
def test_scenario_baselines(ci, bases):
    """Index-days: 14 x equal-weighted mean of the family's series' 14-day mean forecast index."""
    sc = J("scenario.json")
    f = ci[ci["stage"] == "forward"].copy()
    f["idx"] = f["forecast"] / f["series_id"].map(bases["series"]) * 100
    m = f.groupby(["family", "series_id"])["idx"].mean().groupby("family").mean() * 14
    assert "index-days" in sc["baseline_unit"]
    for x in sc["families"]:
        assert close(x["baseline_14d"], m[x["family"]])
    assert sc["assumption_label"] == "Editable demonstration assumptions, not measured UAE uplift"
    assert sc["ecuador_analogue"] == O("eda/ecuador_analogue.json")


# ---------------------------------------------------------------- series sample
@pytest.mark.parametrize("slug,sid", [("12_beverages", "12|BEVERAGES"), ("49_grocery-i", "49|GROCERY I"),
                                      ("27_bread-bakery", "27|BREAD/BAKERY"), ("53_frozen-foods", "53|FROZEN FOODS")])
def test_series_sample(slug, sid, bt, ci, panel, bases):
    s = json.loads((DD / "series" / f"{slug}.json").read_text())
    assert s["series_id"] == sid
    b, pm = bases["series"][sid], bases["promo"][sid]
    X = lambda v: None if v is None or pd.isna(v) else r(float(v) / b * 100, 1)  # noqa: E731
    P = lambda v: r(float(v) / pm * 100, 1)  # noqa: E731
    p = panel[(panel["series_id"] == sid) & (panel["date"] >= "2017-01-01") & (panel["date"] <= "2017-08-15")].sort_values("date")
    exp = [None if c == 1 or pd.isna(v) else X(v) for v, c in zip(p["sales"], p["closed"])]
    assert len(s["history"]["index"]) == len(exp)
    assert all(close(a, e) for a, e in zip(s["history"]["index"], exp))
    # weekly index: Monday weeks from 2015-06-01, mean over open days
    q = panel[(panel["series_id"] == sid) & (panel["closed"] == 0) & panel["sales"].notna()].copy()
    q["wk"] = q["date"] - pd.to_timedelta(q["date"].dt.dayofweek, unit="D")
    wm = q.groupby("wk")["sales"].mean()
    assert s["history"]["week_start"][0] == "2015-06-01" and s["history"]["week_start"][-1] == "2017-08-07"
    for w, v in zip(s["history"]["week_start"], s["history"]["weekly_index"]):
        assert close(v, X(wm.get(pd.Timestamp(w))))
    bb = bt[bt["series_id"] == sid]
    for win in s["holdout"]:
        o = pd.Timestamp(win["origin"])
        c = ci[(ci["series_id"] == sid) & (ci["origin"] == o)].sort_values("horizon")
        for row, (_, x) in zip(win["rows"], c.iterrows()):
            act = None if x["closed"] == 1 or pd.isna(x["actual"]) else x["actual"]
            assert close(row["actual_index"], X(act)) and close(row["p10_index"], X(x["p10"]))
            assert close(row["p90_index"], X(x["p90"])) and close(row["forecast_index"], X(x["forecast"]))
            assert "models_index" not in row  # champion only on holdout windows
    lg = bb[(bb["model"] == "lightgbm") & (bb["variant"] == "A") & (bb["stage"] == "forward")].sort_values("horizon")
    w0 = next(x for x in s["whatif"]["rows"] if x["scale_pct"] == 0)
    assert all(close(a, X(v)) for a, v in zip(w0["daily_index"], lg["forecast"]))
    assert close(w0["forecast_index"], X(lg["forecast"].sum() / 14))
    assert close(w0["promo_index"], P(lg["onpromotion"].mean()))
    assert [x["promo_index"] for x in s["forward"]["rows"]] == [P(v) for v in lg["onpromotion"]]
    fc = ci[(ci["series_id"] == sid) & (ci["stage"] == "forward")].sort_values("horizon")
    assert [x["forecast_index"] for x in s["forward"]["rows"]] == [X(v) for v in fc["forecast"]]
