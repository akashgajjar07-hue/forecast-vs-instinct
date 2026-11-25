"""Forecast Trust Index tests. CI tests use synthetic data; @pytest.mark.data tests read the real
calibration output (outputs/trust_calibration.json, outputs/trust_weights.json)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import trust as TR
from src.backtest import run_backtest
from src.intervals import fit_quantiles
from src.selection import champion_rows
from tests.conftest import make_synthetic_panel


# ---------------------------------------------------------------- synthetic pipeline helpers
def _const(fc_fn):
    def fn(view, origin, horizon, **_):
        rows = []
        for sid, g in view[view["date"] <= origin].groupby("series_id"):
            s = g["sales"].dropna()
            for h in range(1, horizon + 1):
                rows.append({"series_id": sid, "horizon": h, "forecast": fc_fn(s, h)})
        return pd.DataFrame(rows)
    return fn


MODEL_FNS = {
    "naive": _const(lambda s, h: s.iloc[-1]),
    "seasonal_naive": _const(lambda s, h: s.iloc[-7 + (h - 1) % 7]),
    "moving_average": _const(lambda s, h: s.iloc[-28:].mean()),
    "holt_winters": _const(lambda s, h: s.iloc[-7:].mean()),
    "lightgbm": _const(lambda s, h: s.iloc[-14:].mean() * 1.02),
}
CHAMPS = {"BEVERAGES": {"model": "holt_winters"}, "PRODUCE": {"model": "seasonal_naive"}}


def _setup(panel, first="2016-03-01", n=12, n_fit=6):
    origins = list(pd.date_range(first, periods=n, freq="7D"))
    bt = pd.concat([run_backtest(m, f, panel, [("selection", o) for o in origins]) for m, f in MODEL_FNS.items()],
                   ignore_index=True)
    q = fit_quantiles(champion_rows(bt[bt["origin"].isin(origins[:n_fit])], CHAMPS), panel)
    return bt, origins, {o: q for o in origins}


@pytest.fixture(scope="module")
def synth():
    panel = make_synthetic_panel()
    bt, origins, qmap = _setup(panel)
    comp = TR.components(bt, panel, CHAMPS, origins[4:], qmap)
    return panel, bt, origins, qmap, comp


# ---------------------------------------------------------------- bounds
def test_components_and_score_bounded(synth):
    *_, comp = synth
    assert len(comp) == 4 * 8
    for c in TR.COMPONENTS:
        assert comp[c].notna().all(), c
        assert comp[c].between(0, 100).all(), c
    for w in TR.standard_candidates().values():
        s = TR.score(comp, w)
        assert s.between(0, 100).all()


def test_extreme_inputs_still_bounded(synth):
    panel, bt, origins, qmap, _ = synth
    p = panel.copy()
    sid = "2|PRODUCE"
    p.loc[(p["series_id"] == sid) & (p["date"] <= origins[6]) & (p["date"] > origins[6] - pd.Timedelta(days=30)), "sales"] = 0.0
    b = bt.copy()
    b.loc[(b["series_id"] == sid) & (b["model"] == "naive"), "forecast"] *= 50  # wild disagreement
    comp = TR.components(b, p, CHAMPS, [origins[6]], qmap)
    for c in TR.COMPONENTS:
        assert comp[c].between(0, 100).all(), c
    r = comp.set_index("series_id").loc[sid]
    assert r["agreement"] == 0
    assert r["zero_share_56"] > 0.4


def test_component_formulas_match_definitions(synth):
    panel, bt, origins, _, comp = synth
    r = comp.iloc[0]
    T, sid = r["origin"], r["series_id"]
    assert r["accuracy"] == pytest.approx(100 * (1 - min(r["wape_hist"], 50) / 50))
    assert r["interval"] == pytest.approx(100 * (1 - min(abs(r["coverage_hist"] - 80), 40) / 40))
    w = panel[(panel.series_id == sid) & (panel.date > T - pd.Timedelta(days=56)) & (panel.date <= T)]
    ok = w[(w.closed == 0) & w.sales.notna()]["sales"]
    cv = ok.std(ddof=0) / ok.mean()
    assert r["stability"] == pytest.approx(100 * (1 - min(cv, 1)) * (1 - (ok == 0).mean()))
    assert r["completeness"] == pytest.approx(100 * len(ok) / 56)
    tot = bt[(bt.origin == T) & (bt.series_id == sid)].groupby("model")["forecast"].sum()
    assert r["agreement"] == pytest.approx(100 * (1 - min(tot.std(ddof=0) / tot.mean() / 0.5, 1)))
    # accuracy window: only origins with o + 14 <= T, at most 8
    assert r["n_hist_windows"] == len([o for o in origins if o + pd.Timedelta(days=14) <= T][-8:])


# ---------------------------------------------------------------- flags and bands
def test_band_edges():
    assert TR.band(59.99, []) == "Review"
    assert TR.band(60, []) == "Monitor"
    assert TR.band(79.99, []) == "Monitor"
    assert TR.band(80, []) == "Accept"
    assert TR.band(100, []) == "Accept"
    assert TR.band(0, []) == "Review"


def test_hard_flag_forces_review():
    for f in ["low_completeness", "holiday_closure", "promo_above_p95", "missing_accuracy"]:
        assert TR.band(99.0, [f]) == "Review"
    s = pd.Series([95.0, 95.0])
    assert TR.bands_for(s, pd.Series([[], ["holiday_closure"]])).tolist() == ["Accept", "Review"]


def test_low_completeness_flag(synth):
    panel, bt, origins, qmap, _ = synth
    T, sid = origins[6], "1|BEVERAGES"
    p = panel.copy()
    p.loc[(p.series_id == sid) & (p.date <= T) & (p.date > T - pd.Timedelta(days=15)), "sales"] = np.nan
    comp = TR.components(bt, p, CHAMPS, [T], qmap).set_index("series_id")
    assert comp.loc[sid, "completeness"] == pytest.approx(100 * 41 / 56)
    assert "low_completeness" in comp.loc[sid, "hard_flags"]
    assert TR.band(TR.score(comp).loc[sid], comp.loc[sid, "hard_flags"]) == "Review"
    assert "low_completeness" not in comp.loc["2|BEVERAGES", "hard_flags"]


def test_promo_flag(synth):
    panel, bt, origins, qmap, _ = synth
    T, sid = origins[6], "2|BEVERAGES"
    b = bt.copy()
    b.loc[(b.series_id == sid) & (b.origin == T), "onpromotion"] = 500  # planned promo far above history
    base = TR.components(bt, panel, CHAMPS, [T], qmap).set_index("series_id")
    comp = TR.components(b, panel, CHAMPS, [T], qmap).set_index("series_id")
    assert "promo_above_p95" not in base.loc[sid, "hard_flags"]
    assert "promo_above_p95" in comp.loc[sid, "hard_flags"]
    others = [s for s in comp.index if s != sid]
    assert comp.loc[others, "hard_flags"].tolist() == base.loc[others, "hard_flags"].tolist()


def test_holiday_closure_flag():
    panel = make_synthetic_panel(n_days=200, start="2015-08-01")
    bt, origins, qmap = _setup(panel, first="2015-10-06", n=12)
    comp = TR.components(bt, panel, CHAMPS, origins[4:], qmap)
    for T, g in comp.groupby("origin"):
        win = pd.date_range(T + pd.Timedelta(days=1), periods=14).strftime("%m-%d")
        expect = bool(set(win) & {"12-25", "01-01"})
        assert all(("holiday_closure" in f) == expect for f in g["hard_flags"]), T
    assert comp["hard_flags"].apply(lambda f: "holiday_closure" in f).any()


def test_missing_history_flags():
    panel = make_synthetic_panel()
    bt, origins, qmap = _setup(panel, n=3, n_fit=3)
    comp = TR.components(bt, panel, CHAMPS, [origins[0]], qmap)  # no completed window yet
    assert comp["hard_flags"].apply(lambda f: "missing_accuracy" in f and "missing_interval" in f).all()
    assert TR.score(comp).between(0, 100).all()


# ---------------------------------------------------------------- as-of property
def test_components_use_only_information_at_T(synth):
    panel, bt, origins, qmap, _ = synth
    T = origins[7]
    base = TR.components(bt, panel, CHAMPS, [T], qmap)
    rng = np.random.default_rng(3)
    p = panel.copy()
    after = p["date"] > T
    p.loc[after, "sales"] = p.loc[after, "sales"] * rng.uniform(0, 20, after.sum())
    p.loc[after, "closed"] = 1
    b = bt.copy()
    fut = b["target_date"] > T
    b.loc[fut, "actual"] = b.loc[fut, "actual"] * rng.uniform(0, 20, fut.sum())
    b.loc[fut, "closed"] = rng.integers(0, 2, fut.sum())
    b.loc[b["origin"] > T, "forecast"] = 1e9  # later origins' forecasts are not known at T either
    pert = TR.components(b, p, CHAMPS, [T], qmap)
    pd.testing.assert_frame_equal(base, pert)


# ---------------------------------------------------------------- weights
def test_weights_sum_to_one():
    assert sum(C.TRUST_WEIGHTS_DEFAULT.values()) == pytest.approx(1.0)
    TR.check_weights(C.TRUST_WEIGHTS_DEFAULT)
    with pytest.raises(ValueError):
        TR.check_weights({**C.TRUST_WEIGHTS_DEFAULT, "accuracy": 0.4})
    with pytest.raises(ValueError):
        TR.check_weights({"accuracy": 1.2, "interval": -0.2})
    grid = TR.grid_candidates(0.1)
    assert len(grid) == 1001
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in grid.values())
    cands = TR.standard_candidates()
    assert {"default", "equal", "volatility_only"} <= set(cands)


def test_volatility_only_score_equals_stability(synth):
    *_, comp = synth
    s = TR.score(comp, TR.standard_candidates()["volatility_only"])
    np.testing.assert_allclose(s, comp["stability"])


def test_choose_weights_rule():
    w = dict(C.TRUST_WEIGHTS_DEFAULT)
    t = pd.DataFrame([
        {"name": "default", "spearman": -0.30, "spearman_first_half": -0.30, "spearman_second_half": -0.30},
        {"name": "x", "spearman": -0.32, "spearman_first_half": -0.40, "spearman_second_half": -0.25},
        {"name": "y", "spearman": -0.20, "spearman_first_half": -0.2, "spearman_second_half": -0.2}])
    t["weights"] = [w] * 3
    assert TR.choose_weights(t)[0] == "default"  # gain 0.02 < 0.03
    t.loc[1, "spearman"] = -0.35
    assert TR.choose_weights(t)[0] == "default"  # gain ok, but worse on second half
    t.loc[1, "spearman_second_half"] = -0.31
    assert TR.choose_weights(t)[0] == "x"


def test_choose_weights_tie_break_and_identifiability():
    d = dict(C.TRUST_WEIGHTS_DEFAULT)
    far = {"accuracy": 0.9, "interval": 0, "stability": 0, "agreement": 0, "completeness": 0.1}
    near = {"accuracy": 0.5, "interval": 0.1, "stability": 0.2, "agreement": 0.1, "completeness": 0.1}
    t = pd.DataFrame([
        {"name": "default", "weights": d, "spearman": -0.3, "spearman_first_half": -0.3, "spearman_second_half": -0.3},
        {"name": "far", "weights": far, "spearman": -0.5, "spearman_first_half": -0.5, "spearman_second_half": -0.5},
        {"name": "near", "weights": near, "spearman": -0.5, "spearman_first_half": -0.5, "spearman_second_half": -0.5}])
    assert TR.choose_weights(t)[0] == "near"
    comp = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in TR.COMPONENTS}).assign(completeness=100.0)
    keep, const = TR.identifiable_candidates(TR.standard_candidates(0.1), comp)
    assert const == ["completeness"]
    assert {"default", "equal", "volatility_only"} <= set(keep)
    grid = [w for n, w in keep.items() if n.startswith("grid:")]
    assert grid and all(abs(w["completeness"] - 0.1) < 1e-9 for w in grid)


def test_compare_weights_runs(synth):
    panel, bt, origins, qmap, comp = synth
    real = TR.realised_window_wape(champion_rows(bt, CHAMPS))
    tab = TR.compare_weights(comp, real, TR.standard_candidates(0.5), split_origins=origins[4:8])
    assert tab["n"].eq(len(comp)).all()
    assert tab["spearman"].dropna().between(-1, 1).all()


# ---------------------------------------------------------------- priority
def test_priority():
    s = pd.Series([50.0, 90.0, 100.0])
    f = pd.Series([100.0, 300.0, 600.0])
    np.testing.assert_allclose(TR.priority(s, f), [5.0, 3.0, 0.0])


# ---------------------------------------------------------------- validation math
def _hand_example():
    # 6 windows: two per band. abs error / actual chosen by hand.
    scored = pd.DataFrame({
        "series_id": list("abcdef"), "origin": pd.Timestamp("2017-07-04"),
        "score": [90, 85, 70, 65, 40, 30], "band": ["Accept", "Accept", "Monitor", "Monitor", "Review", "Review"],
        "stability": [10, 20, 30, 40, 50, 60]})  # volatility-only ranks exactly backwards
    real = pd.DataFrame({
        "series_id": list("abcdef"), "origin": pd.Timestamp("2017-07-04"),
        "sum_abs_error": [10.0, 10.0, 20.0, 20.0, 60.0, 40.0], "sum_actual": [100.0, 100.0, 100.0, 100.0, 200.0, 100.0]})
    real["wape"] = real["sum_abs_error"] / real["sum_actual"] * 100
    return scored, real


def test_validation_math():
    scored, real = _hand_example()
    v = TR.validate(scored, real)
    assert v["n_windows"] == 6
    # wape: 10,10,20,20,30,40 vs score 90..30 -> perfectly decreasing apart from the tie
    assert v["spearman"] < -0.9
    wb = {r["band"]: r for r in v["wape_by_band"]}
    assert wb["Accept"]["wape"] == pytest.approx(10.0) and wb["Accept"]["n"] == 2
    assert wb["Monitor"]["wape"] == pytest.approx(20.0)
    assert wb["Review"]["wape"] == pytest.approx(100 * 100 / 300, abs=0.01)  # ratio of sums, not mean of ratios (35)
    assert v["error_ratio_review_vs_accept"] == pytest.approx(round(100 / 3 / 10, 2))
    assert v["review_share_of_abs_error"] == pytest.approx(round(100 * 100 / 160, 1))
    assert v["review_share_of_windows"] == pytest.approx(round(100 * 2 / 6, 1))
    assert v["volatility_only_spearman"] > 0.9
    assert v["beats_volatility_baseline"] is True
    assert "did not" not in v["verdict"].lower() and "probab" not in v["verdict"].lower()


def test_validation_honest_when_index_fails():
    scored, real = _hand_example()
    scored["score"] = scored["score"].values[::-1]  # now high score = high error
    scored["stability"] = scored["stability"].values[::-1]
    v = TR.validate(scored, real)
    assert v["spearman"] > 0.9
    assert v["beats_volatility_baseline"] is False
    assert "did not" in v["verdict"].lower() and "NOT beat" in v["verdict"]


def test_validation_fits_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from src.schemas import SCHEMAS
    scored, real = _hand_example()
    v = TR.validate(scored, real)
    jsonschema.validate(v, SCHEMAS["trust.json"]["properties"]["validation"])
    json.dumps(v)


def test_realised_window_wape_ignores_closed_and_missing():
    bt = pd.DataFrame({"series_id": "a", "origin": pd.Timestamp("2017-01-03"), "horizon": [1, 2, 3, 4],
                       "actual": [100.0, 100.0, np.nan, 50.0], "forecast": [90.0, 120.0, 5.0, 0.0],
                       "closed": [0, 0, 0, 1]})
    r = TR.realised_window_wape(bt).iloc[0]
    assert r["wape"] == pytest.approx(15.0) and r["n_days"] == 2


# ---------------------------------------------------------------- real calibration output
CAL_PATH = C.OUTPUTS / "trust_calibration.json"


@pytest.mark.data
def test_real_trust_weights_frozen_and_valid():
    if not (C.OUTPUTS / "trust_weights.sha256").exists():
        pytest.skip("trust weights not frozen yet")
    from src.freeze import verify
    tw = verify("trust_weights")
    TR.check_weights(tw["weights"])
    mins = sorted(b["min"] for b in tw["bands"])
    assert mins[0] == 0 and len(mins) == 3
    assert set(tw["constants"]) >= {"wape_cap", "cov_tol", "cv_cap", "agree_cap", "lookback_days", "n_windows"}


@pytest.mark.data
def test_real_calibration_output():
    if not CAL_PATH.exists():
        pytest.skip("run scripts/06_trust_calibrate.py first")
    cal = json.loads(CAL_PATH.read_text())
    names = {c["name"] for c in cal["candidates"]}
    assert {"default", "equal", "volatility_only"} <= names
    assert cal["chosen"] in names
    assert all(-1 <= c["spearman"] <= 1 for c in cal["candidates"] if c["spearman"] is not None)
    assert cal["n_windows"] > 0 and cal["n_windows"] <= 96 * 8
    split = json.loads((C.CONFIG / "split_calendar.json").read_text())["stages"]
    assert set(cal["origins"]) == set(split["calibration"])  # calibration origins only, no holdout
    assert sum(cal["band_distribution"].values()) == 96 * 8
    for c in TR.COMPONENTS:
        s = cal["component_summary"][c]
        assert 0 <= s["min"] <= s["max"] <= 100
