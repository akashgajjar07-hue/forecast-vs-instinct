import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src.intervals import fit_quantiles, apply_intervals, origin_scales, fit_total_quantiles, apply_total_band, total_rows
from src.selection import selection_table, pick_champions, champion_rows


def _bt(models_err: dict, fams=("BEVERAGES", "DAIRY"), stage="selection", n_or=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, err in models_err.items():
        for f in fams:
            for o in range(n_or):
                for h in range(1, 15):
                    a = 100.0
                    rows.append(dict(model=m, variant="A", stage=stage, origin=pd.Timestamp("2017-01-03") + pd.Timedelta(days=7 * o),
                                     series_id=f"1|{f}", store_nbr=1, family=f, horizon=h,
                                     target_date=pd.Timestamp("2017-01-03") + pd.Timedelta(days=7 * o + h),
                                     forecast=a + err[f], actual=a, closed=0, onpromotion=0, is_holiday=0))
    return pd.DataFrame(rows)


def test_champion_lowest_wape_per_family():
    bt = _bt({"naive": {"BEVERAGES": 10, "DAIRY": 2}, "lightgbm": {"BEVERAGES": 5, "DAIRY": 8}})
    ch = pick_champions(selection_table(bt))
    assert ch["BEVERAGES"]["model"] == "lightgbm" and ch["DAIRY"]["model"] == "naive"
    assert ch["BEVERAGES"]["selection_wape"] == pytest.approx(5.0)


def test_champion_tie_break_on_abs_bias():
    bt = _bt({"a_model": {"BEVERAGES": 5, "DAIRY": 5}, "b_model": {"BEVERAGES": -5, "DAIRY": 5}})
    # identical WAPE; make b_model unbiased-ish on BEVERAGES by mixing signs
    b = bt[bt.model == "b_model"].copy()
    b.loc[b.horizon % 2 == 0, "forecast"] = 105.0
    bt = pd.concat([bt[bt.model == "a_model"], b])
    ch = pick_champions(selection_table(bt))
    assert ch["BEVERAGES"]["model"] == "b_model"


def test_champion_ignores_other_stages():
    bt = pd.concat([_bt({"naive": {"BEVERAGES": 1, "DAIRY": 1}, "lightgbm": {"BEVERAGES": 9, "DAIRY": 9}}),
                    _bt({"naive": {"BEVERAGES": 50, "DAIRY": 50}, "lightgbm": {"BEVERAGES": 0, "DAIRY": 0}}, stage="holdout")])
    ch = pick_champions(selection_table(bt))
    assert ch["BEVERAGES"]["model"] == "naive"
    assert len(champion_rows(bt, ch)) == len(bt) / 2


def test_scale_uses_only_past(synthetic_panel):
    T = pd.Timestamp("2016-05-01")
    s1 = origin_scales(synthetic_panel, [T])
    p2 = synthetic_panel.copy(); p2.loc[p2.date > T, "sales"] *= 1000
    s2 = origin_scales(p2, [T])
    pd.testing.assert_frame_equal(s1, s2)


def test_conformal_coverage_on_exchangeable_data(synthetic_panel):
    rng = np.random.default_rng(1)
    rows = []
    for o in pd.date_range("2016-03-01", periods=12, freq="7D"):
        for sid, fam in synthetic_panel[["series_id", "family"]].drop_duplicates().values:
            for h in range(1, 15):
                f = 100.0
                rows.append(dict(origin=o, series_id=sid, family=fam, horizon=h, forecast=f,
                                 actual=f + rng.normal(0, 10), closed=0, target_date=o + pd.Timedelta(days=h)))
    df = pd.DataFrame(rows)
    cal, test = df[df.origin < "2016-04-20"], df[df.origin >= "2016-04-20"]
    q = fit_quantiles(cal, synthetic_panel)
    out = apply_intervals(test, synthetic_panel, q)
    cov = M.coverage(out.actual, out.p10, out.p90)
    assert 70 <= cov <= 90
    assert (out.p10 >= 0).all() and (out.p10 <= out.forecast).all() and (out.p90 >= out.forecast).all()


def test_total_band():
    rows = pd.DataFrame([dict(origin=pd.Timestamp("2017-01-03"), target_date=pd.Timestamp("2017-01-03") + pd.Timedelta(days=h),
                              horizon=h, series_id=s, forecast=10.0, actual=10.0 + (h % 3 - 1), closed=0)
                         for h in range(1, 15) for s in "ab"])
    qt = fit_total_quantiles(rows)
    t = apply_total_band(total_rows(rows), qt)
    assert (t.p10 <= t.forecast).all() and (t.p90 >= t.forecast).all()


def test_window_band_is_not_sum_of_daily_bounds_and_calibrates():
    from src.intervals import fit_window_quantiles, apply_window_band, window_totals
    rng = np.random.default_rng(3)
    rows = []
    for o in pd.date_range("2016-03-01", periods=40, freq="7D"):
        for s, fam in [("1|BEVERAGES", "BEVERAGES"), ("2|BEVERAGES", "BEVERAGES")]:
            shock = rng.normal(0, 0.1)  # window-level error, shared across the 14 days
            for h in range(1, 15):
                rows.append(dict(origin=o, series_id=s, family=fam, horizon=h, forecast=100.0,
                                 actual=100.0 * (1 + shock), closed=0, target_date=o + pd.Timedelta(days=h)))
    df = pd.DataFrame(rows)
    cal, test = df[df.origin < "2016-08-01"], df[df.origin >= "2016-08-01"]
    q = fit_window_quantiles(cal)
    out = apply_window_band(test, q)
    cov = M.coverage(out.actual, out.p10, out.p90)
    assert 65 <= cov <= 95
    assert (out.p10 <= out.forecast).all() and (out.p90 >= out.forecast).all()
    wt = window_totals(df.assign(closed=np.where((df.horizon == 3) & (df.origin == df.origin.min()), 1, 0)))
    assert (~wt.complete).sum() == 2  # windows with a closed day are excluded from fitting
