"""Baselines + runner. CI tests use `synthetic_panel`; the `data` test checks saved real backtests."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.backtest import run_origin
from src.models import baselines as B
from src.runner import load_backtests, run_model_stages, save_backtest

T = pd.Timestamp("2016-05-03")  # a Tuesday inside the synthetic panel (2016-01-01 + 240 days)
SID = "1|BEVERAGES"


def _sales(p, sid, date):
    return float(p.loc[(p.series_id == sid) & (p.date == pd.Timestamp(date)), "sales"].iloc[0])


def _fc(fc, sid, h):
    return float(fc.loc[(fc.series_id == sid) & (fc.horizon == h), "forecast"].iloc[0])


def _ALL():
    return {
        "naive": (B.naive, {}),
        "seasonal_naive": (B.seasonal_naive, {}),
        "moving_average": (B.moving_average, {"window": 28}),
        "holt_winters": (B.holt_winters, {}),
    }


# ---------- hand-checkable values ----------
def test_naive_is_last_observed(synthetic_panel):
    p = synthetic_panel
    fc = B.naive(p, T, 14)
    assert all(_fc(fc, SID, h) == _sales(p, SID, T) for h in range(1, 15))


def test_naive_skips_closed_origin_day():
    from tests.conftest import make_synthetic_panel
    q = make_synthetic_panel(start="2016-10-01", n_days=120)
    xmas = pd.Timestamp("2016-12-25")
    fc = B.naive(q, xmas, 14)
    assert _fc(fc, SID, 1) == _sales(q, SID, "2016-12-24")


def test_seasonal_naive_same_weekday_last_week(synthetic_panel):
    p = synthetic_panel
    fc = B.seasonal_naive(p, T, 14)
    for h in range(1, 15):
        lag = 7 if h <= 7 else 14
        src = T + pd.Timedelta(days=h - lag)
        assert src <= T and src.weekday() == (T + pd.Timedelta(days=h)).weekday()
        assert _fc(fc, SID, h) == _sales(p, SID, src)


def test_seasonal_naive_falls_back_to_earlier_week_when_closed():
    from tests.conftest import make_synthetic_panel
    q = make_synthetic_panel(start="2016-10-01", n_days=120)
    Tq = pd.Timestamp("2016-12-27")  # h=5 -> 2017-01-01 (closed); h=12 -> 2016-12-25 (closed)
    fc = B.seasonal_naive(q, Tq, 14)
    assert np.isnan(_sales(q, SID, "2016-12-25"))
    # h=5: target 2017-01-01, lag 7 -> 2016-12-25 closed -> 2016-12-18
    assert _fc(fc, SID, 5) == _sales(q, SID, "2016-12-18")
    # h=12: target 2017-01-08, lag 14 -> 2016-12-25 closed -> 2016-12-18
    assert _fc(fc, SID, 12) == _sales(q, SID, "2016-12-18")
    assert _fc(fc, SID, 1) == _sales(q, SID, "2016-12-21")


def test_moving_average_value_and_ignores_closed():
    from tests.conftest import make_synthetic_panel
    q = make_synthetic_panel(start="2016-10-01", n_days=120)
    Tq = pd.Timestamp("2016-12-27")
    fc = B.moving_average(q, Tq, 14, window=7)
    s = q[(q.series_id == SID) & (q.date > Tq - pd.Timedelta(days=7)) & (q.date <= Tq)].sales
    assert s.isna().sum() == 1  # 25 Dec in window
    exp = s.dropna().mean()  # mean of the 6 open days, closed day not counted as 0
    assert _fc(fc, SID, 1) == pytest.approx(exp)
    assert _fc(fc, SID, 1) > s.fillna(0).mean()


# ---------- leakage ----------
@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "moving_average", "holt_winters"])
def test_no_use_of_data_after_origin(synthetic_panel, name):
    fn, kw = _ALL()[name]
    p = synthetic_panel
    a = fn(p, T, 14, **kw)
    q = p.copy()
    after = q.date > T
    q.loc[after, "sales"] = q.loc[after, "sales"] * 50 + 1e6
    q.loc[after, "onpromotion"] = 999
    q.loc[after, "closed"] = 1
    b = fn(q, T, 14, **kw)
    pd.testing.assert_frame_equal(a.sort_values(["series_id", "horizon"]).reset_index(drop=True),
                                  b.sort_values(["series_id", "horizon"]).reset_index(drop=True))


# ---------- Holt-Winters ----------
def test_holt_winters_reasonable(synthetic_panel):
    fc = B.holt_winters(synthetic_panel, T, 14)
    lvl = synthetic_panel[(synthetic_panel.series_id == SID) & (synthetic_panel.date <= T)].sales.tail(28).mean()
    assert np.isfinite(fc.forecast).all()
    assert abs(fc[fc.series_id == SID].forecast.mean() / lvl - 1) < 0.15


def test_holt_winters_fallback_to_seasonal_naive(synthetic_panel):
    B.HW_FALLBACKS.clear()
    bad = lambda job: (None, "fit_error:Injected")
    fc = B.holt_winters(synthetic_panel, T, 14, _fit=bad)
    sn = B.seasonal_naive(synthetic_panel, T, 14)
    m = fc.merge(sn, on=["series_id", "horizon"])
    assert np.allclose(m.forecast_x, m.forecast_y)
    assert len(B.HW_FALLBACKS) == synthetic_panel.series_id.nunique()
    assert B.HW_FALLBACKS[0]["reason"] == "fit_error:Injected"
    B.HW_FALLBACKS.clear()


def test_holt_winters_real_fit_error_is_caught():
    y = np.full(40, np.nan)  # all NaN -> cannot fill -> fallback reason
    f, reason = B._fit_one((y, 14, None))
    assert f is None and reason is not None


def test_fill_same_weekday():
    y = np.arange(21, dtype=float)
    y[10] = np.nan
    y[2] = np.nan
    z = B._fill_same_weekday(y)
    assert z[10] == 3.0 and z[2] == 9.0


# ---------- completeness through the harness ----------
@pytest.mark.parametrize("name", ["naive", "seasonal_naive", "moving_average", "holt_winters"])
def test_completeness_via_harness(synthetic_panel, name):
    fn, kw = _ALL()[name]
    out = run_origin(name, fn, synthetic_panel, T, "selection", **kw)
    assert len(out) == 4 * 14
    assert out.forecast.notna().all() and (out.forecast >= 0).all()
    assert set(out.horizon) == set(range(1, 15))


# ---------- runner ----------
def test_runner_refuses_final_stages(synthetic_panel):
    for st in (["holdout"], ["selection", "holdout"], ["forward"]):
        with pytest.raises(PermissionError):
            run_model_stages("naive", B.naive, synthetic_panel, st, save=False)


def test_runner_truncates_panel_and_saves_merging(synthetic_panel, tmp_path):
    cal = {"stages": {"selection": ["2016-04-05", "2016-04-12"], "calibration": ["2016-05-03"],
                      "holdout": ["2016-06-07"]}}
    seen = []

    def spy(view, origin, horizon, **kw):
        seen.append(view.date.max())
        return B.naive(view, origin, horizon)

    bt = run_model_stages("naive", spy, synthetic_panel, ["selection", "calibration"], cal=cal,
                          out_dir=tmp_path)
    assert max(seen) == pd.Timestamp("2016-05-17")  # last calibration origin + 14, nothing later
    assert len(bt) == 3 * 4 * 14 and set(bt.stage) == {"selection", "calibration"}
    # re-saving one stage replaces only that stage
    save_backtest(bt[bt.stage == "selection"], "naive", tmp_path)
    again = load_backtests(tmp_path)
    assert len(again) == len(bt) and not again.duplicated(["stage", "origin", "series_id", "horizon"]).any()


# ---------- saved real backtests ----------
@pytest.mark.data
def test_saved_real_baseline_backtests():
    names = list(B.BASELINES)
    files = [C.OUTPUTS / "backtests" / f"{n}.parquet" for n in names]
    if not all(f.exists() for f in files):
        pytest.skip("baseline backtests not built (run scripts/02_run_baselines.py)")
    bt = load_backtests(models=names)
    assert set(bt.model) == set(names)
    assert not bt.stage.isin(["holdout", "forward"]).any()
    assert (bt.variant == "A").all()
    assert (bt.forecast >= 0).all() and bt.forecast.notna().all()
    assert (bt.target_date <= pd.Timestamp("2017-06-27")).all()
    sizes = bt.groupby(["model", "origin"]).size()
    assert (sizes == 96 * 14).all()
    assert bt.groupby("model").origin.nunique().eq(C.N_SELECTION + C.N_CALIBRATION).all()
    keys = {m: set(map(tuple, g[["origin", "series_id", "horizon"]].to_numpy())) for m, g in bt.groupby("model")}
    ref = keys[names[0]]
    assert all(k == ref for k in keys.values())
