"""Split calendar, harness masking, holdout guard (CI) + real-data integrity (marked `data`)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.backtest import masked_view, run_origin, evaluable
from src.freeze import freeze, verify, HoldoutGuardError
from src.splits import load_split_calendar, validate_split_calendar, STAGES


# ---------- split calendar (committed config -> runs in CI) ----------
def test_split_calendar_ordering_and_no_overlap():
    cal = load_split_calendar()
    h = cal["horizon"]
    windows = []
    for s in STAGES:
        o = pd.to_datetime(cal["stages"][s])
        windows.append((o.min() + pd.Timedelta(days=1), o.max() + pd.Timedelta(days=h)))
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert e1 < s2, "target windows overlap across stages"
    assert windows[-1][1] <= pd.Timestamp("2017-08-15")
    assert len(cal["stages"]["selection"]) >= 8 and len(cal["stages"]["calibration"]) >= 6
    assert len(cal["stages"]["holdout"]) >= 6


def test_split_validator_rejects_overlap():
    cal = load_split_calendar()
    bad = json.loads(json.dumps(cal))
    bad["stages"]["calibration"][0] = bad["stages"]["selection"][-1]
    with pytest.raises(AssertionError):
        validate_split_calendar(bad)


# ---------- harness masking ----------
def test_masked_view_hides_future_sales_and_promo_B(synthetic_panel):
    T = pd.Timestamp("2016-05-01")
    vA = masked_view(synthetic_panel, T, "A")
    vB = masked_view(synthetic_panel, T, "B")
    assert vA.loc[vA.date > T, "sales"].isna().all()
    assert vA.loc[vA.date > T, "onpromotion"].notna().all()  # A: promo plan known
    assert vB.loc[vB.date > T, "onpromotion"].isna().all()   # B: no future promo
    pd.testing.assert_frame_equal(vA[vA.date <= T], synthetic_panel[synthetic_panel.date <= T])


def _dummy(view, origin, horizon):
    last = view[view.date <= origin].groupby("series_id")["sales"].last()
    return pd.DataFrame([{"series_id": s, "horizon": h, "forecast": v - 1e9 if s.endswith("PRODUCE") else v}
                         for s, v in last.items() for h in range(1, horizon + 1)])


def test_run_origin_completeness_and_clipping(synthetic_panel):
    out = run_origin("naive", _dummy, synthetic_panel, "2016-05-01", "selection")
    assert len(out) == synthetic_panel.series_id.nunique() * C.HORIZON
    assert (out.forecast >= 0).all()  # negative predictions clipped
    assert set(out.horizon) == set(range(1, 15))


def test_run_origin_rejects_incomplete_output(synthetic_panel):
    bad = lambda v, o, h: _dummy(v, o, h).query("horizon < 14")
    with pytest.raises(ValueError):
        run_origin("x", bad, synthetic_panel, "2016-05-01", "selection")


def test_evaluable_excludes_closed():
    from tests.conftest import make_synthetic_panel
    out = run_origin("naive", _dummy, make_synthetic_panel(start="2016-10-01", n_days=120), "2016-12-20", "selection")
    ev = evaluable(out)
    assert (ev.closed == 0).all() and ev.actual.notna().all()
    assert len(ev) < len(out)  # 25 Dec dropped


# ---------- holdout guard ----------
def test_freeze_verify_roundtrip_and_tamper(tmp_path):
    freeze("selection", {"BEVERAGES": "lightgbm"}, root=tmp_path)
    assert verify("selection", root=tmp_path)["BEVERAGES"] == "lightgbm"
    p = tmp_path / "selection.json"
    p.write_text(p.read_text().replace("lightgbm", "naive"))
    with pytest.raises(HoldoutGuardError):
        verify("selection", root=tmp_path)


def test_verify_refuses_missing_hash(tmp_path):
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(HoldoutGuardError):
        verify("selection", root=tmp_path)


# ---------- real data integrity ----------
@pytest.mark.data
def test_panel_keys_unique_and_complete(real_panel):
    p = real_panel
    assert not p.duplicated(["series_id", "date"]).any()
    assert p.series_id.nunique() == 96
    n_days = (p.date.max() - p.date.min()).days + 1
    assert (p.groupby("series_id").size() == n_days).all()  # full daily calendar
    assert p.date.min() == pd.Timestamp(C.MODEL_START)
    assert p.loc[p.is_future == 0, "date"].max() == pd.Timestamp(C.LAST_ACTUAL)


@pytest.mark.data
def test_closed_days_are_nan_not_zero(real_panel):
    p = real_panel
    assert p.loc[p.closed == 1, "sales"].isna().all()
    xmas = p[(p.date.dt.month == 12) & (p.date.dt.day == 25)]
    assert (xmas.closed == 1).all()
    hist = p[(p.is_future == 0) & (p.closed == 0)]
    assert hist.sales.notna().all() and (hist.sales >= 0).all()


@pytest.mark.data
def test_future_rows_have_no_sales(real_panel):
    f = real_panel[real_panel.is_future == 1]
    assert len(f) > 0 and f.sales.isna().all() and f.onpromotion.notna().all()


@pytest.mark.data
def test_scope_criteria():
    s = json.loads((C.CONFIG / "scope.json").read_text())
    st = pd.DataFrame(s["stores"])
    assert len(st) == 12 and st.city.nunique() >= 4 and st.type.nunique() == 5
    assert s["cutoff_exclusive"] == load_split_calendar()["stages"]["selection"][0]
