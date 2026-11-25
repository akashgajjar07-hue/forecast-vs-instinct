"""Kaggle rule 7B: the published site (docs/data) must not redistribute row-level competition data.

CI-safe part (no Kaggle data needed): no published JSON carries a field whose name means raw sales / promo counts or
the old per-series unit fields, and every series file declares its index scale.
Data part (@pytest.mark.data, needs data/processed/panel.parquet): for all 96 series, no published per-series daily
number equals the raw daily sales or promo count of that day, and no published number equals a private index base.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "docs" / "data"
PANEL = ROOT / "data" / "processed" / "panel.parquet"
FILES = sorted(p for p in DD.rglob("*.json") if "_fixtures" not in p.parts) if DD.exists() else []
SERIES = sorted((DD / "series").glob("*.json")) if (DD / "series").exists() else []

# Raw-data field names (and the pre-compliance per-series unit fields) that must never be published.
FORBIDDEN = {"sales", "actual", "onpromotion", "promo_total", "mean_daily_sales", "mean_promo_count",
             "max_promo_count", "forecast_14d", "p10_14d", "p90_14d", "stat_forecast", "priority", "mean",
             "forecast_total", "daily", "hist_max_promo", "hist_p95_promo", "holiday_mean", "normal_mean",
             "mean_sales", "median_sales", "by_family_units", "mean_transactions", "sales_per_txn", "total_sales"}


def keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}.{k}", k
            yield from keys(v, f"{path}.{k}")
    elif isinstance(o, list):
        for x in o:
            yield from keys(x, f"{path}[]")


def numbers(o):
    if isinstance(o, bool) or o is None:
        return
    if isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from numbers(v)
    elif isinstance(o, list):
        for v in o:
            yield from numbers(v)


@pytest.mark.skipif(not FILES, reason="no published docs/data")
def test_no_forbidden_field_names():
    bad = []
    for p in FILES:
        for path, k in keys(json.loads(p.read_text())):
            if k in FORBIDDEN:
                bad.append(f"{p.relative_to(DD)}: {path}")
    assert not bad, bad[:20]


@pytest.mark.skipif(not SERIES, reason="no published series files")
def test_series_files_are_indexed():
    for p in SERIES:
        s = json.loads(p.read_text())
        assert "index" in s["scale"] and "100" in s["scale"], p.name
        assert "index" in s["promo_scale"], p.name
        assert set(s["history"]) == {"start", "index", "week_start", "weekly_index"}, p.name


@pytest.mark.skipif(not FILES, reason="no published docs/data")
def test_scope_has_no_per_series_levels():
    for p in [DD / "scope.json", ROOT / "config" / "scope.json"]:
        if p.exists():
            for row in json.loads(p.read_text()).get("series", []):
                assert not {"mean_daily_sales", "mean_promo_count"} & set(row), p


# ---------------------------------------------------------------------------------------------- data part
@pytest.fixture(scope="module")
def panel():
    if not PANEL.exists() or not SERIES:
        pytest.skip("needs the processed panel and published series files")
    import pandas as pd
    p = pd.read_parquet(PANEL)
    return p.set_index(["series_id", "date"]).sort_index()


def _raw(panel, sid, dates, col):
    import pandas as pd
    return panel.loc[[(sid, pd.Timestamp(d)) for d in dates], col].to_numpy(float)


@pytest.mark.data
def test_no_published_daily_value_equals_raw(panel):
    """All 96 series. Published observed values (history.index, holdout actual_index) are compared with that day's raw
    sales and forward promo_index with the raw promo count: no equality allowed except raw 0 (0 is 0 on any scale).
    Model outputs on the index scale (forecasts, bands, what-if) are compared with raw sales and promo of the same day:
    isolated coincidences can happen, a systematic match (> 1 % of values) cannot."""
    import pandas as pd
    strict, loose, n_strict, n_loose = [], 0, 0, 0
    for p in SERIES:
        s = json.loads(p.read_text())
        sid = s["series_id"]
        start = pd.Timestamp(s["history"]["start"])
        obs = [(start + pd.Timedelta(days=i), v, "sales") for i, v in enumerate(s["history"]["index"])]
        model = []
        for w in s["holdout"]:
            for r in w["rows"]:
                d = pd.Timestamp(r["date"])
                obs.append((d, r["actual_index"], "sales"))
                model += [(d, v) for v in [r["p10_index"], r["p90_index"], r["forecast_index"]]]
        fdates = [pd.Timestamp(r["date"]) for r in s["forward"]["rows"]]
        for d, r in zip(fdates, s["forward"]["rows"]):
            obs.append((d, r["promo_index"], "onpromotion"))
            model += [(d, v) for v in [r["forecast_index"], r["p10_index"], r["p90_index"], *r["models_index"].values()]]
        for w in s["whatif"]["rows"]:
            model += list(zip(fdates, w["daily_index"]))
        for col in ("sales", "onpromotion"):
            o = [(d, v) for d, v, c in obs if c == col and v is not None]
            raw = _raw(panel, sid, [d for d, _ in o], col)
            vals = np.array([v for _, v in o], float)
            eq = np.isclose(vals, raw, rtol=0, atol=1e-9) & (raw != 0)
            strict += [f"{sid} {d:%Y-%m-%d} {col}={x}" for (d, _), x, e in zip(o, raw, eq) if e]
            n_strict += len(o)
        m = [(d, v) for d, v in model if v is not None]
        vals = np.array([v for _, v in m], float)
        for col in ("sales", "onpromotion"):
            raw = _raw(panel, sid, [d for d, _ in m], col)
            loose += int((np.isclose(vals, raw, rtol=0, atol=1e-9) & (raw != 0)).sum())
            n_loose += len(m)
    assert n_strict > 96 * 300 and n_loose > 96 * 500
    assert not strict, strict[:20]
    assert loose / n_loose < 0.01, (loose, n_loose)


@pytest.mark.data
def test_no_index_base_is_published(panel):
    """No number anywhere in docs/data equals a series sales base, the chain base or a promo base at 3-4 decimals
    (the precision these means were published with before; at 1-2 decimals chance matches with thousands of index
    values are expected and say nothing)."""
    p = panel.reset_index()
    w = p[(p["date"] >= "2015-06-01") & (p["date"] <= "2017-02-06") & (p["closed"] == 0) & p["sales"].notna()]
    bases = np.concatenate([w.groupby("series_id")["sales"].mean().to_numpy(), [w.groupby("date")["sales"].sum().mean()]])
    pm = p[(p["date"] >= "2015-06-01") & (p["date"] <= "2017-08-15") & (p["closed"] == 0)]
    promo = pm.groupby("series_id")["onpromotion"].mean().to_numpy()
    sales_secret, promo_secret = set(), set()
    for k in (3, 4):
        sales_secret |= set(np.round(bases, k).tolist())
        promo_secret |= set(np.round(promo, k).tolist())
    nonint = lambda xs: {x for x in xs if x != int(x)}  # noqa: E731  (counts / days can equal a rounded mean)
    sales_secret, promo_secret = nonint(sales_secret), nonint(promo_secret)
    for f in FILES:
        obj = json.loads(f.read_text())
        hit = sales_secret & set(numbers(obj))
        assert not hit, (f.name, sorted(hit)[:5])
        promo_nums = {v for path, k in keys(obj) if "promo" in k.lower() for v in numbers(_get(obj, path))}
        hit = promo_secret & promo_nums
        assert not hit, (f.name, sorted(hit)[:5])


def _get(obj, path):
    """Values at a keys() path (lists are expanded)."""
    parts = [x for x in re_split(path) if x]
    cur = [obj]
    for part in parts:
        nxt = []
        for c in cur:
            if part == "[]":
                nxt += c if isinstance(c, list) else []
            elif isinstance(c, dict) and part in c:
                nxt.append(c[part])
        cur = nxt
    return cur


def re_split(path):
    import re
    return re.findall(r"\[\]|[^.\[\]]+", path)
