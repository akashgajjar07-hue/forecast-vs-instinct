"""Acceptance test (owner rule, Kaggle 7B): the 96 private index bases must NOT be identifiable from published files.

Per-series values are published only as an index (value / base x 100). With a base, every published index turns back
into units, so this test plays the attacker: it collects every linear equation in the 96 unknown bases that the
CURRENT tracked files allow and solves them by least squares.

Equation sources scanned (each used only if the needed per-series coefficients are published):
  * unit-valued aggregates: overview KPI (chain 14-day forecast total, units), arena / holdout_results MAE in units,
    intervals / holdout_results mean width in units, demand weekly family totals in units, scenario family baselines
    in units, the S&OP "Review-band share of volume" sentence;
  * volume-weighted ratios (homogeneous equations): arena WAPE and bias for every holdout slice whose rows can be
    rebuilt from series files (overall, family, store, horizon, horizon bucket, week bucket), holdout_results WAPE
    by family.
The test fails if the system reaches rank 96 or the least-squares bases have a median relative error below 20 %.
Run: .venv/bin/python -m pytest -q tests/test_no_reconstruction.py -s   (prints rank and errors)
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "docs" / "data"
OUT = ROOT / "outputs"
PANEL = ROOT / "data" / "processed" / "panel.parquet"

pytestmark = pytest.mark.data


def J(p):
    return json.loads(Path(p).read_text())


def load_series(dd):
    idx = J(dd / "series_index.json")
    return [s["series_id"] for s in idx], {s["series_id"]: J(dd / "series" / f"{s['slug']}.json") for s in idx}


def holdout_rows(ser):
    """Rows (sid, store, family, origin, h, actual_index, {model_key: forecast_index}, p10, p90) from series files,
    whatever field names the published version uses."""
    out = []
    for sid, s in ser.items():
        for w in s.get("holdout", []):
            for r in w["rows"]:
                a = r.get("actual_index", r.get("actual"))
                models = dict(r.get("models_index", r.get("models", {})) or {})
                if "forecast_index" in r and w.get("champion"):
                    models.setdefault(w["champion"], r["forecast_index"])
                out.append((sid, s["store_nbr"], s["family"], w["origin"], r["h"], a, models,
                            r.get("p10_index", r.get("p10")), r.get("p90_index", r.get("p90"))))
    return out


def build_system(dd=DD, out=OUT):
    sids, ser = load_series(dd)
    pos = {s: i for i, s in enumerate(sids)}
    n = len(sids)
    rows_h, y_h, rows_n, y_n, src = [], [], [], [], {}

    def add(kind, coef, val=0.0, name=""):
        if not np.any(coef):
            return
        (rows_h if kind == "h" else rows_n).append(coef)
        if kind == "n":
            y_n.append(val)
        src[name] = src.get(name, 0) + 1

    hr = [r for r in holdout_rows(ser) if r[5] is not None]
    # forward: chain KPI in units
    fwd = np.zeros(n)
    for sid, s in ser.items():
        fwd[pos[sid]] = sum(r.get("forecast_index", r.get("forecast", 0)) for r in s["forward"]["rows"]) / 100
    ov = J(dd / "overview.json")
    add("n", fwd, ov["kpis"]["forecast_14d_total"]["value"], "overview KPI total (units)")

    # arena holdout slices
    ar = J(dd / "arena.json")
    mae_units = "index" not in str(ar.get("mae_unit", "units")).lower()
    for m in ar["metrics"]:
        if m["stage"] != "holdout":
            continue
        key = m["model"] + ("_B" if m["variant"] == "B" else "")
        st, sv = m["slice_type"], m["slice_value"]
        sel = []
        for r in hr:
            if key not in r[6]:
                continue
            if st == "overall" or (st == "family" and r[2] == sv) or (st == "store" and str(r[1]) == sv) or \
                    (st == "horizon" and str(r[4]) == sv) or \
                    (st == "horizon_bucket" and ((r[4] <= 7) == (sv == "1-7"))):
                sel.append(r)
        if st == "week_bucket":
            wk = {}
            for r in hr:
                if key in r[6] and ((r[4] <= 7) == (sv == "week1")):
                    k = (r[0], r[3])
                    a, f = wk.get(k, (0.0, 0.0))
                    wk[k] = (a + r[5], f + r[6][key])
            E, A, F = np.zeros(n), np.zeros(n), np.zeros(n)
            for (sid, _), (a, f) in wk.items():
                E[pos[sid]] += abs(a - f); A[pos[sid]] += a; F[pos[sid]] += f
        elif sel:
            E, A, F = np.zeros(n), np.zeros(n), np.zeros(n)
            for r in sel:
                E[pos[r[0]]] += abs(r[5] - r[6][key]); A[pos[r[0]]] += r[5]; F[pos[r[0]]] += r[6][key]
        else:
            continue
        if m["wape"] is not None:
            add("h", E - m["wape"] / 100 * A, name="arena WAPE slices")
        if m["bias"] is not None:
            add("h", F - A - m["bias"] / 100 * A, name="arena bias slices")
        if mae_units and m["mae"] is not None and st != "week_bucket" and sel:
            add("n", E / 100, m["mae"] * len(sel), "arena MAE slices (units)")

    # holdout_results.json (frozen): WAPE by family x model, overall MAE, champion mean width
    res = J(out / "holdout_results.json")
    for fr in res["by_family"]:
        for model in ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"]:
            E, A = np.zeros(n), np.zeros(n)
            for r in hr:
                if r[2] == fr["family"] and model in r[6]:
                    E[pos[r[0]]] += abs(r[5] - r[6][model]); A[pos[r[0]]] += r[5]
            if A.any():
                add("h", E - fr[model] / 100 * A, name="holdout_results WAPE by family")
    for lb in res["leaderboard"]:
        key = lb["model"] + ("_B" if lb["variant"] == "B" else "")
        sel = [r for r in hr if key in r[6]]
        if sel and len(sel) == lb["n"]:
            E = np.zeros(n)
            for r in sel:
                E[pos[r[0]]] += abs(r[5] - r[6][key])
            add("n", E / 100, lb["mae"] * lb["n"], "holdout_results leaderboard MAE (units)")
    W = np.zeros(n)
    for r in hr:
        W[pos[r[0]]] += r[8] - r[7]
    cs = res["champion_system"]
    if len(hr) == cs["n"]:
        add("n", W / 100, cs["mean_width"] * cs["n"], "holdout_results mean width (units)")
    iv = J(dd / "intervals.json")
    if "index" not in str(iv.get("width_unit", "")).lower():
        add("n", W / 100, iv["holdout"]["mean_width"] * len(hr), "intervals mean width (units)")
        for fb in iv["holdout"]["by_family"]:
            Wf = np.zeros(n); k = 0
            for r in hr:
                if r[2] == fb["family"]:
                    Wf[pos[r[0]]] += r[8] - r[7]; k += 1
            add("n", Wf / 100, fb["mean_width"] * k, "intervals width by family (units)")

    # demand weekly family totals in units (+ per-series weekly index)
    dem = J(dd / "demand.json")
    if "by_family" in dem["weekly"]:
        for fam, vals in dem["weekly"]["by_family"].items():
            for w, v in zip(dem["weekly"]["week_start"], vals):
                if v is None:
                    continue
                c = np.zeros(n)
                for sid, s in ser.items():
                    if s["family"] != fam:
                        continue
                    h = s["history"]
                    wi = dict(zip(h.get("week_start", []), h.get("weekly_index", [])))
                    if wi.get(w) is not None:
                        c[pos[sid]] = wi[w] * (7 - _closures(w)) / 100
                add("n", c, v, "demand weekly family totals (units)")

    # scenario baselines in units
    sc = J(dd / "scenario.json")
    for f in sc["families"]:
        if "index" in f.get("baseline_unit", "") or "index" in f.get("baseline_source", "").lower():
            continue
        c = np.zeros(n)
        for sid, s in ser.items():
            if s["family"] == f["family"]:
                c[pos[sid]] = fwd[pos[sid]]
        add("n", c, f["baseline_14d"], "scenario family baselines (units)")

    # S&OP sentence: Review-band share of the 14-day forecast volume
    sop = J(dd / "sop.json")
    band = {r["series_id"]: r["trust_band"] for r in sop["rows"]}
    for t in sop["summary"]["discussion_points"]:
        m = re.search(r"Review-band series: they carry ([0-9.]+) % of the 14-day forecast volume", t)
        if m:
            share = float(m.group(1)) / 100
            c = np.array([(1.0 if band[s] == "Review" else 0.0) - share for s in sids]) * fwd
            add("h", c, name="S&OP Review share of volume")
    return sids, np.array(rows_h) if rows_h else np.zeros((0, n)), np.array(rows_n), np.array(y_n), src


def _closures(week_start):
    """Chain-wide closed days (25 Dec, 1 Jan) in a Monday week."""
    import datetime as dt
    d0 = dt.date.fromisoformat(week_start)
    return sum((d0 + dt.timedelta(days=i)).strftime("%m-%d") in ("12-25", "01-01") for i in range(7))


def solve(H, N, y):
    """Least squares on [homogeneous rows; non-homogeneous rows], every row scaled to unit norm."""
    A = np.vstack([H, N]) if len(H) else N
    b = np.concatenate([np.zeros(len(H)), y])
    s = np.linalg.norm(A, axis=1)
    s[s == 0] = 1
    A, b = A / s[:, None], b / s
    rank = int(np.linalg.matrix_rank(A, tol=1e-9 * np.abs(A).max()))
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x, rank, A.shape[0]


def true_bases(sids):
    import pandas as pd
    p = pd.read_parquet(PANEL)
    w = p[(p["date"] >= "2015-06-01") & (p["date"] <= "2017-02-06") & (p["closed"] == 0) & p["sales"].notna()]
    return w.groupby("series_id")["sales"].mean().reindex(sids).to_numpy()


def attack(dd=DD, out=OUT):
    sids, H, N, y, src = build_system(dd, out)
    x, rank, n_eq = solve(H, N, y)
    err = np.abs(x / true_bases(sids) - 1)
    return {"rank": rank, "n_equations": n_eq, "n_unknowns": len(sids), "median_rel_error": float(np.median(err)),
            "p10_rel_error": float(np.quantile(err, 0.1)), "sources": src}


@pytest.mark.skipif(not PANEL.exists() or not (DD / "series_index.json").exists(), reason="needs panel and docs/data")
def test_bases_not_identifiable():
    r = attack()
    print("\nreconstruction attack:", json.dumps(r, indent=1))
    assert r["rank"] < r["n_unknowns"], r
    assert r["median_rel_error"] >= 0.20, r
