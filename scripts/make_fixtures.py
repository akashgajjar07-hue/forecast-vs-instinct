"""Generate PLACEHOLDER fixtures for the dashboard shell: docs/data/_fixtures/.

Every number here is synthetic (seeded random numbers), NOT model output and NOT Favorita data. Every fixture
file carries a "FIXTURE" marker in its strings. The site loads these only with `?fixtures=1` and shows a
"PLACEHOLDER DATA" banner. Delete docs/data/_fixtures/ before deploying.

Usage: .venv/bin/python scripts/make_fixtures.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schemas import SCHEMAS, SERIES_SCHEMA  # noqa: E402

OUT = ROOT / "docs" / "data" / "_fixtures"
F = "FIXTURE"
rng = np.random.default_rng(7)

scope = json.loads((ROOT / "config" / "scope.json").read_text())
cal = json.loads((ROOT / "config" / "split_calendar.json").read_text())
FAMILIES = scope["families"]
STORES = scope["stores"]
MODELS = ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"]
START = date.fromisoformat(cal["model_start"])
LAST = date.fromisoformat(cal["last_actual"])
FWD0 = LAST + timedelta(days=1)
FWD1 = LAST + timedelta(days=14)
SERIES_PICK = [(12, "BREAD/BAKERY"), (12, "BEVERAGES"), (27, "PRODUCE"), (49, "GROCERY I"), (54, "CLEANING"),
               (36, "DAIRY")]
DOW = np.array([0.95, 0.9, 0.92, 0.97, 1.08, 1.2, 1.1])


def slug(store, fam):
    return f"{store}_{re.sub(r'[^a-z0-9]+', '-', fam.lower()).strip('-')}"


def d(x: date) -> str:
    return x.isoformat()


def r1(x):
    return None if x is None else round(float(x), 1)


def band(score, flags=()):
    if flags or score < 60:
        return "Review"
    return "Accept" if score >= 80 else "Monitor"


def synth_series(level, n, seed):
    g = np.random.default_rng(seed)
    days = [START + timedelta(days=i) for i in range(n)]
    dow = np.array([DOW[x.weekday()] for x in days])
    trend = 1 + 0.15 * np.arange(n) / n
    season = 1 + 0.08 * np.sin(np.arange(n) * 2 * np.pi / 365)
    y = level * dow * trend * season * g.lognormal(0, 0.12, n)
    out = [round(float(v), 1) for v in y]
    for i, x in enumerate(days):
        if (x.month, x.day) in [(12, 25), (1, 1)]:
            out[i] = None
    return days, out


LEVEL = {f: float(rng.uniform(100, 2000)) for f in FAMILIES}
SCORES = {}
for s in STORES:
    for f in FAMILIES:
        SCORES[(s["store_nbr"], f)] = float(np.clip(rng.normal(72, 14), 20, 98))


def trust_components(seed):
    g = np.random.default_rng(seed)
    return {k: round(float(g.uniform(40, 100)), 1) for k in ["accuracy", "interval", "stability", "agreement",
                                                              "completeness"]}


def make_series(store, fam, idx):
    n = (LAST - START).days + 1
    level = LEVEL[fam] * (0.6 + 0.1 * idx)
    days, sales = synth_series(level, n, 100 + idx)
    g = np.random.default_rng(200 + idx)
    champ = ["lightgbm", "holt_winters", "seasonal_naive", "lightgbm", "moving_average", "lightgbm"][idx]
    holdout = []
    for o in cal["stages"]["holdout"]:
        od = date.fromisoformat(o)
        rows = []
        for h in range(1, 15):
            t = od + timedelta(days=h)
            a = sales[(t - START).days]
            base = level * DOW[t.weekday()] * 1.13
            models = {m: round(float(base * g.lognormal(0, 0.08)), 1) for m in MODELS}
            c = models[champ]
            rows.append({"date": d(t), "h": h, "actual": a, "p10": round(c * 0.82, 1), "p90": round(c * 1.18, 1),
                         "models": models})
        holdout.append({"origin": o, "champion": champ, "rows": rows})
    frows = []
    for h in range(1, 15):
        t = LAST + timedelta(days=h)
        base = level * DOW[t.weekday()] * 1.15
        models = {m: round(float(base * g.lognormal(0, 0.08)), 1) for m in MODELS}
        c = models[champ]
        frows.append({"date": d(t), "h": h, "forecast": c, "p10": round(c * 0.8, 1), "p90": round(c * 1.2, 1),
                      "onpromotion": float(g.integers(0, 20)), "models": models})
    promo_total = sum(r["onpromotion"] for r in frows)
    fc_total = sum(r["forecast"] for r in frows)
    wrows = []
    for sc in [-100, -50, 0, 50, 100]:
        mult = 1 + 0.0009 * sc * (1 + 0.3 * idx)
        daily = [round(r["forecast"] * mult, 1) for r in frows]
        tot = sum(daily)
        wrows.append({"scale_pct": sc, "promo_total": round(promo_total * (1 + sc / 100), 1),
                      "forecast_total": round(tot, 1), "change_pct": round(100 * (tot / fc_total - 1), 2),
                      "daily": daily, "extrapolation": sc >= 100 or (sc >= 50 and idx % 2 == 0)})
    score = SCORES[(store, fam)]
    flags = [f"{F}: interval coverage below 50 % in last window"] if idx == 2 else []
    return {
        "series_id": f"{store}|{fam}", "store_nbr": store, "family": fam,
        "history": {"start": d(START), "sales": sales},
        "holdout": holdout,
        "forward": {"origin": d(LAST), "champion": champ, "rows": frows},
        "whatif": {"origin": d(LAST), "hist_max_promo": 40.0, "hist_p95_promo": 22.0, "rows": wrows},
        "trust": {"score": round(score, 1), "band": band(score, flags), "components": trust_components(idx),
                  "hard_flags": flags},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "series").mkdir(exist_ok=True)
    for p in (OUT / "series").glob("*.json"):
        p.unlink()
    files = {}
    store_attr = {s["store_nbr"]: s for s in STORES}

    files["meta.json"] = {
        "generated_at": f"{F} 2026-11-26T00:00:00Z",
        "citation": f"{F} citation placeholder: Corporación Favorita Grocery Sales Forecasting (Kaggle).",
        "caveats": [f"{F}: placeholder caveat. Observed sales, not unconstrained demand.",
                    f"{F}: placeholder caveat. Twelve stores, eight families."],
        "data_window": {"start": d(START), "last_actual": d(LAST), "forward_start": d(FWD0), "forward_end": d(FWD1)},
        "split_calendar": cal["stages"],
        "scope": {"n_stores": len(STORES), "n_families": len(FAMILIES), "n_series": len(STORES) * len(FAMILIES),
                  "families": FAMILIES,
                  "stores": [{k: s[k] for k in ["store_nbr", "city", "state", "type", "cluster"]} for s in STORES]},
        "hero": [{"label": f"{F} series tracked", "value": "96", "detail": f"{F} 12 stores x 8 families"},
                 {"label": f"{F} holdout WAPE", "value": "00.0 %", "detail": f"{F} placeholder"},
                 {"label": f"{F} series to review", "value": "00", "detail": f"{F} placeholder"}],
        "headline": f"{F}: headline sentence will come from the pipeline. Placeholder text only.",
    }

    # overview
    hist_days = [LAST - timedelta(days=i) for i in range(89, -1, -1)]
    tot_level = sum(LEVEL.values()) * len(STORES)
    hist = [{"date": d(x), "actual": round(float(tot_level * DOW[x.weekday()] * rng.lognormal(0, 0.05)), 1)}
            for x in hist_days]
    fc = []
    for h in range(1, 15):
        x = LAST + timedelta(days=h)
        v = float(tot_level * DOW[x.weekday()] * 1.02)
        fc.append({"date": d(x), "forecast": round(v, 1), "p10": round(v * 0.9, 1), "p90": round(v * 1.1, 1)})
    keys = sorted(SCORES, key=lambda k: SCORES[k])
    champs = {f: MODELS[i % 5] for i, f in enumerate(FAMILIES)}
    files["overview.json"] = {
        "kpis": {"forecast_14d_total": {"value": round(sum(r["forecast"] for r in fc), 1), "start": d(FWD0),
                                        "end": d(FWD1)},
                 "holdout_wape": {"value": 12.3, "model": f"{F} champion per family"},
                 "holdout_bias": {"value": -1.2, "model": f"{F} champion per family"},
                 "review_exceptions": {"value": sum(1 for k in SCORES if band(SCORES[k]) == "Review"), "of": 96},
                 "interval_coverage": {"value": 77.0, "target": 80.0}},
        "outlook": {"history": hist, "forecast": fc, "band_method": f"{F}: placeholder band method"},
        "top_exceptions": [{"series_id": f"{s}|{f}", "store_nbr": s, "family": f, "trust_score": round(SCORES[(s, f)], 1),
                            "band": band(SCORES[(s, f)]), "reason": f"{F}: placeholder reason"} for s, f in keys[:5]],
        "ml_wins": {"families_won": sum(1 for v in champs.values() if v == "lightgbm"), "n_families": 8,
                    "champions": champs},
        "findings": [f"{F}: finding one placeholder.", f"{F}: finding two placeholder.",
                     f"{F}: finding three placeholder."],
    }

    files["series_index.json"] = [
        {"series_id": f"{s}|{f}", "slug": slug(s, f), "store_nbr": s, "family": f, "city": store_attr[s]["city"],
         "type": store_attr[s]["type"], "cluster": store_attr[s]["cluster"]} for s, f in SERIES_PICK]

    # demand
    weeks = []
    w = START
    while w + timedelta(days=6) <= LAST:
        weeks.append(w)
        w += timedelta(days=7)
    by_fam = {}
    for f in FAMILIES:
        vals = [round(float(LEVEL[f] * 7 * 12 * (1 + 0.1 * np.sin(i / 8)) * rng.lognormal(0, 0.05)), 1)
                for i in range(len(weeks))]
        by_fam[f] = vals
    files["demand.json"] = {
        "weekly": {"week_start": [d(x) for x in weeks], "by_family": by_fam},
        "dow_profile": {f: [round(float(x * rng.uniform(0.97, 1.03)), 3) for x in DOW] for f in FAMILIES},
        "volatility": [{"store_nbr": s["store_nbr"], "family": f, "mean": round(LEVEL[f], 1),
                        "cv": round(float(rng.uniform(0.1, 0.6)), 3), "zero_share": 0.0}
                       for s in STORES for f in FAMILIES],
        "promo_bins": {f: [{"bin": b, "n_days": int(rng.integers(50, 400)),
                            "mean_sales": round(LEVEL[f] * (1 + 0.1 * i), 1),
                            "median_sales": round(LEVEL[f] * (0.97 + 0.1 * i), 1)}
                           for i, b in enumerate(["0", "1-5", "6-15", "16+"])] for f in FAMILIES},
        "promo_frequency": [{"store_nbr": s["store_nbr"], "family": f, "promo_day_share": round(float(rng.uniform(0.1, 0.9)), 3)}
                            for s in STORES for f in FAMILIES],
        "holiday_effect": [{"family": f, "holiday_mean": round(LEVEL[f] * 1.05, 1), "normal_mean": round(LEVEL[f], 1),
                            "change_pct": round(float(rng.uniform(-15, 20)), 1), "n_holiday_days": 30} for f in FAMILIES],
        "notes": [f"{F}: promo bins show association, not causation."],
    }

    # arena
    metrics = []
    slices = [("overall", "all")] + [("family", f) for f in FAMILIES] + [("store", str(s["store_nbr"])) for s in STORES] \
        + [("horizon", str(h)) for h in range(1, 15)] + [("horizon_bucket", b) for b in ["1-7", "8-14"]] \
        + [("promo", b) for b in ["promo", "no_promo"]] + [("holiday", b) for b in ["holiday", "normal"]] \
        + [("week_bucket", b) for b in ["week1", "week2"]]
    skill = {"naive": 1.5, "seasonal_naive": 1.15, "moving_average": 1.25, "holt_winters": 1.05, "lightgbm": 1.0}
    for stage in ["selection", "holdout"]:
        for m in MODELS:
            for v in (["A", "B"] if m == "lightgbm" else ["A"]):
                for st, sv in slices:
                    wape = 12 * skill[m] * (1.04 if v == "B" else 1) * float(rng.uniform(0.85, 1.2))
                    if st == "horizon":
                        wape *= 1 + 0.02 * int(sv)
                    metrics.append({"stage": stage, "model": m, "variant": v, "slice_type": st, "slice_value": sv,
                                    "wape": round(wape, 2), "bias": round(float(rng.normal(0, 3)), 2),
                                    "mae": round(wape * 3, 2), "rmse": round(wape * 4.2, 2), "n": 1000})
    files["arena.json"] = {
        "metrics": metrics,
        "champions": {f: {"model": champs[f], "selection_wape": round(float(rng.uniform(8, 20)), 2),
                          "selection_bias": round(float(rng.normal(0, 2)), 2)} for f in FAMILIES},
        "variant_comparison": [{"stage": st, "family": f, "wape_A": round(float(rng.uniform(8, 18)), 2),
                                "wape_B": round(float(rng.uniform(9, 19)), 2)}
                               for st in ["selection", "holdout"] for f in ["all"] + FAMILIES],
        "notes": [f"{F}: placeholder arena note."],
    }

    files["intervals.json"] = {
        "method": f"{F}: split conformal placeholder", "level": 80.0,
        "quantiles": [{"family": f, "bucket": b, "q_lo": -0.2, "q_hi": 0.2} for f in FAMILIES for b in ["1-7", "8-14"]],
        "calibration_coverage": 80.0,
        "holdout": {"coverage": 77.0, "mean_width": 35.0,
                    "by_family": [{"family": f, "coverage": round(float(rng.uniform(65, 90)), 1),
                                   "mean_width": round(float(rng.uniform(25, 45)), 1)} for f in FAMILIES],
                    "by_horizon": [{"horizon": h, "coverage": round(80 - h * 0.4, 1), "mean_width": round(30 + h, 1)}
                                   for h in range(1, 15)]},
    }

    latest = []
    for (s, f), sc in SCORES.items():
        flags = [f"{F}: placeholder hard flag"] if (s + len(f)) % 17 == 0 else []
        fc14 = round(LEVEL[f] * 14 * 1.1, 1)
        latest.append({"series_id": f"{s}|{f}", "store_nbr": s, "family": f, "score": round(sc, 1),
                       "band": band(sc, flags), "components": trust_components(s * 7 + len(f)), "hard_flags": flags,
                       "forecast_14d": fc14, "p10_14d": round(fc14 * 0.85, 1), "p90_14d": round(fc14 * 1.15, 1),
                       "priority": 0.0})
    tot = sum(r["forecast_14d"] for r in latest)
    for r in latest:
        r["priority"] = round((100 - r["score"]) * r["forecast_14d"] / tot, 4)
    files["trust.json"] = {
        "weights": {"accuracy": 0.35, "interval": 0.2, "stability": 0.2, "agreement": 0.15, "completeness": 0.1},
        "bands": [{"min": 80, "band": "Accept"}, {"min": 60, "band": "Monitor"}, {"min": 0, "band": "Review"}],
        "latest_origin": d(LAST), "latest": latest,
        "validation": {"n_windows": 576, "spearman": 0.21,
                       "wape_by_band": [{"band": "Accept", "n": 300, "wape": 9.1}, {"band": "Monitor", "n": 200, "wape": 12.4},
                                        {"band": "Review", "n": 76, "wape": 18.8}],
                       "error_ratio_review_vs_accept": 2.07, "review_share_of_abs_error": 22.0,
                       "review_share_of_windows": 13.0, "volatility_only_spearman": 0.18,
                       "beats_volatility_baseline": True,
                       "verdict": f"{F}: placeholder verdict. The real sentence states honestly whether the index worked."},
        "calibration_choice": {"candidates": [{"name": f"{F} default weights", "spearman": 0.2}], "chosen": f"{F} default weights"},
    }

    files["scenario.json"] = {
        "assumption_label": "Editable demonstration assumptions, not measured UAE uplift",
        "families": [{"family": f, "baseline_14d": round(LEVEL[f] * 14 * 12, 1),
                      "baseline_source": f"{F}: champion forward forecast placeholder"} for f in FAMILIES],
        "events": [{"id": "ramadan", "label": "Ramadan", "low": 5, "base": 15, "high": 30},
                   {"id": "eid_al_fitr", "label": "Eid al-Fitr", "low": 10, "base": 25, "high": 45},
                   {"id": "eid_al_adha", "label": "Eid al-Adha", "low": 8, "base": 20, "high": 40},
                   {"id": "summer", "label": "Summer", "low": -20, "base": -10, "high": 0},
                   {"id": "promo", "label": "Promo campaign", "low": 5, "base": 12, "high": 25}],
        "ecuador_analogue": [{"family": f, "change_pct": round(float(rng.uniform(-10, 15)), 1), "n_holidays": 20,
                              "window": f"{F} 2015-06-01 to 2017-08-15"} for f in FAMILIES],
    }

    files["sop.json"] = {
        "origin": d(LAST), "horizon_end": d(FWD1),
        "rows": [{"series_id": r["series_id"], "store_nbr": r["store_nbr"], "family": r["family"],
                  "stat_forecast": r["forecast_14d"], "p10": r["p10_14d"], "p90": r["p90_14d"],
                  "trust_score": r["score"], "trust_band": r["band"]} for r in latest],
        "reason_codes": ["Upcoming promotion", "Market intelligence", "Supply disruption", "Exceptional event",
                         "Suspected forecast error"],
        "summary": {"top_risks": [f"{F}: risk placeholder one.", f"{F}: risk placeholder two."],
                    "persistent_bias": [f"{F}: bias placeholder."],
                    "review_queue": [f"{F}: review queue placeholder."],
                    "promo_opportunities": [f"{F}: promo placeholder."],
                    "discussion_points": [f"{F}: discussion point one.", f"{F}: discussion point two.",
                                          f"{F}: discussion point three."]},
        "fva": {"label": "Simulated, rule-based override, not human-planner performance",
                "rule": f"{F}: placeholder rule", "stage": "holdout", "champion_wape": 12.3, "simulated_wape": 12.9,
                "fva_pp": -0.6},
    }

    marker = f"{F}: synthetic placeholder data, not model output. Delete docs/data/_fixtures before deploy."
    for name, obj in files.items():
        if isinstance(obj, dict):
            obj["_fixture"] = marker
        else:
            for item in obj:
                item["_fixture"] = marker
        jsonschema.validate(obj, SCHEMAS[name])
        (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
    for i, (s, f) in enumerate(SERIES_PICK):
        obj = make_series(s, f, i)
        obj["_fixture"] = marker
        jsonschema.validate(obj, SERIES_SCHEMA)
        (OUT / "series" / f"{slug(s, f)}.json").write_text(json.dumps(obj, separators=(",", ":")))
    (OUT / "README_FIXTURE.txt").write_text(
        "FIXTURE / PLACEHOLDER DATA. Synthetic numbers generated by scripts/make_fixtures.py for building the "
        "dashboard shell. Not model output, not Favorita data. Delete this folder before deploying.\n")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {len(files)} files + {len(SERIES_PICK)} series to {OUT} ({size / 1e6:.2f} MB), all schema-valid")


if __name__ == "__main__":
    main()
