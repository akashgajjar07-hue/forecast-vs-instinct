"""Export pipeline outputs to the dashboard JSON files (docs/data/*.json + docs/data/series/<slug>.json).

Every number comes from files in outputs/ (backtests, holdout artefacts, frozen JSON) or from the processed
panel. Nothing here trains a model except the single LightGBM refit at the forward origin that is needed for
the promo what-if grid; that refit uses the frozen config and must reproduce the saved forward forecasts
(asserted to 1e-6). Contract: docs/data_contract.md, src/schemas.py. Definitions: docs/agent_notes/export.md.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from . import config as C
from .freeze import verify
from . import metrics as M
from .backtest import evaluable, masked_view
from .runner import load_backtests
from .schemas import SCHEMAS, SERIES_SCHEMA
from .trust import COMPONENTS, describe_flags

OUT = C.OUTPUTS
FORWARD_ORIGIN = pd.Timestamp(C.LAST_ACTUAL)
FORWARD_START = FORWARD_ORIGIN + pd.Timedelta(days=1)
FORWARD_END = FORWARD_ORIGIN + pd.Timedelta(days=C.HORIZON)
HISTORY_START = pd.Timestamp("2017-01-01")
OUTLOOK_DAYS = 56
WHATIF_SCALES = [-100, -50, 0, 50, 100]
BIAS_RULE_PP = 2.0
MAX_BYTES = 3 * 1024 * 1024
MODEL_LABEL = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "moving_average": "Moving average",
               "holt_winters": "Holt-Winters", "lightgbm": "LightGBM"}
COMP_LABEL = {"accuracy": "Accuracy", "interval": "Interval", "stability": "Stability",
              "agreement": "Agreement", "completeness": "Completeness"}
CITATION = "Alexis Cook, DanB, inversion, Ryan Holbrook (2021). Store Sales – Time Series Forecasting. Kaggle."
REASON_CODES = ["Upcoming promotion", "Market intelligence", "Supply disruption", "Exceptional event",
                "Suspected forecast error"]
# Demonstration values only (same defaults as the dashboard fixtures). Not estimated from any data.
SCENARIO_EVENTS = [
    {"id": "ramadan", "label": "Ramadan", "low": 5, "base": 15, "high": 30},
    {"id": "eid_al_fitr", "label": "Eid al-Fitr", "low": 10, "base": 25, "high": 45},
    {"id": "eid_al_adha", "label": "Eid al-Adha", "low": 8, "base": 20, "high": 40},
    {"id": "summer", "label": "Summer", "low": -20, "base": -10, "high": 0},
    {"id": "promo", "label": "Promo campaign", "low": 5, "base": 12, "high": 25},
    {"id": "custom", "label": "Custom", "low": 0, "base": 0, "high": 0},
]


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def rnd(x, nd: int = 2):
    """Round for publishing; NaN/None -> None."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def slug(store_nbr: int, family: str) -> str:
    return f"{int(store_nbr)}_{re.sub(r'[^a-z0-9]+', '-', family.lower()).strip('-')}"


def fmt_d(d) -> str:
    """'2017-06-28' -> '28 Jun 2017'."""
    return pd.Timestamp(d).strftime("%-d %b %Y")


def iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def fam_label(f: str) -> str:
    return re.sub(r"(^|[\s/])(\S)", lambda m: m.group(1) + m.group(2).upper(), f.lower())


def fmt_int(x) -> str:
    return f"{int(math.floor(float(x) + 0.5)):,}"


def pct(x, nd: int = 1) -> str:
    return f"{x:.{nd}f} %"


def metric_dict(a, f) -> dict:
    r = M.all_metrics(a, f)
    return {"wape": rnd(r.wape, 4), "bias": rnd(r.bias, 4), "mae": rnd(r.mae, 4), "rmse": rnd(r.rmse, 4), "n": int(r.n)}


def dump(obj, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(txt, encoding="utf-8")
    return len(txt.encode("utf-8"))


def read_json(p: Path):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------
def load_inputs() -> dict:
    bt = load_backtests()
    bt["origin"] = pd.to_datetime(bt["origin"])
    bt["target_date"] = pd.to_datetime(bt["target_date"])
    ci = pd.read_parquet(OUT / "holdout" / "champion_intervals.parquet")
    ci["origin"] = pd.to_datetime(ci["origin"])
    ct = pd.read_parquet(OUT / "holdout" / "chain_total.parquet")
    ct["origin"] = pd.to_datetime(ct["origin"])
    ts = pd.read_parquet(OUT / "holdout" / "trust_scores.parquet")
    ts["origin"] = pd.to_datetime(ts["origin"])
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    return {
        "bt": bt, "ci": ci, "ct": ct, "ts": ts, "panel": panel,
        "selection": read_json(OUT / "selection.json"),
        "intervals": read_json(OUT / "intervals.json"),
        "trust_weights": read_json(OUT / "trust_weights.json"),
        "trust_cal": read_json(OUT / "trust_calibration.json"),
        "holdout": read_json(OUT / "holdout_results.json"),
        "demand": read_json(OUT / "eda" / "demand.json"),
        "analogue": read_json(OUT / "eda" / "ecuador_analogue.json"),
        "scope": read_json(C.CONFIG / "scope.json"),
        "cal": read_json(C.CONFIG / "split_calendar.json"),
        "window_intervals": verify("window_intervals"),
        "window_band": read_json(OUT / "window_band_results.json"),
    }


def champions(inp) -> dict:
    return {f: c["model"] for f, c in inp["selection"]["champions"].items()}


def champion_bt(inp, stage: str) -> pd.DataFrame:
    """Champion-system rows (variant A) of one stage, all rows (not only evaluable)."""
    ch = champions(inp)
    b = inp["bt"]
    b = b[(b["stage"] == stage) & (b["variant"] == "A")]
    return b[b["model"] == b["family"].map(ch)]


# ------------------------------------------------------------------------------------------------
# arena
# ------------------------------------------------------------------------------------------------
def arena_rows(bt: pd.DataFrame, stage: str) -> list[dict]:
    rows = []
    ev = evaluable(bt[bt["stage"] == stage]).copy()
    ev["horizon_bucket"] = np.where(ev["horizon"] <= 7, "1-7", "8-14")
    ev["promo"] = np.where(ev["onpromotion"] > 0, "promo", "no_promo")
    ev["holiday"] = np.where(ev["is_holiday"] == 1, "holiday", "normal")
    for (model, variant), g in ev.groupby(["model", "variant"], sort=True):
        base = {"stage": stage, "model": model, "variant": variant}
        rows.append({**base, "slice_type": "overall", "slice_value": "all", **metric_dict(g["actual"], g["forecast"])})
        for st, col in [("family", "family"), ("store", "store_nbr"), ("horizon", "horizon"),
                        ("horizon_bucket", "horizon_bucket"), ("promo", "promo"), ("holiday", "holiday")]:
            for v, gg in g.groupby(col, sort=True):
                rows.append({**base, "slice_type": st, "slice_value": str(v), **metric_dict(gg["actual"], gg["forecast"])})
        wk = M.bucket_totals(g, ["series_id", "origin"])
        for v, gg in wk.groupby("bucket", sort=True):
            rows.append({**base, "slice_type": "week_bucket", "slice_value": v, **metric_dict(gg["actual"], gg["forecast"])})
    return rows


def build_arena(inp) -> dict:
    bt = inp["bt"]
    metrics = []
    for stage in ["selection", "calibration", "holdout"]:
        metrics += arena_rows(bt, stage)
    # identical evaluation population check
    for stage in ["selection", "calibration", "holdout"]:
        ns = {(r["model"], r["variant"]): r["n"] for r in metrics if r["stage"] == stage and r["slice_type"] == "overall"}
        assert len(set(ns.values())) == 1, f"evaluation population differs on {stage}: {ns}"
    champs = {f: {"model": c["model"], "selection_wape": rnd(c["selection_wape"], 4),
                  "selection_bias": rnd(c["selection_bias"], 4)} for f, c in inp["selection"]["champions"].items()}
    vc = []
    lg = bt[bt["model"] == "lightgbm"]
    for stage in ["selection", "holdout"]:
        ev = evaluable(lg[lg["stage"] == stage])
        def w(d, v):
            dd = d[d["variant"] == v]
            return rnd(M.wape(dd["actual"], dd["forecast"]), 4)
        vc.append({"stage": stage, "family": "all", "wape_A": w(ev, "A"), "wape_B": w(ev, "B")})
        for fam in C.FAMILIES:
            e = ev[ev["family"] == fam]
            vc.append({"stage": stage, "family": fam, "wape_A": w(e, "A"), "wape_B": w(e, "B")})
    n_hold = len(inp["cal"]["stages"]["holdout"])
    notes = [
        "Champion rule: " + inp["selection"]["rule"],
        "Actuals are observed sales (units; some families such as produce are fractional), not unconstrained demand: "
        "stock-outs are not visible in the data.",
        f"The holdout ({n_hold} weekly origins, targets {inp['cal']['target_windows']['holdout'][0]} to "
        f"{inp['cal']['target_windows']['holdout'][1]}) was scored once, after champions, interval quantiles and "
        "Trust Index weights were frozen. Nothing was tuned on it.",
        "Every model is scored on the identical population: open target days with an observed actual.",
        "WAPE and bias are ratios of sums over the slice. Week buckets sum the 7 daily forecasts and actuals per "
        "series and origin (days 1–7 and 8–14) before computing the metrics.",
        "Variant A uses the known promotion plan (count of promoted items on the target date); variant B has no "
        "future promotion information. Baselines use no promotion information and appear once (variant A).",
    ]
    return {"metrics": metrics, "champions": champs, "variant_comparison": vc, "notes": notes}


# ------------------------------------------------------------------------------------------------
# intervals
# ------------------------------------------------------------------------------------------------
def holdout_interval_rows(inp) -> pd.DataFrame:
    ci = inp["ci"]
    return evaluable(ci[ci["stage"] == "holdout"])


def build_intervals(inp) -> dict:
    iv = inp["intervals"]
    h = holdout_interval_rows(inp)
    cov = lambda d: M.coverage(d["actual"], d["p10"], d["p90"])
    wid = lambda d: M.mean_width(d["p10"], d["p90"])
    return {
        "method": iv["method"] + " Coverage is an empirical target measured on past windows, not a guarantee.",
        "level": float(iv["level"]) * 100 if iv["level"] <= 1 else float(iv["level"]),
        "quantiles": [{"family": q["family"], "bucket": q["bucket"], "q_lo": q["q_lo"], "q_hi": q["q_hi"], "n": q["n"]}
                      for q in iv["quantiles"]],
        "calibration_coverage": float(iv["calibration_coverage_leave_one_origin_out"]),
        "calibration_coverage_note": "Leave-one-origin-out estimate on the 8 calibration origins "
                                     f"(in-sample: {iv['calibration_coverage_in_sample']:.1f} %).",
        "holdout": {
            "coverage": float(cov(h)), "mean_width": float(wid(h)),
            "by_family": [{"family": f, "coverage": rnd(cov(g), 4), "mean_width": rnd(wid(g), 4)}
                          for f, g in h.groupby("family")],
            "by_horizon": [{"horizon": int(k), "coverage": rnd(cov(g), 4), "mean_width": rnd(wid(g), 4)}
                           for k, g in h.groupby("horizon")],
        },
        "width_unit": "units per day (mean of P90 − P10 over open holdout days)",
        "window_band": build_window_band(inp),
    }


def champ_bias(inp, stage) -> float:
    ch = inp["bt"][(inp["bt"]["stage"] == stage) & (inp["bt"]["variant"] == "A")]
    m = {f: c["model"] for f, c in inp["selection"]["champions"].items()}
    ch = evaluable(ch[ch["model"] == ch["family"].map(m)])
    return float(M.bias(ch["actual"], ch["forecast"]))


def build_window_band(inp) -> dict:
    wq, wb = inp["window_intervals"], inp["window_band"]
    return {
        "method": wq["method"],
        "level": float(wq["level"]) * 100 if wq["level"] <= 1 else float(wq["level"]),
        "fitted_on": wq["fitted_on"],
        "calibration_loo_coverage": float(wq["calibration_loo_coverage"]),
        "holdout_coverage": float(wb["holdout_coverage"]),
        "holdout_mean_width_pct": float(wb["holdout_mean_width_pct"]),
        "holdout_below_p10_pct": float(wb["holdout_below_p10_pct"]),
        "holdout_above_p90_pct": float(wb["holdout_above_p90_pct"]),
        "n_holdout_windows": int(wb["n_holdout_windows"]),
        "by_family": wb["by_family"],
        "replaced": {"what": "sum of the 14 daily P10 / P90 bounds",
                     "holdout_coverage": float(wb["replaced_sum_of_daily_bounds"]["holdout_coverage"]),
                     "holdout_mean_width_pct": float(wb["replaced_sum_of_daily_bounds"]["holdout_mean_width_pct"])},
        "explanation": (f"Below target: {float(wb['holdout_coverage']):.1f} % of holdout 14-day totals fell inside the band "
                        f"vs the 80 % target ({float(wb['holdout_below_p10_pct']):.1f} % below P10, "
                        f"{float(wb['holdout_above_p90_pct']):.1f} % above P90). On the calibration windows the champion "
                        f"under-forecast (bias {champ_bias(inp, 'calibration'):.1f} %), so the band was centred above the "
                        f"forecast; on the holdout the bias was close to zero ({champ_bias(inp, 'holdout'):.1f} %), so totals "
                        "fell below P10 more often. The band is kept exactly as frozen; rolling recalibration is future work."),
        "future_work": "Rolling recalibration: refit the 14-day quantiles each week on the most recent completed windows.",
        "disclosure": "Designed after the holdout had been opened (the independent review showed the summed daily "
                      "bounds covered ~94 %). Quantiles use calibration windows only and were frozen with a hash "
                      "before holdout windows were scored; nothing was tuned on holdout results.",
    }


# ------------------------------------------------------------------------------------------------
# trust
# ------------------------------------------------------------------------------------------------
def forward_champion(inp) -> pd.DataFrame:
    ci = inp["ci"]
    f = ci[ci["stage"] == "forward"].copy()
    assert f["origin"].nunique() == 1 and f["origin"].iloc[0] == FORWARD_ORIGIN
    return f


def latest_trust(inp) -> pd.DataFrame:
    ts = inp["ts"]
    t = ts[ts["origin"] == FORWARD_ORIGIN].copy()
    assert len(t) == 96
    f = forward_champion(inp).groupby("series_id")[["forecast"]].sum()
    t = t.join(f, on="series_id")
    assert np.allclose(t["forecast"], t["forecast_14d"], rtol=0, atol=1e-6), "trust forecast_14d != champion forward sum"
    # 14-day P10/P90 = separately calibrated 14-day-total band (scripts/07b_window_band.py), NOT summed daily bounds
    wb = pd.DataFrame(inp["window_band"]["forward"]).set_index("series_id")
    assert np.allclose(wb.loc[t["series_id"], "forecast_14d"].to_numpy(), t["forecast_14d"].to_numpy(), rtol=0, atol=1e-6)
    t["p10"] = wb.loc[t["series_id"], "p10_14d"].to_numpy()
    t["p90"] = wb.loc[t["series_id"], "p90_14d"].to_numpy()
    return t.sort_values("priority", ascending=False).reset_index(drop=True)


def weakest_component(row, weights) -> tuple[str, float]:
    comps = [c for c in COMPONENTS if weights.get(c, 0) > 0]
    c = min(comps, key=lambda k: row[k])
    return c, float(row[c])


def reason_for(row, weights) -> str:
    flags = list(row["hard_flags"])
    if flags:
        return "; ".join(describe_flags(flags))
    c, v = weakest_component(row, weights)
    return f"Weakest component: {COMP_LABEL[c]} {v:.0f} / 100"


def build_trust(inp) -> dict:
    tw = inp["trust_weights"]
    t = latest_trust(inp)
    tv = inp["holdout"]["trust_validation"]
    cands = {c["name"]: c for c in inp["trust_cal"]["candidates"]}
    chosen = tw["chosen"]
    names = ["default", "equal", "volatility_only", chosen]
    latest = []
    for _, r in t.iterrows():
        latest.append({
            "series_id": r["series_id"], "store_nbr": int(r["store_nbr"]), "family": r["family"],
            "score": rnd(r["score"], 2), "band": r["band"],
            "components": {c: rnd(r[c], 2) for c in COMPONENTS},
            "hard_flags": describe_flags(list(r["hard_flags"])),
            "forecast_14d": rnd(r["forecast_14d"], 2), "p10_14d": rnd(r["p10"], 2), "p90_14d": rnd(r["p90"], 2),
            "priority": rnd(r["priority"], 4),
        })
    validation = {k: tv[k] for k in ["n_windows", "spearman", "wape_by_band", "error_ratio_review_vs_accept",
                                      "review_share_of_abs_error", "review_share_of_windows",
                                      "volatility_only_spearman", "beats_volatility_baseline", "verdict"]}
    validation["band_counts"] = inp["holdout"]["trust_band_counts_holdout"]
    return {
        "weights": {c: float(tw["weights"][c]) for c in COMPONENTS},
        "bands": [{"min": float(b["min"]), "band": b["band"]} for b in tw["bands"]],
        "latest_origin": iso(FORWARD_ORIGIN),
        "latest": latest,
        "validation": validation,
        "calibration_choice": {
            "candidates": [{"name": n, "spearman": cands[n]["spearman"], "weights": cands[n]["weights"]} for n in names],
            "chosen": chosen,
            "rule": tw["choice_rule"],
            "reason": tw["choice_reason"],
            "grid_size": inp["trust_cal"]["grid_size"],
        },
        "notes": [
            "The Trust score is a 0–100 ranking index of how far to rely on a forecast, not a probability.",
            "p10_14d / p90_14d: a separately calibrated P10–P90 band for each series' 14-day total (relative-residual "
            "conformal on 14-day totals, fitted on calibration windows only; see intervals.json → window_band). "
            "It replaced the sum of daily bounds after the independent review.",
            "priority = (100 − score) × the series' share of the total 14-day forecast at this origin.",
        ],
    }


# ------------------------------------------------------------------------------------------------
# overview + meta
# ------------------------------------------------------------------------------------------------
def holdout_family_wapes(inp) -> pd.DataFrame:
    rows = inp["holdout"]["by_family"]
    return pd.DataFrame(rows).set_index("family")[C.MODELS]


def ml_wins(inp) -> dict:
    fw = holdout_family_wapes(inp)
    won = int((fw.idxmin(axis=1) == "lightgbm").sum())
    return {"families_won": won, "n_families": int(len(fw)), "champions": champions(inp),
            "definition": "families where LightGBM (variant A) had the lowest HOLDOUT WAPE of the five models; "
                          "champions were chosen on selection backtests only"}


def best_baseline(inp) -> dict:
    lb = [r for r in inp["holdout"]["leaderboard"] if r["model"] != "lightgbm"]
    return min(lb, key=lambda r: r["wape"])


def lgbm_variant(inp, v) -> dict:
    return next(r for r in inp["holdout"]["leaderboard"] if r["model"] == "lightgbm" and r["variant"] == v)


def chain_outlook(inp) -> dict:
    p = inp["panel"]
    lo = FORWARD_ORIGIN - pd.Timedelta(days=OUTLOOK_DAYS - 1)
    w = p[(p["date"] >= lo) & (p["date"] <= FORWARD_ORIGIN)]
    g = w.groupby("date")
    tot = g["sales"].sum(min_count=1)
    n_open = g.apply(lambda d: int(((d["closed"] == 0) & d["sales"].notna()).sum()))
    history = [{"date": iso(d), "actual": (rnd(tot[d], 1) if n_open[d] == 96 else None)} for d in tot.index]
    ct = inp["ct"]
    f = ct[ct["stage"] == "forward"].sort_values("horizon")
    forecast = [{"date": iso(r.target_date), "forecast": rnd(r.forecast, 1), "p10": rnd(r.p10, 1), "p90": rnd(r.p90, 1)}
                for r in f.itertuples()]
    tq = inp["intervals"]["total_quantiles"]
    method = ("Own split-conformal band for the chain total: relative residuals (actual − forecast) / forecast of the "
              "96-series daily total on the 8 calibration origins, 10th/90th percentiles per horizon bucket "
              f"(days 1–7: {tq[0]['q_lo']*100:+.1f} % / {tq[0]['q_hi']*100:+.1f} %; days 8–14: "
              f"{tq[1]['q_lo']*100:+.1f} % / {tq[1]['q_hi']*100:+.1f} %). Not the sum of series bands. "
              "An empirical 80 % target, not a guarantee.")
    return {"history": history, "forecast": forecast, "band_method": method}


def findings(inp, ivals) -> list[str]:
    hr = inp["holdout"]
    cs = hr["champion_system"]
    bb = best_baseline(inp)
    a, b = lgbm_variant(inp, "A"), lgbm_variant(inp, "B")
    mw = ml_wins(inp)
    fw = holdout_family_wapes(inp)
    tv = hr["trust_validation"]
    n_hold = len(inp["cal"]["stages"]["holdout"])
    tw = inp["cal"]["target_windows"]["holdout"]
    rel = (bb["wape"] - cs["wape"]) / bb["wape"] * 100
    out = [
        f"On the final holdout ({n_hold} weekly origins, targets {tw[0]} to {tw[1]}, scored once) the champion system "
        f"had a WAPE of {cs['wape']:.1f} % against {bb['wape']:.1f} % for the best baseline "
        f"({MODEL_LABEL[bb['model']]}), {rel:.0f} % less absolute error.",
        f"LightGBM had the lowest holdout WAPE of the five models in {mw['families_won']} of {mw['n_families']} families; "
        f"it had been chosen as champion on selection backtests in "
        f"{sum(m == 'lightgbm' for m in mw['champions'].values())} of {mw['n_families']}.",
    ]
    gap = b["wape"] - a["wape"]
    size = "only a small" if abs(gap) < 0.5 else "a clear"
    out.append(f"Knowing the promotion plan made {size} difference: LightGBM holdout WAPE was {a['wape']:.2f} % with the "
               f"known plan (A) and {b['wape']:.2f} % without future promotion information (B), a gap of {gap:+.2f} pp.")
    cov = ivals["holdout"]["coverage"]
    out.append(f"The P10–P90 intervals contained {cov:.1f} % of holdout actuals against an empirical 80 % target "
               f"(calibration leave-one-origin-out estimate {ivals['calibration_coverage']:.1f} %).")
    wb = {r["band"]: r["wape"] for r in tv["wape_by_band"]}
    s = (f"Trust Index on the holdout: Spearman {tv['spearman']:.2f} between score and realised window WAPE "
         f"(volatility-only baseline {tv['volatility_only_spearman']:.2f}); Review windows had "
         f"{tv['error_ratio_review_vs_accept']:.2f}× the WAPE of Accept windows")
    if abs(wb["Review"] - wb["Monitor"]) < 1.0:
        s += (f", but Monitor ({wb['Monitor']:.1f} %) and Review ({wb['Review']:.1f} %) were barely separated, "
              "so the index sorts Accept from the rest better than it splits Monitor from Review.")
    else:
        s += f" (Monitor {wb['Monitor']:.1f} %, Review {wb['Review']:.1f} %)."
    out.append(s)
    cw = fw["lightgbm"]
    hard, easy = cw.idxmax(), cw.idxmin()
    out.append(f"Hardest family to forecast: {fam_label(hard)}, champion holdout WAPE {cw[hard]:.1f} %; easiest: "
               f"{fam_label(easy)} at {cw[easy]:.1f} %.")
    return out


def build_overview(inp, ivals, trust) -> dict:
    hr = inp["holdout"]
    cs = hr["champion_system"]
    fwd = forward_champion(inp)
    lt = latest_trust(inp)
    tw = inp["trust_weights"]["weights"]
    n_lgb = sum(m == "lightgbm" for m in champions(inp).values())
    model_txt = (f"champion per family (LightGBM in {n_lgb} of {len(C.FAMILIES)}), "
                 f"{len(inp['cal']['stages']['holdout'])} holdout origins")
    top = []
    for _, r in lt[lt["band"].isin(["Review", "Monitor"])].head(8).iterrows():
        top.append({"series_id": r["series_id"], "store_nbr": int(r["store_nbr"]), "family": r["family"],
                    "trust_score": rnd(r["score"], 2), "band": r["band"], "reason": reason_for(r, tw),
                    "priority": rnd(r["priority"], 4)})
    return {
        "kpis": {
            "forecast_14d_total": {"value": float(fwd["forecast"].sum()), "start": iso(FORWARD_START), "end": iso(FORWARD_END)},
            "holdout_wape": {"value": float(cs["wape"]), "model": model_txt},
            "holdout_bias": {"value": float(cs["bias"]), "model": model_txt},
            "review_exceptions": {"value": int((lt["band"] == "Review").sum()), "of": int(len(lt))},
            "interval_coverage": {"value": float(ivals["holdout"]["coverage"]), "target": 80.0},
        },
        "outlook": chain_outlook(inp),
        "top_exceptions": top,
        "ml_wins": ml_wins(inp),
        "findings": findings(inp, ivals),
    }


def build_meta(inp, ivals) -> dict:
    hr = inp["holdout"]
    cs = hr["champion_system"]
    bb = best_baseline(inp)
    tv = hr["trust_validation"]
    cal = inp["cal"]
    sc = inp["scope"]
    n_hold = len(cal["stages"]["holdout"])
    cov = ivals["holdout"]["coverage"]
    hero = [
        {"label": "Holdout WAPE, champion system", "value": pct(cs["wape"]),
         "detail": f"vs {pct(bb['wape'])} for the best baseline ({MODEL_LABEL[bb['model']]}); {n_hold} holdout origins, scored once"},
        {"label": "Review vs Accept error", "value": f"{tv['error_ratio_review_vs_accept']:.2f}×",
         "detail": "holdout WAPE of Review-band windows relative to Accept-band windows (Trust score is a 0–100 index, not a probability)"},
        {"label": "Interval coverage", "value": pct(cov),
         "detail": "of holdout actuals inside P10–P90; empirical target 80 %, not a guarantee"},
    ]
    headline = (f"Across {n_hold} weekly forecast origins (targets {fmt_d(inp['cal']['target_windows']['holdout'][0])} – "
                f"{fmt_d(inp['cal']['target_windows']['holdout'][1])}), each scored on days the model had not seen, "
                f"LightGBM cut forecast error to {pct(cs['wape'])} WAPE from "
                f"{pct(bb['wape'])} for the best baseline, and Review-band forecasts went on to show "
                f"{tv['error_ratio_review_vs_accept']:.2f}× the error of Accept-band ones.")
    caveats = [
        "Real historical data from Corporación Favorita (Ecuador), Kaggle. Not UAE data.",
        "Sales are observed sales, not unconstrained demand: stock-outs and lost sales are not visible.",
        f"Scope: {len(sc['stores'])} stores × {len(sc['families'])} families = {len(sc['stores']) * len(sc['families'])} daily series, "
        "stores chosen with data before the first selection origin.",
        "Promotion = count of promoted items in the family on a day. Promotion effects shown are model-predicted "
        "associations, not causal uplift.",
        "Prediction intervals are empirical 80 % targets from calibration residuals, not guarantees.",
        "The Trust score is a 0–100 ranking index, not a probability that the forecast is right.",
        "UAE Scenario Lab values are editable demonstration assumptions, not measured UAE uplift.",
        "Forecast value added (FVA) on the S&OP page is a simulated, rule-based override, not human-planner performance.",
        f"The holdout was evaluated once; the forward window {iso(FORWARD_START)} to {iso(FORWARD_END)} has no actuals.",
    ]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "citation": CITATION,
        "caveats": caveats,
        "data_window": {"start": cal["model_start"], "last_actual": cal["last_actual"],
                        "forward_start": iso(FORWARD_START), "forward_end": iso(FORWARD_END)},
        "split_calendar": {k: cal["stages"][k] for k in ["selection", "calibration", "holdout"]},
        "scope": {"n_stores": len(sc["stores"]), "n_families": len(sc["families"]),
                  "n_series": len(sc["stores"]) * len(sc["families"]), "families": sc["families"],
                  "stores": [{k: s[k] for k in ["store_nbr", "city", "state", "type", "cluster"]} for s in sc["stores"]]},
        "hero": hero,
        "headline": headline,
    }


def build_series_index(inp) -> list[dict]:
    out = []
    for s in inp["scope"]["stores"]:
        for f in inp["scope"]["families"]:
            out.append({"series_id": f"{s['store_nbr']}|{f}", "slug": slug(s["store_nbr"], f), "store_nbr": s["store_nbr"],
                        "family": f, "city": s["city"], "type": s["type"], "cluster": s["cluster"]})
    return out


# ------------------------------------------------------------------------------------------------
# promo what-if (one LightGBM refit at the forward origin)
# ------------------------------------------------------------------------------------------------
def compute_whatif(inp) -> pd.DataFrame:
    """Daily what-if forecasts for every series x scale. Columns: series_id, horizon, target_date, scale_pct,
    forecast, promo (scaled planned promo count)."""
    from .models.lgbm import fit_lightgbm, promo_whatif
    params = inp["selection"]["model_params"]["lightgbm"]
    p = inp["panel"]
    p = p[p["date"] <= FORWARD_END]  # same truncation as the harness (last needed target date)
    view = masked_view(p, FORWARD_ORIGIN, "A")
    fit = fit_lightgbm(view, FORWARD_ORIGIN, C.HORIZON, "A", **params)
    saved = inp["bt"]
    saved = saved[(saved["model"] == "lightgbm") & (saved["variant"] == "A") & (saved["stage"] == "forward")]
    out = []
    for s in WHATIF_SCALES:
        pr = promo_whatif(fit, s)
        sc = fit.scoring[["series_id", "horizon", "promo_t"]].copy()
        pr["promo"] = (sc["promo_t"] * (1 + s / 100.0)).to_numpy()
        pr["scale_pct"] = s
        out.append(pr)
    wi = pd.concat(out, ignore_index=True)
    z = wi[wi["scale_pct"] == 0].merge(saved[["series_id", "horizon", "forecast", "onpromotion"]],
                                       on=["series_id", "horizon"], suffixes=("", "_saved"))
    assert len(z) == 96 * C.HORIZON, len(z)
    err = float((z["forecast"] - z["forecast_saved"]).abs().max())
    assert err <= 1e-6, f"what-if scale 0 differs from saved forward LightGBM forecast by {err}"
    assert np.allclose(z["promo"], z["onpromotion"]), "scoring promo_t differs from planned onpromotion"
    wi.attrs["max_abs_diff_scale0"] = err
    return wi


def promo_history_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Per series, on dates <= forward origin: max daily promo and p95 of the 14-day rolling mean promo
    (same definition as the Trust Index promo flag)."""
    p = panel[panel["date"] <= FORWARD_ORIGIN].sort_values(["series_id", "date"])
    roll = p.groupby("series_id")["onpromotion"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    p = p.assign(roll=roll)
    return pd.DataFrame({"hist_max_promo": p.groupby("series_id")["onpromotion"].max(),
                         "hist_p95_promo": p.groupby("series_id")["roll"].quantile(0.95),
                         "hist_min_roll_promo": p.groupby("series_id")["roll"].min()})


def whatif_table(wi: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    g = wi.groupby(["series_id", "scale_pct"])
    t = pd.DataFrame({"promo_total": g["promo"].sum(), "promo_mean": g["promo"].mean(), "promo_max": g["promo"].max(),
                      "forecast_total": g["forecast"].sum()}).reset_index()
    base = t[t["scale_pct"] == 0].set_index("series_id")["forecast_total"]
    t["change_pct"] = [((ft / base[s] - 1) * 100 if base[s] > 0 else np.nan) for s, ft in zip(t["series_id"], t["forecast_total"])]
    t = t.join(stats, on="series_id")
    t["extrapolation"] = (t["promo_mean"] > t["hist_p95_promo"]) | (t["promo_max"] > t["hist_max_promo"])
    # informational only (not part of the contract's extrapolation flag): plan below anything seen in a 14-day window
    t["below_history"] = t["promo_mean"] < t["hist_min_roll_promo"]
    return t


# ------------------------------------------------------------------------------------------------
# series files
# ------------------------------------------------------------------------------------------------
def build_series(inp, wi: pd.DataFrame, wt: pd.DataFrame) -> dict[str, dict]:
    p = inp["panel"]
    hist = p[(p["date"] >= HISTORY_START) & (p["date"] <= FORWARD_ORIGIN)].sort_values(["series_id", "date"])
    ch = champions(inp)
    bt = inp["bt"]
    fin = bt[bt["stage"].isin(["holdout", "forward"])].copy()
    fin["key"] = np.where(fin["variant"] == "B", fin["model"] + "_B", fin["model"])
    piv = fin.pivot_table(index=["stage", "origin", "series_id", "horizon"], columns="key", values="forecast", aggfunc="first")
    ci = inp["ci"].set_index(["stage", "origin", "series_id", "horizon"]).sort_index()
    lt = latest_trust(inp).set_index("series_id")
    files = {}
    for s in build_series_index(inp):
        sid, fam = s["series_id"], s["family"]
        h = hist[hist["series_id"] == sid]
        assert len(h) == (FORWARD_ORIGIN - HISTORY_START).days + 1
        sales = [None if (c == 1 or pd.isna(v)) else rnd(v, 1) for v, c in zip(h["sales"], h["closed"])]
        holdout = []
        for o in inp["cal"]["stages"]["holdout"]:
            o = pd.Timestamp(o)
            c = ci.loc[("holdout", o, sid)]
            mp = piv.loc[("holdout", o, sid)]
            rows = []
            for hz, r in c.iterrows():
                act = None if (r["closed"] == 1 or pd.isna(r["actual"])) else rnd(r["actual"], 1)
                rows.append({"date": iso(r["target_date"]), "h": int(hz), "actual": act, "p10": rnd(r["p10"], 1),
                             "p90": rnd(r["p90"], 1), "models": {k: rnd(v, 1) for k, v in mp.loc[hz].items()}})
            holdout.append({"origin": iso(o), "champion": ch[fam], "rows": rows})
        c = ci.loc[("forward", FORWARD_ORIGIN, sid)]
        mp = piv.loc[("forward", FORWARD_ORIGIN, sid)]
        frows = [{"date": iso(r["target_date"]), "h": int(hz), "forecast": rnd(r["forecast"], 1), "p10": rnd(r["p10"], 1),
                  "p90": rnd(r["p90"], 1), "onpromotion": rnd(r["onpromotion"], 1),
                  "models": {k: rnd(v, 1) for k, v in mp.loc[hz].items()}} for hz, r in c.iterrows()]
        w = wt[wt["series_id"] == sid].sort_values("scale_pct")
        wd = wi[wi["series_id"] == sid]
        wrows = []
        for r in w.itertuples():
            daily = wd[wd["scale_pct"] == r.scale_pct].sort_values("horizon")["forecast"]
            wrows.append({"scale_pct": int(r.scale_pct), "promo_total": rnd(r.promo_total, 1),
                          "forecast_total": rnd(r.forecast_total, 1), "change_pct": rnd(r.change_pct, 2),
                          "daily": [rnd(x, 1) for x in daily], "extrapolation": bool(r.extrapolation),
                          "below_history": bool(r.below_history)})
        tr = lt.loc[sid]
        files[s["slug"]] = {
            "series_id": sid, "store_nbr": s["store_nbr"], "family": fam,
            "history": {"start": iso(HISTORY_START), "sales": sales},
            "holdout": holdout,
            "forward": {"origin": iso(FORWARD_ORIGIN), "champion": ch[fam], "rows": frows},
            "whatif": {"origin": iso(FORWARD_ORIGIN), "model": "lightgbm (variant A, frozen config, refit at the forward origin)",
                       "hist_max_promo": rnd(w["hist_max_promo"].iloc[0], 2), "hist_p95_promo": rnd(w["hist_p95_promo"].iloc[0], 2),
                       "rows": wrows},
            "trust": {"score": rnd(tr["score"], 2), "band": tr["band"], "components": {k: rnd(tr[k], 2) for k in COMPONENTS},
                      "hard_flags": describe_flags(list(tr["hard_flags"]))},
        }
    return files


# ------------------------------------------------------------------------------------------------
# scenario
# ------------------------------------------------------------------------------------------------
def build_scenario(inp) -> dict:
    fwd = forward_champion(inp)
    tot = fwd.groupby("family")["forecast"].sum()
    src = f"LightGBM champion forecast {iso(FORWARD_START)}..{iso(FORWARD_END)[5:]}, sum of 12 stores (real model output)"
    return {
        "assumption_label": "Editable demonstration assumptions, not measured UAE uplift",
        "families": [{"family": f, "baseline_14d": float(tot[f]),
                      "baseline_source": src if champions(inp)[f] == "lightgbm" else
                      f"{MODEL_LABEL[champions(inp)[f]]} champion forecast, sum of 12 stores (real model output)"}
                     for f in C.FAMILIES],
        "events": SCENARIO_EVENTS,
        "events_note": "Event percentages are placeholders for a planner to edit. They were not estimated from any UAE data.",
        "ecuador_analogue": inp["analogue"],
    }


# ------------------------------------------------------------------------------------------------
# S&OP
# ------------------------------------------------------------------------------------------------
def family_bias(rows: pd.DataFrame) -> pd.Series:
    ev = evaluable(rows)
    g = ev.groupby("family")
    return g.apply(lambda d: M.bias(d["actual"], d["forecast"]))


def persistent_bias(inp) -> list[dict]:
    cal = family_bias(champion_bt(inp, "calibration"))
    hol = family_bias(champion_bt(inp, "holdout"))
    out = []
    for f in C.FAMILIES:
        a, b = cal[f], hol[f]
        if np.sign(a) == np.sign(b) and abs(a) > BIAS_RULE_PP and abs(b) > BIAS_RULE_PP:
            out.append({"family": f, "calibration_bias": float(a), "holdout_bias": float(b)})
    return out


def simulate_fva(inp) -> dict:
    """Simulated rule-based override on holdout actuals (see docs/agent_notes/export.md)."""
    ch_rows = pd.concat([champion_bt(inp, "calibration"), champion_bt(inp, "holdout")])
    ev_all = evaluable(ch_rows)
    hold = evaluable(champion_bt(inp, "holdout")).copy()
    hold["sim"] = hold["forecast"]
    log = []
    for T in sorted(hold["origin"].unique()):
        T = pd.Timestamp(T)
        done = ev_all[ev_all["origin"] + pd.Timedelta(days=C.HORIZON) <= T]
        for fam, d in done.groupby("family"):
            b = M.bias(d["actual"], d["forecast"])
            adj = abs(b) > BIAS_RULE_PP
            m = (hold["origin"] == T) & (hold["family"] == fam)
            if adj:
                hold.loc[m, "sim"] = hold.loc[m, "forecast"] / (1 + b / 100.0)
            log.append({"origin": iso(T), "family": fam, "past_bias": float(b), "adjusted": bool(adj),
                        "n_windows": int(d["origin"].nunique())})
    cw = M.wape(hold["actual"], hold["forecast"])
    sw = M.wape(hold["actual"], hold["sim"])
    return {"champion_wape": float(cw), "simulated_wape": float(sw), "fva_pp": float(cw - sw), "log": log,
            "n_adjusted": int(sum(r["adjusted"] for r in log)), "n_decisions": len(log)}


def build_sop(inp, wt: pd.DataFrame) -> dict:
    lt = latest_trust(inp)
    tw = inp["trust_weights"]["weights"]
    rows = [{"series_id": r["series_id"], "store_nbr": int(r["store_nbr"]), "family": r["family"],
             "stat_forecast": rnd(r["forecast_14d"], 2), "p10": rnd(r["p10"], 2), "p90": rnd(r["p90"], 2),
             "trust_score": rnd(r["score"], 2), "trust_band": r["band"]}
            for _, r in lt.sort_values(["store_nbr", "family"]).iterrows()]
    total = float(lt["forecast_14d"].sum())
    rev = lt[lt["band"] == "Review"]
    name = lambda r: f"Store {int(r['store_nbr'])} · {fam_label(r['family'])}"

    why = lambda r: ("forced by a hard flag" if len(r["hard_flags"]) else "score below 60")
    top_risks = [f"{name(r)}: Review band (score {r['score']:.0f}, {why(r)}), 14-day forecast {fmt_int(r['forecast_14d'])} "
                 f"units (sum of daily P10–P90 {fmt_int(r['p10'])}–{fmt_int(r['p90'])})."
                 for _, r in rev.sort_values("forecast_14d", ascending=False).head(5).iterrows()]

    pb = persistent_bias(inp)
    if pb:
        persist = [f"{fam_label(x['family'])}: champion bias {x['calibration_bias']:+.1f} % on calibration and "
                   f"{x['holdout_bias']:+.1f} % on holdout ({'over' if x['holdout_bias'] > 0 else 'under'}-forecast both times)."
                   for x in pb]
    else:
        persist = [f"No family had a champion bias of the same sign beyond ±{BIAS_RULE_PP:.0f} % on both calibration and holdout."]

    queue = [f"{i + 1}. {name(r)} (score {r['score']:.0f}, priority {r['priority']:.2f}): {reason_for(r, tw)}."
             for i, (_, r) in enumerate(rev.sort_values("priority", ascending=False).iterrows())]

    w0 = wt[wt["scale_pct"] == 0].set_index("series_id")
    w50 = wt[wt["scale_pct"] == 50].set_index("series_id")
    promo = []
    flagged = lt[lt["hard_flags"].apply(lambda f: "promo_above_p95" in list(f))].sort_values("forecast_14d", ascending=False)
    for _, r in flagged.head(4).iterrows():
        promo.append(f"{name(r)}: planned promotions average {r['promo_plan_mean']:.1f} promoted items per day, above this "
                     f"series' historical 95th percentile of 14-day means ({r['promo_p95']:.1f}); the forecast extrapolates "
                     "beyond the promotion intensity the model has seen.")
    cand = w50[~w50["extrapolation"]].sort_values("change_pct", ascending=False).head(3)
    for sid, r in cand.iterrows():
        st, fam = sid.split("|")
        promo.append(f"Store {st} · {fam_label(fam)}: a +50 % promotion plan is associated with a model-predicted change of "
                     f"{r['change_pct']:+.1f} % in the 14-day forecast ({fmt_int(w0.loc[sid, 'forecast_total'])} → "
                     f"{fmt_int(r['forecast_total'])} units); an association learned from history, not a causal uplift.")

    fva = simulate_fva(inp)
    p = inp["panel"]
    last14 = p[(p["date"] > FORWARD_ORIGIN - pd.Timedelta(days=C.HORIZON)) & (p["date"] <= FORWARD_ORIGIN)]["sales"].sum()
    rev_share = rev["forecast_14d"].sum() / total * 100
    disc = [
        f"The chain 14-day statistical forecast is {fmt_int(total)} units, {(total / last14 - 1) * 100:+.1f} % versus the "
        f"last 14 observed days ({fmt_int(last14)} units).",
        f"Start with the {len(rev)} Review-band series: they carry {rev_share:.1f} % of the 14-day forecast volume.",
        (f"Persistent bias: {', '.join(fam_label(x['family']) for x in pb)} — agree whether a family-level correction is justified."
         if pb else
         "No family shows persistent champion bias on both calibration and holdout, so blanket family-level adjustments are "
         "not supported by the evidence."),
        f"In simulation, a rule-based bias-correction override changed holdout WAPE from {fva['champion_wape']:.2f} % to "
        f"{fva['simulated_wape']:.2f} % (FVA {fva['fva_pp']:+.2f} pp). Keep overrides for information the model does not "
        "have and record a reason code.",
    ]
    if len(flagged):
        disc.append(f"{len(flagged)} series have planned promotions above their historical 95th percentile; confirm those "
                    "promotion plans before accepting their forecasts.")
    return {
        "origin": iso(FORWARD_ORIGIN), "horizon_end": iso(FORWARD_END),
        "rows": rows,
        "reason_codes": REASON_CODES,
        "summary": {"top_risks": top_risks, "persistent_bias": persist, "review_queue": queue,
                    "promo_opportunities": promo, "discussion_points": disc},
        "fva": {"label": "Simulated, rule-based override, not human-planner performance",
                "rule": ("For each holdout origin T and family, the champion's bias is measured over all completed windows "
                         "before T (calibration windows and earlier holdout windows with origin + 14 days ≤ T). If |bias| > "
                         f"{BIAS_RULE_PP:.0f} %, the simulated planner divides that family's forecasts by (1 + bias/100); "
                         "otherwise the forecast is kept. fva_pp = champion WAPE − simulated WAPE (positive = the rule "
                         "reduced error)."),
                "stage": "holdout",
                "champion_wape": fva["champion_wape"], "simulated_wape": fva["simulated_wape"], "fva_pp": fva["fva_pp"],
                "n_adjusted": fva["n_adjusted"], "n_decisions": fva["n_decisions"]},
        "notes": ["stat_forecast = 14-day sum of the champion forecast; p10 / p90 = sums of daily bounds (conservative)."],
    }


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
def validate(name: str, obj) -> None:
    schema = SERIES_SCHEMA if name.startswith("series/") else SCHEMAS[name]
    jsonschema.validate(obj, schema)


def build_all(out_dir: Path | None = None, log=print) -> dict:
    out_dir = Path(out_dir or C.DOCS_DATA)
    inp = load_inputs()
    log("inputs loaded")
    ivals = build_intervals(inp)
    trust = build_trust(inp)
    log("refitting LightGBM at the forward origin for the what-if grid ...")
    wi = compute_whatif(inp)
    log(f"what-if scale 0 vs saved forward forecast: max |diff| = {wi.attrs['max_abs_diff_scale0']:.3g}")
    wt = whatif_table(wi, promo_history_stats(inp["panel"]))
    files = {
        "meta.json": build_meta(inp, ivals),
        "overview.json": build_overview(inp, ivals, trust),
        "series_index.json": build_series_index(inp),
        "demand.json": inp["demand"],
        "arena.json": build_arena(inp),
        "intervals.json": ivals,
        "trust.json": trust,
        "scenario.json": build_scenario(inp),
        "sop.json": build_sop(inp, wt),
    }
    assert set(files) == set(SCHEMAS), set(SCHEMAS) ^ set(files)
    for slug_, obj in build_series(inp, wi, wt).items():
        files[f"series/{slug_}.json"] = obj
    sizes = {}
    for name, obj in files.items():
        validate(name, obj)
        sizes[name] = dump(obj, out_dir / name)
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file() and "_fixtures" not in p.parts)
    assert total < MAX_BYTES, f"docs/data is {total} bytes (limit {MAX_BYTES})"
    log(f"wrote {len(files)} files, docs/data total {total / 1024:.1f} KB (excluding _fixtures)")
    return {"sizes": sizes, "total_bytes": total, "whatif_max_diff": wi.attrs["max_abs_diff_scale0"]}
