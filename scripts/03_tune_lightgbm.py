"""Phase 2 (Agent B): LightGBM tuning on SELECTION origins only, then frozen-config backtests.

Holdout guard: the panel handed to the harness is truncated at the end of the calibration target window
(2017-06-27), so no holdout actual can be read, and only selection/calibration origins are ever run.

Outputs
  outputs/lgbm_tuning.json              every candidate (selection WAPE/bias, runtime), choices, quake/oil
  outputs/backtests/lightgbm_A.parquet  chosen config, variant A, selection + calibration
  outputs/backtests/lightgbm_B.parquet  chosen config, variant B, selection + calibration
  outputs/lgbm_feature_importance.json  mean gain share / split count per feature over the backtest origins

Usage: python scripts/03_tune_lightgbm.py            (tune + final)
       python scripts/03_tune_lightgbm.py --final-only (reuse chosen params from lgbm_tuning.json)
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.backtest import run_origin, evaluable  # noqa: E402
from src.features import load_oil  # noqa: E402
from src.metrics import wape, bias, grouped_metrics  # noqa: E402
from src.models.lgbm import DEFAULT_PARAMS, fit_lightgbm, predict  # noqa: E402
from src.splits import load_split_calendar  # noqa: E402

OUT_TUNE = C.OUTPUTS / "lgbm_tuning.json"
OUT_IMP = C.OUTPUTS / "lgbm_feature_importance.json"
BT_DIR = C.OUTPUTS / "backtests"
TUNE_STRIDE = 2  # training-origin stride during the grid (halves runtime); final stride is checked in stage 4


def guarded_panel(cal):
    p = pd.read_parquet(C.PROCESSED / "panel.parquet")
    cutoff = pd.Timestamp(cal["target_windows"]["calibration"][1])
    assert cutoff < pd.Timestamp(cal["stages"]["holdout"][0]) + pd.Timedelta(days=1)
    return p[p["date"] <= cutoff].reset_index(drop=True)


def check_origins(cal, origins):
    hold0 = pd.Timestamp(cal["stages"]["holdout"][0])
    for s, o in origins:
        assert s in ("selection", "calibration") and o < hold0, f"refusing to run {s} {o.date()}"


class _Capture:
    """model_fn wrapper that also keeps feature importances (harness still owns the masked view)."""

    def __init__(self, variant, oil):
        self.variant, self.oil, self.importance, self.times, self.n_train = variant, oil, [], [], []

    def __call__(self, view, origin, horizon, **params):
        t = time.time()
        fit = fit_lightgbm(view, origin, horizon, self.variant, oil=self.oil, **params)
        out = predict(fit)
        self.times.append(time.time() - t)
        self.importance.append(fit.importance)
        self.n_train.append(fit.n_train)
        return out


def run(panel, origins, variant, params, oil):
    cap = _Capture(variant, oil)
    bt = pd.concat([run_origin("lightgbm", cap, panel, o, s, variant, **params) for s, o in origins],
                   ignore_index=True)
    return bt, cap


def score(bt):
    e = evaluable(bt)
    return round(wape(e["actual"], e["forecast"]), 4), round(bias(e["actual"], e["forecast"]), 4)


def tune(panel, cal, oil):
    sel = [("selection", pd.Timestamp(o)) for o in cal["stages"]["selection"]]
    check_origins(cal, sel)
    cands, log = {}, []

    def evaluate(name, stage, **over):
        params = {**DEFAULT_PARAMS, "origin_stride": TUNE_STRIDE, **over}
        t = time.time()
        bt, cap = run(panel, sel, "A", params, oil)
        w, b = score(bt)
        rec = {"name": name, "stage": stage, "params": {k: params[k] for k in sorted(params)},
               "selection_wape": w, "selection_bias": b, "runtime_s": round(time.time() - t, 1),
               "mean_fit_s": round(float(np.mean(cap.times)), 2), "n_train_last": cap.n_train[-1]}
        log.append(rec)
        cands[name] = rec
        print(f"{stage:>10} {name:<28} WAPE {w:7.3f}  bias {b:+7.3f}  {rec['runtime_s']:6.1f}s", flush=True)
        return rec

    best = lambda recs: min(recs, key=lambda r: (round(r["selection_wape"], 2), abs(r["selection_bias"])))

    # Stage 1: objective / target transform
    s1 = [evaluate("tweedie_p1.2", "objective", objective="tweedie", tweedie_variance_power=1.2),
          evaluate("tweedie_p1.5", "objective", objective="tweedie", tweedie_variance_power=1.5),
          evaluate("l2_raw", "objective", objective="regression"),
          evaluate("l1_raw", "objective", objective="regression_l1"),
          evaluate("l2_ratio28", "objective", objective="regression", target="ratio")]
    b1 = best(s1)
    base = {k: b1["params"][k] for k in ("objective", "tweedie_variance_power", "target")}

    # Stage 2: small structural grid around the chosen objective
    s2 = [b1]
    for name, over in [("leaves31", dict(num_leaves=31)), ("leaves127", dict(num_leaves=127)),
                       ("min_leaf30", dict(min_data_in_leaf=30)), ("min_leaf300", dict(min_data_in_leaf=300)),
                       ("lr0.03_n800", dict(learning_rate=0.03, n_estimators=800)),
                       ("lr0.1_n200", dict(learning_rate=0.1, n_estimators=200))]:
        s2.append(evaluate(f"{b1['name']}+{name}", "structure", **base, **over))
    b2 = best(s2)
    chosen = {k: b2["params"][k] for k in ("objective", "tweedie_variance_power", "target", "num_leaves",
                                           "learning_rate", "n_estimators", "min_data_in_leaf")}

    # Stage 3: data traps
    q_drop = evaluate("quake_drop_targets", "quake", **chosen, quake="drop")
    quake = "drop" if (round(q_drop["selection_wape"], 2), abs(q_drop["selection_bias"])) < \
        (round(b2["selection_wape"], 2), abs(b2["selection_bias"])) else "flag"
    chosen["quake"] = quake
    ref = q_drop if quake == "drop" else b2
    oil_rec = evaluate("oil_lagged", "oil", **chosen, use_oil=True)
    use_oil = round(oil_rec["selection_wape"], 2) < round(ref["selection_wape"], 2)
    chosen["use_oil"] = use_oil
    ref = oil_rec if use_oil else ref

    # Stage 4: training-origin stride (all daily origins vs every 2nd day)
    s1_rec = evaluate("stride1", "stride", **chosen, origin_stride=1)
    stride = 1 if round(s1_rec["selection_wape"], 2) < round(ref["selection_wape"], 2) else TUNE_STRIDE
    chosen["origin_stride"] = stride

    decisions = {
        "objective_stage_winner": b1["name"], "structure_stage_winner": b2["name"],
        "quake": {"choice": quake, "flag_wape": b2["selection_wape"], "flag_bias": b2["selection_bias"],
                  "drop_wape": q_drop["selection_wape"], "drop_bias": q_drop["selection_bias"]},
        "oil": {"keep": use_oil, "without_wape": (q_drop if quake == "drop" else b2)["selection_wape"],
                "with_wape": oil_rec["selection_wape"], "rule": "keep only if selection WAPE (2 dp) improves"},
        "origin_stride": {"choice": stride, "stride2_wape": ref["selection_wape"],
                          "stride1_wape": s1_rec["selection_wape"],
                          "rule": "stride 1 only if selection WAPE (2 dp) improves on stride 2"},
    }
    return {**DEFAULT_PARAMS, **chosen}, log, decisions


def by_family(bt):
    g = grouped_metrics(evaluable(bt), "family")
    return {r["family"]: {"wape": round(r["wape"], 4), "bias": round(r["bias"], 4)} for _, r in g.iterrows()}


def final(panel, cal, params, oil):
    origins = [("selection", pd.Timestamp(o)) for o in cal["stages"]["selection"]] + \
              [("calibration", pd.Timestamp(o)) for o in cal["stages"]["calibration"]]
    check_origins(cal, origins)
    BT_DIR.mkdir(parents=True, exist_ok=True)
    results, imps = {}, {}
    for v in ("A", "B"):
        t = time.time()
        bt, cap = run(panel, origins, v, params, oil)
        bt.to_parquet(BT_DIR / f"lightgbm_{v}.parquet", index=False)
        res = {"runtime_s": round(time.time() - t, 1), "mean_fit_s_per_origin": round(float(np.mean(cap.times)), 2)}
        for st in ("selection", "calibration"):
            b = bt[bt["stage"] == st]
            w, bi = score(b)
            res[st] = {"wape": w, "bias": bi, "by_family": by_family(b)}
        results[v] = res
        feats = list(cap.importance[0]["gain"])
        gain = np.array([[imp["gain"][f] for f in feats] for imp in cap.importance])
        share = (gain / gain.sum(1, keepdims=True)).mean(0) * 100
        split = np.array([[imp["split"][f] for f in feats] for imp in cap.importance]).mean(0)
        order = np.argsort(-share)
        imps[v] = [{"feature": feats[i], "gain_share_pct": round(float(share[i]), 3),
                    "mean_splits": round(float(split[i]), 1)} for i in order]
        print(f"final {v}: sel WAPE {res['selection']['wape']:.3f} bias {res['selection']['bias']:+.3f} | "
              f"cal WAPE {res['calibration']['wape']:.3f} bias {res['calibration']['bias']:+.3f} | "
              f"{res['runtime_s']}s", flush=True)
    OUT_IMP.write_text(json.dumps({"method": "mean over selection+calibration origins of per-origin gain share (%)",
                                   "params": params, "variants": imps}, indent=2, default=str))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()
    cal = load_split_calendar()
    panel = guarded_panel(cal)
    oil = load_oil()
    t0 = time.time()
    if args.final_only:
        prev = json.loads(OUT_TUNE.read_text())
        params, log, decisions = prev["chosen_params"], prev["candidates"], prev["decisions"]
        t_tune = prev["runtime_s"]["tuning"]
    else:
        params, log, decisions = tune(panel, cal, oil)
        t_tune = time.time() - t0
    results = final(panel, cal, params, oil)
    out = {
        "selection_metric": "overall WAPE over the 10 selection origins, variant A (tie-break |bias|, WAPE rounded to 2 dp)",
        "tuning_variant": "A (chosen params reused unchanged for variant B)",
        "tuning_origin_stride": TUNE_STRIDE,
        "holdout_guard": "panel truncated at %s; holdout origins never run" % cal["target_windows"]["calibration"][1],
        "candidates": log, "chosen_params": params, "decisions": decisions,
        "final": results, "runtime_s": {"tuning": round(t_tune, 1), "final": round(time.time() - t0 - (0 if args.final_only else t_tune), 1)},
    }
    OUT_TUNE.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"chosen": params, "decisions": decisions}, indent=2, default=str))


if __name__ == "__main__":
    main()
