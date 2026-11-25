"""Phase 2 / Agent A: run the four baselines on the selection + calibration stages.

- tunes the moving-average window on SELECTION only (overall WAPE; all candidates recorded)
- compares Holt-Winters trend settings (none vs additive damped) on SELECTION only
- runs naive, seasonal_naive, moving_average(best window), holt_winters(best trend) on
  selection + calibration and saves outputs/backtests/<model>.parquet (variant A)
- writes outputs/baselines_summary.json

Never touches holdout or forward stages (src.runner refuses them without allow_final).
Usage: python scripts/02_run_baselines.py [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.backtest import evaluable  # noqa: E402
from src.metrics import all_metrics, grouped_metrics  # noqa: E402
from src.models import baselines as B  # noqa: E402
from src.runner import DEV_STAGES, Timer, run_model_stages  # noqa: E402

MA_WINDOWS = [7, 14, 28, 56]
HW_TRENDS = {"none": None, "add_damped": "add"}


def sel_scores(bt: pd.DataFrame) -> dict:
    ev = evaluable(bt[bt["stage"] == "selection"])
    o = all_metrics(ev["actual"], ev["forecast"])
    fam = grouped_metrics(ev, "family").set_index("family")
    return {
        "overall": {"wape": round(o.wape, 3), "bias": round(o.bias, 3), "n": o.n},
        "by_family": {f: {"wape": round(r.wape, 3), "bias": round(r.bias, 3), "n": int(r.n)}
                      for f, r in fam.reindex(C.FAMILIES).iterrows()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    summary = {"stages_run": list(DEV_STAGES), "variant": "A", "runtimes_sec": {}, "selection": {}}

    # 1. moving-average window, selection only
    ma_tuning = {}
    with Timer() as t:
        for w in MA_WINDOWS:
            bt = run_model_stages("moving_average", B.moving_average, panel, ["selection"], save=False, window=w)
            ma_tuning[w] = sel_scores(bt)["overall"]
    summary["runtimes_sec"]["ma_tuning_selection"] = t.seconds
    best_w = min(MA_WINDOWS, key=lambda w: ma_tuning[w]["wape"])
    summary["moving_average"] = {"candidates": {str(w): v for w, v in ma_tuning.items()},
                                 "chosen_window": best_w, "brief_default": 28,
                                 "criterion": "lowest overall selection WAPE"}
    print("MA tuning:", ma_tuning, "-> window", best_w)

    # 2. Holt-Winters trend setting, selection only
    hw_tuning, hw_fb = {}, {}
    with Timer() as t:
        for k, tr in HW_TRENDS.items():
            B.HW_FALLBACKS.clear()
            bt = run_model_stages("holt_winters", B.holt_winters, panel, ["selection"], save=False,
                                  trend=tr, n_jobs=args.n_jobs)
            hw_tuning[k] = sel_scores(bt)["overall"]
            hw_fb[k] = len(B.HW_FALLBACKS)
    summary["runtimes_sec"]["hw_trend_comparison_selection"] = t.seconds
    best_tr = min(HW_TRENDS, key=lambda k: hw_tuning[k]["wape"])
    print("HW tuning:", hw_tuning, "fallbacks", hw_fb, "->", best_tr)

    # 3. final baseline runs on selection + calibration, saved
    specs = {
        "naive": (B.naive, {}),
        "seasonal_naive": (B.seasonal_naive, {}),
        "moving_average": (B.moving_average, {"window": best_w}),
        "holt_winters": (B.holt_winters, {"trend": HW_TRENDS[best_tr], "n_jobs": args.n_jobs}),
    }
    for name, (fn, params) in specs.items():
        B.HW_FALLBACKS.clear()
        with Timer() as t:
            bt = run_model_stages(name, fn, panel, DEV_STAGES, **params)
        summary["runtimes_sec"][name] = t.seconds
        summary["selection"][name] = sel_scores(bt)
        summary["selection"][name]["params"] = {k: v for k, v in params.items() if k != "n_jobs"}
        if name == "holt_winters":
            fb = list(B.HW_FALLBACKS)
            summary["holt_winters"] = {
                "trend_candidates": {k: {**hw_tuning[k], "fallbacks": hw_fb[k]} for k in HW_TRENDS},
                "chosen_trend": best_tr, "fit_days": 364,
                "fallbacks_total": len(fb),
                "fallbacks_by_stage": {s: sum(1 for f in fb if f["origin"] in
                                              {str(o.date()) for o in bt.loc[bt.stage == s, "origin"].unique()})
                                       for s in DEV_STAGES},
                "fallback_events": fb,
                "fits_total": int(bt[["origin", "series_id"]].drop_duplicates().shape[0]),
            }
        print(f"{name}: {t.seconds}s, selection WAPE {summary['selection'][name]['overall']['wape']}")
    B.shutdown_pool()

    out = C.OUTPUTS / "baselines_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
