"""Freeze model selection (selection stage) and interval quantiles (calibration stage)
BEFORE the holdout is touched. Writes outputs/selection.json|.sha256 and
outputs/intervals.json|.sha256. Refuses to run if any holdout/forward rows exist."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, metrics as M  # noqa: E402
from src.backtest import evaluable  # noqa: E402
from src.freeze import freeze  # noqa: E402
from src.intervals import fit_quantiles, apply_intervals, fit_total_quantiles, LO, HI, SCALE_DAYS  # noqa: E402
from src.runner import load_backtests  # noqa: E402
from src.selection import selection_table, pick_champions, champion_rows  # noqa: E402


def main():
    bt = load_backtests()
    assert set(bt["stage"]) <= {"selection", "calibration"}, "holdout rows present before freeze!"
    assert set(bt["model"]) == set(C.MODELS), f"missing models: {set(C.MODELS) - set(bt.model)}"
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel = panel[panel["date"] <= "2017-06-27"]  # nothing after the calibration window is needed

    tbl = selection_table(bt, "A")
    champs = pick_champions(tbl)
    tblB = selection_table(bt[bt.model == "lightgbm"], "B")
    tuning = json.loads((C.OUTPUTS / "lgbm_tuning.json").read_text())
    base = json.loads((C.OUTPUTS / "baselines_summary.json").read_text())
    sel = {
        "rule": "Per family: lowest selection-backtest WAPE (ratio of sums, 10 origins x 12 stores x 14 days); "
                "tie (within 0.05 pp) -> lower |bias|. LightGBM competes with promo variant A (known promo plan).",
        "champions": champs,
        "selection_table": tbl.round(4).to_dict(orient="records"),
        "lightgbm_variant_B_selection": tblB.round(4).to_dict(orient="records"),
        "model_params": {
            "moving_average": {"window": base["moving_average"]["chosen_window"]},
            "holt_winters": {"trend": None if base["holt_winters"]["chosen_trend"] == "none" else "add",
                             "damped": base["holt_winters"]["chosen_trend"] == "add_damped",
                             "fit_days": base["holt_winters"]["fit_days"]},
            "lightgbm": tuning.get("chosen_params", tuning.get("chosen")),
        },
    }
    h_sel = freeze("selection", sel)

    cal = champion_rows(bt[bt.stage == "calibration"], champs, "A")
    q = fit_quantiles(cal, panel)
    in_sample = apply_intervals(evaluable(cal), panel, q)
    # leave-one-origin-out coverage: honest estimate of out-of-sample calibration
    loo = []
    for o in sorted(cal.origin.unique()):
        qo = fit_quantiles(cal[cal.origin != o], panel)
        loo.append(apply_intervals(evaluable(cal[cal.origin == o]), panel, qo))
    loo = pd.concat(loo)
    qt = fit_total_quantiles(cal)
    ints = {
        "method": f"Scaled split-conformal: residual (A-F)/scale_T, scale_T = mean sales of the {SCALE_DAYS} days up to T; "
                  f"quantiles {LO:.2f}/{HI:.2f} per family x horizon bucket from CALIBRATION champion backtests "
                  f"({cal.origin.nunique()} origins). P10/P90 = max(0, F + q*scale_T).",
        "level": C.INTERVAL_LEVEL,
        "quantiles": q.round(6).to_dict(orient="records"),
        "total_quantiles": qt.round(6).to_dict(orient="records"),
        "calibration_coverage_in_sample": M.coverage(in_sample.actual, in_sample.p10, in_sample.p90),
        "calibration_coverage_leave_one_origin_out": M.coverage(loo.actual, loo.p10, loo.p90),
        "quantile_objective_comparison": "not run (optional in brief); see DECISIONS.md",
    }
    h_int = freeze("intervals", ints)
    print(json.dumps({"champions": {f: c["model"] for f, c in champs.items()},
                      "selection_sha256": h_sel, "intervals_sha256": h_int,
                      "cal_cov_in_sample": round(ints["calibration_coverage_in_sample"], 2),
                      "cal_cov_loo": round(ints["calibration_coverage_leave_one_origin_out"], 2)}, indent=2))


if __name__ == "__main__":
    main()
