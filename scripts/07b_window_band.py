"""14-day window-total band per series (added after the independent review).

Why: the dashboard's 14-day "P10-P90" was the SUM of 14 daily bounds, which is not an 80 % band
for the 14-day total (it covered ~94 % of holdout totals). This script replaces it with a band
calibrated directly on 14-day totals.

Protocol (same discipline as the main holdout):
  1. Fit relative-residual quantiles per family on CALIBRATION champion windows ONLY.
  2. Freeze them (outputs/window_intervals.json + .sha256) BEFORE any holdout window is scored.
     If a frozen file already exists, it is verified and reused - never refitted.
  3. Score the holdout windows once and report coverage as it comes out (no tuning).
  4. Apply the frozen band to the forward 14-day forecast (no actuals).
Disclosure: unlike the daily intervals, this band was designed after the holdout had been opened
(the reviewer reported the 94 % coverage of the summed bounds). Its quantiles still use calibration
windows only, and nothing about it was tuned on holdout results.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, metrics as M  # noqa: E402
from src.freeze import freeze, verify, verify_all, HoldoutGuardError  # noqa: E402
from src.intervals import fit_window_quantiles, apply_window_band, LO, HI  # noqa: E402
from src.runner import load_backtests  # noqa: E402
from src.selection import champion_rows  # noqa: E402

NAME = "window_intervals"


def main():
    frozen = verify_all()
    champs = frozen["selection"]["champions"]
    bt = load_backtests()
    ch = champion_rows(bt, champs, "A")

    # 1-2. fit on calibration only, freeze (or verify an existing freeze)
    try:
        wq = verify(NAME)
        q = pd.DataFrame(wq["quantiles"])
        print("reusing frozen", NAME)
    except HoldoutGuardError:
        if (C.OUTPUTS / f"{NAME}.json").exists():
            raise
        cal = ch[ch.stage == "calibration"]
        assert set(cal.stage) == {"calibration"} and cal.target_date.max() <= pd.Timestamp("2017-06-27")
        q = fit_window_quantiles(cal)
        loo = []
        for o in sorted(cal.origin.unique()):
            qo = fit_window_quantiles(cal[cal.origin != o])
            t = apply_window_band(cal[cal.origin == o], qo)
            loo.append(t[t.complete])
        loo = pd.concat(loo)
        freeze(NAME, {
            "method": "Relative-residual split-conformal band on each series' 14-day total: r = (A14 - F14) / F14 on "
                      f"complete calibration windows of the frozen champions; {LO:.2f}/{HI:.2f} quantiles per family; "
                      "P10 = max(0, F14*(1+q_lo)), P90 = F14*(1+q_hi). Same method as the chain-total band.",
            "level": C.INTERVAL_LEVEL,
            "fitted_on": "calibration origins only (8 origins x 96 series = 768 windows)",
            "quantiles": q.round(6).to_dict(orient="records"),
            "calibration_loo_coverage": M.coverage(loo.actual, loo.p10, loo.p90),
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print("froze", NAME)

    # 3. holdout, scored once with the frozen quantiles
    hw = apply_window_band(ch[ch.stage == "holdout"], q)
    hw = hw[hw.complete]
    res = {
        "holdout_coverage": M.coverage(hw.actual, hw.p10, hw.p90),
        "holdout_mean_width_pct": float(((hw.p90 - hw.p10) / hw.forecast).mean() * 100),
        "holdout_below_p10_pct": float((hw.actual < hw.p10).mean() * 100),
        "holdout_above_p90_pct": float((hw.actual > hw.p90).mean() * 100),
        "n_holdout_windows": int(len(hw)),
        "by_family": [
            {"family": f, "coverage": M.coverage(g.actual, g.p10, g.p90),
             "mean_width_pct": float(((g.p90 - g.p10) / g.forecast).mean() * 100), "n": int(len(g))}
            for f, g in hw.groupby("family")],
    }
    # for comparison: the summed daily bounds this band replaces
    from src.intervals import apply_intervals
    daily = apply_intervals(ch[ch.stage == "holdout"], pd.read_parquet(C.PROCESSED / "panel.parquet"),
                            pd.DataFrame(frozen["intervals"]["quantiles"]))
    s = daily.groupby(["series_id", "origin"]).agg(actual=("actual", "sum"), p10=("p10", "sum"), p90=("p90", "sum"),
                                                   forecast=("forecast", "sum")).reset_index()
    res["replaced_sum_of_daily_bounds"] = {"holdout_coverage": M.coverage(s.actual, s.p10, s.p90),
                                           "holdout_mean_width_pct": float(((s.p90 - s.p10) / s.forecast).mean() * 100)}

    # 4. forward 14-day bands
    fw = apply_window_band(ch[ch.stage == "forward"], q)
    res["forward"] = [{"series_id": r.series_id, "family": r.family, "forecast_14d": float(r.forecast),
                       "p10_14d": float(r.p10), "p90_14d": float(r.p90)} for r in fw.itertuples()]
    res["frozen_sha256"] = (C.OUTPUTS / f"{NAME}.sha256").read_text().strip()
    (C.OUTPUTS / "window_band_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("forward", "by_family")}, indent=1))
    print(pd.DataFrame(res["by_family"]).round(1).to_string())


if __name__ == "__main__":
    main()
