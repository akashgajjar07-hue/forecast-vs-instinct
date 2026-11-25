"""THE ONE-TIME HOLDOUT RUN (+ forward forecast from the last actual).

Preconditions (enforced):
  - outputs/selection, intervals, trust_weights .json all frozen and hash-verified
  - outputs/holdout_run.json must NOT exist (holdout already touched) unless --rerun is given,
    in which case results must reproduce the logged digest exactly (reproducibility check only).
Nothing chosen here: every model, parameter, quantile and weight comes from the frozen files.
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, metrics as M, trust as TR  # noqa: E402
from src.backtest import evaluable  # noqa: E402
from src.freeze import verify_all, sha256_file  # noqa: E402
from src.intervals import apply_intervals, apply_total_band, total_rows  # noqa: E402
from src.models import baselines as B  # noqa: E402
from src.models.lgbm import lightgbm_model  # noqa: E402
from src.runner import run_final_stages, save_backtest, load_backtests  # noqa: E402
from src.selection import champion_rows  # noqa: E402

RUN_LOG = C.OUTPUTS / "holdout_run.json"
HOLD_DIR = C.OUTPUTS / "holdout"


def run_models(panel, frozen):
    p = frozen["selection"]["model_params"]
    t0 = time.time()
    run_final_stages("naive", B.naive, panel)
    run_final_stages("seasonal_naive", B.seasonal_naive, panel)
    run_final_stages("moving_average", B.moving_average, panel, window=p["moving_average"]["window"])
    run_final_stages("holt_winters", B.holt_winters, panel, trend=p["holt_winters"]["trend"],
                     fit_days=p["holt_winters"]["fit_days"], n_jobs=4)
    B.shutdown_pool()
    a = run_final_stages("lightgbm", lightgbm_model, panel, variant="A", save=False)
    save_backtest(a, "lightgbm_A")
    b = run_final_stages("lightgbm", functools.partial(lightgbm_model, variant="B"), panel, variant="B", save=False)
    save_backtest(b, "lightgbm_B")
    return round(time.time() - t0, 1)


def main(rerun=False):
    frozen = verify_all()  # raises HoldoutGuardError if any freeze is missing/modified
    if RUN_LOG.exists() and not rerun:
        raise SystemExit("Holdout already run (outputs/holdout_run.json). Use --rerun only to check reproducibility.")
    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    runtime = run_models(panel, frozen)

    bt = load_backtests()
    champs = frozen["selection"]["champions"]
    q = pd.DataFrame(frozen["intervals"]["quantiles"])
    qt = pd.DataFrame(frozen["intervals"]["total_quantiles"])
    HOLD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- champion system with frozen intervals (holdout + forward) ----
    final = bt[bt.stage.isin(["holdout", "forward"])]
    ch = apply_intervals(champion_rows(final, champs, "A"), panel, q)
    ch.to_parquet(HOLD_DIR / "champion_intervals.parquet", index=False)
    tot = apply_total_band(total_rows(champion_rows(final, champs, "A")).assign(
        stage=lambda d: np.where(d.origin == pd.Timestamp(C.LAST_ACTUAL), "forward", "holdout")), qt)
    tot.to_parquet(HOLD_DIR / "chain_total.parquet", index=False)

    # ---- leaderboard (identical evaluation population for every model) ----
    hold = evaluable(bt[bt.stage == "holdout"])
    keys = hold[hold.model == "naive"][["origin", "series_id", "horizon"]]
    assert all(len(g) == len(keys) for _, g in hold.groupby(["model", "variant"])), "evaluation population differs"
    board = []
    for (m, v), g in hold.groupby(["model", "variant"]):
        r = M.all_metrics(g.actual, g.forecast)
        board.append({"model": m, "variant": v, "wape": r.wape, "bias": r.bias, "mae": r.mae, "rmse": r.rmse, "n": r.n})
    chh = evaluable(ch[ch.stage == "holdout"])
    rc = M.all_metrics(chh.actual, chh.forecast)
    by_fam = []
    for fam, g in hold[hold.variant == "A"].groupby("family"):
        row = {"family": fam, "champion": champs[fam]["model"]}
        for m, gm in g.groupby("model"):
            row[m] = M.wape(gm.actual, gm.forecast)
        row["holdout_best"] = min(C.MODELS, key=lambda m: row[m])
        by_fam.append(row)

    # ---- Trust Index at holdout + forward origins with frozen weights ----
    tw = frozen["trust_weights"]
    origins = sorted(final.origin.unique())
    comp = TR.components(bt, panel, champs, origins, {o: q for o in origins}, constants=tw["constants"])
    comp["score"] = TR.score(comp, tw["weights"])
    comp["band"] = TR.bands_for(comp["score"], comp["hard_flags"], tw["bands"])
    comp["priority"] = TR.priority(comp["score"], comp["forecast_14d"], comp["origin"])
    comp.to_parquet(HOLD_DIR / "trust_scores.parquet", index=False)
    realised = TR.realised_window_wape(champion_rows(bt[bt.stage == "holdout"], champs, "A"))
    hs = comp[comp.origin.isin(bt[bt.stage == "holdout"].origin.unique())]
    validation = TR.validate(hs, realised)

    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_hashes": {n: sha256_file(C.OUTPUTS / f"{n}.json") for n in ("selection", "intervals", "trust_weights")},
        "runtime_s": runtime,
        "leaderboard": sorted(board, key=lambda r: r["wape"]),
        "champion_system": {"wape": rc.wape, "bias": rc.bias, "mae": rc.mae, "rmse": rc.rmse, "n": rc.n,
                            "coverage": M.coverage(chh.actual, chh.p10, chh.p90),
                            "mean_width": M.mean_width(chh.p10, chh.p90)},
        "by_family": by_fam,
        "trust_validation": validation,
        "trust_band_counts_holdout": hs.band.value_counts().to_dict(),
    }
    digest = hashlib.sha256(json.dumps({k: res[k] for k in ("leaderboard", "champion_system", "trust_validation")},
                                       sort_keys=True, default=float).encode()).hexdigest()
    res["results_digest"] = digest
    (C.OUTPUTS / "holdout_results.json").write_text(json.dumps(res, indent=2, default=float))
    if rerun and RUN_LOG.exists():
        prev = json.loads(RUN_LOG.read_text())
        print("REPRODUCED" if prev["results_digest"] == digest else "DIGEST MISMATCH", digest)
    else:
        RUN_LOG.write_text(json.dumps({"first_run_at": res["generated_at"], "results_digest": digest,
                                       "frozen_hashes": res["frozen_hashes"]}, indent=2))
    print(json.dumps({k: res[k] for k in ("leaderboard", "champion_system", "trust_validation")}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    main(ap.parse_args().rerun)
