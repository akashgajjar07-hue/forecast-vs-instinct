"""Calibrate and freeze the Forecast Trust Index weights on CALIBRATION origins only.

    .venv/bin/python scripts/06_trust_calibrate.py [--no-wait]

1. Waits (30 s polls, up to 60 min) for the integrator's freezes (outputs/selection.sha256,
   outputs/intervals.sha256) and the LightGBM variant-A backtest; verifies both hashes (refuses otherwise).
2. Loads selection + calibration backtest rows only (asserts no holdout / forward row is in memory) and
   the panel truncated at the last calibration origin.
3. Computes components at the 8 calibration origins. Interval quantiles = re-fitted on SELECTION champion
   residuals (the calibration quantiles do not exist yet at a calibration origin).
4. Compares weight candidates with the pre-stated rule (docs/agent_notes/trust.md), freezes
   `trust_weights` and writes outputs/trust_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import trust as TR  # noqa: E402
from src.freeze import freeze, sha256_file, verify  # noqa: E402
from src.intervals import fit_quantiles  # noqa: E402
from src.runner import BACKTEST_DIR, load_backtests  # noqa: E402
from src.selection import champion_rows  # noqa: E402
from src.splits import load_split_calendar  # noqa: E402

POLL_S, MAX_WAIT_S = 30, 60 * 60
MARGIN = 0.03


def wait_for(paths: list[Path], wait: bool) -> None:
    t0 = time.time()
    while True:
        missing = [p for p in paths if not p.exists()]
        if not missing:
            return
        if not wait or time.time() - t0 > MAX_WAIT_S:
            raise SystemExit(f"refusing to run: missing {[str(p) for p in missing]}")
        print(f"waiting for {[p.name for p in missing]} ...", flush=True)
        time.sleep(POLL_S)


def champions_from(sel: dict) -> dict:
    c = sel.get("champions", sel)
    out = {f: (v if isinstance(v, dict) else {"model": v}) for f, v in c.items() if f in C.FAMILIES}
    assert set(out) == set(C.FAMILIES), f"selection.json: champions for {sorted(out)}"
    return out


def main(wait: bool = True) -> None:
    wait_for([C.OUTPUTS / "selection.sha256", C.OUTPUTS / "intervals.sha256", BACKTEST_DIR / "lightgbm_A.parquet"],
             wait)
    sel = verify("selection")
    verify("intervals")  # used for holdout/forward origins later; verified here so the chain is intact
    champions = champions_from(sel)

    cal = load_split_calendar()
    cal_origins = pd.to_datetime(cal["stages"]["calibration"])
    T_max = cal_origins.max()

    bt = load_backtests()
    bt["origin"] = pd.to_datetime(bt["origin"])
    bt = bt[bt["stage"].isin(["selection", "calibration"]) & (bt["variant"] == "A")].copy()
    assert not bt["stage"].isin(["holdout", "forward"]).any()
    assert bt["origin"].max() <= T_max, "a non-calibration origin slipped in"
    hold_start = pd.Timestamp(cal["target_windows"]["holdout"][0])
    assert bt["target_date"].max() < hold_start, "holdout target dates in memory"
    models_present = sorted(bt["model"].unique())
    assert set(C.MODELS) <= set(models_present), f"backtests missing models: {set(C.MODELS) - set(models_present)}"

    panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
    panel = panel[(panel["date"] >= C.MODEL_START) & (panel["date"] <= T_max)].copy()
    assert panel["date"].max() < hold_start

    # pre-calibration quantiles: selection champion residuals only
    champ = champion_rows(bt, champions)
    q_sel = fit_quantiles(champ[champ["stage"] == "selection"], panel)
    comp = TR.components(bt, panel, champions, cal_origins, {o: q_sel for o in cal_origins})
    assert len(comp) == 96 * len(cal_origins)
    assert (comp["n_models"] == len(C.MODELS)).all(), "agreement needs all five models at every origin"

    realised = TR.realised_window_wape(champ[champ["stage"] == "calibration"])
    cands, constant_components = TR.identifiable_candidates(TR.standard_candidates(0.1), comp)
    tab = TR.compare_weights(comp, realised, cands, split_origins=cal_origins[:4])
    chosen, reason = TR.choose_weights(tab, "default", MARGIN)
    weights = TR.check_weights(cands[chosen])

    # bands on calibration
    s = TR.score(comp, weights)
    bands = [{"min": float(m), "band": b} for m, b in C.TRUST_BANDS]
    b = TR.bands_for(s, comp["hard_flags"], bands)
    dist = b.value_counts().reindex(["Accept", "Monitor", "Review"]).fillna(0).astype(int)
    threshold_note = "kept 80 / 60 (no band empty on calibration)"
    if (dist == 0).any():
        hi, lo = (int(round(v)) for v in np.quantile(s, [2 / 3, 1 / 3]))
        bands = [{"min": float(hi), "band": "Accept"}, {"min": float(lo), "band": "Monitor"}, {"min": 0.0, "band": "Review"}]
        threshold_note = f"band(s) {list(dist[dist == 0].index)} empty at 80/60; thresholds moved to calibration score terciles {hi}/{lo}"
        b = TR.bands_for(s, comp["hard_flags"], bands)
        dist = b.value_counts().reindex(["Accept", "Monitor", "Review"]).fillna(0).astype(int)
    score_only = TR.bands_for(s, None, bands).value_counts().reindex(["Accept", "Monitor", "Review"]).fillna(0).astype(int)
    flag_counts = comp["hard_flags"].explode().dropna().value_counts().to_dict()

    tab = tab.sort_values("spearman").reset_index(drop=True)
    fin = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    row = lambda r: {"name": r["name"], "spearman": fin(r["spearman"]), "p_value": fin(r["p_value"]),
                     "spearman_first_half": fin(r["spearman_first_half"]),
                     "spearman_second_half": fin(r["spearman_second_half"]), "n": int(r["n"]),
                     "weights": {k: round(v, 3) for k, v in r["weights"].items()}}
    named = tab[tab["name"].isin(["default", "equal", "volatility_only"])]
    grid = tab[tab["name"].str.startswith("grid:")]
    j = comp.merge(realised[["series_id", "origin", "wape"]], on=["series_id", "origin"])
    single = {c: fin(TR._spearman(j[c], j["wape"])[0]) for c in TR.COMPONENTS}

    frozen = {
        "weights": weights, "bands": bands, "constants": TR.CONSTANTS, "chosen": chosen,
        "choice_rule": f"most negative Spearman(score, realised window WAPE) on calibration; non-default adopted only if "
                       f"it beats default by >= {MARGIN} and on both calibration halves",
        "choice_reason": reason, "threshold_note": threshold_note,
        "constant_components_on_calibration": constant_components,
        "interval_quantile_source": {"calibration": "fit_quantiles on SELECTION champion residuals",
                                     "holdout_forward": "frozen calibration quantiles in outputs/intervals.json",
                                     "intervals_sha256": sha256_file(C.OUTPUTS / "intervals.json")},
        "agreement_models": {"models": C.MODELS, "lightgbm_variant": "A"},
        "calibration_origins": [o.strftime("%Y-%m-%d") for o in cal_origins],
    }
    digest = freeze("trust_weights", frozen)

    out = {
        "objective": "Spearman(trust score, realised champion window WAPE); more negative = better",
        "rule": frozen["choice_rule"], "margin": MARGIN,
        "origins": frozen["calibration_origins"], "n_windows": int(tab["n"].iloc[0]),
        "candidates": [row(r) for _, r in named.iterrows()] + [row(r) for _, r in grid.head(10).iterrows()],
        "grid_size": int(len(grid)),
        "constant_components_on_calibration": constant_components,
        "grid_rank_of_default": int(tab.index[tab["name"] == "default"][0]) + 1,
        "single_component_spearman": single,
        "chosen": chosen, "choice_reason": reason, "weights": weights,
        "bands": bands, "threshold_note": threshold_note,
        "band_distribution": dist.to_dict(), "band_distribution_score_only": score_only.to_dict(),
        "hard_flag_counts": {k: int(v) for k, v in flag_counts.items()},
        "score_summary": {k: round(float(v), 2) for k, v in s.describe(percentiles=[.1, .25, .5, .75, .9]).items()},
        "component_summary": {c: {k: round(float(v), 2) for k, v in comp[c].describe().items()} for c in TR.COMPONENTS},
        "realised_wape_by_band": {
            bb: round(float(g["sum_abs_error"].sum() / g["sum_actual"].sum() * 100), 2)
            for bb, g in comp.assign(band=b.values).merge(realised, on=["series_id", "origin"]).groupby("band")},
        "trust_weights_sha256": digest,
    }
    (C.OUTPUTS / "trust_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["chosen", "choice_reason", "weights", "band_distribution",
                                          "single_component_spearman", "grid_rank_of_default"]}, indent=2))
    print(pd.DataFrame(out["candidates"])[["name", "spearman", "spearman_first_half", "spearman_second_half"]]
          .to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true")
    main(wait=not ap.parse_args().no_wait)
