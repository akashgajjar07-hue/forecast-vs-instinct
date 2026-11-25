"""Rolling-origin split calendar.

Forecast origin T = end of day T (data up to and including T is known).
Targets are T+1 .. T+14. Origins are weekly (Tuesdays).
Stages are chronological and their TARGET windows never overlap:
    selection  -> model choice, LightGBM tuning, moving-average window
    calibration-> prediction intervals, Trust Index weights/thresholds
    holdout    -> touched once, after everything above is frozen and hashed
"""
from __future__ import annotations

import json
import pandas as pd

from . import config as C

STAGES = ["selection", "calibration", "holdout"]


def build_split_calendar(last_actual=C.LAST_ACTUAL, horizon=C.HORIZON, n_sel=C.N_SELECTION,
                         n_cal=C.N_CALIBRATION, n_hold=C.N_HOLDOUT) -> dict:
    last = pd.Timestamp(last_actual)
    last_hold = last - pd.Timedelta(days=horizon)  # T+14 <= last actual
    while last_hold.weekday() != C.ORIGIN_WEEKDAY:
        last_hold -= pd.Timedelta(days=1)
    wk = pd.Timedelta(days=7)
    hold = [last_hold - wk * i for i in range(n_hold)][::-1]
    # next stage must end its targets before the earlier stage's first target: T_prev + 14 < T_first + 1
    def stage_before(first_origin, n):
        last_o = first_origin - pd.Timedelta(days=horizon)
        while last_o.weekday() != C.ORIGIN_WEEKDAY:
            last_o -= pd.Timedelta(days=1)
        return [last_o - wk * i for i in range(n)][::-1]
    cal = stage_before(hold[0], n_cal)
    sel = stage_before(cal[0], n_sel)
    fmt = lambda xs: [x.strftime("%Y-%m-%d") for x in xs]
    out = {
        "definition": "Origin T = end of day T; targets T+1..T+%d; weekly origins; target windows disjoint across stages." % horizon,
        "horizon": horizon,
        "last_actual": last.strftime("%Y-%m-%d"),
        "model_start": C.MODEL_START,
        "stages": {"selection": fmt(sel), "calibration": fmt(cal), "holdout": fmt(hold)},
        "target_windows": {},
    }
    for s, os_ in out["stages"].items():
        o = pd.to_datetime(os_)
        out["target_windows"][s] = [(o.min() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                    (o.max() + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")]
    validate_split_calendar(out)
    return out


def validate_split_calendar(cal: dict) -> None:
    h = cal["horizon"]
    last = pd.Timestamp(cal["last_actual"])
    prev_end = None
    for s in STAGES:
        o = pd.to_datetime(cal["stages"][s])
        assert len(o) > 0 and o.is_monotonic_increasing and o.is_unique, s
        assert ((o[1:] - o[:-1]).days == 7).all(), f"{s}: origins not weekly"
        start, end = o.min() + pd.Timedelta(days=1), o.max() + pd.Timedelta(days=h)
        assert end <= last, f"{s}: targets beyond last actual"
        if prev_end is not None:
            assert start > prev_end, f"{s}: target window overlaps previous stage"
        prev_end = end
    assert len(cal["stages"]["selection"]) >= 8
    assert len(cal["stages"]["calibration"]) >= 6
    assert len(cal["stages"]["holdout"]) >= 6


def load_split_calendar(path=None) -> dict:
    with open(path or C.CONFIG / "split_calendar.json") as f:
        cal = json.load(f)
    validate_split_calendar(cal)
    return cal


def all_origins(cal: dict, stages=STAGES) -> list[tuple[str, pd.Timestamp]]:
    return [(s, pd.Timestamp(o)) for s in stages for o in cal["stages"][s]]
