"""Generic backtest runner built on src.backtest.run_backtest.

    run_model_stages(name, fn, panel, ["selection", "calibration"], variant="A", **params)

runs a model on every origin of the requested stages and (optionally) saves the rows to
outputs/backtests/<name>.parquet in the contract format (src.backtest.BACKTEST_COLUMNS).
Saving MERGES into an existing file: rows of the same (variant, stage) are replaced,
everything else is kept. So selection/calibration rows written now and holdout/forward rows
written later by the integrator live in one file per model.

Holdout guard: the stages "holdout" and "forward" are refused unless the caller passes
`allow_final=True` (use `run_final_stages`, which is meant to be called once by the
integrator after selection is frozen). The panel is also truncated to the last target date
the requested origins need, so no later actual is ever in memory of the harness call.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from . import config as C
from .backtest import BACKTEST_COLUMNS, run_backtest
from .splits import all_origins, load_split_calendar

BACKTEST_DIR = C.OUTPUTS / "backtests"
DEV_STAGES = ("selection", "calibration")
FINAL_STAGES = ("holdout", "forward")


def stage_origins(stages: Iterable[str], cal: dict | None = None) -> list[tuple[str, pd.Timestamp]]:
    """(stage, origin) pairs. 'forward' = single origin at LAST_ACTUAL (targets are future rows)."""
    cal = cal or load_split_calendar()
    stages = list(stages)
    out = all_origins(cal, [s for s in stages if s != "forward"])
    if "forward" in stages:
        out.append(("forward", pd.Timestamp(C.LAST_ACTUAL)))
    return out


def run_model_stages(name: str, fn: Callable, panel: pd.DataFrame, stages: Iterable[str],
                     variant: str = "A", save: bool = True, out_dir: Path | None = None,
                     cal: dict | None = None, allow_final: bool = False, horizon: int = C.HORIZON,
                     **params) -> pd.DataFrame:
    stages = list(stages)
    final = [s for s in stages if s in FINAL_STAGES]
    if final and not allow_final:
        raise PermissionError(f"stages {final} are final: call run_final_stages() after selection is frozen")
    origins = stage_origins(stages, cal)
    if not origins:
        raise ValueError(f"no origins for stages {stages}")
    last_needed = max(o for _, o in origins) + pd.Timedelta(days=horizon)
    p = panel[panel["date"] <= last_needed]  # actuals beyond the last target are never passed in
    bt = run_backtest(name, fn, p, origins, variant=variant, **params)
    if save:
        save_backtest(bt, name, out_dir)
    return bt


def run_final_stages(name: str, fn: Callable, panel: pd.DataFrame, stages=FINAL_STAGES, variant: str = "A",
                     **params) -> pd.DataFrame:
    """For the integrator ONLY, once, after selection/intervals/trust weights are frozen.
    Verifies the freeze hashes first (src.freeze.verify_all raises HoldoutGuardError otherwise)."""
    from .freeze import verify_all
    verify_all()
    return run_model_stages(name, fn, panel, stages, variant=variant, allow_final=True, **params)


def save_backtest(bt: pd.DataFrame, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or BACKTEST_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    new = bt[BACKTEST_COLUMNS]
    if path.exists():
        old = pd.read_parquet(path)
        keys = set(zip(new["variant"], new["stage"]))
        keep = ~pd.Series(list(zip(old["variant"], old["stage"])), index=old.index).isin(keys)
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["variant", "stage", "origin", "series_id", "horizon"]).reset_index(drop=True)
    new.to_parquet(path, index=False)
    return path


def load_backtests(out_dir: Path | None = None, models: Iterable[str] | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or BACKTEST_DIR)
    files = sorted(out_dir.glob("*.parquet"))
    if models is not None:
        files = [f for f in files if f.stem in set(models)]
    if not files:
        return pd.DataFrame(columns=BACKTEST_COLUMNS)
    return pd.concat([pd.read_parquet(f)[BACKTEST_COLUMNS] for f in files], ignore_index=True)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = round(time.perf_counter() - self.t0, 2)
