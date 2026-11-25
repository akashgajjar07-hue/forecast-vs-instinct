"""Champion-challenger: pick the best model per product family on SELECTION backtests.

Rule: lowest selection WAPE (ratio of sums over all series/origins/horizons in the
family); tie-break (WAPE within 0.05 pp) lower |bias|. LightGBM competes with its
variant A (known promo plan); variant B is reported as a comparison only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .backtest import evaluable

TIE_PP = 0.05


def selection_table(bt: pd.DataFrame, champion_variant: str = "A") -> pd.DataFrame:
    sel = evaluable(bt[(bt["stage"] == "selection") & (bt["variant"] == champion_variant)])
    rows = []
    for (fam, model), g in sel.groupby(["family", "model"]):
        r = M.all_metrics(g["actual"], g["forecast"])
        rows.append({"family": fam, "model": model, "wape": r.wape, "bias": r.bias, "n": r.n})
    return pd.DataFrame(rows)


def pick_champions(tbl: pd.DataFrame) -> dict:
    out = {}
    for fam, g in tbl.groupby("family"):
        g = g.dropna(subset=["wape"]).sort_values("wape")
        best = g.iloc[0]["wape"]
        tied = g[g["wape"] <= best + TIE_PP].assign(absbias=lambda d: d["bias"].abs()).sort_values(["absbias", "wape"])
        c = tied.iloc[0]
        out[fam] = {"model": c["model"], "selection_wape": float(c["wape"]), "selection_bias": float(c["bias"])}
    return out


def champion_rows(bt: pd.DataFrame, champions: dict, variant: str = "A") -> pd.DataFrame:
    """Backtest rows of the champion system (each family's champion model)."""
    m = pd.Series({f: c["model"] for f, c in champions.items()})
    b = bt[bt["variant"] == variant]
    return b[b["model"] == b["family"].map(m)].copy()
