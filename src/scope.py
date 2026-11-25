"""Choose the 12 stores x 8 families (96 series) using ONLY data before the first
selection origin, so scope choice cannot be influenced by evaluation periods."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import config as C
from .data import load_raw, store_day_totals

TYPE_QUOTA = {"A": 2, "B": 2, "C": 3, "D": 3, "E": 2}  # roughly proportional to the 54-store mix, all types covered


def select_scope(cutoff: str, raw=None) -> dict:
    train, stores, _, _ = raw if raw is not None else load_raw()
    cutoff = pd.Timestamp(cutoff)
    tr = train[train["date"] < cutoff]
    tot = store_day_totals(tr)
    win = tot[tot["date"] >= C.MODEL_START]
    known_closures = (win["date"].dt.month.eq(1) & win["date"].dt.day.eq(1))  # New Year: closed chain-wide, flagged not excluded
    first_sale = tot[tot["store_total"] > 0].groupby("store_nbr")["date"].min()
    zero_days = win[(win["store_total"] == 0) & ~known_closures].groupby("store_nbr").size()
    n_days = (cutoff - pd.Timestamp(C.MODEL_START)).days
    obs_days = win.groupby("store_nbr").size()
    fam = tr[(tr["date"] >= C.MODEL_START) & tr["family"].isin(C.FAMILIES)]
    fam = fam.merge(tot, on=["store_nbr", "date"])
    fam_zero = fam[fam["store_total"] > 0].groupby("store_nbr")["sales"].apply(lambda s: (s == 0).mean())

    st = stores.set_index("store_nbr").copy()
    st["first_sale"] = first_sale
    st["closure_days_in_window"] = zero_days.reindex(st.index).fillna(0).astype(int)
    st["family_zero_share"] = fam_zero.reindex(st.index)
    st["volume"] = win.groupby("store_nbr")["store_total"].sum()
    reasons = {}
    for s, r in st.iterrows():
        why = []
        if pd.isna(r.first_sale) or r.first_sale > pd.Timestamp(C.MODEL_START) - pd.Timedelta(days=56):
            why.append("opened too late (needs 56 days of history before model start)")
        if r.closure_days_in_window > 0:
            why.append(f"{r.closure_days_in_window} closure days in modelling window")
        if obs_days.get(s, 0) < n_days - 4:  # 25 Dec missing chain-wide (2 occurrences)
            why.append("missing days")
        if r.family_zero_share > 0.01:
            why.append("zero sales in selected families on open days")
        reasons[int(s)] = why
    st["eligible"] = [len(reasons[int(s)]) == 0 for s in st.index]

    chosen, used_clusters, used_cities = [], set(), set()
    for t, q in TYPE_QUOTA.items():
        cand = st[st["eligible"] & (st["type"] == t)].copy()
        med = cand["volume"].median()
        for _ in range(q):
            cand = cand.drop(index=[c for c in chosen if c in cand.index])
            if cand.empty:
                break
            key = pd.DataFrame({
                "new_cluster": ~cand["cluster"].isin(used_clusters),
                "new_city": ~cand["city"].isin(used_cities),
                "rep": -(cand["volume"] - med).abs(),  # closest to type median = representative
            }, index=cand.index).sort_values(["new_cluster", "new_city", "rep"], ascending=False)
            pick = int(key.index[0])
            chosen.append(pick)
            used_clusters.add(cand.loc[pick, "cluster"]); used_cities.add(cand.loc[pick, "city"])
    chosen = sorted(chosen)
    sel = st.loc[chosen]
    assert len(chosen) == C.N_STORES and sel["city"].nunique() >= 4, "scope criteria not met"

    # per-series stats on the scope window
    p = tr[tr["store_nbr"].isin(chosen) & tr["family"].isin(C.FAMILIES) & (tr["date"] >= C.MODEL_START)]
    p = p.merge(tot, on=["store_nbr", "date"]); p = p[p["store_total"] > 0]
    g = p.groupby(["store_nbr", "family"])
    # No per-series mean sales or mean promo count: those are the private index bases (src/indexing.py) and are
    # never published (Kaggle rules: raw competition data is not redistributed).
    stats = pd.DataFrame({
        "cv": g["sales"].std() / g["sales"].mean(),
        "zero_share": g["sales"].apply(lambda s: (s == 0).mean()),
        "promo_day_share": g["onpromotion"].apply(lambda s: (s > 0).mean()),
        "n_open_days": g.size(),
    }).reset_index()
    stats["series_id"] = stats["store_nbr"].astype(str) + "|" + stats["family"]
    return {
        "cutoff_exclusive": cutoff.strftime("%Y-%m-%d"),
        "window_start": C.MODEL_START,
        "criteria": [
            "Only data before the first selection origin was used.",
            "Store open since >=56 days before the modelling window start (2015-06-01).",
            "No unexplained closure days in the window (chain-wide 1 Jan and 25 Dec closures are flagged, not disqualifying).",
            "No zero-sales days for the 8 families on open days (no recording gaps).",
            "Quota per store type A2/B2/C3/D3/E2; within type prefer an unused cluster, then an unused city, then volume closest to the type median (representative, not just the biggest stores).",
        ],
        "families": C.FAMILIES,
        "stores": [{"store_nbr": int(s), "city": r.city, "state": r.state, "type": r.type, "cluster": int(r.cluster),
                    "first_sale": r.first_sale.strftime("%Y-%m-%d")} for s, r in sel.iterrows()],
        "excluded_stores": {str(k): v for k, v in reasons.items() if v},
        "n_eligible": int(st["eligible"].sum()),
        "series": stats.round(4).to_dict(orient="records"),
    }
