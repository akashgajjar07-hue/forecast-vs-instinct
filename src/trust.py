"""Forecast Trust Index: how much should a planner trust a series' 14-day forecast at origin T?

Five components, each normalised to 0-100 (higher = more trustworthy), each computed from
information available at the END of day T only:

  accuracy     100 * (1 - min(WAPE_hist, WAPE_CAP) / WAPE_CAP)
               WAPE_hist = champion WAPE (ratio of sums, open days with an actual) pooled over the
               last <= N_WINDOWS backtest origins o whose whole target window is known: o + 14 <= T.
  interval     100 * (1 - min(|cov_hist - 80|, COV_TOL) / COV_TOL)
               cov_hist = % of actuals inside the champion's P10-P90 band over the same windows.
               The quantiles used must themselves be known at T (see `quantiles_by_origin`).
  stability    100 * (1 - min(CV_56, CV_CAP) / CV_CAP) * (1 - zero_share_56)
               CV_56 = std / mean of daily sales on open, observed days in the 56 days up to T.
  agreement    100 * (1 - min(cv_models / AGREE_CAP, 1))
               cv_models = std / mean (population std) of the 14-day forecast totals of the five
               models (naive, seasonal_naive, moving_average, holt_winters, lightgbm variant A) at T.
  completeness 100 * (# days in the 56-day lookback that are open and have an observed sale) / 56

score = sum_k w_k * component_k  (weights sum to 1). It is an index, NOT a probability.
Bands: score >= 80 Accept, >= 60 Monitor, else Review. Any hard flag forces Review:
  low_completeness   completeness < 80
  holiday_closure    25 Dec or 1 Jan (chain-wide closure) inside T+1 .. T+14
  promo_above_p95    planned mean promo count over T+1..T+14 > the series' p95 of the 14-day rolling
                     mean promo count over history <= T (the forecast is extrapolating)
  missing_<comp>     a component could not be computed (e.g. no completed backtest window yet)

priority = (100 - score) * (series' share of the total 14-day champion forecast at T)
"""
from __future__ import annotations

import itertools
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .backtest import evaluable
from .intervals import apply_intervals
from .selection import champion_rows

COMPONENTS = ["accuracy", "interval", "stability", "agreement", "completeness"]

# Normalisation constants (frozen with the weights; see docs/agent_notes/trust_methodology.md).
CONSTANTS = {
    "wape_cap": 50.0,          # % WAPE at which the accuracy score hits 0
    "cov_target": C.INTERVAL_LEVEL * 100,  # 80 %
    "cov_tol": 40.0,           # |coverage - 80| (pp) at which the interval score hits 0
    "cv_cap": 1.0,             # daily CV at which the stability score hits 0
    "agree_cap": 0.5,          # cross-model CV at which the agreement score hits 0
    "lookback_days": 56,
    "n_windows": 8,
    "horizon": C.HORIZON,
    "completeness_flag": 80.0,
    "promo_window": 14,
    "promo_quantile": 0.95,
    "closure_mmdd": ["12-25", "01-01"],
}

FLAG_TEXT = {
    "low_completeness": "Data completeness below 80% in the last 56 days",
    "holiday_closure": "Known chain-wide closure (25 Dec / 1 Jan) inside the forecast window",
    "promo_above_p95": "Planned promotion intensity above this series' historical 95th percentile",
    "missing_accuracy": "No completed backtest window yet: accuracy unknown",
    "missing_interval": "No completed backtest window yet: interval reliability unknown",
    "missing_stability": "No observed sales in the last 56 days",
    "missing_agreement": "Fewer than two model forecasts available",
    "missing_completeness": "Completeness could not be computed",
}


def _bands(bands=None) -> list[tuple[float, str]]:
    b = bands if bands is not None else C.TRUST_BANDS
    out = []
    for x in b:
        out.append((float(x["min"]), x["band"]) if isinstance(x, Mapping) else (float(x[0]), x[1]))
    return sorted(out, key=lambda t: -t[0])


# --------------------------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------------------------
def _series_wape(rows: pd.DataFrame) -> pd.Series:
    g = rows.assign(ae=(rows["actual"] - rows["forecast"]).abs()).groupby("series_id")[["ae", "actual"]].sum()
    return (g["ae"] / g["actual"].where(g["actual"] > 0) * 100).rename("wape_hist")


def components(bt: pd.DataFrame, panel: pd.DataFrame, champions: dict, origins: Iterable,
               quantiles_by_origin: Mapping, constants: Mapping | None = None) -> pd.DataFrame:
    """Trust components per (series_id, origin).

    bt                   backtest rows of all models (contract columns). Only variant A is used.
                         Rows needed: champion rows of earlier origins (accuracy/interval history) and
                         all models' rows AT each origin T (agreement, promo plan, forecast volume).
                         Actuals are read only for windows with o + 14 <= T.
    panel                processed panel; only rows with date <= T are read for origin T.
    champions            {family: {"model": ...}} (frozen selection).
    quantiles_by_origin  {origin: DataFrame[family, bucket, q_lo, q_hi]} - the interval quantiles
                         that were KNOWN at that origin (selection-fitted for calibration origins,
                         frozen calibration quantiles for holdout/forward origins).
    """
    k = {**CONSTANTS, **(constants or {})}
    H, L = int(k["horizon"]), int(k["lookback_days"])
    bt = bt[bt["variant"] == "A"].copy()
    bt["origin"] = pd.to_datetime(bt["origin"])
    champ = champion_rows(bt, champions)
    panel = panel.sort_values(["series_id", "date"])
    info = panel.drop_duplicates("series_id").set_index("series_id")[["store_nbr", "family"]]
    series = info.index
    qmap = {pd.Timestamp(o): q for o, q in quantiles_by_origin.items()}

    # 14-day rolling mean promo per series; value at date d uses d-13..d only (as-of safe).
    pw = int(k["promo_window"])
    promo_roll = panel[["series_id", "date"]].copy()
    promo_roll["roll"] = (panel.groupby("series_id")["onpromotion"]
                          .transform(lambda s: s.rolling(pw, min_periods=pw).mean()))

    out = []
    for T in pd.to_datetime(list(origins)):
        d = pd.DataFrame(index=series)
        d["origin"] = T

        # --- accuracy + interval on completed windows -------------------------------------
        done = sorted(o for o in champ["origin"].unique() if o + pd.Timedelta(days=H) <= T)
        done = done[-int(k["n_windows"]):]
        hist = evaluable(champ[champ["origin"].isin(done) & (champ["target_date"] <= T)])
        d["n_hist_windows"] = hist.groupby("series_id")["origin"].nunique().reindex(series).fillna(0).astype(int)
        d["wape_hist"] = _series_wape(hist).reindex(series)
        if len(hist) and T in qmap:
            iv = apply_intervals(hist, panel[panel["date"] <= T], qmap[T])
            inside = (iv["actual"] >= iv["p10"]) & (iv["actual"] <= iv["p90"])
            d["coverage_hist"] = (inside.groupby(iv["series_id"]).mean() * 100).reindex(series)
        else:
            d["coverage_hist"] = np.nan
        d["accuracy"] = 100 * (1 - np.minimum(d["wape_hist"], k["wape_cap"]) / k["wape_cap"])
        d["interval"] = 100 * (1 - np.minimum((d["coverage_hist"] - k["cov_target"]).abs(), k["cov_tol"]) / k["cov_tol"])

        # --- stability + completeness on the 56-day lookback ------------------------------
        w = panel[(panel["date"] > T - pd.Timedelta(days=L)) & (panel["date"] <= T)]
        ok = w[(w["closed"] == 0) & w["sales"].notna()]
        g = ok.groupby("series_id")["sales"]
        mean, std = g.mean().reindex(series), g.std(ddof=0).reindex(series)
        # all-zero lookback: CV undefined, set to 0 -> stability = 100 * 1 * (1 - 1) = 0
        d["cv_56"] = (std / mean.where(mean > 0)).where(mean != 0, 0.0)
        d["zero_share_56"] = g.apply(lambda s: float((s == 0).mean())).reindex(series)
        d["stability"] = 100 * (1 - np.minimum(d["cv_56"], k["cv_cap"]) / k["cv_cap"]) * (1 - d["zero_share_56"])
        d["completeness"] = 100 * ok.groupby("series_id").size().reindex(series).fillna(0) / L

        # --- agreement: dispersion of the models' 14-day totals at T ----------------------
        at = bt[(bt["origin"] == T) & bt["model"].isin(C.MODELS)]
        tot = at.groupby(["series_id", "model"])["forecast"].sum().unstack()
        n_mod = tot.notna().sum(axis=1)
        m_mean, m_std = tot.mean(axis=1), tot.std(axis=1, ddof=0)
        mcv = pd.Series(np.where(m_mean > 0, m_std / m_mean.where(m_mean > 0), 0.0), index=tot.index)
        mcv[n_mod < 2] = np.nan
        d["n_models"] = n_mod.reindex(series).fillna(0).astype(int)
        d["model_cv"] = mcv.reindex(series)
        d["agreement"] = 100 * (1 - np.minimum(d["model_cv"] / k["agree_cap"], 1))

        # --- champion window at T: forecast volume + promo plan ---------------------------
        cT = champ[champ["origin"] == T]
        d["forecast_14d"] = cT.groupby("series_id")["forecast"].sum().reindex(series)
        d["promo_plan_mean"] = cT.groupby("series_id")["onpromotion"].mean().reindex(series)
        pr = promo_roll[promo_roll["date"] <= T].groupby("series_id")["roll"]
        d["promo_p95"] = pr.quantile(k["promo_quantile"]).reindex(series)

        # --- hard flags ------------------------------------------------------------------
        window = pd.date_range(T + pd.Timedelta(days=1), periods=H)
        closure = bool(window.strftime("%m-%d").isin(k["closure_mmdd"]).any())
        flags = []
        for sid, r in d.iterrows():
            f = []
            if pd.notna(r["completeness"]) and r["completeness"] < k["completeness_flag"]:
                f.append("low_completeness")
            if closure:
                f.append("holiday_closure")
            if pd.notna(r["promo_plan_mean"]) and pd.notna(r["promo_p95"]) and r["promo_plan_mean"] > r["promo_p95"]:
                f.append("promo_above_p95")
            f += [f"missing_{c}" for c in COMPONENTS if pd.isna(r[c])]
            flags.append(f)
        d["hard_flags"] = flags
        out.append(d)

    res = pd.concat(out).rename_axis("series_id").reset_index()
    res = res.merge(info.reset_index(), on="series_id", how="left")
    for c in COMPONENTS:
        res[c] = res[c].clip(0, 100)
    lead = ["series_id", "store_nbr", "family", "origin"] + COMPONENTS + ["hard_flags"]
    return res[lead + [c for c in res.columns if c not in lead]]


# --------------------------------------------------------------------------------------------
# score, band, priority
# --------------------------------------------------------------------------------------------
def check_weights(weights: Mapping) -> dict:
    w = {c: float(weights.get(c, 0.0)) for c in COMPONENTS}
    extra = set(weights) - set(COMPONENTS)
    if extra:
        raise ValueError(f"unknown components {extra}")
    if any(v < 0 for v in w.values()):
        raise ValueError("weights must be non-negative")
    if abs(sum(w.values()) - 1) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {sum(w.values())}")
    return w


def score(comp: pd.DataFrame, weights: Mapping = C.TRUST_WEIGHTS_DEFAULT) -> pd.Series:
    """Weighted sum of the 0-100 components. A missing component counts as 0 (and carries a
    missing_* hard flag, so the band is Review anyway)."""
    w = check_weights(weights)
    s = sum(w[c] * comp[c].fillna(0.0) for c in COMPONENTS)
    return s.clip(0, 100).rename("score")


def band(score_value: float, flags=(), bands=None) -> str:
    if flags is not None and len(flags) > 0:
        return "Review"
    for lo, name in _bands(bands):
        if score_value >= lo:
            return name
    return "Review"


def bands_for(scores: pd.Series, flags: pd.Series | None = None, bands=None) -> pd.Series:
    flags = flags if flags is not None else pd.Series([[]] * len(scores), index=scores.index)
    return pd.Series([band(s, f, bands) for s, f in zip(scores, flags)], index=scores.index, name="band")


def priority(scores: pd.Series, forecast_14d: pd.Series, origin: pd.Series | None = None) -> pd.Series:
    """(100 - score) * the series' share of the total 14-day forecast (within each origin)."""
    f = forecast_14d.astype(float)
    tot = f.groupby(origin).transform("sum") if origin is not None else f.sum()
    return ((100 - scores) * f / tot).rename("priority")


def describe_flags(flags) -> list[str]:
    return [FLAG_TEXT.get(f, f) for f in flags]


# --------------------------------------------------------------------------------------------
# realised error, weight comparison, validation
# --------------------------------------------------------------------------------------------
def realised_window_wape(bt_champion: pd.DataFrame) -> pd.DataFrame:
    """Per (series_id, origin): champion WAPE over the 14-day window (open days with an actual)."""
    e = evaluable(bt_champion)
    g = e.assign(abs_error=(e["actual"] - e["forecast"]).abs()).groupby(["series_id", "origin"]).agg(
        sum_abs_error=("abs_error", "sum"), sum_actual=("actual", "sum"), n_days=("actual", "size")).reset_index()
    g["wape"] = g["sum_abs_error"] / g["sum_actual"].where(g["sum_actual"] > 0) * 100
    return g


def _spearman(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic if hasattr(r, "statistic") else r[0]), float(r.pvalue)


def grid_candidates(step: float = 0.1) -> dict[str, dict]:
    n = int(round(1 / step))
    out = {}
    for combo in itertools.product(range(n + 1), repeat=len(COMPONENTS) - 1):
        if sum(combo) > n:
            continue
        ws = list(combo) + [n - sum(combo)]
        w = {c: round(v * step, 10) for c, v in zip(COMPONENTS, ws)}
        out["grid:" + "/".join(f"{v / n:.1f}" for v in ws)] = w
    return out


def standard_candidates(step: float = 0.1) -> dict[str, dict]:
    eq = 1 / len(COMPONENTS)
    cands = {"default": dict(C.TRUST_WEIGHTS_DEFAULT),
             "equal": {c: eq for c in COMPONENTS},
             "volatility_only": {c: (1.0 if c == "stability" else 0.0) for c in COMPONENTS}}
    cands.update(grid_candidates(step))
    return cands


def compare_weights(comp: pd.DataFrame, realised: pd.DataFrame, candidates: Mapping[str, Mapping],
                    split_origins: Iterable | None = None) -> pd.DataFrame:
    """Spearman(score, realised window WAPE) per candidate over (series, origin) windows.
    Negative = higher trust goes with lower realised error (what we want).
    split_origins: origins of the first half; also reports rho on each half for a stability check."""
    j = comp.merge(realised[["series_id", "origin", "wape"]], on=["series_id", "origin"], how="inner")
    j = j[np.isfinite(j["wape"])]
    first = j["origin"].isin(pd.to_datetime(list(split_origins))) if split_origins is not None else None
    rows = []
    for name, w in candidates.items():
        s = score(j, w)
        rho, p = _spearman(s, j["wape"])
        r = {"name": name, "spearman": rho, "p_value": p, "n": int(len(j)), "weights": check_weights(w)}
        if first is not None:
            r["spearman_first_half"] = _spearman(s[first], j.loc[first, "wape"])[0]
            r["spearman_second_half"] = _spearman(s[~first], j.loc[~first, "wape"])[0]
        rows.append(r)
    return pd.DataFrame(rows)


def identifiable_candidates(cands: Mapping[str, Mapping], comp: pd.DataFrame,
                            default: Mapping = C.TRUST_WEIGHTS_DEFAULT) -> tuple[dict, list[str]]:
    """A component that is constant on the calibration windows only shifts every score by the same
    amount: Spearman cannot tell its weight apart. Its weight is not identifiable from calibration, so it
    is held at its default value: grid candidates that move it are dropped (named candidates are kept)."""
    const = [c for c in COMPONENTS if comp[c].nunique(dropna=True) <= 1]
    keep = {n: w for n, w in cands.items()
            if not n.startswith("grid:") or all(abs(w[c] - default[c]) < 1e-9 for c in const)}
    return keep, const


def choose_weights(table: pd.DataFrame, default: str = "default", margin: float = 0.03,
                   tie_tol: float = 1e-6) -> tuple[str, str]:
    """Pre-stated rule (written before results were seen, docs/agent_notes/trust.md):
    take the candidate with the most negative rho; adopt it instead of the default only if
    (1) its rho is at least `margin` more negative than the default's on all calibration windows and
    (2) it is also more negative than the default on EACH half of the calibration origins.
    Otherwise keep the default. Exact ties in rho (within tie_tol) are broken by the smallest
    Euclidean distance to the default weights. Returns (chosen name, reason)."""
    t = table.dropna(subset=["spearman"]).copy()
    dw = check_weights(table.set_index("name").loc[default, "weights"])
    t["_dist"] = t["weights"].apply(lambda w: float(np.sqrt(sum((w[c] - dw[c]) ** 2 for c in COMPONENTS))))
    t["_rank"] = (t["spearman"] / tie_tol).round()
    t = t.sort_values(["_rank", "_dist", "name"])
    d = table.set_index("name").loc[default]
    best = t.iloc[0]
    if best["name"] == default:
        return default, "default has the most negative rho"
    gain = d["spearman"] - best["spearman"]
    halves = ("spearman_first_half" in t and best["spearman_first_half"] < d["spearman_first_half"]
              and best["spearman_second_half"] < d["spearman_second_half"])
    if gain >= margin and halves:
        return best["name"], f"beats default by {gain:.3f} in rho and on both calibration halves"
    why = (f"best candidate {best['name']} beats default by only {gain:.3f} (< {margin})" if gain < margin
           else f"best candidate {best['name']} beats default by {gain:.3f} but not on both halves")
    return default, why + "; default kept"


def validate(scored: pd.DataFrame, realised: pd.DataFrame, vol_col: str = "stability") -> dict:
    """Holdout validation. scored: series_id, origin, score, band, <vol_col> (volatility-only score).
    realised: output of realised_window_wape. Fits the trust.json -> validation schema
    (shares in %, WAPE in %). Windows with zero actual volume (WAPE undefined) are dropped."""
    j = scored.merge(realised, on=["series_id", "origin"], how="inner")
    j = j[np.isfinite(j["wape"])]
    n = int(len(j))
    rho, p = _spearman(j["score"], j["wape"])
    rho_vol, p_vol = _spearman(j[vol_col], j["wape"])
    by_band, wb = [], {}
    for b in ["Accept", "Monitor", "Review"]:
        g = j[j["band"] == b]
        w = float(g["sum_abs_error"].sum() / g["sum_actual"].sum() * 100) if len(g) and g["sum_actual"].sum() > 0 else None
        wb[b] = w
        by_band.append({"band": b, "n": int(len(g)), "wape": None if w is None else round(w, 2)})
    ratio = wb["Review"] / wb["Accept"] if wb["Review"] is not None and wb["Accept"] else None
    tot_err = j["sum_abs_error"].sum()
    rev = j["band"] == "Review"
    err_share = float(j.loc[rev, "sum_abs_error"].sum() / tot_err * 100) if tot_err > 0 else None
    win_share = float(rev.mean() * 100) if n else None
    beats = bool(np.isfinite(rho) and np.isfinite(rho_vol) and rho < rho_vol)
    fin = lambda x, nd=3: None if x is None or not np.isfinite(x) else round(float(x), nd)

    # verdict: honest, plain English, never a probability
    if not np.isfinite(rho):
        verdict = "The Trust Index could not be evaluated on the holdout (too few windows with sales)."
    else:
        parts = []
        if rho <= -0.3 and p < 0.05:
            parts.append(f"The Trust Index ranked holdout windows well: higher scores went with lower realised error "
                         f"(Spearman {rho:.2f}).")
        elif rho < 0 and p < 0.05:
            parts.append(f"The Trust Index ranked holdout windows in the right direction but only weakly "
                         f"(Spearman {rho:.2f}).")
        else:
            parts.append(f"The Trust Index did not rank holdout windows reliably (Spearman {rho:.2f}, p = {p:.2g}); "
                         f"treat its bands as a rough guide only.")
        if ratio is not None:
            parts.append(f"Review windows had {ratio:.1f}x the WAPE of Accept windows.")
        elif wb["Accept"] is None or wb["Review"] is None:
            parts.append("At least one of the Accept / Review bands was empty, so the error ratio is undefined.")
        if err_share is not None and win_share is not None:
            parts.append(f"The Review band held {win_share:.0f}% of windows and {err_share:.0f}% of absolute error.")
        parts.append("It beat the volatility-only baseline." if beats else
                     f"It did NOT beat the volatility-only baseline (Spearman {rho_vol:.2f}), so the extra "
                     f"components added no ranking value on this holdout.")
        verdict = " ".join(parts)
    return {
        "n_windows": n, "spearman": fin(rho), "spearman_p_value": fin(p, 6),
        "wape_by_band": by_band,
        "error_ratio_review_vs_accept": fin(ratio, 2),
        "review_share_of_abs_error": fin(err_share, 1), "review_share_of_windows": fin(win_share, 1),
        "volatility_only_spearman": fin(rho_vol), "beats_volatility_baseline": beats, "verdict": verdict,
    }
