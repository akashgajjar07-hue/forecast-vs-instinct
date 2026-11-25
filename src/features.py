"""Leakage-safe features for the direct multi-horizon LightGBM.

Conventions
-----------
Origin t' = end of day t'. A feature row is (origin t', series s, horizon h) with target date d = t' + h.
Lags are named relative to the origin ("lag k" = the k-th most recent day ending at t'):
    lag_1  = sales[t']        lag_7  = sales[t' - 6]
    lag_14 = sales[t' - 13]   lag_28 = sales[t' - 27]
Rolling mean/std over 7/14/28/56 days are the windows (t'-w, t'], NaN (closed) days ignored.
Same-weekday features: o(h) = 7*ceil(h/7) - h is the offset (0..6) from t' back to the most recent day
<= t' that shares the target's weekday.  sw_last = sales[t' - o(h)],  sw_mean4 = nanmean of
sales[t' - o(h) - 7j], j = 0..3.

Purity: every "origin feature" for t' is a function of view rows with date <= t' only (rolling windows,
shifts and forward fills look strictly backwards).  "Target features" are the calendar of the target
date d (dow, month, payday, holiday flags, quake) plus, for promo variant A only, the planned
onpromotion at d.  Variant B never reads promo at d.  The external oil series (optional) is
forward-filled and read at t' (last known value <= t').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

WINDOWS = (7, 14, 28, 56)
LAGS = (1, 7, 14, 28)
BURN_IN = 56  # first training origin needs a full 56-day window
CAL_TARGET = ["is_holiday", "is_national_holiday", "is_event", "is_workday", "payday"]
CATEGORICALS = ["store_nbr", "family_code", "type_code", "cluster"]


def sw_offset(h) -> np.ndarray:
    h = np.asarray(h)
    return 7 * np.ceil(h / 7).astype(int) - h


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """out[i] = a[i-k] (k >= 0), NaN before the start. Looks strictly backwards."""
    if k == 0:
        return a.copy()
    out = np.full_like(a, np.nan, dtype=float)
    out[k:] = a[:-k]
    return out


def load_oil(path=None) -> pd.Series:
    """Daily WTI price indexed by date, raw (gaps kept). Forward-fill happens per origin in `_Panel`."""
    o = pd.read_csv(path or C.RAW / "oil.csv", parse_dates=["date"])
    return o.set_index("date")["dcoilwtico"]


@dataclass
class _Panel:
    dates: pd.DatetimeIndex
    series: list
    static: pd.DataFrame          # per series: store_nbr, family_code, type_code, cluster
    wide: dict                    # name -> (D, S) float array


def _to_wide(view: pd.DataFrame, oil: pd.Series | None = None) -> _Panel:
    v = view.sort_values(["series_id", "date"])
    dates = pd.date_range(v["date"].min(), v["date"].max(), freq="D")
    series = sorted(v["series_id"].unique())
    cols = ["sales", "onpromotion", "quake"] + CAL_TARGET
    wide = {}
    for c in cols:
        w = v.pivot(index="date", columns="series_id", values=c).reindex(index=dates, columns=series)
        wide[c] = w.to_numpy(dtype=float)
    st = v.drop_duplicates("series_id").set_index("series_id").loc[series]
    fams = sorted(view["family"].unique()) if len(set(view["family"]) - set(C.FAMILIES)) else C.FAMILIES
    static = pd.DataFrame({
        "store_nbr": st["store_nbr"].astype(int).to_numpy(),
        "family_code": [fams.index(f) for f in st["family"]],
        "type_code": [ord(str(t)[0]) - ord("A") for t in st["type"]],
        "cluster": st["cluster"].astype(int).to_numpy(),
    }, index=series)
    if oil is not None:
        # last known price <= day i (ffill only looks back). Price on a day is published end of day.
        o = oil.reindex(pd.date_range(min(oil.index.min(), dates[0]), dates[-1], freq="D")).ffill()
        wide["oil"] = np.repeat(o.reindex(dates).to_numpy(dtype=float)[:, None], len(series), axis=1)
    return _Panel(dates, series, static, wide)


def _origin_feature_arrays(P: _Panel, variant: str, use_oil: bool) -> dict:
    """(D, S) arrays; row i holds features for origin = dates[i], computed from rows <= i only."""
    s = pd.DataFrame(P.wide["sales"])
    f = {}
    for k in LAGS:
        f[f"lag_{k}"] = _shift(P.wide["sales"], k - 1)
    for w in WINDOWS:
        r = s.rolling(w, min_periods=1)
        f[f"roll_mean_{w}"] = r.mean().to_numpy()
        f[f"roll_std_{w}"] = s.rolling(w, min_periods=2).std().to_numpy()
    f["open_share_28"] = s.notna().rolling(28, min_periods=1).mean().to_numpy()
    pr = pd.DataFrame(P.wide["onpromotion"])
    f["promo_origin"] = P.wide["onpromotion"].copy()
    f["promo_roll_7"] = pr.rolling(7, min_periods=1).mean().to_numpy()
    f["promo_roll_28"] = pr.rolling(28, min_periods=1).mean().to_numpy()
    f["quake_share_28"] = pd.DataFrame(P.wide["quake"]).rolling(28, min_periods=1).mean().to_numpy()
    if use_oil:
        f["oil"] = P.wide["oil"].copy()
    return f


def _sameweekday_arrays(sales: np.ndarray) -> tuple[dict, dict]:
    last, mean4 = {}, {}
    for o in range(7):
        stack = np.stack([_shift(sales, o + 7 * j) for j in range(4)])
        last[o] = stack[0]
        with np.errstate(all="ignore"):
            cnt = (~np.isnan(stack)).sum(0)
            mean4[o] = np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)
    return last, mean4


def feature_names(variant: str = "A", use_oil: bool = False, quake: str = "flag") -> list[str]:
    names = ["horizon", "dow", "month", "dom"] + [f"{c}_t" for c in CAL_TARGET]
    if quake == "flag":
        names += ["quake_t", "quake_share_28"]
    names += CATEGORICALS
    names += [f"lag_{k}" for k in LAGS]
    for w in WINDOWS:
        names += [f"roll_mean_{w}", f"roll_std_{w}"]
    names += ["open_share_28", "sw_last", "sw_mean4", "promo_origin", "promo_roll_7", "promo_roll_28"]
    if variant == "A":
        names += ["promo_t", "promo_t_rel28"]
    if use_oil:
        names += ["oil"]
    return names


def _assemble(P: _Panel, F: dict, SW: tuple, oi: np.ndarray, hh: np.ndarray, si: np.ndarray,
              variant: str, use_oil: bool, quake: str) -> pd.DataFrame:
    """Vectorised gather. oi = origin row index, hh = horizon, si = series index (same length)."""
    ti = oi + hh  # target row index (calendar / promo-plan only)
    D = len(P.dates)
    ti_c = np.minimum(ti, D - 1)
    tdates = P.dates[ti_c]
    out = {"horizon": hh.astype(np.int32), "dow": tdates.dayofweek.to_numpy(), "month": tdates.month.to_numpy(),
           "dom": tdates.day.to_numpy()}
    for c in CAL_TARGET:
        out[f"{c}_t"] = P.wide[c][ti_c, si]
    if quake == "flag":
        out["quake_t"] = P.wide["quake"][ti_c, si]
    for c in CATEGORICALS:
        out[c] = P.static[c].to_numpy()[si]
    for name, arr in F.items():
        out[name] = arr[oi, si]
    o = sw_offset(hh)
    last, mean4 = SW
    out["sw_last"] = np.empty(len(oi))
    out["sw_mean4"] = np.empty(len(oi))
    for k in range(7):
        m = o == k
        out["sw_last"][m] = last[k][oi[m], si[m]]
        out["sw_mean4"][m] = mean4[k][oi[m], si[m]]
    if variant == "A":
        pt = P.wide["onpromotion"][ti_c, si]
        out["promo_t"] = pt
        out["promo_t_rel28"] = pt / (F["promo_roll_28"][oi, si] + 1.0)
    df = pd.DataFrame(out)
    df = df[feature_names(variant, use_oil, quake)]
    df.insert(0, "series_id", np.asarray(P.series)[si])
    df.insert(1, "origin", P.dates[oi])
    df.insert(2, "target_date", P.dates[ti_c])
    if (ti >= D).any():
        raise ValueError("target date beyond the view's calendar")
    return df


class FeatureBuilder:
    """Builds training and scoring frames for one forecast origin T from a masked view.

    Training rows: origins t' = first+BURN_IN-1 .. T-1 (every `origin_stride` days, anchored on T-1 so the
    most recent origins are always included), horizons 1..H with t'+h <= T and an observed (open-day) target.
    Scoring rows: origin T, every series x h = 1..H.
    """

    def __init__(self, view: pd.DataFrame, variant: str = "A", use_oil: bool = False, quake: str = "flag",
                 oil: pd.Series | None = None):
        if variant not in ("A", "B"):
            raise ValueError(variant)
        if quake not in ("flag", "drop"):
            raise ValueError(quake)
        self.variant, self.use_oil, self.quake = variant, use_oil, quake
        if use_oil and oil is None:
            oil = load_oil()
        self.P = _to_wide(view, oil if use_oil else None)
        self.F = _origin_feature_arrays(self.P, variant, use_oil)
        self.SW = _sameweekday_arrays(self.P.wide["sales"])

    def _idx(self, T) -> int:
        T = pd.Timestamp(T)
        i = self.P.dates.get_indexer([T])[0]
        if i < 0:
            raise KeyError(f"origin {T.date()} not in view")
        return int(i)

    def origin_features(self, origins) -> pd.DataFrame:
        """Per (origin, series) history features only (no target-date information)."""
        idx = np.array([self._idx(t) for t in origins])
        S = len(self.P.series)
        oi, si = np.repeat(idx, S), np.tile(np.arange(S), len(idx))
        df = pd.DataFrame({k: v[oi, si] for k, v in self.F.items()})
        last, mean4 = self.SW
        for k in range(7):
            df[f"sw_last_o{k}"] = last[k][oi, si]
            df[f"sw_mean4_o{k}"] = mean4[k][oi, si]
        df.insert(0, "series_id", np.asarray(self.P.series)[si])
        df.insert(1, "origin", self.P.dates[oi])
        return df

    def training_frame(self, T, horizon: int = C.HORIZON, origin_stride: int = 1) -> pd.DataFrame:
        iT = self._idx(T)
        first = BURN_IN - 1
        last_o = iT - 1
        if last_o < first:
            raise ValueError("not enough history before origin for training")
        oidx = np.arange(last_o, first - 1, -origin_stride)[::-1]
        S = len(self.P.series)
        H = np.arange(1, horizon + 1)
        oi, hh, si = (a.ravel() for a in np.meshgrid(oidx, H, np.arange(S), indexing="ij"))
        keep = oi + hh <= iT
        oi, hh, si = oi[keep], hh[keep], si[keep]
        y = self.P.wide["sales"][oi + hh, si]
        keep = ~np.isnan(y)
        if self.quake == "drop":
            keep &= self.P.wide["quake"][oi + hh, si] == 0
        oi, hh, si, y = oi[keep], hh[keep], si[keep], y[keep]
        df = _assemble(self.P, self.F, self.SW, oi, hh, si, self.variant, self.use_oil, self.quake)
        df["y"] = y
        return df

    def scoring_frame(self, T, horizon: int = C.HORIZON) -> pd.DataFrame:
        iT = self._idx(T)
        S = len(self.P.series)
        hh, si = (a.ravel() for a in np.meshgrid(np.arange(1, horizon + 1), np.arange(S), indexing="ij"))
        oi = np.full(len(hh), iT)
        return _assemble(self.P, self.F, self.SW, oi, hh, si, self.variant, self.use_oil, self.quake)


def build_origin_features(view, origins, variant="A", use_oil=False, quake="flag", oil=None) -> pd.DataFrame:
    return FeatureBuilder(view, variant, use_oil, quake, oil).origin_features(origins)


def build_training_frame(view, T, horizon=C.HORIZON, variant="A", origin_stride=1, **kw) -> pd.DataFrame:
    return FeatureBuilder(view, variant, **kw).training_frame(T, horizon, origin_stride)


def build_scoring_frame(view, T, horizon=C.HORIZON, variant="A", **kw) -> pd.DataFrame:
    return FeatureBuilder(view, variant, **kw).scoring_frame(T, horizon)
