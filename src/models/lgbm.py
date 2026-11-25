"""Direct multi-horizon global LightGBM (one model, `horizon` is a feature).

Harness entry point: `lightgbm_model(view, origin, horizon, variant="A", **params)`.
For promo what-ifs (variant A) use `fit_lightgbm` -> `LgbmFit`, then `predict(fit, frame)` on a copy of
`fit.scoring` with `promo_t` (and `promo_t_rel28`) rescaled, e.g. via `promo_whatif(fit, scale_pct)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config as C
from ..features import FeatureBuilder, CATEGORICALS, feature_names

# Frozen choice written by scripts/03_tune_lightgbm.py (selection origins only); see outputs/lgbm_tuning.json.
# Chosen on SELECTION origins only (outputs/lgbm_tuning.json -> chosen_params).
DEFAULT_PARAMS = dict(
    objective="regression", tweedie_variance_power=1.2, target="ratio",
    num_leaves=127, learning_rate=0.05, n_estimators=400, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    quake="flag", use_oil=False, origin_stride=1, num_threads=4,
)
FIXED = dict(seed=C.SEED, deterministic=True, force_col_wise=True, verbose=-1)
SCALE_COL = "roll_mean_28"
_MODEL_KEYS = {"objective", "tweedie_variance_power", "num_leaves", "learning_rate", "min_data_in_leaf",
               "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2", "num_threads", "max_bin"}


@dataclass
class LgbmFit:
    booster: lgb.Booster
    scoring: pd.DataFrame          # origin-T scoring frame (series_id, origin, target_date, features...)
    features: list
    params: dict
    variant: str
    n_train: int
    importance: dict = field(default_factory=dict)


def _scale(frame: pd.DataFrame) -> np.ndarray:
    s = frame[SCALE_COL].to_numpy(dtype=float)
    return np.where(np.isnan(s), 0.0, s) + 1.0


def fit_lightgbm(view: pd.DataFrame, origin, horizon: int = C.HORIZON, variant: str = "A", oil=None,
                 **params) -> LgbmFit:
    p = {**DEFAULT_PARAMS, **params}
    fb = FeatureBuilder(view, variant, use_oil=p["use_oil"], quake=p["quake"], oil=oil)
    tr = fb.training_frame(origin, horizon, origin_stride=p["origin_stride"])
    sc = fb.scoring_frame(origin, horizon)
    feats = feature_names(variant, p["use_oil"], p["quake"])
    y = tr["y"].to_numpy(dtype=float)
    weight = None
    if p["target"] == "ratio":           # scale-free: y / (rolling-28 mean + 1), weighted by the scale
        sc_tr = _scale(tr)
        y, weight = y / sc_tr, sc_tr
    elif p["target"] != "raw":
        raise ValueError(p["target"])
    lp = {k: v for k, v in p.items() if k in _MODEL_KEYS}
    if lp.get("objective") != "tweedie":
        lp.pop("tweedie_variance_power", None)
    lp.update(FIXED)
    ds = lgb.Dataset(tr[feats], label=y, weight=weight, categorical_feature=CATEGORICALS, free_raw_data=True)
    booster = lgb.train(lp, ds, num_boost_round=int(p["n_estimators"]))
    imp = {"gain": dict(zip(feats, booster.feature_importance("gain").tolist())),
           "split": dict(zip(feats, booster.feature_importance("split").astype(int).tolist()))}
    return LgbmFit(booster, sc, feats, p, variant, len(tr), imp)


def predict(fit: LgbmFit, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    f = fit.scoring if frame is None else frame
    pred = fit.booster.predict(f[fit.features], num_threads=fit.params.get("num_threads", 4))
    if fit.params["target"] == "ratio":
        pred = pred * _scale(f)
    return pd.DataFrame({"series_id": f["series_id"].to_numpy(), "horizon": f["horizon"].to_numpy(),
                         "target_date": f["target_date"].to_numpy(), "forecast": np.maximum(pred, 0.0)})


def promo_whatif(fit: LgbmFit, scale_pct: float) -> pd.DataFrame:
    """Re-score with planned target-date promo counts scaled by (1 + scale_pct/100). Variant A only.
    Model-predicted association, not a causal effect."""
    if fit.variant != "A":
        raise ValueError("promo what-if needs variant A (target-date promo is a feature)")
    f = fit.scoring.copy()
    f["promo_t"] = f["promo_t"] * (1 + scale_pct / 100.0)
    f["promo_t_rel28"] = f["promo_t"] / (f["promo_roll_28"] + 1.0)
    return predict(fit, f)


def lightgbm_model(view: pd.DataFrame, origin, horizon: int = C.HORIZON, variant: str = "A", **params):
    fit = fit_lightgbm(view, origin, horizon, variant, **params)
    return predict(fit)[["series_id", "horizon", "forecast"]]
