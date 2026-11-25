"""Leakage, completeness and reproducibility tests for src/features.py + src/models/lgbm.py.

CI tests use `synthetic_panel`; `@pytest.mark.data` tests check the real saved backtests."""
import functools

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.backtest import BACKTEST_COLUMNS, masked_view, run_origin
from src.features import FeatureBuilder, feature_names
from src.models.lgbm import fit_lightgbm, lightgbm_model, predict, promo_whatif

T = pd.Timestamp("2016-06-15")
H = 14
FAST = dict(n_estimators=30, num_leaves=15, min_data_in_leaf=20, num_threads=2)


def _perturb(panel, T, promo=True, seed=1):
    rng = np.random.default_rng(seed)
    p = panel.copy()
    after = p["date"] > T
    p.loc[after, "sales"] = rng.uniform(0, 1e5, after.sum())
    if promo:
        p.loc[after, "onpromotion"] = rng.integers(0, 500, after.sum())
    return p


def _model(variant):
    return functools.partial(lightgbm_model, variant=variant)


# (a) perturbing everything after T changes nothing at T
@pytest.mark.parametrize("variant", ["A", "B"])
def test_future_perturbation_masked_view_identical(synthetic_panel, variant):
    pert = _perturb(synthetic_panel, T, promo=(variant == "B"))
    f0 = FeatureBuilder(masked_view(synthetic_panel, T, variant), variant)
    f1 = FeatureBuilder(masked_view(pert, T, variant), variant)
    pd.testing.assert_frame_equal(f0.scoring_frame(T, H), f1.scoring_frame(T, H), check_exact=True)
    pd.testing.assert_frame_equal(f0.training_frame(T, H), f1.training_frame(T, H), check_exact=True)
    b0 = run_origin("lightgbm", _model(variant), synthetic_panel, T, "selection", variant, **FAST)
    b1 = run_origin("lightgbm", _model(variant), pert, T, "selection", variant, **FAST)
    np.testing.assert_array_equal(b0["forecast"].to_numpy(), b1["forecast"].to_numpy())


@pytest.mark.parametrize("variant", ["A", "B"])
def test_future_perturbation_unmasked_feature_functions(synthetic_panel, variant):
    """Feature code alone (no harness mask) must not read sales after T, nor (B) promo after T."""
    pert = _perturb(synthetic_panel, T, promo=(variant == "B"))
    f0, f1 = FeatureBuilder(synthetic_panel, variant), FeatureBuilder(pert, variant)
    pd.testing.assert_frame_equal(f0.origin_features([T]), f1.origin_features([T]), check_exact=True)
    pd.testing.assert_frame_equal(f0.scoring_frame(T, H), f1.scoring_frame(T, H), check_exact=True)
    pd.testing.assert_frame_equal(f0.training_frame(T, H), f1.training_frame(T, H), check_exact=True)
    # and identical to the masked-view features
    fm = FeatureBuilder(masked_view(synthetic_panel, T, variant), variant)
    pd.testing.assert_frame_equal(f0.scoring_frame(T, H), fm.scoring_frame(T, H), check_exact=True)


def test_variant_A_unmasked_promo_perturbation_changes_only_target_promo(synthetic_panel):
    pert = _perturb(synthetic_panel, T, promo=True)
    s0 = FeatureBuilder(synthetic_panel, "A").scoring_frame(T, H)
    s1 = FeatureBuilder(pert, "A").scoring_frame(T, H)
    diff = [c for c in s0.columns if not s0[c].equals(s1[c])]
    assert set(diff) <= {"promo_t", "promo_t_rel28"}


# (b) no feature row for origin T reads a date > T
@pytest.mark.parametrize("variant", ["A", "B"])
def test_truncated_panel_gives_identical_origin_features(synthetic_panel, variant):
    trunc = synthetic_panel[synthetic_panel["date"] <= T]
    origins = [T - pd.Timedelta(days=k) for k in (0, 1, 7, 30)]
    full_f, trunc_f = FeatureBuilder(synthetic_panel, variant), FeatureBuilder(trunc, variant)
    pd.testing.assert_frame_equal(full_f.origin_features(origins), trunc_f.origin_features(origins),
                                  check_exact=True)
    pd.testing.assert_frame_equal(full_f.training_frame(T, H), trunc_f.training_frame(T, H), check_exact=True)


def test_training_rows_respect_origin(synthetic_panel):
    tr = FeatureBuilder(masked_view(synthetic_panel, T, "A"), "A").training_frame(T, H)
    assert (tr["target_date"] <= T).all() and (tr["origin"] < T).all()
    assert ((tr["target_date"] - tr["origin"]).dt.days == tr["horizon"]).all()
    assert tr["y"].notna().all()


# (c) horizon completeness
@pytest.mark.parametrize("variant", ["A", "B"])
def test_horizon_completeness(synthetic_panel, variant):
    sc = FeatureBuilder(masked_view(synthetic_panel, T, variant), variant).scoring_frame(T, H)
    n = synthetic_panel["series_id"].nunique()
    assert len(sc) == n * H and not sc.duplicated(["series_id", "horizon"]).any()
    assert set(sc["horizon"]) == set(range(1, H + 1))
    bt = run_origin("lightgbm", _model(variant), synthetic_panel, T, "selection", variant, **FAST)
    assert len(bt) == n * H and bt["forecast"].notna().all()


# (d) variant B has no target-date promo
def test_variant_B_has_no_target_promo(synthetic_panel):
    names = feature_names("B")
    assert "promo_t" not in names and "promo_t_rel28" not in names
    sc = FeatureBuilder(masked_view(synthetic_panel, T, "B"), "B").scoring_frame(T, H)
    assert not {"promo_t", "promo_t_rel28"} & set(sc.columns)
    assert {"promo_t", "promo_t_rel28"} <= set(feature_names("A"))


# (e) reproducibility
@pytest.mark.parametrize("variant", ["A", "B"])
def test_reproducible(synthetic_panel, variant):
    b0 = run_origin("lightgbm", _model(variant), synthetic_panel, T, "selection", variant, **FAST)
    b1 = run_origin("lightgbm", _model(variant), synthetic_panel, T, "selection", variant, **FAST)
    np.testing.assert_array_equal(b0["forecast"].to_numpy(), b1["forecast"].to_numpy())


# (f) negative predictions clipped by the harness (and by predict())
def test_negative_predictions_clipped(synthetic_panel):
    def neg(view, origin, horizon, **kw):
        out = lightgbm_model(view, origin, horizon, **kw)
        return out.assign(forecast=out["forecast"] - 1e6)
    bt = run_origin("lightgbm", neg, synthetic_panel, T, "selection", "A", **FAST)
    assert (bt["forecast"] >= 0).all() and (bt["forecast"] == 0).all()
    fit = fit_lightgbm(masked_view(synthetic_panel, T, "A"), T, H, "A", objective="regression", **FAST)
    assert (predict(fit)["forecast"] >= 0).all()


def test_ratio_target_and_whatif(synthetic_panel):
    fit = fit_lightgbm(masked_view(synthetic_panel, T, "A"), T, H, "A", target="ratio",
                       objective="regression", **FAST)
    base = predict(fit)
    assert len(base) == synthetic_panel["series_id"].nunique() * H and base["forecast"].notna().all()
    pd.testing.assert_frame_equal(promo_whatif(fit, 0), base)
    assert len(promo_whatif(fit, 100)) == len(base)
    fitB = fit_lightgbm(masked_view(synthetic_panel, T, "B"), T, H, "B", **FAST)
    with pytest.raises(ValueError):
        promo_whatif(fitB, 50)
    assert set(fit.importance["gain"]) == set(fit.features)


# ---------- real saved backtests ----------
@pytest.mark.data
@pytest.mark.parametrize("variant", ["A", "B"])
def test_real_lightgbm_backtests(variant):
    from src.splits import load_split_calendar
    path = C.OUTPUTS / "backtests" / f"lightgbm_{variant}.parquet"
    if not path.exists():
        pytest.skip("run scripts/03_tune_lightgbm.py first")
    bt = pd.read_parquet(path)
    cal = load_split_calendar()
    assert list(bt.columns) == BACKTEST_COLUMNS
    assert set(bt["model"]) == {"lightgbm"} and set(bt["variant"]) == {variant}
    assert set(bt["stage"]) == {"selection", "calibration"}
    assert "holdout" not in set(bt["stage"])
    assert bt["origin"].max() < pd.Timestamp(cal["stages"]["holdout"][0])
    exp = set(pd.to_datetime(cal["stages"]["selection"] + cal["stages"]["calibration"]))
    assert set(bt["origin"]) == exp
    sizes = bt.groupby("origin").size()
    assert (sizes == 96 * 14).all()
    assert not bt.duplicated(["origin", "series_id", "horizon"]).any()
    assert bt["forecast"].notna().all() and (bt["forecast"] >= 0).all()
