"""Metric formulas checked against hand-computed values (no Kaggle data needed)."""
import math
import numpy as np
import pandas as pd
import pytest

from src import metrics as M


def test_wape_hand_computed():
    # |10-8| + |20-25| + |0-1| = 2 + 5 + 1 = 8 ; sum A = 30 -> 26.666..%
    assert M.wape([10, 20, 0], [8, 25, 1]) == pytest.approx(8 / 30 * 100)


def test_bias_sign_positive_means_overforecast():
    # sum(F-A) = (8-10)+(25-20)+(1-0) = 4 ; /30 -> 13.33% over-forecast
    assert M.bias([10, 20, 0], [8, 25, 1]) == pytest.approx(4 / 30 * 100)
    assert M.bias([10], [5]) == pytest.approx(-50.0)


def test_mae_rmse():
    assert M.mae([10, 20, 0], [8, 25, 1]) == pytest.approx(8 / 3)
    assert M.rmse([10, 20, 0], [8, 25, 1]) == pytest.approx(math.sqrt((4 + 25 + 1) / 3))


def test_zero_denominator_returns_nan_with_flag():
    r = M.all_metrics([0, 0, 0], [1, 0, 2])
    assert math.isnan(r.wape) and math.isnan(r.bias)
    assert r.flag == M.ZERO_ACTUALS
    assert r.mae == pytest.approx(1.0)  # MAE still defined


def test_empty_input():
    r = M.all_metrics([], [])
    assert r.flag == M.EMPTY and math.isnan(r.wape) and math.isnan(r.mae)


def test_nan_pairs_dropped():
    assert M.wape([10, np.nan, 10], [5, 99, 10]) == pytest.approx(25.0)


def test_clip_negative_predictions():
    out = M.clip_forecast([-3.0, 0.0, 2.5, np.nan])
    assert out[0] == 0 and out[2] == 2.5 and np.isnan(out[3])


def test_coverage_and_width():
    a = [5, 10, 15, 20]
    lo = [4, 11, 10, 20]
    hi = [6, 12, 20, 25]
    assert M.coverage(a, lo, hi) == pytest.approx(75.0)  # inclusive bounds
    assert M.mean_width(lo, hi) == pytest.approx((2 + 1 + 10 + 5) / 4)


def test_grouped_is_ratio_of_sums():
    df = pd.DataFrame({"g": ["x", "x", "y"], "actual": [10, 30, 0], "forecast": [20, 30, 5]})
    out = M.grouped_metrics(df, "g").set_index("g")
    assert out.loc["x", "wape"] == pytest.approx(10 / 40 * 100)
    assert math.isnan(out.loc["y", "wape"]) and out.loc["y", "flag"] == M.ZERO_ACTUALS


def test_bucket_totals():
    df = pd.DataFrame({"s": 1, "horizon": range(1, 15), "actual": 1.0, "forecast": 2.0})
    b = M.bucket_totals(df, ["s"]).set_index("bucket")
    assert b.loc["week1", "actual"] == 7 and b.loc["week2", "forecast"] == 14
