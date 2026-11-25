"""Shared fixtures. `synthetic_panel` mimics data/processed/panel.parquet (same columns)
so model/feature/leakage tests run in CI without Kaggle data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANEL_PATH = ROOT / "data" / "processed" / "panel.parquet"


def make_synthetic_panel(n_days=240, start="2016-01-01", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    specs = [(1, "BEVERAGES", 500, "A", 1), (1, "PRODUCE", 200, "A", 1), (2, "BEVERAGES", 300, "C", 3),
             (2, "PRODUCE", 50, "C", 3)]
    dow = np.array([1.0, 0.9, 0.9, 0.95, 1.1, 1.4, 1.3])
    for store, fam, level, typ, cl in specs:
        promo = rng.poisson(5, n_days)
        y = level * dow[dates.dayofweek] * (1 + 0.02 * promo) * rng.lognormal(0, 0.1, n_days)
        df = pd.DataFrame({"store_nbr": store, "family": fam, "date": dates, "sales": y.round(1),
                           "onpromotion": promo, "closed": 0})
        rows.append(df)
    p = pd.concat(rows, ignore_index=True)
    xmas = p["date"].dt.strftime("%m-%d").isin(["12-25", "01-01"])
    p.loc[xmas, "closed"] = 1
    p.loc[xmas, "sales"] = np.nan
    p["is_holiday"] = p["date"].dt.strftime("%m-%d").isin(["01-01", "05-01"]).astype(int)
    p["is_workday"] = 0
    p["is_event"] = 0
    p["is_national_holiday"] = p["is_holiday"]
    p["city"] = np.where(p["store_nbr"] == 1, "Quito", "Guayaquil")
    p["state"] = np.where(p["store_nbr"] == 1, "Pichincha", "Guayas")
    p["type"] = np.where(p["store_nbr"] == 1, "A", "C")
    p["cluster"] = np.where(p["store_nbr"] == 1, 1, 3)
    p["payday"] = ((p["date"].dt.day == 15) | p["date"].dt.is_month_end).astype(int)
    p["quake"] = 0
    p["is_future"] = 0
    p["series_id"] = p["store_nbr"].astype(str) + "|" + p["family"]
    return p.sort_values(["series_id", "date"]).reset_index(drop=True)


@pytest.fixture
def synthetic_panel():
    return make_synthetic_panel()


@pytest.fixture(scope="session")
def real_panel():
    if not PANEL_PATH.exists():
        pytest.skip("real panel not built (needs Kaggle data)")
    return pd.read_parquet(PANEL_PATH)
