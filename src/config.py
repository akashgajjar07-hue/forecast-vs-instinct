"""Project-wide constants and paths. Single source of truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CONFIG = ROOT / "config"
DOCS_DATA = ROOT / "docs" / "data"
REPORTS = ROOT / "reports"

SEED = 42

FAMILIES = [
    "BEVERAGES", "DAIRY", "BREAD/BAKERY", "GROCERY I",
    "FROZEN FOODS", "CLEANING", "PERSONAL CARE", "PRODUCE",
]
N_STORES = 12

# Modelling window start: PRODUCE has recording gaps (zero sales in ~55% of open
# store-days) until 2015-05-31. All families start here so every series shares one
# calendar. See docs/DECISIONS.md.
MODEL_START = "2015-06-01"
LAST_ACTUAL = "2017-08-15"
HORIZON = 14
ORIGIN_WEEKDAY = 1  # Tuesday (2017-08-01, the last holdout origin, is a Tuesday)

N_SELECTION = 10
N_CALIBRATION = 8
N_HOLDOUT = 6

# 2016-04-16 Ecuador earthquake (M7.8, Manabí). Flag the month after it.
QUAKE_START = "2016-04-16"
QUAKE_END = "2016-05-15"

INTERVAL_LEVEL = 0.80  # P10-P90
MODELS = ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"]

TRUST_WEIGHTS_DEFAULT = {
    "accuracy": 0.30, "interval": 0.25, "stability": 0.20, "agreement": 0.15, "completeness": 0.10,
}
TRUST_BANDS = [(80, "Accept"), (60, "Monitor"), (0, "Review")]
