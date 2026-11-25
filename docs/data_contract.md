# Data contract

Machine-checkable schemas live in `src/schemas.py` (`SCHEMAS`, `SERIES_SCHEMA`). This page explains what each field
means. **Every number the dashboard shows comes from one of these files, and every file is produced by the pipeline
from `outputs/`.** The site never retrains a model.

General rules
- Dates are ISO `YYYY-MM-DD`. Missing or closed days are `null`, never 0.
- Percentages are percent values (`23.4` means 23.4 %), not fractions.
- Sales are **observed sales** (units, some families are fractional such as kg of produce), not unconstrained demand.
- `family` uses the exact Kaggle labels: BEVERAGES, DAIRY, BREAD/BAKERY, GROCERY I, FROZEN FOODS, CLEANING, PERSONAL CARE, PRODUCE.
- `series_id` = `"<store_nbr>|<family>"`. `slug` = file-safe version used for `series/<slug>.json`
  (`<store_nbr>_<family lower, non-alphanumerics → '-'>`, e.g. `12_bread-bakery`).
- Model ids: `naive`, `seasonal_naive`, `moving_average`, `holt_winters`, `lightgbm`.
- Trust bands: `Accept` (80–100), `Monitor` (60–79), `Review` (0–59 or any hard flag).

## Backtest table (pipeline-internal, `outputs/backtests/*.parquet`, not published)

One row per model × promo variant × origin × series × horizon. Columns (`src/backtest.py::BACKTEST_COLUMNS`):
`model, variant (A|B), stage (selection|calibration|holdout|forward), origin, series_id, store_nbr, family, horizon (1–14),
target_date, forecast (≥0), actual (NaN if closed / future), closed, onpromotion (target date), is_holiday (target date)`.
Baselines don't use promo information, so their variant A and B rows are identical (stored once as `A`).
Evaluation population = `closed == 0 & actual not null`, identical for every model.

## Published files (`docs/data/`)

| File | Loaded by view | Contents |
|---|---|---|
| `meta.json` | all | citation, caveats, data window, split calendar, scope, 3 hero numbers, headline sentence |
| `overview.json` | Executive Overview | KPI cards, chain-level outlook (history + forward forecast with band), top exceptions, ML-win count, findings |
| `series_index.json` | Explorer, Workspace, S&OP | the 96 series with store attributes and slug |
| `demand.json` | Demand Explorer | weekly totals per family, day-of-week profile, volatility map, promo bins and frequency, holiday effect |
| `arena.json` | Model Arena | long table of metrics (stage × model × variant × slice), champion per family, promo variant A vs B |
| `intervals.json` | Workspace, Exceptions, Methodology | conformal quantiles, calibration and holdout coverage/width |
| `trust.json` | Exception Centre, Overview | weights, bands, latest trust score per series with components and hard flags, holdout validation |
| `scenario.json` | UAE Scenario Lab | real baseline per family, editable demo event assumptions, Ecuador analogue |
| `sop.json` | S&OP Demand Review | forward statistical forecast per series, trust band, deterministic summary, simulated FVA |
| `series/<slug>.json` | Forecast Workspace (lazy) | history, holdout forecasts by origin (all models + band), forward forecast, promo what-if grid, trust |

### Field notes
- `overview.kpis.forecast_14d_total`: sum of the champion forward forecast over all 96 series for the 14 days after the last actual (2017-08-16 → 08-29). There are no actuals for that window.
- `overview.kpis.holdout_wape` / `holdout_bias`: the champion-per-family system scored on the final holdout (ratio of sums over all series, horizons and origins).
- `overview.outlook.forecast[].p10/p90`: band for the **chain total**, computed with its own conformal quantiles on calibration residuals of the total. Summing series P10s would be wrong. `band_method` says how it was built.
- `arena.metrics[].slice_type`: `overall | family | store | horizon | horizon_bucket (1-7, 8-14) | promo (promo, no_promo) | holiday (holiday, normal) | week_bucket (week1, week2; metrics on 7-day sums)`.
- `arena.metrics[].variant`: `A` known promo plan, `B` no future promo info. Baselines appear once with `A`.
- `arena.champions[family]`: chosen on **selection** backtests only (lowest WAPE, tie-break lower |bias|).
- `trust.latest[].priority`: sort key for the inbox (higher = look first) = (100 − score) × 14-day forecast volume share.
- `trust.validation`: holdout-only numbers (Spearman of score vs realised window WAPE, etc.). `verdict` is a plain sentence that says honestly whether the index worked.
- `series.whatif.rows[].scale_pct`: −100, −50, 0, +50, +100 change to the planned promo count. `extrapolation` = scaled count beyond the series' historical p95 of the 14-day mean. Wording on the site: "model-predicted change associated with", never "caused".
- `scenario.events[]`: UAE event % changes. These are **editable demonstration assumptions, not measured UAE uplift**.
- `sop.fva`: a **simulated, rule-based** override scored on holdout actuals. It is not human-planner performance.
