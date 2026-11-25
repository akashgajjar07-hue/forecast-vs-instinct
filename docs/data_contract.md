# Data contract

Machine-checkable schemas live in `src/schemas.py` (`SCHEMAS`, `SERIES_SCHEMA`). This page explains what each field
means. **Every number the dashboard shows comes from one of these files, and every file is produced by the pipeline
from `outputs/`.** The site never retrains a model.

General rules
- Dates are ISO `YYYY-MM-DD`. Missing or closed days are `null`, never 0.
- Percentages are percent values (`23.4` means 23.4 %), not fractions.
- Sales are **observed sales** (units, some families are fractional such as kg of produce), not unconstrained demand.
- **Data licence (Kaggle rule 7B).** Built on the Kaggle Store Sales – Time Series Forecasting competition data
  (Corporación Favorita). Raw competition data is not redistributed; the site shows only aggregated/indexed values and
  model outputs. Reproduce by downloading the data from Kaggle after accepting the competition rules.
- **Index scale.** Every per-series (one store × one family) sales-scale value is published as an index, never in units:
  `index = value ÷ base × 100`. The **series base** is the series' mean observed daily sales on open days over the
  training period before the first selection origin, 2015-06-01 to 2017-02-06 (`src/indexing.py`), so 100 = an
  average training-period day. A 14-day quantity is published as its mean daily value over the 14 days ÷ base × 100.
  The **chain base** (overview outlook) is the mean daily total of all 96 series over the same period (days when all
  96 were open). The **promo index** = promoted-item count ÷ the series' mean promoted-item count on open days from
  2015-06-01 up to the forward origin × 100 (`null` if that mean is 0). Bases are never published; converting an index
  back to units needs the competition data, which only someone who downloaded it can do locally.
- **Owner rule (round 2).** Per-series values appear only as % or on the index scale. Aggregates over series are
  **equal-weighted** means of the member series' indexes (family weekly index, the overview outlook, Scenario Lab
  baselines in index-days, weekday profiles), which carry no information about the bases. The only unit value published
  is the portfolio 14-day forecast total of all 96 series (`overview.kpis.forecast_14d_total`). MAE / RMSE
  (`arena.json`) and interval widths (`intervals.json`) are in index points. Holdout windows in the series files carry
  the champion forecast only. `tests/test_no_reconstruction.py` checks that the 96 bases are not identifiable from the
  published files. Field names `sales`, `actual`, `onpromotion`,
  `promo_total`, `mean_daily_sales`, `mean_promo_count`, `max_promo_count` (and the old unit fields `forecast_14d`,
  `p10_14d`, `p90_14d`, `stat_forecast`, `priority`) must not appear in `docs/data` (`tests/test_no_raw_data.py`).
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
| `overview.json` | Executive Overview | KPI cards, chain-level outlook (indexed history + forward forecast with band), top exceptions, ML-win count, findings |
| `series_index.json` | Explorer, Workspace, S&OP | the 96 series with store attributes and slug |
| `demand.json` | Demand Explorer | weekly totals per family (units, 12 stores summed), day-of-week profile, volatility map (cv, zero share), promo bins and promo-day share, holiday effect |
| `arena.json` | Model Arena | long table of metrics (stage × model × variant × slice), champion per family, promo variant A vs B |
| `intervals.json` | Workspace, Exceptions, Methodology | conformal quantiles, calibration and holdout coverage/width |
| `trust.json` | Exception Centre, Overview | weights, bands, latest trust score per series with components and hard flags, holdout validation |
| `scenario.json` | UAE Scenario Lab | real baseline per family, editable demo event assumptions, Ecuador analogue |
| `sop.json` | S&OP Demand Review | forward statistical forecast per series (index), trust band, deterministic summary, simulated FVA |
| `series/<slug>.json` | Forecast Workspace, Demand Explorer (lazy) | indexed daily and weekly history, holdout forecasts by origin (all models + band), forward forecast with promo index, promo what-if grid, trust; all on the index scale |

### Field notes
- `overview.kpis.forecast_14d_total`: sum of the champion forward forecast over all 96 series for the 14 days after the last actual (2017-08-16 → 08-29). There are no actuals for that window.
- `overview.kpis.holdout_wape` / `holdout_bias`: the champion-per-family system scored on the final holdout (ratio of sums over all series, horizons and origins).
- `overview.outlook`: `history[].mean_index` (equal-weighted mean of the 96 series' daily index, `null` unless all 96 were open) and `forecast[].mean_index` (same for the champion forecast). No band: the frozen chain-total band applies to the volume-weighted total; `chain_band_holdout_coverage` and `band_note` report its holdout coverage.
- `overview.top_exceptions[].priority_rank`: position in the trust inbox (1 = first).
- `demand.weekly.family_index[family][]`: equal-weighted family weekly index (plain mean of the 12 store series' weekly index). `demand.dow_profile`: equal-weighted weekday profile (mean 1). `demand.volatility[]`: `cv`, `cv_weekly`, `zero_share`, `n_open_days`. `demand.promo_frequency[]`: `promo_day_share` and `corr_promo_sales`. `demand.promo_bins[family][]`: `bin`, `n_days`, `mean_index` (store-day ÷ its store-family monthly mean), `n_stores`. `demand.holiday_effect[]`: `change_pct` variants and counts only.
- `arena.metrics[].mae / rmse`: index points (`mae_unit`): per row (actual − forecast) ÷ series base × 100, then mean (MAE) or root mean square (RMSE). WAPE and bias unchanged (ratios of sums in units).
- `intervals.holdout.mean_width` (and by family / horizon): mean of (P90 − P10) ÷ series base × 100 (index points); `mean_width_pct` = Σ(P90 − P10) ÷ Σ forecast × 100.
- `scenario.families[].baseline_14d`: index-days (`baseline_unit`) = 14 × the equal-weighted mean over the family's 12 series of the 14-day mean forecast index; 1,400 = 14 average training-period days for an average store. Scenario supply is entered in the same unit.
- `arena.metrics[].slice_type`: `overall | family | store | horizon | horizon_bucket (1-7, 8-14) | promo (promo, no_promo) | holiday (holiday, normal) | week_bucket (week1, week2; metrics on 7-day sums)`.
- `arena.metrics[].variant`: `A` known promo plan, `B` no future promo info. Baselines appear once with `A`.
- `arena.champions[family]`: chosen on **selection** backtests only (lowest WAPE, tie-break lower |bias|).
- `trust.latest[].forecast_14d_index / p10_14d_index / p90_14d_index`: the champion 14-day forecast and the frozen 14-day-total P10–P90 band (`intervals.json → window_band`), each as mean daily value over the 14 days ÷ series base × 100.
- `trust.latest[].priority_rank`: inbox order (1 = look first). Priority = (100 − score) × 14-day forecast volume share is computed in the pipeline; only the rank is published, because the share would reveal unit volumes.
- `sop.rows[]`: `stat_index`, `p10_index`, `p90_index` on the same 14-day index scale. The S&OP CSV export has columns `statistical_index, override_index, final_index`; converting them to units needs the series bases, i.e. the data owner's local copy of the competition data.
- `series/<slug>.json`: `scale` / `promo_scale` state the scales. `history.index` (daily from `history.start`, `null` = closed), `history.week_start` / `history.weekly_index` (Monday weeks from 2015-06-01: mean daily sales over the week's open days ÷ base × 100); `holdout[].rows[]`: `actual_index`, `p10_index`, `p90_index`, `forecast_index` (champion only); `forward.rows[]`: `forecast_index`, `p10_index`, `p90_index`, `promo_index`, `models_index{model}`.
- `trust.validation`: holdout-only numbers (Spearman of score vs realised window WAPE, etc.). `verdict` is a plain sentence that says honestly whether the index worked.
- `series.whatif.rows[].scale_pct`: −100, −50, 0, +50, +100 change to the planned promo count. Each row: `promo_index` (14-day mean planned promo, promo index), `forecast_index` (14-day mean forecast, index), `daily_index[14]`, `change_pct` (unchanged, scale-free). `hist_max_promo_index` / `hist_p95_promo_index` = the series' highest daily promo count and p95 of its 14-day rolling mean, as promo index. `extrapolation` = scaled plan beyond the series' historical p95 of the 14-day mean or its highest day. Wording on the site: "model-predicted change associated with", never "caused".
- `scenario.events[]`: UAE event % changes. These are **editable demonstration assumptions, not measured UAE uplift**.
- `sop.fva`: a **simulated, rule-based** override scored on holdout actuals. It is not human-planner performance.
