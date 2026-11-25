# Data dictionary

This page describes every data table in the project: the raw Kaggle files, the modelling panel, the backtest tables, the frozen output files and the JSON files the dashboard loads. The field-level contract for the dashboard files is in [docs/data_contract.md](../docs/data_contract.md); machine-checkable schemas are in `src/schemas.py`.

**General rules**
- Sales are **observed sales, not unconstrained demand** (no stock-out data). Units vary by family; some are fractional (for example, kilograms of produce).
- Closed or missing days are `NaN` / `null`, **never 0**.
- `onpromotion` is a **count of promoted items**, not a discount depth.
- `series_id` = `"<store_nbr>|<family>"`, for example `36|DAIRY`.

---

## 1. Raw Kaggle files (`data/raw/`, not committed)

Source: Alexis Cook, DanB, inversion, Ryan Holbrook (2021). *Store Sales – Time Series Forecasting*. Kaggle. Corporación Favorita, Ecuador.

| File | Rows | Columns | Used for |
|---|---|---|---|
| `train.csv` | 3,000,888 | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` | daily sales and promotion counts, 54 stores × 33 families, 2013-01-01 to 2017-08-15 |
| `test.csv` | 28,512 | `id`, `date`, `store_nbr`, `family`, `onpromotion` | **planned promotion counts only** for 2017-08-16 onwards (the forward forecast). No sales; never used for evaluation |
| `stores.csv` | 54 | `store_nbr`, `city`, `state`, `type`, `cluster` | store attributes; locale matching for holidays |
| `holidays_events.csv` | 350 | `date`, `type`, `locale`, `locale_name`, `description`, `transferred` | holiday, event and work-day flags |
| `oil.csv` | 1,218 | `date`, `dcoilwtico` | tested as a feature, dropped; EDA only |
| `transactions.csv` | 83,488 | `date`, `store_nbr`, `transactions` | EDA only (future values unknown at forecast time) |
| `sample_submission.csv` | 28,512 | `id`, `sales` | not used |

Key raw columns:

| Column | Meaning |
|---|---|
| `family` | product family label (exact Kaggle text, e.g. `BREAD/BAKERY`, `GROCERY I`) |
| `sales` | units sold that day (observed) |
| `onpromotion` | number of items in the family that were on promotion that day |
| `type` (stores) | store format, A to E |
| `cluster` (stores) | Favorita's grouping of similar stores, 1 to 17 |
| `type` (holidays) | Holiday, Transfer, Bridge, Additional, Work Day, Event |
| `locale` / `locale_name` | National / Regional (matched to store `state`) / Local (matched to store `city`) |
| `transferred` | `True` if the holiday was moved to another date (that date has a `Transfer` row) |

## 2. Modelling panel (`data/processed/panel.parquet`, not committed)

Built by `scripts/01_prepare_data.py` (`src/data.py::build_panel`). One row per store × family × date on a full daily calendar from 2015-06-01 to 2017-08-31: 96 series × 823 days = 79,008 rows.

| Column | Type | Meaning |
|---|---|---|
| `store_nbr` | int | store number |
| `family` | str | product family |
| `date` | date | calendar date |
| `sales` | float | observed sales; **NaN if closed** or in the future |
| `onpromotion` | int | promoted-item count (from `test.csv` for future rows) |
| `is_future` | 0/1 | 1 for 2017-08-16 to 2017-08-31 (1,536 rows): no sales, promotion plan only |
| `closed` | 0/1 | 1 if the date is missing for the store or the store sold nothing across all 33 families (in scope: 25 Dec and 1 Jan, 384 rows) |
| `is_holiday` | 0/1 | a day off for this store: Holiday (not transferred), Transfer, Bridge or Additional, matched by locale |
| `is_workday` | 0/1 | a weekend day made a working day |
| `is_event` | 0/1 | an Event row applies to this store |
| `is_national_holiday` | 0/1 | a national day off |
| `city`, `state`, `type`, `cluster` | | store attributes |
| `payday` | 0/1 | 15th or last day of the month |
| `quake` | 0/1 | 2016-04-16 to 2016-05-15 |
| `series_id` | str | `store_nbr|family` |

## 3. Configuration (`config/`, committed)

| File | Contents |
|---|---|
| `scope.json` | selection cutoff (2017-02-07), window start, the eligibility criteria, the 12 stores with city/state/type/cluster/first sale, the 8 families, excluded stores with reasons, per-series stats |
| `split_calendar.json` | origin definition, horizon 14, last actual 2017-08-15, model start, origins per stage, target window per stage |

## 4. Backtest table (`outputs/backtests/<model>.parquet`, regenerated, not committed)

One row per model × promotion variant × stage × origin × series × horizon. Columns (`src/backtest.py::BACKTEST_COLUMNS`):

| Column | Meaning |
|---|---|
| `model` | `naive`, `seasonal_naive`, `moving_average`, `holt_winters`, `lightgbm` |
| `variant` | `A` (known promotion plan) or `B` (no future promotion information). Baselines are stored once as `A` |
| `stage` | `selection`, `calibration`, `holdout`, `forward` |
| `origin` | forecast origin date T |
| `series_id`, `store_nbr`, `family` | series keys |
| `horizon` | 1 to 14 |
| `target_date` | T + horizon |
| `forecast` | forecast, ≥ 0 |
| `actual` | observed sales on the target date; NaN if closed or in the future |
| `closed` | 1 if the target date is a closure |
| `onpromotion` | promotion count on the target date |
| `is_holiday` | holiday flag on the target date |

**Evaluation population:** `closed == 0` and `actual` not null. It is identical for every model.

Holdout extras (`outputs/holdout/`, regenerated):

| File | Contents |
|---|---|
| `champion_intervals.parquet` | champion rows for holdout and forward origins, plus `scale` (28-day mean at T), `bucket` (1-7 / 8-14), `p10`, `p90` |
| `trust_scores.parquet` | one row per series × origin (calibration, holdout, forward): the five components, `hard_flags`, history inputs (`wape_hist`, `coverage_hist`, `cv_56`, `zero_share_56`, `model_cv`, `promo_plan_mean`, `promo_p95`), `forecast_14d`, `score`, `band`, `priority` |
| `chain_total.parquet` | daily 96-series total: forecast, actual, its own `p10` / `p90` |

## 5. Pipeline outputs (`outputs/`, committed JSON)

| File | Contents |
|---|---|
| `baselines_summary.json` | baseline selection and calibration metrics by family; moving-average window and Holt-Winters trend tuning; fallbacks |
| `lgbm_tuning.json` | every LightGBM tuning candidate with selection WAPE, bias, runtime; chosen parameters; final A/B metrics by stage and family |
| `lgbm_feature_importance.json` | mean gain and split share per feature, variants A and B |
| `selection.json` (+ `.sha256`) | **frozen**: champion per family, selection table, variant-B selection table, model parameters, rule |
| `intervals.json` (+ `.sha256`) | **frozen**: conformal method, 16 family × bucket quantiles, chain-total quantiles, calibration coverage (in-sample, leave-one-origin-out) |
| `trust_weights.json` (+ `.sha256`) | **frozen**: weights, bands, constants, choice rule and reason, interval quantile source |
| `trust_calibration.json` | Trust weight candidates and their Spearman values, grid size, single-component Spearman, band distribution, score and component summaries |
| `holdout_results.json` | leaderboard, champion system (WAPE, bias, MAE, RMSE, coverage, width), WAPE by family × model, Trust validation, band counts |
| `holdout_run.json` | one-time run log: timestamp, results digest, frozen hashes |
| `eda/demand.json`, `eda/ecuador_analogue.json`, `eda/eda_extras.json` | demand analysis results (history up to 2017-06-27 only) |

## 6. Published dashboard files (`docs/data/`)

Every number on the site comes from one of these files, produced by `scripts/08_export_dashboard.py` from `outputs/`. The site never retrains a model. Total size is kept under 3 MB. Field meanings: [docs/data_contract.md](../docs/data_contract.md).

| File | Loaded by view | Contents |
|---|---|---|
| `meta.json` | all | citation, caveats, data window, split calendar, scope, hero numbers, headline |
| `overview.json` | Executive Overview | KPI cards, chain outlook (history + forward forecast with its own band), top exceptions, ML-win count, findings |
| `series_index.json` | Explorer, Workspace, S&OP | the 96 series with store attributes and file slug |
| `demand.json` | Demand Explorer | weekly equal-weighted family index, day-of-week profile, volatility, promotion bins (index) and frequency, holiday effect (% change) |
| `arena.json` | Model Arena | metrics by stage × model × variant × slice (overall, family, store, horizon, horizon bucket, promo, holiday, week bucket), champions, variant A vs B |
| `intervals.json` | Workspace, Exceptions, Methodology | quantiles, calibration and holdout coverage and width |
| `trust.json` | Exception Centre, Overview | weights, bands, latest score per series with components and hard flags, holdout validation, calibration candidates |
| `scenario.json` | UAE Scenario Lab | real 14-day baseline per family in index-days (1,400 = 14 average training-period days for an average store); **editable demonstration event assumptions**; Ecuador holiday analogue |
| `sop.json` | S&OP Demand Review | forward forecast per series with band, summary, reason codes, **simulated** FVA |
| `series/<slug>.json` | Forecast Workspace | history, holdout forecasts by origin (all models + band), forward forecast, promotion what-if grid, trust |

`scope.json` in `docs/data/` is a copy of `config/scope.json`; no view loads it. `_fixtures/` holds synthetic placeholder data for development only and is not deployed.
