# Model card: LightGBM champion (variant A, known promotion plan)

## Summary

| Item | Value |
|---|---|
| Model | One global LightGBM regressor (LightGBM 4.5.0), direct multi-horizon |
| Task | Forecast daily observed sales 1 to 14 days ahead for 96 store × family series |
| Champion in | All 8 families (chosen on selection backtests only; `outputs/selection.json`) |
| Holdout WAPE | **12.57 %** (best simple model, Holt-Winters: 17.23 %) |
| Holdout bias | −0.46 % |
| Interval | 80 % (P10–P90) scaled split-conformal band; holdout coverage 80.32 % |
| Data | Corporación Favorita, Ecuador (Kaggle). **Not UAE data.** |
| Target meaning | **Observed sales, not unconstrained demand** (no stock-out data) |

## Intended use

- **Intended:** a statistical baseline forecast for weekly demand review and replenishment discussion, at store × product-family level, 14 days ahead. It is meant to be read **with** its prediction interval and Trust Index band, and reviewed by a planner.
- **Intended users:** demand planners and analysts, and interviewers or reviewers assessing the method.
- **Out of scope:**
  - SKU-level ordering. The model forecasts family totals.
  - Any market other than the 12 Ecuadorian stores it was trained on, including the UAE, without retraining and revalidation.
  - Estimating causal promotion uplift or price elasticity. There is no price or discount data.
  - December peak planning. The holdout did not include December.
  - Automatic ordering without human review.

## Training data

- **Source:** Kaggle *Store Sales – Time Series Forecasting* (Alexis Cook, DanB, inversion, Ryan Holbrook, 2021). Raw data is not included in the repository.
- **Scope:** stores 12, 14, 27, 29, 31, 36, 38, 39, 49, 51, 53, 54 × families BEVERAGES, DAIRY, BREAD/BAKERY, GROCERY I, FROZEN FOODS, CLEANING, PERSONAL CARE, PRODUCE.
- **Window:** 2015-06-01 to the forecast origin (expanding window; the model is retrained at every origin). The last actual is 2017-08-15.
- **Training examples:** every historical origin t′ (from day 56 of the panel to T − 1) × series × horizon h with target date ≤ T. Closed target days are dropped. 825,888 rows at the last selection origin.
- **Known data issues:** closed days (25 Dec, 1 Jan) are NaN and excluded. PRODUCE's pre-June-2015 recording gap is avoided by the window start. The April–May 2016 earthquake period is kept with a flag. Sales are censored by stock-outs, which cannot be seen.

## Features

35 features in variant A (33 in variant B). Full definitions in [METHODOLOGY.md §4.2](METHODOLOGY.md#42-lightgbm-one-global-direct-multi-horizon-model-srcmodelslgbmpy-srcfeaturespy).

- **History at the origin:** lags 1/7/14/28; rolling mean and standard deviation over 7/14/28/56 days; share of open days; same-weekday last value and 4-week mean.
- **Target-date calendar (known in advance):** weekday, month, day of month, holiday / national holiday / event / work day / payday flags, earthquake flag.
- **Identity:** store, family, store type, cluster (categorical); horizon.
- **Promotion:** count at the origin and its 7- and 28-day means (both variants); **planned count on the target date and its ratio to the recent level (variant A only)**.
- **Not used:** transactions (unknown in the future), oil price (tested and made accuracy worse).

**Target:** sales ÷ (28-day mean at the origin + 1), L2 loss weighted by (28-day mean + 1). Predictions are multiplied back and clipped at 0.

**Top features by mean gain share** (variant A, mean over the 18 selection and calibration origins, `outputs/lgbm_feature_importance.json`): `dow` 19.10 %, `store_nbr` 13.87 %, `dom` 12.29 %, `promo_t` 7.96 %, `family_code` 7.61 %, `month` 6.89 %, `sw_mean4` 3.72 %, `promo_t_rel28` 3.63 %, `roll_mean_28` 2.77 %, `sw_last` 2.48 %. Gain describes how the model splits; it is not a cause-and-effect measure.

## Parameters

`objective=regression` (L2) on the ratio target, `num_leaves=127`, `learning_rate=0.05`, `n_estimators=400`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0`, earthquake as flag, no oil, training-origin stride 1, `num_threads=4`, `seed=42`, deterministic, `force_col_wise`. (`tweedie_variance_power=1.2` is stored but has no effect with L2.)

All choices were made on the 10 selection origins only. The tuning log is in `outputs/lgbm_tuning.json` and summarised in METHODOLOGY.md §4.3. A rerun gives bit-identical forecasts (checked on calibration origin 2017-06-13).

## Performance

Population: open days with an observed actual; every model scored on identical rows. WAPE and bias are ratio of sums. Observed sales, not unconstrained demand.

### By stage

| Stage | Origins | Rows | WAPE % | Bias % |
|---|---|---|---|---|
| Selection | 10 | 13,440 | 11.88 | −0.54 |
| Calibration | 8 | 10,752 | 12.65 | −2.39 |
| **Holdout** | 6 | 8,064 | **12.57** | **−0.46** |

Holdout MAE 208.5 units and RMSE 512.5 units per series-day.

### Holdout by family (champion vs best simple model)

| Family | LightGBM A WAPE % | Best simple model | Its WAPE % | LightGBM A bias % | Interval coverage % |
|---|---|---|---|---|---|
| BEVERAGES | 17.74 | Holt-Winters | 19.99 | −2.42 | 72.2 |
| DAIRY | 9.15 | Holt-Winters | 10.76 | +4.02 | 83.9 |
| BREAD/BAKERY | 10.92 | Holt-Winters | 11.98 | +0.35 | 75.1 |
| GROCERY I | 8.63 | Holt-Winters | 15.48 | +0.31 | 87.8 |
| FROZEN FOODS | 18.66 | Seasonal naive | 22.97 | +3.92 | 81.1 |
| CLEANING | 22.03 | Moving average | 28.90 | −6.31 | 75.9 |
| PERSONAL CARE | 13.88 | Holt-Winters | 23.27 | −0.75 | 86.4 |
| PRODUCE | 8.04 | Holt-Winters | 9.22 | +2.28 | 80.2 |

*Sources: `outputs/holdout_results.json`; bias and coverage from executed notebooks 02 and 03.*

### Other holdout slices

| Slice | LightGBM A WAPE % | Note |
|---|---|---|
| Horizon 1 | 10.1 | best of all models at all 14 horizons |
| Horizons 8–14 (mean) | 13.1 | |
| Weekly total, week 1 / week 2 | 7.42 / 9.39 | moving average second best at this level (8.53 / 10.67) |
| Holiday rows (176) | 17.13 | **moving average better (16.41)** |
| Normal rows (7,888) | 12.47 | |
| Promo days (7,412) / no-promo days (652) | 12.57 / 12.50 | no-promo group is small and mostly two families |
| Series won | 83 of 96 | Holt-Winters won 10, moving average 2, seasonal naive 1 |

### Variant B (no future promotion information)

Same model and parameters without `promo_t` and `promo_t_rel28`.

| Stage | Variant A WAPE % | Variant B WAPE % | B − A (pp) |
|---|---|---|---|
| Selection | 11.88 | 12.05 | +0.16 |
| Calibration | 12.65 | 12.91 | +0.26 |
| Holdout | 12.57 | 12.75 | +0.18 |

On the holdout, B was slightly better in FROZEN FOODS (−0.56 pp), BREAD/BAKERY (−0.23) and DAIRY (−0.12), and had a slightly lower RMSE (508.1 vs 512.5). The planned promotion count helps on average, but it is not the main source of the model's advantage.

## Limitations

- **Censored target.** Trained on observed sales. Stock-outs make true demand look lower, and the model will repeat that.
- **Family-level bias.** Overall bias is near zero, but family biases on the holdout ranged from −6.31 % (CLEANING) to +4.02 % (DAIRY). On calibration the model under-forecast by about 2.4 % overall; this was not re-tuned, because tuning on calibration would contaminate the intervals and Trust weights fitted there.
- **Holidays.** Weaker on holiday days, where a simple moving average did better on a small sample.
- **Series-level losses.** Simple models beat it on 13 of 96 series. The worst case was 54|BEVERAGES (40.95 % vs 35.23 % for Holt-Winters).
- **Season coverage.** The holdout is 6 summer origins (late June to mid August 2017). December, the peak for FROZEN FOODS (30.0 % of its in-window volume), was not tested.
- **Promotion response.** The what-if response to scaling the promotion plan is non-monotonic in many series and is an association only.
- **Intervals** are pooled over weekdays and under-cover Sundays.

## Ethical and usage considerations

- **No personal data.** The data is aggregated store × family daily sales. There are no customers, employees or prices.
- **Risk of misuse:** treating a forecast of observed sales as true demand can lock in past stock-outs (if a product was often out of stock, the forecast stays low, so fewer units are ordered, so it stays out of stock). Use it together with availability data when available.
- **Human in the loop:** the model is designed to be reviewed. The Trust Index directs attention, but it is a ranking index, not a probability, and it is dominated by past accuracy.
- **Transfer to other markets:** results do not transfer to the UAE or any other market without retraining and validation on local data. The UAE Scenario Lab uses hypothetical assumptions.
- **Data licence:** the raw data belongs to Kaggle / Corporación Favorita and is not redistributed.
