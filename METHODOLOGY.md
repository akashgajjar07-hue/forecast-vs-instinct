# Methodology

This document explains how every result in the project was produced, in the order the pipeline runs. Formulas are exact. Numbers come from the files named next to them. Judgement calls are logged with reasons in [docs/DECISIONS.md](docs/DECISIONS.md).

Two rules apply everywhere:

- **Sales are observed sales, not unconstrained demand.** The data has no stock-out information. When a shelf was empty, recorded sales were low, and the models learn that as if it were low demand.
- **`onpromotion` is the number of promoted items** in a store-family on a day. There is no price or discount depth, so every promotion result is an association, not a causal uplift.

---

## 1. Data

### 1.1 Source

Kaggle *Store Sales – Time Series Forecasting* (Alexis Cook, DanB, inversion, Ryan Holbrook, 2021): daily sales of Corporación Favorita, an Ecuadorian grocery retailer, for 54 stores and 33 product families from 2013-01-01 to 2017-08-15. Files used: `train.csv`, `test.csv` (planned promotions only), `stores.csv`, `holidays_events.csv`; `oil.csv` and `transactions.csv` were used in analysis only (see 1.3). Column definitions are in [data/data_dictionary.md](data/data_dictionary.md).

### 1.2 The modelling panel

`src/data.py::build_panel` builds one row per store × family × date on a **full daily calendar** from 2015-06-01 to 2017-08-31: 96 series × 823 days = 79,008 rows. Rows after 2017-08-15 (1,536 rows) are "future" rows with no sales and only the planned promotion count from `test.csv`.

### 1.3 Data traps and how each was handled

| Trap | What we found | What we did |
|---|---|---|
| **Store closures** | 25 December is missing from `train.csv`. On 1 January the stores sold nothing. | A day is **closed** if the date is missing for the store, or the store sold nothing across all 33 families. Sales on closed days are set to **NaN, not 0**, and every metric excludes them. In scope, only 25 Dec and 1 Jan are closures (384 panel rows). A closure is not zero demand. |
| **PRODUCE recording gap** | Before June 2015 PRODUCE shows zero sales on about half of open store-days (48.3 % in the EDA query): a recording gap, not real demand. From 2015-06-01 the zero share is 0. | The modelling window starts on **2015-06-01** for every family, so all 96 series share one calendar and still have about 20 months of history before the first origin. |
| **Earthquake** | The M7.8 earthquake in Manabí on 2016-04-16 was followed by a temporary jump in sales across the chain. In the two Manabí stores (53 Manta, 54 El Carmen) the increase lasted. | A `quake` flag marks 2016-04-16 to 2016-05-15. LightGBM uses it (`quake_t`, `quake_share_28`). Dropping those training rows was tested and was worse (section 4.3). |
| **Payday** | Public-sector wages in Ecuador are paid on the 15th and the last day of the month. | `payday` = 1 on the 15th or the last day of the month. LightGBM also gets `dom` (day of month), which carries the start-of-month effect. |
| **Holidays** | `holidays_events.csv` mixes several types and locales. | See 1.4. |
| **Transactions** | Store transactions track sales closely (pooled daily r = 0.897). | **Not a feature.** Future transactions are unknown at the forecast date. EDA only. |
| **Oil price** | Ecuador's economy depends on oil; a daily oil price is provided. | **Tested and dropped.** Adding lagged oil made selection WAPE worse (12.27 % vs 11.91 %). |

### 1.4 Holiday handling

From `src/data.py::holiday_flags`:

- **Day off** (`is_holiday` = 1): `type == Holiday` and **not transferred**; or `Transfer` (the day a holiday moved to); or `Bridge` (an extra day off joining a holiday to a weekend); or `Additional` (extra days around a holiday, for example the Christmas run-up).
- A holiday with `transferred == True` is **not** a day off on its original date. Its `Transfer` row is.
- **Work Day** (`is_workday`): a weekend day made a working day to make up for a Bridge.
- **Event** (`is_event`): national events such as Mother's Day, Black Friday, the earthquake aftermath and World Cup days.
- **Locale matching:** National rows apply to all stores; Regional rows apply when `locale_name` equals the store's **state**; Local rows apply when `locale_name` equals the store's **city**.
- `is_national_holiday` marks national days off only.

---

## 2. Scope selection

`src/scope.py`, result in `config/scope.json`. Only data **before the first selection origin (2017-02-07)** was used, so the scope cannot be tuned on results.

Eligibility:
1. Store open for at least 56 days before the modelling window start (2015-06-01).
2. No unexplained closure days in the window (the chain-wide 1 Jan and 25 Dec closures are allowed).
3. No zero-sales days on open days for any of the 8 families.

Choice among eligible stores: a quota per store type, **A2 / B2 / C3 / D3 / E2**; within a type, prefer an unused cluster, then an unused city, then volume closest to the type median. The result is stores **12, 14, 27, 29, 31, 36, 38, 39, 49, 51, 53, 54**, covering 11 cities and 10 clusters. All 8 families are kept.

---

## 3. Split calendar and leakage controls

### 3.1 Rolling-origin backtest

A **forecast origin T** is the end of day T. Everything up to and including T is known; we forecast T+1 to T+14. Origins are weekly, on **Tuesdays** (2017-08-01, the last origin whose 14-day window ends on or before the last actual 2017-08-15, is a Tuesday). From `config/split_calendar.json`:

| Stage | Job | Origins | Target dates |
|---|---|---|---|
| **Selection** | pick and tune models, choose champions | 10: 2017-02-07, 02-14, 02-21, 02-28, 03-07, 03-14, 03-21, 03-28, 04-04, 04-11 | 2017-02-08 to 2017-04-25 |
| **Calibration** | fit interval quantiles and Trust Index weights | 8: 2017-04-25, 05-02, 05-09, 05-16, 05-23, 05-30, 06-06, 06-13 | 2017-04-26 to 2017-06-27 |
| **Holdout** | final exam, scored once | 6: 2017-06-27, 07-04, 07-11, 07-18, 07-25, 08-01 | 2017-06-28 to 2017-08-15 |
| **Forward** | the "next 14 days" plan | 1: 2017-08-15 | 2017-08-16 to 2017-08-29 (no actuals) |

Within a stage, windows overlap (weekly origins, 14-day horizon). **Between stages, target windows never overlap**: the last selection target (25 April) is before the first calibration target (26 April), and the last calibration target (27 June) is before the first holdout target (28 June). A test checks this.

Every model is retrained at every origin on all history up to that origin (an expanding window).

### 3.2 Leakage controls

"Leakage" means a model uses information it would not have had on the forecast date. Five controls prevent it:

1. **Masked view** (`src/backtest.py::masked_view`). A model never sees the raw panel. It gets a copy where every sale after T is NaN. In promotion variant B, the promotion count after T is NaN too. In variant A it is kept, because a retailer knows its agreed promotion calendar. This makes look-ahead structurally impossible rather than relying on careful feature code.
2. **Self-filtering models.** Each baseline also filters its input to `date <= T` itself, so it stays leak-free even if handed an unmasked panel (tested).
3. **Origin-relative features.** Every history feature is measured backwards from the origin, never from the target date. Target-date features are only things known in advance: the calendar and, in variant A, the planned promotion count.
4. **Freeze hashes and the holdout guard** (`src/freeze.py`). `selection.json`, `intervals.json` and `trust_weights.json` are written with a SHA-256 hash file. `scripts/07_holdout_and_forward.py` verifies all three before running, refuses if any changed, and refuses a second run if `outputs/holdout_run.json` exists (except a `--rerun` reproducibility check, which must reproduce the logged digest and never overwrites the log).
5. **Truncation.** Scripts that tune or calibrate cut the panel at 2017-06-27 (the last calibration target) before anything runs, and assert that no holdout or forward rows are in memory.

The leakage tests are listed in [reports/VALIDATION_REPORT.md](reports/VALIDATION_REPORT.md). A deliberate one-day look-ahead injected into `lag_1` made 5 of them fail.

---

## 4. Models

All models forecast 14 daily values per series per origin. Forecasts are clipped at 0.

### 4.1 Baselines (`src/models/baselines.py`)

| Model | Rule | Tuning (selection only) |
|---|---|---|
| **Naive** | every day of the next 14 = the last observed sale (closed origin day skipped) | none |
| **Seasonal naive** | day T+h = sales on T+h − 7·ceil(h/7), the same weekday in the most recent week ≤ T. If that day was closed, step back another 7 days | none |
| **Moving average** | every day = mean of the last *w* observed days (closed days skipped, not counted as 0) | *w* ∈ {7, 14, 28, 56}: WAPE 20.778 / 20.248 / **19.683** / 20.677 → **w = 28** |
| **Holt-Winters** | additive level + additive weekly seasonality (period 7) on raw sales, fitted per series on the trailing 364 days (`statsmodels`, `initialization_method="estimated"`). Closed days in the fit window are filled with the value 7 days earlier, for fitting only | trend: none → 14.726; additive damped → 14.924 → **no trend**. If a fit fails, fall back to seasonal naive: 0 fallbacks in 1,728 fits |

*Source: `outputs/baselines_summary.json`.*

### 4.2 LightGBM: one global, direct multi-horizon model (`src/models/lgbm.py`, `src/features.py`)

**Global.** One model is trained across all 96 series, with store and family as categorical features. A quiet series borrows strength from busy ones.

**Direct, not recursive.** A *recursive* model forecasts day 1, feeds that guess back in as a lag, forecasts day 2, and so on, so early errors compound. Here `horizon` (1–14) is a feature, and the model is trained on real pairs "history at origin t′ → actual sale on t′ + h". Every horizon is predicted from the same observed history at T. Nothing has to be simulated forward, and the day-14 forecast is learned from real day-14 examples.

**Training examples.** For a backtest origin T, each example is a historical origin t′, a series and a horizon h with target date d = t′ + h ≤ T. t′ runs from day 56 of the panel (so the 56-day windows are full) to T − 1, every day (stride 1). Rows with a closed target are dropped. At the last selection origin there are 825,888 training rows.

**Scale-free target.** The model predicts y ÷ (roll_mean_28 + 1), where roll_mean_28 is the series' mean sales over the 28 days up to the origin. Training uses L2 loss with sample weight roll_mean_28 + 1, so the loss tracks errors in units. Predictions are multiplied back by the same scale and clipped at 0. This lets one model serve stores that differ 10× in size: the trees learn **shape** (weekday, promotion, holiday) and the level comes from recent history.

**Features** (origin t′ is the end of day t′; d = t′ + h is the target date; "lag 1" is day t′ itself):

| Feature | Definition |
|---|---|
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | sales on t′, t′−6, t′−13, t′−27 |
| `roll_mean_w`, `roll_std_w` (w = 7, 14, 28, 56) | mean and standard deviation of sales over (t′−w, t′]; closed days ignored; mean needs ≥ 1 value, std ≥ 2 |
| `open_share_28` | share of the last 28 days ≤ t′ with an observed sale |
| `sw_last` | sales on t′ − o(h), where o(h) = 7·ceil(h/7) − h: the most recent day ≤ t′ on the target's weekday |
| `sw_mean4` | mean of sales on t′ − o(h) − 7j for j = 0..3 (closed days ignored) |
| `dow`, `month`, `dom` | weekday, month and day of month of d |
| `is_holiday_t`, `is_national_holiday_t`, `is_event_t`, `is_workday_t`, `payday_t` | holiday, event, work-day and payday flags of d for that store |
| `quake_t` | earthquake flag of d |
| `quake_share_28` | share of the last 28 days ≤ t′ inside the earthquake window |
| `store_nbr`, `family_code`, `type_code`, `cluster` | categorical features with fixed codes |
| `horizon` | h |
| `promo_origin`, `promo_roll_7`, `promo_roll_28` | promotion count on t′ and its 7- and 28-day rolling means ending at t′ (both variants) |
| `promo_t` (variant A only) | planned promotion count on d |
| `promo_t_rel28` (variant A only) | promo_t ÷ (promo_roll_28 + 1) |

Variant A has 35 features and variant B 33; the only difference is `promo_t` and `promo_t_rel28`.

### 4.3 LightGBM tuning (selection origins only)

Tuning used variant A, all 10 selection origins and a training-origin stride of 2 to save time. Candidates were ranked by overall selection WAPE rounded to 2 dp, ties to smaller |bias|. Each stage kept the previous stage's winner. From `outputs/lgbm_tuning.json`:

| Stage | Candidate | Selection WAPE % | Bias % |
|---|---|---|---|
| objective | Tweedie p = 1.2 | 12.2985 | −1.6906 |
| objective | Tweedie p = 1.5 | 12.3016 | −1.7694 |
| objective | L2 on raw sales | 12.4739 | −1.4840 |
| objective | L1 on raw sales | 12.2828 | −2.1898 |
| objective | **L2 on ratio to 28-day mean** | **12.0392** | −0.6317 |
| structure | + 31 leaves | 12.0858 | −0.7074 |
| structure | **+ 127 leaves** | **11.9144** | −0.6151 |
| structure | + min_data_in_leaf 30 | 12.0611 | −0.6295 |
| structure | + min_data_in_leaf 300 | 12.0460 | −0.6608 |
| structure | + lr 0.03, 800 rounds | 12.0169 | −0.7124 |
| structure | + lr 0.1, 200 rounds | 12.0831 | −0.6043 |
| earthquake | drop quake-window targets (instead of flag) | 11.9423 | −1.0556 |
| oil | + lagged oil price | 12.2724 | −0.2110 |
| stride | **training origins every day (stride 1)** | **11.8836** | −0.5355 |

Final parameters: objective `regression` (L2) on the ratio target, num_leaves 127, learning_rate 0.05, 400 rounds, min_data_in_leaf 100, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0, earthquake as a flag, no oil, stride 1, 4 threads, seed 42, deterministic. Variant B reuses these parameters without separate tuning.

### 4.4 Promotion variants A and B

| Variant | Promotion information | Business meaning |
|---|---|---|
| **A** | promotion history up to T **plus the planned count on each target date** | the planner has an agreed promotion calendar |
| **B** | promotion history up to T only | no promotion plan is available |

Baselines use no promotion information, so they are stored once (as variant A).

### 4.5 Champion rule (`src/selection.py`, frozen in `outputs/selection.json`)

Per family: the model with the **lowest selection WAPE** (ratio of sums over 10 origins × 12 stores × 14 days = 1,680 rows) is champion. A tie within 0.05 pp goes to the smaller |bias|. LightGBM competes as variant A. Result: **LightGBM is champion in all 8 families**, with selection margins over the runner-up from 0.63 pp (PRODUCE) to 9.03 pp (PERSONAL CARE).

---

## 5. Metrics (`src/metrics.py`)

A = actual, F = forecast, sums over the rows being scored:

| Metric | Formula | Read as |
|---|---|---|
| **WAPE** | Σ\|A − F\| ÷ Σ\|A\| × 100 | % of volume mis-forecast; lower is better; the headline metric |
| **Bias** | Σ(F − A) ÷ ΣA × 100 | positive = over-forecast, negative = under-forecast |
| **MAE** | mean \|A − F\| | average miss per series-day, in units |
| **RMSE** | √ mean (A − F)² | like MAE but punishes large misses more |
| **Coverage** | share of rows with P10 ≤ A ≤ P90, × 100 | target 80 % |
| **Mean width** | mean (P90 − P10) | in units per series-day |

Rules:
- WAPE and bias are always **ratio of sums** within the group, never an average of row-level percentages.
- **Evaluation population:** open days with an observed actual (`src/backtest.py::evaluable`: `closed == 0` and actual not null). Every model is scored on the identical set of rows (8,064 on the holdout).
- **Zero denominators:** if Σ|A| = 0 (or the group is empty), WAPE and bias return NaN with a flag (`zero_actuals` or `empty`), never infinity and never an error.
- **Week buckets:** for each series × origin, the daily forecasts and actuals of horizons 1–7 are summed into a "week 1" total, and horizons 8–14 into "week 2" (`metrics.bucket_totals`). WAPE, bias, MAE and RMSE are then computed on those totals. This is closer to how planners are measured, because daily errors within a week partly cancel.

---

## 6. Prediction intervals: scaled split-conformal (`src/intervals.py`, frozen in `outputs/intervals.json`)

The idea: look at how wrong the champion was on the calibration windows and assume the future will be wrong in a similar way.

1. For each calibration row (8 origins), compute a **scaled residual** r = (A − F) ÷ scale_T, where scale_T is the series' mean sales over the 28 days up to the origin (known at T). Scaling lets one quantile serve a large and a small store.
2. Group residuals by **family × horizon bucket** (days 1–7, days 8–14): 16 groups of 672 residuals each (8 origins × 12 stores × 7 days).
3. In each group take the 10th and 90th percentiles, q_lo and q_hi.
4. For a new forecast F: **P10 = max(0, F + q_lo × scale_T)** and **P90 = max(0, F + q_hi × scale_T)**, with the point forecast kept inside its own band.

Coverage was estimated three ways: in-sample on calibration (79.76 %), **leave-one-origin-out** on calibration (refit 8 times, each time scoring the left-out origin: 79.20 %), and on the holdout (80.32 %).

**Chain total band.** Adding 96 series' P10 values does not give the P10 of the sum, because errors partly cancel. The chain total therefore gets its own band: relative residuals (A − F) ÷ F of the 96-series daily total on the 8 calibration origins, 10th and 90th percentiles per bucket (days 1–7: −5.2 % / +11.2 %; days 8–14: −5.3 % / +10.1 %).

The optional comparison with a LightGBM quantile objective was not run (DECISIONS #16).

---

## 7. Forecast Trust Index (`src/trust.py`, frozen in `outputs/trust_weights.json`)

The Trust Index answers one question at origin T: *how much should I trust this series' 14-day forecast?* It is a 0–100 **index**, not a probability: a score of 85 does not mean an 85 % chance that the forecast is right. It ranks series so that review time goes where it is most needed. Every input is known at the end of day T. A test perturbs every sale and actual after T and checks that the components do not change.

### 7.1 Components (each 0–100, higher = more trustworthy)

Notation: *series* = store × family; *window* = the 14 target days T+1..T+14 of one origin; *evaluable day* = an open day with an observed sale.

| Component | Default weight | Formula |
|---|---|---|
| Historical accuracy | 30 % | 100 × (1 − min(WAPE_hist, 50) / 50) |
| Interval reliability | 25 % | 100 × (1 − min(\|cov_hist − 80\|, 40) / 40) |
| Demand stability | 20 % | 100 × (1 − min(CV_56, 1)) × (1 − zero_share_56) |
| Model agreement | 15 % | 100 × (1 − min(cv_models / 0.5, 1)) |
| Data completeness | 10 % | 100 × n_ok_56 / 56 |

- **WAPE_hist:** the champion's WAPE for this series, pooled over the evaluable days of the last ≤ 8 backtest origins whose whole window was already known at T (o + 14 ≤ T). The 50 % cap was set from selection-stage data, where per-series champion WAPE ranged from about 7 % to 45 %, so the score stays linear over the observed range.
- **cov_hist:** share of evaluable actuals in those same windows inside the champion's P10–P90 band. The band uses only quantiles that existed at T: at calibration origins, quantiles re-fitted on **selection** residuals; at holdout and forward origins, the frozen calibration quantiles. Under-coverage costs more than over-coverage: 40 % coverage scores 0, 100 % scores 50.
- **CV_56, zero_share_56:** population standard deviation ÷ mean, and the share of zero-sales days, over evaluable days in T−55..T. The daily CV includes the weekly pattern.
- **cv_models:** population standard deviation ÷ mean of the 14-day forecast totals of the five models (naive, seasonal naive, moving average, Holt-Winters, LightGBM A). Disagreement between reasonable models signals uncertainty.
- **n_ok_56:** number of evaluable days among the 56 calendar days T−55..T.

### 7.2 Score, bands, hard flags and priority

**score = Σ_k w_k × component_k**, with Σ w_k = 1.

| Band | Rule |
|---|---|
| Accept | score ≥ 80 and no hard flag |
| Monitor | 60 ≤ score < 80 and no hard flag |
| Review | score < 60, **or any hard flag** |

Hard flags (each forces Review):
- `low_completeness`: completeness < 80 % in the 56-day lookback.
- `holiday_closure`: a chain-wide closure (25 Dec or 1 Jan) inside T+1..T+14.
- `promo_above_p95`: the planned mean promotion count over T+1..T+14 is above the series' 95th percentile of its 14-day rolling mean promotion count over all history ≤ T (the forecast would be extrapolating).
- `missing_<component>`: a component could not be computed.

**Priority** (sort key for the exception inbox, higher = look first) = (100 − score) × (series' 14-day champion forecast ÷ total 14-day champion forecast).

### 7.3 Weight selection rule (calibration origins only)

The default weights were a hypothesis. The rule below was written **before** any calibration result was computed:

- **Objective:** Spearman rank correlation between the score and the realised champion WAPE of the same window, over all 96 × 8 = 768 calibration windows. More negative is better.
- **Candidates:** the default (30/25/20/15/10), equal weights, volatility-only (stability alone) and every weight vector on a 0.1-step grid over the five components (1,001 vectors).
- **Rule:** keep the default unless the best candidate is at least **0.03 more negative** than the default on all calibration windows **and** more negative on **each half** of the calibration origins (2017-04-25..05-16 and 05-23..06-13). The margin and the split guard against picking a grid point that is best by chance.
- **Thresholds:** keep 80 / 60 unless a band is empty on calibration.

**Disclosed amendment** (made after a mechanics dry run on baseline-only champions, before LightGBM and the real freezes existed): completeness was 100 on every calibration window, so its weight only shifts all scores and cannot be identified. A component that is constant on calibration keeps its default weight, and grid points that move it are dropped (leaving 220 grid points). Exact ties are broken by the smallest distance to the default weights.

**Outcome** (`outputs/trust_calibration.json`):

| Candidate (accuracy/interval/stability/agreement/completeness) | ρ all | ρ first half | ρ second half |
|---|---|---|---|
| default 0.30/0.25/0.20/0.15/0.10 | −0.5141 | −0.4427 | −0.6099 |
| equal 0.20 each | −0.4781 | −0.4020 | −0.5845 |
| volatility only | −0.4488 | −0.3629 | −0.5833 |
| **chosen 0.7/0.1/0.1/0.0/0.1** | **−0.6501** | −0.5887 | −0.7306 |

The chosen vector beat the default by 0.136 and on both halves, so the rule adopted it. Bands stayed at 80 / 60 (no band was empty). Calibration band counts: Accept 261, Monitor 368, Review 139. Weights, bands and constants were then frozen (trust_weights sha256 `dc75cfb2…54b6`).

### 7.4 Validation design (holdout, run once)

On the 6 holdout origins (576 windows): Spearman ρ of score vs realised window WAPE; realised WAPE per band (ratio of sums); the Review ÷ Accept WAPE ratio; the Review band's share of total absolute error vs its share of windows; and whether the index beats the volatility-only baseline. Results and caveats are in [reports/VALIDATION_REPORT.md](reports/VALIDATION_REPORT.md).

---

## 8. Holdout run and forward forecast

`scripts/07_holdout_and_forward.py` verified the three freeze hashes, ran every model on the 6 holdout origins and the forward origin (2017-08-15), applied the frozen intervals and Trust weights, and wrote `outputs/holdout_results.json` and `outputs/holdout_run.json` (first and only run 2026-11-25T21:04:43Z, results digest `7a7799b0…f73f8`). The forward forecast uses the planned promotion counts in Kaggle's `test.csv` (2017-08-16 onwards). `test.csv` has no sales and is never used for evaluation.

---

## 9. Promotion what-if (`src/models/lgbm.py::promo_whatif`, exported by `src/export.py`)

At the forward origin, the frozen LightGBM variant-A model is refitted once (its unscaled predictions equal the saved forward forecasts exactly). For each series, the planned promotion count on every target day is multiplied by (1 + s/100) for s ∈ {−100, −50, 0, +50, +100}; `promo_t_rel28` is recomputed; the frame is re-scored and rescaled.

- `change_pct` = the 14-day forecast at scale s vs scale 0.
- **`extrapolation`** = the scaled 14-day mean promotion count is above the series' historical p95 of its 14-day rolling mean, **or** the scaled maximum single day is above the historical maximum.
- `below_history` (informational) = the scaled 14-day mean is below the historical minimum 14-day rolling mean.

This is a **model-predicted association**, not a causal effect. The response is non-monotonic in many series (for example, for 49|GROCERY I, −100 % promotions gives +7.4 %), and removing all promotions is below anything in history for 71 of 96 series.

---

## 10. UAE Scenario Lab formulas (`src/scenario.py` = `docs/scenario.js`)

The Scenario Lab is a **hypothetical** planning calculator. The baseline is real (the champion's 14-day forward forecast for a family, summed over 12 stores), but the event percentages (Ramadan 5/15/30 %, Eid al-Fitr 10/25/45 %, Eid al-Adha 8/20/40 %, Summer −20/−10/0 %, Promo campaign 5/12/25 %) are **editable demonstration assumptions, not measured UAE uplift**. The Ecuador holiday analogue shown alongside is a real Ecuadorian measurement (same-weekday comparison, 1–2 weeks before and after, over 19 national holidays), not a UAE estimate.

- Demand per scenario: **D_s = max(0, baseline × (1 + pct_s / 100))**, s ∈ {low, base, high}.
- Supply position: shortfall = max(0, D − supply); excess = max(0, supply − D); coverage % = 100 × supply ÷ D (null if D = 0).
- Cost of ordering plan p if outcome o happens: **cost(p, o) = c_u × max(0, D_o − Q_p) + c_o × max(0, Q_p − D_o)**, with Q_p = D_p, c_u = cost per unit short, c_o = cost per unit over.
- **Recommendation = the plan with minimum expected cost**, using fixed weights low / base / high = 0.25 / 0.5 / 0.25. Ties go to the lower maximum regret, then to the order low, base, high. The minimax-regret plan is also shown when it differs. "Keep current supply" is shown for reference.
- Critical ratio = c_u ÷ (c_u + c_o), shown as a hint.
- Timing: order_in_days = days_to_event − lead_time. Below 0 → "late" (expedite or reallocate); 0 to 3 → "now"; above 3 → "later".

JavaScript and Python give identical outputs on 8 shared test cases (`tests/js/scenario_cases.json`), and so does the Excel Scenario sheet (checked by a Python formula evaluator).

---

## 11. S&OP demand review and simulated FVA (`src/export.py`)

At the forward origin (2017-08-15), for each series:

- `stat_forecast` = 14-day sum of the champion forecast; `p10` / `p90` = sums of the daily bounds (conservative: this is not an interval for the 14-day total).
- Trust score and band, as in section 7.

The summary is generated deterministically from the numbers:

- **Top risks:** up to 5 Review-band series by volume.
- **Persistent bias:** families whose champion bias has the same sign and |bias| > 2 % on **both** calibration and holdout.
- **Review queue:** all Review series sorted by priority, with the reason (the hard flag, or else the weakest non-zero-weight component).
- **Promotion checks:** up to 4 series with the `promo_above_p95` flag (by volume), plus the 3 largest +50 % what-if changes among non-extrapolating rows (worded as associations).
- **Discussion points:** chain total vs the last 14 observed days, Review volume share, bias, FVA, promotion-plan check.

**Simulated Forecast Value Added (FVA).** A real FVA compares the error of the statistical forecast with the error after planner overrides. We have no real overrides, so a **rule-based simulated planner** is used:

> For each holdout origin T and family, measure the champion's bias over all completed windows before T (calibration windows and earlier holdout windows with origin + 14 ≤ T). If |bias| > 2 %, divide that family's forecasts by (1 + bias / 100); otherwise keep the forecast. **FVA (pp) = champion WAPE − simulated WAPE** (positive = the rule reduced error).

Result: 12.57 % → 13.02 %, **FVA −0.45 pp**, with 17 of 48 family-origin decisions adjusted. This is **not human-planner performance**; it only shows that a mechanical "correct last period's bias" habit would have hurt on this holdout.

---

## 12. Exploratory analysis

Fourteen DuckDB SQL files in `sql/`, one per business question, run by `scripts/04_build_eda.py`, use history from 2015-06-01 to 2017-06-27 only (the holdout is never read; the PRODUCE gap check reads pre-window history on purpose). Results are in `outputs/eda/` and `notebooks/01_demand_intelligence.ipynb`. All are associations on observed sales. Exact definitions are at the top of each SQL file.
