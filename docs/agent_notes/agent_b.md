# Agent B: features, direct multi-horizon LightGBM, leakage tests

Files: `src/features.py`, `src/models/lgbm.py`, `scripts/03_tune_lightgbm.py`, `tests/test_features_leakage.py`.
Results: `outputs/lgbm_tuning.json`, `outputs/backtests/lightgbm_{A,B}.parquet`, `outputs/lgbm_feature_importance.json`.
All numbers below are copied from those files. The holdout was never used. The script cuts the panel at
2017-06-27 (the end of the calibration target window) before anything runs, and it asserts that every origin it runs is a
selection or calibration origin dated before 2017-06-27.

## Model
- **One global LightGBM, direct multi-horizon.** `horizon` (1–14) is a feature. The model is retrained at every
  origin T (expanding window), using only the masked view the harness provides.
- **Training examples.** Each example is a historical origin t', a series and a horizon h, with target date d = t'+h ≤ T.
  - t' runs from day 56 of the panel (burn-in, so the 56-day window is full) up to T−1, using every day
    (stride 1, chosen on selection).
  - Rows whose target is NaN (closed day) are dropped.
  - At the last calibration origin there are about 826k training rows (825,888 at the last selection origin).
- **Target (scale-free).** The model predicts y / (roll_mean_28 + 1), where roll_mean_28 is taken at the origin.
  - Training uses L2 loss, with sample weight = roll_mean_28 + 1, so the loss tracks errors in units rather than
    treating every series equally.
  - Predictions are multiplied back by the same scale and clipped at 0.
- **Final parameters.** Saved in `chosen_params` and in `DEFAULT_PARAMS`:
  - objective `regression` (L2), target `ratio`
  - num_leaves 127, learning_rate 0.05, n_estimators 400, min_data_in_leaf 100
  - feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 1.0
  - quake `flag`, use_oil false, origin_stride 1
  - num_threads 4, seed 42, deterministic, force_col_wise
  - (`tweedie_variance_power` is carried in the params but ignored for L2.)
  - A rerun gives bit-identical forecasts. The coordinator checked this on calibration origin 2017-06-13.

## Features (exact definitions)
Origin t' is the end of day t'. Target date d = t'+h. "Lag k" counts back from the origin, so lag 1 is day t' itself.

| Feature | Definition |
|---|---|
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | sales on t', t'−6, t'−13, t'−27 |
| `roll_mean_w`, `roll_std_w` (w = 7, 14, 28, 56) | mean and std of sales over days (t'−w, t']; NaN (closed) days are ignored; mean needs ≥1 value, std needs ≥2 |
| `open_share_28` | share of the last 28 days ≤ t' with observed sales |
| `sw_last` | sales[t' − o(h)], where o(h) = 7·ceil(h/7) − h is the most recent day ≤ t' on the target's weekday |
| `sw_mean4` | NaN-mean of sales[t' − o(h) − 7j] for j = 0..3 |
| `dow`, `month`, `dom` | weekday, month and day of month of d |
| `is_holiday_t`, `is_national_holiday_t`, `is_event_t`, `is_workday_t`, `payday_t` | panel flags of d for that store |
| `quake_t` | quake flag of d |
| `quake_share_28` | share of the last 28 days ≤ t' inside the quake window |
| `store_nbr`, `family_code`, `type_code`, `cluster` | LightGBM categoricals; the codes are fixed (family order from `C.FAMILIES`, type A–E → 0–4) |
| `horizon` | h |
| `promo_origin`, `promo_roll_7`, `promo_roll_28` | onpromotion at t', and its 7- and 28-day rolling means ending at t' (both variants) |
| `promo_t` (variant A only) | planned onpromotion on d |
| `promo_t_rel28` (variant A only) | promo_t / (promo_roll_28 + 1) |

Variant B has no target-date promo feature. The tests check this. Oil and transactions are not used (see below).

**Purity.** Origin features use rolling windows, shifts and forward-fills that only look backwards over a daily
(date × series) matrix, so the row for t' depends only on view rows dated ≤ t'. Target features read only the
calendar for d, plus promo on d in variant A.

## Selection tuning
Setup for every candidate:
- variant A, all 10 selection origins, training-origin stride 2 (to save time)
- base parameters: num_leaves 63, lr 0.05, 400 rounds, min_data_in_leaf 100
- ranking: overall WAPE rounded to 2 dp, ties broken by lower |bias|

| Stage | Candidate | Sel. WAPE | Sel. bias | Runtime (s) |
|---|---|---|---|---|
| objective | tweedie p=1.2 | 12.2985 | −1.6906 | 78.7 |
| objective | tweedie p=1.5 | 12.3016 | −1.7694 | 80.2 |
| objective | l2 raw | 12.4739 | −1.4840 | 66.1 |
| objective | l1 raw | 12.2828 | −2.1898 | 86.9 |
| objective | **l2 on ratio-to-roll28** | **12.0392** | −0.6317 | 71.3 |
| structure | + num_leaves 31 | 12.0858 | −0.7074 | 58.0 |
| structure | **+ num_leaves 127** | **11.9144** | −0.6151 | 98.7 |
| structure | + min_data_in_leaf 30 | 12.0611 | −0.6295 | 67.2 |
| structure | + min_data_in_leaf 300 | 12.0460 | −0.6608 | 61.9 |
| structure | + lr 0.03 / 800 rounds | 12.0169 | −0.7124 | 120.2 |
| structure | + lr 0.1 / 200 rounds | 12.0831 | −0.6043 | 31.4 |
| quake | drop quake-window targets (no quake features) | 11.9423 | −1.0556 | 79.2 |
| oil | + lagged oil | 12.2724 | −0.2110 | 109.6 |
| stride | **stride 1 (all daily origins)** | **11.8836** | −0.5355 | 165.4 |

Decisions:
- **Scale-free target.** Predicting the ratio beat every raw-target objective by about 0.25 pp WAPE and cut the bias
  from about −1.5 to −2.2 down to −0.63. One global model then learns shape (weekday, promo, holiday) and does not
  have to learn each series' level.
- **Earthquake: flag.** Flag WAPE 11.9144 (bias −0.6151) vs dropping training targets in 2016-04-16..05-15:
  11.9423 (bias −1.0556). Keeping the rows with `quake_t` and `quake_share_28` is better and less biased.
- **Oil: dropped.** WAPE was 12.2724 with oil vs 11.9144 without. A single national price adds no useful signal for
  14-day store demand and lets trees split on trend noise. Transactions are never used, because their future values
  are unknown at T.
- **Stride 1.** 11.8836 vs 11.9144 at stride 2. The cost is about 2× runtime.
- Variant B reuses variant A's parameters without separate tuning.
- Because the run is deterministic, the final variant-A selection result exactly reproduces the stride-1 candidate (11.8836).

## Backtest results (frozen config)
WAPE and bias in %. Population: open days with an observed actual.

| | Selection A | Selection B | Calibration A | Calibration B |
|---|---|---|---|---|
| **Overall WAPE** | **11.8836** | **12.0472** | **12.6511** | **12.9141** |
| Overall bias | −0.5355 | −1.1787 | −2.3854 | −2.2236 |
| BEVERAGES | 13.8067 (−2.39) | 14.2523 (−2.89) | 13.6354 (−3.41) | 13.8454 (−3.92) |
| BREAD/BAKERY | 11.5828 (+2.61) | 11.6006 (+2.55) | 10.4716 (−2.78) | 10.5703 (−2.15) |
| CLEANING | 12.7692 (+0.42) | 12.7830 (+0.01) | 19.1309 (−1.82) | 19.7448 (−1.40) |
| DAIRY | 10.6336 (+1.14) | 11.0684 (+0.88) | 10.4987 (+1.00) | 10.6126 (+0.12) |
| FROZEN FOODS | 20.4998 (−0.10) | 19.8365 (−1.04) | 20.2366 (−0.21) | 20.2446 (+0.39) |
| GROCERY I | 11.3794 (−0.28) | 11.2980 (−1.75) | 12.7158 (−3.13) | 12.9863 (−2.53) |
| PERSONAL CARE | 14.8620 (−1.63) | 15.0168 (−1.73) | 16.1887 (−1.50) | 16.9287 (−0.50) |
| PRODUCE | 9.5277 (−0.29) | 9.8074 (−0.11) | 8.2550 (−1.38) | 8.4418 (−1.08) |

Family cells show WAPE with bias in brackets.

- Knowing the promo plan (A) improves WAPE by 0.16 pp on selection and 0.26 pp on calibration.
- Calibration shows a systematic under-forecast of about −2.2 to −2.4 % in both variants, worst in BEVERAGES and
  GROCERY I. The conformal intervals should absorb this. We did not re-tune for it, because that would mean using
  calibration for model choice.

## Feature importance
Values are the mean gain share (%) per origin over all 18 backtest origins (`outputs/lgbm_feature_importance.json`).

| Rank | Variant A | Gain share (%) | Variant B | Gain share (%) |
|---|---|---|---|---|
| 1 | dow | 19.10 | dow | 25.09 |
| 2 | store_nbr | 13.87 | store_nbr | 14.24 |
| 3 | dom | 12.29 | dom | 13.47 |
| 4 | promo_t | 7.96 | family_code | 8.11 |
| 5 | family_code | 7.61 | month | 7.19 |
| 6 | month | 6.89 | sw_mean4 | 4.46 |
| 7 | sw_mean4 | 3.72 | roll_mean_56 | 2.70 |
| 8 | promo_t_rel28 | 3.63 | roll_mean_28 | 2.39 |
| 9 | roll_mean_28 | 2.77 | sw_last | 2.17 |
| 10 | sw_last | 2.48 | roll_std_28 | 2.03 |

The series level is already handled by the target scale (roll_mean_28), so the trees spend their gain on shape:
- weekday
- store × family profile
- payday and month-end effects (via `dom`)
- in A, the planned promo (`promo_t` + `promo_t_rel28` together ≈ 11.6 %)

## Why direct beats recursive here
- **No error compounding.** A recursive model feeds its own h=1 forecast back in as a lag for h=2, and so on. Errors
  at early horizons then become inputs at later ones. The direct model predicts every horizon from the same observed
  history at T.
- **Every horizon is trained on true information available at T.** Training rows pair a real origin t' with real
  targets at t'+h, so the model learns how far ahead each lag is informative. The day-14 forecast is fitted on
  day-14 examples, not on chained 1-step predictions.
- **No need to forecast the drivers.** Rolling stats, same-weekday values and (in B) promo history would all have to
  be simulated forward in a recursive model. The direct model reads them once at T. Known calendar items and (in A)
  the promo plan are joined at the target date.
- **One global model with `horizon` as a feature.** This is cheaper than 14 separate models and shares strength
  across horizons and 96 series.

## For the integrator
- **Harness.** The harness does not pass the variant, so wrap the model:
  `run_origin("lightgbm", functools.partial(lightgbm_model, variant="B"), panel, T, stage, "B")`.
- **What-ifs (variant A only).**
  - Call `fit = fit_lightgbm(masked_view(panel, T, "A"), T, 14, "A")`.
  - `fit` holds the booster, the T scoring frame, the feature list, the params and gain/split importances.
  - `promo_whatif(fit, scale_pct)` multiplies `promo_t` by (1 + scale_pct/100), recomputes `promo_t_rel28`, re-scores
    and rescales, and returns `series_id, horizon, target_date, forecast`.
  - `promo_whatif(fit, 0)` equals `predict(fit)`.
  - For a custom frame, edit a copy of `fit.scoring` and call `predict(fit, frame)`.
  - Report results as a model-predicted association, not a causal effect.
- **Forward / holdout.** Use the same `lightgbm_model` with `DEFAULT_PARAMS`. Forward origin 2017-08-15 has
  promo_t from test.csv rows in the panel.
- **Runtime** (MacBook Air, 4 threads, stride 1).
  - Mean fit + score per origin: 16.3 s for A and 13.6 s for B.
  - Full A run (18 origins): 294.0 s. Full B run: 245.5 s.
  - Tuning grid: 1174.9 s. Final runs: 539.5 s.
- **Rerun.** `python scripts/03_tune_lightgbm.py --final-only` reuses the saved choices.
- **Tests.**
  - `tests/test_features_leakage.py` covers: future-perturbation invariance (masked view and unmasked feature code,
    A and B), truncated-vs-full panel identity, horizon completeness, no target promo in B, reproducibility,
    negative clipping, ratio target and what-if, plus `@pytest.mark.data` checks on the saved backtests.
  - A deliberate one-day look-ahead injected into `lag_1` made 5 tests fail.
