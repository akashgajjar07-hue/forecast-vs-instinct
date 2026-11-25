# Validation report

This report collects the evidence that the results in this project can be trusted: data quality checks, backtest design, leakage controls, reproducibility, interval validation and Trust Index validation. It also lists what failed or is weak. The final section is reserved for an independent review.

Data: Corporación Favorita (Ecuador), Kaggle. Recorded sales are **observed sales, not unconstrained demand** (no stock-out data). Every accuracy number below should be read that way.

---

## 1. Test suite status

Final run on 2026-11-26 after the independent-review fixes (`reports/test_results.txt`, `reports/junit_local.xml`):

| Command | Result |
|---|---|
| `pytest` (all tests, Node on PATH so JS/Python scenario parity runs) | **151 passed, 0 failed, 0 skipped** |
| `pytest -m "not data"` (CI-safe, synthetic fixtures) | **118 passed** |
| `pytest -m data` (real panel and outputs, incl. dashboard reconciliation to 1e-6) | **33 passed** |
| `node --test tests/js/scenario.test.mjs` | **7 passed, 0 failed** |
| `scripts/playwright_check.py` (local and live URL, 8 views × 1440 / 390 px) | PASSED: no console errors, no failed requests, no horizontal scroll |
| GitHub Actions CI (`.github/workflows/ci.yml`) | green (see `gh run list`) |

Earlier, three data tests failed because they still asserted "no holdout rows" after the holdout had (correctly) been run. They now allow holdout/forward rows only if `outputs/holdout_run.json` exists.

---

## 2. Data quality checks

| Check | Evidence | Result |
|---|---|---|
| Panel keys unique and complete | `test_phase1::test_panel_keys_unique_and_complete`: no duplicate (series, date); 96 series; every series has the full daily calendar; starts 2015-06-01; last non-future date 2017-08-15 | pass |
| Closed days are NaN, not 0 | `test_phase1::test_closed_days_are_nan_not_zero`: every closed row has NaN sales; every 25 Dec is closed; every open historical day has a non-negative sale | pass |
| Future rows carry no sales | `test_phase1::test_future_rows_have_no_sales`: rows after 2017-08-15 have NaN sales and a promotion plan | pass |
| Scope rule respected | `test_phase1::test_scope_criteria`: 12 stores, 5 store types, ≥ 4 cities; scope cutoff equals the first selection origin | pass |
| Panel facts | 79,008 rows (96 series × 823 days); 384 closed rows (25 Dec and 1 Jan only, 192 each); 1,536 future rows | from `data/processed/panel.parquet` |
| PRODUCE recording gap | zero share on open store-days 48.3 % before 2015-06-01, 0.0 % from then on (`sql/11_produce_recording_gap.sql`) | handled by window start (DECISIONS #5) |
| Open zero-sales days in scope | 1 each in CLEANING, PERSONAL CARE and FROZEN FOODS over 2015-06-01 to 2017-06-27; 0 elsewhere (`outputs/eda/eda_extras.json`) | negligible |
| EDA reconciles with the panel | `test_eda::test_weekly_totals_reconcile_with_panel` (weekly totals vs panel to 1e-6); `test_demand_dow_profile_mean_one`; schema validation of `demand.json` and the analogue | pass |
| Metrics behave on edge cases | `test_metrics`: hand-computed WAPE; bias sign; zero denominator → NaN with a flag; empty input; NaN pairs dropped; ratio-of-sums grouping; week-bucket totals | pass |
| Dashboard numbers reconcile with outputs | `test_reconciliation` (14 tests): KPIs, hero, findings, arena holdout overall and week buckets, intervals, trust, S&OP totals and FVA, scenario baselines, 4 sample series files, recomputed independently from `outputs/` and the panel | pass |

---

## 3. Backtest design

- **Rolling origin, weekly Tuesdays, 14-day horizon**, expanding training window, retrained at every origin.
- **Three stages with disjoint target windows** (`config/split_calendar.json`):

| Stage | Origins | Target dates | Used for |
|---|---|---|---|
| Selection | 10 (2017-02-07 … 2017-04-11) | 2017-02-08 → 2017-04-25 | tuning, champion choice |
| Calibration | 8 (2017-04-25 … 2017-06-13) | 2017-04-26 → 2017-06-27 | interval quantiles, Trust weights |
| Holdout | 6 (2017-06-27 … 2017-08-01) | 2017-06-28 → 2017-08-15 | final evaluation, once |

- `test_phase1::test_split_calendar_ordering_and_no_overlap` asserts the stage target windows do not overlap, end by 2017-08-15, and have at least 8 / 6 / 6 origins. `test_split_validator_rejects_overlap` checks the validator catches a deliberately overlapping calendar.
- **Same rows for every model:** the holdout scores every model on the identical 8,064 evaluable rows (6 × 96 × 14, no closed target days). Notebook 02 asserts the population size is equal across models.

---

## 4. Leakage checks

### 4.1 `tests/test_features_leakage.py` (LightGBM features)

| Test | What it asserts |
|---|---|
| `test_future_perturbation_masked_view_identical[A,B]` | Replacing every sale after T (and, for B, every promotion count after T) with random values leaves the scoring frame, the training frame and the final forecasts **exactly identical** |
| `test_future_perturbation_unmasked_feature_functions[A,B]` | Even **without** the harness mask, the feature code does not read sales after T (or, for B, promotions after T); results equal the masked-view features |
| `test_variant_A_unmasked_promo_perturbation_changes_only_target_promo` | In variant A, perturbing future promotions changes only `promo_t` and `promo_t_rel28` |
| `test_truncated_panel_gives_identical_origin_features[A,B]` | Features computed on a panel cut at T equal those computed on the full panel, for several origins |
| `test_training_rows_respect_origin` | Every training row has target date ≤ T and origin < T, horizon = target − origin, and a non-missing target |
| `test_horizon_completeness[A,B]` | Exactly one row per series × horizon 1–14, no gaps or duplicates, no missing forecasts |
| `test_variant_B_has_no_target_promo` | Variant B's feature list and scoring frame contain no `promo_t` or `promo_t_rel28` |
| `test_reproducible[A,B]` | Two runs give bit-identical forecasts |
| `test_negative_predictions_clipped` | Negative predictions are clipped to 0 by the harness and by `predict()` |
| `test_ratio_target_and_whatif` | The ratio target works; `promo_whatif(fit, 0)` equals the plain prediction; a what-if is refused for variant B |
| `test_real_lightgbm_backtests[A,B]` (data) | Saved backtests have the right columns, origins, 96 × 14 rows per origin, no duplicates, non-negative forecasts (**currently fails on a stale stage assertion, see 1.1**) |

A deliberate one-day look-ahead injected into `lag_1` made 5 of these tests fail (agent B's notes), which shows the tests can catch real leakage.

### 4.2 `tests/test_phase1.py` (harness and holdout guard)

| Test | What it asserts |
|---|---|
| `test_masked_view_hides_future_sales_and_promo_B` | After T, sales are NaN in both variants; promotion counts stay visible in A and are NaN in B; rows ≤ T are untouched |
| `test_run_origin_completeness_and_clipping` | The harness returns one row per series × horizon 1–14 and clips negative forecasts to 0 |
| `test_run_origin_rejects_incomplete_output` | A model that returns fewer than 14 horizons is rejected |
| `test_evaluable_excludes_closed` | The evaluation population drops closed days (25 Dec) and missing actuals |
| `test_freeze_verify_roundtrip_and_tamper` | A frozen file verifies; after editing it, verification raises `HoldoutGuardError` |
| `test_verify_refuses_missing_hash` | Verification fails if the hash file is missing |

### 4.3 `tests/test_baselines.py`

| Test | What it asserts |
|---|---|
| `test_no_use_of_data_after_origin[naive, seasonal_naive, moving_average, holt_winters]` | Multiplying future sales by 50, setting future promotions to 999 and marking future days closed does not change any forecast, even on an unmasked panel |
| `test_naive_is_last_observed`, `test_naive_skips_closed_origin_day` | Naive repeats the last observed sale, skipping a closed origin day |
| `test_seasonal_naive_same_weekday_last_week`, `..._falls_back_to_earlier_week_when_closed` | Correct same-weekday logic and fallback |
| `test_moving_average_value_and_ignores_closed` | Closed days are skipped, not counted as zero |
| `test_holt_winters_reasonable`, `test_holt_winters_fallback_to_seasonal_naive`, `test_holt_winters_real_fit_error_is_caught`, `test_fill_same_weekday` | Holt-Winters fits sensibly; failures fall back to seasonal naive and are logged |
| `test_completeness_via_harness[...]` | Every baseline returns all 14 horizons for every series |
| `test_runner_refuses_final_stages` | The backtest runner refuses `holdout` and `forward` stages unless explicitly allowed |
| `test_runner_truncates_panel_and_saves_merging` | The runner never passes a panel beyond the last target date needed, and saving merges stages correctly |
| `test_saved_real_baseline_backtests` (data) | Saved baseline backtests are complete (**currently fails on a stale stage assertion, see 1.1**) |

### 4.4 Other leakage guards

- `tests/test_trust.py::test_components_use_only_information_at_T` perturbs every actual after T (and later origins' forecasts) and asserts the Trust components at T are bit-identical.
- `tests/test_selection_intervals.py::test_scale_uses_only_past` checks the interval scale uses only the 28 days up to T; `test_champion_ignores_other_stages` checks champion choice reads only selection rows.
- `scripts/03_tune_lightgbm.py`, `05_select_and_freeze.py` and `06_trust_calibrate.py` truncate the panel at 2017-06-27 and assert no holdout or forward rows are present.
- The EDA reads history up to 2017-06-27 only.

---

## 5. Reproducibility evidence

| Evidence | Detail |
|---|---|
| Deterministic LightGBM | `deterministic=True`, `seed=42`, fixed thread count. A rerun gave **bit-identical forecasts on calibration origin 2017-06-13** (checked by the coordinator; agent B's notes). The final variant-A selection result exactly reproduced the stride-1 tuning candidate (11.8836 %) |
| Freeze hashes | `selection.json` `c1ae56f4…4bab`, `intervals.json` `bfc2c745…2853`, `trust_weights.json` `dc75cfb2…54b6`. The holdout run logged the same three hashes, and notebook 02 recomputes them from the files and confirms all three match |
| Holdout run log | `outputs/holdout_run.json`: first and only run at 2026-11-25T21:04:43Z, results digest `7a7799b0bdce6f02b9066e00020a69978da2d3a558e6518378b761703aad73f8`. Notebook 02 re-derives the digest from `holdout_results.json` and it matches |
| Notebooks recompute, never retrain | Notebooks 02 and 03 recompute every key number from the saved backtests and compare with the JSON: selection table (max difference 0.0), holdout leaderboard (0.0), interval quantiles (0.0), coverage, all Trust validation statistics. A SHA-256 listing of `outputs/`, `config/` and `data/processed/` was identical before and after executing them |
| Forward refit for the what-if | The export refits LightGBM once at the forward origin with the frozen parameters; its unscaled predictions equal the saved forward forecasts (max \|diff\| = 0) |
| Excel workbook | `test_workbook::test_build_is_reproducible`: a rebuild gives identical cell contents (except the build date) |

---

## 6. Accuracy validation (holdout, scored once)

| Model | WAPE % | Bias % | MAE | RMSE |
|---|---|---|---|---|
| LightGBM A | 12.57 | −0.46 | 208.5 | 512.5 |
| LightGBM B | 12.75 | −0.80 | 211.5 | 508.1 |
| Holt-Winters | 17.23 | +1.66 | 285.8 | 606.7 |
| Seasonal naive | 18.73 | −0.03 | 310.6 | 701.5 |
| Moving average | 22.03 | +0.25 | 365.5 | 694.7 |
| Naive | 30.18 | −5.15 | 500.6 | 938.2 |

Stability across stages for the champion: selection 11.88 %, calibration 12.65 %, holdout 12.57 %. Weaknesses found: simple models beat LightGBM on 13 of 96 series; the moving average beat it on the 176 holiday rows (16.41 % vs 17.13 %); family biases on the holdout range from −6.31 % (CLEANING) to +4.02 % (DAIRY).

---

## 7. Interval validation

Scaled split-conformal, P10–P90, quantiles per family × horizon bucket from 8 calibration origins (`outputs/intervals.json`).

| Check | Result |
|---|---|
| Calibration in-sample coverage | 79.76 % |
| Calibration leave-one-origin-out coverage | 79.20 % |
| **Holdout coverage** | **80.32 %** (10.5 % below P10, 9.2 % above P90) |
| Holdout coverage by family | BEVERAGES 72.2, DAIRY 83.9, BREAD/BAKERY 75.1, GROCERY I 87.8, FROZEN FOODS 81.1, CLEANING 75.9, PERSONAL CARE 86.4, PRODUCE 80.2 |
| Holdout coverage by horizon | 87.5, 88.5, 83.3, 73.4, **70.8**, 80.4, 83.9, 85.8, 88.4, 81.4, 70.7, **67.2**, 81.1, 82.1 (horizons 1–14; 5 and 12 are Sundays) |
| Mean width | 615.6 units, 37.3 % of the forecast |
| Chain-total band | 83.3 % coverage over 84 days, width 15.9 % of the forecast |
| Unit test | `test_selection_intervals::test_conformal_coverage_on_exchangeable_data` checks the method reaches its nominal coverage on synthetic exchangeable data |

**Verdict:** on target overall; not uniform. Sundays and three families are under-covered.

---

### 7.x 14-day total band (S&OP / Exception Centre)

| Band | Holdout coverage (target 80 %) | Below P10 / above P90 | Mean width (% of 14-day forecast) |
|---|---|---|---|
| Calibrated 14-day band (published; frozen `outputs/window_intervals.json`, sha256 `3da4fc84e848bb0a…`) | **72.6 %** | 18.9 % / 8.5 % | 20.5 % |
| Sum of 14 daily bounds (replaced) | 94.1 % | | 42.1 % |

576 complete holdout windows (96 series × 6 origins). Calibration leave-one-origin-out estimate: 77.3 %. By family: 61.1 % (BREAD/BAKERY) to 93.1 % (PERSONAL CARE).
**Verdict: below target.** On the calibration windows the champion under-forecast (bias −2.4 %), so the band was centred above the forecast; on the holdout the bias was close to zero (−0.5 %), so 14-day totals fell below P10 more often. The band is kept exactly as frozen and was not adjusted with holdout results.
Rolling recalibration is future work (LIMITATIONS.md). Disclosure: this band was designed after the holdout had been opened, in response to the independent review.

## 8. Trust Index validation

Weights chosen on calibration by a rule stated in advance (METHODOLOGY §7.3), frozen, then tested once on 576 holdout windows (`outputs/holdout_results.json → trust_validation`).

| Statistic | Holdout | Calibration (in-sample for the weights) |
|---|---|---|
| Spearman (score vs realised window WAPE) | **−0.637** (reported p-value 0.0, rounded) | −0.650 |
| Volatility-only baseline Spearman | −0.442 | −0.449 |
| Default weights Spearman (hindsight) | −0.503 | −0.514 |
| WAPE Accept / Monitor / Review | 8.93 / 17.30 / 17.46 % | 9.31 / 15.33 / 18.62 % |
| Windows Accept / Monitor / Review | 225 / 228 / 123 | 261 / 368 / 139 |
| Review ÷ Accept WAPE | 1.96× | |
| Review share of windows / of absolute error | 21.4 % / 26.3 % | |
| Beats volatility-only baseline | yes | |

**Hindsight diagnostics** (notebook 03; they did not change any frozen choice):

| Finding | Number |
|---|---|
| Accuracy component alone, holdout Spearman | −0.647 (slightly better than the full index) |
| Rank correlation of score with the accuracy component | 0.971 |
| Interval component alone | +0.013 (no ranking signal) |
| Stability alone / agreement alone | −0.442 / −0.216 |
| Promo flag fired | 67 windows (11.6 %), realised WAPE 8.71 % vs 13.02 % unflagged |
| Review windows only there because of the promo flag | 65 of 123 |
| Completeness | 100 on all 576 windows (constant) |

**Verdict:** the index ranks windows well and beats the volatility-only baseline, and Accept is clearly separated from the rest. But it is mostly historical accuracy, the promotion flag points the wrong way (it moved easier windows into Review), and Monitor and Review are not separated on realised error.

---

## 9. Other validation

- **Scenario Lab:** JS and Python give identical outputs on 8 shared cases (`tests/test_scenario.py` runs Node and compares field by field). The Excel Scenario sheet matches `run_scenario` on the defaults and on all 8 cases, checked with a Python evaluator of the workbook's own formulas (not with Excel).
- **Excel workbook:** `tests/test_workbook.py` (20 tests reported by the Excel agent): structure, data validation, override logic, forecast totals, exception scores and bands, summary numbers.
- **Dashboard:** `scripts/validate_docs_data.py` validates every published JSON file against `src/schemas.py` and the 3 MB size limit (run in CI). A Playwright end-to-end check passed at 1440 px and 390 px for all 8 views with real data (export agent's notes).

---

## 10. Independent review findings and fixes

A fresh reviewer agent with no access to the build conversation audited the repository (brief: leakage, holdout misuse, metric errors, unreconciled numbers, overclaiming, broken filters, accessibility, mobile). It found **no leakage and no holdout misuse**, and every dashboard/README number it recomputed matched. Findings and fixes:

| # | Severity | Finding | Fix | Verified by |
|---|---|---|---|---|
| 1 | Major | 14-day "P10–P90" in S&OP / Exceptions was the sum of daily bounds (holdout coverage 94.1 %), labelled like an 80 % band | Replaced with a band calibrated on 14-day totals, calibration windows only, frozen before holdout scoring (`scripts/07b_window_band.py`). Holdout coverage **72.6 %**, reported as is with the bias-shift explanation on the dashboard, README, LIMITATIONS and here | `tests/test_reconciliation.py::test_window_band_holdout_coverage`, `test_trust`, `test_sop_totals_and_fva` |
| 2 | Minor | Variant A uses recorded promo counts as the "known plan" | Documented as an upper bound (perfectly executed plan) in METHODOLOGY / MODEL_CARD | docs review |
| 3 | Minor | "Custom" twice in the Scenario Lab event list | Added only if not exported | Playwright |
| 4 | Minor | Overview "Top exceptions" included Accept series | Filtered to Review + Monitor by priority | reconciliation test |
| 5 | Minor | Arena champion WAPE switched to variant B with the toggle | Always variant A (frozen champions) | Playwright |
| 6 | Minor | Explorer charts ignored the store filter; store weeks imputed closed days, Sunday-aligned | Chain-level titles; no imputation; Monday weeks; missing-day hover | Playwright |
| 7 | Minor | README claim about the source of ML gain was untested | Softened to "feature importances suggest (no ablation)" | docs review |
| 8 | Minor | Validation report test status stale | Section 1 regenerated from the final run | this file |
| 9 | Minor | Accessibility: unfocusable scroll regions, page-wide aria-live, chart label without values, mobile clipping | `tabindex`/`role=region`/labels on table wraps; dedicated live region; values on bars; stacked mobile rows ≤ 480 px | Playwright 390 px, screenshots |
| 10 | Nit | "6 weeks of holdout it never saw" | "6 weekly forecast origins (targets 28 Jun – 15 Aug 2017), each scored on days the model had not seen" | dashboard, deck, post |
| 11 | Nit | "Demand" wording; "Promotion opportunities" | Explorer subtitle "Observed sales, not unconstrained demand"; heading "Promotion notes (associations, not causal)" | docs review |
| 12 | Nit | Holdout script used code defaults, not the hashed LightGBM params | Passes frozen params explicitly; `--rerun` **reproduced the holdout digest exactly** (`reports/reproducibility_rerun.txt`) | digest match |
| 13 | Nit | Target-date quake/event flags give early training rows foresight | Not retrained (would change models after the holdout); documented in LIMITATIONS and future work | — |
| 14 | Nit | Model colours differed between Arena and Workspace | Workspace uses the shared `MODEL_COLOR` / `MODEL_DASH` | Playwright screenshots |
| 15 | Nit | Selection WAPE optimistic for the tuned model; in-sample interval component at early holdout origins; LOO overlap | Documented (holdout confirms the ranking) | — |
| 16 | Nit | CSV export open to spreadsheet formula injection | Text fields starting with `= + - @` tab CR prefixed with `'` | code review |
| 17 | Open | Kaggle rules never confirmed | **Owner action**: confirm on the competition Rules tab before promoting the site | — |

UI fix details with file:line: `docs/agent_notes/review_fixes_ui.md`.
