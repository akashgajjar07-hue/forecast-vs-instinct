# Export agent: pipeline outputs → dashboard JSON

## What I built
| Path | Purpose |
|---|---|
| `src/export.py` | Builders for every file in `SCHEMAS` + 96 series files, schema validation, 3 MB guard |
| `scripts/08_export_dashboard.py` | Verifies the freeze hashes (`verify_all`), runs `build_all`, prints sizes (~18 s) |
| `scripts/validate_docs_data.py` | CI step (referenced by `.github/workflows/ci.yml`): validates published docs/data and checks the 3 MB limit. I **overwrote** an existing file at this path without reading it first (the Write call reported "updated"). The new version does the job the CI step names. |
| `tests/test_reconciliation.py` | `@pytest.mark.data`, 14 tests: recomputes the KPIs, hero, findings numbers, arena holdout overall + week buckets, intervals, trust validation/band counts/14-day sums/candidates, S&OP totals + FVA, scenario baselines and 4 sample series files from `outputs/` + panel with independent code |
| `tests/test_schemas_docs.py` | CI-safe: validates whatever exists in `docs/data` and checks the size limit and that no FIXTURE marker is present. Skips cleanly when nothing is published |

Nothing frozen was touched. The holdout script was not re-run. The only model fit is one LightGBM refit at the
forward origin (frozen `selection.json → model_params.lightgbm`, `masked_view(panel ≤ 2017-08-29, 2017-08-15, "A")`,
the same truncation as the harness). Its scale-0 predictions equal the saved forward forecasts (max |diff| = 0, asserted ≤ 1e-6).

## Files produced (compact JSON, total 2,452 KB excluding `_fixtures`, limit 3 MB)
arena 131.0 KB · demand 42.6 · trust 30.8 · sop 21.5 · series_index 12.1 · overview 6.8 · scenario 6.4 · meta 3.3 ·
intervals 3.1 · series/*.json 96 files, 2,174 KB (max 23.5 KB). `docs/data/scope.json` (20.6 KB) was already
there. It is not in the contract and no view loads it. I left it in place, and it counts toward the size limit.

## Definitions
- **Evaluation population**: `evaluable()` (closed == 0 and actual not null). The export asserts that every model/variant has the same n per stage.
- **Arena slices**: overall; family; store; horizon 1..14; horizon_bucket 1-7/8-14; promo = target-date `onpromotion > 0` ("promo") vs 0 ("no_promo"); holiday = target-date `is_holiday` ("holiday"/"normal"); week_bucket = per series × origin, the 7 daily forecasts and actuals of h 1–7 ("week1") and h 8–14 ("week2") are summed (`metrics.bucket_totals` on evaluable rows), then WAPE/bias/MAE/RMSE are computed on the sums (n = number of buckets). Stages: selection, calibration, holdout. LightGBM appears as A and B, baselines as A only. Metrics are rounded to 4 dp.
- **variant_comparison**: LightGBM WAPE A vs B, per family + "all", for selection and holdout.
- **intervals**: `calibration_coverage` = leave-one-origin-out figure (79.20 %). Holdout coverage/width use champion_intervals, holdout stage, evaluable rows. Width is in units/day.
- **trust.latest**: forward-origin (2017-08-15) rows of trust_scores. `forecast_14d` = champion forward sum (asserted equal to trust_scores). `p10_14d`/`p90_14d` = **sums of daily P10/P90** (conservative, stated in `trust.notes`). `priority` = (100 − score) × the series' share of 14-day volume (from trust_scores). Hard flags go through `describe_flags`.
- **calibration_choice**: default, equal, volatility_only and the chosen grid point with their calibration Spearman. `grid_size` = 220 from trust_calibration.json.
- **overview**: forecast_14d_total = sum of the champion forward forecast for 2017-08-16..29. holdout_wape/bias = `champion_system`. review_exceptions = Review count at the forward origin. interval_coverage = holdout coverage vs 80. Outlook history = chain sum of panel sales for the 56 days to 2017-08-15, set to null if any series is closed (none in that window). Outlook forecast/band = chain_total forward rows. top_exceptions = top 8 by priority. The reason is the hard-flag text, or else "Weakest component: X n / 100", taken over components with non-zero weight (agreement has weight 0, so it is never named). ml_wins.families_won = families where LightGBM has the lowest HOLDOUT WAPE of the 5 models (`holdout_results.by_family`). findings = 6 sentences generated from the numbers. The Trust sentence adds a caveat when |Review − Monitor| WAPE < 1 pp.
- **meta.hero**: champion holdout WAPE (detail: best baseline) · Review/Accept error ratio · holdout interval coverage.
- **series files**: history 2017-01-01..2017-08-15 (1 dp, null if closed). holdout: 6 origins × 14 rows: actual, champion p10/p90, models = the 5 model ids (lightgbm = variant A) + `lightgbm_B`. The UI ignores keys that are not in `MODELS`. forward: champion forecast, p10, p90, planned onpromotion, models.
- **what-if**: for scale −100/−50/0/+50/+100, `promo_whatif` scales `promo_t` (and `promo_t_rel28`). promo_total = sum of the scaled plan. change_pct = vs scale 0. `hist_max_promo` = max daily promo on dates ≤ 2017-08-15. `hist_p95_promo` = p95 of the 14-day rolling mean promo (≤ T, same definition as the trust promo flag). **extrapolation** = scaled 14-day mean > hist_p95 OR scaled max daily > hist_max. I also added an informational field, `below_history` = scaled 14-day mean < the historical minimum 14-day rolling mean. It is not part of the contract flag, and the Workspace shows a softer note for it.
- **scenario**: baseline_14d = chain sum (12 stores) of the champion forward forecast per family. Events = the fixture defaults + Custom (0/0/0), labelled as demonstration values. ecuador_analogue is copied verbatim (extra fields kept).
- **sop**: rows = forward origin per series (stat_forecast = 14-day champion sum; p10/p90 = sums of daily bounds; trust score/band). top_risks = up to 5 Review-band series by volume. persistent_bias = families whose champion bias has the same sign with |bias| > 2 % on BOTH calibration and holdout. review_queue = all Review series by priority with their reason. promo_opportunities = up to 4 series with the `promo_above_p95` flag (by volume) + the 3 largest +50 % what-if model-predicted changes among non-extrapolating rows (worded as associations). discussion_points = chain total vs last 14 days, Review volume share, bias, FVA, promo-plan check.
- **FVA (simulated)**: for each holdout origin T and family, the champion bias is measured over evaluable champion rows of all calibration + holdout windows with origin + 14 ≤ T. If |bias| > 2 %, forecasts are divided by (1 + bias/100). fva_pp = champion WAPE − simulated WAPE.

## Key exported numbers
Holdout WAPE 12.571 % (bias −0.458 %), best baseline Holt-Winters 17.23 %. 14-day forecast 2,119,685 units.
Review 22 / 96 (Monitor 39, Accept 35). Interval coverage 80.32 % (LOO calibration 79.20 %). Trust Spearman −0.637,
Review/Accept 1.96×, Monitor 17.3 % vs Review 17.46 %. LightGBM best on holdout in 8/8 families. Persistent bias:
Beverages (−3.4 % calibration, −2.4 % holdout). FVA −0.45 pp (12.57 % → 13.02 %; 17 of 48 family-origin decisions adjusted).

## JS fixes (minimal, wording only; layout untouched)
1. `views/overview.js`: the champion caption said that `families_won` (now holdout-based) was the selection champion count. It now shows both: the selection champion count from the `champions` map, and the holdout wins.
2. `views/workspace.js`: the extrapolation warning compared a 14-day promo total against "p95 of a 14-day window", which is a daily mean. It now states the per-day mean, the p95 of 14-day means and the highest single day. Added a note for `below_history`. The sub-caption now says "precomputed with the frozen LightGBM model (known promo plan)" instead of "by the champion model".
3. `views/scenario_lab.js`: the Ecuador analogue said "vs normal days". The real definition is vs the same weekday 1–2 weeks before/after.
4. `views/exceptions.js`: "(from 4 candidates)" was misleading because only 4 of the compared candidates are published. It now names the default/equal/volatility-only candidates and the 220-point grid.

## Verification
- `scripts/playwright_check.py` (real data): PASSED at 1440 and 390 for all 8 views, 276 interactions, no console errors / failed requests / horizontal scroll. Screenshots in `docs/assets/screenshots/`.
- `pytest tests/test_reconciliation.py tests/test_schemas_docs.py`: 25 passed. `pytest -m "not data"`: 115 passed, 2 skipped. `node --test tests/js/scenario.test.mjs`: 7 pass.

## Open issues
- The what-if is non-monotonic in many series. For example, for 49|GROCERY I, −100 % promo gives +7.4 % (a model artefact). Removing all promotions is below anything in history for 71 of 96 series (`below_history`), but the contract's `extrapolation` flag only covers the upper side.
- 22 series are already flagged as extrapolating at scale 0 (planned promo above p95 or above the historical max). 13 of them carry the Trust `promo_above_p95` hard flag, which is the main reason many high-score Produce/Personal Care series are in Review.
- Overview: the outlook card stretches to the height of the Findings card, which leaves empty space at 1440 px (cosmetic).
- `node --test tests/js/` (directory form, used in CI) fails on local node 22.23 because it is not treated as a directory. Passing the file works. CI uses node 22 from setup-node, so check it there.
- `docs/data/_fixtures/` still exists (agent D: delete before deploy). It is excluded from the size check.
