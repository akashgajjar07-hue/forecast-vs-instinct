# Limitations

What this project cannot tell you, and why. Each point says what the limitation is, why it matters, and what would fix it. Numbers come from the files in `outputs/` and the executed notebooks.

## 1. Data

**Observed sales, not unconstrained demand.** The Kaggle data records what was sold, not what customers wanted to buy. There is no stock-out or on-shelf availability data. When a product ran out, recorded sales were low, and every model (and every accuracy number) treats that as low demand. *Fix:* estimate unconstrained demand using availability or inventory data (for example, by flagging stock-out days and imputing or censoring them).

**Family level, not SKU level.** Each series is a whole product family in one store (for example, all DAIRY in store 36). Real replenishment works at SKU × store level, where series are much sparser and noisier, with many zero days. Family-level WAPE will be much lower than SKU-level WAPE. *Fix:* rebuild at SKU-store level with intermittent-demand methods and hierarchical reconciliation.

**Only 96 series.** 12 of 54 stores × 8 of 33 families. The stores were chosen by a representative rule (type quota, then new cluster, new city, median volume), but they were also filtered to have **no recording gaps and no unexplained closures**, so they are cleaner than the average store.

**Ecuador, not UAE.** The data is from an Ecuadorian grocery chain. Calendar effects (Carnaval, local holidays, payday on the 15th and last day of the month, the 2016 earthquake) are Ecuadorian. Nothing here measures UAE demand.

**Promotion = a count only.** `onpromotion` is the number of promoted items in a store-family on a day. There is no price, discount depth, display or media information. Every promotion result is an **association, not a causal uplift**. In the holdout, about nine rows in ten had some promotion, so "no promotion" days are rare (652 of 8,064 rows, mostly BREAD/BAKERY and FROZEN FOODS).

**No future transactions or oil.** Transactions were excluded because they are unknown at the forecast date; oil was tested and dropped. This is correct for forecasting but means no external economic driver is used.

## 2. Evaluation

**Short holdout.** 6 weekly origins, all in one summer: forecasts for 28 June to 15 August 2017. That is 8,064 scored rows for each model and 576 Trust windows, but only 6 distinct starting points and about 7 weeks of calendar.

**No December peak.** FROZEN FOODS puts 30.0 % of its in-window volume into December, and its hardest season was never in the holdout. Christmas and New Year closures were never inside a holdout window, so the `holiday_closure` hard flag was never tested.

**Few holiday days.** The holdout has only 176 holiday rows per model (one national day, 11 August, plus a few local days). The finding that the moving average beat LightGBM on holidays (16.41 % vs 17.13 % WAPE) is indicative only.

**Family bias hides behind the total.** The champion's overall holdout bias was −0.46 %, but family biases ranged from −6.31 % (CLEANING) to +4.02 % (DAIRY) and +3.92 % (FROZEN FOODS). A single headline bias number is not enough.

**ML does not win everywhere.** LightGBM had the lowest holdout WAPE on 83 of 96 series; simple models won 13 (Holt-Winters 10). At weekly-total level, the moving average and seasonal naive come much closer.

**Calibration under-forecast was not fixed.** On calibration, LightGBM under-forecast by 2.39 % overall. It was not re-tuned, because using calibration for model choices would contaminate the intervals and Trust weights fitted there. It did not carry into the holdout overall (−0.46 %), but BEVERAGES stayed under-forecast (−3.4 % calibration, −2.4 % holdout).

## 3. Prediction intervals

**Empirical targets, not guarantees.** "80 %" is a target measured on past errors. Holdout coverage was 80.32 %, but there is no promise any single interval contains the actual.

**Uneven by family.** Holdout coverage ran from 72.2 % (BEVERAGES) to 87.8 % (GROCERY I). BEVERAGES, BREAD/BAKERY and CLEANING bands were too narrow; GROCERY I, PERSONAL CARE and DAIRY bands too wide.

**Sundays under-covered.** Quantiles are pooled over all weekdays in a 7-day bucket, but Sunday is the peak day with the biggest errors. Holdout coverage at horizons 5 and 12 (both Sundays, since origins are Tuesdays) was 70.8 % and 67.2 %; calibration showed the same pattern (59.0 % and 61.2 % leave-one-origin-out). *Fix:* weekday-aware quantiles.

**14-day totals.** The 14-day P10–P90 shown in the S&OP table and Exception Centre is a separately calibrated band for each series' 14-day total (relative-residual conformal on calibration windows only). It **under-covers**: 72.6 % of holdout 14-day totals fell inside it against the 80 % target (18.9 % below P10, 8.5 % above P90; calibration leave-one-origin-out estimate 77.3 %). It replaced the sum of daily bounds, which covered 94.1 % only because it was about twice as wide (42.1 % vs 20.5 % of the forecast). The band was designed after the holdout had been opened (prompted by the independent review); its quantiles still use calibration windows only and were frozen before holdout scoring.

## 4. Forecast Trust Index

**It is an index, not a probability.** A score of 85 does not mean an 85 % chance the forecast is right.

**It is dominated by historical accuracy.** The chosen weights are 70 % accuracy, 10 % interval, 10 % stability, 0 % agreement, 10 % completeness. On the holdout, accuracy alone ranked windows slightly better (Spearman −0.647) than the full score (−0.637), and the score's rank correlation with the accuracy component was 0.971. The index mostly says "this series was hard recently, so it will be hard again." That is useful, because series difficulty persists, but the other components add little ranking power on this data. Interval reliability alone had a holdout Spearman of 0.013 (no signal).

**The promotion hard flag points the wrong way.** `promo_above_p95` fired on 67 holdout windows (11.6 %). Those windows had *lower* realised WAPE (8.71 %) than unflagged ones (13.02 %), and 65 of the 123 Review windows were in Review only because of the flag. The cause: promotion counts trended up over 2015–2017, so a full-history 95th percentile is easy to exceed. As a result, Monitor (17.30 %) and Review (17.46 %) have almost the same error. *Fix:* a percentile over a recent window, or flag only plans beyond anything in the training data.

**Completeness was constant.** It was 100 on every calibration and holdout window (the scope was chosen to have no gaps), so its 10 % weight just adds 10 points to every score.

**Optimistic weight choice.** The weights were picked from 220 grid points on calibration, so the calibration Spearman (−0.650) is an optimistic estimate. The holdout value (−0.637) held up well, but this was one test.

**Partly in-sample history.** At calibration origins, the accuracy and interval history come partly from selection windows, the same windows the champion was chosen on.

**Ranking mixes two things.** A rank correlation across 96 series mixes "which series are hard" with "which windows are hard". Much of the signal is persistent series difficulty.

## 5. Planning features

**FVA is simulated.** The forecast value added figure (−0.45 pp) comes from a mechanical bias-correction rule, not from real planner overrides. It says nothing about how well a real planner would do.

**The UAE Scenario Lab is hypothetical.** Its baselines are real Ecuadorian model forecasts, but the event percentages (Ramadan, Eid al-Fitr, Eid al-Adha, Summer, Promo campaign) are editable demonstration assumptions, not measured UAE uplift. The expected-cost weights (0.25 / 0.5 / 0.25) are also fixed assumptions.

**Promotion what-if is not causal and not always sensible.** Scaling the promotion plan shows the model's predicted response. It is non-monotonic in many series (for example, −100 % promotions raises the 49|GROCERY I forecast by 7.4 %), and removing all promotions is below anything in history for 71 of 96 series.

## 6. Tools and verification

**Excel formulas were verified by a Python evaluator, not by Excel.** The workbook is written with openpyxl, which does not calculate. LibreOffice was not available. The tests include a small evaluator for the Excel functions the workbook uses and check its results against the Python pipeline. The workbook stores no cached values; Excel calculates on open, but some previewers may show empty formula cells.

**The site does not retrain models.** Models are trained offline; the site loads exported results. The dashboard shows a fixed snapshot with the last actual on 2017-08-15.

**Data licence.** Kaggle's rules page could not be read automatically during the build. No raw rows are published, only model outputs and aggregates, with citation. The owner should confirm the competition rules before sharing derived outputs (DECISIONS #4).


## Future work

- **Rolling recalibration of interval quantiles.** Both the daily and the 14-day bands use quantiles fitted once on 8 calibration origins. The 14-day band under-covered on the holdout (72.6 % vs 80 %) because the calibration-period under-forecast bias (−2.4 %) did not persist (holdout −0.5 %). A production version would refit the quantiles every week on the most recent completed windows (and could add weekday-specific daily quantiles to fix the Sunday under-coverage). The published bands are kept exactly as frozen.
- **Event flags known in advance only.** Mask unforeseeable event flags (earthquake) for target dates after the origin when building training rows (independent review finding 13). This would mean retraining, so it was not done after the holdout.
- **Promotion hard flag** based on a recent-window percentile instead of full history.
