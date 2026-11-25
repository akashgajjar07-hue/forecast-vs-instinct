# Notebooks agent: 02 Forecasting battle, 03 Trust & uncertainty

## Files
| Path | Purpose |
|---|---|
| `scripts/build_notebooks.py` | Generates both notebooks with nbformat (`--execute` also runs them in place via nbconvert's ExecutePreprocessor). |
| `notebooks/02_forecasting_battle.ipynb` | Split calendar, leakage controls, the 5 models, tuning, champion selection, holdout leaderboard and slices (about 0.36 MB executed). |
| `notebooks/03_trust_and_uncertainty.ipynb` | Conformal intervals, coverage, width, Trust Index formulas, weight choice, holdout validation, concerns, one worked example (about 0.38 MB executed). |

Rebuild: `.venv/bin/python scripts/build_notebooks.py --execute` (about 1 min), or build then
`.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/0X_*.ipynb`.

## Rules followed
- Read-only. The notebooks only read `outputs/backtests/*.parquet`, `outputs/holdout/*.parquet`, frozen JSONs, `config/` and
  `data/processed/panel.parquet`. Nothing is trained, the holdout is not re-run, and nothing is written to `outputs/`.
  A SHA-256 listing of `outputs/`, `config/` and `data/processed/` before and after execution was identical.
- Every number is computed in a cell. Key results are recomputed from the backtests and **asserted / compared with the
  saved JSON**: selection table (max diff 0.0), holdout leaderboard (0.0), interval quantiles (0.0), calibration
  coverage in-sample 79.76 % / leave-one-origin-out 79.20 %, holdout coverage 80.32 %, and all Trust validation statistics.
  Notebook 02 also recomputes the three freeze hashes and the holdout results digest. Both match `holdout_run.json`.
- Markdown interpretations were checked against the executed outputs and corrected where the first draft was wrong.
- Cells in notebook 03 that score alternative weightings on the holdout are labelled as hindsight diagnostics. No
  frozen choice was changed.

## Key computed results
**02.** The champion is LightGBM A in all 8 families (selection margin from 0.63 pp in PRODUCE to 9.03 pp in PERSONAL CARE).

| Model | Holdout WAPE (%) | Bias (%) |
|---|---|---|
| LightGBM A | 12.57 | −0.46 |
| LightGBM B | 12.75 | −0.80 |
| Holt-Winters | 17.23 | +1.66 |
| Seasonal naive | 18.73 | −0.03 |
| Moving average | 22.03 | +0.25 |
| Naive | 30.18 | −5.15 |

- MAE / RMSE for LightGBM A: 208.5 / 512.5. LightGBM B has a slightly lower RMSE (508.1).
- LightGBM A is best in 8/8 families and at 14/14 horizons, but per series it is best in only 83/96 (Holt-Winters wins 10).
- Weekly-total WAPE, LightGBM A: week 1 7.42 %, week 2 9.39 %. At the weekly level the moving average becomes the
  second-best model.
- Holiday rows (176): the moving average (16.41 %) beat LightGBM A (17.13 %).
- No-promo rows are only 652 (8 %), mostly FROZEN FOODS and BREAD/BAKERY.
- Variant A beats B by 0.16 / 0.26 / 0.18 pp on selection / calibration / holdout. B is slightly better in FROZEN FOODS,
  BREAD/BAKERY and DAIRY.
- LightGBM A holdout family bias ranges from −6.3 % (CLEANING) to +4.0 % (DAIRY).

**03.**
- Interval coverage on the holdout is 80.32 %: 10.5 % of actuals fell below P10 and 9.2 % above P90.
- By family, holdout coverage runs from 72.2 % (BEVERAGES) to 87.8 % (GROCERY I).
- Sundays (horizons 5 and 12) are under-covered in both calibration and holdout, because the quantiles are pooled
  over weekdays.
- Mean band width is 615.6 units, or 37.3 % of the forecast. The chain-total band covered 83.3 % with a width of 15.9 %
  of the forecast.
- Trust weights are 0.7 / 0.1 / 0.1 / 0 / 0.1, adopted by the pre-stated rule: they beat the default by 0.136 in rho
  and were better on both halves.
- Holdout Spearman is −0.637, against −0.442 for volatility-only and −0.503 for the default weights (hindsight).
- Holdout WAPE by band: Accept 8.93 %, Monitor 17.30 %, Review 17.46 %. The Review band holds 21.4 % of windows and
  26.3 % of absolute error.
- Concerns found:
  - Accuracy alone gives a holdout rho of −0.647, and the score has a rank correlation of 0.971 with the accuracy
    component.
  - Interval reliability alone gives rho 0.013, so it carries no ranking signal on the holdout.
  - The promo flag fired on 67 windows (11.6 %). Those windows had *lower* WAPE (8.71 % vs 13.02 %), and 65 of the
    123 Review windows are there only because of the flag.
  - Completeness was 100 on all 576 windows.
- Worked example: series 36|DAIRY at origin 2017-07-11, chosen as the median-score series without a hard flag.
  - Every component and the score (79.56) were recomputed by hand and matched the saved values, giving band Monitor.
  - Realised WAPE was 6.11 %, and all 14 of 14 days fell inside the band.

## Suggestions for the owner (not acted on, outside this scope)
- Promo hard flag: use a recent-window percentile, or flag only plans beyond the training range.
- Intervals: use weekday-aware quantiles to fix the under-coverage on Sundays.
