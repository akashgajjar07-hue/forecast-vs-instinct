# Agent A: baselines and backtest runner

## What I built
- `src/models/baselines.py`: `naive`, `seasonal_naive`, `moving_average(window)`, `holt_winters(trend, fit_days=364, n_jobs)`, plus `BASELINES` (name to fn). Every model filters `view` to `date <= origin` itself, so it stays leak-free even if someone hands it an unmasked panel. A test checks this.
- `src/runner.py`:
  - `run_model_stages(name, fn, panel, stages, variant="A", save=True, **params)` runs on top of `run_backtest`. It truncates the panel to the last target date the requested origins need, and it **refuses `holdout`/`forward` unless `allow_final=True`**.
  - `run_final_stages(...)` is for the integrator. It calls `src.freeze.verify_all()` first, then runs with `allow_final=True`. Stage `forward` is the single origin `LAST_ACTUAL` (2017-08-15).
  - `save_backtest` merges into `outputs/backtests/<name>.parquet`. It replaces rows with the same (variant, stage) and keeps everything else, so the holdout and forward rows can be added to the same file later.
  - `load_backtests(models=None)` concatenates the files.
  - `stage_origins` and `Timer` are small helpers.
- `scripts/02_run_baselines.py` (`--n-jobs 4`) writes `outputs/backtests/{naive,seasonal_naive,moving_average,holt_winters}.parquet` (selection + calibration, variant A) and `outputs/baselines_summary.json`.
- `tests/test_baselines.py` has 19 CI tests and 1 `data` test.

## Decisions
- **Seasonal naive:** uses the sales on T+h−7·ceil(h/7). If that day is closed, it steps back 7 days at a time. If a series has no same-weekday history at all, it falls back to the last observed value (this never happens in practice).
- **Moving average:** a closed (NaN) day is skipped, not counted as 0. The window was tuned on selection only, on overall WAPE: 7 → 20.778, 14 → 20.248, **28 → 19.683**, 56 → 20.677. **The chosen window is 28**, the same as the brief's default.
- **Holt-Winters:** additive level + additive weekly seasonality on raw sales. It is fitted on the trailing 364 days with `initialization_method="estimated"`.
  - NaN (closed) days in the fitting window are filled with the value from 7 days earlier. Leading NaNs are filled with the value from 7 days later. The filled values are used for fitting only.
  - Trend was compared on selection only: none → WAPE 14.726 (bias −0.493); additive damped → 14.924 (bias −0.387). **No trend was chosen.**
  - If a fit raises or returns non-finite values, that series falls back to seasonal naive. Each fallback is logged in `HW_FALLBACKS` and in the summary JSON. Both selection and calibration had **0 fallbacks out of 1,728 fits**.
  - Fits run in parallel with `ProcessPoolExecutor(4)`.
- Baselines are stored once, as variant A, as `docs/data_contract.md` specifies.

## Selection results (10 origins × 96 series × 14 days, n = 13,440, no closed target days)
| model | WAPE | bias |
|---|---|---|
| naive | 26.228 | −7.905 |
| seasonal_naive | 17.274 | 0.879 |
| moving_average (28) | 19.683 | 0.377 |
| holt_winters (no trend) | **14.726** | −0.493 |

Holt-Winters is the best baseline in every family except FROZEN FOODS. There, seasonal naive scores 24.936, Holt-Winters 31.811, and Holt-Winters has a bias of +9.0. The full table by family is in `outputs/baselines_summary.json`.

## Runtimes (MacBook Air, 4 workers)
Measured in seconds:
- MA tuning: 0.71
- Holt-Winters trend comparison: 15.08
- Final runs: naive 0.35, seasonal naive 0.32, MA 0.33, Holt-Winters 3.66

The whole script took about 21 s wall clock.

## For the integrator
- For the holdout, call `run_final_stages("holt_winters", B.holt_winters, panel, ["holdout"], trend=None, n_jobs=4)`, and similarly for the other baselines (MA with `window=28`). It **requires all three freeze files** (`selection`, `intervals`, `trust_weights`). If the forward run should not need the trust weights, call `run_model_stages(..., ["forward"], allow_final=True)` directly.
- The parquet files hold only the `selection` and `calibration` stages. No target date is after 2017-06-27, and the data test checks this. Once holdout rows are added, that test's "no holdout" assertion has to be relaxed.
- After using `n_jobs>1` in a script, call `B.shutdown_pool()` once at the end. The script needs an `if __name__ == "__main__"` guard because macOS uses spawn.
