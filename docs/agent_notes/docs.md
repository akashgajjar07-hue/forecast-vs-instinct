# Documentation agent notes

## Files written
| Path | Purpose |
|---|---|
| `README.md` | Case study: problem, scope, six business questions with numbers, findings, screenshots, pipeline diagram, reproduction, repo map, limitations, author, citation |
| `METHODOLOGY.md` | Data traps, scope, split calendar, leakage controls, models, features, tuning, metrics, conformal intervals, Trust Index formulas and weight rule, holdout, promo what-if, Scenario Lab formulas, S&OP and simulated FVA |
| `MODEL_CARD.md` | LightGBM champion card |
| `LIMITATIONS.md` | Full limitations list |
| `STUDY_GUIDE.md` | Concepts with real numbers, the 36\|DAIRY worked example, 15 interview questions, next steps |
| `data/data_dictionary.md` | Raw files, panel, backtest and holdout tables, outputs, published JSON |
| `reports/VALIDATION_REPORT.md` | Tests, data quality, backtest design, leakage tests, reproducibility, interval and Trust validation, and a "pending" independent review section |
| `LICENSE` | MIT for code (2026 Akash Gajjar) + data notice |

## Sources
Every number was read from `outputs/*.json`, `config/*.json`, `docs/data/{overview,sop,meta,scenario}.json`, the executed
notebooks 02 and 03 (cell outputs), `data/processed/panel.parquet` (row counts) and the agent notes. Rounding follows the
source (WAPE to 2 dp in prose, 4 dp in the tuning table).

## Things the integrator should know
1. **Three `@pytest.mark.data` tests fail** (run 2026-11-26): `test_baselines.py::test_saved_real_baseline_backtests` and
   `test_features_leakage.py::test_real_lightgbm_backtests[A|B]` still assert that the saved backtests contain only
   selection/calibration rows. The holdout script appended holdout/forward rows as designed. Relax these assertions.
   `pytest -m "not data"`: 115 passed, 2 skipped. `pytest -m data`: 29 passed, 3 failed.
2. **Q6 / S&OP numbers come from `docs/data/sop.json` and `overview.json` as of 2026-11-25T21:12Z** (forecast 2,119,685;
   Review 22 / Monitor 39 / Accept 35; 13 promo-flagged series; FVA −0.45 pp). If the export
   is regenerated, re-check README Q6, METHODOLOGY §11, STUDY_GUIDE §1.17 and Part 5.
3. `presentations/demo.gif` does not exist yet; README references it as requested. The screenshots exist.
4. LinkedIn is a placeholder ("link to be added") in README.
5. Reproduce section: `07_holdout_and_forward.py` is documented with `--rerun`, because `outputs/holdout_run.json` is
   committed and a plain run would refuse. The Kaggle slug `store-sales-time-series-forecasting` is the public
   competition name; it is not referenced anywhere else in the repo.
6. Kaggle rules were not confirmed during the build (DECISIONS #4). README and LIMITATIONS ask the owner to confirm.
7. Runtimes for steps 1, 4, 5, 6 and 9 were not recorded anywhere, so the table leaves them blank.
