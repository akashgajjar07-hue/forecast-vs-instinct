# Compliance agent: Kaggle rule 7B (no redistribution of competition data)

Built on the Kaggle Store Sales – Time Series Forecasting competition data (Corporación Favorita). Raw competition
data is not redistributed; the site shows only aggregated/indexed values and model outputs. Reproduce by downloading
the data from Kaggle after accepting the competition rules.

No model, frozen file (`outputs/*.json` with a `.sha256`, `outputs/holdout_results.json`, `outputs/holdout_run.json`),
metric, threshold or trust score was changed. WAPE, bias, MAE, RMSE, coverage, trust scores and bands are identical;
the export still verifies the freeze hashes and the what-if refit still reproduces the saved forward forecast
(max |diff| = 0).

## Audit list (as verified by the integrator) and resolution

| # | Finding | Resolution |
|---|---|---|
| 1 | `docs/data/series/*.json`: `history.sales`, `holdout[].rows[].actual`, `forward.rows[].onpromotion`, `whatif.rows[].promo_total`, `whatif.hist_max_promo / hist_p95_promo` | All replaced by index fields (mapping below); promo as a promo index |
| 2 | `overview.json → outlook.history[].actual` (chain total, units) | `actual_index` on the chain base; forecast/band too |
| 3 | `demand.json`: `volatility[].mean`, `promo_frequency[].max_promo_count`, `mean_promo_count` | Removed (also from `outputs/eda/demand.json` via `src/eda.py`); promo bins pooling < 3 stores have `mean_sales`/`median_sales` = null |
| 4 | `scope.json` (config + docs): `series[].mean_daily_sales`, `mean_promo_count` | Removed; `src/scope.py` no longer computes them (store selection unchanged: the stats are computed after selection) |
| 5 | Notebook 02 masked-view demo, notebook 03 worked example, `STUDY_GUIDE.md` worked example | Index / promo index values (`scripts/build_notebooks.py`, re-executed); notebook 01 hover/tables with per-series means removed; study guide numbers recomputed on the index scale |
| 6 | `scripts/10_build_presentations.py` used `outlook.history` actuals | Uses the indexed outlook (the forward vs last-14-days ratio is scale-free); review volume share from local `trust_scores.parquet` |
| 7 | Per-series units next to per-series index (trust, sop, series files, workbook) | trust/sop/series in index; `priority` (share of volume) replaced by `priority_rank`; workbook per-series values in index |

Extra items found during the work:
- `outputs/window_band_results.json → forward[]` held per-series 14-day forecast/P10/P90 in units (committed). Moved to
  the git-ignored `outputs/holdout/window_band_forward.parquet` (one-time migration of identical numbers;
  `scripts/07b_window_band.py` now writes the parquet).
- `trust.latest[].priority` and `overview.top_exceptions[].priority` = (100 − score) × share of the 14-day volume:
  with the chain total in units this gave per-series units. Only the rank is published now.
- S&OP summary sentences quoted per-series units and raw promo counts; they now use the index / promo index. The chain
  discussion point no longer quotes the last-14-days actual total in units.
- README Q6 line with the last-14-days chain actual total in units: number removed.

## Index definition (`src/indexing.py`)

- **Series base** = mean observed daily sales of the series on open days (`closed == 0`, sales observed) from
  2015-06-01 to 2017-02-06 (model start to the day before the first selection origin, `config/split_calendar.json`).
- **Index** = value ÷ base × 100 (100 = an average training-period day). Daily values rounded to 1 dp in series files.
- **14-day quantities** (trust, S&OP, what-if totals) = mean daily value over the 14 days ÷ base × 100 (2 dp in trust/sop).
- **Weekly index** = mean daily sales over the open days of a Monday week ÷ base × 100 (weeks 2015-06-01 .. 2017-08-07).
- **Chain base** = mean daily total of all 96 series over the same period, days when all 96 were open.
- **Promo index** = promoted-item count ÷ the series' mean promoted-item count on open days 2015-06-01 .. forward
  origin (2017-08-15) × 100; `null` if that mean is 0 (no series has a zero mean).
- Bases are never published. Tests check that no published number equals a base (3–4 dp).

## Before / after field mapping

| File | Before | After |
|---|---|---|
| series | `history.sales` | `history.index` (+ new `history.week_start`, `history.weekly_index`) |
| series | `holdout[].rows[].actual / p10 / p90 / models` | `actual_index / p10_index / p90_index / models_index` |
| series | `forward.rows[].forecast / p10 / p90 / onpromotion / models` | `forecast_index / p10_index / p90_index / promo_index / models_index` |
| series | `whatif.hist_max_promo / hist_p95_promo` | `hist_max_promo_index / hist_p95_promo_index` |
| series | `whatif.rows[].promo_total / forecast_total / daily` | `promo_index` (14-day mean) / `forecast_index` (14-day mean) / `daily_index`; `change_pct`, `extrapolation`, `below_history` unchanged |
| series | – | `scale`, `promo_scale` (labels) |
| overview | `outlook.history[].actual` | `outlook.history[].actual_index` |
| overview | `outlook.forecast[].forecast / p10 / p90` | `forecast_index / p10_index / p90_index`; `outlook.scale` |
| overview | `top_exceptions[].priority` | `top_exceptions[].priority_rank` |
| trust | `latest[].forecast_14d / p10_14d / p90_14d / priority` | `forecast_14d_index / p10_14d_index / p90_14d_index / priority_rank`; `scale` |
| sop | `rows[].stat_forecast / p10 / p90` | `stat_index / p10_index / p90_index`; `scale`; notes rewritten |
| sop CSV | `statistical_qty, override_qty, final_qty` | `statistical_index, override_index, final_index` (localStorage key `fvi.sop.overrides.v2`) |
| demand | `volatility[].mean`; `promo_frequency[].mean_promo_count / max_promo_count` | removed |
| demand | `promo_bins[].mean_sales / median_sales` | null when the bin pools < 3 stores |
| scope (config + docs) | `series[].mean_daily_sales / mean_promo_count` | removed |
| outputs | `window_band_results.json → forward[]` | `outputs/holdout/window_band_forward.parquet` (git-ignored) |
| workbook | per-series daily/14-day forecast, P10, P90 in units; priority values; override qty | index (Forecast, Exceptions, Overrides); priority rank; chain and family totals stay in units |

Unchanged in units (aggregates across many series, per the design): `demand.weekly.by_family` (12 stores summed),
`demand.holiday_effect`, `scenario.families[].baseline_14d` (12 stores), `overview.kpis.forecast_14d_total` (96 series),
`arena` MAE/RMSE, `intervals.holdout.mean_width` (pooled).

## Files changed

- Pipeline: `src/indexing.py` (new), `src/export.py`, `src/schemas.py`, `src/eda.py`, `src/scope.py`,
  `scripts/07b_window_band.py`, `scripts/09_build_workbook.py`, `scripts/10_build_presentations.py`,
  `scripts/build_notebooks.py`, `scripts/make_fixtures.py`, `scripts/playwright_check.py`.
- Data: `docs/data/*.json`, `docs/data/series/*.json`, `docs/data/scope.json`, `config/scope.json`,
  `outputs/eda/demand.json`, `outputs/window_band_results.json`.
- Site: `docs/js/core.js`, `docs/js/views/{workspace,overview,explorer,exceptions,sop,methodology}.js`, `docs/index.html`
  (footer licence line), `docs/assets/screenshots/*` (regenerated).
- Docs: `README.md` (Data licence section, citation), `METHODOLOGY.md` (§11), `docs/data_contract.md`,
  `docs/agent_notes/{export,excel,notebooks}.md`, `STUDY_GUIDE.md`, this file.
- Tests: `tests/test_reconciliation.py` (index recomputed with independently computed bases), `tests/test_workbook.py`,
  `tests/test_no_raw_data.py` (new).
- Artefacts: notebooks 01–03 (re-executed), `excel/forecast_vs_instinct_planning_workbook.xlsx`, `presentations/*.pdf|pptx`.

## Round 2: closing the reconstruction paths (owner rule)

Rule set by the owner: per-series values only as % or on the index scale; the only unit values allowed are
portfolio-level totals (all 96 series combined). Round 1 left the bases identifiable from the published files
(residual risk 1 of round 1); round 2 closes those paths.

### What changed

| Area | Before (round 1) | After (round 2) |
|---|---|---|
| `demand.json` weekly | family totals in units (12 stores summed) | `weekly.family_index`: equal-weighted family index (plain mean of the 12 series' weekly index) |
| `demand.json` dow_profile | chain-level (volume-weighted) | equal-weighted over series |
| `demand.json` holiday_effect | `holiday_mean` / `normal_mean` in units | % change and counts only |
| `demand.json` promo_bins | `mean_sales` / `median_sales` (null for small bins) | `mean_index` and counts only |
| overview outlook | chain total ÷ chain base, with the chain P10–P90 band | equal-weighted mean index of the 96 series (history + forecast); no band; caption gives the chain band's holdout coverage (83.3 %, 84 days) and why it is not drawn |
| arena MAE / RMSE | units, every slice | index points (per row |A − F| ÷ series base × 100), every stage and slice; `mae_unit` |
| intervals mean width | units per day | index points, plus `mean_width_pct` (Σ width ÷ Σ forecast) |
| scenario baselines | family 14-day totals in units | index-days: 14 × equal-weighted mean 14-day forecast index of the family's 12 series (1,400 = 14 average days); Scenario Lab labels, supply and the recommendation sentence (JS and Python) say index-days |
| series holdout windows | actual + band + all six model lines | actual + band + champion only (`forecast_index`); challengers stay on the forward window. With per-series challenger lines, the volume-weighted holdout WAPE / bias slices of every model gave > 400 homogeneous equations |
| S&OP texts | Review share of forecast volume; chain total vs last 14 days (volume-weighted) | Review share of series; equal-weighted average-series index vs the last 14 days (−2.4 %) |
| `outputs/eda/eda_extras.json` | family totals and volume shares, trend sums, peak-week units, earthquake means, weekday sales/promo means, oil-month chain sales, per-store transaction means and sales per transaction, per-holiday-day 12-store rows | removed or converted to the equal-weighted index / promo index; per-holiday-day rows dropped (summaries kept) |
| Excel workbook | per-family totals and chain daily totals in units | family index-days; equal-weighted portfolio index rows; one stored unit value (all-series 14-day total) plus the portfolio leaderboard MAE/RMSE |
| Notebooks, presentations, Markdown | family/store-level unit values | equal-weighted index / % (see `docs/agent_notes/notebooks.md`, `presentations.md`) |
| Workspace | – | caption under the promotion what-if slider: "Model association, not causal uplift — the data has no price or discount depth" |

No model, frozen file, WAPE, bias, coverage, trust score, band, threshold or weight changed (`arena.json` WAPE/bias
values and `intervals.json` coverages are identical to round 1; the freeze hashes verify).

### Acceptance test: reconstruction attack (`tests/test_no_reconstruction.py`, `@pytest.mark.data`)

The test collects every linear equation in the 96 unknown bases that the current tracked files allow: the KPI total in
units, any unit-valued MAE / width / weekly / baseline aggregate, the volume-weighted WAPE and bias of every arena
holdout slice whose rows can be rebuilt from the series files (overall, family, store, horizon, horizon bucket, week
bucket), `holdout_results.json` WAPE by family and its leaderboard MAE and mean width (frozen, units), and the S&OP
Review-share sentence. Rows are scaled to unit norm and solved by least squares.

| State | Equations | Rank (of 96) | Median relative base error |
|---|---|---|---|
| Round 1 output (index design, before this round) | 1,620 | 96 | 0.21 % (p10 0.02 %) |
| Round 2 output (now) | 89 | 83 | 1,728 % (p10 241 %): not identifiable |
| Round 2 + oracle holiday and promo slice membership (worst case: attacker knows each holdout row's holiday / promo flag) | 97 | 89 | 2,718 % |

The test fails if the rank reaches 96 or the median error falls below 20 %.

## Residual risks

1. **Margin.** The system is 13 ranks short of identification (7 with oracle holiday / promo membership). Any new
   volume-weighted statistic over the published holdout rows (a WAPE by origin, by store × family, by trust band with
   window-level bands, …) or any new unit aggregate adds rank. Publish such statistics equal-weighted or not at all, and
   rerun the test.
2. **Weak prior information** (not linear, not in the test): bases are positive and families have similar levels
   across stores; an attacker with priors could narrow the null space somewhat. The frozen `holdout_results.json`
   (portfolio MAE/RMSE in units, mean width in units) and `ecuador_analogue.json` (per-family min/max of per-holiday
   volume-weighted ratios) are kept unchanged.
3. **Portfolio totals in units** (chain 14-day forecast total, portfolio MAE / RMSE / width) remain by design.
4. Index 0 equals raw 0 (zero sales / zero promoted items are visible as zero).
5. Git history still contains the earlier row-level and unit files; removing them needs a history rewrite.
6. `presentations/demo.webm` / `demo.gif` predate both rounds (re-record with `scripts/11_record_demo.py`).
7. The deployed GitHub Pages site serves the old JSON until the integrator pushes.
