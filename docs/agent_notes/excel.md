# Excel agent: planner workbook

`excel/forecast_vs_instinct_planning_workbook.xlsx` is built by `scripts/09_build_workbook.py` (openpyxl) and checked by
`tests/test_workbook.py`. To rebuild: `.venv/bin/python scripts/09_build_workbook.py`.
The script only reads finished outputs. It refits nothing.

## Inputs (read only)
| File | Used for |
|---|---|
| `outputs/holdout/champion_intervals.parquet` | stage `forward` (origin 2017-08-15, target days 16 to 29 Aug 2017): daily forecast, P10, P90 for each of the 96 series. Stage `holdout`: champion bias and P10-P90 coverage by family |
| `outputs/holdout/trust_scores.parquet` | origin 2017-08-15: components, hard flags (`src.trust.describe_flags` text). Score, band and priority are recomputed by formulas and tested against the parquet |
| `outputs/holdout_results.json` | leaderboard, champion system, WAPE by family x model, trust validation |
| `outputs/selection.json`, `outputs/intervals.json`, `outputs/trust_weights.json`, `config/scope.json` | champion per family, interval method, frozen trust weights (0.7/0.1/0.1/0.0/0.1), scope |

## Sheets
| Sheet | Content | Formulas |
|---|---|---|
| Summary | Title, tagline, data caveats, 19 key numbers | Every key number is a formula: `SUMIF` on Forecast, `COUNTIF` on Exceptions bands, links to Accuracy, Overrides and Scenario |
| Forecast | One row per series x measure (Forecast / P10 / P90). 288 rows, dates as columns E:R, autofilter on Measure | `S` = `SUM(E:R)` per row. Rows 294 to 296 hold chain totals by measure (`SUMIF`) |
| Accuracy | Leaderboard (6 models), champion system, WAPE by family x 5 models, champion bias and coverage by family, trust validation by band | Rank (`RANK`), best model (`INDEX/MATCH` on `MIN`), best WAPE (`MIN`), champion WAPE (`INDEX/MATCH`), gap, band shares, Review/Accept ratio. Conditional formatting: row minimum in teal, colour scale |
| Exceptions | 96 series sorted by priority. Columns: components, hard-flag reasons, planner action, planner note (input) | score = `MIN(100,MAX(0,SUMPRODUCT(components, weights)))`; band = `IF(flag,"Review",IF(>=80,"Accept",IF(>=60,"Monitor","Review")))`; 14-day forecast = `SUMIFS` on Forecast; priority = `(100-score)*fc/SUM(fc)`; rank = `RANK`. Band colours Accept #60F0D8, Monitor #F5C26B, Review #FF6B8A |
| Overrides | One row per series (same order as Forecast). Override qty and comment are input cells. Reason code uses a list validation with the 5 codes. Override qty only accepts a decimal >= 0 | statistical = `=Forecast!S<row>`; band = `INDEX/MATCH` on Exceptions; final = `IF(F="",E,F)`; difference in units and %; a check column for a missing reason; totals row |
| Scenario | Input block, fixed weights, cleaned inputs, scenario table, cost matrix, recommendation, warnings | Everything below the inputs is a formula (see below) |
| About | Kaggle citation, methodology, sheet guide, colour key, how to reproduce | none |

## Scenario sheet = `src/scenario.py`
- Inputs, labelled "Editable demonstration assumptions, not measured UAE uplift". The defaults match the dashboard
  Scenario Lab (`docs/js/views/scenario_lab.js`):
  - family BEVERAGES (B6, list validation)
  - baseline B7: the family's 14-day champion forecast. Since the compliance round 2 it is in **index-days**, not
    units (see "Compliance round 2" below)
  - change % low/base/high 5/15/30 % (the Ramadan preset)
  - supply B11 = `ROUND(baseline,0)` (same index-days scale)
  - lead time 14 days
  - planning date 15 Aug 2017 and event start 12 Sep 2017 (28 days apart)
  - cost under / over 3 / 1
  
  B7 and B11 hold default formulas in input cells, so typing a number overwrites them.
- Fixed weights low/base/high = 0.25/0.5/0.25 and an order-now window of 3 days (`DEFAULT_WEIGHTS`, `REVIEW_WINDOW_DAYS`).
- Negative inputs are cleaned with `MAX(0, ...)` (as `run_scenario` does).
- Demand = `MAX(0, baseline*(1+pct))`; shortfall, excess and coverage are computed per scenario. Coverage is stored
  as a fraction, while Python reports a percentage.
- Cost matrix = `cu*MAX(0,D_o-Q_p)+co*MAX(0,Q_p-D_o)`, plus a "keep current supply" row. Expected cost uses the
  fixed weights. Max regret = `MAX(cost - column MIN)`.
- Recommendation: `INDEX/MATCH(MIN(key))` with key = `IF(expected <= MIN+1E-9, regret, 1E+300)`. So the lowest
  expected cost wins, a tie goes to the lower max regret, and a remaining tie goes to the order Low, Base, High,
  as `argmin_plan` does. Minimax regret uses the mirror key.
- Timing: order-by date = event start − lead time. Days to the order-by date decide the status: below 0 "late",
  0 to 3 "now", above 3 "later".
- The recommendation sentence is built with `CONCATENATE` and `TEXT(INT(x+0.5),"#,##0")` (= `fmt_int`). It is
  character-identical to `run_scenario()["recommendation"]`. Warnings are formulas too.
- Default result: plan **High**; Base and High tie on expected cost and the tie goes to High because its max regret
  is lower, so the default case exercises the tie-break. The order-by date is 29 Aug 2017, 14 days away, so the status
  is "later". The critical ratio is 75%. (The quantities are now in index-days; the unit figures of the first build are
  no longer published.)

## Tests (`tests/test_workbook.py`)
openpyxl does not calculate, and LibreOffice is not installed. The test therefore includes a small evaluator for the
Excel subset the workbook uses: SUM, MIN, MAX, IF (lazy), AND, OR, NOT, INT, ROUND, TEXT, CONCATENATE, INDEX,
MATCH, SUMPRODUCT, SUMIF(S), COUNTIF, COUNTA, COUNT and RANK. It evaluates the workbook's own formula strings,
including the cross-sheet references.

The tests check:
- Structure: the sheets exist; key cells are formulas; the exact formula strings and references in Scenario;
  list validation on `Overrides!G6:G101` with the 5 codes and decimal validation on F; band colours.
- Overrides logic: blank override keeps the statistical forecast; an override flows to the totals; the reason-code
  check works.
- Scenario: Excel equals `run_scenario` on the defaults and on all 8 shared JS/Python cases in
  `tests/js/scenario_cases.json` (custom weights included). The comparison covers every output field and the
  exact recommendation text.
- `@pytest.mark.data`:
  - Forecast totals per series, measure and chain equal the parquet sums, and the header dates are the forward days.
  - Exceptions score, band, priority, sort order and 14-day forecast equal trust_scores.parquet.
  - Summary key numbers match the source files, and the scenario baseline equals the BEVERAGES parquet sum.
  - Accuracy best model equals `holdout_best`.
  - A rebuild produces identical cell contents. The About sheet is excluded because it carries the build date.

Result: 20 passed (15 without data).

## Notes / caveats
- The workbook stores no cached values (openpyxl). `fullCalcOnLoad` is set, so Excel calculates on open. Some
  previewers, such as Quick Look, may show empty formula cells.
- WAPE and bias are stored as fractions with % formats. By family, bias and coverage are shown for the champion only
  (computed from the holdout rows of champion_intervals.parquet; its per-family WAPE reproduces the JSON exactly).
  holdout_results.json has only WAPE by family for the other models.
- The 14-day total of the P10/P90 rows is a sum of daily bounds, not an interval for the 14-day total. The Forecast
  sheet notes this.
- Font: Montserrat is set by name and falls back to the default font if it is not installed.

## Compliance update: per-series values are an index (Kaggle rule 7B)
- Per-series Forecast/P10/P90 cells on the Forecast sheet are a daily **index**: value / the series' mean daily sales
  over 2015-06-01..2017-02-06 (open days, before the first selection origin) x 100, via `src/indexing.series_base`
  on `data/processed/panel.parquet`. The base is never written. Column S = `AVERAGE(E:R)` = 14-day mean index
  (for P10/P90 the mean of daily bounds).
- Units remain only for aggregates: chain totals by measure (rows 294-296, stored daily sums, S = `SUM`) feeding the
  Summary "Total 14-day forecast (units, all series)", and family 14-day totals (rows 299-306, sum of 12 stores)
  feeding the Scenario baseline `SUMIFS`.
- Exceptions: M = 14-day mean index; N = stored priority rank from the pipeline (priority values and volume shares
  are not written, they would reveal units); A = `RANK(N, ..., 1)`. The Review share of 14-day volume is one stored
  aggregate in `Exceptions!M4`, linked from Summary.
- Overrides: statistical / override / final / difference are 14-day mean index values; the total row is a mean
  (`AVERAGE`). Summary rows renamed "Consensus mean 14-day index after overrides" and "Consensus vs statistical
  (mean index)".
- About sheet: "Data licence" note and "Index scale" definition.
- `tests/test_workbook.py`: evaluator gained `AVERAGE`; data tests recompute the base independently from the panel and
  check index values, chain/family unit totals, priority rank order, and that no per-series unit value (daily,
  14-day, priority) is stored anywhere in the workbook.
- The sections above describe the pre-index layout where they mention SUM totals or units per series.

## Compliance round 2 (portfolio-only units)

Owner rule: per-series values only as % or index; the only unit value in the workbook is the all-96-series 14-day
forecast total (plus the leaderboard MAE/RMSE pooled over all 96 series, labelled as such).
- Forecast: the three chain daily unit rows are gone (they gave 14 linear equations on the bases). Rows 294-296 are the
  equal-weighted mean index of the 96 series per day and measure (`AVERAGEIF`); row 298 holds one stored value,
  "All 96 series, 14-day total (units)", in column S, used by Summary.
- Family rows 301-308 are index-days: per day `AVERAGEIFS` of the family's 12 series' Forecast index, S = SUM over the
  14 days = 14 × the equal-weighted mean of the 14-day mean index (1,400 = 14 average training-period days for an
  average store). The Scenario baseline (default BEVERAGES) reads these; labels say index-days, supply is in the same
  units, costs are per index-day (relative). The recommendation sentences take their quantity word from
  `src/scenario.py` (probed at build time) so the Excel/Python/JS parity tests keep matching.
- Accuracy: leaderboard MAE/RMSE headers say "units, all 96 series pooled". The champion width is now the mean of
  (P90 − P10) / base × 100 over evaluable holdout rows ("index points per series-day") plus the mean of the row ratios
  (P90 − P10) / forecast; the unit width row is gone. The family table holds only percentages.
- Exceptions M4: the volume-weighted Review share is replaced by a live, equal-weighted share of the summed 14-day
  index (`SUMIF`/`SUM` over column M). Priority stays a stored rank.
- Tests: the evaluator gained `AVERAGEIF`/`AVERAGEIFS`; `test_no_unit_values_except_portfolio` scans every stored
  number on every sheet against per-series, per-family, per-day and chain daily unit values (forecast/P10/P90 by
  series, family, date, family × date, trust `forecast_14d`, priority) and allows exactly one copy of the all-series
  total.
