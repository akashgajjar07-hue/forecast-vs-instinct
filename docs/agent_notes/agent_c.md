# Agent C: Demand Intelligence / EDA

## What I built

| Path | Purpose |
|---|---|
| `sql/01..14_*.sql` | One DuckDB query per business question; each file opens with a `-- Business question:` comment block, the definitions it uses, the views it expects and its `$params`. |
| `src/eda.py` | `connect()` registers views (`panel`, `holidays`, `transactions`, `oil`, `train`) over the parquet/CSV files (or over DataFrames in tests); `run_sql()` binds only the `$params` a file uses; builders for `demand.json`, the analogue and extras. |
| `scripts/04_build_eda.py` | Writes `outputs/eda/demand.json` (validated against `SCHEMAS["demand.json"]`, 42.6 KB), `outputs/eda/ecuador_analogue.json` (validated against `scenario.json → ecuador_analogue`), `outputs/eda/eda_extras.json`. |
| `notebooks/01_demand_intelligence.ipynb` | Executed, beginner-friendly demand story (plotly, ~0.6 MB); every number comes from the `sql/` files through `src/eda.py` (two small inline queries on the same `panel` view are labelled as such). |
| `tests/test_eda.py` | 7 CI-safe tests (SQL headers, hand-computed analogue on a synthetic frame, cutoff handling, dow mean = 1, synthetic weekly reconciliation, param check) + 4 `@pytest.mark.data` tests (schema validation, size, dow mean, weekly totals vs panel to 1e-6, analogue validation). |

SQL files: 01 weekly family totals · 02 day-of-week profile · 03 volatility · 04 promo bins · 05 promo frequency ·
06 holiday effect · 07 earthquake · 08 payday · 09 transactions vs sales · 10 Ecuador holiday analogue ·
11 PRODUCE recording gap · 12 oil vs sales (monthly) · 13 data overview · 14 weekday promo profile.

## Leakage guard

Every number uses history **2015-06-01 .. 2017-06-27** (the last calibration target date, read from
`config/split_calendar.json`). The holdout (2017-06-28 .. 2017-08-15) is never read. The only exception is
`11_produce_recording_gap.sql`, which reads raw 2013-2015 history *before* the window on purpose (data-quality check).
Weekly values cover complete Monday-Sunday weeks only, so the last published week is 2017-06-19 .. 06-25. Since the
compliance round 2 they are published as the equal-weighted family index (`weekly.family_index`), not unit totals.

## Definitions

- **Closed day**: sales NULL (25 Dec, 1 Jan chain-wide); excluded from every mean, never treated as zero.
- **dow_profile**: per series, weekday mean index; per family, the plain mean over its 12 series, ÷ the mean of the
  7 weekday values (mean = 1); national days off excluded (equal-weighted since the compliance round 2).
- **Volatility**: `cv` = std/mean of daily sales on open days (includes the weekday swing); `cv_weekly` on complete
  weeks; `zero_share` = share of open days with sales = 0.
- **Promo bins**: promoted-item count bins 0, 1-5, 6-15, 16-30, 31-60, 61+ (store-days). `mean_index` = sales ÷ the
  same store-family's mean for that calendar month (removes store size and slow trend). Association only.
- **Holiday effect (06)**: holiday = open store-day with a national day off (Holiday not transferred, Transfer, Bridge,
  Additional); normal = no day off of any locale matched to the store's city/state, no Work Day, no Event, outside the
  earthquake month. `n_holiday_days` = distinct dates. Also `change_pct_excl_additional`.
- **Ecuador holiday analogue (10)** — the window, precisely:
  - Holiday dates H: National rows with (`type='Holiday'` and not transferred) or `type in ('Transfer','Bridge')`,
    within the window. Additional days, Events and Work Days are not holiday dates.
  - Target t = h (+ offset; offset −1 = day before, used only in extras, skipping targets that are themselves days off).
  - Baseline per store: mean of sales on **t−14, t−7, t+7, t+14** (same weekday, ±1–2 weeks), keeping only days that
    are open, inside the window, not a day off / Work Day / Event of any locale for that store and outside the quake
    month; the store needs ≥ 2 valid baseline days and to be open on t.
  - Per holiday × family: `100 × (Σ_stores sales_t ÷ Σ_same stores baseline − 1)`.
  - Family value: simple mean over holidays (each holiday counts once). 19 holidays qualify (25 Dec / 1 Jan are
    closures; 1 May 2016 drops because all its baseline days are in the quake month).
- **Earthquake (07)**: mean daily sales per open store-day, 30 days before 2016-04-16, 2016-04-16..05-15, 30 days after;
  Manabí = stores 53 (Manta), 54 (El Carmen).
- **Payday (08)**: index = sales ÷ same store-family, same month, same weekday mean on non-payday days; payday = 15th or
  month end; day_after = 16th or 1st; days off, Work Days, Events, quake month excluded.

## Key findings (all from the queries; observed sales; associations, not causation)

*(Compliance round 2: the numbers in this list are on the equal-weighted index scale, i.e. per series first, then the
plain mean over a family's 12 series; no unit levels are published.)*

1. **Weekly rhythm**: Sunday is the peak for 6 of 8 families (index 1.27–1.38) and Thursday is the trough for 7 of 8
   (0.76–0.82; FROZEN FOODS is lowest on Monday); FROZEN FOODS peaks on Saturday (1.58).
   PRODUCE peaks **mid-week, Tuesday above all (1.24; Wednesday 1.05)**; produce promotions cluster on Tue/Wed (promo
   index 365 and 264 vs about 13 on other days, 100 = each series' own mean promoted-item count), so the mid-week peak
   is *associated with* a mid-week produce promotion rhythm.
2. **FROZEN FOODS is the planning problem**: a FROZEN FOODS series sells on average 30.0 % of its in-window sales in
   December (other families 7.7–9.9 %); median weekly CV 0.55 vs 0.09–0.18 for the other families; 7 of the 8 most volatile series are FROZEN
   FOODS (store 38's biggest day is ~55× its mean day).
3. **Promotions**: once store size and month are controlled for (`mean_index`), the promoted-item count is only weakly
   associated with sales for GROCERY I, BEVERAGES, CLEANING and DAIRY (bins within 0.97–1.04). The association is
   larger for PRODUCE (61+ items: 1.49) and FROZEN FOODS (6–15: 1.38, 16–30: 1.68), partly calendar-driven.
   No price/discount data: no causal uplift claim.
4. **Holiday analogue** (weekday-matched, 19 holidays): mean change on national days off ranges from **+7.6 %
   (PRODUCE) to +27.8 % (PERSONAL CARE)**; BEVERAGES +17.3 %, GROCERY I +11.9 %. The spread is wide: Carnaval Monday
   and Good Friday are negative for most families (family minima −20 % to −47 %), while the family maxima (+44 % to
   +89 %) come from 1 May 2017, 2 Jan 2017 and 3 Nov 2015. The day *before* a holiday averages negative for 7 of 8 families (13 eves).
5. **Raw holiday comparison** is dominated by the Christmas-run-up Additional days (FROZEN FOODS +227 %); excluding
   them, national days off are associated with −0.6 % (FROZEN FOODS) to +17.3 % (PERSONAL CARE).
6. **Earthquake** (equal-weighted index): the other 10 stores show a temporary jump (quake month +18.0 % vs the month
   before; the month after +4.0 %). The two Manabí stores show a lasting step (+58.8 % in the quake month, +73.9 % in the
   month after). On the equal-weighted index of all 96 series the biggest week of the window is 2016-12-19 (2.20× the
   median week); the earthquake week ranks lower once every store counts the same.
7. **Start of month**: days 1–5 sit 8–16 % above a normal same-weekday day; days 8–14 sit 2–7 % below. The 15th itself
   is not a standout for most families; the "day after payday" (16th / 1st) index is 1.03–1.14.
8. **Transactions** track sales closely (pooled daily r = 0.897; per store 0.73–0.92) — EDA only, not a feature.
   PRODUCE zero share on open store-days was 48.3 % before 2015-06-01 and 0.0 % from then on (recording gap).

## Caveats

- Observed sales, not unconstrained demand (no stock-out data).
- Promo = count of promoted items, no depth; all promo results are associations.
- Holiday analogue averages hide large holiday-to-holiday spread (see min/max in `ecuador_analogue.json`); it is an
  Ecuadorian analogue, not a measurement of any other market.
- Earthquake and payday results are single-event / calendar associations with no control group.
- `n_holiday_days` in `demand.json` counts distinct dates (34 incl. Additional days); `n_holiday_store_days` is extra.
- `ecuador_analogue.json` items carry extra fields (`median_pct`, `min_pct`, `max_pct`); the schema allows extras.
- `demand.json` items carry extra fields (`cv_weekly`, `mean_index`, `n_stores`, `corr_promo_sales`, ...); allowed by
  the schema (`additionalProperties: true`).
