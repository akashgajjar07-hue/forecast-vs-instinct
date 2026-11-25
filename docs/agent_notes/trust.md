# Trust Index agent notes

## What I built

| Path | Purpose |
|---|---|
| `src/trust.py` | Pure functions: `components`, `score`, `band` / `bands_for`, `priority`, `realised_window_wape`, `standard_candidates` / `grid_candidates`, `compare_weights`, `choose_weights`, `validate`, `describe_flags`. Constants in `CONSTANTS`. |
| `scripts/06_trust_calibrate.py` | Verifies the `selection` and `intervals` freezes (waits for `outputs/intervals.sha256`), scores the 8 calibration origins, compares weight candidates, freezes `trust_weights`, writes `outputs/trust_calibration.json`. Asserts that no holdout or forward row is in memory. |
| `tests/test_trust.py` | CI tests on synthetic data + 2 `@pytest.mark.data` tests on the real calibration output. |
| `docs/agent_notes/trust_methodology.md` | Methodology section draft (exact formulas) for METHODOLOGY.md. |

## Pre-stated weight-choice rule (written BEFORE any calibration result was computed)

Objective: Spearman rho between the trust score and the realised champion WAPE of the same
14-day window, over all (series, calibration origin) windows with non-zero actual volume
(96 x 8 = 768 at most). More negative is better. Reported as is (sign kept).

Candidates: (a) default 30/25/20/15/10, (b) equal 20 each, (c) volatility-only (stability alone),
(d) every weight vector on the 0.1-step simplex over the 5 components (1001 vectors; includes (b)-like
and single-component corners).

Rule:
1. Rank all candidates by rho (most negative first).
2. If the best candidate is not the default, adopt it ONLY if
   (i) its rho is at least **0.03** more negative than the default's on all calibration windows, AND
   (ii) it is also more negative than the default on EACH half of the calibration origins
       (first half 2017-04-25..05-16, second half 05-23..06-13).
3. Otherwise keep the default. The margin and split guard against picking a grid point that is
   best by noise: with 1001 candidates the maximum rho is optimistically biased.

**Amendment (made after a mechanics dry run on BASELINE-ONLY champions, before LightGBM and the
real freezes existed; disclosed here for honesty):** the dry run showed that data completeness is 100
for every calibration window, so it is a constant: its weight only shifts all scores and Spearman
cannot identify it (1001 grid points collapse into ties that differ only in completeness weight, and
those ties give very different band distributions). Two mechanical additions, not tuned on outcomes:
- a component that is constant on calibration keeps its DEFAULT weight; grid points that move it are
  dropped (`trust.identifiable_candidates`); default / equal / volatility-only stay as they are;
- exact ties in rho (within 1e-6) are broken by the smallest Euclidean distance to the default weights.
The margin (0.03) and the both-halves condition are unchanged.

Thresholds: keep 80 / 60. Only if a band is EMPTY on calibration under the chosen weights
(after hard flags), replace the two thresholds by the 2/3 and 1/3 quantiles of calibration scores,
rounded to the nearest integer, and document it. Otherwise no threshold tuning.

## Normalisation constants (chosen before calibration, from selection-stage data only)

- `wape_cap = 50 %`: per-series champion WAPE on selection origins (baselines; LightGBM not yet in)
  ranged 7.4-44.5 % (median 16.4, p95 33.0). A 50 % cap keeps the accuracy score linear across
  the whole observed range (never saturates) while a 50 % WAPE series scores 0.
- `cov_tol = 40 pp` (spec value): the score hits 0 at coverage <= 40 %. Over-coverage can be at most
  20 pp (100 %), so it costs at most 50 points: an over-wide band is judged less harmful than an
  over-narrow one. Symmetric formula as specified.
- `cv_cap = 1.0`: 56-day daily CV at 2017-04-11 ranged 0.16-0.70 (median 0.25), so the cap does not bind.
  Daily CV includes the weekly swing (deliberate: the spec asks for raw CV; the weekly pattern is
  shared by most series so it mainly shifts the level).
- `agree_cap = 0.5` (spec value).
- Lookback 56 days; up to 8 completed windows; completeness flag < 80 %.

## As-of rules (what is known at origin T)

- Accuracy / interval history: champion rows of backtest origins o with o + 14 <= T only (last 8).
  The champion map itself is the frozen selection, which is known at every calibration origin
  (the last selection target date 2017-04-25 <= first calibration origin 2017-04-25).
- Interval quantiles: calibration origins use quantiles re-fitted on SELECTION champion residuals
  (`fit_quantiles(champion selection rows)`), because the calibration quantiles in `outputs/intervals.json`
  are fitted on calibration windows that are not complete at T. Holdout / forward origins use the frozen
  calibration quantiles from `outputs/intervals.json` (hash-verified).
- Stability / completeness: panel rows with date in (T-56, T].
- Agreement: the five models' forecasts at T (forecasts only, never actuals).
- Promo flag: the planned promo count on T+1..T+14 (the known promo plan, as in LightGBM variant A)
  vs the p95 of the 14-day rolling mean promo count on dates <= T (whole modelling history).
- Closure flag: calendar only (25 Dec / 1 Jan).
- Test `test_components_use_only_information_at_T` perturbs every actual / sale after T (and later
  origins' forecasts) and asserts the components at T are bit-identical.

## Results

Run: `scripts/06_trust_calibrate.py` after `selection` / `intervals` freezes verified (all 8 family
champions = LightGBM variant A). 768 calibration windows (96 series x 8 origins), all with non-zero volume.
Completeness = 100 on every window -> constant, held at its default 0.10; identifiable grid = 220 points.

| Candidate | rho (all) | rho first half | rho second half |
|---|---|---|---|
| default 0.30/0.25/0.20/0.15/0.10 | -0.5141 | -0.4427 | -0.6099 |
| equal | -0.4781 | -0.4020 | -0.5845 |
| volatility only | -0.4488 | -0.3629 | -0.5833 |
| **grid 0.7/0.1/0.1/0.0/0.1 (chosen)** | **-0.6501** | -0.5887 | -0.7306 |

(order: accuracy/interval/stability/agreement/completeness). Default ranks 62nd of the grid.
Single-component rho: accuracy -0.642, interval -0.206, stability -0.449, agreement -0.190.
Rule outcome: the best grid point beats the default by 0.136 (>= 0.03) and on both halves -> adopted.
Frozen weights: accuracy 0.70, interval 0.10, stability 0.10, agreement 0.00, completeness 0.10;
bands kept at 80 / 60 (no band empty). trust_weights sha256 dc75cfb2...54b6.

Calibration band distribution (with hard flags): Accept 261, Monitor 368, Review 139
(score only: 304 / 417 / 47). Hard flags: promo_above_p95 on 92 windows (12 %), no other flag fired.
Realised calibration WAPE by band (in-sample for the weights): Accept 9.31, Monitor 15.33, Review 18.62.
Score range 28.3-84.9, median 78.7.

## Concerns
- The chosen index is dominated by historical accuracy (70 %); agreement gets 0. The index is largely
  "this series was hard recently, it will be hard again". The holdout test against volatility-only and
  the honest verdict are what will tell whether this holds.
- Choosing among 220 grid points is optimistic: the calibration rho (-0.65) is an upper estimate of
  holdout rho.
- Score max is ~85, so Accept requires near-best accuracy; the band split is driven mostly by WAPE_hist.
- promo_above_p95 fires on 12 % of calibration windows because promo counts trend up over 2015-2017, so
  the full-history p95 is low. Kept as specified (LightGBM trains on that full history, so it is the
  right extrapolation reference), but it puts low-error windows into Review.
- Completeness carries no information on this data (always 100); its 10 % is a constant +10.
- Selection-fitted quantiles are scored partly on the same selection windows (in-sample coverage).

