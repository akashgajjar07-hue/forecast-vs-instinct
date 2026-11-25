## Forecast Trust Index

The Trust Index answers one planner question at forecast origin T: *how much should I trust this
series' 14-day forecast?* It is a 0–100 **index**, not a probability: a score of 85 does not mean
"85 % chance the forecast is right". It ranks series so that review time goes where it is most needed.

Every input is known at the end of day T. Nothing after T is read (a test perturbs every sale and
actual after T and checks the components are unchanged).

### Components (each 0–100, higher = more trustworthy)

Notation: *series* = store × family; *window* = the 14 target days T+1..T+14 of one origin;
*evaluable day* = open (not closed) day with an observed sale. WAPE = Σ|A − F| / ΣA × 100.

| Component | Weight (default) | Formula |
|---|---|---|
| Historical accuracy | 30 % | `100 × (1 − min(WAPE_hist, 50) / 50)` |
| Interval reliability | 25 % | `100 × (1 − min(|cov_hist − 80|, 40) / 40)` |
| Demand stability | 20 % | `100 × (1 − min(CV_56, 1)) × (1 − zero_share_56)` |
| Model agreement | 15 % | `100 × (1 − min(cv_models / 0.5, 1))` |
| Data completeness | 10 % | `100 × n_ok_56 / 56` |

- **WAPE_hist**: the champion model's WAPE for this series, pooled (ratio of sums) over the evaluable
  days of the last ≤ 8 backtest origins o whose whole window is already known (o + 14 ≤ T).
  Cap 50 %: per-series champion WAPE on selection backtests ranged 7–45 %, so the score is linear over
  the observed range and a 50 % WAPE series scores 0.
- **cov_hist**: share (%) of evaluable actuals in those same windows that fell inside the champion's
  P10–P90 band. The band uses only quantiles that existed at T:
  - calibration origins → quantiles re-fitted on **selection** residuals (the calibration quantiles
    are built from windows that are not complete yet at a calibration origin);
  - holdout and forward origins → the frozen calibration quantiles in `outputs/intervals.json`
    (SHA-256 verified).
  Under-coverage is penalised more than over-coverage: 40 % coverage scores 0, 100 % scores 50.
- **CV_56, zero_share_56**: standard deviation ÷ mean (population SD) of daily sales, and share of days
  with zero sales, over evaluable days in the 56 days (T−55..T). The daily CV includes the weekly pattern.
- **cv_models**: standard deviation ÷ mean (population SD) of the five models' 14-day forecast totals
  for this window: naive, seasonal naive, moving average, Holt-Winters, LightGBM (known promo plan, variant A).
  High disagreement between reasonable models signals an uncertain forecast.
- **n_ok_56**: number of evaluable days among the 56 calendar days T−55..T (closed and missing days count against it).

### Score, bands and hard flags

`score = Σ_k w_k × component_k`, with Σ w_k = 1 (frozen weights in `outputs/trust_weights.json`).

| Band | Rule |
|---|---|
| Accept | score ≥ 80 and no hard flag |
| Monitor | 60 ≤ score < 80 and no hard flag |
| Review | score < 60, **or any hard flag** |

Hard flags (each forces Review regardless of the score):
- `low_completeness`: data completeness < 80 % in the 56-day lookback.
- `holiday_closure`: a known chain-wide closure (25 Dec or 1 Jan) falls inside T+1..T+14.
- `promo_above_p95`: the planned promotion intensity for the window (mean promoted-item count over
  T+1..T+14, the known promo plan) is above the series' 95th percentile of its 14-day rolling mean
  promo count over all history ≤ T: the forecast would be extrapolating.
- `missing_<component>`: a component could not be computed (e.g. no completed backtest window).

**Priority** (inbox sort key, higher = look first) = `(100 − score) × (series' 14-day champion forecast ÷ total 14-day champion forecast)`.

### How the weights were chosen (calibration origins only)

The default 30/25/20/15/10 weights are hypotheses. On the 8 calibration origins we compared:
the default, equal weights, a volatility-only index (stability alone) and every weight vector on a
0.1-step grid (1,001 vectors). Objective: Spearman rank correlation between score and the realised
champion WAPE of the same window (96 series × 8 origins); a good index has a strongly **negative**
correlation. The rule, written before the results were computed: keep the default unless another
candidate is at least 0.03 more negative **and** also better on each half of the calibration origins.
Thresholds stay at 80/60 unless a band is empty on calibration. Weights, bands and constants are then
frozen (SHA-256) before the holdout is touched. Results: `outputs/trust_calibration.json`.

Data completeness was 100 on every calibration window, so its weight cannot be learned (it only shifts
every score); it was held at its default 0.10 and ties were broken by closeness to the default weights.

Outcome: default ρ = −0.514, equal ρ = −0.478, volatility-only ρ = −0.449; the best grid vector
(accuracy 0.7, interval 0.1, stability 0.1, agreement 0, completeness 0.1) reached ρ = −0.650 and was
better on both halves, so it replaced the default. Bands stayed at 80/60 (calibration: Accept 261,
Monitor 368, Review 139 of 768 windows). The column "Weight (default)" above shows the starting
hypothesis; the frozen weights are the ones in `outputs/trust_weights.json`.

### Validation (holdout, run once)

On the 6 holdout origins: Spearman ρ of score vs realised window WAPE (with p-value); realised WAPE per
band (ratio of sums); the Review/Accept WAPE ratio; the Review band's share of total absolute error vs
its share of windows; and whether the index beats the volatility-only baseline (more negative ρ).
The verdict is reported as is, including if the index fails.

### Limitations

- Accuracy and interval history for calibration origins come partly from selection windows, the same
  windows the champion was picked on, and the selection-fitted quantiles are scored on those same
  windows: history looks slightly better than truly out-of-sample.
- Sales are observed sales, not unconstrained demand (no stock-out data).
- A rank correlation across 96 series mixes "which series are hard" with "which windows are hard";
  much of the signal is persistent series difficulty.
