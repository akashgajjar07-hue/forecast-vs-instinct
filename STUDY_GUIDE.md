# Study guide

This guide is for preparing to explain *Forecast vs Instinct* in an interview. It explains every concept in plain English using this project's real numbers, walks through one forecast from start to finish, and gives model answers to the hardest questions.

Ground rules to repeat in any interview:
- The data is **real Ecuadorian grocery data** (Corporación Favorita, Kaggle), **not UAE data**.
- Sales are **observed sales, not unconstrained demand**. There is no stock-out data.
- `onpromotion` is a **count of promoted items**. Promotion results are associations, not causal uplift.
- The Trust score is a **0–100 index, not a probability**. Intervals are **empirical targets, not guarantees**.
- The FVA figure is a **simulated rule**, not real planner performance. The UAE Scenario Lab is **hypothetical**.

---

## Part 1. The concepts

### 1.1 The project in one paragraph

I forecast daily sales 14 days ahead for 96 series (12 stores × 8 product families) of an Ecuadorian grocery chain. I compared four simple methods with a LightGBM model on a rolling backtest, picked a champion per family on early data, built 80 % prediction intervals and a Trust Index on later data, froze all those choices with hashes, and then tested everything once on a final 6-week holdout. LightGBM had 12.57 % WAPE against 17.23 % for the best simple method. The intervals covered 80.32 % of actuals. The Trust Index ranked forecasts better than a volatility-only index (Spearman −0.637 vs −0.442), but it turned out to be mostly past accuracy, and its promotion flag pointed the wrong way.

### 1.2 Observed sales vs unconstrained demand

**Unconstrained demand** is what customers wanted to buy. **Observed sales** is what they actually bought, which can be lower when a product is out of stock. If yoghurt sells out at 3 pm, recorded sales for that day are lower than demand. A model trained on observed sales learns that low number as if it were demand, so it may under-forecast next time, which causes another stock-out. This project has no stock-out data, so every forecast here is a forecast of observed sales. That is the first thing to say when presenting any accuracy number.

### 1.3 WAPE

**WAPE (Weighted Absolute Percentage Error)** = total absolute error ÷ total actual sales × 100.

Example from the worked case below (values on the series' index scale, see Part 2): over 14 days, store 36 DAIRY's daily actuals added up to 1,332.9 index points; the forecast missed by a total of 81.5 index points (adding up each day's miss, ignoring sign), so WAPE = 81.5 ÷ 1,332.9 = 6.11 %. WAPE is a ratio, so it is the same whether it is computed in units or on the index.

Why WAPE and not MAPE (the average of daily percentage errors)? MAPE explodes when actual sales are small (a 5-unit miss on a 1-unit day is 500 %) and is undefined on zero days. WAPE weights every unit equally, so big series count more, which matches business impact. In this project WAPE is always a ratio of sums over the group, never an average of percentages. If total actuals are zero, WAPE is reported as missing, not infinity.

### 1.4 Bias

**Bias** = total (forecast − actual) ÷ total actual × 100. Positive means over-forecast (too much stock), negative means under-forecast (risk of stock-outs).

The champion's holdout bias was −0.46 %, almost zero. But that hides family biases from −6.31 % (CLEANING) to +4.02 % (DAIRY). Errors in opposite directions cancel in the total but not on the shelf. **Always check bias by family.**

### 1.5 MAE and RMSE

- **MAE** (mean absolute error) = the average miss per series-day, in units. LightGBM A: 208.5 units.
- **RMSE** (root mean squared error) squares each miss before averaging, so large misses count much more. LightGBM A: 512.5 units.

RMSE much larger than MAE means a few big misses. LightGBM B had a slightly lower RMSE (508.1) than A even though A had a better WAPE: B made slightly fewer very large misses.

### 1.6 The simple methods (baselines)

| Method | Idea | Holdout WAPE |
|---|---|---|
| **Naive** | every day of the next 14 = the last day's sales | 30.18 % |
| **Seasonal naive** | each day = the same weekday last week | 18.73 % |
| **Moving average** | each day = average of the last 28 days | 22.03 % |
| **Holt-Winters** | exponential smoothing of a level plus a weekly pattern | 17.23 % |

Why naive is so bad here: forecast origins are Tuesdays, and a Tuesday sells about 0.955 × an average day (mean over the 96 series). Repeating Tuesday for 14 days misses the weekend peak, so naive under-forecasts (bias −5.15 %).

### 1.7 Seasonal naive

"Same as last week, same weekday." It captures the weekly rhythm (Sunday peak, Thursday trough) with zero fitting. It is the method many planners use by instinct, which is why it is a fair benchmark. Its weakness is that it copies last week's noise and one-off events.

### 1.8 Holt-Winters (exponential smoothing)

Holt-Winters keeps a running **level** (the current typical day) and a **seasonal pattern** (how much each weekday differs from the level). Each new day nudges both, giving more weight to recent days. I used additive weekly seasonality with no trend, fitted per series on the last 364 days. A damped trend was tested and was slightly worse (selection WAPE 14.924 % vs 14.726 %). It was the best simple method on the holdout, and it beat LightGBM on 10 of 96 series.

### 1.9 Gradient boosting (LightGBM)

**Gradient boosting** builds many small decision trees one after another. Each new tree tries to fix the errors the previous trees still make. A decision tree is a set of yes/no questions such as "is it Sunday?" or "is the planned promotion count above 40?". **LightGBM** is a fast implementation of this idea.

How it is used here:
- **One global model** for all 96 series, so a quiet series learns from busy ones.
- **Scale-free target:** it predicts sales ÷ (28-day average + 1), then multiplies back. So the trees learn *shape* (weekday, payday, promotion) and the level comes from recent history. This was the biggest tuning win: 12.04 % selection WAPE vs 12.47 % for the same model on raw sales, with much smaller bias.
- **Most important features** (share of gain): weekday 19.10 %, store 13.87 %, day of month 12.29 % (payday and start-of-month), planned promotions 7.96 %.

### 1.10 Direct vs recursive forecasting

- **Recursive:** forecast tomorrow, then use that forecast as if it were real data to forecast the day after, and so on. Errors on day 1 feed into day 2 and compound.
- **Direct:** forecast every day from the same real history at the forecast date. Here `horizon` (1–14) is a feature, and the model learns from real examples of "history at a date → actual sales h days later".

I used direct because it avoids compounding errors, it does not need to invent future values of features like rolling averages, and the day-14 forecast is trained on real day-14 examples. LightGBM's WAPE rose only gently with horizon: 10.1 % on day 1 to 13.1 % on average over days 8–14.

### 1.11 Leakage

**Leakage** means the model uses information it would not have had on the day the forecast was made. It makes backtests look better than real life. Common examples: a rolling average that accidentally includes the target day, or tuning on the test period.

Controls in this project:
1. Models only see a **masked copy** of the data: all sales after the forecast date are blank.
2. Every history feature is counted **backwards from the forecast date**.
3. Tests **scramble all future data** and check that forecasts do not change by even one bit. A deliberately inserted one-day leak made 5 tests fail.
4. The holdout was **locked**: decisions were hashed before it ran, and the script refuses to run twice.

### 1.12 Rolling-origin backtesting

Instead of one train/test split, the model is re-run at many **forecast origins** (weekly Tuesdays), each time using only data up to that date, exactly as it would have been used in real life. The 24 origins are split into three stages with different jobs:

| Stage | Origins | Target dates | Job |
|---|---|---|---|
| Selection | 10 | 8 Feb – 25 Apr 2017 | tune models, pick champions |
| Calibration | 8 | 26 Apr – 27 Jun 2017 | fit intervals and Trust weights |
| Holdout | 6 | 28 Jun – 15 Aug 2017 | final exam, scored once |

The target dates of the stages never overlap, so nothing graded in the holdout was used to make a choice.

### 1.13 Conformal prediction intervals

A point forecast says "about 668". A **prediction interval** says "probably between 560 and 782". Here the interval is P10 to P90: the aim is that 10 % of actuals fall below and 10 % above, so 80 % inside.

**Split-conformal** means: measure how wrong the model was on a separate set of past windows (calibration), and assume future errors look similar. Steps:
1. For each calibration row, residual = (actual − forecast) ÷ the series' 28-day average (scaling lets one rule serve big and small stores).
2. For each family and week (days 1–7, days 8–14), take the 10th and 90th percentiles of those residuals.
3. New band = forecast + percentile × 28-day average.

### 1.14 Coverage

**Coverage** = the share of actuals that fell inside the band. Target 80 %. Holdout result: **80.32 %** (10.5 % below, 9.2 % above). Before the holdout, a leave-one-origin-out check on calibration estimated 79.20 %, so the estimate was honest.

But coverage is an average: BEVERAGES was 72.2 % (band too narrow), GROCERY I 87.8 % (too wide). Sundays were 70.8 % and 67.2 % because one percentile is shared across all weekdays while Sunday errors are the biggest. **"80 %" is a target, not a guarantee for any single day.**

### 1.15 The Forecast Trust Index

A 0–100 score per series at the forecast date. It is a **ranking index, not a probability**: 85 does not mean "85 % chance of being right".

| Component | Measures | Formula |
|---|---|---|
| Accuracy | recent WAPE for this series | 100 × (1 − min(WAPE, 50) / 50) |
| Interval reliability | did recent bands hit 80 %? | 100 × (1 − min(\|coverage − 80\|, 40) / 40) |
| Stability | how erratic sales were in the last 56 days | 100 × (1 − min(CV, 1)) × (1 − zero share) |
| Agreement | do the five models agree on the 14-day total? | 100 × (1 − min(spread ÷ 0.5, 1)) |
| Completeness | share of the last 56 days with data | 100 × days ok ÷ 56 |

Bands: **Accept** 80+, **Monitor** 60–79, **Review** below 60 or any **hard flag** (data gaps, a store closure in the window, or a planned promotion above the series' historical 95th percentile).

**Weights.** I started with 30/25/20/15/10 as a hypothesis and wrote a rule before seeing results: switch only if another set is at least 0.03 better in Spearman correlation and better on both halves of the calibration period. The winner was **70 % accuracy, 10 % interval, 10 % stability, 0 % agreement, 10 % completeness** (calibration Spearman −0.650 vs −0.514 for the default).

**Spearman correlation** measures whether two things rank the same way (−1 = perfectly opposite). We want high trust to go with low error, so more negative is better.

**Holdout result:** Spearman −0.637 (volatility-only −0.442). WAPE by band: Accept 8.93 %, Monitor 17.30 %, Review 17.46 %. Review windows had 1.96 × the error of Accept ones.

**What I learned:** accuracy alone ranked just as well (−0.647); the score had a 0.971 rank correlation with the accuracy component; the promotion flag fired on windows with *lower* error; completeness was always 100. So it separates Accept from the rest, but not Monitor from Review.

### 1.16 Forecast Value Added (FVA)

**FVA** asks: did a step in the process (for example, a planner's override) make the forecast better or worse? FVA = error before the step − error after. Positive = the step added value.

Here I had no real planner overrides, so I simulated a simple rule: "if a family was biased by more than 2 % in completed past windows, correct its forecast by that bias." On the holdout, WAPE went from 12.57 % to 13.02 %, so **FVA = −0.45 pp** (17 of 48 family-origin decisions adjusted). The lesson is not that planners are useless; it is that mechanically correcting last period's bias did not help here. Real FVA needs real overrides with reason codes.

### 1.17 S&OP (Sales and Operations Planning)

S&OP is the monthly (often weekly at a demand-review level) process where sales, marketing, supply and finance agree on one demand plan and how to supply it. The **demand review** step is where forecasts are challenged. The dashboard's S&OP page builds that pack automatically: the statistical forecast (2,119,685 units for 16–29 August 2017), the 22 Review-band series by priority, persistent biases (BEVERAGES), promotion plans to confirm, and a place to record overrides with a reason code.

### 1.18 Promotion variants A and B

Same model, run twice: **A** knows the planned promotion count for each future day (a retailer agrees its promotion calendar in advance); **B** only knows promotion history. A was better by 0.16, 0.26 and 0.18 percentage points on selection, calibration and holdout. Helpful but small. With no price or discount data, this is an association, not a measured uplift.

---

## Part 2. One forecast, end to end: store 36 DAIRY at origin 11 July 2017

Source: the executed `notebooks/03_trust_and_uncertainty.ipynb`, section 8.

**Index scale.** Raw competition data is not redistributed (Kaggle rules), so every quantity for this one series is shown as an **index**: value ÷ the series' mean daily sales on open days in the training period 2015-06-01 to 2017-02-06 × 100 (100 = an average training-period day). Planned promotions are a **promo index**: count ÷ the series' mean promoted-item count from 2015-06-01 up to T × 100. WAPE, bias, coverage, CV and the Trust components are ratios, so they are identical on either scale. The series was chosen by a fixed rule before looking at its outcome: at the middle holdout origin (the 3rd of 6), the series without a hard flag whose Trust score is closest to that origin's median score (79.57). That gave **36|DAIRY** (store 36, Libertad, type E), score 79.56.

### Step 1. The forecast and its interval

- Origin T = Tuesday 11 July 2017. We forecast 12 to 25 July. Champion for DAIRY: LightGBM, variant A.
- scale_T = average daily sales over the 28 days up to T = index **104.2**.
- DAIRY quantiles from calibration: days 1–7 **[−0.164, +0.172]**; days 8–14 **[−0.178, +0.149]**.

| h | Date | Promo index | Forecast | P10 | P90 | Actual | Inside? |
|---|---|---|---|---|---|---|---|
| 1 | Wed 12 Jul | 548.4 | 105.4 | 88.3 | 123.3 | 96.4 | yes |
| 2 | Thu 13 Jul | 74.8 | 79.7 | 62.6 | 97.6 | 74.0 | yes |
| 3 | Fri 14 Jul | 62.3 | 76.1 | 59.0 | 94.0 | 74.4 | yes |
| 4 | Sat 15 Jul | 78.9 | 105.2 | 88.1 | 123.1 | 100.0 | yes |
| 5 | Sun 16 Jul | 58.2 | 134.2 | 117.1 | 152.1 | 134.0 | yes |
| 6 | Mon 17 Jul | 49.9 | 95.9 | 78.8 | 113.8 | 84.5 | yes |
| 7 | Tue 18 Jul | 45.7 | 105.0 | 87.9 | 122.9 | 99.4 | yes |
| 8 | Wed 19 Jul | 506.8 | 103.3 | 84.8 | 118.8 | 89.1 | yes |
| 9 | Thu 20 Jul | 54.0 | 80.6 | 62.1 | 96.1 | 78.2 | yes |
| 10 | Fri 21 Jul | 58.2 | 77.3 | 58.7 | 92.8 | 72.9 | yes |
| 11 | Sat 22 Jul | 62.3 | 99.7 | 81.2 | 115.2 | 94.5 | yes |
| 12 | Sun 23 Jul | 54.0 | 127.9 | 109.3 | 143.4 | 141.0 | yes |
| 13 | Mon 24 Jul | 58.2 | 91.4 | 72.8 | 106.9 | 94.3 | yes |
| 14 | Tue 25 Jul | 62.3 | 100.5 | 81.9 | 116.0 | 100.3 | yes |

Check day 1 by hand (the band formula is linear, so it works on the index too): P10 = 105.4 + (−0.164 × 104.2) ≈ 105.4 − 17.1 = 88.3. P90 = 105.4 + (0.172 × 104.2) ≈ 105.4 + 17.9 = 123.3.

Notice the shape: the forecast peaks on Sundays (index 134.2 and 127.9) and dips on Thursdays and Fridays. That is the weekly pattern the model learned. The Wednesday promotion spikes (promo index 548.4 and 506.8, about five times the series' usual count) did not produce big forecast jumps, consistent with the small promotion effect for DAIRY.

### Step 2. The five Trust components (all known at T)

| Component | Input | Formula | Value |
|---|---|---|---|
| Accuracy | WAPE over the last 8 completed windows (origins 2 May … 27 June) = 11.63 % | 100 × (1 − 11.63 / 50) | 76.73 |
| Interval | 73.2 % of 112 past days inside the band | 100 × (1 − \|73.2 − 80\| / 40) | 83.04 |
| Stability | CV over 56 days = 0.245, zero share 0 | 100 × (1 − 0.245) × (1 − 0) | 75.47 |
| Agreement | five models' 14-day forecasts as index (mean daily ÷ base × 100): Holt-Winters 101.2, LightGBM 98.7, moving average 104.2, naive 110.2, seasonal naive 100.5; mean 103.0; spread (std ÷ mean) 0.039 | 100 × (1 − 0.039 / 0.5) | 92.17 |
| Completeness | 56 of 56 days open with a sale | 100 × 56 / 56 | 100.00 |

(Inputs are shown rounded; the values are the saved ones, which the notebook recomputes exactly.)

Promotion hard flag check: planned average promotion (promo index 126.7) vs this series' 95th percentile of 14-day averages (promo index 178.8) → **no flag**.

### Step 3. Score and band

| Component | Weight | Value | Weight × value |
|---|---|---|---|
| Accuracy | 0.7 | 76.73 | 53.71 |
| Interval | 0.1 | 83.04 | 8.30 |
| Stability | 0.1 | 75.47 | 7.55 |
| Agreement | 0.0 | 92.17 | 0.00 |
| Completeness | 0.1 | 100.00 | 10.00 |
| **Score** | | | **79.56** |

79.56 is just under 80, so the band is **Monitor**, not Accept. Notice that agreement was very high (the models agreed closely) but has zero weight, so it changed nothing. Completeness adds a flat 10 points, as it does for every series.

### Step 4. What actually happened

- Over the 14 days the actuals averaged index 95.2 and the forecast 98.7; the daily misses added up to 81.5 index points against 1,332.9 index points of actuals.
- **Realised WAPE 6.11 %**, bias **+3.70 %** (a slight over-forecast).
- **14 of 14** days inside the band.
- For comparison, all Monitor windows on the holdout averaged 17.30 % WAPE, and this series' own history was 11.63 %.

**How to explain it:** the index said "typical, keep an eye on it". The fortnight turned out better than typical. One window is one draw; the index is judged on how it ranks hundreds of windows (Spearman −0.637 over 576), not on any single case. DAIRY was one of the families whose bands were wider than needed on the holdout (83.9 % coverage), which is why all 14 days fell inside.

---

## Part 3. The 15 hardest interview questions

**1. "Why should I believe your holdout?"**
Three reasons. First, it was untouched: every choice (champion model, interval quantiles, Trust weights) was made on earlier periods and saved with a SHA-256 hash before the holdout ran. The holdout script checks those hashes, logs a digest of its results, and refuses to run a second time. Second, the target dates of the three stages never overlap, and tests scramble all future data to prove the models cannot see it. Third, the result is consistent with earlier stages: 11.88 % on selection, 12.65 % on calibration, 12.57 % on holdout. What it does *not* prove: it is only 6 origins in one summer, so I would not claim the same accuracy in December.

**2. "Isn't the Trust Index just past accuracy?"**
Largely, yes, and I say so. The weights picked on calibration put 70 % on historical accuracy. On the holdout, accuracy alone ranked windows slightly better (−0.647) than the full index (−0.637), and the two had a rank correlation of 0.971. The honest summary is: "the series that were hard recently stayed hard". That is still useful, because it beats a volatility-only index (−0.442), and series difficulty is persistent. But the other components add little on this data, and I would test them again at SKU level, where data gaps and model disagreement are more common.

**3. "Your promo flag fired on easier windows. What does that tell you?"**
It tells me the flag was badly calibrated for this data. It fired when the planned promotion intensity was above the series' all-history 95th percentile. Promotion counts rose steadily from 2015 to 2017, so recent plans exceeded that threshold easily. On the holdout, the 67 flagged windows had 8.71 % WAPE against 13.02 % for the others, and 65 of the 123 Review windows were there only because of the flag. I kept it as frozen because changing it after seeing the holdout would be cheating. The fix I would test next is a percentile over a recent window, or flagging only plans beyond anything the model saw in training.

**4. "How would this work at SKU level with stock-outs?"**
Three changes. First, demand estimation: I would flag likely stock-out days (zero or unusually low sales while inventory was zero, or using on-shelf availability data) and either exclude them from training or estimate the lost sales, so the model learns demand, not constrained sales. Second, methods: SKU-store series are sparse and intermittent, so I would add methods like Croston or TSB as baselines, use a Tweedie or Poisson-type objective, and judge accuracy at the level decisions are made (for example, weekly SKU-store or SKU-DC). Third, hierarchy: forecast at several levels and reconcile so store totals and SKU totals agree.

**5. "Why WAPE and not MAPE?"**
MAPE averages daily percentage errors, so a small miss on a tiny day can dominate, and it breaks on zero days. WAPE divides total absolute error by total sales, so every unit counts the same, which is closer to the cost of being wrong. I also report bias, because WAPE does not show direction, and weekly-bucket WAPE, because planners order in weekly quantities.

**6. "Your bias is −0.46 %. So the forecast is unbiased?"**
Overall, yes; by family, no. On the holdout CLEANING was under-forecast by 6.31 % and DAIRY over-forecast by 4.02 %. They cancel in the total. BEVERAGES was under-forecast in both calibration (−3.4 %) and holdout (−2.4 %), so the S&OP pack flags it as a persistent bias to discuss.

**7. "Holt-Winters was within about 1 pp in PRODUCE. Would you deploy ML everywhere?"**
Not automatically. LightGBM won on average and in every family, but simple models won 13 of 96 series, and the moving average won on holiday days. In practice I would keep the best simple model running alongside as a challenger, monitor both per series, and use the simpler one where the gain is small and stability or explainability matter. A 1 pp gain in PRODUCE might not justify the extra complexity; a 9 pp gain in PERSONAL CARE clearly does.

**8. "Why direct multi-horizon instead of recursive?"**
A recursive model feeds its own forecasts back in, so a mistake on day 1 becomes an input for day 2 and errors compound. The direct model predicts every horizon from the same real history at the forecast date, with horizon as a feature, so day 14 is learned from real day-14 examples. It also avoids simulating future values of features like rolling averages.

**9. "How do you know there's no leakage?"**
Structurally, the model never receives future sales: the harness blanks them. Then I test it: tests replace every value after the forecast date with random numbers and check the features and forecasts are bit-identical, check that a panel cut at the forecast date gives the same features as the full panel, and check that the baselines ignore the future even without the mask. As a test of the tests, a deliberately inserted one-day look-ahead made 5 of them fail.

**10. "The interval says 80 %. Can I promise the store 80 % availability?"**
No, for three reasons. The interval is for observed sales, not demand. 80 % is an average over many series-days, not a promise for one day: holdout coverage was 80.32 % overall but 72.2 % for BEVERAGES and about 67–71 % on Sundays. And availability depends on stock policy, lead time and replenishment, not only on the forecast band. The band is an input to a safety-stock decision, not the decision itself.

**11. "Knowing the promo plan only helped 0.18 pp. Is the promotion calendar worthless?"**
No. It helped in all three stages (0.16, 0.26, 0.18 pp) and most in GROCERY I and PERSONAL CARE. It is small here because promotions are nearly always on (about 99 % of days in six families) and the data only gives a count of promoted items, not discount depth. With real price and discount data I would expect the promotion plan to matter much more. I would not claim an uplift from this data, only an association.

**12. "Why did a moving average beat your ML model on holidays?"**
Only 176 holiday rows in the holdout, mostly one national holiday and a few local ones, so it is a small sample. Holiday effects vary widely from one holiday to the next (in the EDA, from about −47 % to +89 % by family), so the model has few comparable examples to learn from. A smooth average is hurt less by a day that breaks the weekly pattern. In practice I would review holiday weeks manually or add holiday-specific uplift factors agreed with the commercial team.

**13. "Your FVA was negative. Does that mean planners shouldn't override?"**
No. The FVA here is a **simulated rule**, not real planners. It shows that one mechanical habit (correcting by last period's family bias) made things worse: WAPE went from 12.57 % to 13.02 %. Real planners add value when they have information the model does not, such as a competitor closing or a supply problem. The recommendation is to override with a reason code and measure FVA on real overrides.

**14. "Why only 12 stores? Did you cherry-pick?"**
The stores were chosen by a fixed rule using only data before the first test date: a quota per store type, then preferring new clusters and cities, then volume near the type median. So they cover all 5 store types, 11 cities and 10 clusters, not just the biggest stores. They were also filtered to have no recording gaps, so they are cleaner than average, which I state as a limitation. 96 series was a deliberate size: small enough to inspect each one, large enough for a fair comparison.

**15. "How would you apply this to grocery in Abu Dhabi?"**
The method transfers; the numbers do not. I would retrain on local data, replace the Ecuadorian calendar with UAE events (Ramadan, Eid al-Fitr, Eid al-Adha, the summer slowdown, school terms, public-sector pay dates), and measure their effects on local history. The Scenario Lab in this project shows how a planner would use such estimates, but its percentages are demonstration assumptions, not measured UAE effects. I would also check how Ramadan moves through the calendar each year, because a model that only uses month or week of year would miss it.

---

## Part 4. What I would do next with real company data

1. **SKU × store level.** Forecast where orders are placed, with intermittent-demand methods (Croston, TSB) as baselines and accuracy measured at the decision level.
2. **Stock-out correction / unconstrained demand estimation.** Use inventory and on-shelf availability data to flag censored days and estimate lost sales, so the model learns demand rather than repeating past stock-outs.
3. **ERP and planning system integration.** Feed forecasts, intervals and Trust bands into SAP IBP (or APO in older landscapes) as the statistical baseline, with overrides captured in the tool, rather than in a separate dashboard.
4. **Promotion depth and price.** Add discount depth, price, display and media to estimate real promotion uplift and cannibalisation, and test it against planned vs actual promotion results.
5. **Hierarchical reconciliation.** Forecast at SKU, family, store and DC levels and reconcile them so every level adds up, with intervals that are valid at each level.
6. **Weekly S&OP cadence.** Run the pipeline weekly, publish the exception list before the demand review meeting, and track the Trust Index's hit rate over time.
7. **FVA on real planner overrides.** Record every override with a reason code and measure whether each type of override reduces error, so the team keeps the useful ones.
8. **Fix the known weaknesses.** Weekday-aware interval quantiles (for Sundays), a redesigned promotion flag, and a Trust Index re-tested where completeness and model agreement actually vary.

---

## Part 5. Numbers to know by heart

| What | Number |
|---|---|
| Scope | 12 stores × 8 families = 96 series, 14 days ahead |
| Holdout | 6 origins, 28 Jun – 15 Aug 2017, 8,064 rows per model, scored once |
| LightGBM A holdout WAPE / bias | 12.57 % / −0.46 % |
| Best simple method (Holt-Winters) | 17.23 % |
| Improvement | 4.66 pp, 27 % less mis-forecast volume |
| Series won by LightGBM | 83 of 96 |
| Promo plan gain (A vs B), holdout | 0.18 pp |
| Interval coverage, holdout | 80.32 % (target 80 %) |
| Coverage range by family | 72.2 % – 87.8 % |
| Trust Spearman, holdout | −0.637 (volatility-only −0.442) |
| WAPE Accept / Monitor / Review | 8.93 % / 17.30 % / 17.46 % |
| Accuracy-only Spearman | −0.647 |
| Simulated FVA | −0.45 pp |
| Forward 14-day forecast | 2,119,685 units; 22 Review series |
