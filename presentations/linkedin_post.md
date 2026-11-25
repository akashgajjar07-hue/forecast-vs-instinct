Should a demand planner always trust an AI-generated forecast?

I tested that question on 96 real daily grocery series (12 stores × 8 families, Corporación Favorita, Ecuador) with five forecasting models and a 14-day horizon.

What I found on 6 weekly forecast origins (targets 28 Jun – 15 Aug 2017), each scored on days the model had not seen when it forecast:

• LightGBM cut error to 12.6% WAPE from 17.2% for the best simple baseline.
• But a 28-day moving average still beat it on holiday days (16.4% vs 17.1%).
• Knowing the promotion plan improved WAPE by only 0.18 pp.
• P10–P90 intervals covered 80.3% of actuals against an 80% target.

So I built a Forecast Trust Index: a 0–100 score (not a probability) that sorts forecasts into Accept, Monitor and Review. Review forecasts had 1.96× the error of Accept ones, and it beat a volatility-only rule.

Honest caveat: the index leans heavily on past accuracy, my promotion flag pointed the wrong way, and Monitor and Review were barely separated. This is Ecuadorian data, not UAE data, and observed sales, not true demand.

The lesson for planners: review the few, trust the many, and override only with information the model doesn't have.

Dashboard: https://akashgajjar07-hue.github.io/forecast-vs-instinct/
Code: https://github.com/akashgajjar07-hue/forecast-vs-instinct
Data: Alexis Cook, DanB, inversion, Ryan Holbrook (2021), Store Sales – Time Series Forecasting, Kaggle.

#DemandPlanning #SupplyChain #Forecasting #FMCG #MachineLearning #SOP #AbuDhabi #UAEJobs
