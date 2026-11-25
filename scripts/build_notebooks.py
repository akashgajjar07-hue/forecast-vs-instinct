"""Build notebooks 02 (forecasting battle) and 03 (trust & uncertainty) with nbformat.

    .venv/bin/python scripts/build_notebooks.py            # write both notebooks (unexecuted)
    .venv/bin/python scripts/build_notebooks.py --execute  # write and execute them in place

The notebooks only READ results from outputs/ (backtests, frozen JSONs, holdout files). They never
train a model, never re-run the holdout and never write to outputs/. Every number shown is computed
in a cell; the markdown only interprets numbers that were checked against those cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(s: str):
    return new_markdown_cell(s.strip("\n"))


def code(s: str):
    return new_code_cell(s.strip("\n"))


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

SETUP = r'''
import sys, json, hashlib, warnings
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from src import config as C, metrics as M
from src.backtest import evaluable, masked_view
from src.runner import load_backtests
from src.selection import selection_table, pick_champions, champion_rows

pio.renderers.default = "plotly_mimetype+notebook_connected"
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)

OUT = C.OUTPUTS
J = lambda name: json.loads((OUT / name).read_text())

# Same family colours as notebook 01; one fixed colour per model (never re-assigned)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAM_COLOR = dict(zip(C.FAMILIES, PALETTE))
MODEL_ORDER = ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm A", "lightgbm B"]
MODEL_COLOR = {"naive": "#8a8984", "seasonal_naive": "#eda100", "moving_average": "#e87ba4",
               "holt_winters": "#2a78d6", "lightgbm A": "#eb6834", "lightgbm B": "#6250d6"}
BAND_COLOR = {"Accept": "#1baf7a", "Monitor": "#eda100", "Review": "#e34948"}
LAYOUT = dict(template="simple_white", height=380, margin=dict(l=50, r=20, t=50, b=40),
              font=dict(size=12), legend=dict(title=None))

bt = load_backtests()                                   # every model, every stage, read-only
bt["mv"] = np.where(bt["model"] == "lightgbm", "lightgbm " + bt["variant"], bt["model"])
panel = pd.read_parquet(C.PROCESSED / "panel.parquet")
cal = json.loads((C.CONFIG / "split_calendar.json").read_text())

def table(df, by, models=MODEL_ORDER, value="wape"):
    """Ratio-of-sums metric per group x model (src.metrics.grouped_metrics), models as columns."""
    g = M.grouped_metrics(df, list(by) + ["mv"])
    return g.pivot_table(index=list(by), columns="mv", values=value)[[m for m in models if m in set(g.mv)]]

print(f"backtest rows: {len(bt):,}  |  stages: {sorted(bt.stage.unique())}  |  models: {sorted(bt.mv.unique())}")
'''

# =====================================================================================================
# NOTEBOOK 02
# =====================================================================================================
NB02 = [
md(r'''
# 02 · Forecasting Battle

**Forecast vs Instinct** - real data from **Corporación Favorita** (Ecuador, Kaggle *Store Sales - Time Series
Forecasting*), **12 stores × 8 product families = 96 daily series**, forecast **14 days ahead** from weekly origins.

This notebook tells the modelling story: how the data was split so the final test is honest, how look-ahead was made
impossible, which five models competed, how each was tuned, how a champion was picked per family, and how everything
did on the **holdout, which was evaluated exactly once** after all choices were frozen.

Ground rules (same as notebook 01):

1. **Sales are observed sales, not unconstrained demand.** A day with empty shelves looks like low demand.
2. **`onpromotion` is a count of promoted items**, with no price or discount depth. Promo comparisons are associations.
3. **Nothing is re-trained or re-run here.** Every number is computed in a cell from the saved backtests in
   `outputs/backtests/` and the frozen JSON files in `outputs/`. The notebook never writes to `outputs/`.

**Metrics** (`src/metrics.py`, A = actual, F = forecast, sums over the rows being scored):

| Metric | Formula | Read it as |
|---|---|---|
| WAPE | Σ\|A − F\| / ΣA × 100 | % of volume mis-forecast; **lower is better**; the headline metric |
| Bias | Σ(F − A) / ΣA × 100 | **+ = over-forecast**, − = under-forecast |
| MAE | mean \|A − F\| | average miss per series-day, in units |
| RMSE | √mean (A − F)² | like MAE but punishes big misses more |

WAPE and bias are always **ratio of sums** over the group (never an average of per-row percentages), and every model
is scored on the identical set of rows: open days with an observed actual (`src.backtest.evaluable`).
'''),
code(SETUP),
md(r'''
## 1. The split calendar: three stages, one honest test

A **forecast origin T** is the end of day T: everything up to and including T is known, and we forecast T+1 … T+14.
Origins are weekly (Tuesdays). They are grouped into three stages, each with one job:

| Stage | Job | What is decided there |
|---|---|---|
| **selection** | pick and tune | moving-average window, Holt-Winters trend, LightGBM settings, champion model per family |
| **calibration** | measure uncertainty | prediction-interval quantiles, Trust Index weights |
| **holdout** | final exam | nothing - it is scored once, after the choices above are frozen |

The **target windows** (the dates being forecast) of the three stages never overlap, so no date that is graded in
the holdout was ever used to make a choice.
'''),
code(r'''
rows = []
for stage, origins in cal["stages"].items():
    o = pd.to_datetime(origins)
    rows.append({"stage": stage, "n_origins": len(o), "first_origin": o.min().date(), "last_origin": o.max().date(),
                 "first_target": (o.min() + pd.Timedelta(days=1)).date(),
                 "last_target": (o.max() + pd.Timedelta(days=cal["horizon"])).date()})
split = pd.DataFrame(rows)
for a, b in zip(split.itertuples(), split.iloc[1:].itertuples()):
    gap = (pd.Timestamp(b.first_target) - pd.Timestamp(a.last_target)).days
    assert gap >= 1, "target windows overlap"
    print(f"{a.stage:>11} targets end {a.last_target}  ->  {b.stage} targets start {b.first_target}  (disjoint)")
# the saved backtests use exactly these origins
for stage, origins in cal["stages"].items():
    assert sorted(bt.loc[bt.stage == stage, "origin"].dt.strftime("%Y-%m-%d").unique()) == origins
print("Saved backtests use exactly the calendar origins for every stage.")
split
'''),
code(r'''
win = pd.DataFrame([{"stage": s, "origin": o, "start": pd.Timestamp(o) + pd.Timedelta(days=1),
                     "end": pd.Timestamp(o) + pd.Timedelta(days=cal["horizon"] + 1)}
                    for s, os_ in cal["stages"].items() for o in os_])
fig = px.timeline(win, x_start="start", x_end="end", y="origin", color="stage",
                  color_discrete_map={"selection": "#2a78d6", "calibration": "#eda100", "holdout": "#e34948"},
                  title="Each bar = the 14 target days of one origin")
fig.update_yaxes(autorange="reversed", title=None, tickfont=dict(size=9))
fig.update_layout(**{**LAYOUT, "height": 480}, xaxis_title=None)
fig.show()
'''),
md(r'''
**Reading it:** within a stage the windows overlap (weekly origins, 14-day horizon), but there is a gap between
stages so the last selection target comes before the first calibration target, and the last calibration target
(27 June 2017) comes before the first holdout target (28 June).

### The holdout was touched once

Before the holdout ran, the three decision files (`selection.json`, `intervals.json`, `trust_weights.json`) were
**frozen with a SHA-256 hash**. The holdout script refuses to run if any file changed since, and it logs the hashes and
a digest of its results the first (and only) time it runs (`outputs/holdout_run.json`). The cell below recomputes
the hashes now and re-derives the results digest from `holdout_results.json`.
'''),
code(r'''
run = J("holdout_run.json")
hrows = []
for n in ["selection", "intervals", "trust_weights"]:
    now = hashlib.sha256((OUT / f"{n}.json").read_bytes()).hexdigest()
    frozen = (OUT / f"{n}.sha256").read_text().strip()
    hrows.append({"file": f"{n}.json", "frozen hash": frozen[:16] + "…", "hash logged by holdout run": run["frozen_hashes"][n][:16] + "…",
                  "recomputed now": now[:16] + "…", "all three match": now == frozen == run["frozen_hashes"][n]})
display(pd.DataFrame(hrows))
res = J("holdout_results.json")
digest = hashlib.sha256(json.dumps({k: res[k] for k in ("leaderboard", "champion_system", "trust_validation")},
                                   sort_keys=True, default=float).encode()).hexdigest()
print("Holdout first (and only) run at:", run["first_run_at"])
print("Results digest logged at that run matches holdout_results.json:", digest == run["results_digest"])
'''),
md(r'''
## 2. Leakage controls: making look-ahead impossible

"Leakage" means a model quietly uses information it would not have had on the day the forecast was made. Three
controls stop it:

**1. Masked view.** A model never receives the raw panel. `src.backtest.masked_view(panel, T, variant)` hands it a
copy where every **sale after T is blanked (NaN)**. In **promo variant B** the promoted-item count after T is blanked
too; in **variant A** it is kept, as a planner with an agreed promo calendar would know it. The demo below uses the
first selection origin.
'''),
code(r'''
T0 = pd.Timestamp(cal["stages"]["selection"][0])
sid = "12|BEVERAGES"
one = panel[panel.series_id == sid]
around = lambda d: d[(d.date > T0 - pd.Timedelta(days=3)) & (d.date <= T0 + pd.Timedelta(days=3))].set_index("date")
demo = pd.DataFrame({
    "raw sales": around(one)["sales"], "raw promo": around(one)["onpromotion"],
    "view A sales": around(masked_view(one, T0, "A"))["sales"], "view A promo": around(masked_view(one, T0, "A"))["onpromotion"],
    "view B sales": around(masked_view(one, T0, "B"))["sales"], "view B promo": around(masked_view(one, T0, "B"))["onpromotion"]})
demo.index = [f"{d:%Y-%m-%d}" + ("  <- origin T" if d == T0 else "") for d in demo.index]
print(f"Series {sid}, origin T = {T0:%Y-%m-%d}")
demo
'''),
md(r'''
**2. Direct multi-horizon forecasting.** LightGBM does not forecast day 1 and then feed that guess back in to forecast
day 2 (a *recursive* model, where errors compound). It is one global model with **`horizon` (1-14) as a feature**,
trained on real pairs "history at origin t' → actual sale at t' + h". Every horizon is predicted from the same
observed history at T.

**3. Origin-relative features.** Every history feature is measured **backwards from the origin**, never from the target
date: `lag_1` = sales on T itself, `roll_mean_28` = mean of the 28 days up to T, `sw_last` = the most recent day ≤ T on
the target's weekday. Target-date features are only things known in advance: calendar (weekday, day of month, month,
holiday / event / payday flags) and, in variant A only, the planned promo count. Transactions and oil are not used
(future transactions are unknown at T; oil was tested and dropped, section 5).

The project's tests (`tests/test_features_leakage.py`) perturb every value after T and check that the forecasts do not
change; a deliberately injected one-day look-ahead made 5 of them fail. The cell below confirms from the saved feature
lists that variant B has no target-date promo feature.
'''),
code(r'''
fi = J("lgbm_feature_importance.json")
featA = [r["feature"] for r in fi["variants"]["A"]]
featB = [r["feature"] for r in fi["variants"]["B"]]
print(f"variant A: {len(featA)} features | variant B: {len(featB)} features")
print("only in A (target-date promo plan):", sorted(set(featA) - set(featB)))
print("only in B:", sorted(set(featB) - set(featA)) or "none")
'''),
md(r'''
## 3. The five contenders, explained simply

| Model | One-line idea | Strength | Weakness |
|---|---|---|---|
| **Naive** | tomorrow (and every day of the next 14) = today | zero effort, a floor | ignores the weekly rhythm completely |
| **Seasonal naive** | each day = the same weekday last week | captures the Sunday peak / Thursday trough | copies last week's noise and one-offs |
| **Moving average** | each day = the mean of the last *w* days | smooth, robust | flat line: no weekday shape |
| **Holt-Winters** | exponential smoothing of a level + a weekly pattern, fitted per series on the last 364 days | the classic planner's statistical model | one series at a time; knows nothing about promos or holidays |
| **LightGBM** | one gradient-boosted tree model across all 96 series, learning how sales relate to history, calendar and promo plan | shares patterns across series; uses known drivers | needs careful leakage control and tuning |

LightGBM predicts a **scale-free target**, sales ÷ (28-day mean at the origin + 1), and multiplies back afterwards, so
the trees learn *shape* (weekday, promo, holiday) while the level comes from recent history. It runs in two promo
variants: **A = known promo plan**, **B = no future promo information**.
'''),
md(r'''
## 4. Tuning the simple models (selection stage only)

Two baseline settings were chosen on the 10 selection origins by **lowest overall selection WAPE**: the moving-average
window and whether Holt-Winters should include a (damped) trend.
'''),
code(r'''
bs = J("baselines_summary.json")
ma = pd.DataFrame(bs["moving_average"]["candidates"]).T.rename_axis("window (days)")
ma["chosen"] = ma.index.astype(int) == bs["moving_average"]["chosen_window"]
hw = pd.DataFrame(bs["holt_winters"]["trend_candidates"]).T.rename_axis("Holt-Winters trend")
hw["chosen"] = hw.index == bs["holt_winters"]["chosen_trend"]
display(ma[["wape", "bias", "n", "chosen"]])
display(hw[["wape", "bias", "n", "fallbacks", "chosen"]])
print(f"Holt-Winters fits: {bs['holt_winters']['fits_total']:,}, fallbacks to seasonal naive: {bs['holt_winters']['fallbacks_total']}")
'''),
md(r'''
**Reading it:** a 28-day window is the best of the four moving averages, and adding a damped trend made Holt-Winters
slightly *worse* on selection, so no trend was used. Every Holt-Winters fit converged (no fallbacks).

## 5. Tuning LightGBM (selection stage only)

LightGBM was tuned in stages on the 10 selection origins (variant A, training every 2nd day to save time), ranking
candidates by overall selection WAPE rounded to 2 dp (ties → smaller |bias|). Each stage kept the winner of the
previous one. Variant B reuses the chosen settings unchanged.
'''),
code(r'''
tun = J("lgbm_tuning.json")
cand = pd.DataFrame([{"stage": c["stage"], "candidate": c["name"], "selection WAPE": c["selection_wape"],
                      "selection bias": c["selection_bias"], "runtime (s)": c["runtime_s"]} for c in tun["candidates"]])
cand
'''),
code(r'''
d = tun["decisions"]
print(f"Objective stage winner : {d['objective_stage_winner']}  (L2 loss on sales / 28-day mean)")
print(f"Structure stage winner : {d['structure_stage_winner']}")
print(f"Earthquake month       : '{d['quake']['choice']}'  flag WAPE {d['quake']['flag_wape']:.2f} (bias {d['quake']['flag_bias']:+.2f}) "
      f"vs drop WAPE {d['quake']['drop_wape']:.2f} (bias {d['quake']['drop_bias']:+.2f})")
print(f"Oil price              : keep={d['oil']['keep']}  without {d['oil']['without_wape']:.2f} vs with {d['oil']['with_wape']:.2f}  [{d['oil']['rule']}]")
print(f"Training-origin stride : {d['origin_stride']['choice']}  stride 2 {d['origin_stride']['stride2_wape']:.2f} vs stride 1 {d['origin_stride']['stride1_wape']:.2f}")
pd.Series(tun["chosen_params"]).drop(["tweedie_variance_power", "num_threads"]).to_frame("chosen").T
'''),
md(r'''
**Reading it:**

- **The scale-free target was the biggest win.** Predicting sales relative to the recent 28-day level beat every
  raw-sales objective (Tweedie, L2, L1) and cut the under-forecast bias from about −1.5 to −2.2 % down to about −0.6 %.
- **Earthquake (16 Apr - 15 May 2016): flag it, don't delete it.** Keeping those training rows with a quake flag gave
  a slightly lower WAPE and a smaller bias than dropping them.
- **Oil: dropped.** Adding the lagged national oil price made selection WAPE clearly worse. One national price series
  carries no store-level information for a 14-day horizon, and trends invite spurious splits.
- Training on every day (stride 1) helped a little, so the final model uses it (twice the run time).

(`tweedie_variance_power` is still in the saved parameter set but has no effect with the L2 objective.)

## 6. Choosing a champion per family (selection stage only)

Rule (`src/selection.py`): for each family, the model with the **lowest selection WAPE** (ratio of sums over 10 origins
× 12 stores × 14 days = 1,680 rows) is champion; a tie within 0.05 pp goes to the smaller |bias|. LightGBM competes
with variant A. The cell recomputes the table from the saved backtests and checks it against the frozen
`selection.json`.
'''),
code(r'''
sel = J("selection.json")
st = selection_table(bt)                                   # variant A rows, selection stage
frozen_st = pd.DataFrame(sel["selection_table"])
chk = st.merge(frozen_st, on=["family", "model"], suffixes=("", "_frozen"))
print("max |recomputed - frozen| selection WAPE:", round(float((chk.wape - chk.wape_frozen).abs().max()), 4))
champs = sel["champions"]
assert {f: c["model"] for f, c in pick_champions(st).items()} == {f: c["model"] for f, c in champs.items()}
piv = st.pivot(index="family", columns="model", values="wape").loc[C.FAMILIES, C.MODELS]
piv["champion"] = [champs[f]["model"] for f in piv.index]
piv["runner-up"] = [piv.loc[f, C.MODELS].drop(champs[f]["model"]).idxmin() for f in piv.index]
piv["margin (pp)"] = [piv.loc[f, r] - piv.loc[f, champs[f]["model"]] for f, r in zip(piv.index, piv["runner-up"])]
piv.round(2)
'''),
code(r'''
fig = px.imshow(piv[C.MODELS].round(1), text_auto=True, aspect="auto", color_continuous_scale="Blues",
                title="Selection WAPE % by family and model (lower = better)")
fig.update_layout(**{**LAYOUT, "height": 420}, coloraxis_colorbar=dict(title="WAPE"), xaxis_title=None, yaxis_title=None)
fig.show()
'''),
md(r'''
**Reading it:** LightGBM (variant A) wins selection in **all 8 families**, so the champion system is simply LightGBM A
everywhere. Holt-Winters is the runner-up in seven families; in FROZEN FOODS the runner-up is seasonal naive, because
Holt-Winters over-forecast that family badly on selection (see `baselines_summary.json`: bias about +9 %).
The margin column shows how much daylight there was: under 1 pp for PRODUCE, 4-5 pp for FROZEN FOODS and CLEANING,
and about 9 pp for PERSONAL CARE.

## 7. The holdout leaderboard (6 origins, run once)

All six model/variant combinations are scored on exactly the same 8,064 holdout rows (6 origins × 96 series × 14 days,
open days with an actual). The cell recomputes the leaderboard from the saved backtests and checks it against
`holdout_results.json`.
'''),
code(r'''
hold = evaluable(bt[bt.stage == "holdout"])
sizes = hold.groupby("mv").size()
assert sizes.nunique() == 1, "evaluation population differs between models"
lb = M.grouped_metrics(hold, "mv").set_index("mv").loc[MODEL_ORDER, ["wape", "bias", "mae", "rmse", "n"]].sort_values("wape")
saved = pd.DataFrame(res["leaderboard"]).assign(mv=lambda d: np.where(d.model == "lightgbm", "lightgbm " + d.variant, d.model)).set_index("mv")
print("max |recomputed - saved| WAPE:", float((lb.wape - saved.loc[lb.index, "wape"]).abs().max()))
best_base = lb.drop(["lightgbm A", "lightgbm B"]).wape.idxmin()
print(f"LightGBM A vs best simple model ({best_base}): {lb.loc[best_base,'wape'] - lb.loc['lightgbm A','wape']:.2f} pp lower WAPE, "
      f"{100 * (1 - lb.loc['lightgbm A','wape'] / lb.loc[best_base,'wape']):.0f}% less mis-forecast volume")
pre = panel[(panel.date <= cal["target_windows"]["calibration"][1]) & (panel.closed == 0)].groupby("date").sales.sum()
tue = pre[pre.index.dayofweek == 1].mean() / pre.mean()
print(f"Why naive under-forecasts: origins are Tuesdays, and a Tuesday sells {tue:.3f}x an average day (history up to the "
      f"last calibration target), so repeating Tuesday for 14 days misses the weekend peak.")
lb.round(2)
'''),
code(r'''
fig = go.Figure(go.Bar(x=lb.wape.round(2), y=lb.index, orientation="h", marker_color=[MODEL_COLOR[m] for m in lb.index],
                       text=lb.wape.round(1), textposition="outside", hovertemplate="%{y}: WAPE %{x:.2f}%<extra></extra>"))
fig.update_layout(**{**LAYOUT, "height": 330}, title="Holdout WAPE % (6 origins × 96 series × 14 days)",
                  yaxis=dict(autorange="reversed"), xaxis_title="WAPE %", showlegend=False)
fig.show()
'''),
md(r'''
**Reading it:** LightGBM with the known promo plan is the clear winner on every metric except RMSE, where variant B is
marginally lower (RMSE is dominated by a few very large misses). The ranking of the simple models is the textbook one:
Holt-Winters > seasonal naive > moving average > naive. Naive is the only model with a sizeable bias (about −5 %):
it repeats the origin day, a Tuesday, which sells below an average day (printed above).

The champion system *is* LightGBM A in every family, so its holdout numbers equal the first row
(`holdout_results.json → champion_system` adds interval coverage, covered in notebook 03).

## 8. By family
'''),
code(r'''
fam = table(hold, ["family"]).loc[C.FAMILIES]
fam["best simple model"] = fam[C.MODELS[:-1]].idxmin(axis=1)
fam["gap to best simple (pp)"] = fam[C.MODELS[:-1]].min(axis=1) - fam["lightgbm A"]
fam["selection WAPE (A)"] = [champs[f]["selection_wape"] for f in fam.index]
fam.round(2)
'''),
code(r'''
fig = go.Figure()
for m in ["seasonal_naive", "holt_winters", "lightgbm A"]:
    fig.add_bar(name=m, x=fam.index, y=fam[m].round(2), marker_color=MODEL_COLOR[m])
fig.update_layout(**LAYOUT, barmode="group", bargap=0.25, title="Holdout WAPE % by family: LightGBM A vs the two best simple models",
                  yaxis_title="WAPE %", xaxis_title=None)
fig.show()
'''),
md(r'''
**Reading it:** LightGBM A has the lowest holdout WAPE in every family. The size of the win varies a lot:

- **Big wins** in GROCERY I, CLEANING and PERSONAL CARE, where Holt-Winters and seasonal naive miss by far more.
- **Small wins** in PRODUCE, DAIRY and BREAD/BAKERY, where Holt-Winters is within about 1-2 pp: these families have a
  stable weekly shape that a per-series smoothing model already captures well.
- **CLEANING** is the hardest family in the holdout (22 % vs about 13 % on selection), and every model struggled
  there; its best simple model was the moving average. FROZEN FOODS stays hard, as expected from notebook 01.

## 9. By horizon (1 to 14 days ahead)
'''),
code(r'''
hz = table(hold, ["horizon"])
fig = go.Figure()
for m in MODEL_ORDER:
    fig.add_scatter(x=hz.index, y=hz[m].round(2), name=m, mode="lines+markers", line=dict(color=MODEL_COLOR[m], width=2),
                    marker=dict(size=7), line_dash="dot" if m == "lightgbm B" else "solid")
fig.update_layout(**{**LAYOUT, "height": 420}, title="Holdout WAPE % by forecast horizon (days ahead)",
                  xaxis=dict(title="horizon (days)", dtick=1), yaxis_title="WAPE %")
fig.show()
print("LightGBM A has the lowest WAPE at", int((hz.drop(columns="lightgbm B").idxmin(axis=1) == "lightgbm A").sum()), "of 14 horizons")
print(f"LightGBM A: day 1 {hz.loc[1,'lightgbm A']:.1f}% -> mean of days 8-14 {hz.loc[8:14,'lightgbm A'].mean():.1f}%")
hz.round(1).T
'''),
md(r'''
**Reading it:** error grows only gently with the horizon for the good models, because the weekly pattern - the
biggest source of predictability - is equally known 1 or 14 days ahead. Naive and seasonal naive have a saw-tooth:
on days 7 and 14 the target falls on the origin's weekday, so "repeat today" and "repeat last week" become the same
forecast. The day-to-day wiggles are mostly calendar (which weekdays and holidays fall at each horizon across only
six origins), not a model property.

## 10. Week-1 vs week-2 buckets: how a planner is measured

Replenishment is planned in weekly quantities, so daily errors that cancel within a week don't hurt. Here each
series-origin's 14 days are summed into a **week-1 total (days 1-7)** and a **week-2 total (days 8-14)**
(`src.metrics.bucket_totals`), and WAPE is computed on those totals.
'''),
code(r'''
bk = M.bucket_totals(hold, ["mv", "origin", "series_id"])
bkt = M.grouped_metrics(bk, ["bucket", "mv"]).pivot(index="bucket", columns="mv", values="wape")[MODEL_ORDER]
daily = table(hold.assign(bucket=np.where(hold.horizon <= 7, "week1", "week2")), ["bucket"])
out = pd.concat({"weekly-total WAPE": bkt, "daily WAPE": daily}, axis=0)
out.round(2)
'''),
md(r'''
**Reading it:** at the weekly-total level the errors are much smaller than daily (day-to-day misses partly cancel),
and week 2 is harder than week 1 for every model. LightGBM A stays the best in both buckets.
The ranking of the simple models changes: the **moving average becomes the second-best model** for weekly totals
(its flat line has no weekday shape, which a weekly total does not need), and seasonal naive also overtakes
Holt-Winters. A planner measured on weekly buckets would find simple models much closer to the ML model than the
daily view suggests.

## 11. Promo days vs non-promo days (association, not causation)

A row counts as a **promo day** if at least one item in that store-family was on promotion (`onpromotion > 0`).
'''),
code(r'''
hp = hold.assign(promo=np.where(hold.onpromotion > 0, "promo day", "no promo"))
n = hp[hp.mv == "lightgbm A"].groupby("promo").size().rename("rows (per model)")
pt = table(hp, ["promo"]).join(n)
display(pt.round(2))
share = hp[hp.mv == "lightgbm A"].groupby("family").apply(lambda g: (g.onpromotion > 0).mean() * 100).rename("% promo days")
share.loc[C.FAMILIES].round(1).to_frame().T
'''),
md(r'''
**Reading it:** promotions are almost permanent in this holdout: about nine rows in ten are promo days, and six
families have at least one promoted item on more than 99 % of days, so the "no promo" group is small (652 rows per
model) and mostly BREAD/BAKERY and FROZEN FOODS. For LightGBM A, Holt-Winters and seasonal naive the WAPE differs by
less than 0.5 pp between the groups. Naive does worse on promo days, and LightGBM B does *better* on no-promo days -
but with so few no-promo rows, from two families, these are weak signals. The more useful contrast is between promo
variants A and B (section 13). Neither comparison measures promotional uplift: there is no price or discount data.

## 12. Holiday days vs normal days

`is_holiday` marks a day off for that store (national, regional or local, matched to the store's location).
The holdout contains only a handful of such days, so this is a small-sample check.
'''),
code(r'''
hd = panel[(panel.is_holiday == 1) & panel.date.between(cal["target_windows"]["holdout"][0], cal["target_windows"]["holdout"][1])]
hd_days = hd.groupby("date").agg(stores=("store_nbr", lambda s: sorted(s.unique())), national=("is_national_holiday", "max"))
display(hd_days)
hh = hold.assign(day=np.where(hold.is_holiday == 1, "holiday", "normal"))
n = hh[hh.mv == "lightgbm A"].groupby("day").size().rename("rows (per model)")
table(hh, ["day"]).join(n).round(2)
'''),
md(r'''
**Reading it:** holiday rows are a small share of the holdout (176 rows per model: one national day, 11 August, plus
local/regional days in a few stores). LightGBM A is clearly worse on those days than on normal days, and **it is not
the best model on them: the moving average has the lowest holiday WAPE**. With only 22 holiday store-days this is
indicative, but it fits notebook 01: holiday effects vary a lot from one holiday to the next, so a model that has seen
few comparable days cannot learn them reliably, and a smooth average is hurt less by a day that breaks the weekly pattern.

## 13. Promo variant A (known plan) vs B (no future promo info)

Same model, same settings; the only difference is whether the planned promoted-item count for the target date is a
feature. This is the business question "is it worth getting the promo calendar into the forecast?".
'''),
code(r'''
ab = bt[(bt.model == "lightgbm")]
ab = evaluable(ab)
stage_ab = M.grouped_metrics(ab, ["stage", "variant"]).pivot(index="stage", columns="variant", values=["wape", "bias"])
stage_ab = stage_ab.loc[["selection", "calibration", "holdout"]]
stage_ab[("wape", "B - A (pp)")] = stage_ab[("wape", "B")] - stage_ab[("wape", "A")]
display(stage_ab.round(3))
fam_ab = table(hold, ["family"], ["lightgbm A", "lightgbm B"]).loc[C.FAMILIES]
fam_ab["B - A (pp)"] = fam_ab["lightgbm B"] - fam_ab["lightgbm A"]
fam_ab["% promo days"] = share.loc[C.FAMILIES].values
fam_ab.round(2)
'''),
md(r'''
**Reading it:** knowing the promo plan lowers WAPE in **all three stages**, by a small but consistent margin (a few
tenths of a point overall). By family the picture is mixed: in the holdout the plan helps most in some families and
makes almost no difference, or slightly hurts, in others (see the `B - A` column): the plan helps most in GROCERY I
and PERSONAL CARE, while variant B is slightly better in FROZEN FOODS, BREAD/BAKERY and DAIRY. That is consistent with
notebook 01, where the promoted-item count had only a small association with sales for most families once store size
and month were controlled. A planner would read this as: feed the promo calendar in, it helps on average, but it
is not the main source of LightGBM's advantage.

## 14. What did LightGBM pay attention to?

Mean **gain share** per origin (% of the total split gain), averaged over the 18 selection + calibration origins
(`outputs/lgbm_feature_importance.json`). Gain is "how much the splits on this feature reduced the training loss";
it describes the model, not cause and effect.
'''),
code(r'''
imp = pd.concat([pd.DataFrame(fi["variants"][v]).assign(variant=v) for v in ["A", "B"]])
top = imp.groupby("feature").gain_share_pct.max().sort_values(ascending=False).head(15).index
p = imp[imp.feature.isin(top)]
fig = px.bar(p, x="gain_share_pct", y="feature", color="variant", barmode="group", orientation="h",
             color_discrete_map={"A": MODEL_COLOR["lightgbm A"], "B": MODEL_COLOR["lightgbm B"]},
             category_orders={"feature": list(top)}, title="LightGBM feature importance (mean gain share %, top 15)")
fig.update_layout(**{**LAYOUT, "height": 520}, xaxis_title="gain share %", yaxis_title=None)
fig.show()
imp.pivot(index="feature", columns="variant", values="gain_share_pct").loc[top].round(2).T
'''),
md(r'''
**Reading it:** because the level is already handled by the scale-free target, the trees spend their effort on
**shape**: day of week (`dow`), the store (`store_nbr`) and family profile, day of month (`dom`: start-of-month and
payday rhythm, see notebook 01), and month. In variant A, the planned promo count (`promo_t` and its ratio to the
recent promo level, `promo_t_rel28`) together take a large share; in B that gain is redistributed mostly to weekday
and history features. Plain lags carry little gain: the same-weekday features (`sw_mean4`, `sw_last`) and rolling
means do that job better.

## 15. Where ML won, why - and where simple models were competitive

The table below asks the question per series: on the holdout, which of the five variant-A models had the lowest WAPE
for each of the 96 series?
'''),
code(r'''
ser = table(hold, ["family", "series_id"], C.MODELS[:-1] + ["lightgbm A"])
ser["winner"] = ser.idxmin(axis=1)
wins = ser.groupby(level="family").winner.value_counts().unstack(fill_value=0).reindex(C.FAMILIES).fillna(0).astype(int)
wins.loc["ALL"] = wins.sum()
display(wins)
close = ser[ser.winner != "lightgbm A"].copy()
close["gap to LightGBM A (pp)"] = close["lightgbm A"] - close[C.MODELS[:-1]].min(axis=1)
print(f"Series where a simple model beat LightGBM A on the holdout: {len(close)} of {len(ser)}")
close.sort_values("gap to LightGBM A (pp)", ascending=False).round(2).head(12)
'''),
code(r'''
# Horizon 1 and weekly totals: the two places where simple models came closest
gap_h = (hz[["holt_winters", "seasonal_naive"]].min(axis=1) - hz["lightgbm A"]).rename("best simple - LightGBM A (pp)")
print("Smallest daily gaps by horizon:")
display(gap_h.sort_values().head(4).round(2).to_frame().T)
cal_b = M.grouped_metrics(evaluable(bt[(bt.stage == "calibration") & (bt.mv == "lightgbm A")]), "family").set_index("family").bias
hold_b = M.grouped_metrics(hold[hold.mv == "lightgbm A"], "family").set_index("family").bias
print("LightGBM A bias % by family, calibration vs holdout:")
pd.DataFrame({"calibration": cal_b, "holdout": hold_b}).loc[C.FAMILIES].round(2).T
'''),
md(r'''
**Reading it - why ML won:**

- **It pools 96 series.** Holt-Winters fits each series alone on one year. LightGBM learns the weekday, day-of-month
  and holiday shapes once from all stores and families, so a quiet series borrows strength from busy ones.
- **It knows the calendar ahead of time.** Payday / start-of-month (`dom`), holidays and the promo plan are known for
  the target date; the smoothing models only see past sales.
- **It adapts its level at every origin.** The scale-free target re-anchors to the last 28 days, which keeps bias low
  without a trend term.
- The gains are largest in **GROCERY I, PERSONAL CARE and CLEANING**, where week-to-week shape and promo plans move
  more than in the steady fresh families.

**Where simple models were competitive:**

- **Per series:** a simple model beat LightGBM A on 13 of 96 series, 10 of them Holt-Winters, mostly in BEVERAGES,
  DAIRY and PRODUCE. In PRODUCE, DAIRY and BREAD/BAKERY Holt-Winters is within about 1-2 pp at family level: stable
  weekly shapes don't need a global model.
- **FROZEN FOODS:** seasonal naive, not Holt-Winters, is the best simple model, in selection and holdout alike.
- **Holiday days:** the moving average beat LightGBM A on the 176 holiday rows (section 12).
- **Weekly totals:** the moving average and seasonal naive come much closer at the weekly level (section 10),
  though LightGBM A is still better.
- **Day 1:** the gap between Holt-Winters and LightGBM is narrowest at horizon 1 (about 1.2 pp).

**Honesty notes:** the overall LightGBM under-forecast seen on calibration (about −2.4 %) did not carry into the
holdout (−0.46 % overall), but family biases on the holdout range from about −6 % (CLEANING) to +4 % (DAIRY,
FROZEN FOODS): the near-zero total hides offsetting family errors. Nothing was re-tuned for this, because tuning on
calibration would contaminate the intervals and Trust Index fitted there. The holdout is 6 origins in one summer (late June - mid August 2017), with no
December: FROZEN FOODS' hardest season is not tested.

## 16. Summary
'''),
code(r'''
lgA, hw_, sn = lb.loc["lightgbm A"], lb.loc["holt_winters"], lb.loc["seasonal_naive"]
summary = pd.DataFrame([
    ["Champion (all 8 families, chosen on selection)", "LightGBM, variant A"],
    ["Holdout WAPE - LightGBM A / Holt-Winters / seasonal naive", f"{lgA.wape:.2f}% / {hw_.wape:.2f}% / {sn.wape:.2f}%"],
    ["Holdout bias - LightGBM A", f"{lgA.bias:+.2f}%"],
    ["Families where LightGBM A had the lowest holdout WAPE", f"{int((fam[C.MODELS[:-1]].min(axis=1) > fam['lightgbm A']).sum())} of 8"],
    ["Series where a simple model beat LightGBM A", f"{len(close)} of {len(ser)}"],
    ["Known promo plan (A) vs none (B), holdout WAPE", f"{lb.loc['lightgbm A','wape']:.2f}% vs {lb.loc['lightgbm B','wape']:.2f}%"],
    ["Weekly-total WAPE, LightGBM A (week 1 / week 2)", f"{bkt.loc['week1','lightgbm A']:.2f}% / {bkt.loc['week2','lightgbm A']:.2f}%"],
], columns=["what", "result"]).set_index("what")
summary
'''),
md(r'''
Notebook 03 takes the champion forecasts and asks the next planner question: **how uncertain is each forecast, and
which ones deserve a human look?**
'''),
]

# =====================================================================================================
# NOTEBOOK 03
# =====================================================================================================
SETUP3 = SETUP + r'''
from scipy import stats
from src import trust as TR
from src.intervals import fit_quantiles, apply_intervals, bucket, origin_scales
sel = J("selection.json"); champs = sel["champions"]
ivj = J("intervals.json"); Q = pd.DataFrame(ivj["quantiles"])
tw = J("trust_weights.json"); tcal = J("trust_calibration.json"); res = J("holdout_results.json")
ci = pd.read_parquet(OUT / "holdout" / "champion_intervals.parquet")
ts = pd.read_parquet(OUT / "holdout" / "trust_scores.parquet")
hold_origins = pd.to_datetime(cal["stages"]["holdout"])
print("champion per family:", {f: c["model"] for f, c in champs.items()})
'''

NB03 = [
md(r'''
# 03 · Trust & Uncertainty

**Forecast vs Instinct** - Corporación Favorita (Ecuador), 12 stores × 8 families = 96 daily series, 14-day forecasts.

Notebook 02 picked a champion forecast (LightGBM, known promo plan, in every family). A point forecast alone is not
enough for a planner, who also needs to know:

1. **How wrong could it be?** → an **80 % prediction interval** (P10-P90) for every series-day.
2. **Which forecasts deserve a human look first?** → the **Forecast Trust Index**, a 0-100 score with three bands
   (Accept / Monitor / Review).

Both were fitted on the **calibration** stage (8 origins, late April - June 2017), frozen with SHA-256 hashes, and
then checked once on the **holdout** (6 origins, 28 June - 15 August 2017).

Ground rules:

- **Intervals are empirical, not guarantees.** "80 %" is a target measured on past errors; there is no promise any
  single interval contains the actual.
- **The Trust score is an index, not a probability.** A score of 85 does not mean "85 % chance the forecast is
  right"; it ranks series so review time goes where it is most needed.
- Sales are observed sales, not unconstrained demand; promo = count of promoted items (associations only).
- Nothing is re-trained or re-run: every number is computed here from saved backtests and frozen files.
'''),
code(SETUP3),
md(r'''
## 1. How the intervals are built: scaled split-conformal

The idea is simple: **look at how wrong the champion was on the calibration windows, and assume the future will be
wrong in a similar way.**

1. For every calibration row compute a **scaled residual** r = (actual − forecast) / scale_T, where scale_T is the
   series' mean sales over the 28 days up to the origin (known at T). Scaling lets one quantile serve a big store and
   a small one.
2. Group residuals by **family × horizon bucket** (days 1-7, days 8-14): 8 × 2 = 16 groups, 672 residuals each
   (8 origins × 12 stores × 7 days).
3. In each group take the **10th and 90th percentiles**, q_lo and q_hi.
4. For a new forecast F: **P10 = max(0, F + q_lo × scale_T)**, **P90 = max(0, F + q_hi × scale_T)**
   (the point forecast is kept inside its band).

The cell refits the quantiles from the saved calibration backtests and checks them against the frozen
`intervals.json`.
'''),
code(r'''
cal_rows = champion_rows(bt[bt.stage == "calibration"], champs, "A")
q_re = fit_quantiles(cal_rows, panel)
chk = q_re.merge(Q, on=["family", "bucket"], suffixes=("", "_frozen"))
print("max |refit - frozen| quantile:", float(np.abs(chk[["q_lo", "q_hi"]].values - chk[["q_lo_frozen", "q_hi_frozen"]].values).max().round(6)))
qt = Q.pivot(index="family", columns="bucket", values=["q_lo", "q_hi"]).loc[C.FAMILIES]
qt[("width", "1-7")] = qt[("q_hi", "1-7")] - qt[("q_lo", "1-7")]
qt[("width", "8-14")] = qt[("q_hi", "8-14")] - qt[("q_lo", "8-14")]
print(ivj["method"])
qt.round(3)
'''),
md(r'''
**Reading it:** in most family-buckets q_hi is larger in size than q_lo, so the bands are **asymmetric**: the
champion under-forecast more often than it over-forecast on calibration (its calibration bias was about −2.4 %, see
notebook 02), and the band leans upwards to absorb that. **FROZEN FOODS** gets the widest band (about −33 % to +35 %
of a typical recent day), PRODUCE the narrowest, then DAIRY and BREAD/BAKERY. Week 2 is not systematically wider than week 1, which matches the flat horizon profile in
notebook 02.

## 2. Did the intervals hit 80 %? Calibration vs holdout

Three coverage numbers (share of actuals inside P10-P90):

- **in-sample (calibration):** the quantiles scored on the same rows they were fitted on - close to 80 % by construction;
- **leave-one-origin-out (calibration):** refit the quantiles 8 times, each time leaving one origin out and scoring it -
  an honest preview of out-of-sample coverage;
- **holdout:** the frozen quantiles applied to the 6 holdout origins, run once.
'''),
code(r'''
ins = apply_intervals(evaluable(cal_rows), panel, q_re)
ins["inside"] = (ins.actual >= ins.p10) & (ins.actual <= ins.p90)
loo = []
for o in sorted(cal_rows.origin.unique()):
    qo = fit_quantiles(cal_rows[cal_rows.origin != o], panel)
    loo.append(apply_intervals(evaluable(cal_rows[cal_rows.origin == o]), panel, qo))
loo = pd.concat(loo)
loo["inside"] = (loo.actual >= loo.p10) & (loo.actual <= loo.p90)
ho = evaluable(ci[ci.stage == "holdout"]).copy()
ho["inside"] = (ho.actual >= ho.p10) & (ho.actual <= ho.p90)
ho["below"], ho["above"] = ho.actual < ho.p10, ho.actual > ho.p90
print(f"in-sample      {ins.inside.mean()*100:.2f}%   (intervals.json: {ivj['calibration_coverage_in_sample']:.2f}%)")
print(f"leave-one-out  {loo.inside.mean()*100:.2f}%   (intervals.json: {ivj['calibration_coverage_leave_one_origin_out']:.2f}%)")
print(f"holdout        {ho.inside.mean()*100:.2f}%   (holdout_results.json: {res['champion_system']['coverage']:.2f}%)")
print(f"holdout misses: {ho.below.mean()*100:.1f}% below P10, {ho.above.mean()*100:.1f}% above P90 (target 10% / 10%)")
cov = pd.DataFrame({"calibration in-sample": ins.groupby("family").inside.mean(),
                    "calibration leave-one-origin-out": loo.groupby("family").inside.mean(),
                    "holdout": ho.groupby("family").inside.mean(),
                    "holdout below P10": ho.groupby("family").below.mean(),
                    "holdout above P90": ho.groupby("family").above.mean()}).loc[C.FAMILIES] * 100
cov.round(1)
'''),
code(r'''
fig = go.Figure()
for name, col, dash in [("calibration leave-one-origin-out", "#8a8984", "dot"), ("holdout", "#e34948", "solid")]:
    s = cov[name]
    fig.add_scatter(x=s.round(1), y=s.index, mode="markers", name=name, marker=dict(size=11, color=col, symbol="circle" if name == "holdout" else "diamond"))
fig.add_vline(x=80, line_dash="dash", line_color="#52514e", annotation_text="80% target", annotation_position="top")
fig.update_layout(**{**LAYOUT, "height": 380}, title="P10-P90 coverage by family", xaxis_title="coverage %",
                  yaxis=dict(autorange="reversed"), xaxis_range=[55, 100])
fig.show()
'''),
md(r'''
**Reading it:** overall, the holdout coverage landed very close to the 80 % target, in line with the
leave-one-origin-out preview, and the misses split almost evenly between below P10 and above P90.
**The overall number hides a family spread**:

- **under-covered:** BEVERAGES (misses on both sides, more often above P90), BREAD/BAKERY (mostly below P10) and
  CLEANING - the bands were too narrow for these families in summer;
- **over-covered:** GROCERY I, PERSONAL CARE and DAIRY - the bands were wider than needed, consistent with these
  families' lower holdout WAPE than on calibration (GROCERY I, PERSONAL CARE).

Coverage is an average property over many rows; it says nothing about a single series-day.

### Coverage by horizon
'''),
code(r'''
covh = pd.DataFrame({"calibration leave-one-origin-out": loo.groupby("horizon").inside.mean() * 100,
                     "holdout": ho.groupby("horizon").inside.mean() * 100})
fig = go.Figure()
for name, col in [("calibration leave-one-origin-out", "#8a8984"), ("holdout", "#e34948")]:
    fig.add_scatter(x=covh.index, y=covh[name].round(1), name=name, mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=8))
fig.add_hline(y=80, line_dash="dash", line_color="#52514e")
fig.update_layout(**LAYOUT, title="P10-P90 coverage by horizon", xaxis=dict(title="horizon (days)", dtick=1),
                  yaxis_title="coverage %", yaxis_range=[50, 100])
fig.show()
covh.round(1).T
'''),
md(r'''
**Reading it:** there is no steady widening problem with the horizon, but there is a clear **weekday pattern**: the
two dips, at horizons 5 and 12, are both **Sundays** (origins are Tuesdays), in calibration and holdout alike. Sunday
is the peak sales day (notebook 01), its errors are larger, yet the quantiles are pooled over all seven days of a
bucket, so Sunday bands are too narrow and weekday bands slightly too wide. Quantiles by weekday (or scaled by the
forecast instead of the 28-day mean) would be the natural next improvement. Each horizon point rests on 576 rows
from only 6 calendar days, so individual points are noisy.

## 3. How wide are the bands?

Width = P90 − P10 in units, and relative to the point forecast (width ÷ forecast) so families can be compared.
'''),
code(r'''
ho["width"] = ho.p90 - ho.p10
wd = ho.groupby("family").apply(lambda g: pd.Series({
    "mean width (units)": g.width.mean(), "mean forecast (units)": g.forecast.mean(),
    "width / forecast %": 100 * g.width.sum() / g.forecast.sum(), "coverage %": 100 * g.inside.mean()})).loc[C.FAMILIES]
print(f"Champion system, holdout: mean width {M.mean_width(ho.p10, ho.p90):.1f} units "
      f"(holdout_results.json: {res['champion_system']['mean_width']:.1f}); "
      f"width / forecast {100 * ho.width.sum() / ho.forecast.sum():.1f}%")
display(wd.round(1))
tot = pd.read_parquet(OUT / "holdout" / "chain_total.parquet")
th = tot[(tot.stage == "holdout") & (tot.closed == 0) & tot.actual.notna()]
print(f"Chain total (sum of all 96 series, own quantiles): coverage {M.coverage(th.actual, th.p10, th.p90):.1f}% over {len(th)} days, "
      f"width / forecast {100 * (th.p90 - th.p10).sum() / th.forecast.sum():.1f}%")
'''),
md(r'''
**Reading it:** a band spans from about 28 % of the daily forecast (PRODUCE) to about 68 % (FROZEN FOODS), 37 %
overall. The **chain total** has its own band (fitted on total-level residuals),
because adding 96 series' P10s does not give the P10 of the sum: errors partly cancel, so the total band is far
narrower in relative terms.

## 4. The Forecast Trust Index: components and formulas

The question it answers at origin T: *how much should I trust this series' 14-day forecast?* Every input is known at
the end of day T. Five components, each 0-100 (higher = more trustworthy):

| Component | Formula | What it captures |
|---|---|---|
| Historical accuracy | 100 × (1 − min(WAPE_hist, 50) / 50) | champion WAPE for this series over the last ≤ 8 backtest windows already complete at T |
| Interval reliability | 100 × (1 − min(\|cov_hist − 80\|, 40) / 40) | did the P10-P90 band hit 80 % on those windows? |
| Demand stability | 100 × (1 − min(CV_56, 1)) × (1 − zero_share_56) | how erratic daily sales were over the last 56 days |
| Model agreement | 100 × (1 − min(cv_models / 0.5, 1)) | spread (std ÷ mean) of the 5 models' 14-day totals |
| Data completeness | 100 × n_ok_56 / 56 | share of the last 56 days that are open with an observed sale |

**Score** = Σ weight × component (weights sum to 1). **Bands:** Accept ≥ 80, Monitor 60-80, Review < 60.
**Hard flags force Review** regardless of score: completeness < 80 %, a chain closure (25 Dec / 1 Jan) in the window,
or a planned promo intensity above the series' historical 95th percentile (the forecast would be extrapolating).
**Priority** for the exception inbox = (100 − score) × the series' share of the total 14-day forecast.

Full definitions: `docs/agent_notes/trust_methodology.md`; code: `src/trust.py`. Frozen settings:
'''),
code(r'''
print("Frozen weights :", tw["weights"])
print("Default weights:", C.TRUST_WEIGHTS_DEFAULT)
print("Bands          :", [(b["band"], b["min"]) for b in tw["bands"]])
print("Constants      :", {k: v for k, v in tw["constants"].items() if k in ("wape_cap", "cov_target", "cov_tol", "cv_cap", "agree_cap", "lookback_days", "n_windows", "promo_quantile")})
'''),
md(r'''
## 5. Choosing the weights on calibration, by a rule written in advance

The default weights (30/25/20/15/10) were a hypothesis. On the 8 calibration origins (96 × 8 = 768 windows) the
candidates were: the default, equal weights, **volatility-only** (stability alone, the "naive planner" index), and every
weight vector on a 0.1-step grid. Objective: **Spearman rank correlation between the score and the realised champion
WAPE of the same window**. A useful index has a strongly **negative** correlation (high trust → low error).

**Pre-stated rule** (written before results were computed): keep the default unless another candidate is at least
**0.03 more negative** *and* also better on **each half** of the calibration origins. Because completeness was 100 on
every calibration window (a constant cannot be ranked), its weight was held at the default 0.10 and only the 220 grid
points that keep it there were eligible.
'''),
code(r'''
cand = pd.DataFrame([{"candidate": c["name"], "rho (all 768)": c["spearman"], "rho 1st half": c["spearman_first_half"],
                      "rho 2nd half": c["spearman_second_half"],
                      "weights acc/int/stab/agr/comp": "/".join(f"{c['weights'][k]:.2f}" for k in TR.COMPONENTS)}
                     for c in tcal["candidates"]]).set_index("candidate")
display(cand.round(3))
d, b = cand.loc["default"], cand.loc[tcal["chosen"]]
gain = d["rho (all 768)"] - b["rho (all 768)"]
print(f"Rule check: gain {gain:.3f} >= {tcal['margin']}? {gain >= tcal['margin']} | better on 1st half? {b['rho 1st half'] < d['rho 1st half']} "
      f"| better on 2nd half? {b['rho 2nd half'] < d['rho 2nd half']}")
print("Chosen:", tcal["chosen"], "-", tcal["choice_reason"])
print(f"Default's rank among the {tcal['grid_size']} eligible grid points: {tcal['grid_rank_of_default']}")
print("Single-component rho on calibration:", tcal["single_component_spearman"])
print("Calibration bands (with hard flags):", tcal["band_distribution"], "| realised WAPE by band:", tcal["realised_wape_by_band"])
'''),
md(r'''
**Reading it:** the chosen vector puts **70 % on historical accuracy**, 10 % each on interval reliability, stability
and completeness, and **0 on model agreement**. It beat the default by well over the 0.03 margin and on both halves,
so the rule adopted it. Accuracy on its own is almost as good as the chosen mix; that is a warning sign examined in
section 7. Because 220 grid points were compared, the best calibration rho is optimistic: we should expect the
holdout rho to be a bit weaker.

## 6. Holdout validation (run once, with the frozen weights)

On the 6 holdout origins (96 × 6 = 576 windows) the frozen index is scored against what actually happened. The cell
recomputes every statistic from the saved trust scores and holdout backtests and checks it against
`holdout_results.json`.
'''),
code(r'''
realised = TR.realised_window_wape(champion_rows(bt[bt.stage == "holdout"], champs, "A"))
hs = ts[ts.origin.isin(hold_origins)].copy()
val = TR.validate(hs, realised)
saved_val = res["trust_validation"]
for k in ["n_windows", "spearman", "error_ratio_review_vs_accept", "review_share_of_abs_error", "review_share_of_windows", "volatility_only_spearman"]:
    print(f"{k:32s} recomputed {val[k]!s:>8}   saved {saved_val[k]!s:>8}")
display(pd.DataFrame(val["wape_by_band"]).set_index("band"))
print(val["verdict"])
'''),
code(r'''
jv = hs.merge(realised, on=["series_id", "origin"])
fig = px.scatter(jv, x="score", y="wape", color="band", color_discrete_map=BAND_COLOR, opacity=0.7,
                 category_orders={"band": ["Accept", "Monitor", "Review"]},
                 hover_data={"series_id": True, "origin": "|%Y-%m-%d", "score": ":.1f", "wape": ":.1f"},
                 title="Holdout: Trust score at the origin vs realised 14-day WAPE (each dot = one series-window)")
fig.update_traces(marker_size=7)
for x in (60, 80):
    fig.add_vline(x=x, line_dash="dot", line_color="#52514e")
fig.update_layout(**{**LAYOUT, "height": 440}, xaxis_title="Trust score (0-100 index)", yaxis_title="realised WAPE %",
                  yaxis_range=[0, min(100, float(jv.wape.quantile(0.995)) + 5)])
fig.show()
'''),
md(r'''
**Reading it:**

- **Ranking works.** The Spearman correlation between score and realised error is strongly negative on the holdout,
  close to its calibration value, and better than the volatility-only baseline.
- **Bands separate error.** Review and Monitor windows had roughly twice the WAPE of Accept windows. The Review band
  held about a fifth of the windows but a larger share of the total absolute error, so a planner who only reviews
  that band spends time where more error sits - though not overwhelmingly more.
- **Monitor and Review are hard to tell apart on realised error** (their WAPEs are almost equal). About half of the
  Review band is there because of the promo hard flag, not a low score (section 7b).
- Some high-score windows still missed badly (dots in the upper right): the index ranks *on average*; it is not a
  promise about any one window.

## 7. Concerns, checked honestly

The cells below are **hindsight diagnostics**: they use the holdout to understand the index, and none of them changed
any frozen choice.

### 7a. The weights are dominated by accuracy
'''),
code(r'''
rows = []
for c in TR.COMPONENTS:
    r, _ = TR._spearman(jv[c], jv.wape)
    rows.append({"index": f"{c} only", "holdout rho": r})
for name, w in [("default 30/25/20/15/10", C.TRUST_WEIGHTS_DEFAULT), ("frozen (chosen) weights", tw["weights"])]:
    rows.append({"index": name, "holdout rho": TR._spearman(TR.score(jv, w), jv.wape)[0]})
display(pd.DataFrame(rows).set_index("index").round(3))
print(f"Spearman between the frozen score and the accuracy component alone: {TR._spearman(jv.score, jv.accuracy)[0]:.3f}")
'''),
md(r'''
**Reading it:** on the holdout, historical accuracy alone ranks windows as well as the full frozen score (marginally
better), and the score's rank correlation with the accuracy component alone is about 0.97. Interval reliability on its
own had essentially no ranking power on the holdout. In plain words, **the index mostly says "this
series was hard recently, so it will be hard again"**. That is useful (series difficulty is persistent) but it means
the other components add little ranking power on this data. Completeness is constant, so it has no rho. The frozen
weights still beat the default on the holdout - the choice made on calibration held up.

### 7b. The promo hard flag fires often
'''),
code(r'''
jv["promo_flag"] = jv.hard_flags.apply(lambda f: "promo_above_p95" in list(f))
jv["band_score_only"] = TR.bands_for(jv.score, None, tw["bands"]).values
pf = jv.groupby("promo_flag").apply(lambda g: pd.Series({"windows": len(g), "share %": 100 * len(g) / len(jv),
        "realised WAPE %": 100 * g.sum_abs_error.sum() / g.sum_actual.sum(), "mean score": g.score.mean()}))
display(pf.round(2))
print("Flagged windows by the band their SCORE alone would give:")
display(jv[jv.promo_flag].band_score_only.value_counts().to_frame("windows"))
by_sb = jv.groupby(["band_score_only", "promo_flag"]).apply(lambda g: 100 * g.sum_abs_error.sum() / g.sum_actual.sum()).unstack().round(2)
by_sb.columns = ["not flagged: WAPE %", "promo-flagged: WAPE %"]
forced = int((jv.promo_flag & (jv.band_score_only != "Review")).sum())
print(f"Review windows on the holdout: {int((jv.band == 'Review').sum())}, of which {forced} are there only because of the promo flag")
by_sb.reindex(["Accept", "Monitor", "Review"])
'''),
md(r'''
**Reading it:** the promo flag (planned promo intensity above the series' 95th percentile) fired on about 12 % of
holdout windows, because promoted-item counts trended up over 2015-2017, so the full-history 95th percentile is easy
to exceed. **On this holdout the flag pointed the wrong way:** flagged windows had *lower* realised WAPE than unflagged
ones, overall and within every score band, and almost all of them would have been Accept or Monitor on score alone.
So the flag pushed low-error forecasts into Review and diluted the Review band. It was kept as frozen (LightGBM trains
on that same history, so it was a reasonable "is this unusual?" reference), but it is the first thing to revisit,
e.g. a percentile over a recent window, or a flag only when the plan exceeds anything the model has seen.

### 7c. Completeness is a constant here
'''),
code(r'''
print("Completeness values on holdout windows:", jv.completeness.value_counts().to_dict())
print(f"Score range on holdout: {jv.score.min():.1f} - {jv.score.max():.1f} (median {jv.score.median():.1f})")
print("Holdout bands:", hs.band.value_counts().to_dict())
'''),
md(r'''
**Reading it:** every series was recorded on every open day (notebook 01 chose the scope that way), so completeness
is 100 everywhere and its 10 % weight is a constant +10 points. It would only matter for series with recording gaps.
Also, because accuracy rarely approaches 100, scores top out in the mid-80s: Accept effectively means "near the best
historical accuracy in the portfolio".

## 8. Worked example: one series, one holdout origin, end to end

To make the machinery concrete, we trace **one real series at one holdout origin** from forecast to realised error,
recomputing every number from raw inputs and checking it against the saved scores.

Choice rule (fixed before looking at its outcome): at the **middle holdout origin** (the 3rd of 6), take the series
**without a hard flag whose Trust score is closest to the median score of that origin** - a typical, not a cherry-picked,
case.
'''),
code(r'''
T = hold_origins[2]
at_T = ts[(ts.origin == T) & (ts.hard_flags.apply(len) == 0)]
med = ts[ts.origin == T].score.median()
ex = at_T.iloc[(at_T.score - med).abs().argsort().iloc[0]]
SID, FAMILY = ex.series_id, ex.family
print(f"Origin T = {T:%Y-%m-%d} (forecasting {T + pd.Timedelta(days=1):%Y-%m-%d} .. {T + pd.Timedelta(days=14):%Y-%m-%d})")
print(f"Median score at T = {med:.2f}  ->  chosen series: {SID} (store {ex.store_nbr}, {FAMILY}), saved score {ex.score:.2f}, band {ex.band}")
print(f"Champion for {FAMILY}: {champs[FAMILY]['model']} (variant A)")
'''),
md(r'''
### Step 1 - the forecast and its interval

scale_T = mean of the last 28 days of sales up to T. For days 1-7 the family's q_lo / q_hi for bucket "1-7" apply, for
days 8-14 those of "8-14".
'''),
code(r'''
hist28 = panel[(panel.series_id == SID) & (panel.date > T - pd.Timedelta(days=28)) & (panel.date <= T)]
scale_T = hist28.sales.mean()
w = ci[(ci.series_id == SID) & (ci.origin == T)].sort_values("horizon").copy()
assert np.isclose(scale_T, w.scale.iloc[0])
qf = Q[Q.family == FAMILY].set_index("bucket")
w["p10 (by hand)"] = np.minimum(np.maximum(0, w.forecast + w.bucket.map(qf.q_lo) * scale_T), w.forecast)
w["p90 (by hand)"] = np.maximum(np.maximum(0, w.forecast + w.bucket.map(qf.q_hi) * scale_T), w.forecast)
assert np.allclose(w["p10 (by hand)"], w.p10) and np.allclose(w["p90 (by hand)"], w.p90)
w["inside"] = (w.actual >= w.p10) & (w.actual <= w.p90)
print(f"scale_T = {scale_T:.2f} | {FAMILY} quantiles: 1-7 [{qf.loc['1-7','q_lo']:.3f}, {qf.loc['1-7','q_hi']:.3f}], "
      f"8-14 [{qf.loc['8-14','q_lo']:.3f}, {qf.loc['8-14','q_hi']:.3f}]  (hand-computed bands match the saved ones)")
w[["horizon", "target_date", "onpromotion", "forecast", "p10", "p90", "actual", "inside"]].assign(target_date=lambda d: d.target_date.dt.strftime("%a %d %b")).round(1).set_index("horizon").T
'''),
code(r'''
past = panel[(panel.series_id == SID) & (panel.date > T - pd.Timedelta(days=42)) & (panel.date <= T)]
fig = go.Figure()
fig.add_scatter(x=list(w.target_date) + list(w.target_date[::-1]), y=list(w.p90) + list(w.p10[::-1]), fill="toself",
                fillcolor="rgba(235,104,52,0.18)", line=dict(width=0), name="P10-P90 band", hoverinfo="skip")
fig.add_scatter(x=past.date, y=past.sales, mode="lines", name="sales (known at T)", line=dict(color="#52514e", width=2))
fig.add_scatter(x=w.target_date, y=w.forecast, mode="lines+markers", name="champion forecast", line=dict(color="#eb6834", width=2), marker=dict(size=7))
fig.add_scatter(x=w.target_date, y=w.actual, mode="markers", name="actual (revealed later)", marker=dict(color="#2a78d6", size=9, symbol="diamond"))
fig.add_vline(x=T.timestamp() * 1000, line_dash="dash", line_color="#52514e")
fig.add_annotation(x=T, y=1.02, yref="paper", text="origin T", showarrow=False, yanchor="bottom")
fig.update_layout(**{**LAYOUT, "height": 400}, title=f"{SID}: 6 weeks of history, then the 14-day forecast with its 80% band",
                  yaxis_title="daily sales (units)", xaxis_title=None)
fig.show()
'''),
md(r'''
### Step 2 - each Trust component, recomputed from raw inputs

Everything uses information available at the end of day T only.
'''),
code(r'''
K = tw["constants"]
chA = champion_rows(bt[bt.variant == "A"], champs, "A")
chA = chA[chA.series_id == SID]
done = sorted(o for o in chA.origin.unique() if o + pd.Timedelta(days=K["horizon"]) <= T)[-K["n_windows"]:]
h = evaluable(chA[chA.origin.isin(done) & (chA.target_date <= T)])
wape_hist = 100 * (h.actual - h.forecast).abs().sum() / h.actual.sum()
acc = 100 * (1 - min(wape_hist, K["wape_cap"]) / K["wape_cap"])

hiv = apply_intervals(h, panel[panel.date <= T], Q)
cov_hist = 100 * ((hiv.actual >= hiv.p10) & (hiv.actual <= hiv.p90)).mean()
inter = 100 * (1 - min(abs(cov_hist - K["cov_target"]), K["cov_tol"]) / K["cov_tol"])

l56 = panel[(panel.series_id == SID) & (panel.date > T - pd.Timedelta(days=K["lookback_days"])) & (panel.date <= T)]
ok = l56[(l56.closed == 0) & l56.sales.notna()].sales
cv56, zero56 = ok.std(ddof=0) / ok.mean(), (ok == 0).mean()
stab = 100 * (1 - min(cv56, K["cv_cap"]) / K["cv_cap"]) * (1 - zero56)

tots = bt[(bt.variant == "A") & (bt.origin == T) & (bt.series_id == SID)].groupby("model").forecast.sum()
mcv = tots.std(ddof=0) / tots.mean()
agr = 100 * (1 - min(mcv / K["agree_cap"], 1))
comp_ = 100 * len(ok) / K["lookback_days"]

roll = panel[panel.series_id == SID].sort_values("date").set_index("date").onpromotion.rolling(K["promo_window"], min_periods=K["promo_window"]).mean()
promo_p95 = roll[roll.index <= T].quantile(K["promo_quantile"])
promo_plan = w.onpromotion.mean()

print(f"History windows used (origins with o + 14 <= T): {[f'{o:%m-%d}' for o in done]}")
print("Five models' 14-day forecast totals at T:", {m: round(v) for m, v in tots.items()})
steps = pd.DataFrame([
    ["accuracy", f"WAPE_hist = {wape_hist:.2f}%", f"100 × (1 − min({wape_hist:.2f}, 50)/50)", acc, ex.accuracy],
    ["interval", f"cov_hist = {cov_hist:.1f}% of {len(hiv)} days", f"100 × (1 − min(|{cov_hist:.1f} − 80|, 40)/40)", inter, ex.interval],
    ["stability", f"CV_56 = {cv56:.3f}, zero share = {zero56:.2f}", f"100 × (1 − min({cv56:.3f}, 1)) × (1 − {zero56:.2f})", stab, ex.stability],
    ["agreement", f"5 models' 14-day totals, mean {tots.mean():,.0f}", f"cv_models = {mcv:.3f} → 100 × (1 − min({mcv:.3f}/0.5, 1))", agr, ex.agreement],
    ["completeness", f"{len(ok)} of 56 days open with a sale", f"100 × {len(ok)}/56", comp_, ex.completeness],
], columns=["component", "input", "formula", "by hand", "saved"]).set_index("component")
assert np.allclose(steps["by hand"].astype(float), steps["saved"].astype(float))
print(f"Promo flag check: planned mean promo {promo_plan:.1f} vs series p95 of 14-day mean {promo_p95:.1f} -> "
      f"{'FLAG' if promo_plan > promo_p95 else 'no flag'} (saved: plan {ex.promo_plan_mean:.1f}, p95 {ex.promo_p95:.1f})")
steps.round(2)
'''),
md(r'''
### Step 3 - score, band, and what actually happened
'''),
code(r'''
W = tw["weights"]
contrib = pd.Series({c: W[c] * steps.loc[c, "by hand"] for c in TR.COMPONENTS})
score_hand = contrib.sum()
band_hand = TR.band(score_hand, [], tw["bands"])
display(pd.DataFrame({"weight": pd.Series(W), "component": steps["by hand"].astype(float), "weight × component": contrib}).round(2))
print(f"score = {score_hand:.2f} (saved {ex.score:.2f}) -> band {band_hand} (saved {ex.band})  [an index, not a probability]")

we = evaluable(w)
win_wape = 100 * (we.actual - we.forecast).abs().sum() / we.actual.sum()
win_bias = 100 * (we.forecast - we.actual).sum() / we.actual.sum()
band_wape = {r["band"]: r["wape"] for r in val["wape_by_band"]}
print(f"Realised: 14-day WAPE {win_wape:.2f}%, bias {win_bias:+.2f}%, {int(we.inside.sum())} of {len(we)} days inside the 80% band")
print(f"For comparison, holdout WAPE of all {band_hand} windows: {band_wape[band_hand]:.2f}%; this series' own WAPE_hist was {wape_hist:.2f}%")
r = realised[(realised.series_id == SID) & (realised.origin == T)].iloc[0]
assert np.isclose(r.wape, win_wape)
'''),
md(r'''
**Reading it:** the example series is store 36 DAIRY at origin 11 July 2017. Its score of 79.56 is mostly the
accuracy component (0.7 × 76.73 = 53.71 points), i.e. its recent track record of 11.6 % WAPE; interval, stability and
the constant completeness add the rest. Model agreement was high (the five models' 14-day totals were close) but has
zero weight, so it changed nothing. The score fell just short of 80, so the band is **Monitor**, not Accept.

What happened: a 6.1 % WAPE fortnight, better than the series' own history and far better than the Monitor-band
average, with all 14 actuals inside the band (DAIRY was one of the over-covered families). A slight over-forecast
(+3.7 %) is visible in the chart. One window is one draw: a typical score can come with an unusually good or bad
fortnight, which is why the index is judged on its ranking over hundreds of windows (section 6), not on a single case.

## 9. Summary
'''),
code(r'''
vb = {r["band"]: r for r in val["wape_by_band"]}
pd.DataFrame([
    ["Interval method", "scaled split-conformal, P10/P90 per family × horizon bucket, fitted on 8 calibration origins"],
    ["Coverage: calibration in-sample / leave-one-origin-out / holdout", f"{ins.inside.mean()*100:.1f}% / {loo.inside.mean()*100:.1f}% / {ho.inside.mean()*100:.1f}% (target 80%)"],
    ["Family holdout coverage range", f"{cov.holdout.min():.1f}% ({cov.holdout.idxmin()}) - {cov.holdout.max():.1f}% ({cov.holdout.idxmax()})"],
    ["Mean band width / forecast (holdout)", f"{100 * ho.width.sum() / ho.forecast.sum():.1f}%"],
    ["Trust weights (acc/int/stab/agr/comp)", "/".join(f"{tw['weights'][c]:.1f}" for c in TR.COMPONENTS) + " (chosen on calibration by a pre-stated rule)"],
    ["Holdout Spearman(score, realised WAPE)", f"{val['spearman']:.3f} vs volatility-only {val['volatility_only_spearman']:.3f}"],
    ["Holdout WAPE: Accept / Monitor / Review", f"{vb['Accept']['wape']:.2f}% / {vb['Monitor']['wape']:.2f}% / {vb['Review']['wape']:.2f}%"],
    ["Review band: share of windows / share of absolute error", f"{val['review_share_of_windows']:.1f}% / {val['review_share_of_abs_error']:.1f}%"],
    ["Main caveats", "accuracy-dominated weights; frequent promo flag; completeness constant; 6 summer origins only"],
], columns=["what", "result"]).set_index("what")
'''),
]


def build(execute: bool = False) -> list[Path]:
    paths = []
    for name, cells in [("02_forecasting_battle.ipynb", NB02), ("03_trust_and_uncertainty.ipynb", NB03)]:
        nb = new_notebook(cells=cells, metadata=META)
        path = NB_DIR / name
        nbformat.write(nb, path)
        paths.append(path)
    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        for path in paths:
            nb = nbformat.read(path, as_version=4)
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
            nbformat.write(nb, path)
    for p in paths:
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute both notebooks in place after building")
    build(ap.parse_args().execute)
