"""Build the planner workbook excel/forecast_vs_instinct_planning_workbook.xlsx (openpyxl, reproducible).

Reads only finished pipeline outputs (no model is rerun):
  outputs/holdout/champion_intervals.parquet  forward 14-day forecast + P10/P90, holdout actuals
  outputs/holdout/trust_scores.parquet        Trust Index at the forward origin 2017-08-15
  outputs/holdout_results.json                leaderboard, WAPE by family x model, trust validation
  outputs/selection.json, outputs/intervals.json, outputs/trust_weights.json, config/scope.json

Every calculation a planner sees is a live Excel formula (totals, trust score and band, priority,
consensus, scenario arithmetic). Stored values are only data and finished results.
The Scenario sheet mirrors src/scenario.py exactly (see tests/test_workbook.py).

Run: .venv/bin/python scripts/09_build_workbook.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import scenario as SC  # noqa: E402
from src.backtest import evaluable  # noqa: E402
from src.metrics import bias as m_bias, wape as m_wape  # noqa: E402
from src.trust import describe_flags  # noqa: E402

OUT = ROOT / "excel" / "forecast_vs_instinct_planning_workbook.xlsx"
FORWARD_ORIGIN = "2017-08-15"

# ---------------------------------------------------------------------------------------------
# layout constants (the test imports these, so the workbook and its checks cannot drift apart)
# ---------------------------------------------------------------------------------------------
FC_HEADER_ROW = 4
FC_FIRST = 5
N_SERIES = 96
MEASURES = ["Forecast", "P10", "P90"]
FC_LAST = FC_FIRST + N_SERIES * len(MEASURES) - 1           # 292
FC_TOTAL_ROW = FC_LAST + 2                                  # 294 (chain totals by measure)
FC_DATE_COL1 = 5                                            # E
FC_TOTAL_COL = FC_DATE_COL1 + 14                            # S

EX_WEIGHT_ROW = 5
EX_HEADER_ROW = 6
EX_FIRST = 7
EX_LAST = EX_FIRST + N_SERIES - 1                           # 102

OV_HEADER_ROW = 5
OV_FIRST = 6
OV_LAST = OV_FIRST + N_SERIES - 1                           # 101
OV_TOTAL_ROW = OV_LAST + 1                                  # 102

REASON_CODES = ["Upcoming promotion", "Market intelligence", "Supply disruption", "Exceptional event",
                "Suspected forecast error"]
ASSUMPTION_LABEL = "Editable demonstration assumptions, not measured UAE uplift"
TAGLINE = "Beyond predicting demand. Knowing when to trust the prediction."
DEFAULT_FAMILY = "BEVERAGES"
# Same defaults as the dashboard Scenario Lab (docs/js/views/scenario_lab.js): Ramadan event
# 5/15/30 %, lead time 14 d, event 28 d after the planning date, under:over cost 3:1, supply = baseline.
SCN_DEFAULTS = {"pct_low": 0.05, "pct_base": 0.15, "pct_high": 0.30, "lead": 14, "days_to_event": 28,
                "cu": 3, "co": 1}

# Scenario cell map (the test evaluates these and compares with src/scenario.run_scenario)
S = {
    "family": "B6", "baseline": "B7", "pct_low": "B8", "pct_base": "B9", "pct_high": "B10", "supply": "B11",
    "lead": "B12", "plan_date": "B13", "event_date": "B14", "cu": "B15", "co": "B16",
    "w_low": "B19", "w_base": "B20", "w_high": "B21", "review_window": "B22",
    "baseline_used": "B25", "supply_used": "B26", "cu_used": "B27", "co_used": "B28",
    # scenario table rows 31..33 (Low, Base, High): C demand, D shortfall, E excess, F coverage, G weight
    # cost matrix rows 37..39 (+ 40 keep supply): B qty, C/D/E outcome cost, F expected, G regret, H/I keys
    "recommended": "B43", "rec_qty": "B44", "rec_cost": "B45", "minimax": "B46", "critical": "B47",
    "gap": "B48", "days_to_event": "B49", "order_by": "B50", "order_in": "B51", "status": "B52",
    "timing_msg": "B53", "part1": "B54", "part_timing": "B55", "part2": "B56", "part3": "B57",
    "recommendation": "B58",
}
SCN_ROW = {"low": 31, "base": 32, "high": 33}
MAT_ROW = {"low": 37, "base": 38, "high": 39}
SUPPLY_ROW = 40

# ---------------------------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------------------------
NAVY, NAVY2, ACCENT = "01204C", "0F2A54", "60F0D8"
BAND_FILL = {"Accept": "60F0D8", "Monitor": "F5C26B", "Review": "FF6B8A"}
INPUT_FILL = "E6FBF6"
FONT = "Montserrat"
F_BODY = Font(name=FONT, size=10, color="1A1A2E")
F_BOLD = Font(name=FONT, size=10, bold=True, color="1A1A2E")
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_TITLE = Font(name=FONT, size=16, bold=True, color="FFFFFF")
F_SUB = Font(name=FONT, size=10, italic=True, color=ACCENT)
F_NOTE = Font(name=FONT, size=9, italic=True, color="4A5568")
F_SECTION = Font(name=FONT, size=11, bold=True, color=NAVY)
FILL_NAVY = PatternFill("solid", fgColor=NAVY)
FILL_HEAD = PatternFill("solid", fgColor=NAVY2)
FILL_INPUT = PatternFill("solid", fgColor=INPUT_FILL)
FILL_TOTAL = PatternFill("solid", fgColor="DCE3F0")
THIN = Side(style="thin", color="C5CCE0")
ACC_SIDE = Side(style="medium", color=ACCENT)
B_CELL = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
B_INPUT = Border(left=ACC_SIDE, right=ACC_SIDE, top=ACC_SIDE, bottom=ACC_SIDE)
B_TOTAL = Border(top=Side(style="medium", color=NAVY), bottom=Side(style="medium", color=NAVY))
WRAP = Alignment(wrap_text=True, vertical="top")
NUM, PCT, PCT2, DEC1, DEC3, DATE = "#,##0", "0.0%", "0.00%", "0.0", "0.000", "dd mmm yyyy"


def title(ws, text, sub=None, width_cols=12):
    for c in range(1, width_cols + 1):
        ws.cell(1, c).fill = FILL_NAVY
        ws.cell(2, c).fill = FILL_NAVY
    ws.cell(1, 1, text).font = F_TITLE
    ws.row_dimensions[1].height = 28
    if sub:
        ws.cell(2, 1, sub).font = F_SUB


def header(ws, row, labels, col1=1, fill=FILL_HEAD):
    for i, lab in enumerate(labels):
        c = ws.cell(row, col1 + i, lab)
        c.font, c.fill, c.border = F_HEAD, fill, B_CELL
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[row].height = 32


def put(ws, ref, value, fmt=None, font=F_BODY, fill=None, border=B_CELL, align=None):
    c = ws[ref] if isinstance(ref, str) else ws.cell(*ref)
    c.value = value
    c.font = font
    if fmt:
        c.number_format = fmt
    if fill:
        c.fill = fill
    if border:
        c.border = border
    if align:
        c.alignment = align
    return c


def as_input(c):
    c.fill, c.border = FILL_INPUT, B_INPUT
    return c


def widths(ws, mapping):
    for col, w in mapping.items():
        ws.column_dimensions[col].width = w


def section(ws, row, text, col=1):
    ws.cell(row, col, text).font = F_SECTION


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load():
    ci = pd.read_parquet(ROOT / "outputs/holdout/champion_intervals.parquet")
    fwd = ci[ci["stage"] == "forward"].copy()
    hold = evaluable(ci[ci["stage"] == "holdout"])
    trust = pd.read_parquet(ROOT / "outputs/holdout/trust_scores.parquet")
    trust = trust[trust["origin"] == pd.Timestamp(FORWARD_ORIGIN)].copy()
    res = json.loads((ROOT / "outputs/holdout_results.json").read_text())
    sel = json.loads((ROOT / "outputs/selection.json").read_text())
    intervals = json.loads((ROOT / "outputs/intervals.json").read_text())
    tw = json.loads((ROOT / "outputs/trust_weights.json").read_text())
    scope = json.loads((ROOT / "config/scope.json").read_text())
    assert fwd["series_id"].nunique() == N_SERIES and len(trust) == N_SERIES
    assert fwd["horizon"].nunique() == 14
    return fwd, hold, trust, res, sel, intervals, tw, scope


def series_order(fwd):
    s = fwd[["series_id", "store_nbr", "family"]].drop_duplicates()
    return s.sort_values(["store_nbr", "family"]).reset_index(drop=True)


def fc_row(i_series: int, measure: str) -> int:
    return FC_FIRST + i_series * len(MEASURES) + MEASURES.index(measure)


# ---------------------------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------------------------
def build_forecast(wb, fwd, order, intervals):
    ws = wb.create_sheet("Forecast")
    dates = sorted(fwd["target_date"].unique())
    title(ws, "14-day forecast by series (champion LightGBM, P10/P90)",
          f"Forecast origin {FORWARD_ORIGIN} (last actual). Target days {pd.Timestamp(dates[0]):%d %b} to "
          f"{pd.Timestamp(dates[-1]):%d %b %Y}. Units = observed unit sales. Filter column D to see one measure.",
          FC_TOTAL_COL)
    ws["A3"] = (f"P10/P90 = {int(intervals['level'] * 100)}% scaled split-conformal band per day. The 14-day total of "
                "daily P10 (or P90) is a sum of daily bounds, not a P10/P90 of the 14-day total.")
    ws["A3"].font = F_NOTE
    hdr = ["Series", "Store", "Family", "Measure"] + [pd.Timestamp(d).to_pydatetime() for d in dates] + ["14-day total"]
    header(ws, FC_HEADER_ROW, hdr)
    for j in range(14):
        ws.cell(FC_HEADER_ROW, FC_DATE_COL1 + j).number_format = "ddd dd mmm"
    piv = {m: fwd.pivot(index="series_id", columns="target_date", values=col)
           for m, col in zip(MEASURES, ["forecast", "p10", "p90"])}
    first_d, last_d = get_column_letter(FC_DATE_COL1), get_column_letter(FC_DATE_COL1 + 13)
    for i, r in order.iterrows():
        for m in MEASURES:
            row = fc_row(i, m)
            bold = F_BOLD if m == "Forecast" else F_BODY
            put(ws, (row, 1), r["series_id"], font=bold)
            put(ws, (row, 2), int(r["store_nbr"]))
            put(ws, (row, 3), r["family"])
            put(ws, (row, 4), m, font=bold)
            vals = piv[m].loc[r["series_id"], dates].tolist()
            for j, v in enumerate(vals):
                put(ws, (row, FC_DATE_COL1 + j), float(v), NUM, font=bold)
            put(ws, (row, FC_TOTAL_COL), f"=SUM({first_d}{row}:{last_d}{row})", NUM, font=F_BOLD,
                fill=FILL_TOTAL)
    # chain totals by measure
    for k, m in enumerate(MEASURES):
        row = FC_TOTAL_ROW + k
        put(ws, (row, 1), f"All {N_SERIES} series", font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
        for c in (2, 3):
            put(ws, (row, c), None, fill=FILL_TOTAL, border=B_TOTAL)
        put(ws, (row, 4), m, font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
        for col in range(FC_DATE_COL1, FC_TOTAL_COL + 1):
            L = get_column_letter(col)
            put(ws, (row, col), f'=SUMIF($D${FC_FIRST}:$D${FC_LAST},$D{row},{L}${FC_FIRST}:{L}${FC_LAST})', NUM,
                font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    ws.freeze_panes = ws.cell(FC_FIRST, FC_DATE_COL1)
    ws.auto_filter.ref = f"A{FC_HEADER_ROW}:{get_column_letter(FC_TOTAL_COL)}{FC_LAST}"
    widths(ws, {"A": 22, "B": 7, "C": 16, "D": 10})
    for col in range(FC_DATE_COL1, FC_TOTAL_COL):
        ws.column_dimensions[get_column_letter(col)].width = 11
    ws.column_dimensions[get_column_letter(FC_TOTAL_COL)].width = 14
    return ws


MODEL_COLS = [("naive", "Naive"), ("seasonal_naive", "Seasonal naive"), ("moving_average", "Moving average"),
              ("holt_winters", "Holt-Winters"), ("lightgbm", "LightGBM")]
ACC = {"lb_first": 5, "best_row": 12, "sys_first": 15, "fam_header": 22, "fam_first": 23, "tv_header": 34}
ACC["lb_last"] = ACC["lb_first"] + 5
ACC["fam_last"] = ACC["fam_first"] + 7


def build_accuracy(wb, res, sel, hold):
    ws = wb.create_sheet("Accuracy")
    title(ws, "Holdout accuracy: 6 untouched origins (27 Jun to 1 Aug 2017), 96 series x 14 days",
          "WAPE = sum|A-F| / sum A. Bias = sum(F-A) / sum A (positive = over-forecast). Results from "
          "outputs/holdout_results.json; best-model columns are live formulas.", 12)
    # leaderboard
    section(ws, 3, "Overall leaderboard (all 96 series)")
    header(ws, 4, ["Model", "Variant", "WAPE", "Bias", "MAE (units)", "RMSE (units)", "n (series-days)",
                   "Rank by WAPE"])
    for k, r in enumerate(res["leaderboard"]):
        row = ACC["lb_first"] + k
        name = dict(MODEL_COLS)[r["model"]]
        put(ws, (row, 1), name)
        put(ws, (row, 2), r["variant"])
        put(ws, (row, 3), r["wape"] / 100, PCT)
        put(ws, (row, 4), r["bias"] / 100, PCT2)
        put(ws, (row, 5), r["mae"], DEC1)
        put(ws, (row, 6), r["rmse"], DEC1)
        put(ws, (row, 7), r["n"], NUM)
        put(ws, (row, 8), f"=RANK(C{row},$C${ACC['lb_first']}:$C${ACC['lb_last']},1)", "0")
    a, b = ACC["lb_first"], ACC["lb_last"]
    r = ACC["best_row"]
    put(ws, (r, 1), "Best model on holdout (lowest WAPE)", font=F_BOLD, border=None)
    put(ws, (r, 3), f'=CONCATENATE(INDEX($A${a}:$A${b},MATCH(MIN($C${a}:$C${b}),$C${a}:$C${b},0))," (",'
                    f'INDEX($B${a}:$B${b},MATCH(MIN($C${a}:$C${b}),$C${a}:$C${b},0)),")")', font=F_BOLD)
    put(ws, (r + 1, 1), "Lowest holdout WAPE", font=F_BOLD, border=None)
    put(ws, (r + 1, 3), f"=MIN($C${a}:$C${b})", PCT, font=F_BOLD)
    ws.conditional_formatting.add(f"C{a}:C{b}", ColorScaleRule(start_type="min", start_color=ACCENT,
                                                               mid_type="percentile", mid_value=50,
                                                               mid_color="F5C26B", end_type="max",
                                                               end_color=BAND_FILL["Review"]))
    # champion system
    cs = res["champion_system"]
    section(ws, ACC["sys_first"] - 1, "Champion system (LightGBM variant A for every family, frozen before holdout)")
    champ_lb = res["leaderboard"][0]
    champ_label = f"{dict(MODEL_COLS)[champ_lb['model']]} (variant {champ_lb['variant']})"
    assert all(v["model"] == champ_lb["model"] for v in sel["champions"].values())
    assert abs(champ_lb["wape"] - cs["wape"]) < 1e-9
    rows = [("Champion WAPE", cs["wape"] / 100, PCT), ("Champion bias", cs["bias"] / 100, PCT2),
            ("P10-P90 coverage (target 80%)", cs["coverage"] / 100, PCT),
            ("Mean P10-P90 width (units/day)", cs["mean_width"], DEC1), ("Series-days evaluated", cs["n"], NUM),
            ("Champion", champ_label, None)]
    for k, (lab, v, f) in enumerate(rows):
        put(ws, (ACC["sys_first"] + k, 1), lab, font=F_BOLD)
        put(ws, (ACC["sys_first"] + k, 3), v, f)
    # by family x model
    section(ws, ACC["fam_header"] - 1, "Holdout WAPE by family x model (12 stores per family)")
    header(ws, ACC["fam_header"], ["Family"] + [n for _, n in MODEL_COLS] +
           ["Champion (frozen on selection)", "Best on holdout", "Best WAPE", "Champion WAPE",
            "Champion gap to best (pp)", "Champion bias", "Champion P10-P90 coverage"])
    fam_metrics = {f: (m_bias(g["actual"], g["forecast"]) / 100,
                       float(((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()))
                   for f, g in hold.groupby("family")}
    hc = f"$B${ACC['fam_header']}:$F${ACC['fam_header']}"
    for k, r in enumerate(res["by_family"]):
        row = ACC["fam_first"] + k
        put(ws, (row, 1), r["family"], font=F_BOLD)
        for j, (key, _) in enumerate(MODEL_COLS):
            put(ws, (row, 2 + j), r[key] / 100, PCT)
        champ = dict(MODEL_COLS)[sel["champions"][r["family"]]["model"]]
        put(ws, (row, 7), champ)
        put(ws, (row, 8), f"=INDEX({hc},MATCH(MIN(B{row}:F{row}),B{row}:F{row},0))", font=F_BOLD)
        put(ws, (row, 9), f"=MIN(B{row}:F{row})", PCT)
        put(ws, (row, 10), f"=INDEX(B{row}:F{row},MATCH(G{row},{hc},0))", PCT)
        put(ws, (row, 11), f"=(J{row}-I{row})*100", DEC1)
        put(ws, (row, 12), fam_metrics[r["family"]][0], PCT2)
        put(ws, (row, 13), fam_metrics[r["family"]][1], PCT)
    f1, f2 = ACC["fam_first"], ACC["fam_last"]
    # highlight the row minimum and a colour scale on the grid
    ws.conditional_formatting.add(f"B{f1}:F{f2}", FormulaRule(formula=[f"B{f1}=MIN($B{f1}:$F{f1})"],
                                                              fill=PatternFill("solid", fgColor=ACCENT),
                                                              font=Font(name=FONT, bold=True, color=NAVY)))
    ws.conditional_formatting.add(f"B{f1}:F{f2}", ColorScaleRule(start_type="min", start_color="FFFFFF",
                                                                 end_type="max", end_color="FFC9D5"))
    ws.conditional_formatting.add(f"K{f1}:K{f2}", CellIsRule(operator="greaterThan", formula=["0.05"],
                                                             fill=PatternFill("solid", fgColor="F5C26B")))
    # trust validation (holdout)
    tv = res["trust_validation"]
    t = ACC["tv_header"]
    section(ws, t - 1, "Does the Trust Index rank risk? Holdout check (576 series x origin windows)")
    header(ws, t, ["Trust band", "Windows", "Realised champion WAPE", "Share of windows"])
    for k, b in enumerate(tv["wape_by_band"]):
        row = t + 1 + k
        put(ws, (row, 1), b["band"], font=F_BOLD)
        put(ws, (row, 2), b["n"], NUM)
        put(ws, (row, 3), b["wape"] / 100, PCT)
        put(ws, (row, 4), f"=B{row}/SUM($B${t + 1}:$B${t + 3})", PCT)
        ws.cell(row, 1).fill = PatternFill("solid", fgColor=BAND_FILL[b["band"]])
    r = t + 4
    put(ws, (r, 1), "Review WAPE / Accept WAPE", font=F_BOLD)
    put(ws, (r, 3), f'=INDEX(C{t + 1}:C{t + 3},MATCH("Review",A{t + 1}:A{t + 3},0))/'
                    f'INDEX(C{t + 1}:C{t + 3},MATCH("Accept",A{t + 1}:A{t + 3},0))', '0.00"x"', font=F_BOLD)
    put(ws, (r + 1, 1), "Spearman(score, realised WAPE)", font=F_BOLD)
    put(ws, (r + 1, 3), tv["spearman"], "0.00")
    put(ws, (r + 2, 1), "Verdict", font=F_BOLD)
    put(ws, (r + 2, 2), tv["verdict"], align=WRAP, border=None)
    ws.merge_cells(start_row=r + 2, start_column=2, end_row=r + 2, end_column=9)
    ws.row_dimensions[r + 2].height = 48
    ACC["review_ratio"] = f"C{r}"
    ws.freeze_panes = "B5"
    widths(ws, {"A": 34, "B": 13, "C": 15, "D": 15, "E": 15, "F": 14, "G": 18, "H": 16, "I": 11, "J": 13,
                "K": 14, "L": 12, "M": 14})
    return ws


def build_exceptions(wb, trust, tw, order):
    ws = wb.create_sheet("Exceptions")
    title(ws, f"Exception queue: Trust Index at forecast origin {FORWARD_ORIGIN}",
          "Sorted by priority = (100 - score) x the series' share of the total 14-day forecast. Work from the top.",
          16)
    ws["A3"] = ("Score = SUMPRODUCT(components, frozen weights), clipped 0-100. Band: any hard flag forces Review; "
                "else score >= 80 Accept, >= 60 Monitor, otherwise Review. The score is an index, not a probability.")
    ws["A3"].font = F_NOTE
    comps = ["accuracy", "interval", "stability", "agreement", "completeness"]
    put(ws, (EX_WEIGHT_ROW, 6), "Frozen weight", font=F_BOLD, align=Alignment(horizontal="right"))
    for j, c in enumerate(comps):
        put(ws, (EX_WEIGHT_ROW, 7 + j), tw["weights"][c], "0.00", font=F_BOLD, fill=FILL_TOTAL)
    header(ws, EX_HEADER_ROW, ["Rank", "Series", "Store", "Family", "Trust score", "Band", "Accuracy", "Interval",
                               "Stability", "Agreement", "Completeness", "Hard-flag reasons", "14-day forecast",
                               "Priority", "Planner action", "Planner note"])
    t = trust.sort_values("priority", ascending=False).reset_index(drop=True)
    fc_s = get_column_letter(FC_TOTAL_COL)
    for i, r in t.iterrows():
        row = EX_FIRST + i
        put(ws, (row, 1), f"=RANK(N{row},$N${EX_FIRST}:$N${EX_LAST},0)", "0")
        put(ws, (row, 2), r["series_id"], font=F_BOLD)
        put(ws, (row, 3), int(r["store_nbr"]))
        put(ws, (row, 4), r["family"])
        put(ws, (row, 5), f"=MIN(100,MAX(0,SUMPRODUCT(G{row}:K{row},$G${EX_WEIGHT_ROW}:$K${EX_WEIGHT_ROW})))", DEC1,
            font=F_BOLD)
        put(ws, (row, 6), f'=IF(L{row}<>"","Review",IF(E{row}>=80,"Accept",IF(E{row}>=60,"Monitor","Review")))',
            font=F_BOLD, align=Alignment(horizontal="center"))
        for j, c in enumerate(comps):
            v = r[c]
            put(ws, (row, 7 + j), None if pd.isna(v) else float(v), DEC1)
        flags = list(r["hard_flags"]) if r["hard_flags"] is not None else []
        put(ws, (row, 12), "; ".join(describe_flags(flags)) or None, align=Alignment(wrap_text=True))
        put(ws, (row, 13), f'=SUMIFS(Forecast!${fc_s}${FC_FIRST}:${fc_s}${FC_LAST},Forecast!$A${FC_FIRST}:$A${FC_LAST},'
                           f'B{row},Forecast!$D${FC_FIRST}:$D${FC_LAST},"Forecast")', NUM)
        put(ws, (row, 14), f"=(100-E{row})*M{row}/SUM($M${EX_FIRST}:$M${EX_LAST})", DEC3)
        put(ws, (row, 15), f'=IF(F{row}="Review","Review before release",IF(F{row}="Monitor","Spot-check",'
                           f'"Release as forecast"))')
        as_input(put(ws, (row, 16), None))
    for band, col in BAND_FILL.items():
        ws.conditional_formatting.add(f"F{EX_FIRST}:F{EX_LAST}",
                                      CellIsRule(operator="equal", formula=[f'"{band}"'],
                                                 fill=PatternFill("solid", fgColor=col),
                                                 font=Font(name=FONT, bold=True, color=NAVY)))
    ws.conditional_formatting.add(f"E{EX_FIRST}:E{EX_LAST}",
                                  ColorScaleRule(start_type="num", start_value=0, start_color=BAND_FILL["Review"],
                                                 mid_type="num", mid_value=70, mid_color=BAND_FILL["Monitor"],
                                                 end_type="num", end_value=100, end_color=ACCENT))
    ws.freeze_panes = ws.cell(EX_FIRST, 3)
    ws.auto_filter.ref = f"A{EX_HEADER_ROW}:P{EX_LAST}"
    widths(ws, {"A": 6, "B": 20, "C": 7, "D": 16, "E": 10, "F": 11, "G": 10, "H": 10, "I": 10, "J": 11, "K": 13,
                "L": 48, "M": 14, "N": 10, "O": 22, "P": 30})
    return ws, t


def build_overrides(wb, order):
    ws = wb.create_sheet("Overrides")
    title(ws, "Consensus overrides: statistical forecast vs planner judgement",
          "Type an override quantity (14-day units) in the highlighted column and pick a reason code. "
          "Blank override = keep the statistical forecast.", 12)
    ws["A3"] = ("Statistical forecast links to Forecast!S (14-day total). Final consensus = IF(override blank, "
                "statistical, override). Every override needs a reason code so forecast value added can be measured "
                "later.")
    ws["A3"].font = F_NOTE
    header(ws, OV_HEADER_ROW, ["Series", "Store", "Family", "Trust band", "Statistical 14-day forecast",
                               "Override qty (14 days)", "Reason code", "Planner comment", "Final consensus",
                               "Difference (units)", "Difference %", "Check"])
    dv = DataValidation(type="list", formula1='"' + ",".join(REASON_CODES) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Reason code",
                        error="Pick one of the five reason codes from the list.",
                        promptTitle="Reason code", prompt="Why does judgement differ from the model?",
                        showInputMessage=True)
    dv_qty = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1="0", allow_blank=True,
                            showErrorMessage=True, errorTitle="Override quantity",
                            error="Enter a non-negative 14-day quantity, or leave blank.")
    ws.add_data_validation(dv)
    ws.add_data_validation(dv_qty)
    fc_s = get_column_letter(FC_TOTAL_COL)
    for i, r in order.iterrows():
        row = OV_FIRST + i
        put(ws, (row, 1), r["series_id"], font=F_BOLD)
        put(ws, (row, 2), int(r["store_nbr"]))
        put(ws, (row, 3), r["family"])
        put(ws, (row, 4), f"=INDEX(Exceptions!$F${EX_FIRST}:$F${EX_LAST},MATCH(A{row},Exceptions!$B${EX_FIRST}:"
                          f"$B${EX_LAST},0))", align=Alignment(horizontal="center"))
        put(ws, (row, 5), f"=Forecast!{fc_s}{fc_row(i, 'Forecast')}", NUM)
        as_input(put(ws, (row, 6), None, NUM))
        as_input(put(ws, (row, 7), None))
        as_input(put(ws, (row, 8), None))
        put(ws, (row, 9), f'=IF(F{row}="",E{row},F{row})', NUM, font=F_BOLD)
        put(ws, (row, 10), f"=I{row}-E{row}", NUM)
        put(ws, (row, 11), f"=IF(E{row}=0,0,J{row}/E{row})", PCT)
        put(ws, (row, 12), f'=IF(AND(F{row}<>"",G{row}=""),"Add a reason code",'
                           f'IF(AND(F{row}="",G{row}<>""),"Reason without override",""))')
    dv.add(f"G{OV_FIRST}:G{OV_LAST}")
    dv_qty.add(f"F{OV_FIRST}:F{OV_LAST}")
    t = OV_TOTAL_ROW
    put(ws, (t, 1), "Total", font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    for c in (2, 3, 4, 7, 8, 12):
        put(ws, (t, c), None, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 4), f'=COUNTIF(D{OV_FIRST}:D{OV_LAST},"Review")', '0" Review"', font=F_BOLD, fill=FILL_TOTAL,
        border=B_TOTAL)
    put(ws, (t, 5), f"=SUM(E{OV_FIRST}:E{OV_LAST})", NUM, font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 6), f"=COUNT(F{OV_FIRST}:F{OV_LAST})", '0" overrides"', font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 9), f"=SUM(I{OV_FIRST}:I{OV_LAST})", NUM, font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 10), f"=SUM(J{OV_FIRST}:J{OV_LAST})", NUM, font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 11), f"=IF(E{t}=0,0,J{t}/E{t})", PCT, font=F_BOLD, fill=FILL_TOTAL, border=B_TOTAL)
    put(ws, (t, 12), f'=COUNTIF(L{OV_FIRST}:L{OV_LAST},"?*")', '0" checks open"', font=F_BOLD, fill=FILL_TOTAL,
        border=B_TOTAL)
    for band, col in BAND_FILL.items():
        ws.conditional_formatting.add(f"D{OV_FIRST}:D{OV_LAST}",
                                      CellIsRule(operator="equal", formula=[f'"{band}"'],
                                                 fill=PatternFill("solid", fgColor=col),
                                                 font=Font(name=FONT, bold=True, color=NAVY)))
    ws.conditional_formatting.add(f"L{OV_FIRST}:L{OV_LAST}",
                                  FormulaRule(formula=[f'L{OV_FIRST}<>""'],
                                              font=Font(name=FONT, bold=True, color="B0213F")))
    ws.freeze_panes = ws.cell(OV_FIRST, 2)
    ws.auto_filter.ref = f"A{OV_HEADER_ROW}:L{OV_LAST}"
    widths(ws, {"A": 20, "B": 7, "C": 16, "D": 11, "E": 16, "F": 16, "G": 24, "H": 30, "I": 16, "J": 14, "K": 12,
                "L": 22})
    return ws


def build_scenario(wb, families):
    ws = wb.create_sheet("Scenario")
    title(ws, "Scenario planner: event demand vs supply (mirror of src/scenario.py)", ASSUMPTION_LABEL, 9)
    ws["A3"] = ("Hypothetical planning arithmetic on the assumptions below. Not model output. Change the highlighted "
                "cells; everything else recalculates. Same formulas as the dashboard UAE Scenario Lab.")
    ws["A3"].font = F_NOTE
    fc_s = get_column_letter(FC_TOTAL_COL)
    section(ws, 5, f"Inputs ({ASSUMPTION_LABEL})")
    inputs = [
        ("family", "Family for the baseline", DEFAULT_FAMILY, None,
         "Real Favorita family; the baseline below sums its forecast over the 12 stores."),
        ("baseline", "Baseline units (14 days)",
         f'=SUMIFS(Forecast!${fc_s}${FC_FIRST}:${fc_s}${FC_LAST},Forecast!$C${FC_FIRST}:$C${FC_LAST},{S["family"]},'
         f'Forecast!$D${FC_FIRST}:$D${FC_LAST},"Forecast")', NUM,
         f"Default = real 14-day champion forecast total for the chosen family across the 12 stores "
         f"(Forecast sheet, {DEFAULT_FAMILY} by default). Type a number to overwrite."),
        ("pct_low", "Low scenario change %", SCN_DEFAULTS["pct_low"], PCT,
         "Demonstration default (Ramadan preset in the dashboard)."),
        ("pct_base", "Base scenario change %", SCN_DEFAULTS["pct_base"], PCT, "Demonstration default."),
        ("pct_high", "High scenario change %", SCN_DEFAULTS["pct_high"], PCT, "Demonstration default."),
        ("supply", "Available supply (units)", f"=ROUND({S['baseline']},0)", NUM,
         "Default = baseline rounded (as in the dashboard). Type a number to overwrite."),
        ("lead", "Supplier lead time (days)", SCN_DEFAULTS["lead"], "0", "Demonstration default."),
        ("plan_date", "Planning date (today)", datetime(2017, 8, 15), DATE, "Forecast origin (last actual)."),
        ("event_date", "Event start date", datetime(2017, 9, 12), DATE,
         "Demonstration default: 28 days after the planning date."),
        ("cu", "Cost per unit short (under-stock)", SCN_DEFAULTS["cu"], "#,##0.00",
         "Lost margin / goodwill per unit of unmet demand."),
        ("co", "Cost per unit left over (over-stock)", SCN_DEFAULTS["co"], "#,##0.00",
         "Holding / markdown / waste per unsold unit."),
    ]
    for key, lab, val, fmt, note in inputs:
        ref = S[key]
        put(ws, f"A{ref[1:]}", lab, font=F_BOLD)
        as_input(put(ws, ref, val, fmt, font=F_BOLD))
        ws[f"C{ref[1:]}"] = note
        ws[f"C{ref[1:]}"].font = F_NOTE
    dv = DataValidation(type="list", formula1='"' + ",".join(families) + '"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(S["family"])
    section(ws, 18, "Fixed parameters (as in src/scenario.py, not inputs)")
    for key, lab, val, fmt in [("w_low", "Probability weight: Low", SC.DEFAULT_WEIGHTS["low"], "0%"),
                               ("w_base", "Probability weight: Base", SC.DEFAULT_WEIGHTS["base"], "0%"),
                               ("w_high", "Probability weight: High", SC.DEFAULT_WEIGHTS["high"], "0%"),
                               ("review_window", "Order-now window (days)", SC.REVIEW_WINDOW_DAYS, "0")]:
        put(ws, f"A{S[key][1:]}", lab, font=F_BOLD)
        put(ws, S[key], val, fmt, fill=FILL_TOTAL)
    section(ws, 24, "Cleaned inputs (negatives treated as 0, as in run_scenario)")
    for key, src_key, lab in [("baseline_used", "baseline", "Baseline used"), ("supply_used", "supply", "Supply used"),
                              ("cu_used", "cu", "Under-stock cost used"), ("co_used", "co", "Over-stock cost used")]:
        put(ws, f"A{S[key][1:]}", lab, font=F_BOLD)
        put(ws, S[key], f"=MAX(0,{S[src_key]})", NUM if key in ("baseline_used", "supply_used") else "#,##0.00")
    bu, su, cu, co = (f"${S[k][0]}${S[k][1:]}" for k in ("baseline_used", "supply_used", "cu_used", "co_used"))

    section(ws, 29, "Scenario demand vs available supply")
    header(ws, 30, ["Scenario", "Change %", "Demand (units)", "Shortfall", "Excess", "Coverage (supply / demand)",
                    "Probability weight"])
    labels = {"low": "Low", "base": "Base", "high": "High"}
    pct_ref = {"low": S["pct_low"], "base": S["pct_base"], "high": S["pct_high"]}
    w_ref = {"low": S["w_low"], "base": S["w_base"], "high": S["w_high"]}
    for s, row in SCN_ROW.items():
        put(ws, (row, 1), labels[s], font=F_BOLD)
        put(ws, (row, 2), f"={pct_ref[s]}", PCT)
        put(ws, (row, 3), f"=MAX(0,{bu}*(1+B{row}))", NUM, font=F_BOLD)
        put(ws, (row, 4), f"=MAX(0,C{row}-{su})", NUM)
        put(ws, (row, 5), f"=MAX(0,{su}-C{row})", NUM)
        put(ws, (row, 6), f'=IF(C{row}>0,{su}/C{row},"n/a")', PCT)
        put(ws, (row, 7), f"={w_ref[s]}", "0%")

    section(ws, 35, "Cost matrix: cost(plan, outcome) = under-cost x units short + over-cost x units left over")
    header(ws, 36, ["Plan (order to)", "Plan qty", "If Low happens", "If Base happens", "If High happens",
                    "Expected cost", "Max regret", "Tie-break key (expected)", "Tie-break key (regret)"])
    dem = {s: f"$C${SCN_ROW[s]}" for s in SC.SCENARIOS}
    out_col = {"low": "C", "base": "D", "high": "E"}
    m1, m2 = MAT_ROW["low"], MAT_ROW["high"]
    for p, row in list(MAT_ROW.items()) + [("supply", SUPPLY_ROW)]:
        is_supply = p == "supply"
        put(ws, (row, 1), "Keep current supply" if is_supply else labels[p], font=F_NOTE if is_supply else F_BOLD)
        put(ws, (row, 2), f"={su}" if is_supply else f"={dem[p]}", NUM)
        for o in SC.SCENARIOS:
            put(ws, f"{out_col[o]}{row}", f"={cu}*MAX(0,{dem[o]}-$B{row})+{co}*MAX(0,$B{row}-{dem[o]})", NUM)
        put(ws, (row, 6), f"=${S['w_low'][0]}${S['w_low'][1:]}*C{row}+${S['w_base'][0]}${S['w_base'][1:]}*D{row}+"
                          f"${S['w_high'][0]}${S['w_high'][1:]}*E{row}", NUM, font=F_BOLD)
        if is_supply:
            put(ws, (row, 7), "n/a", align=Alignment(horizontal="right"))
            continue
        put(ws, (row, 7), f"=MAX(C{row}-MIN(C${m1}:C${m2}),D{row}-MIN(D${m1}:D${m2}),E{row}-MIN(E${m1}:E${m2}))",
            NUM)
        put(ws, (row, 8), f"=IF(F{row}<=MIN($F${m1}:$F${m2})+1E-9,G{row},1E+300)", "General", font=F_NOTE)
        put(ws, (row, 9), f"=IF(G{row}<=MIN($G${m1}:$G${m2})+1E-9,F{row},1E+300)", "General", font=F_NOTE)
    ws.conditional_formatting.add(f"A{m1}:F{m2}", FormulaRule(formula=[f'$A{m1}=${S["recommended"][0]}${S["recommended"][1:]}'],
                                                              fill=PatternFill("solid", fgColor=ACCENT),
                                                              font=Font(name=FONT, bold=True, color=NAVY)))
    ws[f"A{SUPPLY_ROW + 1}"] = ("Recommendation = minimum expected cost; ties go to lower max regret, then Low, Base, "
                                "High order (tie-break keys). Minimax regret = minimum max regret.")
    ws[f"A{SUPPLY_ROW + 1}"].font = F_NOTE

    section(ws, 42, "Recommendation")
    pl = f"$A${m1}:$A${m2}"
    rows = [
        ("recommended", "Recommended plan (min expected cost)",
         f"=INDEX({pl},MATCH(MIN($H${m1}:$H${m2}),$H${m1}:$H${m2},0))", None),
        ("rec_qty", "Recommended order-to quantity", f"=INDEX($B${m1}:$B${m2},MATCH({S['recommended']},{pl},0))", NUM),
        ("rec_cost", "Expected cost of that plan", f"=INDEX($F${m1}:$F${m2},MATCH({S['recommended']},{pl},0))", NUM),
        ("minimax", "Minimax-regret plan", f"=INDEX({pl},MATCH(MIN($I${m1}:$I${m2}),$I${m1}:$I${m2},0))", None),
        ("critical", "Critical ratio cu / (cu + co)", f'=IF({cu}+{co}>0,{cu}/({cu}+{co}),"n/a")', PCT),
        ("gap", "Supply gap (plan qty - supply)", f"={S['rec_qty']}-{su}", NUM),
        ("days_to_event", "Days until event start", f"={S['event_date']}-{S['plan_date']}", "0"),
        ("order_by", "Order-by date (event start - lead time)", f"={S['event_date']}-{S['lead']}", DATE),
        ("order_in", "Days until order-by date", f"={S['order_by']}-{S['plan_date']}", "0"),
        ("status", "Timing status", f'=IF({S["order_in"]}<0,"late",IF({S["order_in"]}<={S["review_window"]},"now",'
                                    f'"later"))', None),
        ("timing_msg", "Timing headline", f'=IF({S["status"]}="late","Order-by date already passed",IF({S["status"]}='
                                          f'"now","Decision needed now",CONCATENATE("Order-by in ",'
                                          f'TEXT(INT({S["order_in"]}+0.5),"#,##0")," days")))', None),
    ]
    T = lambda ref: f'TEXT(INT({ref}+0.5),"#,##0")'  # noqa: E731  (fmt_int)
    T100 = lambda ref: f'TEXT(INT({ref}*100+0.5),"#,##0")'  # noqa: E731
    rows += [
        ("part1", "Sentence: plan",
         f'=CONCATENATE("Plan to the ",{S["recommended"]}," scenario (",{T(S["rec_qty"])}," units): lowest expected '
         f'cost (",{T(S["rec_cost"])},") with weights low/base/high = ",{T100(S["w_low"])},"/",{T100(S["w_base"])},'
         f'"/",{T100(S["w_high"])}," % and an under:over cost ratio of ",{T(S["cu_used"])},":",{T(S["co_used"])},".")',
         None),
        ("part_timing", "Sentence: timing (used when supply is short)",
         f'=IF({S["status"]}="late",CONCATENATE("The standard order-by date passed ",{T("-" + S["order_in"])},'
         f'" days ago: expedite, reallocate stock or agree a substitution."),IF({S["status"]}="now",'
         f'CONCATENATE("Order now: the order-by date (event start minus ",{T(S["lead"])}," days lead time) is ",'
         f'{T(S["order_in"])}," days away."),CONCATENATE("Place the replenishment within ",{T(S["order_in"])},'
         f'" days (event start minus lead time).")))', None),
        ("part2", "Sentence: supply",
         f'=IF({S["gap"]}>0.5,CONCATENATE("Available supply is ",{T(S["gap"])}," units short of that plan. ",'
         f'{S["part_timing"]}),IF({S["gap"]}<-0.5,CONCATENATE("Available supply exceeds that plan by ",'
         f'{T("-" + S["gap"])}," units: hold or reallocate the excess rather than reorder."),'
         f'"Available supply matches that plan: no replenishment decision is needed."))', None),
        ("part3", "Sentence: minimax note",
         f'=IF({S["minimax"]}<>{S["recommended"]},CONCATENATE(" A minimax-regret planner would choose the ",'
         f'{S["minimax"]}," scenario instead."),"")', None),
        ("recommendation", "Recommendation", f'=CONCATENATE({S["part1"]}," ",{S["part2"]},{S["part3"]})', None),
    ]
    for key, lab, f, fmt in rows:
        r = S[key][1:]
        put(ws, f"A{r}", lab, font=F_BOLD)
        big = key in ("recommended", "recommendation", "rec_qty", "order_by", "timing_msg")
        c = put(ws, S[key], f, fmt, font=F_BOLD if big else F_BODY,
                align=WRAP if key.startswith("part") or key == "recommendation" else None)
        if key.startswith("part") or key == "recommendation":
            ws.merge_cells(f"B{r}:I{r}")
            ws.row_dimensions[int(r)].height = 30 if key != "recommendation" else 62
        if key == "recommendation":
            c.fill = PatternFill("solid", fgColor="E6FBF6")
            c.border = Border(left=ACC_SIDE)
    for k, (cond, msg) in enumerate([
        (f'NOT(AND({S["pct_low"]}<={S["pct_base"]},{S["pct_base"]}<={S["pct_high"]}))',
         "Low, base and high changes are not in ascending order."),
        (f'{bu}=0', "Baseline is zero, so every scenario demand is zero."),
        (f'AND({cu}=0,{co}=0)', "Both unit costs are zero, so every plan costs nothing."),
    ]):
        r = 60 + k
        put(ws, f"A{r}", "Warning" if k == 0 else None, font=F_BOLD, border=None)
        put(ws, f"B{r}", f'=IF({cond},"{msg}","")', font=Font(name=FONT, bold=True, color="B0213F"), border=None)
    widths(ws, {"A": 40, "B": 16, "C": 16, "D": 16, "E": 16, "F": 16, "G": 16, "H": 14, "I": 14})
    ws.freeze_panes = "A5"
    return ws


def build_summary(wb, res, families):
    ws = wb["Summary"]
    title(ws, "Forecast vs Instinct: demand planning workbook", TAGLINE, 6)
    put(ws, "A4", "What this is", font=F_SECTION, border=None)
    lines = [
        "A planner's working copy of the Forecast vs Instinct pipeline: a 14-day forecast for 96 series "
        "(12 stores x 8 product families), a trust score that says which forecasts need human review, a consensus "
        "override log and an event scenario planner.",
        "Data caveats:",
        "- Real data: Corporacion Favorita grocery chain, Ecuador (Kaggle 'Store Sales - Time Series Forecasting'), "
        "history to 15 Aug 2017.",
        "- Sales are OBSERVED sales, not unconstrained demand: stock-outs and closures cap what was recorded.",
        "- Models were trained and evaluated offline in Python (frozen selection, one-time holdout). This workbook "
        "does not refit anything; forecasts and accuracy are pasted results, all planning arithmetic is live.",
        "- The Scenario sheet uses editable demonstration assumptions, not measured UAE uplift.",
    ]
    for k, t in enumerate(lines):
        c = ws.cell(5 + k, 1, t)
        c.font = F_BOLD if t.endswith(":") else F_BODY
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=5 + k, start_column=1, end_row=5 + k, end_column=6)
        ws.row_dimensions[5 + k].height = 30 if len(t) > 90 else 16
    k0 = 13
    put(ws, f"A{k0 - 1}", "Key numbers (live formulas)", font=F_SECTION, border=None)
    header(ws, k0, ["Measure", "Value", "Source"])
    fc_s = get_column_letter(FC_TOTAL_COL)
    fcr = f"Forecast!${fc_s}${FC_FIRST}:${fc_s}${FC_LAST}"
    fcd = f"Forecast!$D${FC_FIRST}:$D${FC_LAST}"
    exf = f"Exceptions!$F${EX_FIRST}:$F${EX_LAST}"
    exm = f"Exceptions!$M${EX_FIRST}:$M${EX_LAST}"
    d1, d2 = get_column_letter(FC_DATE_COL1), get_column_letter(FC_DATE_COL1 + 13)
    items = [
        ("Series in scope", f"=COUNTA(Overrides!$A${OV_FIRST}:$A${OV_LAST})", "0", "Overrides"),
        ("Forecast window: first day", f"=MIN(Forecast!${d1}${FC_HEADER_ROW}:${d2}${FC_HEADER_ROW})", DATE,
         "Forecast header"),
        ("Forecast window: last day", f"=MAX(Forecast!${d1}${FC_HEADER_ROW}:${d2}${FC_HEADER_ROW})", DATE,
         "Forecast header"),
        ("Total 14-day forecast (units, all series)", f'=SUMIF({fcd},"Forecast",{fcr})', NUM, "Forecast"),
        ("Champion model", f"=Accuracy!$C${ACC['sys_first'] + 5}", None, "Accuracy"),
        ("Holdout WAPE of the champion", f"=Accuracy!$C${ACC['sys_first']}", PCT, "Accuracy"),
        ("Holdout bias of the champion", f"=Accuracy!$C${ACC['sys_first'] + 1}", PCT2, "Accuracy"),
        ("Holdout P10-P90 coverage (target 80%)", f"=Accuracy!$C${ACC['sys_first'] + 2}", PCT, "Accuracy"),
        ("Accept forecasts (release as is)", f'=COUNTIF({exf},"Accept")', "0", "Exceptions"),
        ("Monitor forecasts (spot-check)", f'=COUNTIF({exf},"Monitor")', "0", "Exceptions"),
        ("Review exceptions (human review)", f'=COUNTIF({exf},"Review")', "0", "Exceptions"),
        ("Share of 14-day volume in Review", f'=SUMIF({exf},"Review",{exm})/SUM({exm})', PCT, "Exceptions"),
        ("Holdout WAPE ratio, Review vs Accept windows", f"=Accuracy!{ACC['review_ratio']}", '0.00"x"', "Accuracy"),
        ("Overrides entered", f"=Overrides!$F${OV_TOTAL_ROW}", "0", "Overrides"),
        ("Consensus 14-day total after overrides", f"=Overrides!$I${OV_TOTAL_ROW}", NUM, "Overrides"),
        ("Consensus vs statistical", f"=Overrides!$K${OV_TOTAL_ROW}", PCT, "Overrides"),
        ("Scenario: recommended plan", f'=CONCATENATE(Scenario!{S["recommended"]}," (",Scenario!{S["family"]},")")', None,
         "Scenario"),
        ("Scenario: order-to quantity", f"=Scenario!{S['rec_qty']}", NUM, "Scenario"),
        ("Scenario: order-by date", f"=Scenario!{S['order_by']}", DATE, "Scenario"),
    ]
    SUMMARY_ROWS.clear()
    for k, (lab, f, fmt, src) in enumerate(items):
        r = k0 + 1 + k
        put(ws, (r, 1), lab, font=F_BOLD)
        put(ws, (r, 2), f, fmt, font=F_BOLD, align=Alignment(horizontal="right"))
        put(ws, (r, 3), src, font=F_NOTE)
        SUMMARY_ROWS[lab] = f"B{r}"
    review = SUMMARY_ROWS["Review exceptions (human review)"]
    ws.conditional_formatting.add(review, CellIsRule(operator="greaterThan", formula=["0"],
                                                     fill=PatternFill("solid", fgColor=BAND_FILL["Review"])))
    put(ws, (k0 + len(items) + 2, 1), "Start on the Exceptions sheet: work the Review rows from the top, then record "
                                      "any judgement on Overrides with a reason code.", font=F_NOTE, border=None)
    widths(ws, {"A": 46, "B": 22, "C": 14, "D": 14, "E": 14, "F": 14})
    return ws


SUMMARY_ROWS: dict[str, str] = {}


def build_about(wb, res, sel, intervals, tw, scope):
    ws = wb.create_sheet("About")
    title(ws, "About this workbook", TAGLINE, 4)
    blocks = [
        ("Dataset", [
            "Alexis Cook, DanB, inversion, Ryan Holbrook (2021). Store Sales - Time Series Forecasting. Kaggle. "
            "https://kaggle.com/competitions/store-sales-time-series-forecasting",
            "Corporacion Favorita, Ecuador. Daily unit sales by store and product family, with promotions, holidays "
            "and oil prices. History used to 15 Aug 2017.",
            f"Scope: {len(scope['stores'])} stores x {len(scope['families'])} families = {N_SERIES} series. Stores "
            "chosen as representative by type, cluster and city (config/scope.json), not the biggest.",
        ]),
        ("Methodology", [
            "Models: naive, seasonal naive (7-day), 28-day moving average, Holt-Winters, LightGBM (global, ratio "
            "target). Champion per family = lowest WAPE on 10 selection backtest origins, frozen before the holdout.",
            f"Result: LightGBM variant A won every family. Holdout (6 untouched origins) WAPE "
            f"{res['champion_system']['wape']:.1f}% vs "
            f"{next(x['wape'] for x in res['leaderboard'] if x['model'] == 'seasonal_naive'):.1f}% for seasonal naive.",
            f"Intervals: {intervals['method']}",
            "Trust Index: five 0-100 components (accuracy, interval reliability, stability, model agreement, data "
            "completeness) using only information available at the forecast origin, weighted by frozen weights "
            f"{tw['chosen']} (accuracy/interval/stability/agreement/completeness). Bands Accept >= 80, Monitor >= 60, "
            "Review otherwise; a hard flag (e.g. promotion intensity above the series' historical 95th percentile) "
            "forces Review.",
            "Scenario arithmetic: demand = baseline x (1 + change); cost(plan, outcome) = under-cost x shortfall + "
            "over-cost x excess; recommendation = minimum expected cost with weights 25/50/25 %, ties to lower max "
            "regret; order-by date = event start - lead time. Identical to src/scenario.py and the dashboard.",
        ]),
        ("Sheet guide", [
            "Summary: caveats and key numbers (formulas pulling from the other sheets).",
            "Forecast: 14-day daily forecast with P10/P90 per series; 14-day totals are SUM formulas.",
            "Accuracy: holdout WAPE / bias by model and family; best-model columns are INDEX/MATCH formulas.",
            "Exceptions: Trust Index queue sorted by priority; score, band and priority are live formulas.",
            "Overrides: enter a 14-day override and a reason code; final consensus = IF(blank, statistical, override).",
            "Scenario: event what-if planner. Highlighted cells are inputs; everything else recalculates.",
        ]),
        ("Colour key", [
            "Accept = #60F0D8 (release as forecast). Monitor = #F5C26B (spot-check). Review = #FF6B8A (review "
            "before release). Light mint cells with a teal border are inputs.",
        ]),
        ("Reproduce", [
            "Generated by scripts/09_build_workbook.py from frozen pipeline outputs (.venv/bin/python "
            "scripts/09_build_workbook.py). Checked by tests/test_workbook.py. Formulas calculate when opened in "
            "Excel; openpyxl stores no cached values.",
            f"Built {date.today():%d %b %Y}.",
        ]),
    ]
    r = 4
    for head, items in blocks:
        put(ws, (r, 1), head, font=F_SECTION, border=None)
        r += 1
        for t in items:
            c = ws.cell(r, 1, t)
            c.font, c.alignment = F_BODY, Alignment(wrap_text=True, vertical="top")
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
            ws.row_dimensions[r].height = 15 * max(1, -(-len(t) // 120))
            r += 1
        r += 1
    widths(ws, {"A": 40, "B": 40, "C": 40, "D": 20})
    return ws


def build(out: Path = OUT) -> Path:
    fwd, hold, trust, res, sel, intervals, tw, scope = load()
    order = series_order(fwd)
    families = sorted(order["family"].unique())
    wb = Workbook()
    wb.active.title = "Summary"
    build_forecast(wb, fwd, order, intervals)
    build_accuracy(wb, res, sel, hold)
    build_exceptions(wb, trust, tw, order)
    build_overrides(wb, order)
    build_scenario(wb, families)
    build_summary(wb, res, families)
    build_about(wb, res, sel, intervals, tw, scope)
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.tabColor = ACCENT if ws.title in ("Summary", "Exceptions", "Overrides") else NAVY2
    wb.properties.title = "Forecast vs Instinct planning workbook"
    wb.properties.creator = "Forecast vs Instinct pipeline"
    wb.calculation.fullCalcOnLoad = True
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(ROOT)}")
