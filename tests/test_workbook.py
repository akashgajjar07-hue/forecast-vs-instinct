"""Planner workbook (excel/forecast_vs_instinct_planning_workbook.xlsx, built by scripts/09_build_workbook.py).

openpyxl stores formulas without cached values and LibreOffice is not assumed, so these tests carry a small
evaluator for the Excel subset the workbook uses (SUM, SUMIF(S), COUNTIF, INDEX/MATCH, IF, TEXT, ...). It
evaluates the workbook's own formula strings, so the checks below test what Excel will compute:
  - Forecast 14-day mean index = parquet sum / 14 / base x 100 (base recomputed here from the panel); portfolio and
    family rows = equal-weighted means of the series index (family in index-days); the all-series 14-day total
    (units) = parquet sum; Exceptions score/band/priority rank = trust_scores.parquet (data tests)
  - the only unit values stored anywhere are the all-series 14-day total and the pooled leaderboard MAE/RMSE
  - Scenario sheet = src/scenario.run_scenario on the default inputs and on every shared JS/Python case
"""
import importlib.util
import json
import math
import re
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string, get_column_letter

from src.scenario import SCENARIOS, run_scenario

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "excel" / "forecast_vs_instinct_planning_workbook.xlsx"
CASES = json.loads((ROOT / "tests" / "js" / "scenario_cases.json").read_text())

_spec = importlib.util.spec_from_file_location("build_workbook", ROOT / "scripts" / "09_build_workbook.py")
BW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BW)

SHEETS = ["Summary", "Forecast", "Accuracy", "Exceptions", "Overrides", "Scenario", "About"]
EPOCH = datetime(1899, 12, 30)

pytestmark = pytest.mark.skipif(not XLSX.exists(), reason="workbook not built (scripts/09_build_workbook.py)")


# ---------------------------------------------------------------------------------------------
# minimal Excel formula evaluator
# ---------------------------------------------------------------------------------------------
class Blank(int):
    """An empty cell: 0 in arithmetic, equal to "" and 0 in comparisons."""

    def __new__(cls):
        return super().__new__(cls, 0)

    def __eq__(self, other):
        return other == "" or (isinstance(other, (int, float)) and other == 0) or isinstance(other, Blank)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = int.__hash__


BLANK = Blank()
TOKEN = re.compile(r"""
    (?P<str>"(?:[^"]|"")*")
  | (?P<func>[A-Z][A-Z0-9.]*)\(
  | (?P<ref>(?:(?P<sheet>[A-Za-z_]\w*|'[^']+')!)?\$?(?P<c1>[A-Z]{1,3})\$?(?P<r1>\d+)(?::\$?(?P<c2>[A-Z]{1,3})\$?(?P<r2>\d+))?)
  | (?P<num>\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)
  | (?P<bool>TRUE|FALSE)
  | (?P<op><=|>=|<>|[=<>+\-*/(),])
  | (?P<ws>\s+)
""", re.X)
LAZY = {"IF"}


def _flat(args):
    for a in args:
        if isinstance(a, list):
            yield from (x for x in a if not isinstance(x, (Blank, str)))
        else:
            yield a


def _eq(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def _crit(v, c):
    if c == "?*":
        return isinstance(v, str) and len(v) > 0
    return _eq(v, c)


def _round(x, n=0):
    q = Decimal(1).scaleb(-int(n))
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def _text(x, fmt):
    assert fmt == "#,##0", fmt
    n = int(_round(x))
    return ("-" if n < 0 else "") + f"{abs(n):,}"


FUNCS = {
    "SUM": lambda *a: sum(_flat(a)),
    "AVERAGE": lambda *a: (lambda v: sum(v) / len(v))(list(_flat(a))),
    "MIN": lambda *a: min(_flat(a)),
    "MAX": lambda *a: max(_flat(a)),
    "IF": lambda c, a, b=lambda: False: a() if c() else b(),
    "AND": lambda *a: all(_flat(a)),
    "OR": lambda *a: any(_flat(a)),
    "NOT": lambda a: not a,
    "INT": lambda x: math.floor(x),
    "ABS": abs,
    "ROUND": _round,
    "TEXT": _text,
    "CONCATENATE": lambda *a: "".join(str(x) for x in a),
    "INDEX": lambda rng, n: rng[int(n) - 1],
    "MATCH": lambda v, rng, t=1: next(i + 1 for i, x in enumerate(rng) if _eq(x, v)),
    "SUMPRODUCT": lambda a, b: sum(float(x) * float(y) for x, y in zip(a, b)),
    "SUMIF": lambda rng, c, s: sum(y for x, y in zip(rng, s) if _crit(x, c)),
    "AVERAGEIF": lambda rng, c, s: (lambda v: sum(v) / len(v))([y for x, y in zip(rng, s) if _crit(x, c)]),
    "AVERAGEIFS": lambda s, *cr: (lambda v: sum(v) / len(v))(
        [y for i, y in enumerate(s) if all(_crit(cr[k][i], cr[k + 1]) for k in range(0, len(cr), 2))]),
    "SUMIFS": lambda s, *cr: sum(y for i, y in enumerate(s)
                                 if all(_crit(cr[k][i], cr[k + 1]) for k in range(0, len(cr), 2))),
    "COUNTIF": lambda rng, c: sum(1 for x in rng if _crit(x, c)),
    "COUNTA": lambda rng: sum(1 for x in rng if not isinstance(x, Blank)),
    "COUNT": lambda rng: sum(1 for x in rng if isinstance(x, (int, float)) and not isinstance(x, (Blank, bool))),
    "RANK": lambda v, rng, order=0: 1 + sum(1 for x in rng if (x < v if order else x > v)),
}


class Evaluator:
    def __init__(self, wb):
        self.wb, self.cache, self.overrides = wb, {}, {}

    def set(self, sheet, ref, value):
        self.overrides[(sheet, ref)] = value
        self.cache.clear()

    def raw(self, sheet, ref):
        if (sheet, ref) in self.overrides:
            return self.overrides[(sheet, ref)]
        return self.wb[sheet][ref].value

    def value(self, sheet, ref):
        key = (sheet, ref)
        if key not in self.cache:
            v = self.raw(sheet, ref)
            if v is None:
                v = BLANK
            elif isinstance(v, datetime):
                v = (v - EPOCH) / timedelta(days=1)
            elif isinstance(v, str) and v.startswith("="):
                v = self.formula(sheet, v[1:])
            self.cache[key] = v
        return self.cache[key]

    def range(self, sheet, c1, r1, c2, r2):
        cols = range(column_index_from_string(c1), column_index_from_string(c2) + 1)
        return [self.value(sheet, f"{get_column_letter(c)}{r}") for r in range(int(r1), int(r2) + 1) for c in cols]

    def translate(self, sheet, f):
        out, stack, pos = [], [], 0
        while pos < len(f):
            m = TOKEN.match(f, pos)
            assert m, f"cannot parse {f[pos:]!r} in {f!r}"
            pos = m.end()
            if m["ws"]:
                continue
            if m["str"]:
                out.append(repr(m["str"][1:-1].replace('""', '"')))
            elif m["func"]:
                name = m["func"]
                assert name in FUNCS, f"unsupported function {name}"
                stack.append(name)
                out.append(f"F[{name!r}](" + ("lambda: " if name in LAZY else ""))
            elif m["ref"]:
                sh = (m["sheet"] or sheet).strip("'")
                if m["c2"]:
                    out.append(f"E.range({sh!r},{m['c1']!r},{m['r1']},{m['c2']!r},{m['r2']})")
                else:
                    out.append(f"E.value({sh!r},'{m['c1']}{m['r1']}')")
            elif m["num"]:
                out.append(m["num"])
            elif m["bool"]:
                out.append("True" if m["bool"] == "TRUE" else "False")
            else:
                op = m["op"]
                if op == "(":
                    stack.append("(")
                    out.append("(")
                elif op == ")":
                    stack.pop()
                    out.append(")")
                elif op == ",":
                    out.append(", lambda: " if stack and stack[-1] in LAZY else ",")
                else:
                    out.append({"=": "==", "<>": "!="}.get(op, op))
        return "".join(out)

    def formula(self, sheet, f):
        return eval(self.translate(sheet, f), {"F": FUNCS, "E": self})  # noqa: S307 (own workbook formulas)


@pytest.fixture(scope="module")
def wb():
    return load_workbook(XLSX)


@pytest.fixture
def ev(wb):
    return Evaluator(wb)


def is_formula(cell):
    return isinstance(cell.value, str) and cell.value.startswith("=")


# ---------------------------------------------------------------------------------------------
# structure (runs wherever the workbook exists)
# ---------------------------------------------------------------------------------------------
def test_opens_with_required_sheets(wb):
    assert wb.sheetnames == SHEETS


def test_key_cells_are_formulas(wb):
    fc, S = wb["Forecast"], BW.S
    tot = get_column_letter(BW.FC_TOTAL_COL)
    for r in range(BW.FC_FIRST, BW.FC_LAST + 1):
        assert fc[f"{tot}{r}"].value == f"=AVERAGE(E{r}:R{r})"
        for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL):
            assert isinstance(fc.cell(r, c).value, float)
    for r in range(BW.FC_PORT_ROW, BW.FC_PORT_ROW + 3):
        assert fc[f"{tot}{r}"].value == f"=AVERAGE(E{r}:R{r})"
        assert all(fc.cell(r, c).value.startswith("=AVERAGEIF(") for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL))
    for r in range(BW.FC_FAM_FIRST, BW.FC_FAM_LAST + 1):
        assert fc[f"{tot}{r}"].value == f"=SUM(E{r}:R{r})"
        assert all(fc.cell(r, c).value.startswith("=AVERAGEIFS(") for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL))
    # the single unit value: one stored all-series 14-day total, no per-day chain values
    assert isinstance(fc[f"{tot}{BW.FC_TOTAL_ROW}"].value, float)
    assert all(fc.cell(BW.FC_TOTAL_ROW, c).value is None or isinstance(fc.cell(BW.FC_TOTAL_ROW, c).value, str)
               for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL))
    summary = wb["Summary"]
    vals = [summary[f"B{r}"] for r in range(14, 33) if summary[f"A{r}"].value]
    assert len(vals) >= 15 and all(is_formula(c) for c in vals)
    ex = wb["Exceptions"]
    for r in range(BW.EX_FIRST, BW.EX_LAST + 1):
        assert all(is_formula(ex[f"{c}{r}"]) for c in "AEFMO")
        assert ex[f"N{r}"].value == r - BW.EX_FIRST + 1  # stored priority rank, never the priority value
    ov = wb["Overrides"]
    for r in range(BW.OV_FIRST, BW.OV_LAST + 1):
        assert all(is_formula(ov[f"{c}{r}"]) for c in "DEIJKL")
        assert ov[f"I{r}"].value == f'=IF(F{r}="",E{r},F{r})'
        assert ov[f"F{r}"].value is None and ov[f"F{r}"].fill.fgColor.rgb.endswith(BW.INPUT_FILL)
    assert all(is_formula(ov[f"{c}{BW.OV_TOTAL_ROW}"]) for c in "EFIJK")
    acc = wb["Accuracy"]
    for r in range(BW.ACC["fam_first"], BW.ACC["fam_last"] + 1):
        assert acc[f"H{r}"].value.startswith("=INDEX(") and "MATCH(MIN(" in acc[f"H{r}"].value
        assert acc[f"I{r}"].value == f"=MIN(B{r}:F{r})"
    sc = wb["Scenario"]
    calc = [k for k in S if k not in ("family", "pct_low", "pct_base", "pct_high", "lead", "plan_date",
                                      "event_date", "cu", "co", "w_low", "w_base", "w_high", "review_window")]
    assert all(is_formula(sc[S[k]]) for k in calc)
    for row in list(BW.SCN_ROW.values()) + list(BW.MAT_ROW.values()):
        assert all(is_formula(sc[f"{c}{row}"]) for c in "BCDEF")


def test_scenario_formula_references(wb):
    sc, S = wb["Scenario"], BW.S
    assert sc[S["baseline"]].value == (f"=SUMIFS(Forecast!$S${BW.FC_FAM_FIRST}:$S${BW.FC_FAM_LAST},"
                                       f"Forecast!$C${BW.FC_FAM_FIRST}:$C${BW.FC_FAM_LAST},B6)")
    assert sc[S["family"]].value == BW.DEFAULT_FAMILY
    assert sc["C31"].value == "=MAX(0,$B$25*(1+B31))"
    assert sc["C37"].value == "=$B$27*MAX(0,$C$31-$B37)+$B$28*MAX(0,$B37-$C$31)"
    assert sc["E37"].value == "=$B$27*MAX(0,$C$33-$B37)+$B$28*MAX(0,$B37-$C$33)"
    assert sc["F38"].value == "=$B$19*C38+$B$20*D38+$B$21*E38"
    assert sc[S["recommended"]].value == "=INDEX($A$37:$A$39,MATCH(MIN($H$37:$H$39),$H$37:$H$39,0))"
    assert sc[S["order_by"]].value == "=B14-B12"
    assert sc["A2"].value == "Editable demonstration assumptions, not measured UAE uplift"


def test_overrides_data_validation(wb):
    ov = wb["Overrides"]
    lists = [dv for dv in ov.data_validations.dataValidation if dv.type == "list"]
    assert len(lists) == 1
    dv = lists[0]
    assert str(dv.sqref) == f"G{BW.OV_FIRST}:G{BW.OV_LAST}"
    assert dv.formula1.strip('"').split(",") == BW.REASON_CODES
    assert any(dv.type == "decimal" and str(dv.sqref) == f"F{BW.OV_FIRST}:F{BW.OV_LAST}"
               for dv in ov.data_validations.dataValidation)


def test_band_conditional_formatting(wb):
    rules = [(str(rng.sqref), r) for rng in wb["Exceptions"].conditional_formatting for r in rng.rules]
    fills = {r.formula[0].strip('"'): r.dxf.fill.fgColor.rgb[-6:] for s, r in rules
             if s == f"F{BW.EX_FIRST}:F{BW.EX_LAST}" and r.type == "cellIs"}
    assert fills == {"Accept": "60F0D8", "Monitor": "F5C26B", "Review": "FF6B8A"}


def test_overrides_consensus_logic(ev):
    r = BW.OV_FIRST
    stat = ev.value("Overrides", f"E{r}")
    assert ev.value("Overrides", f"I{r}") == stat and ev.value("Overrides", f"L{r}") == ""
    ev.set("Overrides", f"F{r}", 150.0)
    assert ev.value("Overrides", f"I{r}") == 150.0
    assert ev.value("Overrides", f"K{r}") == pytest.approx(150 / stat - 1)
    assert ev.value("Overrides", f"L{r}") == "Add a reason code"
    assert ev.value("Overrides", f"F{BW.OV_TOTAL_ROW}") == 1
    mean_stat = ev.value("Overrides", f"E{BW.OV_TOTAL_ROW}")  # mean index over the 96 series
    assert ev.value("Overrides", f"I{BW.OV_TOTAL_ROW}") == pytest.approx(mean_stat + (150 - stat) / BW.N_SERIES)
    ev.set("Overrides", f"G{r}", "Upcoming promotion")
    assert ev.value("Overrides", f"L{r}") == ""


# ---------------------------------------------------------------------------------------------
# scenario sheet == src/scenario.py
# ---------------------------------------------------------------------------------------------
def excel_scenario(ev):
    v = lambda k: ev.value("Scenario", BW.S[k])  # noqa: E731
    sc = lambda ref: ev.value("Scenario", ref)  # noqa: E731
    out = {
        "demand": {s: sc(f"C{BW.SCN_ROW[s]}") for s in SCENARIOS},
        "position": {s: {"shortfall": sc(f"D{BW.SCN_ROW[s]}"), "excess": sc(f"E{BW.SCN_ROW[s]}"),
                         "coverage_pct": (None if sc(f"F{BW.SCN_ROW[s]}") == "n/a" else 100 * sc(f"F{BW.SCN_ROW[s]}"))}
                     for s in SCENARIOS},
        "cost_matrix": {p: {o: sc(f"{c}{BW.MAT_ROW[p]}") for o, c in zip(SCENARIOS, "CDE")} for p in SCENARIOS},
        "expected_cost": {p: sc(f"F{BW.MAT_ROW[p]}") for p in SCENARIOS},
        "max_regret": {p: sc(f"G{BW.MAT_ROW[p]}") for p in SCENARIOS},
        "supply_cost": {o: sc(f"{c}{BW.SUPPLY_ROW}") for o, c in zip(SCENARIOS, "CDE")},
        "supply_expected_cost": sc(f"F{BW.SUPPLY_ROW}"),
        "recommended_plan": v("recommended").lower(),
        "minimax_regret_plan": v("minimax").lower(),
        "critical_ratio": None if v("critical") == "n/a" else v("critical"),
        "order": {"order_in_days": v("order_in"), "status": v("status")},
        "supply_gap": v("gap"),
        "recommendation": v("recommendation"),
        "warnings": [w for w in (sc(f"B{r}") for r in (60, 61, 62)) if w],
    }
    inputs = {"baseline": v("baseline"), "pct_low": 100 * v("pct_low"), "pct_base": 100 * v("pct_base"),
              "pct_high": 100 * v("pct_high"), "supply": v("supply"), "cost_under": v("cu"), "cost_over": v("co"),
              "lead_time_days": v("lead"), "days_to_event": v("event_date") - v("plan_date"),
              "weights": {"low": v("w_low"), "base": v("w_base"), "high": v("w_high")}}
    return out, inputs


def assert_same(xl, py):
    for k, val in py.items():
        if isinstance(val, dict):
            assert_same(xl[k], val)
        elif isinstance(val, float):
            assert xl[k] == pytest.approx(val, rel=1e-9, abs=1e-6), k
        else:
            assert xl[k] == val, k


def compare(ev):
    xl, inputs = excel_scenario(ev)
    py = run_scenario(inputs)
    assert_same(xl, py)
    return xl, py, inputs


def test_scenario_defaults_match_python(ev):
    xl, py, inputs = compare(ev)
    assert inputs["pct_low"] == pytest.approx(5) and inputs["pct_high"] == pytest.approx(30)
    assert inputs["days_to_event"] == 28 and inputs["lead_time_days"] == 14
    assert inputs["supply"] == round(inputs["baseline"])
    assert py["recommended_plan"] == "high" and xl["recommendation"] == py["recommendation"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_scenario_cases_match_python(ev, case):
    S = BW.S
    put = lambda k, val: ev.set("Scenario", S[k], val)  # noqa: E731
    put("baseline", case["baseline"])
    for k in ("pct_low", "pct_base", "pct_high"):
        put(k, case[k] / 100)
    put("supply", case["supply"])
    put("cu", case["cost_under"])
    put("co", case["cost_over"])
    put("lead", case["lead_time_days"])
    put("event_date", datetime(2017, 8, 15) + timedelta(days=case["days_to_event"]))
    w = case.get("weights", {"low": 0.25, "base": 0.5, "high": 0.25})
    for s in SCENARIOS:
        put(f"w_{s}", w[s])
    xl, inputs = excel_scenario(ev)
    py = run_scenario({**case, "weights": w})
    assert_same(xl, py)


# ---------------------------------------------------------------------------------------------
# data-dependent: workbook values vs the pipeline outputs
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def outputs():
    pd = pytest.importorskip("pandas")
    p = ROOT / "outputs" / "holdout"
    panel_path = ROOT / "data" / "processed" / "panel.parquet"
    if not (p / "champion_intervals.parquet").exists() or not panel_path.exists():
        pytest.skip("pipeline outputs or panel not present")
    ci = pd.read_parquet(p / "champion_intervals.parquet")
    fwd = ci[ci["stage"] == "forward"]
    tr = pd.read_parquet(p / "trust_scores.parquet")
    tr = tr[tr["origin"] == pd.Timestamp(BW.FORWARD_ORIGIN)]
    # index base recomputed independently: mean open-day sales 2015-06-01..2017-02-06 (day before 1st selection origin)
    panel = pd.read_parquet(panel_path)
    first_sel = json.loads((ROOT / "config" / "split_calendar.json").read_text())["stages"]["selection"][0]
    w = panel[(panel["date"] >= "2015-06-01") & (panel["date"] < first_sel) & (panel["closed"] == 0)]
    base = w.dropna(subset=["sales"]).groupby("series_id")["sales"].mean()
    return fwd, tr, base


@pytest.mark.data
def test_forecast_index_and_totals_equal_parquet(wb, ev, outputs):
    fwd, _, base = outputs
    fc = wb["Forecast"]
    tot = get_column_letter(BW.FC_TOTAL_COL)
    sums = fwd.groupby("series_id")[["forecast", "p10", "p90"]].sum()
    seen = set()
    for r in range(BW.FC_FIRST, BW.FC_LAST + 1):
        sid, m = fc[f"A{r}"].value, fc[f"D{r}"].value
        seen.add(sid)
        col = {"Forecast": "forecast", "P10": "p10", "P90": "p90"}[m]
        exp = sums.loc[sid, col] / 14 / base[sid] * 100
        assert ev.value("Forecast", f"{tot}{r}") == pytest.approx(exp, abs=1e-4)
    assert seen == set(fwd["series_id"])
    assert ev.value("Forecast", f"{tot}{BW.FC_TOTAL_ROW}") == pytest.approx(fwd["forecast"].sum(), rel=1e-12)
    idx = sums.div(base.loc[sums.index], axis=0) * 100 / 14  # per-series 14-day mean index
    for k, col in enumerate(["forecast", "p10", "p90"]):
        assert ev.value("Forecast", f"{tot}{BW.FC_PORT_ROW + k}") == pytest.approx(idx[col].mean(), abs=1e-4)
    fam = fwd[["series_id", "family"]].drop_duplicates().set_index("series_id")["family"]
    idays = (idx["forecast"] * 14).groupby(fam).mean()  # index-days: 14 x equal-weighted family mean
    for r in range(BW.FC_FAM_FIRST, BW.FC_FAM_LAST + 1):
        assert ev.value("Forecast", f"{tot}{r}") == pytest.approx(idays[fc[f"C{r}"].value], abs=1e-3)
    # header dates are the 14 forward target days
    days = [fc.cell(BW.FC_HEADER_ROW, c).value for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL)]
    assert [d.date() for d in days] == sorted({d.date() for d in fwd["target_date"]})


@pytest.mark.data
def test_no_unit_values_except_portfolio(wb, outputs):
    """Only two kinds of unit values may be stored: the all-series 14-day forecast total and the leaderboard MAE/RMSE
    pooled over all 96 series. No stored number equals a per-series, per-family or per-day unit value (daily or 14-day
    forecast/P10/P90 by series, family or day, chain daily totals, trust forecast_14d or priority)."""
    fwd, tr, _ = outputs
    forbidden = set()
    for c in ["forecast", "p10", "p90"]:
        forbidden |= {round(float(v), 6) for v in fwd[c]}
        for keys in (["series_id"], ["family"], ["target_date"], ["family", "target_date"]):
            forbidden |= {round(float(v), 6) for v in fwd.groupby(keys)[c].sum()}
        if c != "forecast":  # chain P10 / P90 totals are not published either
            forbidden.add(round(float(fwd[c].sum()), 6))
    forbidden |= {round(float(v), 6) for v in tr["forecast_14d"]}
    forbidden |= {round(float(v), 4) for v in tr["priority"]}
    total = float(fwd["forecast"].sum())
    found_total = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                if isinstance(v, float) and abs(v) > 1:
                    if abs(v - total) < 1e-6:
                        found_total += 1
                        continue
                    assert round(v, 6) not in forbidden and round(v, 4) not in forbidden, (ws.title, v)
    assert found_total == 1
    # the only unit-labelled stored numbers outside the Forecast total are the leaderboard MAE / RMSE columns
    acc = wb["Accuracy"]
    assert "all 96 series" in acc["E4"].value and "all 96 series" in acc["F4"].value


@pytest.mark.data
def test_summary_and_scenario_baseline(ev, outputs):
    fwd, tr, base = outputs
    rows = {ev.wb["Summary"][f"A{r}"].value: f"B{r}" for r in range(14, 40) if ev.wb["Summary"][f"A{r}"].value}
    val = lambda lab: ev.value("Summary", rows[lab])  # noqa: E731
    assert val("Total 14-day forecast (units, all series)") == pytest.approx(fwd["forecast"].sum())
    assert val("Review exceptions (human review)") == (tr["band"] == "Review").sum()
    assert val("Accept forecasts (release as is)") == (tr["band"] == "Accept").sum()
    assert val("Series in scope") == 96
    res = json.loads((ROOT / "outputs" / "holdout_results.json").read_text())
    assert val("Holdout WAPE of the champion") == pytest.approx(res["champion_system"]["wape"] / 100)
    idx14 = fwd.groupby("series_id")["forecast"].sum() / 14 / base.loc[fwd["series_id"].unique()] * 100
    band = tr.set_index("series_id")["band"]
    rev = idx14[band.loc[idx14.index] == "Review"].sum() / idx14.sum()  # equal weight per series, not volume
    assert val("Review share of the 14-day forecast index (equal-weighted)") == pytest.approx(rev, abs=1e-6)
    mean_idx = ev.value("Overrides", f"E{BW.OV_TOTAL_ROW}")
    assert val("Consensus mean 14-day index after overrides") == pytest.approx(mean_idx)
    assert val("Consensus vs statistical (mean index)") == pytest.approx(0)
    f = fwd[fwd["family"] == BW.DEFAULT_FAMILY]
    s14 = f.groupby("series_id")["forecast"].sum()
    idays = float((s14 / base.loc[s14.index] * 100).mean())  # 14 x mean of 14-day mean index = mean 14-day index sum
    assert ev.value("Scenario", BW.S["baseline"]) == pytest.approx(idays, abs=1e-3)
    acc = {ev.wb["Accuracy"][f"A{r}"].value: r for r in range(BW.ACC["sys_first"], BW.ACC["champion_row"] + 1)}
    h = pytest.importorskip("pandas").read_parquet(ROOT / "outputs/holdout/champion_intervals.parquet")
    h = h[(h["stage"] == "holdout") & (h["closed"] == 0) & h["actual"].notna()]
    w = ((h["p90"] - h["p10"]) / h["series_id"].map(base) * 100).mean()
    assert ev.value("Accuracy", f"C{acc['Mean P10-P90 width (index points per series-day)']}") == pytest.approx(w)


@pytest.mark.data
def test_exceptions_match_trust_scores(ev, outputs):
    fwd, tr, base = outputs
    t = tr.set_index("series_id")
    order = list(tr.sort_values("priority", ascending=False)["series_id"])
    s14 = fwd.groupby("series_id")["forecast"].sum()
    for r in range(BW.EX_FIRST, BW.EX_LAST + 1):
        sid = ev.value("Exceptions", f"B{r}")
        k = r - BW.EX_FIRST
        assert sid == order[k]
        assert ev.value("Exceptions", f"E{r}") == pytest.approx(t.loc[sid, "score"], abs=1e-9)
        assert ev.value("Exceptions", f"F{r}") == t.loc[sid, "band"]
        assert ev.value("Exceptions", f"M{r}") == pytest.approx(s14[sid] / 14 / base[sid] * 100, abs=1e-4)
        assert ev.value("Exceptions", f"N{r}") == k + 1 and ev.value("Exceptions", f"A{r}") == k + 1
        assert bool(ev.value("Exceptions", f"L{r}")) == (len(t.loc[sid, "hard_flags"]) > 0)


@pytest.mark.data
def test_accuracy_best_model_formula(ev):
    res = json.loads((ROOT / "outputs" / "holdout_results.json").read_text())
    names = dict(BW.MODEL_COLS)
    for k, r in enumerate(res["by_family"]):
        row = BW.ACC["fam_first"] + k
        assert ev.value("Accuracy", f"A{row}") == r["family"]
        assert ev.value("Accuracy", f"H{row}") == names[r["holdout_best"]]
        assert ev.value("Accuracy", f"I{row}") == pytest.approx(min(r[m] for m in names) / 100)


@pytest.mark.data
def test_build_is_reproducible(tmp_path):
    p = BW.build(tmp_path / "wb.xlsx")
    fresh, wb = load_workbook(p), load_workbook(XLSX)  # fresh copy: the shared fixture grows when cells are read
    assert fresh.sheetnames == wb.sheetnames
    for name in SHEETS:
        if name == "About":  # carries the build date
            continue
        a, b = fresh[name], wb[name]
        assert a.max_row == b.max_row and a.max_column == b.max_column
        for ra, rb in zip(a.iter_rows(values_only=True), b.iter_rows(values_only=True)):
            assert ra == rb
