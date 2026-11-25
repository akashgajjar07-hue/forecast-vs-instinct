"""Planner workbook (excel/forecast_vs_instinct_planning_workbook.xlsx, built by scripts/09_build_workbook.py).

openpyxl stores formulas without cached values and LibreOffice is not assumed, so these tests carry a small
evaluator for the Excel subset the workbook uses (SUM, SUMIF(S), COUNTIF, INDEX/MATCH, IF, TEXT, ...). It
evaluates the workbook's own formula strings, so the checks below test what Excel will compute:
  - Forecast totals = parquet sums; Exceptions score/band/priority = trust_scores.parquet (data tests)
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
        assert fc[f"{tot}{r}"].value == f"=SUM(E{r}:R{r})"
        for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL):
            assert isinstance(fc.cell(r, c).value, float)
    for r in range(BW.FC_TOTAL_ROW, BW.FC_TOTAL_ROW + 3):
        assert all(is_formula(fc.cell(r, c)) for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL + 1))
    summary = wb["Summary"]
    vals = [summary[f"B{r}"] for r in range(14, 33) if summary[f"A{r}"].value]
    assert len(vals) >= 15 and all(is_formula(c) for c in vals)
    ex = wb["Exceptions"]
    for r in range(BW.EX_FIRST, BW.EX_LAST + 1):
        assert all(is_formula(ex[f"{c}{r}"]) for c in "AEFMNO")
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
    assert sc[S["baseline"]].value.startswith("=SUMIFS(Forecast!$S$5:$S$292,Forecast!$C$5:$C$292,B6,")
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
    ev.set("Overrides", f"F{r}", 1000.0)
    assert ev.value("Overrides", f"I{r}") == 1000.0
    assert ev.value("Overrides", f"K{r}") == pytest.approx(1000 / stat - 1)
    assert ev.value("Overrides", f"L{r}") == "Add a reason code"
    assert ev.value("Overrides", f"F{BW.OV_TOTAL_ROW}") == 1
    total_stat = ev.value("Overrides", f"E{BW.OV_TOTAL_ROW}")
    assert ev.value("Overrides", f"I{BW.OV_TOTAL_ROW}") == pytest.approx(total_stat - stat + 1000)
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
    if not (p / "champion_intervals.parquet").exists():
        pytest.skip("pipeline outputs not present")
    ci = pd.read_parquet(p / "champion_intervals.parquet")
    fwd = ci[ci["stage"] == "forward"]
    tr = pd.read_parquet(p / "trust_scores.parquet")
    tr = tr[tr["origin"] == pd.Timestamp(BW.FORWARD_ORIGIN)]
    return fwd, tr


@pytest.mark.data
def test_forecast_totals_equal_parquet(wb, ev, outputs):
    fwd, _ = outputs
    fc = wb["Forecast"]
    tot = get_column_letter(BW.FC_TOTAL_COL)
    sums = fwd.groupby("series_id")[["forecast", "p10", "p90"]].sum()
    seen = set()
    for r in range(BW.FC_FIRST, BW.FC_LAST + 1):
        sid, m = fc[f"A{r}"].value, fc[f"D{r}"].value
        seen.add(sid)
        col = {"Forecast": "forecast", "P10": "p10", "P90": "p90"}[m]
        assert ev.value("Forecast", f"{tot}{r}") == pytest.approx(sums.loc[sid, col], rel=1e-12)
    assert seen == set(fwd["series_id"])
    for k, col in enumerate(["forecast", "p10", "p90"]):
        assert ev.value("Forecast", f"{tot}{BW.FC_TOTAL_ROW + k}") == pytest.approx(fwd[col].sum(), rel=1e-12)
    # header dates are the 14 forward target days
    days = [fc.cell(BW.FC_HEADER_ROW, c).value for c in range(BW.FC_DATE_COL1, BW.FC_TOTAL_COL)]
    assert [d.date() for d in days] == sorted({d.date() for d in fwd["target_date"]})


@pytest.mark.data
def test_summary_and_scenario_baseline(ev, outputs):
    fwd, tr = outputs
    rows = {ev.wb["Summary"][f"A{r}"].value: f"B{r}" for r in range(14, 40) if ev.wb["Summary"][f"A{r}"].value}
    val = lambda lab: ev.value("Summary", rows[lab])  # noqa: E731
    assert val("Total 14-day forecast (units, all series)") == pytest.approx(fwd["forecast"].sum())
    assert val("Review exceptions (human review)") == (tr["band"] == "Review").sum()
    assert val("Accept forecasts (release as is)") == (tr["band"] == "Accept").sum()
    assert val("Series in scope") == 96
    res = json.loads((ROOT / "outputs" / "holdout_results.json").read_text())
    assert val("Holdout WAPE of the champion") == pytest.approx(res["champion_system"]["wape"] / 100)
    assert val("Consensus 14-day total after overrides") == pytest.approx(fwd["forecast"].sum())
    base = fwd[fwd["family"] == BW.DEFAULT_FAMILY]["forecast"].sum()
    assert ev.value("Scenario", BW.S["baseline"]) == pytest.approx(base)


@pytest.mark.data
def test_exceptions_match_trust_scores(ev, outputs):
    _, tr = outputs
    t = tr.set_index("series_id")
    prev = math.inf
    for r in range(BW.EX_FIRST, BW.EX_LAST + 1):
        sid = ev.value("Exceptions", f"B{r}")
        assert ev.value("Exceptions", f"E{r}") == pytest.approx(t.loc[sid, "score"], abs=1e-9)
        assert ev.value("Exceptions", f"F{r}") == t.loc[sid, "band"]
        assert ev.value("Exceptions", f"M{r}") == pytest.approx(t.loc[sid, "forecast_14d"])
        pr = ev.value("Exceptions", f"N{r}")
        assert pr == pytest.approx(t.loc[sid, "priority"], rel=1e-9)
        assert pr <= prev and ev.value("Exceptions", f"A{r}") == r - BW.EX_FIRST + 1
        prev = pr
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
