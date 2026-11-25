"""Scenario Lab formulas: Python mirror (src/scenario.py) vs JS (docs/scenario.js) on the same cases."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.scenario import cost_matrix, fmt_int, max_regret, order_timing, plan_cost, run_scenario, scenario_demand

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "js" / "scenario_cases.json").read_text())


def test_demand_and_clamp():
    d = scenario_demand(1000, {"low": -10, "base": 10, "high": -150})
    assert d["low"] == pytest.approx(900) and d["base"] == pytest.approx(1100) and d["high"] == 0


def test_costs_and_regret():
    assert plan_cost(900, 1300, 3, 1) == 1200
    assert plan_cost(1300, 900, 3, 1) == 400
    m = cost_matrix({"low": 900, "base": 1100, "high": 1300}, 3, 1)
    assert max_regret(m) == {"low": 1200, "base": 600, "high": 400}


def test_timing_and_format():
    assert order_timing(21, 14) == {"order_in_days": 7, "status": "later"}
    assert order_timing(10, 8)["status"] == "now"
    assert order_timing(5, 12)["status"] == "late"
    assert fmt_int(1234567.5) == "1,234,568" and fmt_int(-2.5) == "-2" and fmt_int(None) == "n/a"


def test_worked_example():
    r = run_scenario(CASES[0])
    assert r["recommended_plan"] == "high"
    assert r["expected_cost"] == pytest.approx({"low": 600, "base": 200, "high": 200})
    assert r["recommendation"] == (
        "Plan to the High scenario (1,300 index-days): lowest expected cost (200) with weights low/base/high = 25/50/25 % "
        "and an under:over cost ratio of 3:1. Available supply is 200 index-days short of that plan. "
        "Place the replenishment within 7 days (event start minus lead time).")


def _assert_same(a, b, path="r"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and b == pytest.approx(a, abs=1e-9), path
    else:
        assert a == b, path


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_js_and_python_agree():
    script = (
        "import { runScenario } from './docs/scenario.js';"
        "import { readFileSync } from 'node:fs';"
        "const c = JSON.parse(readFileSync('tests/js/scenario_cases.json','utf8'));"
        "console.log(JSON.stringify(c.map(runScenario)));"
    )
    out = subprocess.run([NODE, "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True,
                         check=True, timeout=60)
    js = json.loads(out.stdout)
    for case, j in zip(CASES, js):
        _assert_same(run_scenario(case), j, case["name"])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_node_unit_tests_pass():
    out = subprocess.run([NODE, "--test", "tests/js/scenario.test.mjs"], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
