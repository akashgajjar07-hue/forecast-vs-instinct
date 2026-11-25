"""UAE Scenario Lab formulas: exact Python mirror of docs/scenario.js (pure functions).

HYPOTHETICAL planning arithmetic on user-entered assumptions. Not model output.
See docs/scenario.js for the formula list; tests/test_scenario.py checks both agree.
"""
from __future__ import annotations

import math

SCENARIOS = ["low", "base", "high"]
DEFAULT_WEIGHTS = {"low": 0.25, "base": 0.5, "high": 0.25}
REVIEW_WINDOW_DAYS = 3
LABEL = {"low": "Low", "base": "Base", "high": "High"}


def _num(x) -> float:
    """JS `Number(x) || 0` semantics for the inputs we accept."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def fmt_int(x) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    n = math.floor(x + 0.5)
    return ("-" if n < 0 else "") + f"{abs(n):,}"


def scenario_demand(baseline: float, pct: dict) -> dict:
    return {s: max(0.0, baseline * (1 + pct[s] / 100)) for s in SCENARIOS}


def supply_position(demand: float, supply: float) -> dict:
    return {"shortfall": max(0.0, demand - supply), "excess": max(0.0, supply - demand),
            "coverage_pct": (100 * supply) / demand if demand > 0 else None}


def plan_cost(q: float, d: float, cu: float, co: float) -> float:
    return cu * max(0.0, d - q) + co * max(0.0, q - d)


def cost_matrix(demand: dict, cu: float, co: float) -> dict:
    return {p: {o: plan_cost(demand[p], demand[o], cu, co) for o in SCENARIOS} for p in SCENARIOS}


def expected_cost(matrix: dict, weights: dict) -> dict:
    out = {}
    for p in SCENARIOS:
        t = 0.0
        for o in SCENARIOS:
            t += weights[o] * matrix[p][o]
        out[p] = t
    return out


def max_regret(matrix: dict) -> dict:
    best = {o: min(matrix[p][o] for p in SCENARIOS) for o in SCENARIOS}
    return {p: max(matrix[p][o] - best[o] for o in SCENARIOS) for p in SCENARIOS}


def argmin_plan(primary: dict, secondary: dict) -> str:
    best = SCENARIOS[0]
    for p in SCENARIOS[1:]:
        d = primary[p] - primary[best]
        if d < -1e-9 or (abs(d) <= 1e-9 and secondary[p] < secondary[best] - 1e-9):
            best = p
    return best


def order_timing(days_to_event: float, lead_time_days: float) -> dict:
    order_in = days_to_event - lead_time_days
    if order_in < 0:
        status = "late"
    elif order_in <= REVIEW_WINDOW_DAYS:
        status = "now"
    else:
        status = "later"
    return {"order_in_days": order_in, "status": status}


def run_scenario(inp: dict) -> dict:
    weights = inp.get("weights") or DEFAULT_WEIGHTS
    warnings = []
    baseline = max(0.0, _num(inp.get("baseline")))
    supply = max(0.0, _num(inp.get("supply")))
    cu = max(0.0, _num(inp.get("cost_under")))
    co = max(0.0, _num(inp.get("cost_over")))
    pct = {"low": _num(inp.get("pct_low")), "base": _num(inp.get("pct_base")), "high": _num(inp.get("pct_high"))}
    if not (pct["low"] <= pct["base"] <= pct["high"]):
        warnings.append("Low, base and high changes are not in ascending order.")
    if baseline == 0:
        warnings.append("Baseline is zero, so every scenario demand is zero.")
    if cu == 0 and co == 0:
        warnings.append("Both unit costs are zero, so every plan costs nothing.")

    demand = scenario_demand(baseline, pct)
    position = {s: supply_position(demand[s], supply) for s in SCENARIOS}
    matrix = cost_matrix(demand, cu, co)
    expected = expected_cost(matrix, weights)
    regret = max_regret(matrix)
    recommended = argmin_plan(expected, regret)
    minimax = argmin_plan(regret, expected)
    critical = cu / (cu + co) if cu + co > 0 else None
    lead = _num(inp.get("lead_time_days"))
    timing = order_timing(_num(inp.get("days_to_event")), lead)
    supply_cost = {o: plan_cost(supply, demand[o], cu, co) for o in SCENARIOS}
    supply_expected = 0.0
    for o in SCENARIOS:
        supply_expected += weights[o] * supply_cost[o]

    q = demand[recommended]
    gap = q - supply
    parts = [
        f"Plan to the {LABEL[recommended]} scenario ({fmt_int(q)} index-days): lowest expected cost "
        f"({fmt_int(expected[recommended])}) with weights low/base/high = {fmt_int(weights['low'] * 100)}/"
        f"{fmt_int(weights['base'] * 100)}/{fmt_int(weights['high'] * 100)} % and an under:over cost ratio of "
        f"{fmt_int(cu)}:{fmt_int(co)}."
    ]
    if gap > 0.5:
        parts.append(f"Available supply is {fmt_int(gap)} index-days short of that plan.")
        if timing["status"] == "late":
            parts.append(f"The standard order-by date passed {fmt_int(-timing['order_in_days'])} days ago: "
                         "expedite, reallocate stock or agree a substitution.")
        elif timing["status"] == "now":
            parts.append(f"Order now: the order-by date (event start minus {fmt_int(lead)} days lead time) is "
                         f"{fmt_int(timing['order_in_days'])} days away.")
        else:
            parts.append(f"Place the replenishment within {fmt_int(timing['order_in_days'])} days "
                         "(event start minus lead time).")
    elif gap < -0.5:
        parts.append(f"Available supply exceeds that plan by {fmt_int(-gap)} index-days: hold or reallocate the excess "
                     "rather than reorder.")
    else:
        parts.append("Available supply matches that plan: no replenishment decision is needed.")
    if minimax != recommended:
        parts.append(f"A minimax-regret planner would choose the {LABEL[minimax]} scenario instead.")

    return {
        "demand": demand, "position": position, "cost_matrix": matrix, "expected_cost": expected,
        "max_regret": regret, "supply_cost": supply_cost, "supply_expected_cost": supply_expected,
        "recommended_plan": recommended, "minimax_regret_plan": minimax, "critical_ratio": critical,
        "order": timing, "supply_gap": gap, "recommendation": " ".join(parts), "warnings": warnings,
    }
