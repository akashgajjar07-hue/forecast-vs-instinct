// UAE Scenario Lab: pure functions (no DOM, no I/O). Mirrored exactly in src/scenario.py.
// HYPOTHETICAL: these formulas turn user-entered assumptions into a planning range. They are not model output.
//
// Formulas (percent inputs are percent values, 10 means +10 %):
//   demand_s          = max(0, baseline * (1 + pct_s / 100))            s in {low, base, high}
//   shortfall_s       = max(0, demand_s - supply)
//   excess_s          = max(0, supply - demand_s)
//   coverage_pct_s    = 100 * supply / demand_s  (null when demand_s == 0)
//   cost(plan p, o)   = cu * max(0, D_o - Q_p) + co * max(0, Q_p - D_o),   Q_p = D_p
//   expected_cost(p)  = sum_o w_o * cost(p, o),  default w = {low .25, base .5, high .25}
//   regret(p, o)      = cost(p, o) - min_p' cost(p', o);  max_regret(p) = max_o regret(p, o)
//   recommended plan  = argmin expected_cost; ties -> lower max_regret -> order low, base, high
//   critical_ratio    = cu / (cu + co)  (newsvendor service-level hint)
//   order_in_days     = days_to_event - lead_time_days   (order-by date = event start - lead time)

export const SCENARIOS = ["low", "base", "high"];
export const DEFAULT_WEIGHTS = { low: 0.25, base: 0.5, high: 0.25 };
export const REVIEW_WINDOW_DAYS = 3;
const LABEL = { low: "Low", base: "Base", high: "High" };

export function fmtInt(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "n/a";
  const n = Math.floor(x + 0.5);
  const s = String(Math.abs(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return (n < 0 ? "-" : "") + s;
}

export function scenarioDemand(baseline, pct) {
  const out = {};
  for (const s of SCENARIOS) out[s] = Math.max(0, baseline * (1 + pct[s] / 100));
  return out;
}

export function supplyPosition(demand, supply) {
  return {
    shortfall: Math.max(0, demand - supply),
    excess: Math.max(0, supply - demand),
    coverage_pct: demand > 0 ? (100 * supply) / demand : null,
  };
}

export function planCost(q, d, cu, co) {
  return cu * Math.max(0, d - q) + co * Math.max(0, q - d);
}

export function costMatrix(demand, cu, co) {
  const m = {};
  for (const p of SCENARIOS) {
    m[p] = {};
    for (const o of SCENARIOS) m[p][o] = planCost(demand[p], demand[o], cu, co);
  }
  return m;
}

export function expectedCost(matrix, weights) {
  const e = {};
  for (const p of SCENARIOS) {
    let t = 0;
    for (const o of SCENARIOS) t += weights[o] * matrix[p][o];
    e[p] = t;
  }
  return e;
}

export function maxRegret(matrix) {
  const best = {};
  for (const o of SCENARIOS) best[o] = Math.min(...SCENARIOS.map((p) => matrix[p][o]));
  const r = {};
  for (const p of SCENARIOS) r[p] = Math.max(...SCENARIOS.map((o) => matrix[p][o] - best[o]));
  return r;
}

export function argminPlan(primary, secondary) {
  let bestP = SCENARIOS[0];
  for (const p of SCENARIOS.slice(1)) {
    const d = primary[p] - primary[bestP];
    if (d < -1e-9 || (Math.abs(d) <= 1e-9 && secondary[p] < secondary[bestP] - 1e-9)) bestP = p;
  }
  return bestP;
}

export function orderTiming(daysToEvent, leadTimeDays) {
  const orderIn = daysToEvent - leadTimeDays;
  let status;
  if (orderIn < 0) status = "late";
  else if (orderIn <= REVIEW_WINDOW_DAYS) status = "now";
  else status = "later";
  return { order_in_days: orderIn, status };
}

export function runScenario(inp) {
  const weights = inp.weights || DEFAULT_WEIGHTS;
  const warnings = [];
  const baseline = Math.max(0, Number(inp.baseline) || 0);
  const supply = Math.max(0, Number(inp.supply) || 0);
  const cu = Math.max(0, Number(inp.cost_under) || 0);
  const co = Math.max(0, Number(inp.cost_over) || 0);
  const pct = { low: Number(inp.pct_low) || 0, base: Number(inp.pct_base) || 0, high: Number(inp.pct_high) || 0 };
  if (!(pct.low <= pct.base && pct.base <= pct.high)) warnings.push("Low, base and high changes are not in ascending order.");
  if (baseline === 0) warnings.push("Baseline is zero, so every scenario demand is zero.");
  if (cu === 0 && co === 0) warnings.push("Both unit costs are zero, so every plan costs nothing.");

  const demand = scenarioDemand(baseline, pct);
  const position = {};
  for (const s of SCENARIOS) position[s] = supplyPosition(demand[s], supply);
  const matrix = costMatrix(demand, cu, co);
  const expected = expectedCost(matrix, weights);
  const regret = maxRegret(matrix);
  const recommended = argminPlan(expected, regret);
  const minimax = argminPlan(regret, expected);
  const critical = cu + co > 0 ? cu / (cu + co) : null;
  const timing = orderTiming(Number(inp.days_to_event) || 0, Number(inp.lead_time_days) || 0);
  const supplyCost = {};
  for (const o of SCENARIOS) supplyCost[o] = planCost(supply, demand[o], cu, co);
  let supplyExpected = 0;
  for (const o of SCENARIOS) supplyExpected += weights[o] * supplyCost[o];

  const q = demand[recommended];
  const gap = q - supply;
  const parts = [];
  parts.push(
    `Plan to the ${LABEL[recommended]} scenario (${fmtInt(q)} units): lowest expected cost (${fmtInt(expected[recommended])}) ` +
      `with weights low/base/high = ${fmtInt(weights.low * 100)}/${fmtInt(weights.base * 100)}/${fmtInt(weights.high * 100)} % ` +
      `and an under:over cost ratio of ${fmtInt(cu)}:${fmtInt(co)}.`
  );
  if (gap > 0.5) {
    parts.push(`Available supply is ${fmtInt(gap)} units short of that plan.`);
    if (timing.status === "late") parts.push(`The standard order-by date passed ${fmtInt(-timing.order_in_days)} days ago: expedite, reallocate stock or agree a substitution.`);
    else if (timing.status === "now") parts.push(`Order now: the order-by date (event start minus ${fmtInt(Number(inp.lead_time_days) || 0)} days lead time) is ${fmtInt(timing.order_in_days)} days away.`);
    else parts.push(`Place the replenishment within ${fmtInt(timing.order_in_days)} days (event start minus lead time).`);
  } else if (gap < -0.5) {
    parts.push(`Available supply exceeds that plan by ${fmtInt(-gap)} units: hold or reallocate the excess rather than reorder.`);
  } else {
    parts.push("Available supply matches that plan: no replenishment decision is needed.");
  }
  if (minimax !== recommended) parts.push(`A minimax-regret planner would choose the ${LABEL[minimax]} scenario instead.`);

  return {
    demand,
    position,
    cost_matrix: matrix,
    expected_cost: expected,
    max_regret: regret,
    supply_cost: supplyCost,
    supply_expected_cost: supplyExpected,
    recommended_plan: recommended,
    minimax_regret_plan: minimax,
    critical_ratio: critical,
    order: timing,
    supply_gap: gap,
    recommendation: parts.join(" "),
    warnings,
  };
}
