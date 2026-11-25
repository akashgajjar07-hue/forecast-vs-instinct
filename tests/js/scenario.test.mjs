// Run: node --test tests/js/scenario.test.mjs
// Same cases as tests/test_scenario.py (tests/js/scenario_cases.json); pytest also compares JS vs Python outputs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  runScenario, scenarioDemand, supplyPosition, planCost, costMatrix, maxRegret, orderTiming, fmtInt,
} from "../../docs/scenario.js";

const here = dirname(fileURLToPath(import.meta.url));
const CASES = JSON.parse(readFileSync(join(here, "scenario_cases.json"), "utf8"));
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: ${a} != ${b}`);

test("demand applies percent changes and clamps at zero", () => {
  const d = scenarioDemand(1000, { low: -10, base: 10, high: -150 });
  close(d.low, 900, "low");
  close(d.base, 1100, "base");
  assert.equal(d.high, 0);
});

test("supply position", () => {
  assert.deepEqual(supplyPosition(900, 1100), { shortfall: 0, excess: 200, coverage_pct: (100 * 1100) / 900 });
  assert.deepEqual(supplyPosition(0, 10), { shortfall: 0, excess: 10, coverage_pct: null });
});

test("asymmetric cost", () => {
  assert.equal(planCost(900, 1300, 3, 1), 1200);
  assert.equal(planCost(1300, 900, 3, 1), 400);
  const m = costMatrix({ low: 900, base: 1100, high: 1300 }, 3, 1);
  assert.equal(m.low.high, 1200);
  assert.equal(m.high.low, 400);
  assert.deepEqual(maxRegret(m), { low: 1200, base: 600, high: 400 });
});

test("order timing", () => {
  assert.deepEqual(orderTiming(21, 14), { order_in_days: 7, status: "later" });
  assert.deepEqual(orderTiming(10, 8), { order_in_days: 2, status: "now" });
  assert.deepEqual(orderTiming(5, 12), { order_in_days: -7, status: "late" });
});

test("fmtInt rounds half up and adds separators", () => {
  assert.equal(fmtInt(1234567.5), "1,234,568");
  assert.equal(fmtInt(-2.5), "-2");
  assert.equal(fmtInt(null), "n/a");
});

test("worked example (tie on expected cost broken by regret)", () => {
  const r = runScenario(CASES[0]);
  assert.equal(r.recommended_plan, "high");
  close(r.expected_cost.base, 200, "E base");
  close(r.expected_cost.high, 200, "E high");
  close(r.expected_cost.low, 600, "E low");
  close(r.supply_gap, 200, "gap");
  assert.equal(
    r.recommendation,
    "Plan to the High scenario (1,300 units): lowest expected cost (200) with weights low/base/high = 25/50/25 % " +
      "and an under:over cost ratio of 3:1. Available supply is 200 units short of that plan. " +
      "Place the replenishment within 7 days (event start minus lead time)."
  );
});

test("every case runs and is internally consistent", () => {
  for (const c of CASES) {
    const r = runScenario(c);
    for (const s of ["low", "base", "high"]) {
      assert.ok(r.demand[s] >= 0, c.name);
      const p = r.position[s];
      assert.ok(p.shortfall === 0 || p.excess === 0, c.name);
    }
    assert.ok(["low", "base", "high"].includes(r.recommended_plan));
    assert.equal(typeof r.recommendation, "string");
  }
  assert.ok(runScenario(CASES[5]).warnings.length >= 2);
});
