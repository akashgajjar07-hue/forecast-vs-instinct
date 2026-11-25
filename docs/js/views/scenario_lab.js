import { load, esc, fmtNum, fmtPct, plot, C, viewHead, options, setParams, famLabel, showEmpty } from "../core.js";
import { runScenario, SCENARIOS, DEFAULT_WEIGHTS } from "../../scenario.js";

const LABEL = { low: "Low", base: "Base", high: "High" };
const DEFAULTS = { lead: 14, days: 28, cu: 3, co: 1 };

export async function render(el, params) {
  const sc = await load("scenario.json");
  const events = [...sc.events, { id: "custom", label: "Custom", low: 0, base: 0, high: 0 }];
  const fams = sc.families.map((f) => f.family);
  const famRow = (f) => sc.families.find((x) => x.family === f);
  const state = {
    event: events.some((e) => e.id === params.event) ? params.event : events[0].id,
    family: fams.includes(params.family) ? params.family : fams[0],
  };

  el.innerHTML = `
  ${viewHead("06", "UAE Scenario Lab", "A what-if calculator for planning around UAE events. It runs entirely in your browser on the numbers you enter.", "coil",
    '<p style="margin-top:10px"><span class="hyp-badge" role="note">◇ Hypothetical scenario – not model output</span></p>')}
  <div class="hypothetical" aria-label="Hypothetical scenario calculator">
    <p class="small"><b>${esc(sc.assumption_label)}.</b> Event percentages below are editable demonstration assumptions, not measured UAE uplift. No UAE data was used anywhere in this project.</p>
    <div class="card">
      <h2>Inputs</h2>
      <div class="inputs-grid">
        <label class="field"><span>Event</span><select id="sl-event">${options(events.map((e) => [e.id, e.label]), state.event)}</select></label>
        <label class="field"><span>Category</span><select id="sl-family">${options(fams.map((f) => [f, famLabel(f)]), state.family)}</select></label>
        <label class="field"><span>Baseline units (14 days)</span><input type="number" id="sl-baseline" min="0" step="1"></label>
        <label class="field"><span>Low change %</span><input type="number" id="sl-low" step="1"></label>
        <label class="field"><span>Base change %</span><input type="number" id="sl-base" step="1"></label>
        <label class="field"><span>High change %</span><input type="number" id="sl-high" step="1"></label>
        <label class="field"><span>Available supply (units)</span><input type="number" id="sl-supply" min="0" step="1"></label>
        <label class="field"><span>Supplier lead time (days)</span><input type="number" id="sl-lead" min="0" step="1" value="${DEFAULTS.lead}"></label>
        <label class="field"><span>Days until event start</span><input type="number" id="sl-days" min="0" step="1" value="${DEFAULTS.days}"></label>
        <label class="field"><span>Cost per unit under-forecast</span><input type="number" id="sl-cu" min="0" step="0.1" value="${DEFAULTS.cu}"></label>
        <label class="field"><span>Cost per unit over-forecast</span><input type="number" id="sl-co" min="0" step="0.1" value="${DEFAULTS.co}"></label>
      </div>
      <p class="small muted" id="sl-baseline-note" style="margin-top:10px"></p>
      <button type="button" class="btn sm" id="sl-reset">Reset baseline and supply to the real forecast</button>
    </div>
    <div class="card">
      <div class="card-head"><h2>Scenario demand vs supply</h2><span class="muted small">hypothetical</span></div>
      <div class="range-cards" id="sl-range"></div>
      <div class="grid g2" style="margin-top:14px">
        <div id="sl-chart" class="chart short"></div>
        <div class="table-wrap" id="sl-pos"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><h2>Cost of each plan</h2><span class="muted small" id="sl-cr"></span></div>
      <div class="table-wrap" id="sl-cost"></div>
      <p class="caption">cost(plan, outcome) = under-cost × units short + over-cost × units left over. Expected cost uses weights low/base/high = ${DEFAULT_WEIGHTS.low * 100}/${DEFAULT_WEIGHTS.base * 100}/${DEFAULT_WEIGHTS.high * 100} %. Max regret = worst gap to the best plan for that outcome.</p>
    </div>
    <div class="card">
      <h2>Recommended action</h2>
      <div class="reco" id="sl-reco" aria-live="polite"></div>
      <div id="sl-warn"></div>
    </div>
  </div>
  <section class="card real-panel">
    <div class="card-head"><h2>Ecuador analogue</h2><span class="real-badge">Real data · Favorita, Ecuador</span></div>
    <p class="small">Observed change in mean daily sales on Ecuador national holidays vs normal days. <b>Not a UAE estimate</b>: a different country, calendar and shopping culture. Shown only as a sense-check for the size of the assumptions above.</p>
    <div id="sl-analogue" class="chart short"></div>
  </section>`;

  const sel = (id) => el.querySelector(id);
  const num = (id) => { const v = parseFloat(sel(id).value); return Number.isFinite(v) ? v : 0; };

  function setEvent() {
    const e = events.find((x) => x.id === state.event);
    sel("#sl-low").value = e.low; sel("#sl-base").value = e.base; sel("#sl-high").value = e.high;
  }
  function setBaseline() {
    const f = famRow(state.family);
    sel("#sl-baseline").value = Math.round(f.baseline_14d);
    sel("#sl-supply").value = Math.round(f.baseline_14d);
    sel("#sl-baseline-note").innerHTML = `Default baseline = real 14-day champion forecast total for ${esc(famLabel(state.family))}, Favorita Ecuador (source: ${esc(f.baseline_source)}). Default supply = that baseline. Both are editable.`;
  }

  function compute() {
    const r = runScenario({
      baseline: num("#sl-baseline"), pct_low: num("#sl-low"), pct_base: num("#sl-base"), pct_high: num("#sl-high"),
      supply: num("#sl-supply"), lead_time_days: num("#sl-lead"), days_to_event: num("#sl-days"), cost_under: num("#sl-cu"), cost_over: num("#sl-co"),
    });
    const supply = Math.max(0, num("#sl-supply"));
    sel("#sl-range").innerHTML = SCENARIOS.map((s) => `<div class="range-card${s === r.recommended_plan ? " rec" : ""}"><span class="small muted">${LABEL[s]} scenario${s === r.recommended_plan ? " · recommended plan" : ""}</span><b>${fmtNum(r.demand[s])}</b><span class="small">units · ${fmtPct(r.position[s].coverage_pct, 0)} covered</span></div>`).join("");
    sel("#sl-pos").innerHTML = `<table><thead><tr><th>Outcome</th><th class="n">Demand</th><th class="n">Shortfall</th><th class="n">Excess</th><th class="n">Coverage</th></tr></thead><tbody>
      ${SCENARIOS.map((s) => `<tr><td>${LABEL[s]}</td><td class="n">${fmtNum(r.demand[s])}</td><td class="n">${fmtNum(r.position[s].shortfall)}</td><td class="n">${fmtNum(r.position[s].excess)}</td><td class="n">${fmtPct(r.position[s].coverage_pct, 0)}</td></tr>`).join("")}
      </tbody></table><p class="caption">Against available supply of ${fmtNum(supply)} units. Coverage = supply ÷ demand.</p>`;
    plot(sel("#sl-chart"), [
      { x: SCENARIOS.map((s) => LABEL[s]), y: SCENARIOS.map((s) => r.demand[s]), type: "bar", name: "Scenario demand", marker: { color: SCENARIOS.map((s) => (s === r.recommended_plan ? C.champion : C.periwinkle)) }, hovertemplate: "%{x}: %{y:,.0f} units<extra></extra>" },
      { x: SCENARIOS.map((s) => LABEL[s]), y: SCENARIOS.map(() => supply), type: "scatter", mode: "lines", name: "Available supply", line: { color: C.text, dash: "dash", width: 2 }, hovertemplate: "supply %{y:,.0f}<extra></extra>" },
    ], { yaxis: { title: { text: "units (14 days)" }, rangemode: "tozero" }, xaxis: { type: "category" } }, `Hypothetical scenario demand low ${Math.round(r.demand.low)}, base ${Math.round(r.demand.base)}, high ${Math.round(r.demand.high)} units against supply ${Math.round(supply)}.`);
    sel("#sl-cr").textContent = r.critical_ratio === null ? "" : `critical ratio cu/(cu+co) = ${fmtNum(r.critical_ratio, 2)}`;
    sel("#sl-cost").innerHTML = `<table><thead><tr><th>Plan (order to)</th>${SCENARIOS.map((o) => `<th class="n">If ${LABEL[o].toLowerCase()} happens</th>`).join("")}<th class="n">Expected cost</th><th class="n">Max regret</th></tr></thead><tbody>
      ${SCENARIOS.map((p) => `<tr class="${p === r.recommended_plan ? "is-selected" : ""}"><td>${LABEL[p]} · ${fmtNum(r.demand[p])}${p === r.recommended_plan ? " ★" : ""}</td>${SCENARIOS.map((o) => `<td class="n">${fmtNum(r.cost_matrix[p][o])}</td>`).join("")}<td class="n${p === r.recommended_plan ? " best" : ""}">${fmtNum(r.expected_cost[p])}</td><td class="n">${fmtNum(r.max_regret[p])}</td></tr>`).join("")}
      <tr><td class="muted">Keep current supply · ${fmtNum(supply)}</td>${SCENARIOS.map((o) => `<td class="n muted">${fmtNum(r.supply_cost[o])}</td>`).join("")}<td class="n muted">${fmtNum(r.supply_expected_cost)}</td><td class="n muted">–</td></tr>
      </tbody></table>`;
    const timing = r.order.status === "late" ? "Order-by date already passed" : r.order.status === "now" ? "Decision needed now" : `Order-by in ${fmtNum(r.order.order_in_days)} days`;
    sel("#sl-reco").innerHTML = `<p style="margin:0 0 6px"><b>${esc(timing)}.</b> Order-by date = event start − supplier lead time.</p><p style="margin:0">${esc(r.recommendation)}</p>`;
    sel("#sl-warn").innerHTML = r.warnings.map((w) => `<p class="note warn small" style="margin-top:10px">${esc(w)}</p>`).join("");
  }

  function drawAnalogue() {
    const rows = sc.ecuador_analogue.filter((r) => r.change_pct !== null);
    const box = sel("#sl-analogue");
    if (!rows.length) { showEmpty(box, "No Ecuador holiday analogue published."); return; }
    plot(box, [{
      x: rows.map((r) => famLabel(r.family)), y: rows.map((r) => r.change_pct), type: "bar", name: "Observed holiday change",
      marker: { color: rows.map((r) => (r.family === state.family ? C.skyData : "rgba(104,190,243,0.35)")), line: { color: C.skyData, width: 1 } },
      customdata: rows.map((r) => [r.n_holidays, r.window]), hovertemplate: "%{x}: %{y:+.1f} %<br>%{customdata[0]} holidays, %{customdata[1]}<extra></extra>",
    }], { yaxis: { title: { text: "% vs normal days" }, ticksuffix: " %" }, xaxis: { tickangle: -30 }, showlegend: false },
    "Ecuador analogue: observed percent change in sales on national holidays by family. Real Favorita data, not a UAE estimate.");
  }

  sel("#sl-event").addEventListener("change", (e) => { state.event = e.target.value; setParams({ event: state.event, family: state.family }); setEvent(); compute(); });
  sel("#sl-family").addEventListener("change", (e) => { state.family = e.target.value; setParams({ event: state.event, family: state.family }); setBaseline(); compute(); drawAnalogue(); });
  sel("#sl-reset").addEventListener("click", () => { setBaseline(); compute(); });
  el.querySelectorAll(".inputs-grid input").forEach((i) => i.addEventListener("input", compute));

  setEvent(); setBaseline(); compute(); drawAnalogue();
}
