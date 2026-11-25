import { load, esc, fmtNum, plot, C, viewHead, options, setParams, famLabel, showEmpty, MODELS, MODEL_LABEL, MODEL_COLOR, MODEL_DASH, tableWrapAttrs } from "../core.js";

const SLICES = [
  ["overall", "Overall"], ["family", "Family"], ["store", "Store"], ["horizon", "Horizon (days ahead)"],
  ["horizon_bucket", "Horizon bucket"], ["promo", "Promotion days"], ["holiday", "Holiday days"], ["week_bucket", "Week bucket (7-day sums)"],
];
const METRICS = [["wape", "WAPE (%)"], ["bias", "Bias (%)"], ["mae", "MAE (units)"], ["rmse", "RMSE (units)"]];
const SLICE_ORDER = { horizon_bucket: ["1-7", "8-14"], promo: ["promo", "no_promo"], holiday: ["holiday", "normal"], week_bucket: ["week1", "week2"] };
const SLICE_LABEL = { promo: "promo days", no_promo: "no-promo days", holiday: "holidays", normal: "normal days", week1: "week 1", week2: "week 2", all: "All" };

function sliceLabel(type, v) {
  if (type === "family") return famLabel(v);
  if (type === "store") return `Store ${v}`;
  if (type === "horizon") return `h${v}`;
  return SLICE_LABEL[v] || v;
}
const better = (metric, a, b) => (metric === "bias" ? Math.abs(a) < Math.abs(b) : a < b);

export async function render(el, params) {
  const [arena, meta] = await Promise.all([load("arena.json"), load("meta.json")]);
  const M = arena.metrics;
  const stages = [...new Set(M.map((r) => r.stage))].sort((a, b) => (a === "holdout" ? -1 : b === "holdout" ? 1 : a.localeCompare(b)));
  const slices = SLICES.filter(([s]) => M.some((r) => r.slice_type === s));
  const hasB = M.some((r) => r.variant === "B");
  const state = {
    stage: stages.includes(params.stage) ? params.stage : stages[0],
    slice: slices.some(([s]) => s === params.slice) ? params.slice : "overall",
    metric: METRICS.some(([m]) => m === params.metric) ? params.metric : "wape",
    variant: params.variant === "B" && hasB ? "B" : "A",
  };

  el.innerHTML = `
  ${viewHead("03", "Model Arena", "Five models scored on identical windows. Lower WAPE and MAE are better; bias closest to zero is better. Holdout is the final, untouched test; champions were chosen on selection only.", "coil")}
  <div class="controls" role="group" aria-label="Arena filters">
    <label class="field"><span>Stage</span><select id="ar-stage">${options(stages.map((s) => [s, s === "holdout" ? "Holdout (final test)" : s === "selection" ? "Selection (model choice)" : s]), state.stage)}</select></label>
    <label class="field"><span>Slice</span><select id="ar-slice">${options(slices, state.slice)}</select></label>
    <label class="field"><span>Metric</span><select id="ar-metric">${options(METRICS, state.metric)}</select></label>
    ${hasB ? `<label class="field"><span>LightGBM promo info</span><select id="ar-variant">${options([["A", "A · known promo plan"], ["B", "B · no future promo info"]], state.variant)}</select></label>` : ""}
  </div>
  <section class="card"><div class="card-head"><h2 id="ar-title">Scores</h2><span class="muted small" id="ar-sub"></span></div><div id="ar-chart" class="chart tall"></div></section>
  <section class="card"><h2 id="ar-table-title">Table</h2><div class="table-wrap scroll-y" id="ar-table" ${tableWrapAttrs("Arena metrics by model and slice, scrollable")}></div><p class="caption">★ marks the best model in each row. Baselines do not use promotion information, so they appear once.</p></section>
  <div class="grid g2">
    <section class="card"><h2>Champion per family</h2><div class="table-wrap" id="ar-champs" ${tableWrapAttrs("Champion per family table, scrollable")}></div><p class="caption">Lowest selection WAPE, tie-break lower |bias|. Holdout column shows how the frozen champion (LightGBM variant A, known promo plan) then did on the untouched test, whatever the promo-info toggle above.</p></section>
    <section class="card"><h2>Value of promotion information</h2><div id="ar-var" class="chart"></div><p class="caption" id="ar-var-cap"></p></section>
  </div>
  ${arena.notes && arena.notes.length ? `<section class="card"><h2>Notes</h2><ul class="clean">${arena.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></section>` : ""}`;

  const sel = (id) => el.querySelector(id);
  const bind = (id, key) => sel(id)?.addEventListener("change", (e) => { state[key] = e.target.value; setParams({ stage: state.stage, slice: state.slice, metric: state.metric, variant: state.variant }); draw(); });
  bind("#ar-stage", "stage"); bind("#ar-slice", "slice"); bind("#ar-metric", "metric"); bind("#ar-variant", "variant");

  const pick = (model, sliceType, stage = state.stage) => {
    const v = model === "lightgbm" || M.some((r) => r.model === model && r.variant === state.variant && r.stage === stage) ? state.variant : "A";
    let rows = M.filter((r) => r.stage === stage && r.model === model && r.slice_type === sliceType && r.variant === v);
    if (!rows.length) rows = M.filter((r) => r.stage === stage && r.model === model && r.slice_type === sliceType && r.variant === "A");
    return rows;
  };
  const metricLabel = () => METRICS.find(([m]) => m === state.metric)[1];

  function sliceValues() {
    const vals = [...new Set(M.filter((r) => r.stage === state.stage && r.slice_type === state.slice).map((r) => r.slice_value))];
    if (state.slice === "family") return meta.scope.families.filter((f) => vals.includes(f));
    if (state.slice === "store" || state.slice === "horizon") return vals.sort((a, b) => Number(a) - Number(b));
    if (SLICE_ORDER[state.slice]) return SLICE_ORDER[state.slice].filter((v) => vals.includes(v)).concat(vals.filter((v) => !SLICE_ORDER[state.slice].includes(v)));
    return vals;
  }

  function drawChart() {
    const box = sel("#ar-chart");
    const models = MODELS.filter((m) => M.some((r) => r.model === m && r.stage === state.stage));
    sel("#ar-title").textContent = `${metricLabel()} · ${state.stage} · ${SLICES.find(([s]) => s === state.slice)[1]}`;
    sel("#ar-sub").textContent = state.variant === "B" ? "LightGBM shown without future promo info" : "LightGBM shown with known promo plan";
    if (!models.length) { showEmpty(box, "No metrics for this stage."); return; }
    const ysuf = state.metric === "wape" || state.metric === "bias" ? " %" : "";
    if (state.slice === "overall") {
      const rows = models.map((m) => pick(m, "overall")[0]);
      plot(box, [{
        x: models.map((m) => MODEL_LABEL[m]), y: rows.map((r) => r?.[state.metric] ?? null), type: "bar",
        marker: { color: models.map((m) => MODEL_COLOR[m]) }, text: rows.map((r) => (r && r[state.metric] !== null ? fmtNum(r[state.metric], 1) : "n/a")),
        textposition: "outside", textfont: { color: C.text }, hovertemplate: "%{x}: %{y:.2f}<extra></extra>", cliponaxis: false,
      }], { yaxis: { title: { text: metricLabel() }, ticksuffix: ysuf }, showlegend: false }, `${metricLabel()} by model, ${state.stage}, overall.`);
      return;
    }
    const xs = sliceValues();
    if (!xs.length) { showEmpty(box, "No metrics for this slice."); return; }
    const line = state.slice === "horizon";
    const traces = models.map((m, i) => {
      const rows = pick(m, state.slice);
      return {
        x: xs.map((v) => sliceLabel(state.slice, v)), y: xs.map((v) => rows.find((r) => r.slice_value === v)?.[state.metric] ?? null),
        name: MODEL_LABEL[m], type: line ? "scatter" : "bar", mode: line ? "lines+markers" : undefined,
        marker: { color: MODEL_COLOR[m], symbol: ["circle", "square", "diamond", "triangle-up", "x"][i] },
        line: line ? { color: MODEL_COLOR[m], width: m === "lightgbm" ? 2.6 : 1.6, dash: MODEL_DASH[m] } : undefined,
        hovertemplate: `${MODEL_LABEL[m]} · %{x}: %{y:.2f}<extra></extra>`,
      };
    });
    plot(box, traces, { barmode: "group", yaxis: { title: { text: metricLabel() }, ticksuffix: ysuf, zeroline: true }, xaxis: { tickangle: state.slice === "family" || state.slice === "store" ? -30 : 0, type: "category" }, hovermode: "closest" },
      `${metricLabel()} by model for each ${state.slice} slice, ${state.stage} stage.`);
  }

  function drawTable() {
    const box = sel("#ar-table");
    const models = MODELS.filter((m) => M.some((r) => r.model === m && r.stage === state.stage));
    if (!models.length) { box.innerHTML = '<div class="empty">No metrics for this stage.</div>'; return; }
    if (state.slice === "overall") {
      sel("#ar-table-title").textContent = `All metrics · ${state.stage} · overall`;
      const rows = models.map((m) => [m, pick(m, "overall")[0]]);
      const best = {};
      for (const [k] of METRICS) for (const [m, r] of rows) if (r && r[k] !== null && (best[k] === undefined || better(k, r[k], rows.find(([mm]) => mm === best[k])[1][k]))) best[k] = m;
      box.innerHTML = `<table><thead><tr><th>Model</th>${METRICS.map(([, l]) => `<th class="n">${l}</th>`).join("")}<th class="n">n</th></tr></thead><tbody>
        ${rows.map(([m, r]) => `<tr><td><span class="swatch" style="background:${MODEL_COLOR[m]}"></span> ${MODEL_LABEL[m]}</td>${METRICS.map(([k]) => `<td class="n${best[k] === m ? " best" : ""}">${fmtNum(r?.[k], 2)}</td>`).join("")}<td class="n">${fmtNum(r?.n)}</td></tr>`).join("")}
      </tbody></table>`;
      return;
    }
    sel("#ar-table-title").textContent = `${metricLabel()} · ${state.stage} · by ${state.slice.replace("_", " ")}`;
    const xs = sliceValues();
    if (!xs.length) { box.innerHTML = '<div class="empty">No metrics for this slice.</div>'; return; }
    const data = Object.fromEntries(models.map((m) => [m, pick(m, state.slice)]));
    box.innerHTML = `<table><thead><tr><th>${esc(state.slice.replace("_", " "))}</th>${models.map((m) => `<th class="n">${MODEL_LABEL[m]}</th>`).join("")}</tr></thead><tbody>
      ${xs.map((v) => {
        const vals = models.map((m) => data[m].find((r) => r.slice_value === v)?.[state.metric] ?? null);
        let bi = -1;
        vals.forEach((x, i) => { if (x !== null && (bi < 0 || better(state.metric, x, vals[bi]))) bi = i; });
        return `<tr><td>${esc(sliceLabel(state.slice, v))}</td>${vals.map((x, i) => `<td class="n${i === bi ? " best" : ""}">${fmtNum(x, 2)}</td>`).join("")}</tr>`;
      }).join("")}</tbody></table>`;
  }

  function drawChamps() {
    const box = sel("#ar-champs");
    const ch = Object.entries(arena.champions);
    if (!ch.length) { box.innerHTML = '<div class="empty">No champions published.</div>'; return; }
    box.innerHTML = `<table><thead><tr><th>Family</th><th>Champion</th><th class="n">Selection WAPE</th><th class="n">Selection bias</th><th class="n">Holdout WAPE</th></tr></thead><tbody>
      ${ch.map(([f, c]) => {
        const h = M.find((r) => r.stage === "holdout" && r.model === c.model && r.slice_type === "family" && r.slice_value === f && r.variant === "A"); // frozen champions are always variant A
        return `<tr><td>${esc(famLabel(f))}</td><td><span class="swatch" style="background:${MODEL_COLOR[c.model]}"></span> ${MODEL_LABEL[c.model] || esc(c.model)}</td><td class="n">${fmtNum(c.selection_wape, 1)} %</td><td class="n">${fmtNum(c.selection_bias, 1)} %</td><td class="n">${h ? fmtNum(h.wape, 1) + " %" : "n/a"}</td></tr>`;
      }).join("")}</tbody></table>`;
  }

  function drawVariant() {
    const box = sel("#ar-var");
    const rows = arena.variant_comparison.filter((r) => r.stage === state.stage);
    sel("#ar-var-cap").textContent = `LightGBM WAPE (${state.stage}) with the known promotion plan (A) vs without future promotion information (B). A gap shows how much the model leans on knowing the promo calendar.`;
    if (!rows.length) { showEmpty(box, `No A vs B comparison for the ${state.stage} stage.`); return; }
    const x = rows.map((r) => (r.family === "all" || r.family === "ALL" ? "All" : famLabel(r.family)));
    plot(box, [
      { x, y: rows.map((r) => r.wape_A), type: "bar", name: "A · known promo plan", marker: { color: C.champion }, hovertemplate: "%{x} A: %{y:.2f} %<extra></extra>" },
      { x, y: rows.map((r) => r.wape_B), type: "bar", name: "B · no future promo info", marker: { color: C.orchid, pattern: { shape: "/", fgcolor: C.panelDeep, size: 6 } }, hovertemplate: "%{x} B: %{y:.2f} %<extra></extra>" },
    ], { barmode: "group", yaxis: { title: { text: "WAPE (%)" }, ticksuffix: " %" }, xaxis: { tickangle: -30, type: "category" } }, `LightGBM WAPE with and without promotion information by family, ${state.stage}.`);
  }

  function draw() { drawChart(); drawTable(); drawChamps(); drawVariant(); }
  draw();
}
