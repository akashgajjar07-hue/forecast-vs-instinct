import { load, loadSeries, esc, fmtNum, fmtPct, fmtDate, addDays, plot, bandTrace, C, viewHead, options, setParams, famLabel, showEmpty, showLoading, pill, MODELS, MODEL_LABEL, CHALLENGER_COLORS, CHALLENGER_DASH, purgeCharts } from "../core.js";

const SCALES = [-100, -50, 0, 50, 100];
const CONTEXT_DAYS = 56;
const COMP_LABEL = { accuracy: "Accuracy", interval: "Interval", stability: "Stability", agreement: "Agreement", completeness: "Completeness" };

export async function render(el, params, ctx) {
  const [index, meta] = await Promise.all([load("series_index.json"), load("meta.json")]);
  if (!index.length) {
    el.innerHTML = viewHead("04", "Forecast Workspace", "Pick a series to inspect its forecasts.", "torus") + '<div class="empty">No series published.</div>';
    return;
  }
  const famOrder = meta.scope.families;
  const stores = [...new Set(index.map((s) => s.store_nbr))].sort((a, b) => a - b);
  const start = index.find((s) => s.slug === params.series) || index[0];
  const state = {
    store: start.store_nbr, family: start.family, slug: start.slug, origin: params.origin || "forward",
    scale: SCALES.includes(Number(params.scale)) ? Number(params.scale) : 0,
    models: new Set((params.models || "").split(",").filter((m) => MODELS.includes(m))),
  };
  let ser = null;

  el.innerHTML = `
  ${viewHead("04", "Forecast Workspace", "One series at a time: actuals, the champion forecast with its P10–P90 band, challengers on demand, and a promotion what-if from the precomputed grid.", "torus")}
  <div class="controls" role="group" aria-label="Series selection">
    <label class="field"><span>Store</span><select id="ws-store"></select></label>
    <label class="field"><span>Family</span><select id="ws-family"></select></label>
    <label class="field"><span>Forecast origin</span><select id="ws-origin"></select></label>
  </div>
  <div class="grid g-2-1">
    <section class="card">
      <div class="card-head"><h2 id="ws-title">Series</h2><span class="muted small" id="ws-sub"></span></div>
      <div id="ws-chart" class="chart tall"></div>
      <fieldset style="border:0;padding:0;margin:10px 0 0"><legend class="small muted" style="margin-bottom:6px">Overlay challenger models</legend><div class="chips" id="ws-models"></div></fieldset>
      <p class="caption" id="ws-cap"></p>
    </section>
    <section class="card" id="ws-side"></section>
  </div>
  <section class="card">
    <div class="card-head"><h2>Promotion what-if</h2><span class="muted small" id="ws-wi-sub"></span></div>
    <div class="grid g2">
      <div>
        <label class="field" for="ws-scale"><span>Change to planned promotions: <b id="ws-scale-val"></b></span></label>
        <input type="range" id="ws-scale" min="0" max="4" step="1" list="ws-scale-ticks">
        <datalist id="ws-scale-ticks">${SCALES.map((s, i) => `<option value="${i}" label="${s > 0 ? "+" : ""}${s} %"></option>`).join("")}</datalist>
        <div class="small muted" style="display:flex;justify-content:space-between">${SCALES.map((s) => `<span>${s > 0 ? "+" : ""}${s} %</span>`).join("")}</div>
        <div id="ws-wi-text" style="margin-top:12px"></div>
      </div>
      <div id="ws-wi-chart" class="chart short"></div>
    </div>
  </section>`;

  const sel = (id) => el.querySelector(id);
  const storeSel = sel("#ws-store"), famSel = sel("#ws-family"), originSel = sel("#ws-origin"), scale = sel("#ws-scale");
  storeSel.innerHTML = options(stores.map((s) => [s, `Store ${s} · ${index.find((x) => x.store_nbr === s).city}`]), state.store);
  scale.value = String(SCALES.indexOf(state.scale));

  function fillFamilies() {
    const fams = index.filter((s) => s.store_nbr === Number(state.store)).map((s) => s.family).sort((a, b) => famOrder.indexOf(a) - famOrder.indexOf(b));
    if (!fams.includes(state.family)) state.family = fams[0];
    famSel.innerHTML = options(fams.map((f) => [f, famLabel(f)]), state.family);
  }
  fillFamilies();

  const push = () => setParams({ series: state.slug, origin: state.origin === "forward" ? null : state.origin, scale: state.scale || null, models: [...state.models].join(",") || null });

  async function loadCurrent() {
    const entry = index.find((s) => s.store_nbr === Number(state.store) && s.family === state.family);
    state.slug = entry.slug;
    push();
    showLoading(sel("#ws-chart"), "Loading series…");
    const slug = state.slug;
    const data = await loadSeries(slug);
    if (!ctx.isCurrent() || slug !== state.slug) return;
    ser = data;
    const origins = ser.holdout.map((h) => h.origin);
    if (state.origin !== "forward" && !origins.includes(state.origin)) state.origin = "forward";
    originSel.innerHTML = options([["forward", `Forward · from ${fmtDate(addDays(ser.forward.origin, 1))} (no actuals yet)`], ...origins.map((o) => [o, `Holdout · origin ${fmtDate(o)}`])], state.origin);
    drawAll();
  }

  storeSel.addEventListener("change", () => { state.store = Number(storeSel.value); fillFamilies(); loadCurrent(); });
  famSel.addEventListener("change", () => { state.family = famSel.value; loadCurrent(); });
  originSel.addEventListener("change", () => { state.origin = originSel.value; push(); drawAll(); });
  scale.addEventListener("input", () => { state.scale = SCALES[Number(scale.value)]; push(); drawMain(); drawWhatIf(); });

  const champion = () => (state.origin === "forward" ? ser.forward.champion : ser.holdout.find((h) => h.origin === state.origin)?.champion);

  function drawModelChips() {
    const ch = champion();
    const box = sel("#ws-models");
    const challengers = MODELS.filter((m) => m !== ch);
    box.innerHTML = challengers.map((m, i) => `<button type="button" class="chip" data-model="${m}" aria-pressed="${state.models.has(m)}"><span class="swatch" style="background:${CHALLENGER_COLORS[i]}"></span> ${MODEL_LABEL[m]}</button>`).join("");
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const m = b.dataset.model;
      if (state.models.has(m)) state.models.delete(m); else state.models.add(m);
      b.setAttribute("aria-pressed", String(state.models.has(m)));
      push();
      drawMain();
    }));
  }

  function drawMain() {
    const box = sel("#ws-chart");
    const ch = champion();
    const fwd = state.origin === "forward";
    const win = fwd ? ser.forward : ser.holdout.find((h) => h.origin === state.origin);
    if (!win || !win.rows.length) { showEmpty(box, "No forecast rows for this origin."); return; }
    const origin = win.origin;
    const startIdx = Math.max(0, (new Date(origin) - new Date(ser.history.start)) / 864e5 - CONTEXT_DAYS + 1);
    const endIdx = (new Date(origin) - new Date(ser.history.start)) / 864e5;
    const hx = [], hy = [];
    for (let i = startIdx; i <= endIdx && i < ser.history.sales.length; i++) { hx.push(addDays(ser.history.start, i)); hy.push(ser.history.sales[i]); }
    const x = win.rows.map((r) => r.date);
    const traces = [
      { x: hx, y: hy, type: "scatter", mode: "lines", name: "Actual", line: { color: C.actual, width: 1.6 } },
      ...bandTrace(x, win.rows.map((r) => r.p10), win.rows.map((r) => r.p90)),
    ];
    if (!fwd) traces.push({ x, y: win.rows.map((r) => r.actual), type: "scatter", mode: "lines+markers", name: "Actual (window)", line: { color: C.actual, width: 1.6 }, marker: { size: 4, color: C.actual }, showlegend: false });
    const challengers = MODELS.filter((m) => m !== ch);
    challengers.forEach((m, i) => {
      if (!state.models.has(m)) return;
      traces.push({ x, y: win.rows.map((r) => r.models?.[m] ?? null), type: "scatter", mode: "lines", name: MODEL_LABEL[m], line: { color: CHALLENGER_COLORS[i], width: 1.6, dash: CHALLENGER_DASH[i] } });
    });
    traces.push({ x, y: win.rows.map((r) => (fwd ? r.forecast : r.models?.[ch] ?? null)), type: "scatter", mode: "lines", name: `Champion · ${MODEL_LABEL[ch] || ch}`, line: { color: C.champion, width: 2.6 } });
    if (fwd && state.scale !== 0) {
      const w = ser.whatif.rows.find((r) => r.scale_pct === state.scale);
      if (w) traces.push({ x, y: w.daily, type: "scatter", mode: "lines", name: `What-if promo ${state.scale > 0 ? "+" : ""}${state.scale} %`, line: { color: C.Monitor, width: 2, dash: "dot" } });
    }
    plot(box, traces, {
      hovermode: "x unified", yaxis: { title: { text: "units / day" }, rangemode: "tozero" },
      shapes: [{ type: "line", x0: origin, x1: origin, yref: "paper", y0: 0, y1: 1, line: { color: C.muted, dash: "dot", width: 1 } }],
      annotations: [{ x: origin, yref: "paper", y: 1, text: "origin", showarrow: false, font: { color: C.muted, size: 10 }, xanchor: "right", yanchor: "top" }],
    }, `Store ${ser.store_nbr} ${famLabel(ser.family)}: actual sales and ${fwd ? "forward" : "holdout"} champion forecast from origin ${origin} with P10 to P90 band.`);
    sel("#ws-title").textContent = `Store ${ser.store_nbr} · ${famLabel(ser.family)}`;
    sel("#ws-sub").textContent = fwd ? "forward forecast" : `holdout origin ${fmtDate(origin)}`;
    sel("#ws-cap").textContent = fwd
      ? `Forward window ${fmtDate(x[0])} – ${fmtDate(x[x.length - 1])}: no actuals exist yet. Planned promotions come from the published promo calendar.`
      : "Holdout window: the model saw data only up to the origin; actuals in the window were never used for training or selection.";
  }

  function drawSide() {
    const box = sel("#ws-side");
    purgeCharts(box);
    const ch = champion();
    const fwd = state.origin === "forward";
    const win = fwd ? ser.forward : ser.holdout.find((h) => h.origin === state.origin);
    const t = ser.trust;
    let windowStats = "";
    if (fwd) {
      const tot = win.rows.reduce((a, r) => a + r.forecast, 0);
      const promo = win.rows.reduce((a, r) => a + (r.onpromotion || 0), 0);
      windowStats = `<div class="stat-row"><div><b>${fmtNum(tot)}</b>14-day champion total</div><div><b>${fmtNum(promo)}</b>planned promo item-days</div></div>`;
    } else {
      const ok = win.rows.filter((r) => r.actual !== null);
      const f = (r) => r.models?.[ch] ?? 0;
      const sa = ok.reduce((a, r) => a + r.actual, 0);
      const wape = sa > 0 ? (100 * ok.reduce((a, r) => a + Math.abs(r.actual - f(r)), 0)) / sa : null;
      const bias = sa > 0 ? (100 * ok.reduce((a, r) => a + (f(r) - r.actual), 0)) / sa : null;
      const cov = ok.length ? (100 * ok.filter((r) => r.actual >= r.p10 && r.actual <= r.p90).length) / ok.length : null;
      windowStats = `<div class="stat-row"><div><b>${fmtPct(wape)}</b>window WAPE</div><div><b>${fmtPct(bias, 1, true)}</b>window bias</div><div><b>${fmtPct(cov, 0)}</b>days inside band</div></div>
        <p class="small muted">Computed in the browser from this window's ${ok.length} open days.</p>`;
    }
    box.innerHTML = `
      <h2>Champion & trust</h2>
      <p class="small">Champion for ${esc(famLabel(ser.family))}: <b style="color:${C.champion}">${MODEL_LABEL[ch] || esc(ch)}</b></p>
      ${windowStats}
      <h3 style="margin-top:14px">Trust Index ${pill(t.band)}</h3>
      <div class="stat-row"><div><b>${fmtNum(t.score, 0)} / 100</b>latest score (ranking index, not a probability)</div></div>
      <div id="ws-comp" class="chart short" style="height:190px"></div>
      ${t.hard_flags.length ? `<ul class="clean">${t.hard_flags.map((f) => `<li class="flag">${esc(f)}</li>`).join("")}</ul>` : '<p class="small muted">No hard flags.</p>'}`;
    const comps = Object.keys(COMP_LABEL).filter((k) => k in t.components);
    plot(sel("#ws-comp"), [{
      y: comps.map((k) => COMP_LABEL[k]), x: comps.map((k) => t.components[k]), type: "bar", orientation: "h",
      marker: { color: comps.map((k) => (t.components[k] >= 80 ? C.Accept : t.components[k] >= 60 ? C.Monitor : C.Review)) },
      hovertemplate: "%{y}: %{x:.0f}<extra></extra>",
    }], { xaxis: { range: [0, 100], title: { text: "component score" } }, yaxis: { autorange: "reversed" }, margin: { l: 96, t: 4, b: 36 }, showlegend: false },
    `Trust Index components for store ${ser.store_nbr} ${famLabel(ser.family)}.`);
  }

  function drawWhatIf() {
    const wi = ser.whatif;
    const row = wi.rows.find((r) => r.scale_pct === state.scale);
    const base = wi.rows.find((r) => r.scale_pct === 0);
    sel("#ws-scale-val").textContent = `${state.scale > 0 ? "+" : ""}${state.scale} %`;
    scale.setAttribute("aria-valuetext", `${state.scale > 0 ? "plus " : state.scale < 0 ? "minus " : ""}${Math.abs(state.scale)} percent`);
    sel("#ws-wi-sub").textContent = `forward window from ${fmtDate(addDays(wi.origin, 1))} · precomputed with the frozen LightGBM model (known promo plan)`;
    const text = sel("#ws-wi-text");
    if (!row) {
      text.innerHTML = '<div class="empty">No what-if row for this change.</div>';
    } else {
      const extrap = row.extrapolation
        ? `<p class="note bad small" role="alert"><b>Extrapolation warning.</b> ${fmtNum(row.promo_total)} planned promo item-days (${fmtNum(row.promo_total / 14, 1)} promoted items per day on average) is beyond this series' historical range (95th percentile of 14-day means ${fmtNum(wi.hist_p95_promo, 1)} per day, highest single day ${fmtNum(wi.hist_max_promo)}). The model has not seen promotion this intense; treat the number as indicative only.</p>`
        : row.below_history
          ? `<p class="note warn small">This plan has fewer promotions than any 14-day window in this series' history, so the model is also outside the range it learned from; treat the number as indicative only.</p>`
          : "";
      text.innerHTML = `
        <p>Planned promotions: <b>${fmtNum(row.promo_total)}</b> item-days${base ? ` (baseline plan ${fmtNum(base.promo_total)})` : ""}.</p>
        <p class="verdict">Model-predicted change associated with this promotion plan: <b style="color:${C.champion}">${fmtPct(row.change_pct, 1, true)}</b> (${fmtNum(row.forecast_total)} units over 14 days).</p>
        <p class="small muted">This is what the model predicts when only the promotion input changes. It is an association learned from history, not a measured causal uplift.${state.origin !== "forward" ? " The grid applies to the forward window; switch the origin to Forward to see the daily line." : ""}</p>
        ${extrap}`;
    }
    const xs = wi.rows.map((r) => `${r.scale_pct > 0 ? "+" : ""}${r.scale_pct} %`);
    plot(sel("#ws-wi-chart"), [{
      x: xs, y: wi.rows.map((r) => r.forecast_total), type: "bar", name: "14-day forecast",
      marker: { color: wi.rows.map((r) => (r.scale_pct === state.scale ? C.champion : C.periwinkle)), pattern: { shape: wi.rows.map((r) => (r.extrapolation ? "x" : "")), fgcolor: C.panelDeep, size: 7 } },
      customdata: wi.rows.map((r) => [r.change_pct, r.extrapolation ? "extrapolation" : "within range"]),
      hovertemplate: "promo %{x}: %{y:,.0f} units (%{customdata[0]:+.1f} %, %{customdata[1]})<extra></extra>",
    }], { xaxis: { title: { text: "change to planned promotions" }, type: "category" }, yaxis: { title: { text: "14-day units" } }, showlegend: false },
    "Precomputed promotion what-if grid: 14-day forecast total for each change to planned promotions. Hatched bars are extrapolations.");
  }

  function drawAll() { drawModelChips(); drawMain(); drawSide(); drawWhatIf(); }
  await loadCurrent();
}
