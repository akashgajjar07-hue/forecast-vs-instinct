import { load, loadSeries, esc, fmtNum, fmtPct, fmtDate, addDays, plot, bandTrace, C, viewHead, options, setParams, famLabel, showEmpty, showLoading, pill, MODELS, MODEL_LABEL, MODEL_COLOR, MODEL_DASH, purgeCharts, INDEX_AXIS, INDEX_NOTE, PROMO_INDEX_NOTE } from "../core.js";

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
  ${viewHead("04", "Forecast Workspace", "One series at a time: actuals, the champion forecast with its P10–P90 band, challengers on demand, and a promotion what-if from the precomputed grid. All values are indexed (training-period mean = 100).", "torus")}
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
        <p class="caption" style="margin-top:6px">Model association, not causal uplift — the data has no price or discount depth</p>
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
    originSel.innerHTML = options([["forward", `Forward · ${fmtDate(addDays(ser.forward.origin, 1))} (no actuals)`], ...origins.map((o) => [o, `Holdout · origin ${fmtDate(o)}`])], state.origin);
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
    if (state.origin !== "forward") {
      // Holdout windows publish the champion only: per-series challenger lines next to the Arena's holdout accuracy
      // slices would let unit volumes be reconstructed (Kaggle competition rules).
      box.innerHTML = '<p class="small muted" style="margin:0">Challenger lines are shown for the forward window only. Compare all five models on the holdout in the Model Arena.</p>';
      return;
    }
    const challengers = MODELS.filter((m) => m !== ch);
    box.innerHTML = challengers.map((m) => `<button type="button" class="chip" data-model="${m}" aria-pressed="${state.models.has(m)}"><span class="swatch" style="background:${MODEL_COLOR[m]}"></span> ${MODEL_LABEL[m]}</button>`).join("");
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
    for (let i = startIdx; i <= endIdx && i < ser.history.index.length; i++) { hx.push(addDays(ser.history.start, i)); hy.push(ser.history.index[i]); }
    const x = win.rows.map((r) => r.date);
    const traces = [
      { x: hx, y: hy, type: "scatter", mode: "lines", name: "Actual", line: { color: C.actual, width: 1.6 } },
      ...bandTrace(x, win.rows.map((r) => r.p10_index), win.rows.map((r) => r.p90_index)),
    ];
    if (!fwd) traces.push({ x, y: win.rows.map((r) => r.actual_index), type: "scatter", mode: "lines+markers", name: "Actual (window)", line: { color: C.actual, width: 1.6 }, marker: { size: 4, color: C.actual }, showlegend: false });
    const challengers = MODELS.filter((m) => m !== ch);
    challengers.forEach((m) => {
      if (!fwd || !state.models.has(m)) return;
      // Same colour and dash per model as in the Model Arena.
      traces.push({ x, y: win.rows.map((r) => r.models_index?.[m] ?? null), type: "scatter", mode: "lines", name: MODEL_LABEL[m], line: { color: MODEL_COLOR[m] || C.periwinkle, width: 1.6, dash: MODEL_DASH[m] || "dash" } });
    });
    traces.push({ x, y: win.rows.map((r) => r.forecast_index ?? null), type: "scatter", mode: "lines", name: `Champion · ${MODEL_LABEL[ch] || ch}`, line: { color: MODEL_COLOR[ch] || C.champion, width: 2.6 } });
    if (fwd && state.scale !== 0) {
      const w = ser.whatif.rows.find((r) => r.scale_pct === state.scale);
      if (w) traces.push({ x, y: w.daily_index, type: "scatter", mode: "lines", name: `What-if promo ${state.scale > 0 ? "+" : ""}${state.scale} %`, line: { color: C.Monitor, width: 2, dash: "dot" } });
    }
    plot(box, traces, {
      hovermode: "x unified", yaxis: { title: { text: INDEX_AXIS }, rangemode: "tozero" },
      shapes: [{ type: "line", x0: origin, x1: origin, yref: "paper", y0: 0, y1: 1, line: { color: C.muted, dash: "dot", width: 1 } }],
      annotations: [{ x: origin, yref: "paper", y: 1, text: "origin", showarrow: false, font: { color: C.muted, size: 10 }, xanchor: "right", yanchor: "top" }],
    }, `Store ${ser.store_nbr} ${famLabel(ser.family)}: indexed actual sales and ${fwd ? "forward" : "holdout"} champion forecast from origin ${origin} with P10 to P90 band (training-period mean = 100).`);
    sel("#ws-title").textContent = `Store ${ser.store_nbr} · ${famLabel(ser.family)}`;
    sel("#ws-sub").textContent = fwd ? "forward forecast" : `holdout origin ${fmtDate(origin)}`;
    sel("#ws-cap").textContent = fwd
      ? `Forward window ${fmtDate(x[0])} – ${fmtDate(x[x.length - 1])}: no actuals exist yet. Planned promotions come from the published promo calendar. ${INDEX_NOTE}`
      : `Holdout window: the model saw data only up to the origin; actuals in the window were never used for training or selection. ${INDEX_NOTE}`;
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
      const mean = win.rows.reduce((a, r) => a + r.forecast_index, 0) / win.rows.length;
      const pr = win.rows.filter((r) => r.promo_index !== null && r.promo_index !== undefined);
      const promo = pr.length ? pr.reduce((a, r) => a + r.promo_index, 0) / pr.length : null;
      windowStats = `<div class="stat-row"><div><b>${fmtNum(mean, 1)}</b>14-day champion forecast, index (mean day)</div><div><b>${fmtNum(promo, 0)}</b>planned promotions, promo index (14-day mean)</div></div>`;
    } else {
      // WAPE, bias and coverage are scale-free, so they are the same on the index scale as in units.
      const ok = win.rows.filter((r) => r.actual_index !== null);
      const f = (r) => r.forecast_index ?? 0;
      const sa = ok.reduce((a, r) => a + r.actual_index, 0);
      const wape = sa > 0 ? (100 * ok.reduce((a, r) => a + Math.abs(r.actual_index - f(r)), 0)) / sa : null;
      const bias = sa > 0 ? (100 * ok.reduce((a, r) => a + (f(r) - r.actual_index), 0)) / sa : null;
      const cov = ok.length ? (100 * ok.filter((r) => r.actual_index >= r.p10_index && r.actual_index <= r.p90_index).length) / ok.length : null;
      windowStats = `<div class="stat-row"><div><b>${fmtPct(wape)}</b>window WAPE</div><div><b>${fmtPct(bias, 1, true)}</b>window bias</div><div><b>${fmtPct(cov, 0)}</b>days inside band</div></div>
        <p class="small muted">Computed in the browser from this window's ${ok.length} open days.</p>`;
    }
    box.innerHTML = `
      <h2>Champion & trust</h2>
      <p class="small">Champion for ${esc(famLabel(ser.family))}: <b style="color:${MODEL_COLOR[ch] || C.champion}">${MODEL_LABEL[ch] || esc(ch)}</b></p>
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
      text: comps.map((k) => fmtNum(t.components[k], 0)), textposition: "outside", textfont: { color: C.text }, cliponaxis: false,
    }], { xaxis: { range: [0, 110], title: { text: "component score (0–100)" } }, yaxis: { autorange: "reversed" }, margin: { l: 96, t: 4, b: 36 }, showlegend: false },
    `Trust Index components for store ${ser.store_nbr} ${famLabel(ser.family)}: ${comps.map((k) => `${COMP_LABEL[k]} ${Math.round(t.components[k])}`).join(", ")}.`);
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
        ? `<p class="note bad small" role="alert"><b>Extrapolation warning.</b> A planned promotion intensity of promo index ${fmtNum(row.promo_index, 0)} (14-day mean) is beyond this series' historical range (95th percentile of 14-day means: promo index ${fmtNum(wi.hist_p95_promo_index, 0)}; highest single day: ${fmtNum(wi.hist_max_promo_index, 0)}). The model has not seen promotion this intense; treat the number as indicative only.</p>`
        : row.below_history
          ? `<p class="note warn small">This plan has fewer promotions than any 14-day window in this series' history, so the model is also outside the range it learned from; treat the number as indicative only.</p>`
          : "";
      text.innerHTML = `
        <p>Planned promotions: promo index <b>${fmtNum(row.promo_index, 0)}</b> (14-day mean)${base ? ` (baseline plan ${fmtNum(base.promo_index, 0)})` : ""}.</p>
        <p class="verdict">Model-predicted change associated with this promotion plan: <b style="color:${C.champion}">${fmtPct(row.change_pct, 1, true)}</b> (14-day forecast index ${fmtNum(row.forecast_index, 1)}).</p>
        <p class="small muted">This is what the model predicts when only the promotion input changes. It is an association learned from history, not a measured causal uplift. ${esc(PROMO_INDEX_NOTE)}${state.origin !== "forward" ? " The grid applies to the forward window; switch the origin to Forward to see the daily line." : ""}</p>
        ${extrap}`;
    }
    const xs = wi.rows.map((r) => `${r.scale_pct > 0 ? "+" : ""}${r.scale_pct} %`);
    plot(sel("#ws-wi-chart"), [{
      x: xs, y: wi.rows.map((r) => r.forecast_index), type: "bar", name: "14-day forecast index",
      marker: { color: wi.rows.map((r) => (r.scale_pct === state.scale ? C.champion : C.periwinkle)), pattern: { shape: wi.rows.map((r) => (r.extrapolation ? "x" : "")), fgcolor: C.panelDeep, size: 7 } },
      customdata: wi.rows.map((r) => [r.change_pct, r.extrapolation ? "extrapolation" : "within range"]),
      hovertemplate: "promo %{x}: index %{y:,.1f} (%{customdata[0]:+.1f} %, %{customdata[1]})<extra></extra>",
    }], { xaxis: { title: { text: "change to planned promotions" }, type: "category" }, yaxis: { title: { text: INDEX_AXIS } }, showlegend: false },
    "Precomputed promotion what-if grid: 14-day forecast index (mean day, training-period mean = 100) for each change to planned promotions. Hatched bars are extrapolations.");
  }

  function drawAll() { drawModelChips(); drawMain(); drawSide(); drawWhatIf(); }
  await loadCurrent();
}
