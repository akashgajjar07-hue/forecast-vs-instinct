import { load, loadSeries, esc, fmtNum, fmtPct, plot, C, viewHead, options, setParams, slugOf, famLabel, showEmpty, showLoading, addDays } from "../core.js";

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export async function render(el, params, ctx) {
  const [meta, dem, index] = await Promise.all([load("meta.json"), load("demand.json"), load("series_index.json")]);
  const families = meta.scope.families;
  const stores = meta.scope.stores;
  const state = {
    family: families.includes(params.family) ? params.family : "ALL",
    store: stores.some((s) => String(s.store_nbr) === params.store) ? params.store : "ALL",
  };

  el.innerHTML = `
  ${viewHead("02", "Demand Explorer", "Observed sales patterns across 12 stores and 8 families: level, weekly rhythm, volatility, promotions and holidays.", "prism")}
  <div class="controls" role="group" aria-label="Explorer filters">
    <label class="field"><span>Family</span><select id="ex-family">${options([["ALL", "All families"], ...families.map((f) => [f, famLabel(f)])], state.family)}</select></label>
    <label class="field"><span>Store</span><select id="ex-store">${options([["ALL", "All stores"], ...stores.map((s) => [s.store_nbr, `Store ${s.store_nbr} · ${s.city} (type ${s.type})`])], state.store)}</select></label>
    <p class="small muted" style="margin:0;flex:1 1 220px">Store-level history comes from per-series files and needs one family selected.</p>
  </div>
  <section class="card"><div class="card-head"><h2 id="ex-weekly-title">Weekly sales</h2><span class="muted small">observed units per week</span></div><div id="ex-weekly" class="chart"></div></section>
  <div class="grid g2">
    <section class="card"><h2>Day-of-week profile</h2><div id="ex-dow" class="chart short"></div><p class="caption">Index: mean sales on that weekday relative to the series mean (1.0 = average day). Monday first.</p></section>
    <section class="card"><h2>Holiday effect</h2><div id="ex-hol" class="chart short"></div><p class="caption">Mean sales on national holidays vs normal days, per family. Observed difference, not a causal estimate.</p></section>
  </div>
  <section class="card"><div class="card-head"><h2>Volatility map</h2><span class="muted small">coefficient of variation of daily sales, store × family</span></div><div id="ex-vol" class="chart tall"></div><p class="caption">Higher CV = harder to forecast. The selected store or family is outlined.</p></section>
  <div class="grid g2">
    <section class="card"><h2 id="ex-pf-title">Promotion frequency</h2><div id="ex-pf" class="chart short"></div><p class="caption">Share of open days with at least one item on promotion.</p></section>
    <section class="card"><h2 id="ex-pb-title">Sales by promotion intensity</h2><div id="ex-pb" class="chart short"></div>
      <p class="note warn small" style="margin-top:8px"><b>Association, not causation.</b> Promotions are planned for periods that may already sell more (seasons, paydays). These bins describe what was observed together.</p></section>
  </div>
  ${dem.notes && dem.notes.length ? `<section class="card"><h2>Notes</h2><ul class="clean">${dem.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></section>` : ""}`;

  const fSel = el.querySelector("#ex-family");
  const sSel = el.querySelector("#ex-store");
  const onChange = () => {
    state.family = fSel.value;
    state.store = sSel.value;
    setParams({ family: state.family === "ALL" ? null : state.family, store: state.store === "ALL" ? null : state.store });
    draw();
  };
  fSel.addEventListener("change", onChange);
  sSel.addEventListener("change", onChange);

  async function drawWeekly() {
    const box = el.querySelector("#ex-weekly");
    const title = el.querySelector("#ex-weekly-title");
    const wk = dem.weekly;
    if (state.store === "ALL") {
      let y;
      if (state.family === "ALL") {
        y = wk.week_start.map((_, i) => {
          let t = 0, any = false;
          for (const f of families) { const v = wk.by_family[f]?.[i]; if (v !== null && v !== undefined) { t += v; any = true; } }
          return any ? t : null;
        });
        title.textContent = "Weekly sales · all families, all stores";
      } else {
        y = wk.by_family[state.family] || [];
        title.textContent = `Weekly sales · ${famLabel(state.family)}, all stores`;
      }
      if (!y.length) { showEmpty(box, "No weekly data for this selection."); return; }
      plot(box, [{ x: wk.week_start, y, type: "scatter", mode: "lines", name: "Weekly sales", line: { color: C.actual, width: 1.6 } }],
        { yaxis: { rangemode: "tozero", title: { text: "units / week" } }, hovermode: "x unified" }, `${title.textContent}: weekly observed sales line chart.`);
      return;
    }
    if (state.family === "ALL") {
      title.textContent = `Weekly sales · store ${state.store}`;
      showEmpty(box, "Choose a family to see one store's weekly history.");
      return;
    }
    const slug = slugOf(Number(state.store), state.family);
    title.textContent = `Weekly sales · ${famLabel(state.family)}, store ${state.store}`;
    if (!index.some((s) => s.slug === slug)) { showEmpty(box, `No series file published for store ${state.store} × ${famLabel(state.family)}.`); return; }
    showLoading(box, "Loading series…");
    const ser = await loadSeries(slug);
    if (!ctx.isCurrent() || fSel.value !== state.family || sSel.value !== state.store) return;
    const x = [], y = [];
    const s = ser.history.sales;
    for (let i = 0; i + 7 <= s.length; i += 7) {
      const wkVals = s.slice(i, i + 7).filter((v) => v !== null);
      x.push(addDays(ser.history.start, i));
      y.push(wkVals.length ? (wkVals.reduce((a, b) => a + b, 0) * 7) / wkVals.length : null);
    }
    plot(box, [{ x, y, type: "scatter", mode: "lines", name: "Weekly sales", line: { color: C.actual, width: 1.6 } }],
      { yaxis: { rangemode: "tozero", title: { text: "units / week" } }, hovermode: "x unified" }, `${title.textContent}: weekly observed sales line chart.`);
  }

  function drawDow() {
    const box = el.querySelector("#ex-dow");
    const prof = dem.dow_profile;
    let y, name;
    if (state.family === "ALL") {
      const fams = Object.keys(prof);
      if (!fams.length) { showEmpty(box, "No day-of-week profile available."); return; }
      y = DOW.map((_, i) => fams.reduce((a, f) => a + prof[f][i], 0) / fams.length);
      name = "Average across families";
    } else {
      y = prof[state.family];
      name = famLabel(state.family);
      if (!y) { showEmpty(box, "No profile for this family."); return; }
    }
    plot(box, [{ x: DOW, y, type: "bar", name, marker: { color: C.periwinkle }, hovertemplate: "%{x}: %{y:.2f}<extra></extra>" }],
      { yaxis: { title: { text: "index" } }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.text2, dash: "dot", width: 1 } }] },
      `Day-of-week sales index, ${name}.`);
  }

  function drawHoliday() {
    const box = el.querySelector("#ex-hol");
    const rows = dem.holiday_effect;
    if (!rows.length) { showEmpty(box, "No holiday effect data."); return; }
    plot(box, [{
      x: rows.map((r) => famLabel(r.family)), y: rows.map((r) => r.change_pct), type: "bar", name: "Change on holidays",
      marker: { color: rows.map((r) => (r.change_pct >= 0 ? C.champion : C.orchid)), opacity: rows.map((r) => (state.family === "ALL" || r.family === state.family ? 1 : 0.3)) },
      customdata: rows.map((r) => [r.holiday_mean, r.normal_mean, r.n_holiday_days]),
      hovertemplate: "%{x}<br>%{y:+.1f} %<br>holiday mean %{customdata[0]:,.0f} vs normal %{customdata[1]:,.0f}<br>%{customdata[2]} holiday days<extra></extra>",
    }], { yaxis: { title: { text: "% vs normal days" }, ticksuffix: " %" }, xaxis: { tickangle: -30 } }, "Holiday effect: percent change in mean sales on national holidays by family.");
  }

  function drawVol() {
    const box = el.querySelector("#ex-vol");
    const v = dem.volatility;
    if (!v.length) { showEmpty(box, "No volatility data."); return; }
    const sList = [...new Set(v.map((r) => r.store_nbr))].sort((a, b) => a - b);
    const fList = families.filter((f) => v.some((r) => r.family === f));
    const z = sList.map((s) => fList.map((f) => v.find((r) => r.store_nbr === s && r.family === f)?.cv ?? null));
    const ylab = sList.map((s) => `Store ${s}`);
    const xlab = fList.map(famLabel);
    const shapes = [];
    if (state.family !== "ALL" && fList.includes(state.family)) {
      const i = fList.indexOf(state.family);
      shapes.push({ type: "rect", x0: i - 0.5, x1: i + 0.5, y0: -0.5, y1: sList.length - 0.5, line: { color: C.text, width: 2 } });
    }
    if (state.store !== "ALL") {
      const j = sList.indexOf(Number(state.store));
      if (j >= 0) shapes.push({ type: "rect", x0: -0.5, x1: fList.length - 0.5, y0: j - 0.5, y1: j + 0.5, line: { color: C.text, width: 2 } });
    }
    plot(box, [{
      z, x: xlab, y: ylab, type: "heatmap", colorscale: [[0, "#0B2F66"], [0.5, "#9090F0"], [1, "#60F0D8"]],
      hovertemplate: "%{y} · %{x}<br>CV %{z:.2f}<extra></extra>", colorbar: { title: { text: "CV", font: { color: C.text2 } }, tickfont: { color: C.text2 }, outlinewidth: 0, thickness: 12 },
      xgap: 2, ygap: 2,
    }], { xaxis: { tickangle: -30, gridcolor: "rgba(0,0,0,0)" }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" }, margin: { l: 70, b: 70 }, shapes },
    "Volatility heatmap: coefficient of variation of daily sales for each store and family.");
  }

  function drawPromoFreq() {
    const box = el.querySelector("#ex-pf");
    const title = el.querySelector("#ex-pf-title");
    const pf = dem.promo_frequency;
    if (!pf.length) { showEmpty(box, "No promotion frequency data."); return; }
    let x, y;
    if (state.store !== "ALL") {
      const rows = pf.filter((r) => r.store_nbr === Number(state.store));
      x = rows.map((r) => famLabel(r.family));
      y = rows.map((r) => r.promo_day_share * 100);
      title.textContent = `Promotion frequency · store ${state.store}, by family`;
    } else {
      const sList = [...new Set(pf.map((r) => r.store_nbr))].sort((a, b) => a - b);
      x = sList.map((s) => `Store ${s}`);
      y = sList.map((s) => {
        const rows = pf.filter((r) => r.store_nbr === s && (state.family === "ALL" || r.family === state.family));
        return rows.length ? (100 * rows.reduce((a, r) => a + r.promo_day_share, 0)) / rows.length : null;
      });
      title.textContent = `Promotion frequency · ${state.family === "ALL" ? "mean of all families" : famLabel(state.family)}, by store`;
    }
    if (!x.length) { showEmpty(box, "No promotion data for this selection."); return; }
    plot(box, [{ x, y, type: "bar", name: "Promo days", marker: { color: C.skyData }, hovertemplate: "%{x}: %{y:.0f} %<extra></extra>" }],
      { yaxis: { title: { text: "% of open days" }, range: [0, 100], ticksuffix: " %" }, xaxis: { tickangle: -30 } }, `${title.textContent}, bar chart.`);
  }

  function drawPromoBins() {
    const box = el.querySelector("#ex-pb");
    const title = el.querySelector("#ex-pb-title");
    const pb = dem.promo_bins;
    const fams = Object.keys(pb);
    if (!fams.length) { showEmpty(box, "No promotion bin data."); return; }
    if (state.family === "ALL") {
      const bins = pb[fams[0]].map((b) => b.bin);
      const y = bins.map((_, i) => {
        const vals = fams.map((f) => (pb[f][i] && pb[f][0].mean_sales ? (100 * pb[f][i].mean_sales) / pb[f][0].mean_sales : null)).filter((v) => v !== null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      });
      title.textContent = "Sales by promotion intensity · index across families";
      plot(box, [{ x: bins, y, type: "bar", name: "Mean sales index", marker: { color: C.periwinkle }, hovertemplate: "%{x} items on promo: %{y:.0f}<extra></extra>" }],
        { xaxis: { title: { text: "items on promotion that day" }, type: "category" }, yaxis: { title: { text: `index (bin "${bins[0]}" = 100)` } } },
        "Mean sales index by number of items on promotion, averaged across families.");
      return;
    }
    const rows = pb[state.family] || [];
    title.textContent = `Sales by promotion intensity · ${famLabel(state.family)}`;
    if (!rows.length) { showEmpty(box, "No promotion bins for this family."); return; }
    const x = rows.map((r) => r.bin);
    plot(box, [
      { x, y: rows.map((r) => r.mean_sales), type: "bar", name: "Mean", marker: { color: C.periwinkle }, customdata: rows.map((r) => r.n_days), hovertemplate: "%{x}: mean %{y:,.0f} (%{customdata} days)<extra></extra>" },
      { x, y: rows.map((r) => r.median_sales), type: "bar", name: "Median", marker: { color: C.orchid }, hovertemplate: "%{x}: median %{y:,.0f}<extra></extra>" },
    ], { barmode: "group", xaxis: { title: { text: "items on promotion that day" }, type: "category" }, yaxis: { title: { text: "units / day" } } },
    `Mean and median daily sales by promotion intensity for ${famLabel(state.family)}.`);
  }

  function draw() {
    drawDow(); drawHoliday(); drawVol(); drawPromoFreq(); drawPromoBins();
    return drawWeekly();
  }
  await draw();
}
