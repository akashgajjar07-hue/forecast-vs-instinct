import { load, esc, fmtNum, fmtPct, fmtSigned, fmtDate, fmtCompact, plot, bandTrace, C, pill, shape, viewHref, slugOf, famLabel, MODEL_LABEL } from "../core.js";

export async function render(el) {
  const [meta, ov, index] = await Promise.all([load("meta.json"), load("overview.json"), load("series_index.json")]);
  const slugs = new Set(index.map((s) => s.slug));
  const k = ov.kpis;
  const cov = k.interval_coverage;
  const covGap = cov.value - cov.target;
  const champs = Object.entries(ov.ml_wins.champions || {});

  el.innerHTML = `
  <section class="hero" aria-labelledby="hero-title">
    ${shape("torus", "s-torus")}${shape("coil", "s-coil")}${shape("prism", "s-prism")}${shape("orb", "s-orb")}
    <h1 id="hero-title" class="hero-title" tabindex="-1">FORECAST <span class="vs">vs</span> INSTINCT</h1>
    <p class="hero-tag">Beyond predicting demand. Knowing when to trust the prediction.</p>
    <p class="hero-headline">${esc(meta.headline)}</p>
    <div class="hero-nums">
      ${meta.hero.map((h) => `<div class="hero-num"><div class="hero-num-value">${esc(h.value)}</div><div class="hero-num-label">${esc(h.label)}</div><div class="hero-num-detail">${esc(h.detail)}</div></div>`).join("")}
    </div>
  </section>

  <section class="kpis" aria-label="Key indicators">
    <div class="card kpi"><div class="kpi-label">14-day forecast</div><div class="kpi-value">${fmtCompact(k.forecast_14d_total.value)}</div>
      <div class="kpi-detail">units, ${fmtDate(k.forecast_14d_total.start)} – ${fmtDate(k.forecast_14d_total.end)} (no actuals yet)</div></div>
    <div class="card kpi"><div class="kpi-label">Holdout WAPE</div><div class="kpi-value">${fmtPct(k.holdout_wape.value)}</div>
      <div class="kpi-detail">${esc(k.holdout_wape.model)}</div></div>
    <div class="card kpi"><div class="kpi-label">Holdout bias</div><div class="kpi-value">${fmtPct(k.holdout_bias.value, 1, true)}</div>
      <div class="kpi-detail">${k.holdout_bias.value > 0 ? "over-forecast" : k.holdout_bias.value < 0 ? "under-forecast" : "no bias"} · ${esc(k.holdout_bias.model)}</div></div>
    <div class="card kpi"><div class="kpi-label">Review exceptions</div><div class="kpi-value">${fmtNum(k.review_exceptions.value)} <span class="muted small">of ${fmtNum(k.review_exceptions.of)}</span></div>
      <div class="kpi-detail">series in the Review band · <a href="${viewHref("exceptions", { band: "Review" })}">open inbox</a></div></div>
    <div class="card kpi"><div class="kpi-label">Interval coverage</div><div class="kpi-value">${fmtPct(cov.value)}</div>
      <div class="kpi-detail">target ${fmtPct(cov.target)} · ${fmtSigned(covGap)} pp</div>
      <div class="meter" role="img" aria-label="Coverage ${fmtPct(cov.value)} against target ${fmtPct(cov.target)}"><span style="width:${Math.max(0, Math.min(100, cov.value))}%"></span><i style="left:${Math.max(0, Math.min(100, cov.target))}%"></i></div></div>
  </section>

  <div class="grid g-2-1">
    <section class="card">
      <div class="card-head"><h2>Chain outlook</h2><span class="muted small">all ${fmtNum(meta.scope.n_series)} series, daily units</span></div>
      <div id="ov-outlook" class="chart tall"></div>
      <p class="caption">Actuals (white) to ${fmtDate(meta.data_window.last_actual)}, champion forecast (mint) with P10–P90 band. Band: ${esc(ov.outlook.band_method)}</p>
    </section>
    <section class="card">
      <h2>Findings</h2>
      <ul class="clean">${ov.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
      <h3 style="margin-top:14px">Champion per family</h3>
      <p class="small muted">Chosen on selection backtests only: LightGBM is champion in ${fmtNum(champs.filter(([, m]) => m === "lightgbm").length)} of ${fmtNum(champs.length)} families. On the untouched holdout, LightGBM had the lowest WAPE of the five models in ${fmtNum(ov.ml_wins.families_won)} of ${fmtNum(ov.ml_wins.n_families)} families.</p>
      <div>${champs.map(([f, m]) => `<span class="tag">${esc(famLabel(f))}: <b>${esc(MODEL_LABEL[m] || m)}</b></span>`).join("")}</div>
    </section>
  </div>

  <section class="card">
    <div class="card-head"><h2>Top exceptions</h2><a href="#/exceptions">All exceptions →</a></div>
    ${ov.top_exceptions.length ? `<div class="table-wrap"><table>
      <thead><tr><th>Series</th><th class="n">Trust score</th><th>Band</th><th>Reason</th><th></th></tr></thead>
      <tbody>${ov.top_exceptions.map((x) => {
        const slug = slugOf(x.store_nbr, x.family);
        const link = slugs.has(slug) ? `<a href="${viewHref("workspace", { series: slug })}">Workspace</a>` : `<a href="${viewHref("exceptions", { series: x.series_id })}">Details</a>`;
        return `<tr><td>Store ${x.store_nbr} · ${esc(famLabel(x.family))}</td><td class="n">${fmtNum(x.trust_score, 0)}</td><td>${pill(x.band)}</td><td>${esc(x.reason)}</td><td>${link}</td></tr>`;
      }).join("")}</tbody></table></div>` : '<div class="empty">No exceptions in the latest run.</div>'}
    <p class="caption">Trust score is a 0–100 ranking index of how far to rely on a forecast, not a probability.</p>
  </section>`;

  const h = ov.outlook.history;
  const f = ov.outlook.forecast;
  const fx = f.map((r) => r.date);
  plot(
    el.querySelector("#ov-outlook"),
    [
      { x: h.map((r) => r.date), y: h.map((r) => r.actual), type: "scatter", mode: "lines", name: "Actual", line: { color: C.actual, width: 1.6 }, connectgaps: false },
      ...bandTrace(fx, f.map((r) => r.p10), f.map((r) => r.p90)),
      { x: fx, y: f.map((r) => r.forecast), type: "scatter", mode: "lines", name: "Champion forecast", line: { color: C.champion, width: 2.4 } },
    ],
    { hovermode: "x unified", yaxis: { title: { text: "units / day" }, rangemode: "tozero" } },
    `Chain outlook: daily actual sales to ${meta.data_window.last_actual} and 14-day champion forecast with P10 to P90 band.`
  );
}
