import { load, esc, fmtNum, fmtPct, plot, C, viewHead, options, setParams, famLabel, showEmpty, pill, viewHref, slugOf, BANDS, purgeCharts, WINDOW_BAND_LABEL, fillWindowBandCaption, tableWrapAttrs, INDEX_NOTE } from "../core.js";

const COMP = ["accuracy", "interval", "stability", "agreement", "completeness"];
const COMP_LABEL = { accuracy: "Accuracy", interval: "Interval", stability: "Stability", agreement: "Agreement", completeness: "Completeness" };

export async function render(el, params) {
  const [trust, index, meta] = await Promise.all([load("trust.json"), load("series_index.json"), load("meta.json")]);
  const slugs = new Set(index.map((s) => s.slug));
  const all = [...trust.latest].sort((a, b) => a.priority_rank - b.priority_rank);
  const fams = meta.scope.families.filter((f) => all.some((r) => r.family === f));
  const state = {
    band: BANDS.includes(params.band) ? params.band : "ALL",
    family: fams.includes(params.family) ? params.family : "ALL",
    series: params.series || null,
  };
  const v = trust.validation;
  const counts = Object.fromEntries(BANDS.map((b) => [b, all.filter((r) => r.band === b).length]));

  el.innerHTML = `
  ${viewHead("05", "Exception Centre", "A trust inbox: which forecasts to look at first. Priority = (100 − trust score) × share of 14-day volume, so big, shaky series rise to the top (only the rank is published).", "prism")}
  <div class="controls" role="group" aria-label="Inbox filters">
    <div class="field"><span id="ex-band-lbl">Trust band</span><div class="chips" role="group" aria-labelledby="ex-band-lbl" id="ex-bands">
      ${[["ALL", `All (${all.length})`], ...BANDS.map((b) => [b, `${b} (${counts[b]})`])].map(([b, l]) => `<button type="button" class="chip" data-band="${b}" aria-pressed="${state.band === b}">${l}</button>`).join("")}
    </div></div>
    <label class="field"><span>Family</span><select id="ex-fam">${options([["ALL", "All families"], ...fams.map((f) => [f, famLabel(f)])], state.family)}</select></label>
  </div>
  <div class="grid g-2-1">
    <section class="card">
      <div class="card-head"><h2>Trust inbox</h2><span class="muted small" id="ex-count"></span></div>
      <div class="table-wrap scroll-y" id="ex-table" ${tableWrapAttrs("Trust inbox table, scrollable")}></div>
      <p class="caption">Trust score: a 0–100 ranking index built from five components (weights below). It is not a probability that the forecast is right. Bands: Accept 80–100, Monitor 60–79, Review below 60 or any hard flag.</p>
    </section>
    <section class="card" id="ex-detail"></section>
  </div>
  <section class="card">
    <div class="card-head"><h2>Does the Trust Index work?</h2><span class="muted small">holdout only · ${fmtNum(v.n_windows)} series-windows</span></div>
    <div class="grid g2">
      <div><div id="ex-val" class="chart"></div><p class="caption">Realised holdout WAPE of the champion forecast for windows the index placed in each band. If the index works, Review should show the highest error.</p></div>
      <div>
        <p class="verdict">${esc(v.verdict)}</p>
        <div class="stat-row">
          <div><b>${fmtNum(v.spearman, 2)}</b>Spearman, score vs realised WAPE</div>
          <div><b>${fmtNum(v.volatility_only_spearman, 2)}</b>volatility-only baseline</div>
          <div><b>${v.beats_volatility_baseline ? "Yes" : "No"}</b>beats the baseline</div>
        </div>
        <div class="stat-row">
          <div><b>${v.error_ratio_review_vs_accept === null ? "n/a" : fmtNum(v.error_ratio_review_vs_accept, 2) + "×"}</b>Review vs Accept WAPE</div>
          <div><b>${fmtPct(v.review_share_of_abs_error, 0)}</b>of absolute error in Review</div>
          <div><b>${fmtPct(v.review_share_of_windows, 0)}</b>of windows in Review</div>
        </div>
        <p class="small muted">Rank correlation between the trust score at the origin and the error realised afterwards, compared with ranking by volatility alone. Weights chosen on calibration windows: ${esc(trust.calibration_choice.chosen)}${trust.calibration_choice.grid_size ? ` (compared with default, equal and volatility-only weights and a ${fmtNum(trust.calibration_choice.grid_size)}-point weight grid)` : trust.calibration_choice.candidates.length > 1 ? ` (from ${trust.calibration_choice.candidates.length} candidates)` : ""}.</p>
        <h3>Component weights</h3>
        <div>${COMP.map((k) => `<span class="tag">${COMP_LABEL[k]} ${fmtNum(trust.weights[k] * (trust.weights[k] <= 1 ? 100 : 1), 0)} %</span>`).join("")}</div>
      </div>
    </div>
  </section>`;

  const sel = (id) => el.querySelector(id);
  const filtered = () => all.filter((r) => (state.band === "ALL" || r.band === state.band) && (state.family === "ALL" || r.family === state.family));
  const push = () => setParams({ band: state.band === "ALL" ? null : state.band, family: state.family === "ALL" ? null : state.family, series: state.series });

  sel("#ex-bands").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    state.band = b.dataset.band;
    sel("#ex-bands").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    push(); drawTable(); drawDetail();
  }));
  sel("#ex-fam").addEventListener("change", (e) => { state.family = e.target.value; push(); drawTable(); drawDetail(); });

  function drawTable() {
    const rows = filtered();
    sel("#ex-count").textContent = `${rows.length} series`;
    const box = sel("#ex-table");
    if (!rows.length) { box.innerHTML = '<div class="empty" role="status">No series match these filters.</div>'; return; }
    if (!rows.some((r) => r.series_id === state.series)) state.series = rows[0].series_id;
    box.innerHTML = `<table><thead><tr><th class="n">#</th><th>Series</th><th class="n">Score</th><th>Band</th><th>Hard flags</th><th class="n">14-day forecast index</th><th class="n">Priority rank</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="${r.series_id === state.series ? "is-selected" : ""}"><td class="n">${i + 1}</td><td>Store ${r.store_nbr} · ${esc(famLabel(r.family))}</td><td class="n">${fmtNum(r.score, 0)}</td><td>${pill(r.band)}</td>
        <td>${r.hard_flags.length ? `<span class="flag">${r.hard_flags.length}</span>` : '<span class="muted">–</span>'}</td><td class="n">${fmtNum(r.forecast_14d_index, 1)}</td><td class="n">${fmtNum(r.priority_rank)}</td>
        <td><button type="button" class="btn sm" data-sid="${esc(r.series_id)}" aria-label="Inspect store ${r.store_nbr} ${esc(famLabel(r.family))}">Inspect</button></td></tr>`).join("")}
    </tbody></table>`;
    box.querySelectorAll("button[data-sid]").forEach((b) => b.addEventListener("click", () => { state.series = b.dataset.sid; push(); drawTable(); drawDetail(); sel("#ex-detail h2")?.focus(); }));
  }

  function drawDetail() {
    const box = sel("#ex-detail");
    purgeCharts(box);
    const r = all.find((x) => x.series_id === state.series && filtered().includes(x));
    if (!r) { box.innerHTML = '<h2 tabindex="-1">Series detail</h2><div class="empty">Select a series in the inbox.</div>'; return; }
    const slug = slugOf(r.store_nbr, r.family);
    box.innerHTML = `
      <h2 tabindex="-1">Store ${r.store_nbr} · ${esc(famLabel(r.family))}</h2>
      <p>${pill(r.band)} <b class="num">${fmtNum(r.score, 0)}</b> / 100</p>
      <div id="ex-comp" class="chart short"></div>
      <h3>Hard-flag reasons</h3>
      ${r.hard_flags.length ? `<ul class="clean">${r.hard_flags.map((f) => `<li class="flag">${esc(f)}</li>`).join("")}</ul>` : '<p class="small muted">None. The band comes from the score alone.</p>'}
      <p class="small">14-day forecast index <b class="num">${fmtNum(r.forecast_14d_index, 1)}</b>; ${WINDOW_BAND_LABEL}: <span class="num">${fmtNum(r.p10_14d_index, 1)} – ${fmtNum(r.p90_14d_index, 1)}</span> (mean day over the 14 days; ${esc(INDEX_NOTE)})</p>
      <p class="caption" id="ex-wb" hidden></p>
      ${slugs.has(slug) ? `<a class="btn sm" href="${viewHref("workspace", { series: slug })}">Open in Forecast Workspace →</a>` : '<p class="small muted">No per-series file published for this series.</p>'}`;
    fillWindowBandCaption(box.querySelector("#ex-wb"));
    plot(box.querySelector("#ex-comp"), [{
      y: COMP.map((k) => COMP_LABEL[k]), x: COMP.map((k) => r.components[k]), type: "bar", orientation: "h", name: "Component score",
      marker: { color: COMP.map((k) => (r.components[k] >= 80 ? C.Accept : r.components[k] >= 60 ? C.Monitor : C.Review)) },
      customdata: COMP.map((k) => trust.weights[k]), hovertemplate: "%{y}: %{x:.0f} (weight %{customdata:.2f})<extra></extra>",
      text: COMP.map((k) => fmtNum(r.components[k], 0)), textposition: "outside", textfont: { color: C.text }, cliponaxis: false,
    }], { xaxis: { range: [0, 110], title: { text: "component score (0–100)" } }, yaxis: { autorange: "reversed" }, margin: { l: 96, t: 4 }, showlegend: false },
    `Trust components for store ${r.store_nbr} ${famLabel(r.family)}: ${COMP.map((k) => `${COMP_LABEL[k]} ${Math.round(r.components[k])}`).join(", ")}.`);
  }

  function drawValidation() {
    const box = sel("#ex-val");
    const rows = BANDS.map((b) => v.wape_by_band.find((x) => x.band === b)).filter(Boolean);
    if (!rows.length) { showEmpty(box, "No validation results."); return; }
    plot(box, [{
      x: rows.map((r) => r.band), y: rows.map((r) => r.wape), type: "bar", name: "Holdout WAPE",
      marker: { color: rows.map((r) => C[r.band]) }, text: rows.map((r) => `${fmtNum(r.wape, 1)} % · n=${r.n}`), textposition: "outside",
      textfont: { color: C.text }, cliponaxis: false, hovertemplate: "%{x}: %{y:.1f} % WAPE<extra></extra>",
    }], { yaxis: { title: { text: "realised WAPE (%)" }, ticksuffix: " %", rangemode: "tozero" }, showlegend: false, margin: { t: 24 } },
    `Holdout WAPE by trust band: ${rows.map((r) => `${r.band} ${r.wape === null ? "n/a" : r.wape.toFixed(1) + "%"}`).join(", ")}.`);
  }

  drawTable(); drawDetail(); drawValidation();
}
