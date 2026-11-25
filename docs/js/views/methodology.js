import { load, esc, fmtDate, fmtNum, fmtPct, addDays, dayDiff, viewHead, C, REPO, tableWrapAttrs, INDEX_NOTE, PROMO_INDEX_NOTE } from "../core.js";

export const DATA_LICENCE = "Built on the Kaggle Store Sales – Time Series Forecasting competition data (Corporación Favorita). Raw competition data is not redistributed; the site shows only aggregated/indexed values and model outputs. Reproduce by downloading the data from Kaggle after accepting the competition rules.";

// OWNER: paste your LinkedIn profile URL here (left empty on purpose; the card shows a placeholder until filled).
const LINKEDIN_URL = "";
const GITHUB_URL = "https://github.com/akashgajjar07-hue";

const STAGE_STYLE = {
  selection: { label: "Selection", color: () => C.periwinkle, note: "pick the champion per family" },
  calibration: { label: "Calibration", color: () => C.orchid, note: "fit the interval quantiles" },
  holdout: { label: "Holdout", color: () => C.champion, note: "final test, scored once" },
  forward: { label: "Forward", color: () => C.skyData, note: "the next 14 days, no actuals" },
};

function timeline(meta) {
  const cal = meta.split_calendar;
  const dw = meta.data_window;
  const H = 14;
  const stages = ["selection", "calibration", "holdout"].filter((s) => cal[s] && cal[s].length).map((s) => ({
    id: s, origins: cal[s], start: cal[s][0], end: addDays(cal[s][cal[s].length - 1], H),
  }));
  stages.push({ id: "forward", origins: [dw.last_actual], start: dw.last_actual, end: dw.forward_end });
  const z0 = addDays(stages[0].start, -21);
  const z1 = addDays(dw.forward_end, 7);
  const W = 1000, X0 = 30, X1 = 970, AX = 236;
  const span = dayDiff(z0, z1);
  const x = (d) => X0 + ((X1 - X0) * dayDiff(z0, d)) / span;
  const full = dayDiff(dw.start, z1);
  const fx = (d) => X0 + ((X1 - X0) * dayDiff(dw.start, d)) / full;
  let svg = `<svg class="timeline-svg" viewBox="0 0 ${W} 330" role="img" aria-labelledby="tl-title tl-desc">
    <title id="tl-title">Split calendar</title>
    <desc id="tl-desc">Training history from ${dw.start}. ${stages.map((s) => `${STAGE_STYLE[s.id].label} targets ${s.start} to ${s.end} with ${s.origins.length} weekly origin${s.origins.length > 1 ? "s" : ""}`).join(". ")}.</desc>
    <g font-family="Montserrat, sans-serif" font-size="12" fill="${C.text2}">
    <text x="${X0}" y="18" fill="${C.muted}">Full window ${fmtDate(dw.start)} – ${fmtDate(dw.forward_end)}</text>
    <line x1="${X0}" x2="${X1}" y1="32" y2="32" stroke="${C.muted}" stroke-width="2" stroke-linecap="round"/>
    <rect x="${fx(z0)}" y="26" width="${fx(z1) - fx(z0)}" height="12" rx="6" fill="none" stroke="${C.text}" stroke-width="1.5"/>
    <text x="${fx(z0) - 6}" y="52" text-anchor="end" fill="${C.muted}">zoomed below ↓</text>
    <line x1="${X0}" x2="${X1}" y1="${AX}" y2="${AX}" stroke="${C.muted}" stroke-width="1.5"/>`;
  stages.forEach((s, i) => {
    const st = STAGE_STYLE[s.id];
    const a = x(s.start), b = x(s.end);
    const r = (b - a) / 2;
    const h = Math.min(150, Math.max(60, r * 0.9));
    const col = st.color();
    svg += `<path d="M ${a} ${AX} A ${r} ${h} 0 0 1 ${b} ${AX}" fill="none" stroke="${col}" stroke-width="5" stroke-linecap="round" opacity=".95"/>
      <path d="M ${a} ${AX} A ${r} ${h} 0 0 1 ${b} ${AX}" fill="${col}" opacity=".07"/>`;
    s.origins.forEach((o) => { svg += `<circle cx="${x(o)}" cy="${AX}" r="3.5" fill="${col}"/>`; });
    const cx = (a + b) / 2;
    const ty = AX - h - 12;
    const lx = Math.min(Math.max(cx, X0 + 100), X1 - 100);
    svg += `<text x="${cx}" y="${ty}" text-anchor="middle" fill="${C.text}" font-weight="600" font-size="13">${st.label}</text>
      <text x="${lx}" y="${AX + 22 + (i % 2) * 34}" text-anchor="middle" fill="${col}" font-size="11.5">${fmtDate(s.start)} – ${fmtDate(s.end)}</text>
      <text x="${lx}" y="${AX + 37 + (i % 2) * 34}" text-anchor="middle" fill="${C.muted}" font-size="11">${s.origins.length} origin${s.origins.length > 1 ? "s" : ""} · ${st.note}</text>`;
  });
  svg += `<line x1="${x(dw.last_actual)}" x2="${x(dw.last_actual)}" y1="${AX - 175}" y2="${AX + 6}" stroke="${C.text}" stroke-dasharray="3 4"/>
    <text x="${x(dw.last_actual) - 6}" y="${AX - 162}" text-anchor="end" fill="${C.text}" font-size="11">last actual ${fmtDate(dw.last_actual)}</text></g></svg>`;
  return svg;
}

export async function render(el) {
  const [meta, intervals, trust] = await Promise.all([load("meta.json"), load("intervals.json"), load("trust.json").catch(() => null)]);
  const docs = [["README.md", "Project overview"], ["METHODOLOGY.md", "Methodology"], ["MODEL_CARD.md", "Model card"], ["LIMITATIONS.md", "Limitations"], ["docs/DECISIONS.md", "Decision log"]];
  const steps = [
    ["Prepare", `Daily panel for ${meta.scope.n_stores} stores × ${meta.scope.n_families} families from ${fmtDate(meta.data_window.start)}. Closed days are missing, not zero.`],
    ["Split by time", "Weekly origins in three stages with disjoint 14-day target windows."],
    ["Backtest", "Five models forecast 1–14 days ahead from every origin on the same evaluation days."],
    ["Select", "Champion per family on selection backtests only."],
    ["Calibrate", `Conformal ${fmtNum(intervals.level <= 1 ? intervals.level * 100 : intervals.level, 0)} % intervals from calibration residuals.`],
    ["Score trust", "Five-component Trust Index per series, validated on the holdout."],
    ["Publish", "Exported JSON only. The site never retrains a model."],
  ];

  el.innerHTML = `
  ${viewHead("08", "Methodology & Limitations", "How the numbers were made, what was done to stop the future leaking into the past, and what this project cannot tell you.", "prism")}
  <section class="card"><h2>Pipeline</h2><div class="steps">${steps.map(([t, d], i) => `<div class="step"><span class="ring-badge sm" aria-hidden="true">${String(i + 1).padStart(2, "0")}</span><div><h3>${t}</h3><p>${esc(d)}</p></div></div>`).join("")}</div></section>
  <section class="card"><div class="card-head"><h2>Split calendar</h2><span class="muted small">origin = end of day T, targets T+1 … T+14</span></div>
    <div class="table-wrap" ${tableWrapAttrs("Validation timeline, scrollable")}><div style="min-width:640px">${timeline(meta)}</div></div></section>
  <div class="grid g2">
    <section class="card"><h2>Leakage controls</h2><ul class="clean">
      <li>Models receive a masked copy of the data: sales after the origin are blanked, and for promo variant B so are future promotions.</li>
      <li>Selection, calibration and holdout target windows never overlap.</li>
      <li>Champions are chosen on selection backtests only; interval quantiles come from calibration residuals only.</li>
      <li>The holdout is used once, for reporting. Nothing is tuned on it.</li>
      <li>The store scope was fixed using data before the first selection origin.</li>
      <li>Kaggle's test file supplies only the planned promotion counts for the forward window, never sales.</li>
    </ul></section>
    <section class="card"><h2>Prediction intervals</h2>
      <p>${esc(intervals.method)}</p>
      <div class="stat-row">
        <div><b>${fmtPct(intervals.level <= 1 ? intervals.level * 100 : intervals.level, 0)}</b>nominal level</div>
        <div><b>${fmtPct(intervals.calibration_coverage)}</b>calibration coverage</div>
        <div><b>${fmtPct(intervals.holdout.coverage)}</b>holdout coverage</div>
        <div><b>${fmtNum(intervals.holdout.mean_width, 1)}</b>mean width (holdout, index points)</div>
        ${typeof intervals.holdout.mean_width_pct === "number" ? `<div><b>${fmtPct(intervals.holdout.mean_width_pct)}</b>width as % of forecast</div>` : ""}
      </div>
      ${trust ? `<p class="small muted">Trust bands: ${trust.bands.map((b) => `${esc(b.band)} ≥ ${fmtNum(b.min)}`).join(", ")}; any hard flag forces Review.</p>` : ""}
    </section>
  </div>
  <section class="card"><h2>Data caveats and limitations</h2><ul class="clean">${meta.caveats.map((c) => `<li>${esc(c)}</li>`).join("")}${intervals.window_band && typeof intervals.window_band.explanation === "string" && intervals.window_band.explanation.trim() ? `<li><b>14-day total interval:</b> ${esc(intervals.window_band.explanation)}</li>` : ""}</ul>
    <p class="small muted">Citation: ${esc(meta.citation)}</p><p class="small muted">Results generated ${esc(meta.generated_at)}.</p></section>
  <section class="card" id="data-licence"><h2>Data licence</h2>
    <p>${esc(DATA_LICENCE)}</p>
    <ul class="clean small">
      <li>Per-series values (one store × one family) are shown only as an index. ${esc(INDEX_NOTE)}</li>
      <li>${esc(PROMO_INDEX_NOTE)} Raw promoted-item counts are not published.</li>
      <li>Family and portfolio series (Demand Explorer, chain outlook, Scenario Lab baselines in index-days) are equal-weighted means of the member series' indexes. The only unit value published is the portfolio 14-day forecast total of all 96 series. MAE, RMSE and interval widths are in index points.</li>
      <li>Index bases are never published. Converting an index back to units needs the competition data, which anyone can download from Kaggle after accepting the rules and process with the repository's scripts.</li>
    </ul></section>
  <div class="grid g2">
    <section class="card"><h2>Project documents</h2><p class="small">Full write-ups live in the repository.</p>
      <div class="links">${docs.map(([p, l]) => `<a class="btn sm" href="${REPO}/blob/main/${p}" target="_blank" rel="noopener">${l} ↗</a>`).join("")}<a class="btn sm" href="${REPO}" target="_blank" rel="noopener">Repository ↗</a></div></section>
    <section class="card"><h2>Author</h2>
      <div class="author">
        <svg class="avatar" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="24" fill="none" stroke="url(#g-irid)" stroke-width="9"/><text x="32" y="38" text-anchor="middle" font-family="Unbounded, sans-serif" font-weight="800" font-size="15" fill="#fff">AG</text></svg>
        <div><h3 style="margin:0">Akash Gajjar</h3><p class="small" style="margin:2px 0 8px">MSc Supply Chain Management, University of Liverpool · demand planning</p>
          <div class="links"><a class="btn sm" href="${GITHUB_URL}" target="_blank" rel="noopener">GitHub ↗</a>${
            LINKEDIN_URL ? `<a class="btn sm" href="${LINKEDIN_URL}" target="_blank" rel="noopener">LinkedIn ↗</a>` : '<span class="tag" title="Owner to add LinkedIn URL">LinkedIn: link to be added</span>'
          }</div></div>
      </div></section>
  </div>`;
}
