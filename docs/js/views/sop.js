import { load, esc, fmtNum, fmtPct, fmtSigned, fmtDate, addDays, viewHead, options, setParams, famLabel, pill, storeGet, storeSet, storeDel, BANDS, WINDOW_BAND_LABEL, fillWindowBandCaption, tableWrapAttrs } from "../core.js";

const USER_KEY = "fvi.sop.user";

export function csvCell(v) {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
// Text fields only: neutralise spreadsheet formula injection (=, +, -, @, tab, CR) with a leading quote.
// Numeric quantities are passed through csvCell directly so they stay numeric.
export function csvText(v) {
  const s = v === null || v === undefined ? "" : String(v);
  return csvCell(/^[=+\-@\t\r]/.test(s) ? `'${s}` : s);
}
const q2 = (x) => (x === null || x === undefined || !Number.isFinite(x) ? "" : String(Math.round(x * 100) / 100));

export async function render(el, params) {
  const [sop, meta] = await Promise.all([load("sop.json"), load("meta.json")]);
  // v2: overrides are entered on the index scale (v1 held units and is not reused)
  const KEY = `fvi.sop.overrides.v2:${sop.origin}`;
  const famOrder = meta.scope.families;
  const rowsAll = [...sop.rows].sort((a, b) => a.store_nbr - b.store_nbr || famOrder.indexOf(a.family) - famOrder.indexOf(b.family));
  const stores = [...new Set(rowsAll.map((r) => r.store_nbr))];
  const fams = famOrder.filter((f) => rowsAll.some((r) => r.family === f));
  const horizonStart = addDays(sop.origin, 1);
  let ov = storeGet(KEY, {}) || {};
  let storageOk = storeSet(KEY, ov);
  const state = {
    family: fams.includes(params.family) ? params.family : "ALL",
    store: stores.map(String).includes(params.store) ? params.store : "ALL",
    band: BANDS.includes(params.band) ? params.band : "ALL",
  };
  const s = sop.summary;
  const fva = sop.fva;
  const fvaConsistent = Math.abs(fva.champion_wape - fva.simulated_wape - fva.fva_pp) < 0.06;

  const list = (items) => (items.length ? `<ul class="clean">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : '<p class="small muted">Nothing flagged.</p>');

  el.innerHTML = `
  ${viewHead("07", "S&OP Demand Review", `Consensus forecast for ${fmtDate(horizonStart)} – ${fmtDate(sop.horizon_end)}. Start from the statistical forecast, override only with a reason, export in an ERP-ready flat file.`, "torus")}
  <section class="card">
    <h2>Demand review summary</h2>
    <p class="small muted">Generated from the published results by fixed rules (no language model). Origin ${fmtDate(sop.origin)}.</p>
    <div class="grid g3">
      <div><h3>Top risks</h3>${list(s.top_risks)}</div>
      <div><h3>Persistent bias</h3>${list(s.persistent_bias)}</div>
      <div><h3>Review queue</h3>${list(s.review_queue)}</div>
      <div><h3>Promotion notes (associations, not causal)</h3>${list(s.promo_opportunities)}</div>
      <div class="span2"><h3>Discussion points for the meeting</h3><ol class="clean">${s.discussion_points.map((x) => `<li>${esc(x)}</li>`).join("")}</ol></div>
    </div>
  </section>
  <div class="controls" role="group" aria-label="Review filters">
    <label class="field"><span>Family</span><select id="sp-fam">${options([["ALL", "All families"], ...fams.map((f) => [f, famLabel(f)])], state.family)}</select></label>
    <label class="field"><span>Store</span><select id="sp-store">${options([["ALL", "All stores"], ...stores.map((x) => [x, `Store ${x}`])], state.store)}</select></label>
    <label class="field"><span>Trust band</span><select id="sp-band">${options([["ALL", "All bands"], ...BANDS.map((b) => [b, b])], state.band)}</select></label>
    <label class="field"><span>Planner (export user)</span><input type="text" id="sp-user" maxlength="40" autocomplete="off"></label>
    <button type="button" class="btn primary" id="sp-export">Export CSV</button>
    <button type="button" class="btn" id="sp-clear">Clear overrides</button>
  </div>
  <section class="card">
    <div class="card-head"><h2>Consensus table</h2><span class="muted small" id="sp-count"></span></div>
    <p class="note small" id="sp-storage" ${storageOk ? "hidden" : ""}>Browser storage is unavailable, so overrides last only while this page is open. Export before leaving.</p>
    <div class="table-wrap scroll-y" id="sp-table" ${tableWrapAttrs("S&OP consensus table, scrollable")}></div>
    <p class="caption">Quantities are 14-day forecasts on the index scale: the mean daily value over the 14 days ÷ the series' mean daily sales in the training period (1 Jun 2015 – 6 Feb 2017) × 100, so 100 = an average training-period day. Enter overrides on the same scale. Final = override if entered, otherwise the statistical (champion) forecast. Overrides are saved in this browser only. The CSV has one row per series for the 14-day bucket, dated at the horizon start, with index columns (statistical_index, override_index, final_index); converting them back to units needs each series' base, which only the data owner can compute from their local copy of the competition data. Per-series units are not published.</p>
    <p class="caption" id="sp-wb" hidden></p>
  </section>
  <section class="card">
    <div class="card-head"><h2>Forecast value added</h2><span class="hyp-badge">Simulated</span></div>
    <p><b>${esc(fva.label)}.</b> Simulated, rule-based override, not human-planner performance.</p>
    <p class="small">Rule: ${esc(fva.rule)} · scored on the ${esc(fva.stage)} stage.</p>
    <div class="stat-row">
      <div><b>${fmtPct(fva.champion_wape)}</b>champion WAPE</div>
      <div><b>${fmtPct(fva.simulated_wape)}</b>with simulated override</div>
      <div><b>${fmtSigned(fva.fva_pp)} pp</b>FVA</div>
    </div>
    <p class="small muted">${fvaConsistent ? `FVA = champion WAPE − override WAPE. ${fva.fva_pp > 0 ? "Positive: the rule reduced error." : fva.fva_pp < 0 ? "Negative: the rule added error, so touching the forecast destroyed value." : "Zero: no change."}` : "FVA as published by the pipeline."}</p>
  </section>`;

  const sel = (id) => el.querySelector(id);
  const userIn = sel("#sp-user");
  userIn.value = storeGet(USER_KEY, "planner") || "planner";
  userIn.addEventListener("input", () => storeSet(USER_KEY, userIn.value));

  const visible = () => rowsAll.filter((r) => (state.family === "ALL" || r.family === state.family) && (state.store === "ALL" || String(r.store_nbr) === state.store) && (state.band === "ALL" || r.trust_band === state.band));
  const finalOf = (r) => (ov[r.series_id] && Number.isFinite(ov[r.series_id].qty) ? ov[r.series_id].qty : r.stat_index);
  const save = () => {
    storageOk = storeSet(KEY, ov);
    sel("#sp-storage").hidden = storageOk;
  };

  function totals(rows) {
    // index values of different series do not add up: show the mean index of the filtered rows
    const st = rows.length ? rows.reduce((a, r) => a + r.stat_index, 0) / rows.length : null;
    const fi = rows.length ? rows.reduce((a, r) => a + finalOf(r), 0) / rows.length : null;
    const n = rows.filter((r) => ov[r.series_id] && Number.isFinite(ov[r.series_id].qty)).length;
    return { st, fi, n };
  }
  function updateTotals(rows) {
    const t = totals(rows);
    sel("#sp-tot-stat").textContent = fmtNum(t.st, 1);
    sel("#sp-tot-final").textContent = fmtNum(t.fi, 1);
    sel("#sp-count").textContent = `${rows.length} series · ${t.n} overridden · mean consensus index ${t.fi >= t.st ? "+" : ""}${fmtNum(t.fi - t.st, 1)} vs statistical`;
  }

  function drawTable() {
    const rows = visible();
    const box = sel("#sp-table");
    if (!rows.length) { box.innerHTML = '<div class="empty" role="status">No series match these filters.</div>'; sel("#sp-count").textContent = "0 series"; return; }
    const reasonOpts = (cur) => options([["", "Select reason…"], ...sop.reason_codes.map((c) => [c, c])], cur || "");
    box.innerHTML = `<table class="stack-sm"><thead><tr><th>Store</th><th>Family</th><th class="n">Statistical index</th><th class="n">${WINDOW_BAND_LABEL}, index</th><th>Trust</th><th class="n">Planner override (index)</th><th>Reason code</th><th class="n">Final consensus index</th></tr></thead><tbody>
      ${rows.map((r, i) => {
        const o = ov[r.series_id] || {};
        const id = `sp-o-${i}`;
        return `<tr data-sid="${esc(r.series_id)}"><td data-label="Store">${r.store_nbr}</td><td data-label="Family">${esc(famLabel(r.family))}</td><td class="n" data-label="Statistical index">${fmtNum(r.stat_index, 1)}</td><td class="n" data-label="14-day total P10–P90, index">${fmtNum(r.p10_index, 1)} – ${fmtNum(r.p90_index, 1)}</td>
          <td data-label="Trust">${pill(r.trust_band)} <span class="muted small num">${fmtNum(r.trust_score, 0)}</span></td>
          <td class="n" data-label="Planner override"><label class="sr-only" for="${id}">Override for store ${r.store_nbr} ${esc(famLabel(r.family))}</label><input type="number" min="0" step="0.1" id="${id}" class="ov-qty" value="${Number.isFinite(o.qty) ? o.qty : ""}" placeholder="–"></td>
          <td data-label="Reason code"><label class="sr-only" for="${id}-r">Reason for store ${r.store_nbr} ${esc(famLabel(r.family))}</label><select id="${id}-r" class="ov-reason">${reasonOpts(o.reason)}</select><span class="flag small need" ${Number.isFinite(o.qty) && !o.reason ? "" : "hidden"}>reason needed</span></td>
          <td class="n final" data-label="Final consensus index">${fmtNum(finalOf(r), 1)}</td></tr>`;
      }).join("")}
      <tr class="total-row"><td colspan="2"><b>Mean index (filtered)</b></td><td class="n" data-label="Statistical index"><b id="sp-tot-stat"></b></td><td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td><td class="n" data-label="Final consensus index"><b id="sp-tot-final"></b></td></tr>
    </tbody></table>`;
    updateTotals(rows);
    box.querySelectorAll("tr[data-sid]").forEach((tr) => {
      const sid = tr.dataset.sid;
      const r = rowsAll.find((x) => x.series_id === sid);
      const qty = tr.querySelector(".ov-qty");
      const reason = tr.querySelector(".ov-reason");
      const refresh = () => {
        const o = ov[sid] || {};
        tr.querySelector(".final").textContent = fmtNum(finalOf(r), 1);
        tr.querySelector(".need").hidden = !(Number.isFinite(o.qty) && !o.reason);
        updateTotals(rows);
      };
      qty.addEventListener("input", () => {
        const v = parseFloat(qty.value);
        const o = { ...(ov[sid] || {}) };
        if (Number.isFinite(v) && v >= 0) o.qty = v; else delete o.qty;
        if (o.qty === undefined && !o.reason) delete ov[sid]; else ov[sid] = o;
        save(); refresh();
      });
      reason.addEventListener("change", () => {
        const o = { ...(ov[sid] || {}) };
        if (reason.value) o.reason = reason.value; else delete o.reason;
        if (o.qty === undefined && !o.reason) delete ov[sid]; else ov[sid] = o;
        save(); refresh();
      });
    });
  }

  const push = () => setParams({ family: state.family === "ALL" ? null : state.family, store: state.store === "ALL" ? null : state.store, band: state.band === "ALL" ? null : state.band });
  sel("#sp-fam").addEventListener("change", (e) => { state.family = e.target.value; push(); drawTable(); });
  sel("#sp-store").addEventListener("change", (e) => { state.store = e.target.value; push(); drawTable(); });
  sel("#sp-band").addEventListener("change", (e) => { state.band = e.target.value; push(); drawTable(); });
  sel("#sp-clear").addEventListener("click", () => {
    if (Object.keys(ov).length && !window.confirm("Clear all overrides for this review?")) return;
    ov = {};
    storeDel(KEY);
    drawTable();
  });
  sel("#sp-export").addEventListener("click", () => {
    const user = (userIn.value || "planner").trim() || "planner";
    // Index scale (series training-period mean daily sales = 100). Converting to units needs the owner's local data.
    const lines = ["date,location,product_family,statistical_index,override_index,final_index,reason_code,user"];
    for (const r of rowsAll) {
      const o = ov[r.series_id] || {};
      lines.push([
        csvCell(horizonStart), csvText(`STORE_${r.store_nbr}`), csvText(r.family),
        csvCell(q2(r.stat_index)), csvCell(Number.isFinite(o.qty) ? q2(o.qty) : ""), csvCell(q2(finalOf(r))),
        csvText(o.reason || ""), csvText(user),
      ].join(","));
    }
    const blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `demand_review_${horizonStart}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  });

  drawTable();
  fillWindowBandCaption(sel("#sp-wb"));
}
