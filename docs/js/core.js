// Shared helpers: data loading, formatting, routing params, Plotly defaults, decorative shapes.

const qs = new URLSearchParams(location.search);
export const FIXTURES = qs.get("fixtures") === "1";
export const DATA_BASE = FIXTURES ? "data/_fixtures/" : "data/";
export const REPO = "https://github.com/akashgajjar07-hue/forecast-vs-instinct";

// ---------- data ----------
const cache = new Map();
export class DataError extends Error {}

export function load(name) {
  if (!cache.has(name)) {
    const p = fetch(DATA_BASE + name, { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new DataError(`${name} is not available yet (HTTP ${r.status}).`);
      return r.json();
    });
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return cache.get(name);
}
export const loadSeries = (slug) => load(`series/${slug}.json`);

// ---------- storage (must work without it) ----------
export function storeGet(key, fallback) {
  try {
    const v = window.localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch (e) {
    return fallback;
  }
}
export function storeSet(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (e) {
    return false;
  }
}
export function storeDel(key) {
  try {
    window.localStorage.removeItem(key);
  } catch (e) {
    /* storage unavailable */
  }
}

// ---------- formatting ----------
const isNum = (x) => typeof x === "number" && Number.isFinite(x);
export function fmtNum(x, dp = 0) {
  if (!isNum(x)) return "n/a";
  return x.toLocaleString("en-GB", { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
export function fmtCompact(x) {
  if (!isNum(x)) return "n/a";
  const a = Math.abs(x);
  if (a >= 1e6) return fmtNum(x / 1e6, 2) + " M";
  if (a >= 1e4) return fmtNum(x / 1e3, 1) + " k";
  return fmtNum(x, 0);
}
export function fmtPct(x, dp = 1, sign = false) {
  if (!isNum(x)) return "n/a";
  const s = sign && x > 0 ? "+" : "";
  return s + fmtNum(x, dp) + " %";
}
export function fmtSigned(x, dp = 1) {
  if (!isNum(x)) return "n/a";
  return (x > 0 ? "+" : "") + fmtNum(x, dp);
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function fmtDate(iso) {
  if (!iso) return "n/a";
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}
export function addDays(iso, n) {
  const t = new Date(iso + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}
export function dayDiff(a, b) {
  return Math.round((new Date(b + "T00:00:00Z") - new Date(a + "T00:00:00Z")) / 864e5);
}
export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
export function slugOf(store, family) {
  return `${store}_${family.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
}
export const titleCase = (f) => f.toLowerCase().replace(/(^|[\s/])\S/g, (c) => c.toUpperCase());
export const famLabel = (f) => titleCase(f);

// ---------- routing params ----------
export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [view, query = ""] = raw.split("?");
  return { view: view || "overview", params: Object.fromEntries(new URLSearchParams(query)) };
}
export function setParams(updates) {
  const { view, params } = parseHash();
  const next = { ...params, ...updates };
  for (const k of Object.keys(next)) if (next[k] === null || next[k] === undefined || next[k] === "") delete next[k];
  const q = new URLSearchParams(next).toString();
  const hash = `#/${view}${q ? "?" + q : ""}`;
  try {
    history.replaceState(null, "", location.pathname + location.search + hash);
  } catch (e) {
    /* ignore */
  }
}
export function viewHref(view, params = {}) {
  const q = new URLSearchParams(params).toString();
  return `#/${view}${q ? "?" + q : ""}`;
}

// ---------- colours ----------
const css = getComputedStyle(document.documentElement);
const tok = (n) => css.getPropertyValue(n).trim();
export const C = {
  bg: tok("--bg"), panel: tok("--panel"), panelDeep: tok("--panel-deep"), card: tok("--card"),
  text: tok("--text"), text2: tok("--text-2"), muted: tok("--text-muted"), sky: tok("--sky"),
  actual: tok("--c-actual"), champion: tok("--c-champion"), periwinkle: tok("--c-periwinkle"), orchid: tok("--c-orchid"),
  skyData: tok("--c-sky"), violet: tok("--c-violet"), band: tok("--band-fill"),
  Accept: tok("--accept"), Monitor: tok("--monitor"), Review: tok("--review"),
};
export const MODELS = ["naive", "seasonal_naive", "moving_average", "holt_winters", "lightgbm"];
export const MODEL_LABEL = { naive: "Naive", seasonal_naive: "Seasonal naive", moving_average: "Moving average", holt_winters: "Holt-Winters", lightgbm: "LightGBM" };
export const MODEL_COLOR = { lightgbm: C.champion, holt_winters: C.periwinkle, moving_average: C.skyData, seasonal_naive: C.orchid, naive: C.violet };
export const CHALLENGER_COLORS = [C.periwinkle, C.orchid, C.skyData, C.violet];
export const CHALLENGER_DASH = ["dash", "dot", "dashdot", "longdash"];
export const BANDS = ["Review", "Monitor", "Accept"];

// ---------- Plotly ----------
const AXIS = { gridcolor: "rgba(154,164,173,0.16)", zerolinecolor: "rgba(154,164,173,0.35)", linecolor: "rgba(154,164,173,0.35)", tickcolor: "rgba(154,164,173,0.35)", automargin: true, tickfont: { color: "#D7D0E7" }, title: { font: { color: "#D7D0E7", size: 12 } } };
function merge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b || {})) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) && a[k] && typeof a[k] === "object" ? merge(a[k], v) : v;
  }
  return out;
}
// Replace a chart element with a fresh, empty copy instead of Plotly.purge(): purge deletes the
// internal layout while Plotly may still have an auto-margin redraw queued, and that queued redraw
// then throws (seen on slower networks). A detached old node lets the pending redraw finish harmlessly.
// A caller may still hold a node that freshEl() replaced: resolve it again by id.
export function live(el) {
  return el && !el.isConnected && el.id ? document.getElementById(el.id) || el : el;
}
export function freshEl(el) {
  el = live(el);
  if (!el) return el;
  const n = document.createElement(el.tagName);
  for (const a of el.attributes) n.setAttribute(a.name, a.value);
  n.classList.remove("js-plotly-plot");
  n.removeAttribute("role");
  n.removeAttribute("aria-label");
  el.replaceWith(n);
  return n;
}
export function plot(el, traces, layout = {}, label = "") {
  el = live(el);
  if (!el) return;
  if (!el.classList.contains("js-plotly-plot") || el.querySelector(":scope > .empty, :scope > .loading")) {
    el = freshEl(el);
  }
  if (label) {
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", label);
  }
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Montserrat, Segoe UI, sans-serif", color: C.text2, size: 12 },
    margin: { l: 54, r: 14, t: 10, b: 40 }, xaxis: AXIS, yaxis: AXIS,
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: C.text2, size: 11 }, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: C.panelDeep, bordercolor: C.champion, font: { color: C.text, family: "Montserrat, sans-serif" } },
    showlegend: traces.length > 1,
  };
  const L = merge(base, layout);
  if (layout.xaxis) L.xaxis = merge(AXIS, layout.xaxis);
  if (layout.yaxis) L.yaxis = merge(AXIS, layout.yaxis);
  // Plotly writes computed ranges/types back into the layout object: never hand it shared defaults.
  window.Plotly.react(el, traces, JSON.parse(JSON.stringify(L)), { displayModeBar: false, responsive: true });
}
export function purgeCharts(root) {
  // Detach old charts (see freshEl); callers re-render the container's HTML afterwards.
  root.querySelectorAll(".js-plotly-plot").forEach((el) => freshEl(el));
}
export function bandTrace(x, lo, hi, name = "P10–P90 band") {
  return [
    { x, y: hi, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, name: "P90" },
    { x, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: C.band, name, hoverinfo: "skip" },
  ];
}

// ---------- markup helpers ----------
export function pill(band) {
  return `<span class="pill ${esc(band)}">${esc(band)}</span>`;
}
export function empty(msg) {
  return `<div class="empty" role="status">${esc(msg)}</div>`;
}
export function showEmpty(el, msg) {
  el = freshEl(el);
  if (!el) return;
  el.innerHTML = empty(msg);
}
export function showLoading(el, msg = "Loading…") {
  el = live(el);
  if (!el) return;
  if (el.classList.contains("js-plotly-plot")) el = freshEl(el);
  el.innerHTML = `<p class="loading">${esc(msg)}</p>`;
}
export function errorCard(err) {
  const msg = err instanceof DataError ? err.message : "Something went wrong while drawing this view.";
  return `<div class="card error-card" role="alert"><h2>Data not available</h2><p>${esc(msg)}</p>
    <p class="muted small">The pipeline publishes these files to <code>docs/data/</code>. Add <code>?fixtures=1</code> to the URL to preview the layout with placeholder data.</p></div>`;
}
export function options(list, selected) {
  return list.map(([v, l]) => `<option value="${esc(v)}"${String(v) === String(selected) ? " selected" : ""}>${esc(l)}</option>`).join("");
}

// ---------- decorative shapes (aria-hidden) ----------
let coilPath = null;
function coil() {
  if (!coilPath) {
    const pts = [];
    for (let t = 0; t <= Math.PI * 2 * 4.25; t += 0.12) pts.push([50 + 34 * Math.cos(t), 20 + t * 5.6 + 9 * Math.sin(t)]);
    coilPath = "M" + pts.map((p) => p.map((v) => v.toFixed(1)).join(" ")).join(" L");
  }
  return coilPath;
}
export function shape(kind, cls = "") {
  if (kind === "torus") return `<div class="shape shape-glow ${cls}" aria-hidden="true"><div class="torus"></div></div>`;
  if (kind === "orb") return `<div class="shape ${cls}" aria-hidden="true"><div class="orb"></div></div>`;
  if (kind === "prism") {
    return `<div class="shape shape-glow ${cls}" aria-hidden="true"><svg viewBox="0 0 100 100" focusable="false">
      <polygon points="50,4 8,78 50,96" fill="url(#g-prism-a)"/>
      <polygon points="50,4 92,78 50,96" fill="url(#g-prism-b)"/>
      <polygon points="50,4 50,96 8,78" fill="url(#g-sheen)" opacity=".55"/>
      <polyline points="50,4 50,96" stroke="rgba(255,255,255,.7)" stroke-width="1" fill="none"/>
    </svg></div>`;
  }
  if (kind === "coil") {
    const d = coil();
    return `<div class="shape shape-glow ${cls}" aria-hidden="true"><svg viewBox="0 0 100 190" focusable="false">
      <path d="${d}" fill="none" stroke="url(#g-irid-v)" stroke-width="11" stroke-linecap="round"/>
      <path d="${d}" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="2.5" stroke-linecap="round" transform="translate(-2 -3)"/>
    </svg></div>`;
  }
  return "";
}
export function viewHead(num, title, lede, kind = "torus", extra = "") {
  return `<header class="view-head">${shape(kind)}<span class="ring-badge" aria-hidden="true">${num}</span>
    <div class="text"><h1 tabindex="-1">${esc(title)}</h1><p>${lede}</p>${extra}</div></header>`;
}
