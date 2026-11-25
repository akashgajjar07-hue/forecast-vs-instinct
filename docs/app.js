// Shell: navigation, hash routing, lazy view rendering.
import { FIXTURES, parseHash, purgeCharts, errorCard, esc } from "./js/core.js";
import * as overview from "./js/views/overview.js";
import * as explorer from "./js/views/explorer.js";
import * as arena from "./js/views/arena.js";
import * as workspace from "./js/views/workspace.js";
import * as exceptions from "./js/views/exceptions.js";
import * as scenario from "./js/views/scenario_lab.js";
import * as sop from "./js/views/sop.js";
import * as methodology from "./js/views/methodology.js";

export const ROUTES = [
  { id: "overview", num: "01", title: "Executive Overview", mod: overview },
  { id: "explorer", num: "02", title: "Demand Explorer", mod: explorer },
  { id: "arena", num: "03", title: "Model Arena", mod: arena },
  { id: "workspace", num: "04", title: "Forecast Workspace", mod: workspace },
  { id: "exceptions", num: "05", title: "Exception Centre", mod: exceptions },
  { id: "scenario", num: "06", title: "UAE Scenario Lab", mod: scenario, hyp: true },
  { id: "sop", num: "07", title: "S&OP Demand Review", mod: sop },
  { id: "methodology", num: "08", title: "Methodology & Limitations", mod: methodology },
];

const main = document.getElementById("view");
const navList = document.getElementById("nav-list");
const menuBtn = document.getElementById("menu-btn");
let renderToken = 0;
let firstRender = true;

if (FIXTURES) {
  document.getElementById("fixture-banner").hidden = false;
  document.body.classList.add("has-banner");
  document.title = "PLACEHOLDER · Forecast vs Instinct";
}

navList.innerHTML = ROUTES.map(
  (r) => `<li><a href="#/${r.id}" data-view="${r.id}"><span class="nav-num" aria-hidden="true">${r.num}</span>${esc(r.title)}${
    r.hyp ? '<span class="nav-hyp" title="Hypothetical">HYP</span>' : ""
  }</a></li>`
).join("");

function setMenu(open) {
  document.body.classList.toggle("nav-open", open);
  menuBtn.setAttribute("aria-expanded", String(open));
}
menuBtn.addEventListener("click", () => setMenu(!document.body.classList.contains("nav-open")));
navList.addEventListener("click", (e) => {
  if (e.target.closest("a")) setMenu(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && document.body.classList.contains("nav-open")) {
    setMenu(false);
    menuBtn.focus();
  }
});
document.addEventListener("click", (e) => {
  if (document.body.classList.contains("nav-open") && !e.target.closest("#sidenav") && !e.target.closest("#menu-btn")) setMenu(false);
});

async function render() {
  const { view, params } = parseHash();
  const route = ROUTES.find((r) => r.id === view) || ROUTES[0];
  const token = ++renderToken;
  navList.querySelectorAll("a").forEach((a) => {
    if (a.dataset.view === route.id) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `${FIXTURES ? "PLACEHOLDER · " : ""}${route.title} · Forecast vs Instinct`;
  purgeCharts(main);
  const container = document.createElement("div");
  container.className = "view";
  container.dataset.view = route.id;
  main.replaceChildren(container);
  container.innerHTML = '<p class="loading">Loading…</p>';
  try {
    await route.mod.render(container, params, { isCurrent: () => token === renderToken });
  } catch (err) {
    if (token !== renderToken) return;
    console.warn(err);
    container.innerHTML = errorCard(err);
  }
  if (token !== renderToken) return;
  if (!firstRender) {
    const h1 = container.querySelector("h1");
    if (h1) h1.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
  firstRender = false;
  main.setAttribute("data-ready", route.id);
}

window.addEventListener("hashchange", render);
function start() {
  if (!location.hash) history.replaceState(null, "", location.pathname + location.search + "#/overview");
  render();
}
if (window.Plotly) start();
else window.addEventListener("load", start, { once: true });
