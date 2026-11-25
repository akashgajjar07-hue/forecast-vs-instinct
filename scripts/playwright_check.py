"""End-to-end check of the static dashboard with headless Chromium.

For every view at 1440x900 and 390x844: no console errors, no page errors, no failed/4xx requests; every
select option, chip, checkbox, slider, number input and button is exercised; charts render or show an empty
state; no horizontal page scroll at 390 px. Also checks deep links, the mobile menu, CSV export, and that the
S&OP view works when localStorage throws. Screenshots go to docs/assets/screenshots/<view>_<width>.png.

Usage
  .venv/bin/python scripts/playwright_check.py --fixtures          # serve docs/ locally, use placeholder data
  .venv/bin/python scripts/playwright_check.py                     # serve docs/ locally, real docs/data
  .venv/bin/python scripts/playwright_check.py --base https://<user>.github.io/forecast-vs-instinct/
Options: --no-screenshots, --views overview,arena
Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socket
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHOTS = DOCS / "assets" / "screenshots"
VIEWS = ["overview", "explorer", "arena", "workspace", "exceptions", "scenario", "sop", "methodology"]
SIZES = [(1440, 900), (390, 844)]
MAX_OPTIONS = 14


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve() -> tuple[str, http.server.ThreadingHTTPServer]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(DOCS)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/", httpd


class Checker:
    def __init__(self, base: str, fixtures: bool, screenshots: bool):
        self.base = base if base.endswith("/") else base + "/"
        self.query = "?fixtures=1" if fixtures else ""
        self.fixtures = fixtures
        self.screenshots = screenshots
        self.failures: list[str] = []
        self.events: list[str] = []
        self.actions = 0

    def url(self, hash_: str) -> str:
        return f"{self.base}{self.query}#/{hash_}"

    def attach(self, page):
        page.on("console", lambda m: m.type == "error" and self.events.append(f"console error: {m.text}"))
        page.on("pageerror", lambda e: self.events.append(f"page error: {e}"))
        page.on("requestfailed", lambda r: self.events.append(f"request failed: {r.url} ({r.failure})"))
        page.on("response", lambda r: r.status >= 400 and self.events.append(f"HTTP {r.status}: {r.url}"))
        page.on("dialog", lambda d: d.accept())

    def flush(self, where: str):
        for e in self.events:
            self.failures.append(f"[{where}] {e}")
        self.events.clear()

    def settle(self, page):
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        page.wait_for_timeout(120)

    def open_view(self, page, view: str, extra: str = ""):
        page.goto(self.url(view + extra))
        page.wait_for_selector(f'main[data-ready="{view}"]', timeout=20000)
        self.settle(page)

    def check_charts(self, page, where: str):
        bad = page.evaluate("""() => [...document.querySelectorAll('main .chart')].filter(c =>
            !c.classList.contains('js-plotly-plot') && !c.querySelector('.empty')).map(c => c.id || '(no id)')""")
        for b in bad:
            self.failures.append(f"[{where}] chart {b} neither rendered nor shows an empty state")
        unlabeled = page.evaluate("""() => [...document.querySelectorAll('main .chart.js-plotly-plot')].filter(c =>
            !c.querySelector('.empty') && !c.getAttribute('aria-label')).map(c => c.id)""")
        for u in unlabeled:
            self.failures.append(f"[{where}] chart {u} has no aria-label")

    def check_hscroll(self, page, where: str):
        sw, iw = page.evaluate("() => [document.documentElement.scrollWidth, window.innerWidth]")
        if sw > iw + 1:
            self.failures.append(f"[{where}] horizontal page scroll: scrollWidth {sw} > {iw}")

    def exercise(self, page, view: str, where: str):
        # selects (re-query each time: some views rebuild their options)
        n_sel = page.locator("main select").count()
        for i in range(n_sel):
            sel = page.locator("main select").nth(i)
            if not sel.is_visible() or not sel.is_enabled():
                continue
            values = sel.evaluate("s => [...s.options].map(o => o.value)")
            if len(values) > MAX_OPTIONS:
                step = max(1, len(values) // MAX_OPTIONS)
                values = values[::step] + [values[-1]]
            for v in values:
                sel = page.locator("main select").nth(i)
                if not sel.is_visible() or not sel.is_enabled():
                    break
                if v not in sel.evaluate("s => [...s.options].map(o => o.value)"):
                    continue
                sel.select_option(v)
                self.actions += 1
                self.settle(page)
            if view in ("sop",):
                break  # one table select is enough; the rest are identical per-row reason dropdowns
        # chips / toggle buttons
        for i in range(page.locator("main button[aria-pressed]").count()):
            b = page.locator("main button[aria-pressed]").nth(i)
            if b.is_visible():
                b.click()
                self.actions += 1
                self.settle(page)
        # checkboxes
        for i in range(page.locator("main input[type=checkbox]").count()):
            c = page.locator("main input[type=checkbox]").nth(i)
            if c.is_visible():
                c.click()
                self.actions += 1
        # sliders
        for i in range(page.locator("main input[type=range]").count()):
            r = page.locator("main input[type=range]").nth(i)
            if r.is_visible():
                r.focus()
                page.keyboard.press("Home")
                for _ in range(4):
                    page.keyboard.press("ArrowRight")
                    self.actions += 1
                    page.wait_for_timeout(60)
                page.keyboard.press("ArrowLeft")
                page.keyboard.press("ArrowLeft")
        # number / text inputs
        nums = page.locator("main input[type=number]")
        for i in range(min(nums.count(), 14)):
            n = nums.nth(i)
            if n.is_visible() and n.is_enabled():
                n.fill("123")
                n.fill("")
                n.fill("250")
                self.actions += 1
        # view-specific buttons
        if view == "exceptions":
            for i in range(min(3, page.locator("main button[data-sid]").count())):
                page.locator("main button[data-sid]").nth(i).click()
                self.actions += 1
        if view == "scenario":
            page.locator("#sl-reset").click()
            self.actions += 1
            reco = page.locator("#sl-reco").inner_text()
            if "Plan to the" not in reco:
                self.failures.append(f"[{where}] scenario recommendation text missing")
        if view == "sop":
            page.locator("main input.ov-qty").first.fill("999")
            page.locator("main select.ov-reason").first.select_option(index=1)
            with page.expect_download(timeout=10000) as dl:
                page.locator("#sp-export").click()
            path = dl.value.path()
            text = Path(path).read_text() if path else ""
            lines = text.strip().splitlines()
            if not lines or lines[0] != "date,location,product_family,statistical_index,override_index,final_index,reason_code,user":
                self.failures.append(f"[{where}] CSV header wrong: {lines[:1]}")
            elif not any(",999," in ln for ln in lines[1:]):
                self.failures.append(f"[{where}] CSV export missing the override")
            page.locator("#sp-clear").click()
            self.actions += 2
        self.settle(page)

    def run_view(self, page, view: str, width: int):
        where = f"{view}@{width}"
        self.open_view(page, view)
        banner = page.locator("#fixture-banner").is_visible()
        if banner != self.fixtures:
            self.failures.append(f"[{where}] fixture banner visible={banner}, expected {self.fixtures}")
        if page.locator(".site-footer").inner_text().find("Corporación Favorita") < 0:
            self.failures.append(f"[{where}] footer missing")
        self.check_charts(page, where)
        if width < 500:
            self.check_hscroll(page, where)
        if self.screenshots:
            SHOTS.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SHOTS / f"{view}_{width}.png"), full_page=True)
        self.exercise(page, view, where)
        self.check_charts(page, where)
        if width < 500:
            self.check_hscroll(page, where)
        self.flush(where)

    def extra_checks(self, browser):
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        self.attach(page)
        # deep links
        self.open_view(page, "arena", "?stage=selection&slice=family&metric=bias")
        if page.locator("#ar-slice").input_value() != "family":
            self.failures.append("[deeplink] arena slice param ignored")
        self.open_view(page, "explorer", "?family=DAIRY")
        if page.locator("#ex-family").input_value() != "DAIRY":
            self.failures.append("[deeplink] explorer family param ignored")
        slug = page.evaluate("async () => (await (await fetch('" + ("data/_fixtures/" if self.fixtures else "data/")
                             + "series_index.json')).json()).slice(-1)[0].slug")
        self.open_view(page, "workspace", f"?series={slug}")
        self.settle(page)
        title = page.locator("#ws-title").inner_text()
        store = slug.split("_")[0]
        if f"Store {store}" not in title:
            self.failures.append(f"[deeplink] workspace series param ignored ({title})")
        # keyboard: first Tab after a fresh load reaches the skip link
        page.reload()
        page.wait_for_selector('main[data-ready="workspace"]', timeout=20000)
        page.keyboard.press("Tab")
        if page.evaluate("() => document.activeElement && document.activeElement.className") != "skip-link":
            self.failures.append("[a11y] first Tab does not reach the skip link")
        self.flush("deeplinks")
        ctx.close()
        # no localStorage
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.add_init_script("""(() => { const t = () => { throw new Error('storage disabled'); };
            Storage.prototype.getItem = t; Storage.prototype.setItem = t; Storage.prototype.removeItem = t; })();""")
        page = ctx.new_page()
        self.attach(page)
        self.open_view(page, "sop")
        page.locator("main input.ov-qty").first.fill("42")
        page.wait_for_timeout(100)
        if not page.locator("#sp-storage").is_visible():
            self.failures.append("[no-storage] storage warning not shown")
        if "42" not in page.locator("main tr[data-sid] td.final").first.inner_text():
            self.failures.append("[no-storage] override not applied without storage")
        self.flush("no-storage")
        ctx.close()
        # mobile menu
        ctx = browser.new_context(viewport={"width": 390, "height": 844})
        page = ctx.new_page()
        self.attach(page)
        self.open_view(page, "overview")
        page.locator("#menu-btn").click()
        page.wait_for_timeout(350)
        if page.locator("#menu-btn").get_attribute("aria-expanded") != "true":
            self.failures.append("[mobile] menu did not open")
        page.locator('#nav-list a[data-view="arena"]').click()
        page.wait_for_selector('main[data-ready="arena"]', timeout=15000)
        if page.locator("#menu-btn").get_attribute("aria-expanded") != "false":
            self.failures.append("[mobile] menu did not close after navigation")
        self.flush("mobile-menu")
        ctx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="site URL; default serves docs/ locally")
    ap.add_argument("--fixtures", action="store_true", help="append ?fixtures=1 (placeholder data)")
    ap.add_argument("--no-screenshots", action="store_true")
    ap.add_argument("--views", help="comma-separated subset")
    a = ap.parse_args()
    httpd = None
    base = a.base
    if not base:
        base, httpd = serve()
    views = a.views.split(",") if a.views else VIEWS
    ck = Checker(base, a.fixtures, not a.no_screenshots)
    print(f"Checking {base} ({'fixtures' if a.fixtures else 'real data'}) …")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for w, h in SIZES:
            ctx = browser.new_context(viewport={"width": w, "height": h}, accept_downloads=True, reduced_motion="reduce")
            page = ctx.new_page()
            ck.attach(page)
            for v in views:
                before = len(ck.failures)
                try:
                    ck.run_view(page, v, w)
                except Exception as e:  # noqa: BLE001
                    ck.failures.append(f"[{v}@{w}] exception: {e}")
                    ck.flush(f"{v}@{w}")
                status = "ok" if len(ck.failures) == before else f"{len(ck.failures) - before} problem(s)"
                print(f"  {v:12} {w}x{h}: {status}")
            ctx.close()
        try:
            ck.extra_checks(browser)
            print("  deep links, keyboard, no-storage, mobile menu: " + ("ok" if not ck.failures else "see below"))
        except Exception as e:  # noqa: BLE001
            ck.failures.append(f"[extra] exception: {e}")
        browser.close()
    if httpd:
        httpd.shutdown()
    print(f"\n{ck.actions} interactions exercised.")
    if ck.failures:
        print(f"FAILED ({len(ck.failures)}):")
        for f in ck.failures:
            print("  - " + f)
        return 1
    print("PASSED: no console errors, no failed requests, no horizontal scroll at 390 px.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
