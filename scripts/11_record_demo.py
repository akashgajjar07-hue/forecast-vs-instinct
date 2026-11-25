"""Record a 60-90 s walkthrough of the LIVE site (Overview -> Model Arena -> Workspace ->
Exceptions -> Scenario Lab -> S&OP) to presentations/demo.webm, and build a small README GIF
(presentations/demo.gif) from frames captured along the way."""
import shutil
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations"
BASE = sys.argv[1] if len(sys.argv) > 1 else "https://akashgajjar07-hue.github.io/forecast-vs-instinct/"
W, H = 1440, 900
frames = []


def snap(page):
    p = OUT / f"_frame_{len(frames):03d}.png"
    page.screenshot(path=str(p))
    frames.append(p)


def dwell(page, seconds, shots=2):
    for _ in range(shots):
        page.wait_for_timeout(int(seconds * 1000 / shots))
        snap(page)


def scroll(page, px, seconds=2.5):
    steps = 10
    for _ in range(steps):
        page.mouse.wheel(0, px / steps)
        page.wait_for_timeout(int(seconds * 1000 / steps))
    snap(page)


def main():
    vid_dir = OUT / "_video"
    shutil.rmtree(vid_dir, ignore_errors=True)
    t0 = time.time()
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(vid_dir),
                            record_video_size={"width": W, "height": H}, reduced_motion="no-preference")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE + "#/overview", wait_until="networkidle")
        dwell(page, 4.5, 3)
        scroll(page, 700); dwell(page, 2.5, 1); scroll(page, 700); dwell(page, 3, 1)

        page.goto(BASE + "#/arena", wait_until="networkidle"); page.wait_for_timeout(800)
        dwell(page, 3.5, 2)
        sel = page.locator("select").nth(1)
        if sel.count():
            for v in ["family", "horizon"]:
                try:
                    sel.select_option(v); dwell(page, 3, 1)
                except Exception:
                    pass
        scroll(page, 800); dwell(page, 3, 1)

        page.goto(BASE + "#/workspace?series=49_grocery-i&origin=forward", wait_until="networkidle")
        page.wait_for_timeout(1200); dwell(page, 3.5, 2)
        slider = page.locator("input[type=range]")
        if slider.count():
            for v in ["1", "3", "4"]:
                slider.first.evaluate("(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); }", v)
                dwell(page, 2.5, 1)
        scroll(page, 600); dwell(page, 3, 1)

        page.goto(BASE + "#/exceptions", wait_until="networkidle"); page.wait_for_timeout(800)
        dwell(page, 3.5, 2); scroll(page, 900); dwell(page, 2.5, 1)

        page.goto(BASE + "#/scenario", wait_until="networkidle"); page.wait_for_timeout(800)
        dwell(page, 3.5, 2); scroll(page, 700); dwell(page, 2.5, 1)

        page.goto(BASE + "#/sop", wait_until="networkidle"); page.wait_for_timeout(800)
        dwell(page, 3.5, 2); scroll(page, 800); dwell(page, 2.5, 1)

        video = page.video
        ctx.close(); b.close()
        dur = time.time() - t0
        shutil.move(video.path(), OUT / "demo.webm")
        shutil.rmtree(vid_dir, ignore_errors=True)
    if errors:
        print("PAGE ERRORS:", errors)
    imgs = [Image.open(f).convert("RGB").resize((720, 450), Image.LANCZOS) for f in frames]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in imgs]
    pal[0].save(OUT / "demo.gif", save_all=True, append_images=pal[1:], duration=1400, loop=0, optimize=True)
    for f in frames:
        f.unlink()
    print(f"recorded ~{dur:.0f}s, {len(imgs)} gif frames; webm {(OUT / 'demo.webm').stat().st_size / 1e6:.1f} MB, "
          f"gif {(OUT / 'demo.gif').stat().st_size / 1e6:.1f} MB")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
