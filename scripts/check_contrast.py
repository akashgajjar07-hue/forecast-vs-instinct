"""Validate the dashboard colour tokens: WCAG contrast and colour-vision-deficiency separation.

Checks
1. Text tokens >= 4.5:1 against every surface (background, panels, card).
2. Data and status colours >= 3:1 (WCAG 1.4.11 non-text) against every surface; status colours are also used
   as text (band labels), so they must reach 4.5:1 too.
3. Colours that appear together in one chart stay distinguishable: pairwise CIEDE2000 dE >= MIN_DE_NORMAL with
   normal vision and >= MIN_DE_CVD after simulating deuteranopia, protanopia and tritanopia (Machado et al. 2009,
   severity 1.0, applied in linear RGB). Lines also differ by dash pattern on the site, so colour is never the
   only cue.

Tokens are read from docs/styles.css (:root custom properties) so the check stays in sync with the site.
Usage: python scripts/check_contrast.py   (exit 1 on any failure)
"""
from __future__ import annotations

import itertools
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "docs" / "styles.css"

SURFACES = ["--bg", "--panel", "--panel-deep", "--card"]
TEXT = ["--text", "--text-2", "--text-muted", "--sky"]
DATA = ["--c-actual", "--c-champion", "--c-periwinkle", "--c-orchid", "--c-sky", "--c-violet"]
STATUS = ["--accept", "--monitor", "--review"]
GROUPS = {"model lines (actual + 5 models)": DATA, "trust status": STATUS}
MIN_TEXT, MIN_GRAPHIC = 4.5, 3.0
MIN_DE_NORMAL, MIN_DE_CVD = 12.0, 6.0

MACHADO = {
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def tokens() -> dict:
    root = re.search(r":root\s*{(.*?)}", CSS.read_text(), re.S).group(1)
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*(#[0-9A-Fa-f]{6})\s*;", root)}


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_srgb(c):
    c = min(1.0, max(0.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def luminance(rgb):
    r, g, b = (to_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted([luminance(hex_rgb(a)), luminance(hex_rgb(b))], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def simulate(rgb, kind):
    lin = [to_lin(c) for c in rgb]
    m = MACHADO[kind]
    return [to_srgb(sum(m[i][j] * lin[j] for j in range(3))) for i in range(3)]


def lab(rgb):
    r, g, b = (to_lin(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = [t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116 for t in (x, y, z)]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def ciede2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dL, dC = L2 - L1, c2p - c1p
    dh = 0.0 if c1p * c2p == 0 else (h2p - h1p if abs(h2p - h1p) <= 180 else h2p - h1p - 360 * math.copysign(1, h2p - h1p))
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    Lbar, Cbar = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbar = (h1p + h2p) / 2
    else:
        hbar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbar - 30)) + 0.24 * math.cos(math.radians(2 * hbar))
         + 0.32 * math.cos(math.radians(3 * hbar + 6)) - 0.20 * math.cos(math.radians(4 * hbar - 63)))
    sl = 1 + 0.015 * (Lbar - 50) ** 2 / math.sqrt(20 + (Lbar - 50) ** 2)
    sc = 1 + 0.045 * Cbar
    sh = 1 + 0.015 * Cbar * t
    rt = -2 * math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)) * math.sin(math.radians(60 * math.exp(-((hbar - 275) / 25) ** 2)))
    return math.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh))


def main() -> int:
    tok = tokens()
    failures = []
    surf = {s: tok[s] for s in SURFACES}
    print("Surfaces: " + ", ".join(f"{k}={v}" for k, v in surf.items()))
    print(f"\n{'token':16} {'hex':8} {'need':>5} " + " ".join(f"{s:>12}" for s in SURFACES) + "  result")
    for group, need in [(TEXT, MIN_TEXT), (DATA, MIN_GRAPHIC), (STATUS, MIN_TEXT)]:
        for t in group:
            ratios = [contrast(tok[t], surf[s]) for s in SURFACES]
            ok = min(ratios) >= need
            if not ok:
                failures.append(f"{t} contrast {min(ratios):.2f} < {need}")
            print(f"{t:16} {tok[t]:8} {need:>5} " + " ".join(f"{r:>12.2f}" for r in ratios) + ("  PASS" if ok else "  FAIL"))
    brand_violet = "#6030D8"
    print(f"\n(info) brand violet {brand_violet} on --bg: {contrast(brand_violet, tok['--bg']):.2f}:1 -> decorative gradient only, never data.")

    for gname, group in GROUPS.items():
        print(f"\nPairwise CIEDE2000 dE, {gname} (normal >= {MIN_DE_NORMAL}, simulated CVD >= {MIN_DE_CVD})")
        print(f"{'pair':36} {'normal':>7} {'deutan':>7} {'protan':>7} {'tritan':>7}  result")
        for a, b in itertools.combinations(group, 2):
            ra, rb = hex_rgb(tok[a]), hex_rgb(tok[b])
            des = [ciede2000(lab(ra), lab(rb))]
            for k in ["deuteranopia", "protanopia", "tritanopia"]:
                des.append(ciede2000(lab(simulate(ra, k)), lab(simulate(rb, k))))
            ok = des[0] >= MIN_DE_NORMAL and min(des[1:]) >= MIN_DE_CVD
            if not ok:
                failures.append(f"{a} vs {b} dE {[round(x, 1) for x in des]}")
            print(f"{a + ' / ' + b:36} " + " ".join(f"{x:>7.1f}" for x in des) + ("  PASS" if ok else "  FAIL"))

    print()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("All contrast and colour-vision checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
