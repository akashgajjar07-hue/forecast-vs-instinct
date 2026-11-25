/* Forecast vs Instinct: slide painter.
   Reads the display list injected by scripts/10_build_presentations.py into <script id="deck-data">
   and paints every slide with absolutely positioned elements. Charts are flat inline SVG.
   No numbers live in this file: everything comes from the injected JSON. */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("deck-data").textContent);
  const W = DATA.canvas.w, H = DATA.canvas.h;
  const C = {
    text: "#FFFFFF", text2: "#D7D0E7", muted: "#9AA4AD", sky: "#7BBDE6",
    mint: "#60F0D8", peri: "#9090F0", orchid: "#D685C2", dsky: "#68BEF3", violet: "#AE3DFF",
    accept: "#60F0D8", monitor: "#F5C26B", review: "#FF6B8A",
    grid: "rgba(144,144,240,0.16)", axis: "rgba(144,144,240,0.35)", navy: "#00112D",
  };
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fmt = (v, d = 1) => (v < 0 && Number(Math.abs(v).toFixed(d)) !== 0 ? "\u2212" : "") + Math.abs(v).toFixed(d);
  const T = (x, y, s, o = {}) =>
    `<text x="${x}" y="${y}" font-size="${o.fs}" fill="${o.fill || C.text2}" font-weight="${o.fw || 500}"` +
    ` text-anchor="${o.anchor || "start"}" dominant-baseline="${o.base || "middle"}"` +
    (o.ff ? ` font-family="${o.ff}"` : "") + `>${esc(s)}</text>`;
  const HEAD = "Unbounded, 'Segoe UI', sans-serif";

  /* rounded-end horizontal bar anchored at x0 (square at baseline, 4px radius at data end) */
  function hbar(x0, y, len, h, fill) {
    if (len <= 0) return "";
    const r = Math.min(4, len, h / 2);
    return `<path d="M${x0} ${y} H${x0 + len - r} Q${x0 + len} ${y} ${x0 + len} ${y + r} V${y + h - r} Q${x0 + len} ${y + h} ${x0 + len - r} ${y + h} H${x0} Z" fill="${fill}"/>`;
  }
  function vbar(x, y0, w, len, fill) { // grows upward from baseline y0
    if (len <= 0) return "";
    const r = Math.min(4, len, w / 2), top = y0 - len;
    return `<path d="M${x} ${y0} V${top + r} Q${x} ${top} ${x + r} ${top} H${x + w - r} Q${x + w} ${top} ${x + w} ${top + r} V${y0} Z" fill="${fill}"/>`;
  }
  function legend(items, x, y, fs) {
    let out = "", cx = x;
    items.forEach((it) => {
      out += it.dash
        ? `<line x1="${cx}" y1="${y}" x2="${cx + fs * 1.4}" y2="${y}" stroke="${it.color}" stroke-width="2.5" stroke-dasharray="6 5"/>`
        : `<circle cx="${cx + fs * 0.45}" cy="${y}" r="${fs * 0.42}" fill="${it.color}"/>`;
      const off = it.dash ? fs * 1.9 : fs * 1.2;
      out += T(cx + off, y, it.label, { fs, fill: C.text2 });
      cx += off + it.label.length * fs * 0.6 + fs * 1.4;
    });
    return out;
  }

  const charts = {
    leaderboard(p, w, h, fs) {
      const rows = p.rows, labW = p.labelW || fs * 13, rightW = fs * 7.5;
      const max = Math.max(...rows.map((r) => r.wape)) * 1.08;
      const plotW = w - labW - rightW, rowH = h / rows.length, bh = Math.min(rowH * 0.5, fs * 1.6);
      let s = "";
      const rho = p.fmt === "rho";
      (rho ? [0, 0.2, 0.4, 0.6] : [0, 10, 20, 30]).filter((t) => t < max).forEach((t) => {
        const x = labW + (t / max) * plotW;
        s += `<line x1="${x}" y1="0" x2="${x}" y2="${h - fs * 0.2}" stroke="${C.grid}" stroke-width="1"/>`;
      });
      rows.forEach((r, i) => {
        const y = i * rowH + (rowH - bh) / 2, len = (r.wape / max) * plotW;
        s += T(labW - fs * 0.8, y + bh / 2, r.label, { fs, anchor: "end", fill: r.champion ? C.text : C.text2, fw: r.champion ? 700 : 500 });
        s += hbar(labW, y, len, bh, r.champion ? C.mint : (r.muted ? "rgba(144,144,240,0.55)" : C.peri));
        s += T(labW + len + fs * 0.5, y + bh / 2, rho ? "\u2212" + r.wape.toFixed(3) : fmt(r.wape) + "%", { fs: fs * 1.05, fill: C.text, fw: 700 });
        if (r.bias !== undefined) s += T(w, y + bh / 2, "bias " + fmt(r.bias) + "%", { fs: fs * 0.82, anchor: "end", fill: C.muted });
      });
      s += `<line x1="${labW}" y1="0" x2="${labW}" y2="${h}" stroke="${C.axis}" stroke-width="1"/>`;
      return s;
    },

    dumbbell(p, w, h, fs) {
      const rows = p.rows, labW = p.labelW || fs * 10, top = fs * 2.4, bottom = fs * 1.8;
      const max = p.max, plotW = w - labW - fs * 4.5, rowH = (h - top - bottom) / rows.length;
      const X = (v) => labW + (v / max) * plotW;
      let s = legend([{ label: p.aLabel, color: C.mint }, { label: p.bLabel, color: C.orchid }], labW, fs * 0.7, fs * 0.9);
      for (let t = 0; t <= max; t += 10) {
        s += `<line x1="${X(t)}" y1="${top - fs * 0.6}" x2="${X(t)}" y2="${h - bottom}" stroke="${C.grid}"/>`;
        s += T(X(t), h - bottom + fs * 0.9, t + "%", { fs: fs * 0.8, anchor: "middle", fill: C.muted });
      }
      rows.forEach((r, i) => {
        const y = top + i * rowH + rowH / 2, xa = X(r.a), xb = X(r.b);
        s += T(labW - fs * 0.8, y, r.label, { fs: fs * 0.9, anchor: "end" });
        s += `<line x1="${xa}" y1="${y}" x2="${xb}" y2="${y}" stroke="rgba(215,208,231,0.45)" stroke-width="2"/>`;
        s += `<circle cx="${xb}" cy="${y}" r="${fs * 0.42}" fill="${C.orchid}" stroke="#0F2A54" stroke-width="2"/>`;
        s += `<circle cx="${xa}" cy="${y}" r="${fs * 0.42}" fill="${C.mint}" stroke="#0F2A54" stroke-width="2"/>`;
        s += T(xa - fs * 0.7, y, fmt(r.a), { fs: fs * 0.82, anchor: "end", fill: C.text, fw: 700 });
        s += T(xb + fs * 0.7, y, fmt(r.b) + (r.bTag ? " " + r.bTag : ""), { fs: fs * 0.82, fill: C.text2 });
      });
      return s;
    },

    diverging(p, w, h, fs) {
      const rows = p.rows, labW = p.labelW || fs * 10, top = fs * 2.4, bottom = fs * 1.6;
      const lim = p.lim, plotW = w - labW - fs * 1.5, cx = labW + plotW / 2, rowH = (h - top - bottom) / rows.length;
      const bh = Math.min(rowH * 0.56, fs * 1.4);
      const X = (v) => cx + (v / lim) * (plotW / 2);
      let s = legend([{ label: p.posLabel, color: C.dsky }, { label: p.negLabel, color: C.orchid }], labW, fs * 0.7, fs * 0.85);
      [-lim, -lim / 2, 0, lim / 2, lim].forEach((t) => {
        s += `<line x1="${X(t)}" y1="${top - fs * 0.4}" x2="${X(t)}" y2="${h - bottom}" stroke="${t === 0 ? C.axis : C.grid}" stroke-width="${t === 0 ? 1.5 : 1}"/>`;
        s += T(X(t), h - bottom + fs * 0.85, (t > 0 ? "+" : "") + fmt(t, 2), { fs: fs * 0.75, anchor: "middle", fill: C.muted });
      });
      rows.forEach((r, i) => {
        const y = top + i * rowH + (rowH - bh) / 2, v = r.v, len = (Math.abs(v) / lim) * (plotW / 2);
        s += T(labW - fs * 0.8, y + bh / 2, r.label, { fs: fs * 0.88, anchor: "end", fill: r.strong ? C.text : C.text2, fw: r.strong ? 700 : 500 });
        const col = v >= 0 ? C.dsky : C.orchid;
        if (v >= 0) s += hbar(cx, y, len, bh, col);
        else s += `<g transform="translate(${2 * cx} 0) scale(-1 1)">${hbar(cx, y, len, bh, col)}</g>`;
        const tx = v >= 0 ? cx + len + fs * 0.45 : cx - len - fs * 0.45;
        s += T(tx, y + bh / 2, (v > 0 ? "+" : "") + fmt(v, 2), { fs: fs * 0.8, anchor: v >= 0 ? "start" : "end", fill: C.text, fw: 700 });
      });
      return s;
    },

    dots(p, w, h, fs) { // dot plot vs target line (no zero baseline needed for dots)
      const rows = p.rows, labW = p.labelW || fs * 10, top = fs * 1.9, bottom = fs * 1.7;
      const [lo, hi] = p.domain, plotW = w - labW - fs * 3.2, rowH = (h - top - bottom) / rows.length;
      const X = (v) => labW + ((v - lo) / (hi - lo)) * plotW;
      let s = "";
      for (let t = lo; t <= hi; t += p.step) {
        s += `<line x1="${X(t)}" y1="${top - fs * 0.5}" x2="${X(t)}" y2="${h - bottom}" stroke="${C.grid}"/>`;
        s += T(X(t), h - bottom + fs * 0.9, t + "%", { fs: fs * 0.78, anchor: "middle", fill: C.muted });
      }
      const xt = X(p.target);
      s += `<line x1="${xt}" y1="${top - fs * 0.9}" x2="${xt}" y2="${h - bottom}" stroke="${C.text}" stroke-width="2" stroke-dasharray="7 6"/>`;
      s += T(xt, fs * 0.55, p.targetLabel, { fs: fs * 0.8, anchor: "middle", fill: C.text, fw: 600 });
      rows.forEach((r, i) => {
        const y = top + i * rowH + rowH / 2, x = X(r.v);
        s += T(labW - fs * 0.8, y, r.label, { fs: fs * 0.88, anchor: "end" });
        s += `<line x1="${xt}" y1="${y}" x2="${x}" y2="${y}" stroke="${r.under ? C.orchid : C.dsky}" stroke-width="2.5"/>`;
        s += `<circle cx="${x}" cy="${y}" r="${fs * 0.42}" fill="${r.under ? C.orchid : C.dsky}" stroke="#0F2A54" stroke-width="2"/>`;
        const right = r.v >= p.target;
        s += T(right ? x + fs * 0.75 : x - fs * 0.75, y, fmt(r.v) + "%", { fs: fs * 0.8, anchor: right ? "start" : "end", fill: C.text, fw: 700 });
      });
      return s;
    },

    lollipopV(p, w, h, fs) { // vertical lollipops from the target line, one per horizon
      const rows = p.rows, left = fs * 2.6, top = fs * 1.2, bottom = fs * 2.6;
      const [lo, hi] = p.domain, plotW = w - left - fs * 0.5, plotH = h - top - bottom, colW = plotW / rows.length;
      const Y = (v) => top + (1 - (v - lo) / (hi - lo)) * plotH;
      let s = "";
      for (let t = lo; t <= hi; t += p.step) {
        s += `<line x1="${left}" y1="${Y(t)}" x2="${w}" y2="${Y(t)}" stroke="${C.grid}"/>`;
        s += T(left - fs * 0.4, Y(t), t + "%", { fs: fs * 0.72, anchor: "end", fill: C.muted });
      }
      const yt = Y(p.target);
      s += `<line x1="${left}" y1="${yt}" x2="${w}" y2="${yt}" stroke="${C.text}" stroke-width="2" stroke-dasharray="7 6"/>`;
      rows.forEach((r, i) => {
        const x = left + i * colW + colW / 2, y = Y(r.v), col = r.v < p.target ? C.orchid : C.dsky;
        s += `<line x1="${x}" y1="${yt}" x2="${x}" y2="${y}" stroke="${col}" stroke-width="2.5"/>`;
        s += `<circle cx="${x}" cy="${y}" r="${fs * 0.36}" fill="${col}" stroke="#0F2A54" stroke-width="2"/>`;
        s += T(x, h - bottom + fs * 0.8, r.label, { fs: fs * 0.72, anchor: "middle", fill: C.muted });
        if (r.flag) s += T(x, h - bottom + fs * 1.85, r.flag, { fs: fs * 0.72, anchor: "middle", fill: C.orchid, fw: 700 });
      });
      return s;
    },

    bands(p, w, h, fs) {
      const rows = p.rows, top = fs * 2.2, bottom = fs * 3.4, left = fs * 0.5;
      const max = p.max, plotH = h - top - bottom, colW = (w - left) / rows.length, bw = Math.min(colW * 0.5, fs * 6);
      let s = "";
      const base = h - bottom;
      for (let t = 0; t <= max; t += 5) {
        const y = base - (t / max) * plotH;
        s += `<line x1="${left}" y1="${y}" x2="${w}" y2="${y}" stroke="${C.grid}"/>`;
      }
      rows.forEach((r, i) => {
        const x = left + i * colW + (colW - bw) / 2, len = (r.v / max) * plotH;
        s += vbar(x, base, bw, len, r.color);
        s += T(x + bw / 2, base - len - fs * 0.8, fmt(r.v, 1) + "%", { fs: fs * 1.15, anchor: "middle", fill: C.text, fw: 700 });
        s += T(x + bw / 2, base + fs * 1.0, r.label, { fs: fs * 0.95, anchor: "middle", fill: C.text, fw: 700 });
        s += T(x + bw / 2, base + fs * 2.2, r.sub, { fs: fs * 0.75, anchor: "middle", fill: C.muted });
      });
      s += `<line x1="${left}" y1="${base}" x2="${w}" y2="${base}" stroke="${C.axis}"/>`;
      return s;
    },

    weights(p, w, h, fs) {
      const rows = p.rows, barH = Math.min(h * 0.32, fs * 2.4), gap = 2;
      let s = "", x = 0;
      const live = rows.filter((r) => r.v > 0), tot = live.reduce((a, r) => a + r.v, 0);
      const usable = w - gap * (live.length - 1);
      live.forEach((r, i) => {
        const sw = (r.v / tot) * usable;
        const rr = 4, isFirst = i === 0, isLast = i === live.length - 1;
        s += `<rect x="${x}" y="0" width="${sw}" height="${barH}" fill="${r.color}" rx="${isFirst || isLast ? rr : 0}"/>`;
        if (sw > fs * 4) s += T(x + sw / 2, barH / 2, Math.round(r.v * 100) + "%", { fs: fs * 0.95, anchor: "middle", fill: C.navy, fw: 700 });
        x += sw + gap;
      });
      // legend rows
      const ly = barH + fs * 1.4, colW = w / 2;
      rows.forEach((r, i) => {
        const lx = (i % 2) * colW, yy = ly + Math.floor(i / 2) * fs * 1.5;
        s += `<rect x="${lx}" y="${yy - fs * 0.4}" width="${fs * 0.8}" height="${fs * 0.8}" rx="3" fill="${r.v > 0 ? r.color : "none"}" stroke="${r.color}" stroke-width="1.5"/>`;
        s += T(lx + fs * 1.2, yy, r.label + " " + Math.round(r.v * 100) + "%", { fs: fs * 0.82, fill: r.v > 0 ? C.text2 : C.muted });
      });
      return s;
    },

    heat(p, w, h, fs) {
      const rows = p.rows, cols = p.cols, labW = p.labelW || fs * 9, top = fs * 1.6;
      const cw = (w - labW) / cols.length, ch = (h - top) / rows.length;
      let s = "";
      cols.forEach((c, j) => { s += T(labW + j * cw + cw / 2, fs * 0.6, c, { fs: fs * 0.78, anchor: "middle", fill: C.muted, fw: 600 }); });
      rows.forEach((r, i) => {
        s += T(labW - fs * 0.6, top + i * ch + ch / 2, r.label, { fs: fs * 0.82, anchor: "end" });
        r.v.forEach((v, j) => {
          const a = Math.max(0, Math.min(1, (v - p.lo) / (p.hi - p.lo)));
          const op = 0.08 + a * 0.88;
          s += `<rect x="${labW + j * cw + 1}" y="${top + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="4" fill="${C.dsky}" fill-opacity="${op.toFixed(3)}"/>`;
          s += T(labW + j * cw + cw / 2, top + i * ch + ch / 2, v.toFixed(2), { fs: fs * 0.74, anchor: "middle", fill: op > 0.62 ? C.navy : C.text, fw: op > 0.62 ? 700 : 500 });
        });
      });
      return s;
    },

    timeline(p, w, h, fs) {
      const d0 = Date.parse(p.start), d1 = Date.parse(p.end), DAY = 86400000;
      const X = (d) => ((Date.parse(d) - d0) / (d1 - d0)) * w;
      const base = h * 0.6, arcH = h * 0.26;
      let s = "";
      // month ticks
      p.months.forEach((m) => {
        const x = X(m.d);
        if (x > w - fs * 4.5) return;
        s += `<line x1="${x}" y1="${base}" x2="${x}" y2="${h - fs * 1.4}" stroke="${C.grid}"/>`;
        s += T(x + fs * 0.3, h - fs * 0.7, m.label, { fs: fs * 0.78, fill: C.muted });
      });
      s += `<line x1="0" y1="${base}" x2="${w}" y2="${base}" stroke="${C.axis}" stroke-width="1.5"/>`;
      p.stages.forEach((st) => {
        st.origins.forEach((o) => {
          const xa = X(o), xb = xa + (p.horizon * DAY / (d1 - d0)) * w;
          s += `<path d="M${xa} ${base} Q${(xa + xb) / 2} ${base - arcH * 1.9} ${xb} ${base}" fill="none" stroke="${st.color}" stroke-width="2" stroke-opacity="0.85"/>`;
        });
        st.origins.forEach((o) => { s += `<circle cx="${X(o)}" cy="${base}" r="${fs * 0.3}" fill="${st.color}" stroke="#0F2A54" stroke-width="2"/>`; });
        // target window bracket
        const ta = X(st.targets[0]), tb = X(st.targets[1]);
        s += `<rect x="${ta}" y="${base + fs * 0.9}" width="${tb - ta}" height="${fs * 0.45}" rx="${fs * 0.22}" fill="${st.color}" opacity="0.9"/>`;
        s += T((ta + tb) / 2, base + fs * 2.2, st.targetLabel, { fs: fs * 0.72, anchor: "middle", fill: C.text2 });
        // stage label
        const cx = (X(st.origins[0]) + X(st.origins[st.origins.length - 1])) / 2 + (st.dx || 0);
        s += T(cx, fs * 0.7, st.name, { fs: fs * 0.95, anchor: "middle", fill: C.text, fw: 700 });
        s += T(cx, fs * 1.85, st.sub, { fs: fs * 0.75, anchor: "middle", fill: st.color, fw: 600 });
      });
      if (p.forward) {
        const fa = X(p.forward.start), fb = X(p.forward.end);
        s += `<line x1="${fa}" y1="${base + fs * 1.12}" x2="${fb}" y2="${base + fs * 1.12}" stroke="${C.text}" stroke-width="${fs * 0.42}" stroke-dasharray="${fs * 0.5} ${fs * 0.35}"/>`;
        const cx = (fa + fb) / 2;
        s += T(cx, fs * 0.7, p.forward.name, { fs: fs * 0.95, anchor: "middle", fill: C.text, fw: 700 });
        s += T(cx, fs * 1.85, p.forward.sub, { fs: fs * 0.75, anchor: "middle", fill: C.text2, fw: 600 });
      }
      return s;
    },
  };

  /* ---------- shapes ---------- */
  function coilPath() {
    const pts = [];
    for (let t = 0; t <= Math.PI * 2 * 4.25; t += 0.12) pts.push([50 + 34 * Math.cos(t), 20 + t * 5.6 + 9 * Math.sin(t)]);
    return "M" + pts.map((q) => q.map((v) => v.toFixed(1)).join(" ")).join(" L");
  }
  const DEFS = `<defs>
    <linearGradient id="g-irid-v" x1="0" y1="0" x2="0.4" y2="1"><stop offset="0" stop-color="#60F0D8"/><stop offset=".3" stop-color="#9090F0"/><stop offset=".6" stop-color="#D890D8"/><stop offset="1" stop-color="#6030D8"/></linearGradient>
    <linearGradient id="g-prism-a" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset=".3" stop-color="#9090F0"/><stop offset="1" stop-color="#6030D8"/></linearGradient>
    <linearGradient id="g-prism-b" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#60F0D8"/><stop offset=".55" stop-color="#D890D8"/><stop offset="1" stop-color="#3A1E9E"/></linearGradient>
    <linearGradient id="g-sheen" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".6" stop-color="#fff" stop-opacity=".8"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient></defs>`;
  function shape(kind, label, size) {
    if (kind === "torus") return `<div class="shape shape-glow"><div class="torus"></div></div>`;
    if (kind === "orb") return `<div class="shape"><div class="orb"></div></div>`;
    if (kind === "prism") return `<div class="shape shape-glow" style="padding:10%"><svg viewBox="0 0 100 100">${DEFS}
        <polygon points="50,4 8,78 50,96" fill="url(#g-prism-a)"/><polygon points="50,4 92,78 50,96" fill="url(#g-prism-b)"/>
        <polygon points="50,4 50,96 8,78" fill="url(#g-sheen)" opacity=".55"/><polyline points="50,4 50,96" stroke="rgba(255,255,255,.7)" stroke-width="1" fill="none"/></svg></div>`;
    if (kind === "coil") { const d = coilPath(); return `<div class="shape shape-glow" style="padding:8%"><svg viewBox="0 0 100 190">${DEFS}
        <path d="${d}" fill="none" stroke="url(#g-irid-v)" stroke-width="11" stroke-linecap="round"/>
        <path d="${d}" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="2.5" stroke-linecap="round" transform="translate(-2 -3)"/></svg></div>`; }
    if (kind === "badge") return `<div class="ring-badge" style="font-size:${size * 0.3}px">${esc(label)}</div>`;
    return "";
  }

  /* ---------- painter ---------- */
  const root = document.getElementById("deck");
  const overflow = [];
  DATA.slides.forEach((sl, si) => {
    const slide = document.createElement("section");
    slide.className = "slide";
    slide.id = "slide-" + (si + 1);
    slide.style.width = W + "px";
    slide.style.height = H + "px";
    slide.innerHTML = `<div class="bg" id="bg-${si + 1}"></div>`;
    sl.els.forEach((e) => {
      const d = document.createElement("div");
      d.className = "el";
      if (e.id) d.id = e.id;
      Object.assign(d.style, { left: e.x + "px", top: e.y + "px", width: e.w + "px", height: e.h + "px" });
      if (e.k === "rect") {
        d.className += " card" + (e.variant ? " " + e.variant : "");
        if (e.stroke) d.style.borderColor = e.stroke;
        if (e.r !== undefined) d.style.borderRadius = e.r + "px";
      } else if (e.k === "text") {
        d.className += " tx" + (e.valign ? " v-" + e.valign : "");
        d.innerHTML = e.paras.map((p) => {
          const st = [`font-size:${p.size}px`, `line-height:${p.lh}`, `color:${p.color}`, `font-weight:${p.weight}`,
            `text-align:${p.align || "left"}`, `margin-top:${p.sb || 0}px`, `--lh:${p.lh}`];
          if (p.ls) st.push(`letter-spacing:${p.ls}em`);
          if (p.upper) st.push("text-transform:uppercase");
          if (p.bullet) st.push(`padding-left:${p.size * 1.1}px`, `--bc:${p.bulletColor || C.sky}`);
          const runs = p.runs.map((r) => {
            const rs = [];
            if (r.c) rs.push(`color:${r.c}`);
            if (r.b) rs.push("font-weight:700");
            if (r.i) rs.push("font-style:italic");
            return `<span${rs.length ? ` style="${rs.join(";")}"` : ""}>${esc(r.t)}</span>`;
          }).join("");
          return `<p class="${p.font === "head" ? "f-head" : ""}${p.bullet ? " bul" : ""}${p.grad ? " grad-text" : ""}" style="${st.join(";")}">${runs}</p>`;
        }).join("");
      } else if (e.k === "chart") {
        d.className += " chart";
        d.innerHTML = `<svg width="${e.w}" height="${e.h}" viewBox="0 0 ${e.w} ${e.h}">${charts[e.chart](e.p, e.w, e.h, e.fs)}</svg>`;
      } else if (e.k === "asset") {
        d.innerHTML = shape(e.shape, e.label, e.w);
      } else if (e.k === "table") {
        const fs = e.fs, pad = `${fs * 0.5}px ${fs * 0.6}px`;
        const head = e.cols.map((c) => `<th class="${c.align === "right" ? "num" : ""}" style="font-size:${fs * 0.72}px;padding:${pad};width:${c.w}px">${esc(c.t)}</th>`).join("");
        const body = e.rows.map((r) => "<tr>" + r.map((c, j) => {
          const al = e.cols[j].align === "right" ? "num" : "";
          const inner = c.pill ? `<span class="pill" style="background:${c.pill}">${esc(c.t)}</span>` : esc(c.t);
          return `<td class="${al}" style="font-size:${fs}px;padding:${pad};height:${e.rowH}px;${c.c ? "color:" + c.c + ";" : ""}${c.b ? "font-weight:700;" : ""}">${inner}</td>`;
        }).join("") + "</tr>").join("");
        d.innerHTML = `<table class="tbl"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      }
      slide.appendChild(d);
    });
    root.appendChild(slide);
  });

  function checkOverflow() {
    document.querySelectorAll(".tx, .el table").forEach((n) => {
      const box = n.classList.contains("tx") ? n : n.parentElement;
      if (box.scrollHeight > box.clientHeight + 2 || box.scrollWidth > box.clientWidth + 2) {
        overflow.push({ slide: box.closest(".slide").id, id: box.id || "", text: box.textContent.slice(0, 60),
          over_h: box.scrollHeight - box.clientHeight, over_w: box.scrollWidth - box.clientWidth });
      }
    });
    // chart text escaping its box
    document.querySelectorAll(".chart svg").forEach((svg) => {
      const host = svg.parentElement.getBoundingClientRect();
      svg.querySelectorAll("text").forEach((t) => {
        const b = t.getBoundingClientRect();
        if (b.left < host.left - 2 || b.right > host.right + 2 || b.top < host.top - 2 || b.bottom > host.bottom + 2)
          overflow.push({ slide: svg.closest(".slide").id, id: svg.parentElement.id, text: "chart text: " + t.textContent,
            over_w: Math.round(Math.max(host.left - b.left, b.right - host.right)), over_h: Math.round(Math.max(host.top - b.top, b.bottom - host.bottom)) });
      });
    });
    window.__overflow = overflow;
    window.__ready = true;
  }
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => setTimeout(checkOverflow, 50));
})();
