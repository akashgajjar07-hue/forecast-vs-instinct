"""Build the case-study deck (PDF + PPTX), the LinkedIn carousel (PDF) and the LinkedIn post draft.

Every number is read here from finished pipeline outputs (nothing is trained or re-scored):
  outputs/holdout_results.json, outputs/selection.json, outputs/intervals.json, outputs/trust_weights.json,
  outputs/trust_calibration.json, outputs/lgbm_feature_importance.json, outputs/eda/*.json,
  config/scope.json, config/split_calendar.json,
  outputs/backtests/*.parquet + outputs/holdout/*.parquet (recomputed slices, cross-checked against the JSONs),
  docs/data/*.json (the published dashboard export of the same outputs: arena slices, forward origin, S&OP, scenario).
Notes in docs/agent_notes/notebooks.md are cross-checked: the numbers quoted there are recomputed and asserted.

Pipeline: display lists (Python) -> presentations/build/{deck,carousel}.html (templates in presentations/src/)
-> Playwright (Chromium) -> PDF + PNG previews + chart/shape screenshots -> python-pptx deck.

Run: .venv/bin/python scripts/10_build_presentations.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentations"
SRC = OUT / "src"
BUILD = OUT / "build"
PREV = OUT / "previews"
ASSETS = BUILD / "assets"

REPO_URL = "https://github.com/akashgajjar07-hue/forecast-vs-instinct"
LIVE_URL = "https://akashgajjar07-hue.github.io/forecast-vs-instinct/"

# design tokens (docs/styles.css)
BG, PANEL, DEEP, CARD = "#01204C", "#001634", "#00112D", "#0F2A54"
TEXT, TEXT2, MUTED, SKY = "#FFFFFF", "#D7D0E7", "#9AA4AD", "#7BBDE6"
MINT, PERI, ORCHID, DSKY, VIOLET = "#60F0D8", "#9090F0", "#D685C2", "#68BEF3", "#AE3DFF"
ACCEPT, MONITOR, REVIEW = "#60F0D8", "#F5C26B", "#FF6B8A"
BAND_COL = {"Accept": ACCEPT, "Monitor": MONITOR, "Review": REVIEW}
CAVEAT_STROKE = "rgba(245, 194, 107, 0.55)"
MINUS = "−"


def jload(rel):
    return json.loads((ROOT / rel).read_text())


def sg(v, d=1, plus=False):
    s = f"{abs(v):,.{d}f}"
    if v < 0 and round(abs(v), d) != 0:
        return MINUS + s
    return ("+" + s) if plus and v > 0 else s


def pct(v, d=1, plus=False):
    return sg(v, d, plus) + "%"


def title_case(fam):
    return {"BREAD/BAKERY": "Bread/Bakery", "GROCERY I": "Grocery I"}.get(fam, fam.title())


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"cross-check failed for {what}: {a} vs {b}"


# ----------------------------------------------------------------------------- numbers
def wape(df):
    return 100 * (df.actual - df.forecast).abs().sum() / df.actual.sum()


def bias(df):
    return 100 * (df.forecast - df.actual).sum() / df.actual.sum()


def evaluable(df):
    return df[(df.closed == 0) & df.actual.notna()]


def compute():
    hr = jload("outputs/holdout_results.json")
    sel = jload("outputs/selection.json")
    itv = jload("outputs/intervals.json")
    tw = jload("outputs/trust_weights.json")
    tc = jload("outputs/trust_calibration.json")
    fi = jload("outputs/lgbm_feature_importance.json")
    tune = jload("outputs/lgbm_tuning.json")
    scope = jload("config/scope.json")
    cal = jload("config/split_calendar.json")
    extras = jload("outputs/eda/eda_extras.json")
    demand = jload("outputs/eda/demand.json")
    analogue = jload("outputs/eda/ecuador_analogue.json")
    arena = jload("docs/data/arena.json")
    dint = jload("docs/data/intervals.json")
    trust = jload("docs/data/trust.json")
    ov = jload("docs/data/overview.json")
    sop = jload("docs/data/sop.json")
    scen = jload("docs/data/scenario.json")
    meta = jload("docs/data/meta.json")
    nb_notes = (ROOT / "docs/agent_notes/notebooks.md").read_text()
    b_notes = (ROOT / "docs/agent_notes/agent_b.md").read_text()

    N = {}
    # --- scope
    N["n_stores"] = len(scope["stores"])
    N["n_fam"] = len(scope["families"])
    N["n_series"] = N["n_stores"] * N["n_fam"]
    N["n_eligible"] = scope["n_eligible"]
    N["n_cities"] = len({s["city"] for s in scope["stores"]})
    N["n_types"] = len({s["type"] for s in scope["stores"]})
    N["model_start"] = cal["model_start"]
    N["last_actual"] = cal["last_actual"]
    N["horizon"] = cal["horizon"]
    N["points_per_cycle"] = N["n_series"] * N["horizon"]
    N["stages"] = {k: v for k, v in cal["stages"].items()}
    N["target_windows"] = cal["target_windows"]

    # --- EDA
    dow = demand["dow_profile"]
    N["dow"] = dow
    N["sunday_peak"] = sum(1 for f, v in dow.items() if int(np.argmax(v)) == 6)
    # owner rule: no family volume shares (volume-weighted); describe families by volatility instead
    cvf = pd.DataFrame(demand["volatility"]).groupby("family")["cv"].mean()
    N["top_cv_fam"], N["top_cv"] = cvf.idxmax(), float(cvf.max())
    N["txn_corr"] = extras["transactions"]["pooled_corr_daily"]
    N["produce_zero_before"] = 100 * extras["produce_recording_gap"]["zero_share_before_window"]["PRODUCE"]
    an = {a["family"]: a["change_pct"] for a in analogue}
    N["analogue_min"] = min(an.items(), key=lambda t: t[1])
    N["analogue_max"] = max(an.items(), key=lambda t: t[1])
    N["analogue_n"] = analogue[0]["n_holidays"]

    # --- leaderboard
    lb = hr["leaderboard"]
    N["lb"] = lb
    champ = next(r for r in lb if r["model"] == "lightgbm" and r["variant"] == "A")
    base = sorted([r for r in lb if r["model"] != "lightgbm"], key=lambda r: r["wape"])
    N["champ_wape"], N["champ_bias"] = champ["wape"], champ["bias"]
    N["best_base"] = base[0]
    N["err_cut"] = 100 * (1 - champ["wape"] / base[0]["wape"])
    N["n_holdout_pts"] = champ["n"]
    N["n_holdout_origins"] = len(cal["stages"]["holdout"])
    lgb_b = next(r for r in lb if r["model"] == "lightgbm" and r["variant"] == "B")
    N["b_wape"] = lgb_b["wape"]
    fam = hr["by_family"]
    N["fam_champ"] = {r["family"]: r["lightgbm"] for r in fam}
    N["fam_wins"] = sum(1 for r in fam if r["holdout_best"] == "lightgbm")
    N["sel_champs"] = sum(1 for v in sel["champions"].values() if v["model"] == "lightgbm")
    lo = min(N["fam_champ"].items(), key=lambda t: t[1]); hi = max(N["fam_champ"].items(), key=lambda t: t[1])
    N["fam_lo"], N["fam_hi"] = lo, hi
    bases = ["naive", "seasonal_naive", "moving_average", "holt_winters"]
    N["fam_best_base"] = {r["family"]: min(((m, r[m]) for m in bases), key=lambda t: t[1]) for r in fam}

    # arena slices (holdout)
    am = [m for m in arena["metrics"] if m["stage"] == "holdout"]

    def sl(model, variant, st, sv):
        return next(m for m in am if m["model"] == model and m["variant"] == variant and m["slice_type"] == st and m["slice_value"] == sv)
    N["hol_ma"] = sl("moving_average", "A", "holiday", "holiday")["wape"]
    N["hol_lgb"] = sl("lightgbm", "A", "holiday", "holiday")["wape"]
    N["hol_n"] = sl("lightgbm", "A", "holiday", "holiday")["n"]
    N["nopromo_n"] = sl("lightgbm", "A", "promo", "no_promo")["n"]
    hz = {}
    for m in am:
        if m["slice_type"] == "horizon" and not (m["model"] == "lightgbm" and m["variant"] == "B"):
            hz.setdefault(m["slice_value"], []).append((m["wape"], m["model"]))
    N["hz_total"] = len(hz)
    N["hz_wins"] = sum(1 for v in hz.values() if min(v)[1] == "lightgbm")

    # per-series wins from backtests (holdout stage)
    bt = {}
    for m in ["lightgbm_A", "lightgbm_B"] + bases:
        bt[m] = pd.read_parquet(ROOT / f"outputs/backtests/{m}.parquet")
    per = {}
    for m in ["lightgbm_A"] + bases:
        h = evaluable(bt[m][bt[m].stage == "holdout"])
        per[m] = h.groupby("series_id").apply(wape, include_groups=False)
    per = pd.DataFrame(per)
    best = per.idxmin(axis=1)
    N["series_wins"] = int((best == "lightgbm_A").sum())
    N["series_hw"] = int((best == "holt_winters").sum())
    m_ = re.search(r"best in only (\d+)/(\d+) \(Holt-Winters wins (\d+)\)", nb_notes)
    assert m_ and int(m_.group(1)) == N["series_wins"] and int(m_.group(3)) == N["series_hw"], "series wins mismatch"
    close(wape(evaluable(bt["lightgbm_A"][bt["lightgbm_A"].stage == "holdout"])), champ["wape"], 1e-6, "holdout WAPE")

    # A vs B by stage
    ab = {}
    for st in ["selection", "calibration", "holdout"]:
        a = wape(evaluable(bt["lightgbm_A"][bt["lightgbm_A"].stage == st]))
        b = wape(evaluable(bt["lightgbm_B"][bt["lightgbm_B"].stage == st]))
        ab[st] = (a, b, b - a)
    N["ab"] = ab
    close(ab["holdout"][1], lgb_b["wape"], 1e-6, "variant B holdout")
    vc = [r for r in arena["variant_comparison"] if r["stage"] == "holdout" and r["family"] != "all"]
    N["ab_fam"] = sorted(((r["family"], r["wape_B"] - r["wape_A"]) for r in vc), key=lambda t: -t[1])
    N["b_better"] = [f for f, d in N["ab_fam"] if d < 0]
    fiA = {r["feature"]: r["gain_share_pct"] for r in fi["variants"]["A"]}
    N["fi_promo"] = fiA["promo_t"] + fiA["promo_t_rel28"]
    N["fi_top"] = max(fiA.items(), key=lambda t: t[1])
    N["promo_rank"] = sorted(fiA, key=lambda k: -fiA[k]).index("promo_t") + 1
    m_ = re.search(r"A deliberate one-day look-ahead injected into `lag_1` made (\d+) tests fail", b_notes)
    assert m_, "agent_b note on look-ahead test not found"
    N["lookahead_fail"] = int(m_.group(1))
    N["tune_n"] = len(tune["candidates"])

    # --- intervals
    ci = pd.read_parquet(ROOT / "outputs/holdout/champion_intervals.parquet")
    hci = evaluable(ci[ci.stage == "holdout"])
    N["cov"] = 100 * ((hci.actual >= hci.p10) & (hci.actual <= hci.p90)).mean()
    close(N["cov"], hr["champion_system"]["coverage"], 1e-6, "coverage")
    N["below"] = 100 * (hci.actual < hci.p10).mean()
    N["above"] = 100 * (hci.actual > hci.p90).mean()
    N["width_pct"] = 100 * (hci.p90 - hci.p10).sum() / hci.forecast.sum()
    N["width"] = hr["champion_system"]["mean_width"]
    N["width_index"] = jload("docs/data/intervals.json")["holdout"]["mean_width"]  # index points (series base = 100)
    N["cov_target"] = 100 * itv["level"]
    N["cov_loo"] = itv["calibration_coverage_leave_one_origin_out"]
    N["cov_fam"] = [(r["family"], r["coverage"]) for r in dint["holdout"]["by_family"]]
    N["cov_hz"] = [(r["horizon"], r["coverage"]) for r in dint["holdout"]["by_horizon"]]
    o0 = pd.Timestamp(cal["stages"]["holdout"][0])
    N["hz_dow"] = {h: (o0 + pd.Timedelta(days=h)).day_name()[:3] for h, _ in N["cov_hz"]}
    N["sun_hz"] = [h for h, d in N["hz_dow"].items() if d == "Sun"]
    flat = " ".join(nb_notes.split())
    for want, got in [("10.5 % of actuals fell below P10", N["below"]), ("9.2 % above P90", N["above"]), ("37.3 % of the forecast", N["width_pct"])]:
        assert want in flat, f"notebooks.md no longer says: {want}"
        close(round(got, 1), float(want.split(" ")[0]), 1e-9, want)

    # --- trust
    tv = hr["trust_validation"]
    N["tv"] = tv
    N["weights"] = tw["weights"]
    N["bands_thr"] = {b["band"]: b["min"] for b in tw["bands"]}
    cands = {c["name"]: c for c in tc["candidates"]}
    N["cal_rho_chosen"] = cands[tw["chosen"]]["spearman"]
    N["cal_rho_default"] = cands["default"]["spearman"]
    N["grid_n"] = trust["calibration_choice"]["grid_size"]
    ts = pd.read_parquet(ROOT / "outputs/holdout/trust_scores.parquet")
    win = hci.groupby(["series_id", "origin"]).apply(
        lambda g: pd.Series({"ae": (g.actual - g.forecast).abs().sum(), "a": g.actual.sum()}), include_groups=False).reset_index()
    win["wape"] = 100 * win.ae / win.a
    t = ts.merge(win, on=["series_id", "origin"])
    assert len(t) == tv["n_windows"]
    close(spearmanr(t.score, t.wape)[0], tv["spearman"], 6e-4, "trust spearman")
    close(spearmanr(t.stability, t.wape)[0], tv["volatility_only_spearman"], 6e-4, "volatility spearman")
    dw = cands["default"]["weights"]
    N["hold_rho_default"] = spearmanr(sum(t[k] * v for k, v in dw.items()), t.wape)[0]
    N["rho_acc_only"] = spearmanr(t.accuracy, t.wape)[0]
    N["rho_score_acc"] = spearmanr(t.score, t.accuracy)[0]
    fl = t.hard_flags.apply(len) > 0
    N["flag_n"] = int(fl.sum())
    N["flag_share"] = 100 * fl.mean()
    N["flag_wape"] = 100 * t[fl].ae.sum() / t[fl].a.sum()
    N["noflag_wape"] = 100 * t[~fl].ae.sum() / t[~fl].a.sum()
    N["review_by_flag_only"] = int(((t.band == "Review") & (t.score >= N["bands_thr"]["Monitor"])).sum())
    N["review_n"] = int((t.band == "Review").sum())
    for a_, b_ in [(N["flag_n"], 67), (round(N["flag_wape"], 2), 8.71), (round(N["noflag_wape"], 2), 13.02),
                   (N["review_by_flag_only"], 65), (round(N["rho_score_acc"], 3), 0.971), (round(N["hold_rho_default"], 3), -0.503),
                   (round(N["rho_acc_only"], 3), -0.647)]:
        assert str(b_).replace("-", "−") in nb_notes or str(b_) in nb_notes, f"note value {b_} not in notebooks.md"
        close(a_, b_, 1e-9, f"notebooks.md value {b_}")
    N["wb"] = {r["band"]: r for r in tv["wape_by_band"]}

    # --- forward origin / planner
    lat = trust["latest"]
    N["fwd_origin"] = trust["latest_origin"]
    N["fwd_counts"] = {b: sum(1 for r in lat if r["band"] == b) for b in ["Accept", "Monitor", "Review"]}
    # per-series 14-day volumes are not published (index only): use the pipeline's trust scores (units, local)
    tf = ts[ts.origin == pd.Timestamp(N["fwd_origin"])]
    assert len(tf) == 96 and dict(zip(tf.series_id, tf.band)) == {r["series_id"]: r["band"] for r in lat}
    tot = float(tf.forecast_14d.sum())
    # owner rule: no volume-weighted shares (they add equations on the private bases); share of series instead
    N["review_series_share"] = 100 * float((tf.band == "Review").mean())
    N["fwd_total"] = ov["kpis"]["forecast_14d_total"]["value"]
    close(tot, N["fwd_total"], 1e-3, "forward total")
    # outlook = equal-weighted mean index of the 96 series: compare the forecast with the last 14 observed days
    hist = [h["mean_index"] for h in ov["outlook"]["history"]][-N["horizon"]:]
    fidx = [f["mean_index"] for f in ov["outlook"]["forecast"]]
    hist = [v for v in hist if v is not None]
    N["fwd_vs_last"] = 100 * ((sum(fidx) / len(fidx)) / (sum(hist) / len(hist)) - 1)
    N["fwd_start"], N["fwd_end"] = meta["data_window"]["forward_start"], meta["data_window"]["forward_end"]
    N["top_exc"] = ov["top_exceptions"][:5]
    N["fva"] = sop["fva"]
    # persistent bias: Beverages champion bias on calibration and holdout
    cA = evaluable(bt["lightgbm_A"][(bt["lightgbm_A"].stage == "calibration") & (bt["lightgbm_A"].family == "BEVERAGES")])
    N["bev_bias_cal"] = bias(cA)
    N["bev_bias_hold"] = bias(hci[hci.family == "BEVERAGES"])
    close(bias(hci), champ["bias"], 1e-6, "holdout bias sign convention")
    assert "Beverages" in sop["summary"]["persistent_bias"][0]
    N["events"] = scen["events"]
    N["citation"] = meta["citation"]
    return N


# ----------------------------------------------------------------------------- display-list builder
RUN_RE = re.compile(r"(\*\*.+?\*\*|__.+?__|\{\{#[0-9A-Fa-f]{6}:.+?\}\})")


def runs(s):
    out = []
    for part in RUN_RE.split(s):
        if not part:
            continue
        if part.startswith("**"):
            out.append({"t": part[2:-2], "b": True, "c": TEXT})
        elif part.startswith("{{#"):
            out.append({"t": part[10:-2], "b": True, "c": part[2:9]})
        elif part.startswith("__"):
            out.append({"t": part[2:-2], "b": True, "c": MINT})
        else:
            out.append({"t": part})
    return out


def P(s, size, color=TEXT2, font="body", weight=500, lh=1.35, bullet=False, sb=0, align="left", **kw):
    d = {"runs": runs(s), "size": size, "color": color, "font": font, "weight": weight, "lh": lh,
         "bullet": bullet, "sb": sb, "align": align}
    d.update(kw)
    return d


class Slide:
    def __init__(self, idx, fmt):
        self.idx, self.fmt, self.els, self.notes = idx, fmt, [], ""

    def add(self, e):
        e["id"] = f"s{self.idx}-e{len(self.els)}"
        self.els.append(e)
        return e

    def text(self, x, y, w, h, paras, valign=None):
        return self.add({"k": "text", "x": x, "y": y, "w": w, "h": h, "paras": paras, "valign": valign})

    def rect(self, x, y, w, h, variant=None, stroke=None, r=20):
        return self.add({"k": "rect", "x": x, "y": y, "w": w, "h": h, "variant": variant, "stroke": stroke, "r": r})

    def asset(self, shape, x, y, w, h, label=None):
        return self.add({"k": "asset", "shape": shape, "x": x, "y": y, "w": w, "h": h, "label": label})

    def chart(self, chart, x, y, w, h, p, fs):
        return self.add({"k": "chart", "chart": chart, "x": x, "y": y, "w": w, "h": h, "p": p, "fs": fs})

    def table(self, x, y, w, h, cols, rows, fs, rowH):
        return self.add({"k": "table", "x": x, "y": y, "w": w, "h": h, "cols": cols, "rows": rows, "fs": fs, "rowH": rowH})

    # composite blocks
    def card(self, x, y, w, h, title=None, paras=(), pad=32, ts=24, tcolor=TEXT, stroke=None, variant=None, eyebrow=None, ecolor=SKY):
        self.rect(x, y, w, h, variant=variant, stroke=stroke)
        ps = []
        if eyebrow:
            ps.append(P(eyebrow, ts * 0.68, ecolor, weight=700, ls=0.12, upper=True))
        if title:
            ps.append(P(title, ts, tcolor, font="head", weight=700, lh=1.2, sb=8 if eyebrow else 0))
        for i, p in enumerate(paras):
            if i == 0 and (title or eyebrow) and not p.get("sb"):
                p = dict(p, sb=ts * 0.6)
            ps.append(p)
        self.text(x + pad, y + pad - 4, w - 2 * pad, h - 2 * pad + 8, ps)

    def kpi(self, x, y, w, h, value, label, detail=None, color=MINT, vs=64, ls=22, ds=18, pad=28, variant=None, stroke=None):
        self.rect(x, y, w, h, variant=variant, stroke=stroke)
        ps = [P(value, vs, color, font="head", weight=800, lh=1.05), P(label, ls, TEXT, weight=700, sb=vs * 0.22, lh=1.25)]
        if detail:
            ps.append(P(detail, ds, MUTED, sb=6, lh=1.35))
        self.text(x + pad, y + pad - 4, w - 2 * pad, h - 2 * pad + 8, ps, valign="middle")

    def chart_card(self, x, y, w, h, title, chart, p, fs, caption=None, pad=30, ts=22, cap_h=None, eyebrow=None, ecolor=SKY):
        self.rect(x, y, w, h)
        th = 0
        ps = []
        if eyebrow:
            ps.append(P(eyebrow, ts * 0.72, ecolor, weight=700, ls=0.12, upper=True)); th += ts * 0.72 * 1.35 + 6
        if title:
            ps.append(P(title, ts, TEXT, font="head", weight=700, lh=1.2, sb=6 if eyebrow else 0)); th += ts * 1.25 + 4
        if ps:
            self.text(x + pad, y + pad - 4, w - 2 * pad, th + 8, ps)
        cap_h = cap_h if cap_h is not None else (fs * 2.9 if caption else 0)
        cy = y + pad + th + (fs * 0.9 if ps else 0)
        ch = y + h - pad - cap_h - cy
        self.chart(chart, x + pad, cy, w - 2 * pad, ch, p, fs)
        if caption:
            self.text(x + pad, y + h - pad - cap_h + 6, w - 2 * pad, cap_h, [P(caption, fs * 0.82, MUTED, lh=1.35)], valign="bottom")


# deck geometry (1920 x 1080 px; 1 px = 6350 EMU = 0.5 pt)
DW, DH, DM = 1920, 1080, 96
BODY_Y, BODY_B = 286, 990
TOTAL_DECK = 10


def deck_header(s, num, kicker, title, lede=None):
    s.asset("badge", DM, 72, 78, 78, label=num)
    s.text(200, 66, 1620, 30, [P(kicker, 19, SKY, weight=700, ls=0.14, upper=True)])
    s.text(200, 96, 1640, 72, [P(title, 50, TEXT, font="head", weight=800, lh=1.25)])
    if lede:
        s.text(200, 178, 1624, 92, [P(lede, 24, TEXT2, lh=1.4)])


def deck_footer(s, page, N, dark=False):
    s.text(DM, 1022, 1300, 30, [P(f"Data: Corporación Favorita store sales (Ecuador), Kaggle · observed sales, not unconstrained demand · "
                                 f"{N['n_series']} series · Forecast vs Instinct", 15, MUTED)])
    s.text(DW - DM - 200, 1022, 200, 30, [P(f"{page:02d} / {TOTAL_DECK:02d}", 15, MUTED, align="right", weight=600)])


def deck(N):
    S = []
    new = lambda: S.append(Slide(len(S) + 1, "deck")) or S[-1]
    wb = N["wb"]
    hold_from, hold_to = N["target_windows"]["holdout"]

    # 1 ---------------------------------------------------------------- title
    s = new()
    s.asset("torus", 1180, -150, 820, 820)
    s.asset("prism", 1500, 600, 260, 260)
    s.asset("coil", 1700, 470, 190, 360)
    s.asset("orb", 1130, 520, 120, 120)
    s.text(DM, 120, 1200, 32, [P("Case study · AI-powered FMCG demand planning", 20, SKY, weight=700, ls=0.14, upper=True)])
    s.text(DM, 166, 1250, 260, [P("Forecast", 116, TEXT, font="head", weight=800, lh=1.04),
                                P("vs Instinct", 116, TEXT, font="head", weight=800, lh=1.04, grad=True)])
    s.text(DM, 440, 1150, 50, [P("AI-Powered FMCG Demand Planning & Forecast Intelligence", 32, TEXT, weight=600)])
    s.text(DM, 494, 1150, 46, [P("Beyond predicting demand. Knowing when to trust the prediction.", 28, MINT, weight=500, i=True)])
    s.text(DM, 552, 1100, 70, [P(f"{N['n_stores']} stores × {N['n_fam']} product families = {N['n_series']} daily series of real grocery sales "
                                 f"(Corporación Favorita, Ecuador). Five models compete; one index says which forecasts to trust.", 22, TEXT2, lh=1.45)])
    kw = 440
    bb = N["best_base"]
    s.kpi(DM, 690, kw, 250, pct(N["champ_wape"]), "Holdout WAPE, LightGBM champion",
          f"vs {pct(bb['wape'])} for the best baseline ({'Holt-Winters' if bb['model']=='holt_winters' else bb['model']}); {N['n_holdout_origins']} holdout origins, scored once",
          color=MINT, vs=60)
    s.kpi(DM + kw + 28, 690, kw, 250, f"{N['tv']['error_ratio_review_vs_accept']:.2f}×", "Review vs Accept error",
          "holdout WAPE of Review-band windows relative to Accept-band (Trust Index is a 0–100 index, not a probability)", color=REVIEW, vs=60)
    s.kpi(DM + 2 * (kw + 28), 690, kw, 250, pct(N["cov"]), "P10–P90 interval coverage",
          f"of holdout actuals inside the band; empirical {N['cov_target']:.0f}% target, not a guarantee", color=SKY, vs=60)
    s.text(DM, 972, 1500, 34, [P("**Akash Gajjar** · MSc Supply Chain Management, University of Liverpool (2025) · retail operations & inventory, Primark and Uniqlo", 19, TEXT2)])
    s.text(DM, 1016, 1700, 30, [P(f"Data: Alexis Cook, DanB, inversion, Ryan Holbrook (2021), Store Sales – Time Series Forecasting, Kaggle. Real Ecuadorian data, not UAE data.", 15, MUTED)])
    s.notes = "Title. Headline numbers: champion holdout WAPE, Review/Accept error ratio, interval coverage."

    # 2 ---------------------------------------------------------------- dilemma
    s = new()
    deck_header(s, "01", "The planner's dilemma", "Which forecasts can I sign off unchecked?",
                "Every week a planner receives a statistical forecast for every store and family. The hard question is not only "
                "“what will sell?” but “which of these numbers can I trust without checking?”")
    cw, gap = 554, 33
    cards = [
        ("Too many numbers", [P(f"__{N['n_series']}__ daily series × __{N['horizon']}__-day horizon = **{N['points_per_cycle']:,} forecast points** "
                                "in every weekly cycle, in this small scope alone.", 24, lh=1.45)]),
        ("Accuracy is uneven", [P(f"Champion holdout WAPE by family runs from **{pct(N['fam_lo'][1])}** ({title_case(N['fam_lo'][0])}) "
                                  f"to **{pct(N['fam_hi'][1])}** ({title_case(N['fam_hi'][0])}). One overall accuracy number hides that spread.", 24, lh=1.45)]),
        ("Attention is finite", [P(f"At the latest origin the Trust Index sends **{N['fwd_counts']['Review']} of {N['n_series']}** series to Review. "
                                   f"That is **{pct(N['review_series_share'])}** of the series to check first.", 24, lh=1.45)]),
    ]
    for i, (t, ps) in enumerate(cards):
        s.card(DM + i * (cw + gap), 300, cw, 300, t, ps, eyebrow=f"0{i + 1}", ts=28)
    s.text(DM, 640, 1728, 40, [P("What this project builds", 24, TEXT, font="head", weight=700)])
    steps = [("Forecast", MINT, "Five models compete on rolling backtests; a champion is chosen per family on selection data only."),
             ("Range", DSKY, "Empirical P10–P90 intervals, calibrated on separate weeks, give every forecast a plausible range."),
             ("Trust", ORCHID, "A 0–100 Forecast Trust Index ranks how far to rely on each 14-day forecast: Accept, Monitor or Review."),
             ("Act", MONITOR, "Exceptions queue, S&OP summary and a scenario lab turn the ranking into planner actions.")]
    sw = (1728 - 3 * 24) / 4
    for i, (t, col, d) in enumerate(steps):
        s.card(DM + i * (sw + 24), 700, sw, 280, t, [P(d, 20, lh=1.45)], tcolor=col, ts=30, eyebrow=f"Step {i + 1}", variant="panel")
    deck_footer(s, 2, N)

    # 3 ---------------------------------------------------------------- data & scope
    s = new()
    deck_header(s, "02", "Data & scope", "Real data, honest scope",
                f"Daily sales from Corporación Favorita, an Ecuadorian grocery chain (Kaggle). Modelling window "
                f"{N['model_start']} to {N['last_actual']}; stores chosen with data from before the first selection origin only.")
    kp = [(f"{N['n_stores']}", f"stores, of {N['n_eligible']} eligible", f"{N['n_types']} store types, {N['n_cities']} cities"),
          (f"{N['n_fam']}", "product families", f"most volatile: {title_case(N['top_cv_fam'])}, mean daily CV {N['top_cv']:.2f}"),
          (f"{N['n_series']}", "daily series", f"forecast {N['horizon']} days ahead from each weekly origin")]
    for i, (v, l, d) in enumerate(kp):
        s.kpi(DM, 300 + i * 232, 470, 212, v, l, d, color=[MINT, SKY, ORCHID][i], vs=58, ls=21, ds=17)
    fams = list(N["dow"].keys())
    s.chart_card(DM + 494, 300, 640, 676, "Weekday sales index", "heat",
                 {"rows": [{"label": title_case(f), "v": N["dow"][f]} for f in fams], "cols": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                  "lo": min(min(v) for v in N["dow"].values()), "hi": max(max(v) for v in N["dow"].values()), "labelW": 150},
                 fs=19, caption=f"Equal-weighted index over each family's 12 store series, weekday mean = 1.00 (2015-06 to 2017-06, holdout excluded). Sunday is the peak for "
                                f"{N['sunday_peak']} of {N['n_fam']} families.", eyebrow="Demand rhythm")
    s.card(DM + 494 + 664, 300, 1728 - 494 - 664, 676, "Read this first", [
        P("**Ecuador, not the UAE.** Real historical data used as a stand-in; no Gulf numbers are claimed.", 19, bullet=True, lh=1.42, bulletColor=MONITOR),
        P("**Observed sales, not unconstrained demand.** Stock-outs and lost sales are invisible.", 19, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P("**Promo = count of promoted items.** No price or depth, so promo effects are associations, not causal uplift.", 19, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P(f"**Recording gap.** Produce had {pct(N['produce_zero_before'])} zero-sales days before June 2015, so the window starts {N['model_start']}.", 19, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P(f"**Transactions track sales** (r = {N['txn_corr']:.2f}) but are unknown at forecast time, so they are not a feature.", 19, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
    ], stroke=CAVEAT_STROKE, eyebrow="Caveats", ecolor=MONITOR)
    deck_footer(s, 3, N)

    # 4 ---------------------------------------------------------------- method
    s = new()
    st = N["stages"]; tw_ = N["target_windows"]
    fmt_d = lambda d: pd.Timestamp(d).strftime("%-d %b")
    deck_header(s, "03", "Method", "A calendar that cannot leak",
                f"Rolling weekly origins, {N['horizon']}-day horizon. Each stage answers one question, and its target windows never overlap "
                f"the next stage. The holdout was scored once, after everything else was frozen.")
    start = (pd.Timestamp(st["selection"][0]) - pd.Timedelta(days=9)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(N["fwd_end"]) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    months = [{"d": d.strftime("%Y-%m-%d"), "label": d.strftime("%b %Y")} for d in pd.date_range(start, end, freq="MS")]
    tl = {"start": start, "end": end, "horizon": N["horizon"], "months": months, "stages": [
        {"name": "Selection", "sub": f"{len(st['selection'])} origins · pick champion per family", "color": PERI, "origins": st["selection"],
         "targets": tw_["selection"], "targetLabel": f"targets {fmt_d(tw_['selection'][0])} – {fmt_d(tw_['selection'][1])}"},
        {"name": "Calibration", "sub": f"{len(st['calibration'])} origins · fit intervals + Trust weights", "color": ORCHID, "origins": st["calibration"],
         "targets": tw_["calibration"], "targetLabel": f"targets {fmt_d(tw_['calibration'][0])} – {fmt_d(tw_['calibration'][1])}"},
        {"name": "Holdout", "sub": f"{len(st['holdout'])} origins · scored once", "color": MINT, "origins": st["holdout"], "dx": -40,
         "targets": tw_["holdout"], "targetLabel": f"targets {fmt_d(tw_['holdout'][0])} – {fmt_d(tw_['holdout'][1])}"}],
        "forward": {"start": N["fwd_start"], "end": N["fwd_end"], "name": "Forward", "sub": "no actuals"}}
    s.chart_card(DM, 296, 1728, 360, None, "timeline", tl, fs=20, pad=34)
    s.card(DM, 680, 560, 300, "Five models compete", [
        P("Naive · Seasonal naive · 28-day moving average · Holt-Winters", 20, lh=1.45),
        P("**LightGBM**: one global, direct multi-horizon model, retrained at every origin; horizon is a feature, so errors never compound.", 20, lh=1.45, sb=12)], ts=24)
    s.card(DM + 584, 680, 560, 300, "Leakage controls", [
        P("Each origin sees a masked view: nothing after day T.", 20, bullet=True, lh=1.4),
        P(f"Future-perturbation tests; an injected one-day look-ahead made **{N['lookahead_fail']} tests fail**.", 20, bullet=True, lh=1.4, sb=10),
        P("Transactions and oil excluded (unknown or unhelpful at T).", 20, bullet=True, lh=1.4, sb=10)], ts=24)
    s.card(DM + 1168, 680, 560, 300, "Frozen, then tested", [
        P("Champions, interval quantiles and Trust weights were hashed and frozen first.", 20, bullet=True, lh=1.4, bulletColor=MINT),
        P(f"Holdout: **{N['n_holdout_origins']} origins, {N['n_holdout_pts']:,} forecast points**, scored once. Nothing tuned on it.", 20, bullet=True, lh=1.4, sb=10, bulletColor=MINT)], ts=24)
    deck_footer(s, 4, N)

    # 5 ---------------------------------------------------------------- battle
    s = new()
    names = {"lightgbm": "LightGBM", "holt_winters": "Holt-Winters", "seasonal_naive": "Seasonal naive", "moving_average": "Moving average (28d)", "naive": "Naive"}
    deck_header(s, "04", "The forecasting battle", "Machine learning won, but not everywhere",
                f"On {N['n_holdout_origins']} weekly forecast origins (targets {fmt_d(hold_from)} – {fmt_d(hold_to)} 2017), each scored on days the model had not seen when it forecast, LightGBM cut WAPE to "
                f"{pct(N['champ_wape'])} from {pct(bb['wape'])} for the best baseline: {N['err_cut']:.0f}% less absolute error.")
    rows = []
    for r in N["lb"]:
        lab = names[r["model"]] + (" (plan)" if r["model"] == "lightgbm" and r["variant"] == "A" else " (no plan)" if r["model"] == "lightgbm" else "")
        rows.append({"label": lab, "wape": r["wape"], "bias": r["bias"], "champion": r["model"] == "lightgbm" and r["variant"] == "A",
                     "muted": r["model"] == "lightgbm" and r["variant"] == "B"})
    s.chart_card(DM, 296, 840, 500, "Holdout leaderboard, WAPE", "leaderboard", {"rows": rows, "labelW": 290}, fs=20,
                 caption="WAPE = Σ|actual − forecast| ÷ Σ actual over all holdout points; bias = Σ(forecast − actual) ÷ Σ actual. "
                         "“Plan” = known promotion plan.", eyebrow="All 96 series")
    short = {"holt_winters": "HW", "seasonal_naive": "SN", "moving_average": "MA", "naive": "N"}
    drows = [{"label": title_case(f), "a": N["fam_champ"][f], "b": N["fam_best_base"][f][1], "bTag": short[N["fam_best_base"][f][0]]}
             for f in sorted(N["fam_champ"], key=lambda f: N["fam_champ"][f])]
    s.chart_card(DM + 864, 296, 864, 500, "LightGBM vs best simple model", "dumbbell",
                 {"rows": drows, "max": 30, "aLabel": "LightGBM", "bLabel": "Best baseline in that family", "labelW": 170}, fs=20,
                 caption="Holdout WAPE by family. HW = Holt-Winters, SN = seasonal naive.", eyebrow="By family")
    s.kpi(DM, 820, 560, 160, f"{N['fam_wins']}/{N['n_fam']} · {N['hz_wins']}/{N['hz_total']}", "families · horizons won by LightGBM", None, color=MINT, vs=40, ls=19, pad=26)
    s.kpi(DM + 584, 820, 560, 160, f"{N['series_wins']}/{N['n_series']}", "series won by LightGBM",
          f"Holt-Winters is best on {N['series_hw']} individual series", color=SKY, vs=40, ls=19, ds=16, pad=26)
    s.kpi(DM + 1168, 820, 560, 160, f"{pct(N['hol_ma'])} vs {pct(N['hol_lgb'])}", "Holiday days: moving average beat LightGBM",
          f"{N['hol_n']} holiday rows. Simple models still earn a seat.", color=MONITOR, vs=36, ls=19, ds=16, pad=26, stroke=CAVEAT_STROKE)
    deck_footer(s, 5, N)

    # 6 ---------------------------------------------------------------- promo A vs B
    s = new()
    a_h, b_h, d_h = N["ab"]["holdout"]
    deck_header(s, "05", "Promotion information", "Does knowing the promo plan help?",
                "Variant A sees the planned count of promoted items on each target day; variant B sees only promotion history. "
                "Same model, same parameters, same holdout.")
    s.kpi(DM, 296, 520, 330, f"{sg(d_h, 2, plus=True)} pp", "WAPE gained by knowing the plan",
          f"Holdout: {pct(a_h, 2)} with the plan (A) vs {pct(b_h, 2)} without (B). "
          f"Selection {sg(N['ab']['selection'][2], 2, True)} pp, calibration {sg(N['ab']['calibration'][2], 2, True)} pp.", color=DSKY, vs=60)
    s.card(DM, 650, 520, 330, "What the model leans on", [
        P(f"Planned promo features carry **{pct(N['fi_promo'])}** of LightGBM gain (promo_t ranks #{N['promo_rank']}).", 20, lh=1.45),
        P(f"The weekday is first at **{pct(N['fi_top'][1])}**: rhythm matters more than the promo plan.", 20, lh=1.45, sb=12)], ts=24, eyebrow="Feature importance")
    rows = [{"label": "All families", "v": d_h, "strong": True}] + [{"label": title_case(f), "v": d} for f, d in N["ab_fam"]]
    lim = max(0.6, np.ceil(max(abs(r["v"]) for r in rows) * 1.3 / 0.2) * 0.2)
    s.chart_card(DM + 544, 296, 640, 684, "Gain from the plan, by family", "diverging",
                 {"rows": rows, "lim": float(lim), "posLabel": "Plan helped (A better)", "negLabel": "B better", "labelW": 170}, fs=19,
                 caption="Holdout WAPE of B minus A, percentage points.", eyebrow="Holdout")
    s.card(DM + 1208, 296, 520, 684, "Read with care", [
        P("**Count, not depth.** The data has the number of promoted items, not price or discount.", 20, bullet=True, lh=1.42, bulletColor=MONITOR),
        P("**Association, not causation.** The model learns what tends to happen when more items are on promotion.", 20, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P(f"**Few clean comparisons.** Only {N['nopromo_n']} of {N['n_holdout_pts']:,} holdout rows ({pct(100 * N['nopromo_n'] / N['n_holdout_pts'], 0)}) had no promotion.", 20, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P(f"B was slightly better in {', '.join(title_case(f) for f in N['b_better'])}.", 20, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
    ], stroke=CAVEAT_STROKE, eyebrow="Caveats", ecolor=MONITOR)
    deck_footer(s, 6, N)

    # 7 ---------------------------------------------------------------- uncertainty
    s = new()
    deck_header(s, "06", "Uncertainty", "A range, not a point",
                f"Scaled split-conformal P10–P90 bands, fitted on calibration weeks only. Target {N['cov_target']:.0f}% coverage; "
                f"achieved {pct(N['cov'])} on the holdout.")
    s.kpi(DM, 296, 420, 220, pct(N["cov"]), f"holdout coverage vs {N['cov_target']:.0f}% target",
          f"calibration estimate (leave-one-origin-out) {pct(N['cov_loo'])}", color=MINT, vs=56, ls=20, ds=17)
    s.kpi(DM, 532, 420, 220, f"{pct(N['below'])} / {pct(N['above'])}", "below P10 / above P90",
          "misses are close to balanced", color=SKY, vs=44, ls=20, ds=17)
    s.kpi(DM, 768, 420, 212, pct(N["width_pct"]), "mean band width vs forecast",
          f"about {N['width_index']:,.0f} index points per series-day", color=ORCHID, vs=48, ls=20, ds=17)
    s.chart_card(DM + 444, 296, 640, 684, "Coverage by family", "dots",
                 {"rows": [{"label": title_case(f), "v": v, "under": v < N["cov_target"]} for f, v in sorted(N["cov_fam"], key=lambda t: t[1])],
                  "domain": [60, 95], "step": 5, "target": N["cov_target"], "targetLabel": f"Target {N['cov_target']:.0f}%", "labelW": 170}, fs=19,
                 caption="Share of holdout actuals inside P10–P90. Orchid = below target.", eyebrow="Holdout")
    hz_rows = [{"label": f"h{h}", "v": v, "flag": N["hz_dow"][h] if N["hz_dow"][h] in ("Sat", "Sun") else None} for h, v in N["cov_hz"]]
    s.chart_card(DM + 1108, 296, 620, 430, "Coverage by horizon", "lollipopV",
                 {"rows": hz_rows, "domain": [60, 95], "step": 5, "target": N["cov_target"]}, fs=18, eyebrow="Holdout", cap_h=0)
    wk_low = [h for h, v in N["cov_hz"] if v < N["cov_target"] and N["hz_dow"][h] in ("Sat", "Sun")]
    assert set(N["sun_hz"]) <= set(wk_low)
    s.card(DM + 1108, 750, 620, 230, None, [
        P(f"**Empirical, not a guarantee.** {title_case(min(N['cov_fam'], key=lambda t: t[1])[0])} reached only "
          f"{pct(min(v for _, v in N['cov_fam']))}. Weekend horizons ({', '.join(f'h{h}' for h in wk_low)}) fall below target, Sundays most, because quantiles are pooled across weekdays.", 19, lh=1.45)],
           stroke=CAVEAT_STROKE, eyebrow="Caveat", ecolor=MONITOR, pad=28)
    deck_footer(s, 7, N)

    # 8 ---------------------------------------------------------------- trust index
    s = new()
    w_ = N["weights"]
    deck_header(s, "07", "Forecast Trust Index", "Knowing when to trust the prediction",
                f"A 0–100 ranking index, not a probability. Weights were chosen on calibration weeks by a rule written in advance "
                f"({N['grid_n']} candidates), then tested once on {N['tv']['n_windows']} holdout windows (series × origin).")
    wrows = [{"label": "Accuracy", "v": w_["accuracy"], "color": MINT}, {"label": "Interval", "v": w_["interval"], "color": DSKY},
             {"label": "Stability", "v": w_["stability"], "color": ORCHID}, {"label": "Completeness", "v": w_["completeness"], "color": PERI},
             {"label": "Agreement", "v": w_["agreement"], "color": VIOLET}]
    s.chart_card(DM, 296, 540, 300, "Frozen weights", "weights", {"rows": wrows}, fs=19, eyebrow="Index recipe", cap_h=0)
    thr = N["bands_thr"]
    s.card(DM, 616, 540, 364, "Bands", [
        P(f"{{{{{ACCEPT}:Accept}}}} score ≥ {thr['Accept']:.0f}", 20, lh=1.4),
        P(f"{{{{{MONITOR}:Monitor}}}} score {thr['Monitor']:.0f}–{thr['Accept']:.0f}", 20, lh=1.4, sb=8),
        P(f"{{{{{REVIEW}:Review}}}} below {thr['Monitor']:.0f}, or a hard flag (e.g. planned promotions above the series' 95th percentile)", 20, lh=1.4, sb=8),
        P(f"Review windows: **{pct(N['tv']['review_share_of_windows'])}** of windows, **{pct(N['tv']['review_share_of_abs_error'])}** of absolute error.", 19, lh=1.4, sb=14, color=TEXT2)], ts=24, eyebrow="How to read it")
    brows = [{"label": b, "v": N["wb"][b]["wape"], "sub": f"{N['wb'][b]['n']} windows", "color": BAND_COL[b]} for b in ["Accept", "Monitor", "Review"]]
    s.chart_card(DM + 564, 296, 560, 450, "Holdout WAPE by band", "bands", {"rows": brows, "max": 20}, fs=20, eyebrow="Validation",
                 caption=f"Review windows had {N['tv']['error_ratio_review_vs_accept']:.2f}× the WAPE of Accept windows.", cap_h=44)
    rrows = [{"label": "Trust Index (frozen)", "wape": abs(N["tv"]["spearman"]), "champion": True},
             {"label": "Default weights*", "wape": abs(N["hold_rho_default"])},
             {"label": "Volatility only", "wape": abs(N["tv"]["volatility_only_spearman"])}]
    s.chart_card(DM + 1148, 296, 580, 450, "Ranking power |ρ|", "leaderboard", {"rows": rrows, "labelW": 230, "fmt": "rho"}, fs=19, eyebrow="vs baseline",
                 caption=f"Spearman ρ, score vs realised window WAPE: {sg(N['tv']['spearman'], 3)} vs {sg(N['tv']['volatility_only_spearman'], 3)} "
                         f"(volatility only). *Default = hindsight diagnostic.", cap_h=70)
    s.card(DM + 564, 770, 1164, 210, None, [
        P(f"**Dominated by history:** rank correlation {N['rho_score_acc']:.2f} with the accuracy component alone (ρ {sg(N['rho_acc_only'], 3)}).  "
          f"**Promo flag pointed the wrong way:** {N['flag_n']} flagged windows had lower WAPE ({pct(N['flag_wape'])} vs {pct(N['noflag_wape'])}); "
          f"{N['review_by_flag_only']} of {N['review_n']} Review windows were there only because of it.  "
          f"**Monitor ≈ Review** ({pct(N['wb']['Monitor']['wape'])} vs {pct(N['wb']['Review']['wape'])}): it separates Accept from the rest best.", 18.5, lh=1.45)],
           stroke=CAVEAT_STROKE, eyebrow="Honest caveats", ecolor=MONITOR, pad=26)
    deck_footer(s, 8, N)

    # 9 ---------------------------------------------------------------- planner actions
    s = new()
    fc = N["fwd_counts"]
    deck_header(s, "08", "From index to action", "What a planner does with it",
                f"Forward origin {N['fwd_origin']}: a {N['horizon']}-day plan for {fmt_d(N['fwd_start'])} – {fmt_d(N['fwd_end'])} 2017 "
                f"with no actuals yet, triaged by the Trust Index.")
    bw3 = 228
    for i, b in enumerate(["Accept", "Monitor", "Review"]):
        s.kpi(DM + i * (bw3 + 18), 296, bw3, 150, str(fc[b]), b, "series", color=BAND_COL[b], vs=46, ls=20, ds=15, pad=22)
    s.rect(DM, 466, 720, 514)
    s.text(DM + 28, 488, 664, 40, [P("Exceptions queue · top 5 by priority", 22, TEXT, font="head", weight=700)])
    trows = []
    for r in N["top_exc"]:
        reason = r["reason"].replace("Weakest component: ", "").replace(" / 100", "/100")
        if len(reason) > 22:
            reason = "Promo plan above p95" if "promo" in reason.lower() else reason[:22] + "…"
        trows.append([{"t": f"{r['store_nbr']} · {title_case(r['family'])}", "c": TEXT, "b": True},
                      {"t": f"{r['trust_score']:.0f}"}, {"t": r["band"], "pill": BAND_COL[r["band"]]}, {"t": reason}])
    s.table(DM + 20, 540, 680, 420, [{"t": "Store · family", "w": 230}, {"t": "Score", "w": 80, "align": "right"},
                                    {"t": "Band", "w": 130}, {"t": "Weakest component", "w": 240}], trows, fs=18, rowH=68)
    s.card(DM + 744, 296, 480, 684, "S&OP pack", [
        P(f"Chain 14-day forecast **{N['fwd_total']:,.0f} units**; the average series is {pct(N['fwd_vs_last'], 1, True)} vs its last 14 observed days.", 20, bullet=True, lh=1.42),
        P(f"Persistent bias to discuss: **Beverages** under-forecast on calibration ({pct(N['bev_bias_cal'])}) and holdout ({pct(N['bev_bias_hold'])}).", 20, bullet=True, lh=1.42, sb=14),
        P(f"Simulated rule-based override (not a human planner) made holdout WAPE **worse**: {pct(N['fva']['champion_wape'], 2)} → "
          f"{pct(N['fva']['simulated_wape'], 2)} (FVA {sg(N['fva']['fva_pp'], 2)} pp).", 20, bullet=True, lh=1.42, sb=14, bulletColor=MONITOR),
        P("Lesson: override only with information the model lacks, and log a reason code.", 20, lh=1.42, sb=14, color=MINT, weight=600)], ts=26, eyebrow="Consensus meeting")
    ev = N["events"][0]
    s.card(DM + 1248, 296, 480, 684, "Scenario Lab", [
        P("**Explicitly hypothetical.** Gulf event uplifts (Ramadan, Eids, summer) are editable placeholder inputs, not measured UAE data.", 20, bullet=True, lh=1.42, bulletColor=MONITOR),
        P(f"Ecuadorian reference only: national days off ran {pct(N['analogue_min'][1], 1, True)} ({title_case(N['analogue_min'][0])}) to "
          f"{pct(N['analogue_max'][1], 1, True)} ({title_case(N['analogue_max'][0])}) vs the same weekday, over {N['analogue_n']} holidays.", 20, bullet=True, lh=1.42, sb=14),
        P("Low / base / high demand → shortfall, excess and cost; the plan with minimum expected cost is recommended, with an order-by date from lead time.", 20, bullet=True, lh=1.42, sb=14)],
           ts=26, eyebrow="What if", stroke=CAVEAT_STROKE, ecolor=MONITOR)
    deck_footer(s, 9, N)

    # 10 --------------------------------------------------------------- about
    s = new()
    s.asset("torus", 1240, 420, 700, 700)
    s.asset("coil", 1640, 60, 200, 380)
    s.asset("prism", 1380, 140, 220, 220)
    s.asset("badge", DM, 96, 78, 78, label="09")
    s.text(200, 92, 1100, 30, [P("About me", 19, SKY, weight=700, ls=0.14, upper=True)])
    s.text(200, 122, 1100, 80, [P("Akash Gajjar", 64, TEXT, font="head", weight=800, lh=1.15)])
    s.text(DM, 240, 1100, 300, [
        P("MSc Supply Chain Management, University of Liverpool (2025).", 26, TEXT, weight=600, lh=1.45),
        P("Retail operations and inventory experience at Primark and Uniqlo: I have seen what happens on the shop floor when a forecast is trusted blindly, and when it is ignored.", 24, TEXT2, lh=1.5, sb=14),
        P("Looking for **demand planning** roles in **Abu Dhabi** and the wider UAE.", 24, TEXT2, lh=1.5, sb=14)])
    s.card(DM, 520, 1060, 250, "Explore the project", [
        P(f"Live dashboard  **{LIVE_URL}**", 21, lh=1.5, color=TEXT2),
        P(f"Code & notebooks  **{REPO_URL}**", 21, lh=1.5, sb=10, color=TEXT2),
        P("Python · DuckDB SQL · LightGBM · conformal intervals · Plotly · Excel planning workbook", 18, MUTED, lh=1.45, sb=14)], ts=26, eyebrow="Links")
    s.text(DM, 820, 1100, 130, [
        P("Data citation", 16, SKY, weight=700, ls=0.12, upper=True),
        P(f"{N['citation']} Real Corporación Favorita (Ecuador) data; not UAE data. Observed sales, not unconstrained demand. "
          "Trust score is a 0–100 index, not a probability. Intervals are empirical, not guarantees.", 16, MUTED, lh=1.45, sb=6)])
    deck_footer(s, 10, N)
    return S


# ----------------------------------------------------------------------------- carousel (1080 x 1080)
CW = 1080
CM = 72
TOTAL_CAR = 10


def car_head(s, num, kicker, title, tsize=54, th=None):
    s.asset("badge", CM, 64, 68, 68, label=num)
    s.text(160, 78, 860, 40, [P(kicker, 22, SKY, weight=700, ls=0.12, upper=True)], valign="middle")
    s.text(CM, 160, CW - 2 * CM, th or 150, [P(title, tsize, TEXT, font="head", weight=800, lh=1.18)])


def car_foot(s, page):
    s.text(CM, 1000, 700, 40, [P("Forecast vs Instinct · Akash Gajjar", 20, MUTED, weight=600)], valign="middle")
    s.text(CW - CM - 300, 1000, 300, 40, [P(f"{page}/{TOTAL_CAR}" + ("  →" if page < TOTAL_CAR else ""), 22, TEXT2, weight=700, align="right")], valign="middle")


def carousel(N):
    S = []
    new = lambda: S.append(Slide(len(S) + 1, "car")) or S[-1]
    wb, tv = N["wb"], N["tv"]
    bb = N["best_base"]

    s = new()  # 1 hook
    s.asset("torus", 600, 560, 600, 600)
    s.asset("prism", 860, 40, 170, 170)
    s.text(CM, 120, 700, 40, [P("Demand planning case study", 24, SKY, weight=700, ls=0.12, upper=True)])
    s.text(CM, 190, 936, 420, [P("Should a demand planner always trust an AI forecast?", 72, TEXT, font="head", weight=800, lh=1.14)])
    s.text(CM, 600, 600, 280, [P(f"I built 5 forecasting models on **{N['n_series']} real grocery series**, then a Trust Index that says "
                                 "**which forecasts to accept and which to review.**", 34, TEXT2, lh=1.4)])
    s.text(CM, 910, 400, 60, [P("Swipe →", 30, MINT, weight=700, font="head")], valign="middle")
    car_foot(s, 1)

    s = new()  # 2 setup
    car_head(s, "01", "The setup", "Real data, honest scope")
    tiles = [(str(N["n_stores"]), "stores", MINT), (str(N["n_fam"]), "product families", SKY),
             (str(N["n_series"]), "daily series", ORCHID), (f"{N['horizon']} days", "forecast horizon", MONITOR)]
    for i, (v, l, c) in enumerate(tiles):
        x = CM + (i % 2) * 476; y = 330 + (i // 2) * 230
        s.kpi(x, y, 460, 210, v, l, None, color=c, vs=64, ls=28, pad=32)
    s.text(CM, 800, 936, 180, [P("Corporación Favorita grocery chain, **Ecuador** (Kaggle). **Not UAE data.** "
                                 "Observed sales, not unconstrained demand: stock-outs are invisible.", 30, TEXT2, lh=1.4)])
    car_foot(s, 2)

    s = new()  # 3 battle
    car_head(s, "02", "The forecasting battle", "Machine learning won on unseen weeks")
    s.kpi(CM, 340, 936, 230, f"{pct(N['champ_wape'])} vs {pct(bb['wape'])}", "Holdout error (WAPE): LightGBM vs best baseline",
          f"{N['err_cut']:.0f}% less absolute error · {N['n_holdout_origins']} holdout weeks, scored once", color=MINT, vs=68, ls=28, ds=24)
    names = {"lightgbm": "LightGBM", "holt_winters": "Holt-Winters", "seasonal_naive": "Seasonal naive", "moving_average": "Moving avg", "naive": "Naive"}
    rows = [{"label": names[r["model"]], "wape": r["wape"], "champion": r["model"] == "lightgbm"} for r in N["lb"] if r["variant"] == "A"]
    s.chart(  "leaderboard", CM, 610, 936, 360, {"rows": rows, "labelW": 290}, 30)
    car_foot(s, 3)

    s = new()  # 4 but
    car_head(s, "03", "But not everywhere", "Simple models still earn a seat")
    s.kpi(CM, 340, 936, 250, f"{pct(N['hol_ma'])} vs {pct(N['hol_lgb'])}", "Holiday days: a 28-day moving average beat LightGBM",
          f"{N['hol_n']} holiday rows on the holdout", color=MONITOR, vs=64, ls=28, ds=24, stroke=CAVEAT_STROKE)
    s.kpi(CM, 610, 456, 340, f"{N['series_wins']}/{N['n_series']}", "series won by LightGBM",
          f"Holt-Winters best on {N['series_hw']}", color=SKY, vs=60, ls=26, ds=22)
    s.kpi(CM + 480, 610, 456, 340, f"{N['fam_lo'][1]:.0f}–{N['fam_hi'][1]:.0f}%", "family WAPE range",
          f"{title_case(N['fam_lo'][0])} easiest, {title_case(N['fam_hi'][0])} hardest", color=ORCHID, vs=60, ls=26, ds=22)
    car_foot(s, 4)

    s = new()  # 5 promo
    a_h, b_h, d_h = N["ab"]["holdout"]
    car_head(s, "04", "Promotion information", "Knowing the promo plan helped a little")
    s.kpi(CM, 340, 936, 250, f"{sg(d_h, 2, True)} pp", "lower WAPE with the planned promotion count",
          f"{pct(a_h, 2)} with the plan vs {pct(b_h, 2)} without it", color=DSKY, vs=72, ls=28, ds=24)
    s.card(CM, 620, 936, 340, None, [
        P("**Promo = count of promoted items**, no price or depth.", 30, bullet=True, lh=1.4, bulletColor=MONITOR),
        P("**Association, not causation.** It shows what tends to happen, not what a promotion causes.", 30, bullet=True, lh=1.4, sb=18, bulletColor=MONITOR),
        P(f"The weekday mattered more: **{pct(N['fi_top'][1])}** of model gain.", 30, bullet=True, lh=1.4, sb=18, bulletColor=MONITOR)],
           stroke=CAVEAT_STROKE, pad=40)
    car_foot(s, 5)

    s = new()  # 6 intervals
    car_head(s, "05", "Uncertainty", "A range, not a single number")
    s.kpi(CM, 340, 456, 270, pct(N["cov"]), f"of actuals inside P10–P90", f"target {N['cov_target']:.0f}%", color=MINT, vs=64, ls=26, ds=24)
    s.kpi(CM + 480, 340, 456, 270, pct(N["cov_loo"]), "expected from calibration", "estimated before the holdout", color=SKY, vs=64, ls=26, ds=24)
    lo_f = min(N["cov_fam"], key=lambda t: t[1])
    s.card(CM, 640, 936, 320, "Empirical, not a guarantee", [
        P(f"{title_case(lo_f[0])} reached only **{pct(lo_f[1])}**.", 30, bullet=True, lh=1.4, bulletColor=MONITOR),
        P("Sundays were under-covered: one band for every weekday.", 30, bullet=True, lh=1.4, sb=14, bulletColor=MONITOR)],
           ts=32, stroke=CAVEAT_STROKE, pad=40)
    car_foot(s, 6)

    s = new()  # 7 trust
    car_head(s, "06", "Forecast Trust Index", "Knowing when to trust the prediction", tsize=50)
    brows = [{"label": b, "v": wb[b]["wape"], "sub": f"{wb[b]['n']} windows", "color": BAND_COL[b]} for b in ["Accept", "Monitor", "Review"]]
    s.chart_card(CM, 330, 936, 470, "Holdout WAPE by trust band", "bands", {"rows": brows, "max": 20}, fs=28, ts=30, pad=36, cap_h=0)
    s.text(CM, 820, 936, 170, [P(f"A 0–100 index, **not a probability**. Review forecasts had **{tv['error_ratio_review_vs_accept']:.2f}×** the error of "
                                 f"Accept ones, and it beat a volatility-only rule (ρ {sg(tv['spearman'], 2)} vs {sg(tv['volatility_only_spearman'], 2)}).", 28, TEXT2, lh=1.4)])
    car_foot(s, 7)

    s = new()  # 8 caveats
    car_head(s, "07", "Honest caveats", "Where the index falls short")
    items = [("Dominated by history", f"{N['rho_score_acc']:.2f} rank correlation with past accuracy alone."),
             ("Promo flag pointed the wrong way", f"Flagged windows had lower error ({pct(N['flag_wape'])} vs {pct(N['noflag_wape'])})."),
             ("Monitor ≈ Review", f"{pct(wb['Monitor']['wape'])} vs {pct(wb['Review']['wape'])} WAPE. It separates Accept from the rest best.")]
    for i, (t, d) in enumerate(items):
        s.card(CM, 330 + i * 216, 936, 196, t, [P(d, 28, lh=1.35)], ts=30, stroke=CAVEAT_STROKE, pad=32, tcolor=MONITOR)
    car_foot(s, 8)

    s = new()  # 9 planner
    car_head(s, "08", "What a planner does", "Review the few, trust the many")
    fc = N["fwd_counts"]
    for i, b in enumerate(["Accept", "Monitor", "Review"]):
        s.kpi(CM + i * 316, 330, 304, 220, str(fc[b]), b, "series", color=BAND_COL[b], vs=64, ls=28, ds=22)
    s.card(CM, 580, 936, 380, None, [
        P(f"The **{fc['Review']} Review series** ({pct(N['review_series_share'])} of all series): start there.", 30, bullet=True, lh=1.4),
        P(f"A simulated rule-based override made error **worse** ({pct(N['fva']['champion_wape'], 2)} → {pct(N['fva']['simulated_wape'], 2)}).", 30, bullet=True, lh=1.4, sb=16),
        P("Override only with information the model doesn't have.", 30, lh=1.4, sb=16, color=MINT, weight=700)], pad=40)
    car_foot(s, 9)

    s = new()  # 10 links
    s.asset("torus", 720, -220, 560, 560)
    s.asset("badge", CM, 64, 68, 68, label="09")
    s.text(160, 78, 600, 40, [P("Explore it yourself", 22, SKY, weight=700, ls=0.12, upper=True)], valign="middle")
    s.text(CM, 170, 700, 200, [P("Akash Gajjar", 62, TEXT, font="head", weight=800, lh=1.15)])
    s.text(CM, 260, 660, 220, [P("MSc Supply Chain Management, University of Liverpool (2025). Retail ops & inventory at Primark and Uniqlo. "
                                 "Seeking demand planning roles in Abu Dhabi.", 28, TEXT2, lh=1.4)])
    s.card(CM, 500, 936, 380, None, [
        P("Live dashboard", 22, SKY, weight=700, ls=0.1, upper=True),
        P("akashgajjar07-hue.github.io/forecast-vs-instinct", 30, TEXT, weight=700, lh=1.3, sb=6),
        P("Code, notebooks, SQL", 22, SKY, weight=700, ls=0.1, upper=True, sb=26),
        P("github.com/akashgajjar07-hue/forecast-vs-instinct", 30, TEXT, weight=700, lh=1.3, sb=6),
        P("Data: Alexis Cook, DanB, inversion, Ryan Holbrook (2021), Store Sales – Time Series Forecasting, Kaggle.", 20, MUTED, lh=1.4, sb=26)], pad=40)
    car_foot(s, 10)
    return S


# ----------------------------------------------------------------------------- post
def post(N):
    fmt_dd = lambda d: pd.Timestamp(d).strftime("%-d %b")
    bb = N["best_base"]; tv = N["tv"]; wb = N["wb"]
    txt = f"""Should a demand planner always trust an AI-generated forecast?

I tested that question on {N['n_series']} real daily grocery series ({N['n_stores']} stores × {N['n_fam']} families, Corporación Favorita, Ecuador) with five forecasting models and a {N['horizon']}-day horizon.

What I found on {N['n_holdout_origins']} weekly forecast origins (targets {fmt_dd(N['target_windows']['holdout'][0])} – {fmt_dd(N['target_windows']['holdout'][1])} 2017), each scored on days the model had not seen when it forecast:

• LightGBM cut error to {pct(N['champ_wape'])} WAPE from {pct(bb['wape'])} for the best simple baseline.
• But a 28-day moving average still beat it on holiday days ({pct(N['hol_ma'])} vs {pct(N['hol_lgb'])}).
• Knowing the promotion plan improved WAPE by only {sg(N['ab']['holdout'][2], 2)} pp.
• P10–P90 intervals covered {pct(N['cov'])} of actuals against an {N['cov_target']:.0f}% target.

So I built a Forecast Trust Index: a 0–100 score (not a probability) that sorts forecasts into Accept, Monitor and Review. Review forecasts had {tv['error_ratio_review_vs_accept']:.2f}× the error of Accept ones, and it beat a volatility-only rule.

Honest caveat: the index leans heavily on past accuracy, my promotion flag pointed the wrong way, and Monitor and Review were barely separated. This is Ecuadorian data, not UAE data, and observed sales, not true demand.

The lesson for planners: review the few, trust the many, and override only with information the model doesn't have.

Dashboard: {LIVE_URL}
Code: {REPO_URL}
Data: Alexis Cook, DanB, inversion, Ryan Holbrook (2021), Store Sales – Time Series Forecasting, Kaggle.

#DemandPlanning #SupplyChain #Forecasting #FMCG #MachineLearning #SOP #AbuDhabi #UAEJobs
"""
    assert txt.startswith("Should a demand planner always trust an AI-generated forecast?")
    body = txt.split("\nDashboard:")[0]
    words = len(re.findall(r"\S+", txt))
    return txt, words, len(re.findall(r"\S+", body))


# ----------------------------------------------------------------------------- HTML + Playwright
def write_html(name, slides, w, h):
    tpl = (SRC / f"{name}.html").read_text()
    data = {"canvas": {"w": w, "h": h}, "slides": [{"els": s.els} for s in slides]}
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (tpl.replace("/*__THEME_CSS__*/", (SRC / "theme.css").read_text())
               .replace("/*__RENDER_JS__*/", (SRC / "render.js").read_text())
               .replace("/*__DATA__*/", js))
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / f"{name}.html"
    out.write_text(html)
    return out


def render(name, html, w, h, slides, want_assets):
    from playwright.sync_api import sync_playwright
    issues = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        pg.goto(html.as_uri(), wait_until="networkidle")
        pg.wait_for_function("window.__ready === true", timeout=30000)
        fonts = pg.evaluate("[document.fonts.check('800 40px Unbounded'), document.fonts.check('500 20px Montserrat')]")
        if not all(fonts):
            issues.append(f"{name}: web fonts not loaded {fonts} (fallback used)")
        issues += [f"{name}: overflow {o}" for o in pg.evaluate("window.__overflow")]
        pdf = OUT / ("case_study_deck.pdf" if name == "deck" else "linkedin_carousel.pdf")
        pg.pdf(path=str(pdf), width=f"{w}px", height=f"{h}px", print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        br.close()

        # previews (half size)
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=0.5)
        pg.goto(html.as_uri(), wait_until="networkidle")
        pg.wait_for_function("window.__ready === true", timeout=30000)
        for old in PREV.glob(f"{name}_*.png"):
            old.unlink()
        for i in range(len(slides)):
            pg.locator(f"#slide-{i + 1}").screenshot(path=str(PREV / f"{name}_{i + 1:02d}.png"))
        br.close()

        if want_assets:
            ASSETS.mkdir(parents=True, exist_ok=True)
            for old in ASSETS.glob("*.png"):
                old.unlink()
            br = p.chromium.launch()
            pg = br.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            pg.goto(html.as_uri(), wait_until="networkidle")
            pg.wait_for_function("window.__ready === true", timeout=30000)
            # background (one is enough: identical on every slide)
            pg.evaluate("document.querySelectorAll('#slide-1 .el').forEach(e => e.style.visibility='hidden')")
            pg.locator("#slide-1").screenshot(path=str(ASSETS / "bg.png"), type="png")
            pg.evaluate("document.querySelectorAll('#slide-1 .el').forEach(e => e.style.visibility='')")
            pg.add_style_tag(content=".bg{display:none} .slide{background:transparent !important}")
            for s in slides:
                for e in s.els:
                    if e["k"] in ("chart", "asset"):
                        pg.locator(f"#{e['id']}").screenshot(path=str(ASSETS / f"{e['id']}.png"), omit_background=True)
            br.close()
    return issues


# ----------------------------------------------------------------------------- PPTX
def build_pptx(slides, path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Pt
    from lxml import etree

    PX = 6350

    def rgb(hx):
        hx = hx.lstrip("#")
        return RGBColor.from_string(hx.upper())

    def blend(css, bg=CARD):
        m = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", css)
        if not m:
            return css
        r, g, b, a = int(m[1]), int(m[2]), int(m[3]), float(m[4])
        bb = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(a * c + (1 - a) * d):02X}" for c, d in zip((r, g, b), bb))

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(DW * PX), Emu(DH * PX)
    blank = prs.slide_layouts[6]
    for s in slides:
        sl = prs.slides.add_slide(blank)
        fill = sl.background.fill
        fill.solid(); fill.fore_color.rgb = rgb(BG)
        sl.shapes.add_picture(str(ASSETS / "bg.png"), 0, 0, Emu(DW * PX), Emu(DH * PX))
        for e in s.els:
            x, y, w, h = (Emu(int(round(v * PX))) for v in (e["x"], e["y"], e["w"], e["h"]))
            k = e["k"]
            if k == "rect":
                shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
                shp.adjustments[0] = min(0.5, e.get("r", 20) / min(e["w"], e["h"]))
                col = {"panel": PANEL, "deep": DEEP}.get(e.get("variant") or "", CARD)
                shp.fill.solid(); shp.fill.fore_color.rgb = rgb(col)
                shp.line.color.rgb = rgb(blend(e.get("stroke") or "rgba(144, 144, 240, 0.18)", col))
                shp.line.width = Emu(PX * 1.5)
                shp.shadow.inherit = False
            elif k in ("chart", "asset"):
                sl.shapes.add_picture(str(ASSETS / f"{e['id']}.png"), x, y, w, h)
            elif k == "text":
                tb = sl.shapes.add_textbox(x, y, w, h)
                tf = tb.text_frame
                tf.word_wrap = True
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = {"middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}.get(e.get("valign"), MSO_ANCHOR.TOP)
                for i, pa in enumerate(e["paras"]):
                    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    para.alignment = {"right": PP_ALIGN.RIGHT, "center": PP_ALIGN.CENTER}.get(pa.get("align"), PP_ALIGN.LEFT)
                    para.line_spacing = pa["lh"] * 0.92
                    if pa.get("sb"):
                        para.space_before = Pt(pa["sb"] * 0.5)
                    if pa.get("bullet"):
                        pPr = para._p.get_or_add_pPr()
                        ind = int(pa["size"] * 1.1 * PX)
                        pPr.set("marL", str(ind)); pPr.set("indent", str(-ind))
                        bc = etree.SubElement(pPr, qn("a:buClr"))
                        etree.SubElement(bc, qn("a:srgbClr")).set("val", (pa.get("bulletColor") or SKY).lstrip("#").upper())
                        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
                        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
                    for r in pa["runs"]:
                        run = para.add_run()
                        t = r["t"]
                        if pa.get("upper"):
                            t = t.upper()
                        run.text = t
                        f = run.font
                        f.name = "Unbounded" if pa["font"] == "head" else "Montserrat"
                        f.size = Pt(pa["size"] * 0.5)
                        f.bold = bool(r.get("b")) or pa["weight"] >= 700
                        f.italic = bool(r.get("i") or pa.get("i"))
                        f.color.rgb = rgb(r.get("c") or (MINT if pa.get("grad") else pa["color"]))
                        if pa.get("ls"):
                            rPr = run._r.get_or_add_rPr()
                            rPr.set("spc", str(int(pa["ls"] * pa["size"] * 0.5 * 100)))
            elif k == "table":
                nrows, ncols = len(e["rows"]) + 1, len(e["cols"])
                gs = sl.shapes.add_table(nrows, ncols, x, y, w, Emu(int(e["rowH"] * nrows * PX)))
                tbl = gs.table
                tblPr = tbl._tbl.tblPr
                for attr in ("firstRow", "bandRow"):
                    tblPr.set(attr, "0")
                tot = sum(c["w"] for c in e["cols"])
                for j, c in enumerate(e["cols"]):
                    tbl.columns[j].width = Emu(int(e["w"] * c["w"] / tot * PX))
                for i in range(nrows):
                    tbl.rows[i].height = Emu(int(e["rowH"] * PX))
                    for j in range(ncols):
                        cell = tbl.cell(i, j)
                        cell.fill.solid(); cell.fill.fore_color.rgb = rgb(PANEL if i == 0 else CARD)
                        cell.margin_left = cell.margin_right = Emu(int(e["fs"] * 0.6 * PX))
                        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                        tf = cell.text_frame
                        para = tf.paragraphs[0]
                        para.alignment = PP_ALIGN.RIGHT if e["cols"][j].get("align") == "right" else PP_ALIGN.LEFT
                        run = para.add_run()
                        if i == 0:
                            run.text = e["cols"][j]["t"].upper(); color, size, bold = MUTED, e["fs"] * 0.72, True
                        else:
                            c = e["rows"][i - 1][j]
                            run.text = c["t"]; size = e["fs"]
                            color = c.get("pill") or c.get("c") or TEXT2
                            bold = bool(c.get("b") or c.get("pill"))
                        run.font.name = "Montserrat"; run.font.size = Pt(size * 0.5)
                        run.font.bold = bold; run.font.color.rgb = rgb(color)
        if s.notes:
            sl.notes_slide.notes_text_frame.text = s.notes
    prs.save(path)


# ----------------------------------------------------------------------------- main
def main():
    N = compute()
    D = deck(N)
    assert len(D) == TOTAL_DECK
    Cs = carousel(N)
    assert len(Cs) == TOTAL_CAR
    PREV.mkdir(parents=True, exist_ok=True)
    issues = []
    h1 = write_html("deck", D, DW, DH)
    issues += render("deck", h1, DW, DH, D, want_assets=True)
    h2 = write_html("carousel", Cs, CW, CW)
    issues += render("carousel", h2, CW, CW, Cs, want_assets=False)
    build_pptx(D, OUT / "case_study_deck.pptx")
    txt, words, _ = post(N)
    (OUT / "linkedin_post.md").write_text(txt)
    assert 150 <= words <= 250, f"post has {words} words"
    print(f"post words: {words}")
    for f in ["case_study_deck.pdf", "case_study_deck.pptx", "linkedin_carousel.pdf", "linkedin_post.md"]:
        print(f"{f}: {(OUT / f).stat().st_size / 1024:,.0f} KB")
    if issues:
        print("ISSUES:")
        for i in issues:
            print("  -", i)
    else:
        print("no overflow / font issues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
