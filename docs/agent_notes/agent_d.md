# Agent D: dashboard shell and design system

Static single-page app for GitHub Pages. Vanilla HTML/CSS/ES modules plus a vendored Plotly build. It reads only the
files in `docs/data_contract.md` / `src/schemas.py`. **No real numbers are hard-coded.** Every number comes from `docs/data/*.json`.

## Structure
| File | Role |
|---|---|
| `docs/index.html` | Shell: skip link, mobile top bar, left nav, `<main id="view">`, fixed footer, placeholder banner, shared SVG gradients |
| `docs/styles.css` | Design tokens (`:root`) and all styles. `scripts/check_contrast.py` reads the tokens from here |
| `docs/app.js` | Router (`#/<view>?k=v`), nav, mobile menu, lazy view rendering, chart purge on view change |
| `docs/js/core.js` | Data loader (cached fetch; `?fixtures=1` → `data/_fixtures/`), formatting, `setParams` (deep links via `replaceState`), Plotly defaults, try/catch localStorage helpers, decorative shapes |
| `docs/js/views/*.js` | One module per view (`render(el, params, ctx)`) |
| `docs/scenario.js` | Pure Scenario Lab formulas (ES module), mirrored in `src/scenario.py` |
| `docs/assets/vendor/plotly-cartesian-2.35.2.min.js` | `plotly.js-cartesian-dist-min@2.35.2` via `npm pack`, sha256 `72409fd95b505d17772f2dd9bb81a8672ce6d5b61dcf61ba1fd8442c9ea3e180` (the cartesian build because the basic build has no heatmap). Licence alongside |
| `docs/data/_fixtures/` | PLACEHOLDER data from `scripts/make_fixtures.py` (seeded synthetic numbers, each file has `_fixture` marker and "FIXTURE" strings, schema-validated). 9 files + 6 series |
| `scripts/check_contrast.py` | WCAG + CVD check of tokens |
| `scripts/playwright_check.py` | E2E check (local server or `--base URL`), screenshots |
| `tests/test_scenario.py`, `tests/js/scenario.test.mjs`, `tests/js/scenario_cases.json` | Shared cases. pytest also runs node and compares JS vs Python output field by field |

## Views → JSON
| # | Route | Loads | Deep-link params |
|---|---|---|---|
| 01 | `#/overview` | meta, overview, series_index | none |
| 02 | `#/explorer` | meta, demand, series_index (+ `series/<slug>` when store + family chosen) | `family`, `store` |
| 03 | `#/arena` | arena, meta | `stage`, `slice`, `metric`, `variant` |
| 04 | `#/workspace` | series_index, meta, `series/<slug>` | `series`, `origin`, `scale`, `models` |
| 05 | `#/exceptions` | trust, series_index, meta | `band`, `family`, `series` |
| 06 | `#/scenario` | scenario | `event`, `family` |
| 07 | `#/sop` | sop, meta | `family`, `store`, `band` |
| 08 | `#/methodology` | meta, intervals, trust | none |

Assumptions about the data that the integrator should confirm:
- `demand.dow_profile[family]` is Monday-first (pandas `dayofweek`).
- `intervals.level` may be 80 or 0.8 (both handled). `trust.weights` may be fractions or percents (both handled).
- `sop.fva.fva_pp`: if it equals `champion_wape − simulated_wape` (±0.06), the site reads it as "positive = rule reduced error". Otherwise it shows the number without interpretation.
- `arena.variant_comparison[].family` may be `"all"` for the overall row.
- Explorer store view and Workspace use `series_index` slugs. Links to the Workspace appear only for slugs in the index.
- S&OP CSV: `date` = `sop.origin + 1 day`, `location` = `STORE_<nbr>`, quantities rounded to 2 dp, one row per series (all rows, not only the filtered ones), CRLF line endings. localStorage key is `fvi.sop.overrides.v1:<origin>`.

## Design tokens
Background `#01204C`; panels `#001634` / `#00112D`; card `#0F2A54`, border `rgba(144,144,240,.18)`, radius 20px.
Text `#FFFFFF` / `#D7D0E7` / muted `#9AA4AD`; accent sky `#7BBDE6`. The brand gradient `#6030D8→#9090F0→#D890D8→#60F0D8` is decorative only.
Data: actual `#FFFFFF`, champion `#60F0D8`, band mint 18%, challengers periwinkle `#9090F0`, orchid `#D685C2`,
sky `#68BEF3`, violet `#AE3DFF`. I tuned orchid, sky and violet away from the brief's hues for CVD separation, and dropped `#6030D8` because it is 2.2:1 on navy. Lines also differ by dash pattern.
Status: Accept `#60F0D8`, Monitor `#F5C26B`, Review `#FF6B8A` (always shown with a text label).
Fonts: Unbounded 700/800 (headings), Montserrat 400/500/600 (body), `tabular-nums` for numbers.
Shapes are pure CSS/SVG: a conic-gradient torus, an SVG prism and a coil, and an orb. They appear only in the hero and section headers, and `prefers-reduced-motion` turns their animation off.

## Contrast results (`scripts/check_contrast.py`, all PASS)
Minimum ratio across bg/panel/panel-deep/card: text 14.19, text-2 9.51, muted 5.60, sky 6.94; data: champion 10.11,
periwinkle 5.02, orchid 5.40, sky 6.93, violet 3.32 (≥3 graphics); status (≥4.5 because they are also text): accept 10.11,
monitor 8.66, review 5.22. CIEDE2000 pairwise (Machado 2009 deutan/protan/tritan, severity 1): every model pair ≥12 normal
and ≥6 simulated. The tightest pairs are actual vs champion under protanopia at 8.1, and periwinkle vs sky under deutan at 8.1 (where they also differ by dash). Status pairs are ≥16.

## Scenario Lab formulas (`docs/scenario.js` = `src/scenario.py`)
- demand_s = max(0, baseline × (1 + pct_s/100)), s ∈ {low, base, high}
- shortfall = max(0, D − supply); excess = max(0, supply − D); coverage % = 100·supply/D (null if D = 0)
- cost(plan p, outcome o) = cu·max(0, D_o − Q_p) + co·max(0, Q_p − D_o), with Q_p = D_p
- **Recommendation = minimum expected cost** with fixed weights low/base/high = 0.25/0.5/0.25. Ties go to lower max regret, then to the order low, base, high. The view also reports the minimax-regret plan when it differs. The "keep current supply" plan is shown for reference.
- critical ratio = cu/(cu+co) (shown as a hint)
- Timing: order_in_days = days_to_event − lead_time (order-by = event start − lead time). Below 0 → "late" (expedite or reallocate); 0–3 → "now"; above 3 → "later".
- Recommendation text is deterministic and identical in JS and Python (`fmtInt` = floor(x+0.5) with comma separators).

## Verification
- `node --test tests/js/scenario.test.mjs`: 7 pass. `pytest tests/test_scenario.py`: 6 pass, including the JS = Python cross-check on 8 cases.
- `playwright_check.py --fixtures`: PASS at 1440×900 and 390×844 for all 8 views, with 244 interactions. Also checked: deep links, first Tab reaching the skip link, the S&OP view with localStorage throwing, the mobile menu, CSV header and the override in the export. I deleted the fixture-mode screenshots.

## Known gaps / integrator to-dos
1. **Delete `docs/data/_fixtures/` before deploy.** Then remove or regenerate it locally with `scripts/make_fixtures.py`. If you delete it, the `?fixtures=1` mode simply 404s.
2. Run `scripts/playwright_check.py` (no `--fixtures`) against the real `docs/data`, then with `--base <pages URL>`. This regenerates the screenshots.
3. Payload: Plotly is 1.36 MB and the fixture series files are ~25 KB each. 96 real series of similar size are about 2.4 MB, but they load lazily, one per selection. Keep `docs/data` under 3 MB in total.
4. LinkedIn URL: set `LINKEDIN_URL` at the top of `docs/js/views/methodology.js`. Until then the card shows "link to be added".
5. Google Fonts is the only external request. Without it the fonts fall back to the system font.
6. In real mode, any missing JSON shows a "Data not available" card in that view. Before the data exists, the browser logs the 404.
