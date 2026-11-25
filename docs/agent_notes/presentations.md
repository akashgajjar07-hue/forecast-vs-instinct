# Presentation agent: case-study deck, LinkedIn carousel, post draft

Rebuild everything: `.venv/bin/python scripts/10_build_presentations.py` (about 20 s, needs network for Google Fonts).

## Files
| Path | Purpose |
|---|---|
| `scripts/10_build_presentations.py` | Reads the outputs, computes and cross-checks every number, builds the slide display lists, renders them with Playwright, writes the PPTX and the post |
| `presentations/src/deck.html`, `carousel.html` | HTML templates (1920×1080 and 1080×1080). The build inlines `theme.css` and `render.js` and injects the data as JSON |
| `presentations/src/theme.css` | Dashboard tokens (`docs/styles.css`): cards, ring badge, torus/prism/coil/orb shapes |
| `presentations/src/render.js` | Paints the display list. Flat SVG charts: leaderboard, dumbbell, diverging, dots, lollipop, bands, weights, heatmap, timeline. Contains no numbers. It also reports text or chart labels that overflow their box |
| `presentations/build/` | Rendered HTML plus `assets/` (chart, shape and badge PNGs and the background, used by the PPTX). Regenerated on every run |
| `presentations/case_study_deck.pdf` / `.pptx` | 10 slides |
| `presentations/linkedin_carousel.pdf` | 10 square pages |
| `presentations/linkedin_post.md` | Post draft (215 words including links and hashtags, checked to be 150–250) |
| `presentations/previews/*.png` | Half-size previews of every slide and page |

## Design
- One display list per slide, with absolute px boxes for rect, text, chart, asset and table elements. The HTML and the PPTX are painted from the same list, so their layouts match.
  - PPTX scale: 1 px = 6350 EMU = 0.5 pt.
  - Text, cards, KPI tiles and the exceptions table are native PPTX objects.
  - Charts, the glossy shapes and the ring badges are transparent PNG screenshots of the HTML elements, taken at 2× scale.
- Fonts: Unbounded for headings and Montserrat for body text, from Google Fonts. The build checks with `document.fonts.check` that they loaded.
  - The PPTX names the same fonts. If they are not installed, PowerPoint substitutes its own font.
  - Text boxes are sized from the HTML layout, so a much wider substitute font could wrap differently.
- Shapes appear on the title and about slides and on carousel pages 1 and 10 only. Charts are flat.
- Status colours always come with a text label.

## Number provenance (all read in code)
- `holdout_results.json`: the leaderboard, family WAPE, Trust validation (ρ, WAPE by band, Review share) and coverage.
- `docs/data/arena.json`: the holdout holiday slice (moving average 16.41 vs LightGBM 17.13, 176 rows), the no-promo row count, horizon wins, and A vs B by family.
- Recomputed from `outputs/backtests/*.parquet` and `outputs/holdout/*.parquet`, then asserted against the JSON or against `notebooks.md`:
  - holdout WAPE, A/B by stage, and per-series wins (83/96, Holt-Winters 10)
  - share of actuals below P10 / above P90 (10.5 / 9.2) and band width as a share of the forecast (37.3 %)
  - Trust Spearman, volatility-only ρ, default-weights ρ (−0.503, a hindsight diagnostic), accuracy-only ρ (−0.647) and score-vs-accuracy correlation (0.971)
  - the promo flag (67 windows, 8.71 % vs 13.02 %; 65 of 123 Review windows flagged only by it)
  - Beverages bias, calibration and holdout
- `docs/data/{trust,overview,sop,scenario,meta}.json`: forward-origin band counts, Review volume share, the chain 14-day total vs the last 14 days, top exceptions, FVA (simulated), the citation and the placeholder event labels.
- `outputs/eda/*.json`: weekday profile, volume share, transactions r, Produce recording gap and the holiday analogue range.
- `config/*.json`: scope and split calendar. `agent_b.md`: the look-ahead test count (5), extracted with a regex.

## Caveats stated in the materials
These appear in the deck, the carousel and the post:
- Ecuador, not UAE data.
- Observed sales, not unconstrained demand.
- Promotion is a count of promoted items, so its effects are associations.
- Intervals are empirical, not guarantees.
- The Trust score is a 0–100 index, not a probability.
- The Trust Index is dominated by historical accuracy.
- The promo hard flag pointed the wrong way.
- Monitor and Review are barely separated.
- The moving average beat LightGBM on holiday days.
- FVA is a simulated rule, not a human planner.
- Scenario Lab uplifts are placeholders.

## Known limits
- The deck PDF is about 9 MB, because Chromium rasterises the blurred and masked background glows.
- The PPTX was checked structurally (shapes, text, table) but was not opened in PowerPoint here.
- The coverage-by-horizon chart flags Saturdays and Sundays. Both weekend horizons fall below 80 %; Sundays are the lowest, as noted in `notebooks.md`.
