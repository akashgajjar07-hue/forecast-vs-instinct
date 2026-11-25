"""Phase 1: split calendar, scope selection, modelling panel.

Outputs
  config/split_calendar.json   stage origins (committed)
  config/scope.json            chosen stores/families + per-series stats (committed)
  docs/data/scope.json         same, for the dashboard
  data/processed/panel.parquet row-level panel (NOT committed - Kaggle data)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import build_panel, load_raw  # noqa: E402
from src.scope import select_scope  # noqa: E402
from src.splits import build_split_calendar  # noqa: E402


def main():
    for d in (C.CONFIG, C.PROCESSED, C.DOCS_DATA, C.OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cal = build_split_calendar()
    (C.CONFIG / "split_calendar.json").write_text(json.dumps(cal, indent=2))
    scope = select_scope(cal["stages"]["selection"][0], raw=raw)
    (C.CONFIG / "scope.json").write_text(json.dumps(scope, indent=2))
    (C.DOCS_DATA / "scope.json").write_text(json.dumps(scope, separators=(",", ":")))
    stores = [s["store_nbr"] for s in scope["stores"]]
    panel = build_panel(stores, raw=raw)
    panel.to_parquet(C.PROCESSED / "panel.parquet", index=False)
    print(f"panel rows={len(panel):,} series={panel.series_id.nunique()} "
          f"dates={panel.date.min().date()}..{panel.date.max().date()} closed_rows={int(panel.closed.sum())}")


if __name__ == "__main__":
    main()
