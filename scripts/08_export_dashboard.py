"""Export pipeline outputs to the dashboard JSON (docs/data/*.json + docs/data/series/<slug>.json).

Reads only outputs/ (backtests, holdout artefacts, frozen JSON), config/ and the processed panel. Verifies the
freeze hashes first and never touches the holdout script or the frozen files. The only model fit is one LightGBM
refit at the forward origin (frozen config) for the promo what-if grid; its scale-0 predictions must equal the saved
forward forecasts (asserted to 1e-6). Every file is validated against src/schemas.py and the total docs/data size
(excluding _fixtures) must stay under 3 MB.

Usage: .venv/bin/python scripts/08_export_dashboard.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.export import build_all  # noqa: E402
from src.freeze import verify_all  # noqa: E402


def main():
    t0 = time.perf_counter()
    verify_all()  # frozen selection / intervals / trust_weights unchanged
    res = build_all(C.DOCS_DATA)
    for name, size in sorted(res["sizes"].items()):
        if not name.startswith("series/"):
            print(f"  {name:22s} {size / 1024:8.1f} KB")
    ser = [s for n, s in res["sizes"].items() if n.startswith("series/")]
    print(f"  series/*.json x{len(ser):3d}    {sum(ser) / 1024:8.1f} KB (max {max(ser) / 1024:.1f} KB)")
    print(f"  total docs/data        {res['total_bytes'] / 1024:8.1f} KB")
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
