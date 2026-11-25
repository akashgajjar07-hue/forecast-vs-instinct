"""Validate every published docs/data/*.json (and the 96 series files) against src/schemas.py
and check the payload budget (< 3 MB). Exit 1 on any failure. Used in CI and before deploy.
--allow-fixtures tolerates docs/data/_fixtures (local development only)."""
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schemas import SCHEMAS, SERIES_SCHEMA  # noqa: E402

DATA = ROOT / "docs" / "data"
BUDGET = 3 * 1024 * 1024
N_SERIES = 96


def main() -> int:
    errors = []
    for name, schema in SCHEMAS.items():
        p = DATA / name
        if not p.exists():
            errors.append(f"missing {name}")
            continue
        try:
            jsonschema.validate(json.loads(p.read_text()), schema)
        except jsonschema.ValidationError as e:
            errors.append(f"{name}: {e.message} at {list(e.absolute_path)}")
    series = sorted((DATA / "series").glob("*.json"))
    if len(series) != N_SERIES:
        errors.append(f"expected {N_SERIES} series files, found {len(series)}")
    for p in series:
        try:
            jsonschema.validate(json.loads(p.read_text()), SERIES_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"series/{p.name}: {e.message}")
    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file() and "_fixtures" not in f.parts)
    if size >= BUDGET:
        errors.append(f"docs/data payload {size / 1e6:.2f} MB exceeds 3 MB")
    if (DATA / "_fixtures").exists() and "--allow-fixtures" not in sys.argv:
        errors.append("docs/data/_fixtures must be deleted before deploy")
    print(f"docs/data: {len(SCHEMAS)} contract files + {len(series)} series, {size / 1024:.1f} KB, {len(errors)} errors")
    for e in errors:
        print("FAIL", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
