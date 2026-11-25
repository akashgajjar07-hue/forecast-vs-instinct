"""Validate published dashboard JSON (docs/data, excluding _fixtures) against src/schemas.py (used by CI).
Exits 0 with a note when nothing is published yet; exits 1 on any schema error or if the payload is >= 3 MB."""
import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schemas import SCHEMAS, SERIES_SCHEMA  # noqa: E402

DD = ROOT / "docs" / "data"


def main() -> int:
    errors, n = [], 0
    for name, schema in SCHEMAS.items():
        p = DD / name
        if p.exists():
            n += 1
            try:
                jsonschema.validate(json.loads(p.read_text()), schema)
            except jsonschema.ValidationError as e:
                errors.append(f"{name}: {e.message}")
    for p in sorted((DD / "series").glob("*.json")) if (DD / "series").exists() else []:
        n += 1
        try:
            jsonschema.validate(json.loads(p.read_text()), SERIES_SCHEMA)
        except jsonschema.ValidationError as e:
            errors.append(f"series/{p.name}: {e.message}")
    if n == 0:
        print("docs/data: nothing published yet, skipped")
        return 0
    size = sum(p.stat().st_size for p in DD.rglob("*") if p.is_file() and "_fixtures" not in p.parts)
    if size >= 3 * 1024 * 1024:
        errors.append(f"docs/data is {size} bytes (limit 3 MB)")
    for e in errors:
        print("ERROR", e)
    print(f"validated {n} files, {size / 1024:.1f} KB, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
