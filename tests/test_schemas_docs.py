"""CI-safe: validate whatever dashboard JSON exists in docs/data (not _fixtures) against src/schemas.py.
Skips cleanly when docs/data has no published files."""
import json
from pathlib import Path

import jsonschema
import pytest

from src.schemas import SCHEMAS, SERIES_SCHEMA

DD = Path(__file__).resolve().parents[1] / "docs" / "data"
FILES = sorted(p for p in DD.glob("*.json") if p.name in SCHEMAS) if DD.exists() else []
SERIES = sorted((DD / "series").glob("*.json")) if (DD / "series").exists() else []


@pytest.mark.skipif(not FILES, reason="no published docs/data/*.json")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_published_file_matches_schema(path):
    jsonschema.validate(json.loads(path.read_text()), SCHEMAS[path.name])


@pytest.mark.skipif(not SERIES, reason="no published docs/data/series/*.json")
def test_series_files_match_schema():
    for p in SERIES:
        jsonschema.validate(json.loads(p.read_text()), SERIES_SCHEMA)


@pytest.mark.skipif(not FILES, reason="no published docs/data/*.json")
def test_payload_under_3mb_and_no_fixture_markers():
    files = [p for p in DD.rglob("*.json") if "_fixtures" not in p.parts]
    assert sum(p.stat().st_size for p in files) < 3 * 1024 * 1024
    for p in files:
        assert "FIXTURE" not in p.read_text(), p
