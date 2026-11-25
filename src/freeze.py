"""Freeze decisions before the holdout is touched.

freeze(name, obj) writes outputs/<name>.json and outputs/<name>.sha256.
verify(name) recomputes the hash and raises HoldoutGuardError if the hash file is
missing or the JSON was modified afterwards. The holdout script verifies
`selection`, `intervals` and `trust_weights` before running and refuses otherwise.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config as C

FROZEN = ["selection", "intervals", "trust_weights"]


class HoldoutGuardError(RuntimeError):
    pass


def _paths(name: str, root: Path | None = None):
    root = root or C.OUTPUTS
    return root / f"{name}.json", root / f"{name}.sha256"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(name: str, obj: dict, root: Path | None = None) -> str:
    j, h = _paths(name, root)
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    digest = sha256_file(j)
    h.write_text(digest + "\n")
    return digest


def verify(name: str, root: Path | None = None) -> dict:
    j, h = _paths(name, root)
    if not h.exists():
        raise HoldoutGuardError(f"{h.name} missing: freeze '{name}' before touching the holdout")
    if not j.exists():
        raise HoldoutGuardError(f"{j.name} missing")
    if sha256_file(j) != h.read_text().strip():
        raise HoldoutGuardError(f"{j.name} was modified after it was frozen")
    return json.loads(j.read_text())


def verify_all(root: Path | None = None) -> dict:
    return {n: verify(n, root) for n in FROZEN}
