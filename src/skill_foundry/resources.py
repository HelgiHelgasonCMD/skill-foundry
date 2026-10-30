"""Locate bundled JSON schemas (works from a source checkout and an installed wheel)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent
_CANDIDATES = (_PKG / "_schemas", _PKG.parent.parent / "schemas")


def schema_dir() -> Path:
    for c in _CANDIDATES:
        if c.is_dir():
            return c
    raise FileNotFoundError("schemas directory not found")


@cache
def load_schema(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((schema_dir() / name).read_text(encoding="utf-8"))
    return data


EXPERIENCE_BUNDLE_SCHEMA = "experience-bundle-v1.schema.json"
# sha256 of the canonical cross-repository contract, copied verbatim. A test pins it.
EXPERIENCE_BUNDLE_SCHEMA_SHA256 = "6d649ed25c144a64c1be77fe696500b0f8865763cc71fc107ae2bee54298fc36"
FEEDBACK_SCHEMA = "outcome-feedback-v1.schema.json"
EVAL_SUITE_SCHEMA = "eval-suite-v1.schema.json"
