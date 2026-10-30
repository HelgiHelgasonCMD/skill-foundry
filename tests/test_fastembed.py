"""Real CPU sentence embeddings (downloads a small model; CI job only)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import BUNDLE
from skill_foundry.config import Settings
from skill_foundry.service import Foundry

pytestmark = [
    pytest.mark.fastembed,
    pytest.mark.skipif(not os.environ.get("CI"), reason="downloads a model; CI only"),
]


def test_semantic_embeddings_retrieve_paraphrases(tmp_path: Path) -> None:
    f = Foundry(Settings(home=tmp_path / "sf", workspace="demo", embedding="fastembed"))
    try:
        assert f.index.provider.semantic
        f.ingest(BUNDLE)
        hits = f.index.search("demo", "lock down the CI pipeline's third-party dependencies", 3)
        assert hits[0].experience_id.startswith("demo/gha-")
    finally:
        f.close()
