"""Server-mode Qdrant integration (CI service container). Embedded mode is the local
default; server behaviour (payload indexes, performance) is not assumed identical."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from conftest import BUNDLE, SUITE
from skill_foundry.config import Settings
from skill_foundry.retrieval.index import COLLECTION
from skill_foundry.service import Foundry

pytestmark = [
    pytest.mark.qdrant_server,
    pytest.mark.skipif(not os.environ.get("SF_QDRANT_URL"), reason="SF_QDRANT_URL not set"),
]


def test_server_mode_hybrid_and_isolation(tmp_path: Path) -> None:
    ws = f"ci-{uuid.uuid4().hex[:8]}"
    f = Foundry(
        Settings(
            home=tmp_path / "sf",
            workspace=ws,
            embedding="hashing",
            qdrant_url=os.environ["SF_QDRANT_URL"],
        )
    )
    try:
        assert f.index.mode == "server"
        f.index.rebuild()  # fresh collection for this run
        f.ingest(BUNDLE)
        h = f.index.health(ws)
        assert h.healthy, h.reasons
        info = f.index.client.get_collection(COLLECTION)
        assert "workspace" in (info.payload_schema or {})  # keyword index exists on the server
        hits = f.index.search(ws, "pin third-party actions and restrict token permissions", 5)
        assert hits[0].experience_id.startswith(f"{ws}/gha-")
        assert all(h.experience_id.startswith(f"{ws}/") for h in hits)
        d = f.discover(SUITE)
        assert any(s["name"] == "fix-failing-pytest-suite" for s in d["specs"])
    finally:
        f.close()
