"""Registry migrations/versioning and post-publication monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from skill_foundry.monitoring.feedback import ingest_feedback, monitor, propose_revision
from skill_foundry.registry.db import Database, SchemaVersionError
from skill_foundry.registry.orm import Base


def test_migrations_match_orm(tmp_path: Path) -> None:
    db = Database(tmp_path / "r.db")
    with db.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert diff == []
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    assert db.integrity_check() == "ok"
    db.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    db = Database(tmp_path / "r.db")
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE sf_meta SET value='99' WHERE key='schema_version'"))
    db.close()
    with pytest.raises(SchemaVersionError):
        Database(tmp_path / "r.db")


def _feedback(tmp_path: Path, pkg: str | None, n_fail: int, invoked: bool = True) -> Path:
    events = [
        {
            "event_id": f"e{i}-{pkg}-{invoked}",
            "skill_name": "fix-failing-pytest-suite",
            "package_sha256": pkg,
            "skill_invoked": invoked,
            "outcome": "failure",
            "verified": True,
            "failure_summary": "suite uses tox-managed environments",
            "observed_at": "2026-10-01T00:00:00Z",
            "task_summary": "t",
        }
        for i in range(n_fail)
    ]
    events.append(
        {
            "event_id": f"ok-{pkg}-{invoked}",
            "skill_name": "fix-failing-pytest-suite",
            "package_sha256": pkg,
            "outcome": "success",
            "verified": True,
            "observed_at": "2026-10-01T00:00:00Z",
            "task_summary": "t",
        }
    )
    p = tmp_path / f"fb-{pkg}-{invoked}.json"
    p.write_text(
        json.dumps(
            {"schema_version": "outcome-feedback.v1", "feedback_id": p.stem, "events": events}
        )
    )
    return p


def test_feedback_attribution_and_revision(
    approved: dict[str, Any], target: Path, tmp_path: Path
) -> None:
    f = approved["foundry"]
    vid = approved["version_id"]
    f.publish(vid, target, apply=True)
    pkg = f.registry.get_version(vid).package_sha256
    r1 = ingest_feedback(_feedback(tmp_path, "0" * 64, 5), f.registry)  # wrong hash
    assert r1["attributed"] == 0
    r2 = ingest_feedback(_feedback(tmp_path, pkg, 3, invoked=False), f.registry)
    assert r2["attributed"] == 1  # only the success event without skill_invoked=False
    assert not monitor(f.registry, vid)["regression_suspected"]
    assert propose_revision(f.registry, vid)["proposed"] is False
    ingest_feedback(_feedback(tmp_path, pkg, 3), f.registry)
    m = monitor(f.registry, vid)
    assert m["regression_suspected"] and m["verified_failures"] == 3
    prop = propose_revision(f.registry, vid)
    assert prop["proposed"]
    new = f.spec(prop["spec_hash"])
    assert any("tox-managed" in c.text for c in new.contraindications)
    # nothing was generated, approved or published automatically
    assert [v.version_id for v in f.registry.versions()] == [vid]
