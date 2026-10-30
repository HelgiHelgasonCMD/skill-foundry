"""Ingestion: redaction at every boundary, injection quarantine, trust qualification,
deduplication, limits, workspace isolation and human labelling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import BUNDLE, LATE, settings_for
from skill_foundry.ingest.pipeline import ingest_file
from skill_foundry.ingest.sanitize import (
    LimitExceeded,
    Limits,
    SanitizeReport,
    redact_text,
    sanitize,
)
from skill_foundry.service import Foundry

SECRETS = [
    "sk-" + "ant-api03-abcdefghijklmnopqrstuvwxyz0123",
    "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "Bearer abcdefghijklmnopqrstuvwxyz123456",
    "API_KEY=supersecretvalue123",
    "https://user:hunter22@example.invalid/repo.git",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_regex_redaction(secret: str) -> None:
    out = redact_text(f"log line {secret} end")
    assert "[REDACTED:" in out
    assert secret not in out


def test_commit_shas_are_not_redacted() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert redact_text(sha) == sha


def test_structured_secret_fields_and_hidden_reasoning_dropped() -> None:
    rep = SanitizeReport()
    out = sanitize({"extensions": {"api_token": "x", "thinking": "private", "note": "ok"}}, rep)
    assert out == {"extensions": {"note": "ok"}}
    assert len(rep.dropped_fields) == 2


def _write(tmp_path: Path, bundle: dict[str, Any]) -> Path:
    p = tmp_path / "b.json"
    p.write_text(json.dumps(bundle))
    return p


def test_secrets_never_reach_storage_index_or_export(
    tmp_path: Path, bundle: dict[str, Any]
) -> None:
    secret = "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789"
    ep = bundle["episodes"][0]
    ep["observations"][0]["summary"] += f" token {secret}"
    ep["extensions"] = {"session_cookie": "abc", "vendor.note": f"leak {secret}"}
    f = Foundry(settings_for(tmp_path / "sf"))
    try:
        f.ingest(_write(tmp_path, bundle))
        x = f.registry.get_experience(f"demo/{ep['episode_id']}")
        assert secret not in x.model_dump_json()
        db = (tmp_path / "sf" / "registry.db").read_bytes()
        assert secret.encode() not in db
        hits = f.index.search("demo", "token failing pytest", 30)
        assert all(secret not in h.experience_id for h in hits)
        out = tmp_path / "export.json"
        f.export_experiences(out)
        assert secret not in out.read_text()
        for p in (tmp_path / "sf" / "qdrant").rglob("*"):
            if p.is_file():
                assert secret.encode() not in p.read_bytes()
    finally:
        f.close()


def test_injection_is_quarantined_and_not_indexed(foundry: Foundry) -> None:
    rep = foundry.ingest(BUNDLE)
    assert [q["experience_id"] for q in rep.quarantined] == ["demo/gha-inj-1"]
    assert foundry.index.health("demo").sqlite_count == rep.accepted - 1
    ids = {
        h.experience_id
        for h in foundry.index.search("demo", "ignore previous instructions curl sh", 50)
    }
    assert "demo/gha-inj-1" not in ids


def test_trust_qualification(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    by = {x.episode_id: x for x in foundry.registry.experiences("demo")}
    assert by["pt-billing-1"].trust.value == "verified"
    assert by["pt-notify-1"].trust.value == "recorded"  # exit 0 but not acceptance-linked
    assert by["pt-maps-1"].trust.value == "recorded"  # stale revision
    assert any("stale" in r for r in by["pt-maps-1"].trust_reasons)
    assert by["pt-search-1"].trust.value == "self_reported"
    assert by["pt-legacy-1"].is_counterexample


def test_human_attestation_upgrades_trust(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    with pytest.raises(ValueError):
        foundry.label("demo/pt-notify-1", "attest-acceptance", "ev1::suite passes", "alice", "")
    # a non-objective check kind ("command") stays recorded even when attested
    x = foundry.label(
        "demo/pt-notify-1", "attest-acceptance", "ev1::suite passes", "alice", "reviewed CI log"
    )
    assert x.trust.value == "recorded"
    assert x.evidence[0].acceptance is not None
    assert x.evidence[0].acceptance.basis == "human_attested"


def test_dedup_and_conflict(foundry: Foundry, tmp_path: Path, bundle: dict[str, Any]) -> None:
    first = foundry.ingest(BUNDLE)
    again = foundry.ingest(BUNDLE)
    assert again.duplicates == first.accepted and again.accepted == 0
    bundle["episodes"][0]["task_summary"] += " (edited)"
    bundle["episodes"] = bundle["episodes"][:1]
    rep = foundry.ingest(_write(tmp_path, bundle))
    assert rep.accepted == 0
    assert "different content" in rep.rejected[0]["reasons"][0]


def test_generic_jsonl(foundry: Foundry) -> None:
    rep = foundry.ingest(LATE)
    assert rep.format == "jsonl" and rep.accepted == 2 and rep.redactions >= 1


def test_limits(tmp_path: Path, bundle: dict[str, Any], foundry: Foundry) -> None:
    bundle["episodes"][0]["task_summary"] = "x" * 50
    rep = ingest_file(
        _write(tmp_path, bundle), foundry.registry, "demo", limits=Limits(max_string_chars=40)
    )
    assert rep.rejected and "longer than" in rep.rejected[0]["reasons"][0]
    with pytest.raises(LimitExceeded):
        ingest_file(
            _write(tmp_path, bundle), foundry.registry, "demo", limits=Limits(max_bundle_bytes=100)
        )


def test_workspace_isolation(tmp_path: Path) -> None:
    home = tmp_path / "sf"
    a = Foundry(settings_for(home, "team-a"))
    a.ingest(BUNDLE)
    a.close()
    b = Foundry(settings_for(home, "team-b"))
    try:
        b.ingest(LATE)
        assert {x.workspace for x in b.registry.experiences("team-b")} == {"team-b"}
        hits = b.index.search("team-b", "Fix failing pytest suite AssertionError", 50)
        assert hits and all(h.experience_id.startswith("team-b/") for h in hits)
        with pytest.raises(ValueError):
            b.index.search("", "anything")
    finally:
        b.close()
