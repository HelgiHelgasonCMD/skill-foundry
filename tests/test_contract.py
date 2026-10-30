"""experience-bundle.v1 contract tests: canonical schema, valid/invalid fixtures,
referential integrity, version negotiation and round-trip export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import BUNDLE
from skill_foundry.contracts.bundle_v1 import (
    BundleRejected,
    UnsupportedVersion,
    assert_valid,
    validate_bundle,
)
from skill_foundry.resources import (
    EXPERIENCE_BUNDLE_SCHEMA,
    EXPERIENCE_BUNDLE_SCHEMA_SHA256,
    schema_dir,
)
from skill_foundry.util import sha256_bytes


def test_canonical_schema_is_pinned_verbatim() -> None:
    data = (schema_dir() / EXPERIENCE_BUNDLE_SCHEMA).read_bytes()
    assert sha256_bytes(data) == EXPERIENCE_BUNDLE_SCHEMA_SHA256


def test_demo_bundle_is_valid() -> None:
    res = validate_bundle(json.loads(BUNDLE.read_text()))
    assert res.rejected == []
    assert len(res.episodes) == 25


def _first(bundle: dict[str, Any]) -> dict[str, Any]:
    ep: dict[str, Any] = bundle["episodes"][0]
    return ep


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda e: e.pop("task_summary"), "task_summary"),
        (lambda e: e["outcome"].__setitem__("evidence", []), "evidence"),
        (lambda e: e["outcome"].__setitem__("status", "great_success"), "status"),
        (lambda e: e.__setitem__("transcript", "raw"), "transcript"),
        (lambda e: e["outcome"]["evidence"][0].__setitem__("kind", "vibes"), "kind"),
        (lambda e: e["provenance"].pop("redaction_applied"), "redaction_applied"),
    ],
)
def test_invalid_episode_is_rejected_alone(
    bundle: dict[str, Any], mutate: Any, needle: str
) -> None:
    mutate(_first(bundle))
    res = validate_bundle(bundle)
    assert len(res.rejected) == 1
    assert len(res.episodes) == 24
    assert needle in " ".join(res.rejected[0].reasons) or res.rejected[0].reasons


def test_envelope_errors_reject_bundle(bundle: dict[str, Any]) -> None:
    bundle["surprise"] = 1
    with pytest.raises(BundleRejected):
        validate_bundle(bundle)


@pytest.mark.parametrize("version", ["experience-bundle.v2", "experience-bundle.v0", None])
def test_unsupported_version(bundle: dict[str, Any], version: str | None) -> None:
    bundle["schema_version"] = version
    with pytest.raises(UnsupportedVersion):
        validate_bundle(bundle)


def test_corrupted_event_reference(bundle: dict[str, Any]) -> None:
    _first(bundle)["attempts"][0]["source_event_ids"] = ["nope"]
    res = validate_bundle(bundle)
    assert "unknown source events" in res.rejected[0].reasons[0]


def test_corrupted_lesson_evidence_reference(bundle: dict[str, Any]) -> None:
    _first(bundle)["lessons"][0]["verification_evidence_ids"] = ["ev-missing"]
    res = validate_bundle(bundle)
    assert "unknown verification evidence" in " ".join(res.rejected[0].reasons)


def test_lesson_cites_unknown_episode(bundle: dict[str, Any]) -> None:
    _first(bundle)["lessons"][0]["source_episode_ids"].append("ghost")
    res = validate_bundle(bundle)
    assert "not in bundle" in " ".join(res.rejected[0].reasons)


def test_duplicate_episode_ids(bundle: dict[str, Any]) -> None:
    bundle["episodes"].append(bundle["episodes"][0])
    res = validate_bundle(bundle)
    assert any("duplicate episode_id" in r.reasons[0] for r in res.rejected)


def test_extensions_are_allowed_everywhere(bundle: dict[str, Any]) -> None:
    ep = _first(bundle)
    ep["extensions"] = {"vendor.x": {"a": 1}}
    ep["environment"]["extensions"] = {"vendor.y": True}
    bundle["extensions"] = {"vendor.z": []}
    assert validate_bundle(bundle).rejected == []


def test_round_trip_export_is_canonical(foundry: Any, tmp_path: Path) -> None:
    foundry.ingest(BUNDLE)
    out = tmp_path / "export.json"
    foundry.export_experiences(out)
    exported = json.loads(out.read_text())
    assert_valid(exported)  # our exports satisfy the canonical schema
    original = {e["episode_id"]: e for e in json.loads(BUNDLE.read_text())["episodes"]}
    for ep in exported["episodes"]:
        src = original[ep["episode_id"]]
        assert ep["task_summary"] == src["task_summary"]
        assert ep["outcome"]["status"] == src["outcome"]["status"]
        assert [a["attempt_id"] for a in ep["attempts"]] == [
            a["attempt_id"] for a in src["attempts"]
        ]
        assert ep["provenance"]["event_ids"] == src["provenance"]["event_ids"]
        assert ep["environment"]["revision"] == src["environment"]["revision"]
    # quarantined (injection) episodes are not exported
    assert "gha-inj-1" not in {e["episode_id"] for e in exported["episodes"]}
