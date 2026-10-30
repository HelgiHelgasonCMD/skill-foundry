"""experience-bundle.v1 contract: validation, referential integrity, round-trip export.

The JSON Schema in ``schemas/experience-bundle-v1.schema.json`` is the canonical
cross-repository contract and is copied verbatim. This module adds checks a JSON
Schema cannot express (cross-references between IDs) and never infers trust from
schema validity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from skill_foundry.resources import EXPERIENCE_BUNDLE_SCHEMA, load_schema

SCHEMA_VERSION = "experience-bundle.v1"
SUPPORTED_VERSIONS = frozenset({SCHEMA_VERSION})

# Skill Foundry's own data only ever lives under these namespaced extension keys.
EXT_ACCEPTANCE = "skill_foundry.acceptance"
EXT_WORKTREE = "skill_foundry.worktree_fingerprint"
EXT_LINEAGE = "skill_foundry.task_lineage"


class UnsupportedVersion(ValueError):
    pass


class BundleRejected(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class EpisodeRejection:
    index: int
    episode_id: str | None
    reasons: list[str]


@dataclass
class BundleParseResult:
    export_id: str
    exported_at: str | None
    episodes: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[EpisodeRejection] = field(default_factory=list)


def _validator() -> Draft202012Validator:
    return Draft202012Validator(
        load_schema(EXPERIENCE_BUNDLE_SCHEMA), format_checker=FormatChecker()
    )


def check_version(data: Any) -> None:
    if not isinstance(data, dict):
        raise BundleRejected(["bundle must be a JSON object"])
    version = data.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedVersion(
            f"unsupported schema_version {version!r}; supported: {sorted(SUPPORTED_VERSIONS)}"
        )


def validate_bundle(data: Any) -> BundleParseResult:
    """Schema-validate a bundle and split it into accepted / rejected episodes.

    Envelope-level errors reject the whole bundle. Errors inside one episode reject
    only that episode, with reasons.
    """
    check_version(data)
    per_episode: dict[int, list[str]] = {}
    envelope: list[str] = []
    for err in _validator().iter_errors(data):
        path = list(err.absolute_path)
        if len(path) >= 2 and path[0] == "episodes" and isinstance(path[1], int):
            loc = "/".join(str(p) for p in path[2:]) or "<episode>"
            per_episode.setdefault(path[1], []).append(f"schema: {loc}: {err.message}")
        else:
            loc = "/".join(str(p) for p in path) or "<root>"
            envelope.append(f"schema: {loc}: {err.message}")
    if envelope:
        raise BundleRejected(envelope)

    result = BundleParseResult(export_id=data["export_id"], exported_at=data.get("exported_at"))
    episodes: list[dict[str, Any]] = data["episodes"]
    ids = Counter(e.get("episode_id") for e in episodes if isinstance(e, dict))
    bundle_episode_ids = {i for i in ids if isinstance(i, str)}
    bundle_evidence_ids = {
        ev.get("evidence_id")
        for i, ep in enumerate(episodes)
        if i not in per_episode
        for ev in ep["outcome"]["evidence"]
    }
    for i, ep in enumerate(episodes):
        reasons = list(per_episode.get(i, []))
        if not reasons:
            if ids[ep["episode_id"]] > 1:
                reasons.append(f"duplicate episode_id {ep['episode_id']!r} within bundle")
            reasons += integrity_errors(ep, bundle_episode_ids, bundle_evidence_ids)
        if reasons:
            eid = ep.get("episode_id") if isinstance(ep, dict) else None
            result.rejected.append(EpisodeRejection(i, eid, reasons))
        else:
            result.episodes.append(ep)
    return result


def integrity_errors(
    ep: dict[str, Any], bundle_episode_ids: set[str], bundle_evidence_ids: set[Any]
) -> list[str]:
    """Cross-reference checks the JSON Schema cannot express."""
    errors: list[str] = []
    events = set(ep["provenance"]["event_ids"])

    def check_events(owner: str, refs: list[str]) -> None:
        missing = sorted(set(refs) - events)
        if missing:
            errors.append(f"{owner} references unknown source events {missing}")

    def unique(kind: str, values: list[str]) -> None:
        dup = sorted(v for v, c in Counter(values).items() if c > 1)
        if dup:
            errors.append(f"duplicate {kind} ids {dup}")

    unique("observation", [o["observation_id"] for o in ep["observations"]])
    unique("attempt", [a["attempt_id"] for a in ep["attempts"]])
    unique("evidence", [e["evidence_id"] for e in ep["outcome"]["evidence"]])
    unique("lesson", [x["lesson_id"] for x in ep["lessons"]])
    for o in ep["observations"]:
        check_events(f"observation {o['observation_id']}", o["source_event_ids"])
    for a in ep["attempts"]:
        check_events(f"attempt {a['attempt_id']}", a["source_event_ids"])
    for ev in ep["outcome"]["evidence"]:
        check_events(f"evidence {ev['evidence_id']}", ev.get("source_event_ids", []))
    for lesson in ep["lessons"]:
        lid = lesson["lesson_id"]
        missing_ev = sorted(set(lesson["verification_evidence_ids"]) - bundle_evidence_ids)
        if missing_ev:
            errors.append(f"lesson {lid} cites unknown verification evidence {missing_ev}")
        missing_ep = sorted(set(lesson["source_episode_ids"]) - bundle_episode_ids)
        if missing_ep:
            errors.append(f"lesson {lid} cites episodes not in bundle {missing_ep}")
    return errors


def make_bundle(export_id: str, episodes: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "export_id": export_id,
        "episodes": episodes,
    }
    bundle.update(extra)
    return bundle


def assert_valid(bundle: dict[str, Any]) -> None:
    """Raise if ``bundle`` (e.g. one we export) violates the canonical schema."""
    errors = sorted(_validator().iter_errors(bundle), key=lambda e: list(e.absolute_path))
    if errors:
        raise BundleRejected([f"{list(e.absolute_path)}: {e.message}" for e in errors])
