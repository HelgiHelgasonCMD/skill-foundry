"""Conversion between bundle episodes (contract dicts) and domain ``Experience``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from skill_foundry.contracts.bundle_v1 import EXT_ACCEPTANCE, EXT_LINEAGE, EXT_WORKTREE
from skill_foundry.domain.models import (
    AcceptanceLink,
    Attempt,
    Environment,
    EvidenceItem,
    Experience,
    Lesson,
    Observation,
    OutcomeStatus,
    Provenance,
    TrustLevel,
)


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _acceptance(ext: dict[str, Any]) -> AcceptanceLink | None:
    raw = ext.get(EXT_ACCEPTANCE)
    if not isinstance(raw, dict):
        return None
    criteria = raw.get("criteria")
    if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
        return None
    rev = raw.get("tested_revision")
    fp = raw.get("worktree_fingerprint")
    return AcceptanceLink(
        criteria=[c for c in criteria if c.strip()],
        tested_revision=rev if isinstance(rev, str) else None,
        worktree_fingerprint=fp if isinstance(fp, str) else None,
        basis="producer_declared",
    )


def episode_to_experience(
    ep: dict[str, Any], *, workspace: str, export_id: str, content_hash: str
) -> Experience:
    env = ep["environment"]
    env_ext = env.get("extensions") or {}
    wt = env_ext.get(EXT_WORKTREE)
    environment = Environment(
        repo_id=env["repo_id"],
        revision=env["revision"],
        versions=dict(env["versions"]),
        runtime=env.get("runtime"),
        platform=env.get("platform"),
        worktree_fingerprint=wt if isinstance(wt, str) else None,
    )
    evidence = [
        EvidenceItem(
            evidence_id=e["evidence_id"],
            kind=e["kind"],
            summary=e["summary"],
            observed_at=_dt(e["observed_at"]),
            revision=e.get("revision"),
            exit_code=e.get("exit_code"),
            source_event_ids=list(e.get("source_event_ids", [])),
            acceptance=_acceptance(e.get("extensions") or {}),
        )
        for e in ep["outcome"]["evidence"]
    ]
    lineage = (ep.get("extensions") or {}).get(EXT_LINEAGE)
    return Experience(
        experience_id=f"{workspace}/{ep['episode_id']}",
        workspace=workspace,
        episode_id=ep["episode_id"],
        task_summary=ep["task_summary"],
        environment=environment,
        observations=[
            Observation(
                observation_id=o["observation_id"],
                kind=o["kind"],
                summary=o["summary"],
                error_code=o.get("error_code"),
                timestamp=_dt(o.get("timestamp")),
                source_event_ids=list(o["source_event_ids"]),
            )
            for o in ep["observations"]
        ],
        attempts=[
            Attempt(
                attempt_id=a["attempt_id"],
                action_summary=a["action_summary"],
                status=a["status"],
                result_summary=a.get("result_summary"),
                source_event_ids=list(a["source_event_ids"]),
            )
            for a in ep["attempts"]
        ],
        outcome_status=OutcomeStatus(ep["outcome"]["status"]),
        evidence=evidence,
        lessons=[
            Lesson(**{k: v for k, v in lesson.items() if k != "extensions"})
            for lesson in ep["lessons"]
        ],
        provenance=Provenance(
            source=ep["provenance"]["source"],
            event_ids=list(ep["provenance"]["event_ids"]),
            redaction_applied=ep["provenance"]["redaction_applied"],
            bundle_export_id=export_id,
        ),
        trust=TrustLevel.SELF_REPORTED,
        trust_reasons=[],
        content_hash=content_hash,
        task_lineage=lineage if isinstance(lineage, str) else None,
    )


def experience_to_episode(x: Experience) -> dict[str, Any]:
    """Export a stored experience back to the canonical contract shape.

    Only sanitised fields are exported; Skill Foundry-specific data goes under
    namespaced ``extensions`` keys so any v1 consumer can ignore it.
    """
    env: dict[str, Any] = {
        "repo_id": x.environment.repo_id,
        "revision": x.environment.revision,
        "versions": dict(x.environment.versions),
        "runtime": x.environment.runtime,
        "platform": x.environment.platform,
    }
    if x.environment.worktree_fingerprint:
        env["extensions"] = {EXT_WORKTREE: x.environment.worktree_fingerprint}

    def ev(e: EvidenceItem) -> dict[str, Any]:
        d: dict[str, Any] = {
            "evidence_id": e.evidence_id,
            "kind": e.kind,
            "summary": e.summary,
            "observed_at": e.observed_at.isoformat(),
            "revision": e.revision,
            "exit_code": e.exit_code,
            "source_event_ids": list(e.source_event_ids),
        }
        if e.acceptance and e.acceptance.basis == "producer_declared":
            d["extensions"] = {
                EXT_ACCEPTANCE: {
                    "criteria": list(e.acceptance.criteria),
                    "tested_revision": e.acceptance.tested_revision,
                    "worktree_fingerprint": e.acceptance.worktree_fingerprint,
                }
            }
        return d

    episode: dict[str, Any] = {
        "episode_id": x.episode_id,
        "task_summary": x.task_summary,
        "environment": env,
        "observations": [
            {
                k: v
                for k, v in {
                    "observation_id": o.observation_id,
                    "kind": o.kind,
                    "summary": o.summary,
                    "error_code": o.error_code,
                    "timestamp": o.timestamp.isoformat() if o.timestamp else None,
                    "source_event_ids": list(o.source_event_ids),
                }.items()
                if not (k == "timestamp" and v is None)
            }
            for o in x.observations
        ],
        "attempts": [
            {
                "attempt_id": a.attempt_id,
                "action_summary": a.action_summary,
                "status": a.status,
                "result_summary": a.result_summary,
                "source_event_ids": list(a.source_event_ids),
            }
            for a in x.attempts
        ],
        "outcome": {"status": x.outcome_status.value, "evidence": [ev(e) for e in x.evidence]},
        "lessons": [lesson.model_dump() for lesson in x.lessons],
        "provenance": {
            "source": x.provenance.source,
            "event_ids": list(x.provenance.event_ids),
            # Exported content has always passed through our sanitiser.
            "redaction_applied": True,
        },
    }
    if x.task_lineage:
        episode["extensions"] = {EXT_LINEAGE: x.task_lineage}
    return episode
