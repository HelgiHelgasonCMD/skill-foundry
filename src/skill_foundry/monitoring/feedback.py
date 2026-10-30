"""Post-publication monitoring.

Outcome events are attributed to a skill version only when they carry the exact
package hash of an installed version. Unattributed events are stored but never
counted against (or for) a version. A suspected regression proposes a revised
ProcedureSpec (new contraindications); it never regenerates, approves or
publishes anything by itself, and model self-grading is not an input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import select

from skill_foundry.domain.models import ProcedureSpec, SupportedStatement
from skill_foundry.ingest.sanitize import redact_text
from skill_foundry.procedure.infer import spec_hash
from skill_foundry.registry import orm
from skill_foundry.registry.repo import Registry
from skill_foundry.resources import FEEDBACK_SCHEMA, load_schema
from skill_foundry.util import utcnow

MIN_ATTRIBUTED_FAILURES = 2
MAX_FAILURE_RATE = 0.3


def ingest_feedback(path: Path, registry: Registry) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = list(
        Draft202012Validator(
            load_schema(FEEDBACK_SCHEMA), format_checker=FormatChecker()
        ).iter_errors(data)
    )
    if errors:
        raise ValueError("invalid feedback bundle: " + "; ".join(e.message for e in errors[:5]))
    stored = attributed = duplicates = 0
    with registry.db.tx() as s:
        installed = {
            (r.version_id.split("@")[0], r.package_sha256): r.version_id
            for r in s.scalars(select(orm.InstallationRow))
        }
        for ev in data["events"]:
            if s.get(orm.FeedbackRow, ev["event_id"]) is not None:
                duplicates += 1
                continue
            version_id = installed.get((ev["skill_name"], ev.get("package_sha256")))
            if ev.get("skill_invoked") is False:
                version_id = None  # installed but not used: not evidence about the skill
            clean = json.loads(redact_text(json.dumps(ev)))
            s.add(
                orm.FeedbackRow(
                    event_id=ev["event_id"],
                    feedback_id=data["feedback_id"],
                    skill_name=ev["skill_name"],
                    package_sha256=ev.get("package_sha256"),
                    attributed_version_id=version_id,
                    outcome=ev["outcome"],
                    verified=bool(ev.get("verified", False)),
                    payload=clean,
                    created_at=utcnow(),
                )
            )
            stored += 1
            attributed += version_id is not None
    return {
        "feedback_id": data["feedback_id"],
        "stored": stored,
        "attributed": attributed,
        "unattributed": stored - attributed,
        "duplicates": duplicates,
    }


def monitor(registry: Registry, version_id: str) -> dict[str, Any]:
    with registry.db.tx() as s:
        rows = s.scalars(
            select(orm.FeedbackRow).where(orm.FeedbackRow.attributed_version_id == version_id)
        ).all()
        unattributed = s.scalars(
            select(orm.FeedbackRow).where(
                orm.FeedbackRow.skill_name == version_id.split("@")[0],
                orm.FeedbackRow.attributed_version_id.is_(None),
            )
        ).all()
        n = len(rows)
        verified_fail = [r for r in rows if r.outcome == "failure" and r.verified]
        fails = [r for r in rows if r.outcome == "failure"]
        rate = len(fails) / n if n else None
        suspected = len(verified_fail) >= MIN_ATTRIBUTED_FAILURES and (rate or 0) > MAX_FAILURE_RATE
        return {
            "version_id": version_id,
            "attributed_events": n,
            "failures": len(fails),
            "verified_failures": len(verified_fail),
            "failure_rate": round(rate, 3) if rate is not None else None,
            "unattributed_events": len(unattributed),
            "regression_suspected": suspected,
            "rule": f">= {MIN_ATTRIBUTED_FAILURES} verified attributed failures and failure rate "
            f"> {MAX_FAILURE_RATE}",
            "failure_summaries": sorted(
                {r.payload.get("failure_summary") or "" for r in verified_fail} - {""}
            ),
            "note": "Observational post-publication data; not a controlled comparison.",
        }


def propose_revision(registry: Registry, version_id: str) -> dict[str, Any]:
    status = monitor(registry, version_id)
    if not status["regression_suspected"]:
        return {"proposed": False, "reason": "no suspected regression", "monitor": status}
    cv = registry.get_version(version_id)
    spec = registry.get_spec(cv.spec_hash)
    with registry.db.tx() as s:
        rows = s.scalars(
            select(orm.FeedbackRow).where(
                orm.FeedbackRow.attributed_version_id == version_id,
                orm.FeedbackRow.outcome == "failure",
                orm.FeedbackRow.verified.is_(True),
            )
        ).all()
        added = [
            SupportedStatement(
                text="Post-publication failure: "
                + (r.payload.get("failure_summary") or "unspecified"),
                kind="observed",
                support=[f"feedback:{r.event_id}"],
            )
            for r in rows
        ]
    new = ProcedureSpec.model_validate(
        {
            **spec.model_dump(),
            "contraindications": [*spec.contraindications, *added],
            "assumptions": [
                *spec.assumptions,
                f"Revision proposed from {len(rows)} attributed post-publication "
                f"failure(s) of {version_id}; requires review and fresh evaluation.",
            ],
        }
    )
    h = spec_hash(new)
    registry.save_spec(new, h)
    return {
        "proposed": True,
        "spec_hash": h,
        "added_contraindications": [a.text for a in added],
        "next": f"sf generate {h[:12]} --suite <suite>; then validate, evaluate and review",
        "monitor": status,
    }
