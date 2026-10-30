"""Human labelling and correction of imported experiences (always audited)."""

from __future__ import annotations

from skill_foundry.domain.models import AcceptanceLink, Experience, OutcomeStatus
from skill_foundry.ingest.qualify import qualify
from skill_foundry.registry import orm
from skill_foundry.registry.repo import NotFound, Registry
from skill_foundry.util import utcnow

LABEL_KEYS = ("outcome", "attest-acceptance", "quarantine", "lineage", "note")


def apply_label(
    registry: Registry, experience_id: str, key: str, value: str, author: str, reason: str
) -> Experience:
    if not reason.strip():
        raise ValueError("a reason is required for every label/correction")
    if not author.strip():
        raise ValueError("an author is required")
    with registry.db.tx() as s:
        row = s.get(orm.ExperienceRow, experience_id)
        if row is None:
            raise NotFound(experience_id)
        x = Experience.model_validate(row.payload)
        if key == "outcome":
            x.outcome_status = OutcomeStatus(value)
        elif key == "attest-acceptance":
            # value: "<evidence_id>::<criterion>[;;<criterion>...]"
            ev_id, _, crit = value.partition("::")
            criteria = [c.strip() for c in crit.split(";;") if c.strip()]
            if not criteria:
                raise ValueError("attest-acceptance needs '<evidence_id>::<criterion>'")
            found = False
            evidence = []
            for e in x.evidence:
                if e.evidence_id == ev_id:
                    found = True
                    e = e.model_copy(
                        update={
                            "acceptance": AcceptanceLink(
                                criteria=criteria,
                                tested_revision=e.revision or x.environment.revision,
                                worktree_fingerprint=x.environment.worktree_fingerprint,
                                basis="human_attested",
                                attested_by=author,
                            )
                        }
                    )
                evidence.append(e)
            if not found:
                raise NotFound(f"evidence {ev_id} in {experience_id}")
            x.evidence = evidence
        elif key == "quarantine":
            x.quarantined = value.lower() in ("1", "true", "yes")
            x.quarantine_reasons = (
                [*x.quarantine_reasons, f"quarantined by {author}: {reason}"]
                if x.quarantined
                else []
            )
        elif key == "lineage":
            x.task_lineage = value
        elif key == "note":
            pass
        else:
            raise ValueError(f"unknown label key {key!r}; expected one of {LABEL_KEYS}")
        x.labels = {**x.labels, key: value}
        x = qualify(x)
        registry.save_experience(s, x)
        s.add(
            orm.LabelRow(
                experience_id=experience_id,
                key=key,
                value=value,
                author=author,
                reason=reason,
                created_at=utcnow(),
            )
        )
        return x
