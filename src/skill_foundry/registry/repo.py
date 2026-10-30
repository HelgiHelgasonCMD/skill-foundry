"""Repository layer: maps ORM rows to domain models. Every write is transactional."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from skill_foundry.domain.lifecycle import check_transition
from skill_foundry.domain.models import (
    CandidateVersion,
    CapabilityStatus,
    CheckStatus,
    Experience,
    LifecycleState,
    ProcedureSpec,
    ReviewItem,
    TaskCluster,
)
from skill_foundry.registry import orm
from skill_foundry.registry.db import Database
from skill_foundry.util import sha256_bytes, utcnow


class NotFound(LookupError):
    pass


def experience_row(x: Experience, batch_id: str) -> orm.ExperienceRow:
    observed = max((e.observed_at for e in x.evidence), default=None)
    return orm.ExperienceRow(
        experience_id=x.experience_id,
        workspace=x.workspace,
        episode_id=x.episode_id,
        content_hash=x.content_hash,
        outcome_status=x.outcome_status.value,
        trust=x.trust.value,
        repo_id=x.environment.repo_id,
        revision=x.environment.revision,
        env_fingerprint=x.environment.fingerprint(),
        quarantined=x.quarantined,
        observed_at=observed,
        batch_id=batch_id,
        payload=x.model_dump(mode="json"),
        ingested_at=x.ingested_at or utcnow(),
    )


def enqueue_index(s: Session, x: Experience, op: str = "upsert") -> None:
    s.add(
        orm.IndexOutbox(
            workspace=x.workspace,
            experience_id=x.experience_id,
            content_hash=x.content_hash,
            op=op,
            status="pending",
            attempts=0,
            updated_at=utcnow(),
        )
    )


class Registry:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ experiences
    def get_experience(self, experience_id: str) -> Experience:
        with self.db.tx() as s:
            row = s.get(orm.ExperienceRow, experience_id)
            if row is None:
                raise NotFound(experience_id)
            return Experience.model_validate(row.payload)

    def experiences(
        self,
        workspace: str,
        *,
        include_quarantined: bool = False,
        since: datetime | None = None,
        repo_id: str | None = None,
        ids: Iterable[str] | None = None,
    ) -> list[Experience]:
        """Workspace scoping is mandatory: there is no cross-workspace query."""
        q = select(orm.ExperienceRow).where(orm.ExperienceRow.workspace == workspace)
        if not include_quarantined:
            q = q.where(orm.ExperienceRow.quarantined.is_(False))
        if since is not None:
            q = q.where(orm.ExperienceRow.observed_at >= since)
        if repo_id is not None:
            q = q.where(orm.ExperienceRow.repo_id == repo_id)
        if ids is not None:
            q = q.where(orm.ExperienceRow.experience_id.in_(list(ids)))
        with self.db.tx() as s:
            rows = s.scalars(q.order_by(orm.ExperienceRow.experience_id)).all()
            return [Experience.model_validate(r.payload) for r in rows]

    def save_experience(self, s: Session, x: Experience) -> None:
        row = s.get(orm.ExperienceRow, x.experience_id)
        assert row is not None
        row.payload = x.model_dump(mode="json")
        row.trust = x.trust.value
        row.outcome_status = x.outcome_status.value
        row.quarantined = x.quarantined
        row.content_hash = x.content_hash
        enqueue_index(s, x, "delete" if x.quarantined else "upsert")

    # ------------------------------------------------------------------ clusters/specs
    def save_clusters(self, run_id: str, clusters: list[TaskCluster]) -> None:
        with self.db.tx() as s:
            for c in clusters:
                s.merge(
                    orm.ClusterRow(
                        cluster_id=c.cluster_id,
                        workspace=c.workspace,
                        classification=c.classification.value,
                        eligible=c.eligible_for_synthesis,
                        payload=c.model_dump(mode="json"),
                        discovery_run=run_id,
                        created_at=utcnow(),
                    )
                )

    def clusters(self, workspace: str) -> list[TaskCluster]:
        with self.db.tx() as s:
            latest = s.scalar(
                select(orm.ClusterRow.discovery_run)
                .where(orm.ClusterRow.workspace == workspace)
                .order_by(orm.ClusterRow.created_at.desc())
                .limit(1)
            )
            if latest is None:
                return []
            rows = s.scalars(
                select(orm.ClusterRow).where(
                    orm.ClusterRow.workspace == workspace, orm.ClusterRow.discovery_run == latest
                )
            ).all()
            return [TaskCluster.model_validate(r.payload) for r in rows]

    def get_cluster(self, cluster_id: str) -> TaskCluster:
        with self.db.tx() as s:
            rows = s.scalars(
                select(orm.ClusterRow).where(orm.ClusterRow.cluster_id.startswith(cluster_id))
            ).all()
            if len(rows) != 1:
                raise NotFound(f"cluster {cluster_id!r} ({len(rows)} matches)")
            return TaskCluster.model_validate(rows[0].payload)

    def save_spec(self, spec: ProcedureSpec, spec_hash: str) -> None:
        with self.db.tx() as s:
            if s.get(orm.SpecRow, spec_hash) is None:
                s.add(
                    orm.SpecRow(
                        spec_hash=spec_hash,
                        name=spec.name,
                        workspace=spec.workspace,
                        cluster_id=spec.cluster_id,
                        payload=spec.model_dump(mode="json"),
                        created_at=utcnow(),
                    )
                )

    def get_spec(self, spec_hash: str) -> ProcedureSpec:
        with self.db.tx() as s:
            rows = s.scalars(
                select(orm.SpecRow).where(orm.SpecRow.spec_hash.startswith(spec_hash))
            ).all()
            if len(rows) != 1:
                raise NotFound(f"procedure spec {spec_hash!r}")
            return ProcedureSpec.model_validate(rows[0].payload)

    def specs(self, name: str | None = None) -> list[tuple[str, ProcedureSpec]]:
        q = select(orm.SpecRow).order_by(orm.SpecRow.created_at)
        if name:
            q = q.where(orm.SpecRow.name == name)
        with self.db.tx() as s:
            return [(r.spec_hash, ProcedureSpec.model_validate(r.payload)) for r in s.scalars(q)]

    # ------------------------------------------------------------------ versions
    @staticmethod
    def _to_version(r: orm.VersionRow) -> CandidateVersion:
        return CandidateVersion(
            version_id=r.version_id,
            name=r.name,
            version=r.version,
            package_sha256=r.package_sha256,
            spec_hash=r.spec_hash,
            state=LifecycleState(r.state),
            spec_compliance=CheckStatus(r.spec_compliance),
            lint_status=CheckStatus(r.lint_status),
            applicability_status=CheckStatus(r.applicability_status),
            capability_status=CapabilityStatus(r.capability_status),
            evidence_strength=r.evidence_strength,
            review_items=[ReviewItem.model_validate(i) for i in r.review_items],
            overridden=r.overridden,
            created_at=r.created_at,
        )

    def add_version(
        self,
        cv: CandidateVersion,
        files: dict[str, bytes],
        manifest: dict[str, Any],
        changelog: str,
        parent: str | None,
        actor: str,
    ) -> None:
        with self.db.tx() as s:
            if s.get(orm.VersionRow, cv.version_id) is not None:
                raise ValueError(f"{cv.version_id} already exists; versions are immutable")
            s.add(
                orm.VersionRow(
                    version_id=cv.version_id,
                    name=cv.name,
                    version=cv.version,
                    package_sha256=cv.package_sha256,
                    spec_hash=cv.spec_hash,
                    state=cv.state.value,
                    spec_compliance=cv.spec_compliance.value,
                    lint_status=cv.lint_status.value,
                    applicability_status=cv.applicability_status.value,
                    capability_status=cv.capability_status.value,
                    evidence_strength=cv.evidence_strength,
                    review_items=[i.model_dump() for i in cv.review_items],
                    overridden=False,
                    manifest=manifest,
                    changelog=changelog,
                    parent_version_id=parent,
                    created_at=cv.created_at or utcnow(),
                )
            )
            s.flush()
            for path, data in sorted(files.items()):
                digest = sha256_bytes(data)
                if s.get(orm.Blob, digest) is None:
                    s.add(orm.Blob(sha256=digest, data=data))
                s.add(orm.VersionFile(version_id=cv.version_id, path=path, sha256=digest))
            s.add(
                orm.LifecycleEvent(
                    version_id=cv.version_id,
                    from_state=LifecycleState.HYPOTHESIS.value,
                    to_state=cv.state.value,
                    actor=actor,
                    reason=f"generated from procedure spec {cv.spec_hash[:12]}",
                    created_at=utcnow(),
                )
            )

    def get_version(self, ref: str) -> CandidateVersion:
        """``ref`` is ``name@version`` or ``name`` (latest)."""
        with self.db.tx() as s:
            return self._to_version(self._version_row(s, ref))

    def _version_row(self, s: Session, ref: str) -> orm.VersionRow:
        if "@" in ref:
            row = s.get(orm.VersionRow, ref)
        else:
            row = s.scalars(
                select(orm.VersionRow)
                .where(orm.VersionRow.name == ref)
                .order_by(orm.VersionRow.created_at.desc())
                .limit(1)
            ).first()
        if row is None:
            raise NotFound(f"candidate version {ref!r}")
        return row

    def manifest(self, version_id: str) -> dict[str, Any]:
        with self.db.tx() as s:
            return dict(self._version_row(s, version_id).manifest)

    def changelog(self, version_id: str) -> str:
        with self.db.tx() as s:
            return self._version_row(s, version_id).changelog

    def versions(self, name: str | None = None) -> list[CandidateVersion]:
        q = select(orm.VersionRow).order_by(orm.VersionRow.name, orm.VersionRow.created_at)
        if name:
            q = q.where(orm.VersionRow.name == name)
        with self.db.tx() as s:
            return [self._to_version(r) for r in s.scalars(q)]

    def package_files(self, version_id: str) -> dict[str, bytes]:
        """The stored artifact bytes. Never regenerated."""
        with self.db.tx() as s:
            rows = s.execute(
                select(orm.VersionFile.path, orm.Blob.data)
                .join(orm.Blob, orm.Blob.sha256 == orm.VersionFile.sha256)
                .where(orm.VersionFile.version_id == version_id)
            ).all()
            return {p: bytes(d) for p, d in rows}

    def update_version(self, version_id: str, **fields: Any) -> CandidateVersion:
        with self.db.tx() as s:
            row = self._version_row(s, version_id)
            for k, v in fields.items():
                if k == "review_items":
                    v = [i.model_dump() for i in v]
                elif hasattr(v, "value"):
                    v = v.value
                setattr(row, k, v)
            return self._to_version(row)

    def transition(
        self, version_id: str, target: LifecycleState, actor: str, reason: str
    ) -> CandidateVersion:
        with self.db.tx() as s:
            row = self._version_row(s, version_id)
            current = LifecycleState(row.state)
            check_transition(current, target)
            row.state = target.value
            s.add(
                orm.LifecycleEvent(
                    version_id=row.version_id,
                    from_state=current.value,
                    to_state=target.value,
                    actor=actor,
                    reason=reason,
                    created_at=utcnow(),
                )
            )
            return self._to_version(row)

    def history(self, version_id: str) -> list[dict[str, Any]]:
        with self.db.tx() as s:
            rows = s.scalars(
                select(orm.LifecycleEvent)
                .where(orm.LifecycleEvent.version_id == version_id)
                .order_by(orm.LifecycleEvent.id)
            ).all()
            return [
                {
                    "from": r.from_state,
                    "to": r.to_state,
                    "actor": r.actor,
                    "reason": r.reason,
                    "at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    def add_check(self, version_id: str, kind: str, status: str, report: dict[str, Any]) -> None:
        with self.db.tx() as s:
            s.add(
                orm.CheckResult(
                    version_id=version_id,
                    kind=kind,
                    status=status,
                    report=report,
                    created_at=utcnow(),
                )
            )

    def checks(self, version_id: str) -> list[dict[str, Any]]:
        with self.db.tx() as s:
            rows = s.scalars(
                select(orm.CheckResult)
                .where(orm.CheckResult.version_id == version_id)
                .order_by(orm.CheckResult.id)
            ).all()
            return [
                {"kind": r.kind, "status": r.status, "report": r.report, "at": r.created_at}
                for r in rows
            ]

    # ------------------------------------------------------------------ misc counts
    def counts(self, workspace: str) -> dict[str, int]:
        with self.db.tx() as s:

            def c(*conds: Any) -> int:
                return int(
                    s.scalar(
                        select(func.count())
                        .select_from(orm.ExperienceRow)
                        .where(orm.ExperienceRow.workspace == workspace, *conds)
                    )
                    or 0
                )

            return {
                "experiences": c(),
                "verified": c(orm.ExperienceRow.trust == "verified"),
                "quarantined": c(orm.ExperienceRow.quarantined.is_(True)),
                "verified_success": c(
                    orm.ExperienceRow.outcome_status == "verified_success",
                    orm.ExperienceRow.trust == "verified",
                    orm.ExperienceRow.quarantined.is_(False),
                ),
                "counterexamples": c(orm.ExperienceRow.outcome_status == "verified_failure"),
            }

    def mark_outbox(self, ids: list[int], status: str, error: str | None = None) -> None:
        with self.db.tx() as s:
            s.execute(
                update(orm.IndexOutbox)
                .where(orm.IndexOutbox.id.in_(ids))
                .values(
                    status=status,
                    last_error=error,
                    attempts=orm.IndexOutbox.attempts + 1,
                    updated_at=utcnow(),
                )
            )
