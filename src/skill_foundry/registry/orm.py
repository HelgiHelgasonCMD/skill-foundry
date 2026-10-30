"""SQLAlchemy persistence models. Kept separate from ``skill_foundry.domain``.

SQLite is the source of truth. The Qdrant index is derived from these tables and
can be rebuilt at any time (see ``retrieval.index``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Bump together with a new Alembic revision.
SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[Any]: JSON}


class Meta(Base):
    __tablename__ = "sf_meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class IngestBatch(Base):
    __tablename__ = "ingest_batches"
    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(Text)
    source_sha256: Mapped[str] = mapped_column(String(64))
    export_id: Mapped[str | None] = mapped_column(String(256))
    format: Mapped[str] = mapped_column(String(32))
    stats: Mapped[dict[str, Any]] = mapped_column(JSON)
    rejections: Mapped[list[Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ExperienceRow(Base):
    __tablename__ = "experiences"
    experience_id: Mapped[str] = mapped_column(String(400), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    episode_id: Mapped[str] = mapped_column(String(256))
    content_hash: Mapped[str] = mapped_column(String(64))
    outcome_status: Mapped[str] = mapped_column(String(32))
    trust: Mapped[str] = mapped_column(String(32))
    repo_id: Mapped[str | None] = mapped_column(String(256))
    revision: Mapped[str | None] = mapped_column(String(128))
    env_fingerprint: Mapped[str] = mapped_column(String(256))
    quarantined: Mapped[bool] = mapped_column(Boolean, default=False)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    batch_id: Mapped[str] = mapped_column(ForeignKey("ingest_batches.batch_id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)  # sanitised domain JSON
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        UniqueConstraint("workspace", "content_hash", name="uq_experience_content"),
        Index("ix_experience_ws_status", "workspace", "outcome_status", "trust"),
    )


class LabelRow(Base):
    __tablename__ = "experience_labels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.experience_id"), index=True)
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class IndexOutbox(Base):
    __tablename__ = "index_outbox"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    experience_id: Mapped[str] = mapped_column(String(400))
    content_hash: Mapped[str] = mapped_column(String(64))
    op: Mapped[str] = mapped_column(String(16))  # upsert | delete
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending | done | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class IndexMetaRow(Base):
    __tablename__ = "index_meta"
    collection: Mapped[str] = mapped_column(String(128), primary_key=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ClusterRow(Base):
    __tablename__ = "clusters"
    cluster_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    classification: Mapped[str] = mapped_column(String(32))
    eligible: Mapped[bool] = mapped_column(Boolean)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    discovery_run: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SpecRow(Base):
    __tablename__ = "procedure_specs"
    spec_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    workspace: Mapped[str] = mapped_column(String(128))
    cluster_id: Mapped[str] = mapped_column(ForeignKey("clusters.cluster_id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Blob(Base):
    __tablename__ = "blobs"
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)


class VersionRow(Base):
    __tablename__ = "candidate_versions"
    version_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    package_sha256: Mapped[str] = mapped_column(String(64))
    spec_hash: Mapped[str] = mapped_column(ForeignKey("procedure_specs.spec_hash"))
    state: Mapped[str] = mapped_column(String(32))
    spec_compliance: Mapped[str] = mapped_column(String(16))
    lint_status: Mapped[str] = mapped_column(String(16))
    applicability_status: Mapped[str] = mapped_column(String(16))
    capability_status: Mapped[str] = mapped_column(String(32))
    evidence_strength: Mapped[str] = mapped_column(String(16))
    review_items: Mapped[list[Any]] = mapped_column(JSON)
    overridden: Mapped[bool] = mapped_column(Boolean, default=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    changelog: Mapped[str] = mapped_column(Text)
    parent_version_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("name", "version", name="uq_name_version"),)


class VersionFile(Base):
    __tablename__ = "version_files"
    version_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_versions.version_id"), primary_key=True
    )
    path: Mapped[str] = mapped_column(String(512), primary_key=True)
    sha256: Mapped[str] = mapped_column(ForeignKey("blobs.sha256"))


class LifecycleEvent(Base):
    __tablename__ = "lifecycle_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32))
    actor: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CheckResult(Base):
    __tablename__ = "check_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # spec | lint | applicability
    status: Mapped[str] = mapped_column(String(16))
    report: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SynthesisLog(Base):
    __tablename__ = "synthesis_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    input_sha256: Mapped[str] = mapped_column(String(64))
    output_sha256: Mapped[str] = mapped_column(String(64))
    input_redacted: Mapped[str | None] = mapped_column(Text)  # cleared after retention
    output_redacted: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalRunRow(Base):
    __tablename__ = "eval_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # applicability | capability
    runner_kind: Mapped[str] = mapped_column(String(16))  # static | fake | mock | live
    status: Mapped[str] = mapped_column(String(32))
    report: Mapped[dict[str, Any]] = mapped_column(JSON)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ReviewRow(Base):
    __tablename__ = "review_decisions"
    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    package_sha256: Mapped[str] = mapped_column(String(64))
    reviewer: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(16))
    rationale: Mapped[str] = mapped_column(Text)
    override: Mapped[bool] = mapped_column(Boolean)
    overridden_gates: Mapped[list[Any]] = mapped_column(JSON)
    acknowledged_items: Mapped[list[Any]] = mapped_column(JSON)
    policy_result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InstallationRow(Base):
    __tablename__ = "installations"
    install_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("candidate_versions.version_id"), index=True)
    package_sha256: Mapped[str] = mapped_column(String(64))
    decision_id: Mapped[str] = mapped_column(ForeignKey("review_decisions.decision_id"))
    publisher: Mapped[str] = mapped_column(String(64))
    target_root: Mapped[str] = mapped_column(Text)
    destination: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(32))
    before_hashes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after_hashes: Mapped[dict[str, Any]] = mapped_column(JSON)
    previous_install_id: Mapped[str | None] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class InstallEvent(Base):
    __tablename__ = "installation_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    install_id: Mapped[str] = mapped_column(ForeignKey("installations.install_id"), index=True)
    action: Mapped[str] = mapped_column(String(32))  # install | rollback | recover
    detail: Mapped[dict[str, Any]] = mapped_column(JSON)
    actor: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FeedbackRow(Base):
    __tablename__ = "feedback_events"
    event_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    feedback_id: Mapped[str] = mapped_column(String(256))
    skill_name: Mapped[str] = mapped_column(String(64), index=True)
    package_sha256: Mapped[str | None] = mapped_column(String(64))
    attributed_version_id: Mapped[str | None] = mapped_column(String(100))
    outcome: Mapped[str] = mapped_column(String(16))
    verified: Mapped[bool] = mapped_column(Boolean)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
