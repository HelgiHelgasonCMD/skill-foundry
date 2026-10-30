"""Domain models (pure Pydantic, no persistence concerns).

Terminology used throughout the code base:

* *observed*  - a fact copied (after sanitisation) from an imported experience.
* *reported*  - a claim made by the producer of the experience (e.g. a lesson);
                 never treated as proof on its own.
* *inferred*  - derived by Skill Foundry from several observed facts.
* *model_suggestion* - proposed by an LLM synthesiser with no supporting evidence;
                 always requires human review.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- evidence


class TrustLevel(StrEnum):
    """How much an outcome claim can be relied upon.

    ``self_reported`` - the agent or producer says so; schema validity only.
    ``recorded``      - a check was captured, but it is not linked to the task's
                        acceptance criteria and tested state.
    ``verified``      - an objective, acceptance-linked check on the tested revision.
    """

    SELF_REPORTED = "self_reported"
    RECORDED = "recorded"
    VERIFIED = "verified"


class AcceptanceLink(Frozen):
    """Links a piece of evidence to the task's acceptance criteria and tested state."""

    criteria: list[str]
    tested_revision: str | None = None
    worktree_fingerprint: str | None = None
    basis: Literal["producer_declared", "human_attested"] = "producer_declared"
    attested_by: str | None = None


class EvidenceItem(Frozen):
    evidence_id: str
    kind: Literal["test", "ci", "command", "replay", "external_check", "other"]
    summary: str
    observed_at: datetime
    revision: str | None = None
    exit_code: int | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    acceptance: AcceptanceLink | None = None
    trust: TrustLevel = TrustLevel.SELF_REPORTED
    trust_reasons: list[str] = Field(default_factory=list)


class Observation(Frozen):
    observation_id: str
    kind: str
    summary: str
    error_code: str | None = None
    timestamp: datetime | None = None
    source_event_ids: list[str] = Field(default_factory=list)


class Attempt(Frozen):
    attempt_id: str
    action_summary: str
    status: Literal["unknown", "failed", "completed"]
    result_summary: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)


class Lesson(Frozen):
    lesson_id: str
    problem: str
    observed_symptoms: list[str]
    failed_approaches: list[str]
    successful_approach: str | None
    preconditions: list[str]
    contraindications: list[str]
    versions: dict[str, str]
    verification_evidence_ids: list[str]
    source_episode_ids: list[str]
    confidence_basis: str


class Environment(Frozen):
    repo_id: str | None
    revision: str | None
    versions: dict[str, str]
    runtime: str | None = None
    platform: str | None = None
    worktree_fingerprint: str | None = None

    def runtime_family(self) -> str | None:
        if not self.runtime:
            return None
        return self.runtime.strip().lower().split()[0].rstrip("0123456789.")

    def platform_family(self) -> str | None:
        if not self.platform:
            return None
        return self.platform.strip().lower().split("-")[0].split()[0]

    def fingerprint(self) -> str:
        """Coarse environment identity used for independence and compatibility."""
        minors = sorted(f"{k}={version_prefix(v, 2)}" for k, v in self.versions.items())
        return "|".join([self.runtime_family() or "?", self.platform_family() or "?", *minors])


def version_prefix(version: str, parts: int) -> str:
    """``version_prefix("v3.12.4", 2) == "3.12"``."""
    return ".".join(version.strip().lstrip("vV").split(".")[:parts])


class Provenance(Frozen):
    source: str
    event_ids: list[str]
    redaction_applied: bool
    bundle_export_id: str | None = None
    bundle_sha256: str | None = None
    sanitizer_version: str | None = None


class OutcomeStatus(StrEnum):
    UNKNOWN = "unknown"
    CLAIMED_SUCCESS = "claimed_success"
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"


class Experience(Mutable):
    """One imported, sanitised episode scoped to a workspace."""

    experience_id: str  # "<workspace>/<episode_id>"
    workspace: str
    episode_id: str
    task_summary: str
    environment: Environment
    observations: list[Observation]
    attempts: list[Attempt]
    outcome_status: OutcomeStatus
    evidence: list[EvidenceItem]
    lessons: list[Lesson]
    provenance: Provenance
    trust: TrustLevel
    trust_reasons: list[str]
    content_hash: str
    task_lineage: str | None = None  # producer-declared lineage, if any
    quarantined: bool = False
    quarantine_reasons: list[str] = Field(default_factory=list)
    redactions: int = 0
    labels: dict[str, str] = Field(default_factory=dict)
    ingested_at: datetime | None = None

    @property
    def is_positive(self) -> bool:
        return (
            self.outcome_status == OutcomeStatus.VERIFIED_SUCCESS
            and self.trust == TrustLevel.VERIFIED
            and not self.quarantined
        )

    @property
    def is_counterexample(self) -> bool:
        return self.outcome_status == OutcomeStatus.VERIFIED_FAILURE and not self.quarantined

    def text(self) -> str:
        """Sanitised text used for retrieval. Never includes hidden reasoning."""
        parts = [self.task_summary]
        parts += [o.summary for o in self.observations]
        parts += [a.action_summary for a in self.attempts]
        parts += [lesson.problem for lesson in self.lessons]
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- mining


class ScoreFactor(Frozen):
    name: str
    value: float
    weight: float
    contribution: float
    explanation: str


class EvidenceScore(Frozen):
    total: float
    label: Literal["insufficient", "weak", "moderate", "strong"]
    factors: list[ScoreFactor]


class IndependenceUnit(Frozen):
    unit_id: str
    lineage: str
    repo_id: str | None
    env_fingerprint: str
    experience_ids: list[str]


class RejectedMember(Frozen):
    experience_id: str
    reason: str


class Contradiction(Frozen):
    kind: str
    detail: str
    experience_ids: list[str]


class Reusability(StrEnum):
    ONE_OFF = "one_off"
    ENVIRONMENT_SPECIFIC = "environment_specific"
    REUSABLE = "reusable"
    INSUFFICIENT = "insufficient_evidence"


class TaskCluster(Frozen):
    cluster_id: str
    workspace: str
    member_ids: list[str]
    representative_id: str
    min_pairwise: float
    mean_pairwise: float
    pairwise: dict[str, float]  # "a|b" -> similarity, for inspection
    outliers: list[str]
    rejected: list[RejectedMember]
    counterexample_ids: list[str]
    contradictions: list[Contradiction]
    units: list[IndependenceUnit]
    classification: Reusability
    classification_reasons: list[str]
    evidence_score: EvidenceScore
    eligible_for_synthesis: bool
    exploratory: bool
    config_hash: str


# --------------------------------------------------------------------------- procedure


StepOrigin = Literal["observed_inference", "lesson_reported", "model_suggestion"]
StatementKind = Literal["observed", "reported", "inferred", "model_suggestion"]


class SupportedStatement(Frozen):
    text: str
    kind: StatementKind
    support: list[str]  # experience ids (or evidence ids prefixed "ev:")


class ProcedureStep(Frozen):
    step_id: str
    action: str
    commands: list[str]
    rationale: str
    support: list[str]
    support_units: int
    support_ratio: float
    origin: StepOrigin = "observed_inference"
    needs_review: bool = False


class DecisionBranch(Frozen):
    branch_id: str
    condition: str
    condition_basis: Literal["environment", "observation", "unresolved"]
    steps: list[ProcedureStep]
    support: list[str]
    needs_review: bool = False


class VerificationRequirement(Frozen):
    description: str
    commands: list[str]
    expected: str
    evidence_refs: list[str]  # "experience_id#evidence_id"


class RecoveryAction(Frozen):
    when: str
    action: str
    commands: list[str]
    support: list[str]


class EnvironmentCompatibility(Frozen):
    supported: list[str]
    unsupported: list[str]
    notes: list[str]


class ExcludedStep(Frozen):
    action: str
    reason: str
    support: list[str]


class ProcedureSpec(Frozen):
    """Typed intermediate representation between evidence and SKILL.md."""

    spec_version: Literal["procedure-spec.v1"] = "procedure-spec.v1"
    name: str
    title: str
    trigger: str
    in_scope: list[str]
    out_of_scope: list[str]
    preconditions: list[SupportedStatement]
    steps: list[ProcedureStep]
    branches: list[DecisionBranch]
    verification: list[VerificationRequirement]
    recovery: list[RecoveryAction]
    failed_approaches: list[SupportedStatement]
    contraindications: list[SupportedStatement]
    environments: EnvironmentCompatibility
    assumptions: list[str]
    excluded_steps: list[ExcludedStep]
    cluster_id: str
    workspace: str
    source_experience_ids: list[str]
    counterexample_ids: list[str]
    evidence_score: EvidenceScore
    exploratory: bool
    inference_config_hash: str
    keywords: list[str]


# --------------------------------------------------------------------------- lifecycle


class LifecycleState(StrEnum):
    HYPOTHESIS = "hypothesis"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


class CheckStatus(StrEnum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class CapabilityStatus(StrEnum):
    NOT_MEASURED = "not_measured"
    SMOKE_ONLY = "smoke_only"  # fake / mock runner only: never evidence of benefit
    INCONCLUSIVE = "inconclusive"  # live, but sample too small or no difference
    NO_BENEFIT = "no_benefit"
    MEASURED_BENEFIT = "measured_benefit"
    REGRESSION = "regression"


class FileEntry(Frozen):
    path: str
    sha256: str
    size: int


class ArtifactManifest(Frozen):
    """Immutable record of exactly what a candidate version is and how it was made."""

    manifest_version: Literal["artifact-manifest.v1"] = "artifact-manifest.v1"
    skill_name: str
    version: str
    package_sha256: str
    files: list[FileEntry]
    spec_hash: str
    spec_version: str
    evidence_snapshot: dict[str, str]  # experience_id -> content_hash
    synthesizer: dict[str, Any]
    template_version: str
    retrieval_config: dict[str, Any]
    clustering_config: dict[str, Any]
    eval_suite: dict[str, str] | None
    created_at: datetime
    skill_foundry_version: str


class ValidationIssue(Frozen):
    code: str
    severity: Literal["error", "warning"]
    message: str
    path: str | None = None


class ValidationReport(Frozen):
    kind: Literal["spec", "lint"]
    passed: bool
    issues: list[ValidationIssue]
    reference_validator: str | None = None


class ReviewItem(Frozen):
    item_id: str
    detail: str


class CandidateVersion(Mutable):
    version_id: str  # "<name>@<version>"
    name: str
    version: str
    package_sha256: str
    spec_hash: str
    state: LifecycleState
    spec_compliance: CheckStatus = CheckStatus.NOT_RUN
    lint_status: CheckStatus = CheckStatus.NOT_RUN
    applicability_status: CheckStatus = CheckStatus.NOT_RUN
    capability_status: CapabilityStatus = CapabilityStatus.NOT_MEASURED
    evidence_strength: str = "insufficient"
    review_items: list[ReviewItem] = Field(default_factory=list)
    overridden: bool = False
    created_at: datetime | None = None


class ReviewDecision(Frozen):
    decision_id: str
    version_id: str
    package_sha256: str
    reviewer: str
    decision: Literal["approve", "reject"]
    rationale: str
    override: bool
    overridden_gates: list[str]
    acknowledged_items: list[str]
    created_at: datetime


class PolicyGate(Frozen):
    gate: str
    passed: bool
    detail: str


class PolicyResult(Frozen):
    allowed: bool
    gates: list[PolicyGate]

    @property
    def failed(self) -> list[PolicyGate]:
        return [g for g in self.gates if not g.passed]
