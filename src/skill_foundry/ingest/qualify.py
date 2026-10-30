"""Evidence qualification.

Schema-valid evidence is *not* trusted evidence. An experience reaches
``verified`` only if at least one evidence item:

* is an objective check kind (test / ci / replay / external_check),
* exited with code 0,
* is acceptance-linked: it names the task's acceptance criteria and the tested
  revision (and working-tree fingerprint when one is known), and
* that tested state matches the episode's recorded environment.

A successful command that is not linked to acceptance criteria stays ``recorded``:
an exit code of zero shows that *something* ran, not that the task was solved.
"""

from __future__ import annotations

from skill_foundry.domain.models import (
    EvidenceItem,
    Experience,
    OutcomeStatus,
    TrustLevel,
)

OBJECTIVE_KINDS = frozenset({"test", "ci", "replay", "external_check"})
QUALIFIER_VERSION = "qualifier.v1"


def qualify_evidence(item: EvidenceItem, x: Experience) -> EvidenceItem:
    reasons: list[str] = []
    env_rev = x.environment.revision
    if item.exit_code is None:
        reasons.append("no exit code recorded")
    elif item.exit_code != 0:
        reasons.append(f"exit code {item.exit_code}")
    if item.kind not in OBJECTIVE_KINDS:
        reasons.append(f"kind '{item.kind}' is not an objective acceptance check")
    acc = item.acceptance
    if acc is None or not acc.criteria:
        reasons.append("not linked to task acceptance criteria")
    else:
        tested = acc.tested_revision or item.revision
        if tested is None or env_rev is None:
            reasons.append("tested revision unknown; cannot tie result to the tested state")
        elif tested != env_rev:
            reasons.append(f"stale evidence: tested {tested[:12]} but episode is at {env_rev[:12]}")
        if item.revision is not None and env_rev is not None and item.revision != env_rev:
            reasons.append("evidence revision differs from episode revision")
        wt_env = x.environment.worktree_fingerprint
        if acc.worktree_fingerprint and wt_env and acc.worktree_fingerprint != wt_env:
            reasons.append("working-tree fingerprint mismatch")

    if not reasons:
        basis = acc.basis if acc else "?"
        return item.model_copy(
            update={"trust": TrustLevel.VERIFIED, "trust_reasons": [f"acceptance-linked ({basis})"]}
        )
    level = TrustLevel.RECORDED if item.exit_code is not None else TrustLevel.SELF_REPORTED
    return item.model_copy(update={"trust": level, "trust_reasons": reasons})


def qualify(x: Experience) -> Experience:
    """Return a copy of ``x`` with per-evidence and per-experience trust assigned."""
    evidence = [qualify_evidence(e, x) for e in x.evidence]
    reasons: list[str] = []
    status = x.outcome_status
    if status in (OutcomeStatus.UNKNOWN, OutcomeStatus.CLAIMED_SUCCESS):
        trust = TrustLevel.SELF_REPORTED
        reasons.append(f"outcome is '{status.value}': producer claim only")
    elif status == OutcomeStatus.VERIFIED_SUCCESS:
        verified = [e for e in evidence if e.trust == TrustLevel.VERIFIED]
        if verified:
            trust = TrustLevel.VERIFIED
            reasons.append(
                f"{len(verified)} acceptance-linked check(s): "
                + ", ".join(e.evidence_id for e in verified)
            )
        else:
            trust = TrustLevel.RECORDED
            reasons.append("verified_success claimed, but no evidence item qualifies:")
            reasons += [f"  {e.evidence_id}: {'; '.join(e.trust_reasons)}" for e in evidence]
    else:  # verified_failure: a counterexample. Failure evidence needs less: a failing check.
        failing = [e for e in evidence if e.exit_code not in (None, 0)]
        trust = TrustLevel.VERIFIED if failing else TrustLevel.RECORDED
        reasons.append(
            "failure confirmed by non-zero check(s)"
            if failing
            else "failure not confirmed by a check"
        )
    if not x.provenance.redaction_applied:
        reasons.append("producer did not redact; Skill Foundry sanitiser applied")
    return x.model_copy(update={"evidence": evidence, "trust": trust, "trust_reasons": reasons})
