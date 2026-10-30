"""Promotion policy.

Technical validity (a well-formed SKILL.md) and measured effectiveness are separate
gates. By default nothing is promotable without *live* paired evidence of benefit;
a human may override, and the override is recorded and displayed forever after.
"""

from __future__ import annotations

from dataclasses import dataclass

from skill_foundry.domain.models import (
    CandidateVersion,
    CapabilityStatus,
    CheckStatus,
    PolicyGate,
    PolicyResult,
)


@dataclass(frozen=True)
class PromotionPolicy:
    require_capability: frozenset[CapabilityStatus] = frozenset({CapabilityStatus.MEASURED_BENEFIT})
    require_applicability: bool = True
    min_evidence_strength: tuple[str, ...] = ("moderate", "strong")

    def evaluate(
        self, version: CandidateVersion, acknowledged: set[str] | None = None
    ) -> PolicyResult:
        ack = acknowledged or set()
        open_items = [i.item_id for i in version.review_items if i.item_id not in ack]
        gates = [
            PolicyGate(
                gate="spec_compliance",
                passed=version.spec_compliance == CheckStatus.PASS,
                detail=f"spec validation: {version.spec_compliance.value}",
            ),
            PolicyGate(
                gate="content_lint",
                passed=version.lint_status in (CheckStatus.PASS, CheckStatus.WARN),
                detail=f"content/security lint: {version.lint_status.value}",
            ),
            PolicyGate(
                gate="applicability",
                passed=(not self.require_applicability)
                or version.applicability_status == CheckStatus.PASS,
                detail=f"applicability & safety eval: {version.applicability_status.value}",
            ),
            PolicyGate(
                gate="no_regression",
                passed=version.capability_status != CapabilityStatus.REGRESSION,
                detail="critical regression detected"
                if version.capability_status == CapabilityStatus.REGRESSION
                else "no regression recorded",
            ),
            PolicyGate(
                gate="measured_capability",
                passed=version.capability_status in self.require_capability,
                detail=f"capability status: {version.capability_status.value} "
                f"(required: {', '.join(sorted(s.value for s in self.require_capability))})",
            ),
            PolicyGate(
                gate="evidence_strength",
                passed=version.evidence_strength in self.min_evidence_strength,
                detail=f"evidence strength: {version.evidence_strength}",
            ),
            PolicyGate(
                gate="review_items_acknowledged",
                passed=not open_items,
                detail="all review items acknowledged"
                if not open_items
                else f"unacknowledged: {', '.join(open_items)}",
            ),
        ]
        return PolicyResult(allowed=all(g.passed for g in gates), gates=gates)


# Gates that an override can never bypass: a malformed or unsafe package must be
# fixed rather than overridden. Regressions and missing measurements can be
# overridden by a named human with a rationale; the override is recorded.
NON_OVERRIDABLE = frozenset({"spec_compliance", "content_lint"})
