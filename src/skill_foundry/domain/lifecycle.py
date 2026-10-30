"""Lifecycle state machine for candidate skill versions."""

from __future__ import annotations

from skill_foundry.domain.models import LifecycleState as S

TRANSITIONS: dict[S, frozenset[S]] = {
    S.HYPOTHESIS: frozenset({S.CANDIDATE, S.DEPRECATED}),
    S.CANDIDATE: frozenset({S.VALIDATED, S.DEPRECATED}),
    S.VALIDATED: frozenset({S.APPROVED, S.CANDIDATE, S.DEPRECATED}),
    S.APPROVED: frozenset({S.PUBLISHED, S.DEPRECATED}),
    S.PUBLISHED: frozenset({S.ROLLED_BACK, S.DEPRECATED}),
    S.ROLLED_BACK: frozenset({S.PUBLISHED, S.DEPRECATED}),
    S.DEPRECATED: frozenset(),
}


class InvalidTransition(ValueError):
    pass


def check_transition(current: S, target: S) -> None:
    if target not in TRANSITIONS[current]:
        allowed = ", ".join(sorted(t.value for t in TRANSITIONS[current])) or "none"
        raise InvalidTransition(
            f"illegal lifecycle transition {current.value} -> {target.value} (allowed: {allowed})"
        )
