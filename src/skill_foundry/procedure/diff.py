"""Structured comparison of two ProcedureSpecs (what changed in the *evidence-derived*
procedure, independent of wording)."""

from __future__ import annotations

from typing import Any

from skill_foundry.domain.models import ProcedureSpec


def _set(items: list[Any], attr: str) -> set[str]:
    return {getattr(i, attr) for i in items}


def spec_diff(old: ProcedureSpec | None, new: ProcedureSpec) -> dict[str, Any]:
    if old is None:
        return {"initial": True, "steps_added": [s.action for s in new.steps]}
    out: dict[str, Any] = {"initial": False}
    for field, attr in (
        ("steps", "action"),
        ("contraindications", "text"),
        ("preconditions", "text"),
        ("failed_approaches", "text"),
    ):
        a, b = _set(getattr(old, field), attr), _set(getattr(new, field), attr)
        if b - a:
            out[f"{field}_added"] = sorted(b - a)
        if a - b:
            out[f"{field}_removed"] = sorted(a - b)
    ob = {b.condition for b in old.branches}
    nb = {b.condition for b in new.branches}
    if nb - ob:
        out["branches_added"] = sorted(nb - ob)
    if ob - nb:
        out["branches_removed"] = sorted(ob - nb)
    added_src = sorted(set(new.source_experience_ids) - set(old.source_experience_ids))
    removed_src = sorted(set(old.source_experience_ids) - set(new.source_experience_ids))
    if added_src:
        out["evidence_added"] = added_src
    if removed_src:
        out["evidence_removed"] = removed_src
    if old.evidence_score.total != new.evidence_score.total:
        out["evidence_score"] = [old.evidence_score.total, new.evidence_score.total]
    return out


def changelog_text(diff: dict[str, Any], version: str) -> str:
    if diff.get("initial"):
        return f"{version}: initial candidate ({len(diff['steps_added'])} evidence-backed steps)."
    parts = [f"{version}:"]
    for k, v in diff.items():
        if k == "initial":
            continue
        parts.append(f"{k.replace('_', ' ')}: {v}")
    if len(parts) == 1:
        parts.append("re-rendered with no change to the procedure spec")
    return " ".join(parts)
