"""Render a ProcedureSpec (+ synthesiser wording) into an Agent Skill package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from skill_foundry.domain.models import ProcedureSpec
from skill_foundry.procedure.infer import spec_hash

TEMPLATE_VERSION = "skill-template.v1"
_TEMPLATES = Path(__file__).resolve().parent / "templates"


def _yaml_str(value: str) -> str:
    # Emit a safely quoted single-line YAML scalar.
    dumped: str = yaml.safe_dump(str(value), default_style='"', width=10_000).strip()
    return dumped.removesuffix("\n...").strip()


def _env() -> Environment:
    env = Environment(  # noqa: S701 - output is Markdown, not HTML
        loader=FileSystemLoader(_TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.filters["yaml_str"] = _yaml_str
    env.filters["capitalize_first"] = lambda s: s[:1].upper() + s[1:]
    env.filters["lower_first"] = lambda s: s[:1].lower() + s[1:]
    return env


def build_description(spec: ProcedureSpec, override: str | None = None) -> str:
    if override:
        return override[:1024]
    verbs = "; ".join(s.action.split(" with ")[0] for s in spec.steps[:4])
    contra = [c.text for c in spec.contraindications if c.kind == "reported"][:1]
    desc = (
        f"{spec.title}: an evidence-backed procedure ({verbs}). "
        f"Use when the task is to {spec.trigger[0].lower() + spec.trigger[1:]}."
    )
    if spec.keywords:
        desc += f" Keywords: {', '.join(spec.keywords)}."
    if contra:
        desc += f" Do not use when {contra[0]}."
    return desc[:1024]


def build_compatibility(spec: ProcedureSpec) -> str:
    runtimes = sorted({fp.split("|")[0] for fp in spec.environments.supported})
    text = (
        f"Evidence from {', '.join(runtimes)} environments "
        f"({len(spec.environments.supported)} fingerprints). Requires the tools named in the "
        "procedure; no network access is assumed."
    )
    return text[:500]


def evidence_markdown(spec: ProcedureSpec, lineage: dict[str, str]) -> str:
    lines = [
        f"# Evidence for `{spec.name}`",
        "",
        "Every procedure step below lists the experiences that support it. Experience IDs "
        "refer to records in the Skill Foundry registry (`sf lineage` shows the full chain "
        "down to evidence items and source events).",
        "",
        f"- Cluster: `{spec.cluster_id}`",
        f"- Evidence score: **{spec.evidence_score.total} ({spec.evidence_score.label})**",
        "",
        "| factor | value | weight | contribution | explanation |",
        "|---|---|---|---|---|",
    ]
    for f in spec.evidence_score.factors:
        lines.append(f"| {f.name} | {f.value} | {f.weight} | {f.contribution} | {f.explanation} |")
    lines += [
        "",
        "## Steps",
        "",
        "| step | action | lineages | supporting experiences |",
        "|---|---|---|---|",
    ]
    for s in spec.steps:
        lines.append(
            f"| {s.step_id} | {s.action} | {s.support_units} | "
            f"{', '.join(f'`{e}`' for e in s.support)} |"
        )
    for b in spec.branches:
        for s in b.steps:
            lines.append(
                f"| {s.step_id} | {b.condition}: {s.action} | {s.support_units} | "
                f"{', '.join(f'`{e}`' for e in s.support)} |"
            )
    lines += ["", "## Verification evidence", ""]
    for v in spec.verification:
        lines.append(f"- {v.description}: {', '.join(f'`{r}`' for r in v.evidence_refs)}")
    lines += ["", "## Counterexamples", ""]
    lines += [f"- `{c}`" for c in spec.counterexample_ids] or ["- none known"]
    lines += ["", "## Excluded (single-lineage) steps", ""]
    lines += [
        f"- {e.action}: {e.reason} ({', '.join(f'`{x}`' for x in e.support)})"
        for e in spec.excluded_steps
    ] or ["- none"]
    lines += ["", "## Independent lineages", ""]
    groups: dict[str, list[str]] = {}
    for exp, lin in sorted(lineage.items()):
        groups.setdefault(lin, []).append(exp)
    for lin, exps in sorted(groups.items()):
        lines.append(f"- `{lin}`: {', '.join(f'`{e}`' for e in exps)}")
    return "\n".join(lines) + "\n"


def render_package(
    spec: ProcedureSpec,
    *,
    version: str,
    wording: dict[str, str] | None = None,
    description: str | None = None,
    suggestions: list[str] | None = None,
    lineage: dict[str, str],
) -> dict[str, bytes]:
    words = {s.step_id: s.action for s in spec.steps}
    words.update({s.step_id: s.action for b in spec.branches for s in b.steps})
    words.update(wording or {})
    members = set(spec.source_experience_ids)
    ctx: dict[str, Any] = {
        "spec": spec,
        "title": spec.title,
        "description": build_description(spec, description),
        "compatibility": build_compatibility(spec),
        "version": version,
        "spec_hash": spec_hash(spec),
        "wording": words,
        "suggestions": suggestions or [],
        "n_units": len({lin for e, lin in lineage.items() if e in members}),
        "n_members": len(spec.source_experience_ids),
    }
    skill_md = _env().get_template("SKILL.md.j2").render(**ctx)
    triggers = {
        "positive": [spec.trigger],
        "negative": [c.text for c in spec.contraindications],
        "note": "Derived from training evidence only. Held-out evaluation cases live elsewhere.",
    }
    return {
        "SKILL.md": skill_md.encode(),
        "references/evidence.md": evidence_markdown(spec, lineage).encode(),
        "references/procedure-spec.json": (
            json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode(),
        "tests/fixtures/trigger-cases.json": (json.dumps(triggers, indent=2) + "\n").encode(),
    }
