"""Deterministic ProcedureSpec inference from a cluster's evidence.

Every element of a ProcedureSpec carries its support (experience IDs). Support is
counted in independent task lineages, never raw episodes. Nothing is invented:

* invariant steps  - completed in >= ``invariant_ratio`` of lineages;
* decision branches - completed in >= 2 lineages but fewer than the invariant
                      ratio; the condition is derived from observations or the
                      environment, or is explicitly marked unresolved;
* single-lineage steps are excluded (listed with the reason);
* verification     - only from acceptance-verified evidence;
* recovery         - only where a failed attempt was followed by a completed one;
* contraindications - from verified counterexamples and producer lessons
                      (the latter marked ``reported``).

Lesson text from producers is ``reported``: it never becomes a step unless a
completed attempt with the same signature was observed.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from skill_foundry.domain.models import (
    DecisionBranch,
    EnvironmentCompatibility,
    ExcludedStep,
    Experience,
    ProcedureSpec,
    ProcedureStep,
    RecoveryAction,
    SupportedStatement,
    TaskCluster,
    TrustLevel,
    VerificationRequirement,
)
from skill_foundry.mining.features import (
    _BACKTICK,
    extract_commands,
    normalize_command,
    step_signature,
)
from skill_foundry.util import jaccard, sha256_json, tokens

SPEC_INFERENCE_VERSION = "procedure-infer.v1"


@dataclass(frozen=True)
class InferenceConfig:
    invariant_ratio: float = 0.75
    min_branch_units: int = 2
    min_reported_units: int = 2
    max_keywords: int = 8

    def hash(self) -> str:
        return sha256_json({**asdict(self), "version": SPEC_INFERENCE_VERSION})


def generic_action(text: str) -> str:
    """Strip episode-specific detail (after ':') and parameterise inline commands."""
    head = (
        text.split(":", 1)[0].strip()
        if ":" in text and not text.strip().startswith("`")
        else text.strip()
    )
    # A colon inside backticks is not a detail separator.
    if head.count("`") % 2:
        head = text.strip()

    def sub(m: re.Match[str]) -> str:
        span = m.group(1)
        if not extract_commands(f"`{span}`"):
            return m.group(0)
        return f"`{normalize_command(span, keep_placeholders=True)}`"

    return imperative(_BACKTICK.sub(sub, head).rstrip("."))


_IMPERATIVE = {
    "reproduced": "Reproduce",
    "re-ran": "Re-run",
    "reran": "Re-run",
    "ran": "Run",
    "edited": "Edit",
    "inspected": "Inspect",
    "fixed": "Fix",
    "compared": "Compare",
    "validated": "Validate",
    "added": "Add",
    "pinned": "Pin",
    "replaced": "Replace",
    "located": "Locate",
    "restored": "Restore",
    "updated": "Update",
    "changed": "Change",
    "increased": "Increase",
    "rewrote": "Rewrite",
    "checked": "Check",
    "verified": "Verify",
    "removed": "Remove",
    "reverted": "Revert",
    "installed": "Install",
    "configured": "Configure",
    "read": "Read",
    "diffed": "Diff",
    "bisected": "Bisect",
    "isolated": "Isolate",
    "used": "Use",
}


def imperative(action: str) -> str:
    """'Reproduced the failure ...' -> 'Reproduce the failure ...' (known verbs only)."""
    first, _, rest = action.partition(" ")
    verb = _IMPERATIVE.get(first.lower())
    return f"{verb} {rest}".strip() if verb else action


def slugify(text: str, limit: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:limit].strip("-")
    return slug or "skill"


def _common_head(members: list[Experience]) -> str:
    heads = [m.task_summary.split(":", 1)[0].strip() for m in members]
    toks = [h.split() for h in heads]
    prefix: list[str] = []
    for col in zip(*toks, strict=False):
        if len(set(w.lower() for w in col)) == 1:
            prefix.append(col[0])
        else:
            break
    if len(prefix) >= 2:
        return " ".join(prefix)
    return Counter(heads).most_common(1)[0][0]


class _StepStats:
    def __init__(self) -> None:
        self.units: set[str] = set()
        self.exps: set[str] = set()
        self.positions: list[float] = []
        self.texts: list[tuple[str, str]] = []  # (experience_id, action)


def infer_spec(
    cluster: TaskCluster,
    members: list[Experience],
    counterexamples: list[Experience],
    cfg: InferenceConfig | None = None,
    name: str | None = None,
) -> ProcedureSpec:
    cfg = cfg or InferenceConfig()
    unit_of = {e: u.unit_id for u in cluster.units for e in u.experience_ids}
    n_units = len(cluster.units) or 1
    by_id = {m.experience_id: m for m in members}

    stats: dict[str, _StepStats] = defaultdict(_StepStats)
    for m in members:
        completed = [a for a in m.attempts if a.status == "completed"]
        seen: Counter[str] = Counter()
        for pos, a in enumerate(completed):
            base = step_signature(a.action_summary)
            seen[base] += 1
            sig = base if seen[base] == 1 else f"{base} #{seen[base]}"
            st = stats[sig]
            st.units.add(unit_of[m.experience_id])
            st.exps.add(m.experience_id)
            st.positions.append(pos / max(len(completed) - 1, 1))
            st.texts.append((m.experience_id, a.action_summary))

    def make_step(sig: str, st: _StepStats, idx: int, prefix: str = "s") -> ProcedureStep:
        text = sorted(st.texts)[0][1]
        action = generic_action(text)
        cmds = sorted({normalize_command(c) for _, t in st.texts for c in extract_commands(t)})
        ratio = len(st.units) / n_units
        return ProcedureStep(
            step_id=f"{prefix}{idx}",
            action=action,
            commands=cmds,
            rationale=f"completed in {len(st.units)}/{n_units} independent lineages "
            f"({len(st.exps)} episodes)",
            support=sorted(st.exps),
            support_units=len(st.units),
            support_ratio=round(ratio, 3),
        )

    ordered = sorted(stats.items(), key=lambda kv: (median(kv[1].positions), kv[0]))
    steps: list[ProcedureStep] = []
    branch_sigs: list[tuple[str, _StepStats]] = []
    excluded: list[ExcludedStep] = []
    for sig, st in ordered:
        ratio = len(st.units) / n_units
        if ratio >= cfg.invariant_ratio:
            steps.append(make_step(sig, st, len(steps) + 1))
        elif len(st.units) >= cfg.min_branch_units:
            branch_sigs.append((sig, st))
        else:
            excluded.append(
                ExcludedStep(
                    action=generic_action(sorted(st.texts)[0][1]),
                    reason=f"observed in only {len(st.units)} lineage(s); treated as task-specific",
                    support=sorted(st.exps),
                )
            )

    branches: list[DecisionBranch] = []
    for i, (sig, st) in enumerate(branch_sigs, start=1):
        cond, basis = _branch_condition(st.exps, members)
        branches.append(
            DecisionBranch(
                branch_id=f"b{i}",
                condition=cond,
                condition_basis=basis,
                steps=[make_step(sig, st, 1, prefix=f"b{i}.")],
                support=sorted(st.exps),
                needs_review=basis == "unresolved",
            )
        )

    verification = _verification(members)
    recovery = _recovery(members)
    failed = _failed_approaches(members, cfg, unit_of)
    contra, unsupported = _contraindications(members, counterexamples, cfg, unit_of)
    pre = _reported(members, "preconditions", cfg, unit_of)
    supported_envs = sorted({u.env_fingerprint for u in cluster.units})
    head = _common_head(members)
    signals = Counter(o.error_code for m in members for o in m.observations if o.error_code)
    common_signals = [
        s for s, c in signals.most_common() if c >= max(2, round(len(members) * 0.75))
    ]
    keywords = _keywords(members, cfg)

    assumptions = [
        f"Evidence covers {len(cluster.units)} independent lineage(s) across "
        f"{len({u.repo_id for u in cluster.units})} repositories; environments outside "
        f"{', '.join(supported_envs)} are untested.",
    ]
    for b in branches:
        if b.condition_basis == "unresolved":
            assumptions.append(
                f"Branch {b.branch_id}: the condition for "
                f"'{b.steps[0].action}' could not be derived from evidence."
            )
    prose = sum(1 for m in members for lesson in m.lessons if lesson.successful_approach)
    if prose:
        assumptions.append(
            f"{prose} producer lesson(s) describe the approach in prose. Lesson text is treated "
            "as reported: it informs preconditions and contraindications, never steps."
        )
    if cluster.exploratory:
        assumptions.append("EXPLORATORY: generated below the configured evidence thresholds.")

    spec_name = name or slugify(head)
    return ProcedureSpec(
        name=spec_name,
        title=head[0].upper() + head[1:],
        trigger=f"{head}"
        + (f" (typical signals: {', '.join(common_signals)})" if common_signals else ""),
        in_scope=[
            f"Tasks of the form: {head}",
            f"Environments resembling: {', '.join(supported_envs)}",
        ],
        out_of_scope=[c.text for c in contra],
        preconditions=pre,
        steps=steps,
        branches=branches,
        verification=verification,
        recovery=recovery,
        failed_approaches=failed,
        contraindications=contra,
        environments=EnvironmentCompatibility(
            supported=supported_envs,
            unsupported=unsupported,
            notes=["Fingerprint = runtime family | platform family | tool major.minor versions."],
        ),
        assumptions=assumptions,
        excluded_steps=excluded,
        cluster_id=cluster.cluster_id,
        workspace=cluster.workspace,
        source_experience_ids=sorted(by_id),
        counterexample_ids=sorted(c.experience_id for c in counterexamples),
        evidence_score=cluster.evidence_score,
        exploratory=cluster.exploratory,
        inference_config_hash=cfg.hash(),
        keywords=keywords,
    )


def spec_hash(spec: ProcedureSpec) -> str:
    return sha256_json(spec.model_dump(mode="json"))


def _branch_condition(support: set[str], members: list[Experience]) -> tuple[str, str]:
    sup = [m for m in members if m.experience_id in support]
    rest = [m for m in members if m.experience_id not in support]

    def obs_keys(m: Experience) -> set[tuple[str, str | None]]:
        return {(o.kind, o.error_code) for o in m.observations}

    shared = set.intersection(*(obs_keys(m) for m in sup)) if sup else set()
    distinguishing = [k for k in sorted(shared, key=str) if not any(k in obs_keys(m) for m in rest)]
    if distinguishing:
        kind, code = distinguishing[0]
        example = next(
            o.summary for o in sup[0].observations if (o.kind, o.error_code) == (kind, code)
        )
        return (
            f"when you observe {kind}"
            + (f" ({code})" if code else "")
            + f', e.g. "{generic_action(example)}"',
            "observation",
        )
    for key in sorted({k for m in sup for k in m.environment.versions}):
        vals = {m.environment.versions.get(key) for m in sup}
        if (
            len(vals) == 1
            and None not in vals
            and not any(m.environment.versions.get(key) in vals for m in rest)
        ):
            return f"when {key} is {vals.pop()}", "environment"
    return "condition not derivable from evidence (reviewer must decide)", "unresolved"


def _verification(members: list[Experience]) -> list[VerificationRequirement]:
    groups: dict[str, dict[str, Any]] = {}
    for m in members:
        for e in m.evidence:
            if e.trust != TrustLevel.VERIFIED or not e.acceptance:
                continue
            cmds = sorted({normalize_command(c) for c in extract_commands(e.summary)})
            key = " && ".join(cmds) or e.kind
            g = groups.setdefault(
                key, {"cmds": cmds, "criteria": set(), "refs": [], "kind": e.kind}
            )
            g["criteria"].update(e.acceptance.criteria)
            g["refs"].append(f"{m.experience_id}#{e.evidence_id}")
    out = []
    for key, g in sorted(groups.items()):
        out.append(
            VerificationRequirement(
                description=(
                    f"Run `{key}` and require exit code 0"
                    if g["cmds"]
                    else f"Confirm with a passing {g['kind']} check"
                ),
                commands=g["cmds"],
                expected="; ".join(sorted(g["criteria"])),
                evidence_refs=sorted(g["refs"]),
            )
        )
    return out


def _recovery(members: list[Experience]) -> list[RecoveryAction]:
    found: dict[tuple[str, str], RecoveryAction] = {}
    for m in members:
        atts = m.attempts
        for a, b in itertools.pairwise(atts):
            if a.status == "failed" and b.status == "completed" and a.result_summary:
                key = (step_signature(a.action_summary), step_signature(b.action_summary))
                if jaccard(key[0].split(), key[1].split()) < 0.3:
                    continue  # unrelated next step: not a recovery of this failure
                prev = found.get(key)
                support = sorted({*(prev.support if prev else []), m.experience_id})
                found[key] = RecoveryAction(
                    when=f'the attempt "{generic_action(a.action_summary)}" failed with '
                    f"{a.result_summary.split(':')[0]}",
                    action=generic_action(b.action_summary),
                    commands=sorted(
                        {normalize_command(c) for c in extract_commands(b.action_summary)}
                    ),
                    support=support,
                )
    return list(found.values())


def _failed_approaches(
    members: list[Experience], cfg: InferenceConfig, unit_of: dict[str, str]
) -> list[SupportedStatement]:
    out: list[SupportedStatement] = []
    for m in members:
        for a, nxt in zip(m.attempts, [*m.attempts[1:], None], strict=False):
            if a.status != "failed":
                continue
            if (
                nxt
                and jaccard(
                    step_signature(a.action_summary).split(),
                    step_signature(nxt.action_summary).split(),
                )
                >= 0.3
            ):
                continue  # covered as a recovery action
            why = f" ({a.result_summary})" if a.result_summary else ""
            out.append(
                SupportedStatement(
                    text=f"{generic_action(a.action_summary)}{why}",
                    kind="observed",
                    support=[m.experience_id],
                )
            )
    out += _reported(members, "failed_approaches", cfg, unit_of, min_units=1)
    return out


def _reported(
    members: list[Experience],
    field: str,
    cfg: InferenceConfig,
    unit_of: dict[str, str],
    min_units: int | None = None,
) -> list[SupportedStatement]:
    support: dict[str, set[str]] = defaultdict(set)
    for m in members:
        for lesson in m.lessons:
            for text in getattr(lesson, field):
                support[text.strip()].add(m.experience_id)
    need = cfg.min_reported_units if min_units is None else min_units
    out = []
    for text, exps in sorted(support.items()):
        if len({unit_of[e] for e in exps}) >= need:
            out.append(SupportedStatement(text=text, kind="reported", support=sorted(exps)))
    return out


def _contraindications(
    members: list[Experience],
    counterexamples: list[Experience],
    cfg: InferenceConfig,
    unit_of: dict[str, str],
) -> tuple[list[SupportedStatement], list[str]]:
    out: list[SupportedStatement] = []
    member_envs = {m.environment.fingerprint() for m in members}
    unsupported = sorted({c.environment.fingerprint() for c in counterexamples} - member_envs)
    by_text: dict[str, set[str]] = defaultdict(set)
    for c in counterexamples:
        reasons = [
            a.result_summary for a in c.attempts if a.status == "failed" and a.result_summary
        ]
        for r in reasons[:1]:
            by_text[f"Observed failure: {r}"].add(c.experience_id)
        for lesson in c.lessons:
            for t in lesson.contraindications:
                by_text[t].add(c.experience_id)
    for text, exps in sorted(by_text.items()):
        kind = "observed" if text.startswith("Observed failure") else "reported"
        out.append(SupportedStatement(text=text, kind=kind, support=sorted(exps)))
    out += _reported(members, "contraindications", cfg, unit_of)
    return out, unsupported


def _keywords(members: list[Experience], cfg: InferenceConfig) -> list[str]:
    df: Counter[str] = Counter()
    for m in members:
        df.update(
            set(
                tokens(m.task_summary.split(":", 1)[0])
                + [o.error_code.lower() for o in m.observations if o.error_code]
            )
        )
    need = max(2, int(len(members) * 0.75))
    return [t for t, c in df.most_common() if c >= need and len(t) > 2][: cfg.max_keywords]
