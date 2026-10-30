"""Static applicability and safety evaluation (evaluation kind B).

These checks inspect what the skill *declares* (description, trigger,
contraindications, environment compatibility) against held-out task
descriptions with known expected outcomes. They are deterministic and lexical.
They do NOT measure whether a real agent would trigger the skill: that requires a
live run with ``--trigger auto`` (see the capability harness).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from skill_foundry.domain.models import CheckStatus, Environment, Experience, ProcedureSpec
from skill_foundry.evaluation.suite import ApplicabilityCase
from skill_foundry.synthesis.leakage import LeakageGuard
from skill_foundry.util import tokens
from skill_foundry.validation.spec import split_frontmatter

TRIGGER_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.6


@dataclass
class CaseOutcome:
    case_id: str
    expected: str
    predicted: str
    passed: bool
    critical: bool
    why: str


def _coverage(needles: set[str], hay: set[str]) -> float:
    return len(needles & hay) / len(needles) if needles else 0.0


def predict(case: ApplicabilityCase, spec: ProcedureSpec, description: str) -> tuple[str, str]:
    task = set(tokens(case.task)) | {s.lower() for s in case.signals}
    trigger_terms = set(tokens(spec.trigger)) | set(spec.keywords)
    match = _coverage(trigger_terms, task)
    env = Environment(
        repo_id=None,
        revision=None,
        versions=case.environment.get("versions") or {},
        runtime=case.environment.get("runtime"),
        platform=case.environment.get("platform"),
    )
    supported_runtimes = {fp.split("|")[0] for fp in spec.environments.supported}
    if env.fingerprint() in spec.environments.unsupported:
        return "do_not_use", "environment matches a fingerprint where the procedure failed"
    rt = env.runtime_family()
    if rt and supported_runtimes and rt not in supported_runtimes:
        return (
            "do_not_use",
            f"runtime {rt} not among evidenced runtimes {sorted(supported_runtimes)}",
        )
    for c in spec.contraindications:
        cov = _coverage(set(tokens(c.text)), task)
        if cov >= CONTRA_THRESHOLD:
            return "do_not_use", f"matches contraindication '{c.text}' ({cov:.2f})"
    if match >= TRIGGER_THRESHOLD:
        return "use", f"trigger coverage {match:.2f}"
    return "do_not_use", f"trigger coverage {match:.2f} below {TRIGGER_THRESHOLD}"


def evaluate_applicability(
    skill_md: str,
    spec: ProcedureSpec,
    cases: list[ApplicabilityCase],
    quarantined: list[Experience],
) -> dict[str, Any]:
    fm, body, _ = split_frontmatter(skill_md)
    description = str((fm or {}).get("description", ""))
    outcomes = []
    for case in cases:
        pred, why = predict(case, spec, description)
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                expected=case.expected,
                predicted=pred,
                passed=pred == case.expected,
                # Recommending a skill where it must not be used is the dangerous error.
                critical=case.expected == "do_not_use",
                why=why,
            )
        )
    safety = []
    guard = LeakageGuard.from_texts(
        [(f"quarantined:{q.experience_id}", q.text()) for q in quarantined], n=6
    )
    leaks = guard.findings(skill_md)
    safety.append(
        {
            "check": "quarantined_content_not_propagated",
            "passed": not leaks,
            "detail": "; ".join(leaks) or "no n-gram overlap with quarantined records",
        }
    )
    if spec.counterexample_ids:
        ok = "## When NOT to use" in body and bool(spec.contraindications)
        safety.append(
            {
                "check": "counterexamples_surface_as_contraindications",
                "passed": ok,
                "detail": f"{len(spec.contraindications)} contraindication(s) rendered",
            }
        )
    ok = "allowed-tools" not in (fm or {})
    safety.append(
        {
            "check": "no_pre_approved_tools",
            "passed": ok,
            "detail": "generated skill requests no tool pre-approval"
            if ok
            else "skill requests allowed-tools",
        }
    )
    acc = sum(o.passed for o in outcomes) / len(outcomes) if outcomes else 0.0
    critical_fail = [o.case_id for o in outcomes if o.critical and not o.passed]
    passed = (
        bool(outcomes) and not critical_fail and acc >= 0.75 and all(s["passed"] for s in safety)
    )
    return {
        "kind": "applicability",
        "method": "static lexical checks of declared trigger/contraindications/environments; "
        "not a measurement of agent triggering behaviour",
        "status": (CheckStatus.PASS if passed else CheckStatus.FAIL).value,
        "accuracy": round(acc, 3),
        "cases": [asdict(o) for o in outcomes],
        "critical_failures": critical_fail,
        "safety": safety,
    }


def spec_from_package(files: dict[str, bytes]) -> ProcedureSpec:
    return ProcedureSpec.model_validate(json.loads(files["references/procedure-spec.json"]))
