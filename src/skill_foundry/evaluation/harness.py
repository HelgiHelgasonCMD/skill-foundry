"""Paired baseline-vs-candidate capability evaluation.

For every (case, repeat): both arms start from byte-identical copies of the same
task snapshot, run under identical limits with the same runner configuration,
in a randomised (seeded, recorded) order, and are judged only by independent
verifiers with hidden assets. Workspaces are deleted afterwards; traces are kept.
"""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from skill_foundry import __version__
from skill_foundry.domain.models import CapabilityStatus
from skill_foundry.evaluation.runners import AgentRunner, AgentRunResult, Arm, RunTask
from skill_foundry.evaluation.suite import EvalCase, EvalSuite
from skill_foundry.evaluation.verifiers import (
    VERIFIER_VERSION,
    VerifierResult,
    VerifyContext,
    make_verifier,
)
from skill_foundry.sandbox.base import Sandbox
from skill_foundry.util import package_hash, sha256_bytes, tree_hashes, utcnow

MIN_PAIRS_FOR_CLAIM = 10


@dataclass
class ArmResult:
    case_id: str
    repeat: int
    arm: Arm
    order: int
    success: bool
    critical_failed: list[str]
    verifiers: list[dict[str, Any]]
    run: dict[str, Any]
    failure_category: str | None


@dataclass
class PairResult:
    case_id: str
    repeat: int
    baseline: ArmResult
    candidate: ArmResult

    @property
    def regression(self) -> list[str]:
        """Critical verifiers that pass without the skill but fail with it."""
        base_ok = {v["verifier_id"] for v in self.baseline.verifiers if v["passed"]}
        return [
            v["verifier_id"]
            for v in self.candidate.verifiers
            if v["critical"] and not v["passed"] and v["verifier_id"] in base_ok
        ]


@dataclass
class CapabilityReport:
    runner_kind: str
    status: CapabilityStatus
    pairs: list[PairResult]
    summary: dict[str, Any]
    notes: list[str]
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_kind": self.runner_kind,
            "status": self.status.value,
            "summary": self.summary,
            "notes": self.notes,
            "manifest": self.manifest,
            "pairs": [
                {
                    "case_id": p.case_id,
                    "repeat": p.repeat,
                    "baseline": asdict(p.baseline),
                    "candidate": asdict(p.candidate),
                    "regression": p.regression,
                }
                for p in self.pairs
            ],
        }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _category(run: AgentRunResult, verifiers: list[VerifierResult]) -> str | None:
    if run.status != "completed":
        return run.status
    if any(not v.passed and v.critical for v in verifiers):
        return "verifier_failed"
    return None


def run_arm(
    case: EvalCase,
    arm: Arm,
    repeat: int,
    order: int,
    runner: AgentRunner,
    sandbox: Sandbox,
    run_root: Path,
    skill_name: str,
    skill_files: dict[str, bytes],
    trigger: Literal["auto", "explicit"],
) -> ArmResult:
    run_dir = run_root / case.case_id / f"r{repeat}" / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    shutil.copytree(case.workspace, workspace, symlinks=False)
    try:
        task = RunTask(
            case=case,
            arm=arm,
            workspace=workspace,
            run_dir=run_dir,
            skill_name=skill_name,
            skill_files=skill_files if arm == "candidate" else None,
            trigger=trigger,
        )
        run = runner.run(task)
        ctx = VerifyContext(
            workspace=workspace,
            snapshot=case.workspace,
            hidden=case.hidden,
            sandbox=sandbox,
            timeout_s=float(case.limits["timeout_s"]),
        )
        results = [make_verifier(v).verify(ctx) for v in case.verifiers]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)  # deterministic cleanup; traces kept
        shutil.rmtree(run_dir / "skillroot", ignore_errors=True)
        shutil.rmtree(run_dir / "claude-config", ignore_errors=True)
    crit = [r.verifier_id for r in results if r.critical and not r.passed]
    return ArmResult(
        case_id=case.case_id,
        repeat=repeat,
        arm=arm,
        order=order,
        success=not crit and run.status == "completed",
        critical_failed=crit,
        verifiers=[r.to_dict() for r in results],
        run=run.to_dict(),
        failure_category=_category(run, results),
    )


def run_capability(
    suite: EvalSuite,
    track: str,
    *,
    runner: AgentRunner,
    sandbox: Sandbox,
    run_root: Path,
    skill_name: str,
    skill_files: dict[str, bytes],
    repeats: int = 1,
    seed: int = 0,
    trigger: Literal["auto", "explicit"] = "auto",
) -> CapabilityReport:
    cases, _ = suite.track(track)
    rng = random.Random(seed)
    pairs: list[PairResult] = []
    snapshots = {c.case_id: package_hash(tree_hashes(c.workspace)) for c in cases}
    for case in cases:
        for r in range(repeats):
            arms: list[Arm] = ["baseline", "candidate"]
            rng.shuffle(arms)
            res = {
                arm: run_arm(
                    case, arm, r, i, runner, sandbox, run_root, skill_name, skill_files, trigger
                )
                for i, arm in enumerate(arms)
            }
            pairs.append(PairResult(case.case_id, r, res["baseline"], res["candidate"]))
    report = summarise(pairs, runner.kind)
    report.manifest = {
        "skill_foundry_version": __version__,
        "runner": runner.describe(),
        "sandbox": sandbox.name,
        "suite": suite.manifest(),
        "track": track,
        "task_snapshots": snapshots,
        "candidate_package_sha256": package_hash(
            {k: sha256_bytes(v) for k, v in skill_files.items()}
        ),
        "verifier_version": VERIFIER_VERSION,
        "limits": {c.case_id: c.limits for c in cases},
        "repeats": repeats,
        "order_seed": seed,
        "trigger": trigger,
        "created_at": utcnow().isoformat(),
    }
    return report


def summarise(pairs: list[PairResult], runner_kind: str) -> CapabilityReport:
    n = len(pairs)
    kb = sum(p.baseline.success for p in pairs)
    kc = sum(p.candidate.success for p in pairs)
    b = sum(p.baseline.success and not p.candidate.success for p in pairs)
    c = sum(p.candidate.success and not p.baseline.success for p in pairs)
    regressions = [(p.case_id, p.repeat, p.regression) for p in pairs if p.regression]

    def mean(arm: str, key: str) -> float | None:
        vals = [getattr(p, arm).run.get(key) for p in pairs]
        nums = [v for v in vals if isinstance(v, int | float)]
        return round(sum(nums) / len(nums), 3) if nums and len(nums) == len(vals) else None

    def failures(arm: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in pairs:
            cat = getattr(p, arm).failure_category
            if cat:
                out[cat] = out.get(cat, 0) + 1
        return out

    def skill_obs(key: str) -> dict[str, int]:
        vals = [p.candidate.run["skill"].get(key) for p in pairs]
        return {"true": vals.count(True), "false": vals.count(False), "unknown": vals.count(None)}

    summary: dict[str, Any] = {
        "pairs": n,
        "baseline": {
            "successes": kb,
            "rate": round(kb / n, 3) if n else None,
            "wilson95": wilson(kb, n),
            "mean_tool_calls": mean("baseline", "tool_calls"),
            "mean_failed_actions": mean("baseline", "failed_actions"),
            "mean_duration_s": mean("baseline", "duration_s"),
            "mean_cost_usd": mean("baseline", "cost_usd"),
            "failures": failures("baseline"),
        },
        "candidate": {
            "successes": kc,
            "rate": round(kc / n, 3) if n else None,
            "wilson95": wilson(kc, n),
            "mean_tool_calls": mean("candidate", "tool_calls"),
            "mean_failed_actions": mean("candidate", "failed_actions"),
            "mean_duration_s": mean("candidate", "duration_s"),
            "mean_cost_usd": mean("candidate", "cost_usd"),
            "failures": failures("candidate"),
        },
        "discordant": {"baseline_only": b, "candidate_only": c},
        "critical_regressions": regressions,
        "skill_observed": {k: skill_obs(k) for k in ("discoverable", "invoked", "used")},
    }
    notes = []
    if n < MIN_PAIRS_FOR_CLAIM:
        summary["p_value"] = None
        notes.append(
            f"Only {n} paired run(s); at least {MIN_PAIRS_FOR_CLAIM} are required before "
            "any significance is computed. Treat differences as anecdotal."
        )
    else:
        summary["p_value"] = round(exact_mcnemar(b, c), 4)
    if runner_kind != "live":
        notes.append(
            f"runner_kind={runner_kind}: this is a SMOKE TEST of the harness, sandbox "
            "and verifiers. It is NOT evidence that the skill improves an agent."
        )
        status = CapabilityStatus.SMOKE_ONLY
    elif regressions:
        status = CapabilityStatus.REGRESSION
    elif n < MIN_PAIRS_FOR_CLAIM:
        status = CapabilityStatus.INCONCLUSIVE
    elif c > b and summary["p_value"] is not None and summary["p_value"] < 0.05:
        status = CapabilityStatus.MEASURED_BENEFIT
    elif c <= b:
        status = CapabilityStatus.NO_BENEFIT
    else:
        status = CapabilityStatus.INCONCLUSIVE
    if runner_kind == "live" and summary["skill_observed"]["invoked"]["true"] == 0:
        notes.append(
            "The candidate skill was never observed being invoked; any difference "
            "cannot be attributed to the skill."
        )
    return CapabilityReport(
        runner_kind=runner_kind, status=status, pairs=pairs, summary=summary, notes=notes
    )
