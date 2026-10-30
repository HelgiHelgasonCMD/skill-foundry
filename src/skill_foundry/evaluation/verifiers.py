"""Objective task verifiers. The agent's own claim of success is never consulted.

Verifiers that execute code (pytest, command) run inside the sandbox on a *copy*
of the post-run workspace, with hidden assets mounted read-only.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from skill_foundry.sandbox.base import Sandbox, SandboxPolicy

VERIFIER_VERSION = "verifiers.v1"


@dataclass
class VerifierResult:
    verifier_id: str
    type: str
    passed: bool
    critical: bool
    detail: str
    tests_passed: int | None = None
    tests_failed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyContext:
    workspace: Path  # post-run workspace (host)
    snapshot: Path  # original, untouched case workspace
    hidden: Path | None
    sandbox: Sandbox
    timeout_s: float = 120.0


class Verifier(Protocol):
    verifier_id: str
    critical: bool

    def verify(self, ctx: VerifyContext) -> VerifierResult: ...


class PytestVerifier:
    def __init__(self, spec: dict[str, Any]):
        self.verifier_id = spec["id"]
        self.critical = bool(spec.get("critical", True))
        self.tests = spec.get("tests", "hidden:tests")
        self.args = list(spec.get("args", ["-q"]))

    def verify(self, ctx: VerifyContext) -> VerifierResult:
        with tempfile.TemporaryDirectory(prefix="sf-verify-") as tmp:
            work = Path(tmp) / "work"
            shutil.copytree(ctx.workspace, work, symlinks=False)
            ro: dict[str, Path] = {}
            if self.tests.startswith("hidden:"):
                if ctx.hidden is None:
                    return VerifierResult(
                        self.verifier_id,
                        "pytest",
                        False,
                        self.critical,
                        "case declares hidden tests but has no hidden dir",
                    )
                ro["hidden"] = ctx.hidden
            policy = SandboxPolicy(mounts_rw={"work": work}, mounts_ro=ro, timeout_s=ctx.timeout_s)
            work_mp = ctx.sandbox.mount_point("work", policy)
            policy.env["PYTHONPATH"] = work_mp
            if self.tests.startswith("hidden:"):
                target = (
                    f"{ctx.sandbox.mount_point('hidden', policy)}/{self.tests.split(':', 1)[1]}"
                )
            else:
                target = self.tests
            junit = f"{work_mp}/.sf-junit.xml"
            argv = [
                "python",
                "-m",
                "pytest",
                *self.args,
                "-p",
                "no:cacheprovider",
                f"--rootdir={work_mp}",
                f"--junitxml={junit}",
                target,
            ]
            res = ctx.sandbox.run(argv, policy)
            passed_n, failed_n = _junit_counts(work / ".sf-junit.xml")
            ok = res.ok and failed_n == 0 and (passed_n or 0) > 0
            tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
            detail = (
                f"exit={res.exit_code}{' TIMEOUT' if res.timed_out else ''} "
                f"passed={passed_n} failed={failed_n}: {' | '.join(tail)}"
            )
            return VerifierResult(
                self.verifier_id, "pytest", ok, self.critical, detail[:500], passed_n, failed_n
            )


def _junit_counts(path: Path) -> tuple[int | None, int]:
    if not path.exists():
        return None, 0
    root = ET.parse(path).getroot()  # noqa: S314 - file produced by pytest in our sandbox
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return tests - bad - skipped, bad + skipped  # a skipped hidden test counts as failure


class FilesVerifier:
    def __init__(self, spec: dict[str, Any]):
        self.verifier_id = spec["id"]
        self.critical = bool(spec.get("critical", True))
        self.must_exist = list(spec.get("must_exist", []))
        self.must_not_exist = list(spec.get("must_not_exist", []))
        self.unchanged = list(spec.get("unchanged", []))

    def verify(self, ctx: VerifyContext) -> VerifierResult:
        problems = []
        for rel in self.must_exist:
            if not (ctx.workspace / rel).exists():
                problems.append(f"missing {rel}")
        for rel in self.must_not_exist:
            if (ctx.workspace / rel).exists():
                problems.append(f"unexpected {rel}")
        for rel in self.unchanged:
            a, b = ctx.workspace / rel, ctx.snapshot / rel
            if not a.exists() or a.read_bytes() != b.read_bytes():
                problems.append(f"{rel} was modified (must stay unchanged)")
        return VerifierResult(
            self.verifier_id,
            "files",
            not problems,
            self.critical,
            "; ".join(problems) or "all file constraints satisfied",
        )


_PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


class YamlConstraintVerifier:
    RULES = ("permissions_declared", "actions_pinned", "no_pull_request_target")

    def __init__(self, spec: dict[str, Any]):
        self.verifier_id = spec["id"]
        self.critical = bool(spec.get("critical", True))
        self.file = spec["file"]
        self.rules = list(spec.get("rules", self.RULES))

    def verify(self, ctx: VerifyContext) -> VerifierResult:
        path = ctx.workspace / self.file
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            return VerifierResult(
                self.verifier_id,
                "yaml_constraints",
                False,
                self.critical,
                f"cannot parse {self.file}: {exc}",
            )
        if not isinstance(doc, dict):
            return VerifierResult(
                self.verifier_id,
                "yaml_constraints",
                False,
                self.critical,
                "workflow is not a mapping",
            )
        problems = []
        triggers = doc.get("on", doc.get(True))  # YAML 1.1 parses bare `on` as True
        if "permissions_declared" in self.rules:
            perms = doc.get("permissions")
            if perms is None or perms == "write-all":
                problems.append("no least-privilege top-level permissions block")
        if "actions_pinned" in self.rules:
            for job_name, job in (doc.get("jobs") or {}).items():
                for step in (job or {}).get("steps") or []:
                    uses = (step or {}).get("uses")
                    if not uses or uses.startswith("./") or uses.startswith("docker://"):
                        continue
                    if not _PINNED.match(str(uses)):
                        problems.append(f"job {job_name}: '{uses}' is not pinned to a commit SHA")
        if "no_pull_request_target" in self.rules:
            names = (
                triggers
                if isinstance(triggers, list)
                else list(triggers or {})
                if isinstance(triggers, dict)
                else [triggers]
            )
            if "pull_request_target" in names:
                problems.append("uses pull_request_target")
        return VerifierResult(
            self.verifier_id,
            "yaml_constraints",
            not problems,
            self.critical,
            "; ".join(problems) or f"{self.file}: all rules satisfied",
        )


class CommandVerifier:
    def __init__(self, spec: dict[str, Any]):
        self.verifier_id = spec["id"]
        self.critical = bool(spec.get("critical", True))
        self.argv = list(spec["argv"])
        self.expect_exit = int(spec.get("expect_exit", 0))
        self.contains = spec.get("expect_stdout_contains")

    def verify(self, ctx: VerifyContext) -> VerifierResult:
        with tempfile.TemporaryDirectory(prefix="sf-verify-") as tmp:
            work = Path(tmp) / "work"
            shutil.copytree(ctx.workspace, work)
            res = ctx.sandbox.run(
                self.argv, SandboxPolicy(mounts_rw={"work": work}, timeout_s=ctx.timeout_s)
            )
        ok = res.exit_code == self.expect_exit and not res.timed_out
        if ok and self.contains:
            ok = self.contains in res.stdout
        return VerifierResult(
            self.verifier_id,
            "command",
            ok,
            self.critical,
            f"exit={res.exit_code} (expected {self.expect_exit})",
        )


def make_verifier(spec: dict[str, Any]) -> Verifier:
    kind = spec["type"]
    if kind == "pytest":
        return PytestVerifier(spec)
    if kind == "files":
        return FilesVerifier(spec)
    if kind == "yaml_constraints":
        return YamlConstraintVerifier(spec)
    if kind == "command":
        return CommandVerifier(spec)
    raise ValueError(f"unknown verifier type {kind!r}")
