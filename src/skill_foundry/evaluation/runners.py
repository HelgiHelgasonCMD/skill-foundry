"""Agent runners.

``runner_kind`` is recorded on every result and controls what a result may claim:

* ``fake`` - ``FakeScriptedRunner`` replays hand-written per-arm action scripts.
  Differences between arms reflect the scripts, not any effect of a skill. It
  exercises the harness, sandbox and verifiers ("smoke test") only.
* ``mock`` - ``ClaudeCodeRunner`` pointed at a fake ``claude`` executable (tests).
  Exercises argument construction, isolation and trace parsing only.
* ``live`` - ``ClaudeCodeRunner`` against the real CLI. The only kind that can
  produce evidence about agent capability.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from skill_foundry.evaluation.suite import EvalCase
from skill_foundry.ingest.sanitize import redact_text
from skill_foundry.sandbox.base import BwrapSandbox, Sandbox, SandboxPolicy
from skill_foundry.util import package_hash, sha256_bytes

RunnerKind = Literal["fake", "mock", "live"]
Arm = Literal["baseline", "candidate"]


class ContaminationError(RuntimeError):
    """The baseline arm could see a skill, or the candidate arm the wrong one."""


@dataclass
class SkillObservation:
    """What we could observe about the skill during a run (None = not observable)."""

    available: bool  # the package was placed where the runner exposes skills
    discoverable: bool | None = None  # listed by the agent at session start
    invoked: bool | None = None  # agent explicitly invoked/loaded the skill
    used: bool | None = None  # agent read skill resources / followed a step after invoking
    evidence: list[str] = field(default_factory=list)


@dataclass
class RunTask:
    case: EvalCase
    arm: Arm
    workspace: Path  # fresh copy, writable
    run_dir: Path  # private per-run directory for traces/config
    skill_name: str | None
    skill_files: dict[str, bytes] | None  # exact candidate package (candidate arm only)
    trigger: Literal["auto", "explicit"] = "auto"


@dataclass
class AgentRunResult:
    status: Literal["completed", "timeout", "budget_exceeded", "max_turns", "crash", "error"]
    tool_calls: int
    failed_actions: int
    duration_s: float
    skill: SkillObservation
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    trace_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRunner(Protocol):
    kind: RunnerKind
    name: str

    def describe(self) -> dict[str, Any]: ...

    def run(self, task: RunTask) -> AgentRunResult: ...


def install_isolated_skill(task: RunTask, root: Path) -> Path | None:
    """Place the exact candidate package under ``root/.claude/skills/<name>``.

    Baseline: returns None and asserts no skill is visible in the workspace.
    """
    if (task.workspace / ".claude" / "skills").exists():
        raise ContaminationError(f"case workspace for {task.case.case_id} contains .claude/skills")
    if task.arm == "baseline":
        if task.skill_files:
            raise ContaminationError("baseline arm was given skill files")
        return None
    if not task.skill_files or not task.skill_name:
        raise ContaminationError("candidate arm has no skill package")
    dest = root / ".claude" / "skills" / task.skill_name
    dest.mkdir(parents=True)
    hashes = {}
    for rel, data in sorted(task.skill_files.items()):
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        hashes[rel] = sha256_bytes(p.read_bytes())
    expected = package_hash({k: sha256_bytes(v) for k, v in task.skill_files.items()})
    if package_hash(hashes) != expected:
        raise ContaminationError("installed candidate does not match the package hash")
    return root


class FakeScriptedRunner:
    kind: RunnerKind = "fake"
    name = "fake-scripted"

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def describe(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "kind": self.kind,
            "sandbox": self.sandbox.name,
            "note": "replays per-arm scripts; not an agent; no capability evidence",
        }

    def run(self, task: RunTask) -> AgentRunResult:
        install_isolated_skill(task, task.run_dir / "skillroot")
        script_path = task.case.fake_scripts.get(task.arm)
        if script_path is None:
            raise ValueError(f"case {task.case.case_id} has no fake script for arm {task.arm}")
        actions = json.loads(script_path.read_text(encoding="utf-8"))["actions"]
        calls = failed = 0
        elapsed = 0.0
        trace = []
        status: Literal["completed", "timeout"] = "completed"
        for act in actions:
            calls += 1
            op = act["op"]
            if op in ("replace", "write"):
                target = (task.workspace / act["path"]).resolve()
                if not target.is_relative_to(task.workspace.resolve()):
                    raise ValueError("fake script path escapes the workspace")
                if op == "write":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(act["content"], encoding="utf-8")
                    trace.append({"op": op, "path": act["path"]})
                else:
                    text = target.read_text(encoding="utf-8")
                    if act["find"] not in text:
                        failed += 1
                        trace.append({"op": op, "path": act["path"], "error": "find not found"})
                        continue
                    target.write_text(
                        text.replace(act["find"], act["replace"], 1), encoding="utf-8"
                    )
                    trace.append({"op": op, "path": act["path"]})
            elif op == "run":
                res = self.sandbox.run(
                    list(act["argv"]),
                    SandboxPolicy(
                        mounts_rw={"work": task.workspace},
                        timeout_s=float(task.case.limits["timeout_s"]),
                    ),
                )
                elapsed += res.duration_s
                if not res.ok:
                    failed += 1
                if res.timed_out:
                    status = "timeout"
                trace.append({"op": op, "argv": act["argv"], "exit": res.exit_code})
            else:
                raise ValueError(f"unknown fake op {op!r}")
        trace_path = task.run_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=1), encoding="utf-8")
        return AgentRunResult(
            status=status,
            tool_calls=calls,
            failed_actions=failed,
            duration_s=round(elapsed, 3),
            skill=SkillObservation(
                available=task.arm == "candidate",
                evidence=["scripted runner: skill discovery/invocation not applicable"],
            ),
            trace_path=str(trace_path),
            notes=["FAKE: scripted actions; arm differences are authored, not measured"],
        )


DEFAULT_ALLOWED_TOOLS = (
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Skill",
    "Bash(python -m pytest:*)",
    "Bash(git log:*)",
    "Bash(git diff:*)",
    "Bash(git status:*)",
)


class ClaudeCodeRunner:
    """Opt-in adapter for the Claude Code CLI in headless (`-p`) mode.

    Isolation:
      * a private, empty ``CLAUDE_CONFIG_DIR`` per run (no user skills/settings/memory),
      * ``--setting-sources project`` and ``--strict-mcp-config`` (no user settings, no MCP),
      * the candidate package is exposed via ``--add-dir <run>/skillroot`` containing
        ``.claude/skills/<name>``; the baseline gets no skill root,
      * a deny-by-default tool allowlist; never ``--dangerously-skip-permissions``,
      * executed inside bwrap with network enabled (the CLI needs the API). bwrap cannot
        filter by destination, so this boundary is weaker than the offline sandbox.

    Whether the real CLI discovers skills from ``--add-dir`` in this configuration is
    UNVERIFIED until a live run is logged; the trace parser records what was observed.
    """

    name = "claude-code"

    def __init__(
        self,
        *,
        binary: str | None = None,
        model: str | None = None,
        kind: RunnerKind = "live",
        sandbox: Sandbox | None = None,
        allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS,
        bare: bool = False,
        pass_api_key: bool = False,
        pass_oauth_token: bool = False,
        permission_mode: str = "acceptEdits",
    ):
        found = binary or shutil.which("claude")
        if not found:
            raise RuntimeError("claude CLI not found on PATH")
        self.binary = str(Path(found).resolve())
        self.model = model
        self.kind = kind
        self.allowed_tools = allowed_tools
        self.bare = bare
        self.pass_api_key = pass_api_key
        self.pass_oauth_token = pass_oauth_token
        self.permission_mode = permission_mode
        self.sandbox = sandbox or BwrapSandbox(extra_ro=[Path(self.binary).parent])

    def version(self) -> str:
        try:
            out = subprocess.run(
                [self.binary, "--version"], capture_output=True, text=True, timeout=30, check=False
            )
            return out.stdout.strip() or out.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unknown ({exc})"

    def describe(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "kind": self.kind,
            "binary": self.binary,
            "cli_version": self.version(),
            "model": self.model,
            "allowed_tools": list(self.allowed_tools),
            "bare": self.bare,
            "permission_mode": self.permission_mode,
            "sandbox": self.sandbox.name,
            "skill_loading": "--add-dir <skillroot> with .claude/skills/<name> (UNVERIFIED "
            "against the real CLI unless kind=live and observed)",
        }

    def build_argv(self, task: RunTask, skillroot_mp: str | None) -> list[str]:
        prompt = task.case.prompt
        if task.trigger == "explicit" and task.arm == "candidate" and task.skill_name:
            prompt = f"/{task.skill_name} {prompt}"
        argv = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--setting-sources",
            "project",
            "--permission-mode",
            self.permission_mode,
            "--allowedTools",
            *self.allowed_tools,
        ]
        if self.model:
            argv += ["--model", self.model]
        if "max_turns" in task.case.limits:
            argv += ["--max-turns", str(task.case.limits["max_turns"])]
        if "max_budget_usd" in task.case.limits:
            argv += ["--max-budget-usd", str(task.case.limits["max_budget_usd"])]
        if self.bare:
            argv.append("--bare")
        if skillroot_mp:
            argv += ["--add-dir", skillroot_mp]
        return argv

    def run(self, task: RunTask) -> AgentRunResult:
        skillroot = task.run_dir / "skillroot"
        skillroot.mkdir(parents=True, exist_ok=True)
        installed = install_isolated_skill(task, skillroot)
        config = task.run_dir / "claude-config"
        config.mkdir(parents=True, exist_ok=True)
        ro = {"skillroot": skillroot} if installed else {}
        policy = SandboxPolicy(
            mounts_rw={"work": task.workspace, "claude-config": config},
            mounts_ro=ro,
            network=self.kind == "live",
            timeout_s=float(task.case.limits["timeout_s"]),
            memory_mb=4096,
        )
        policy.env["CLAUDE_CONFIG_DIR"] = self.sandbox.mount_point("claude-config", policy)
        policy.env["DISABLE_AUTOUPDATER"] = "1"
        secrets: list[str] = []
        for wanted, var in (
            (self.pass_api_key, "ANTHROPIC_API_KEY"),
            (self.pass_oauth_token, "CLAUDE_CODE_OAUTH_TOKEN"),
        ):
            value = os.environ.get(var)
            if wanted and value:
                policy.env[var] = value
                secrets.append(value)
        mp = self.sandbox.mount_point("skillroot", policy) if installed else None
        argv = self.build_argv(task, mp)
        res = self.sandbox.run(argv, policy)
        trace_path = task.run_dir / "trace.jsonl"
        stdout = res.stdout
        for secret in secrets:  # exact-value scrub: regex detectors may miss a token format
            stdout = stdout.replace(secret, "[REDACTED:credential]")
        trace_path.write_text(redact_text(stdout), encoding="utf-8")
        # Parse with the candidate's name in both arms so baseline contamination is visible.
        parsed = parse_stream(stdout, task.skill_name)
        status = parsed.pop("status")
        if res.timed_out:
            status = "timeout"
        elif res.exit_code != 0 and status == "completed":
            status = "crash"
        skill = SkillObservation(available=bool(installed), **parsed.pop("skill"))
        if not installed:
            skill.evidence.append("baseline: no skill root exposed")
            if skill.invoked or skill.discoverable:
                raise ContaminationError("baseline run could see or invoke the candidate skill")
        return AgentRunResult(
            status=status,
            skill=skill,
            duration_s=res.duration_s,
            trace_path=str(trace_path),
            **parsed,
        )


def parse_stream(stdout: str, skill_name: str | None) -> dict[str, Any]:
    """Parse ``--output-format stream-json`` lines into run metrics + skill observations."""
    tool_calls = failed = 0
    discoverable: bool | None = None
    invoked = used = False
    evidence: list[str] = []
    result: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init" and skill_name:
            listed = [*map(str, ev.get("skills") or []), *map(str, ev.get("slash_commands") or [])]
            discoverable = any(s.split(":")[-1].lstrip("/") == skill_name for s in listed)
            evidence.append(
                f"init lists {len(listed)} skills/commands; "
                f"{'includes' if discoverable else 'does not include'} {skill_name}"
            )
        elif etype == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if block.get("type") != "tool_use":
                    continue
                tool_calls += 1
                name, inp = block.get("name"), block.get("input") or {}
                if skill_name and name == "Skill" and skill_name in json.dumps(inp):
                    invoked = True
                    evidence.append("Skill tool invoked with the candidate")
                elif skill_name and invoked and "skillroot" in json.dumps(inp):
                    used = True
                    evidence.append(f"{name} accessed skill resources after invocation")
        elif etype == "user":
            for block in (ev.get("message") or {}).get("content") or []:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("is_error")
                ):
                    failed += 1
        elif etype == "result":
            result = ev
    sub = str(result.get("subtype", ""))
    status = (
        "completed"
        if sub == "success"
        else "max_turns"
        if "max_turns" in sub
        else "budget_exceeded"
        if "budget" in sub
        else "error"
        if result
        else "crash"
    )
    usage = result.get("usage") or {}
    return {
        "status": status,
        "tool_calls": tool_calls,
        "failed_actions": failed,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": result.get("total_cost_usd"),
        "num_turns": result.get("num_turns"),
        "notes": [],
        "skill": {
            "discoverable": discoverable if skill_name else None,
            "invoked": invoked if skill_name else None,
            "used": used if skill_name else None,
            "evidence": evidence,
        },
    }
