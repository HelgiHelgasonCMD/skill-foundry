"""Verifiers, paired harness, honest statistics and the Claude Code adapter (mocked)."""

from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from conftest import SUITE, TRACK
from skill_foundry.domain.models import CapabilityStatus
from skill_foundry.evaluation.harness import (
    ArmResult,
    PairResult,
    exact_mcnemar,
    summarise,
    wilson,
)
from skill_foundry.evaluation.runners import (
    ClaudeCodeRunner,
    ContaminationError,
    FakeScriptedRunner,
    RunTask,
    parse_stream,
)
from skill_foundry.evaluation.suite import EvalSuite
from skill_foundry.evaluation.verifiers import (
    FilesVerifier,
    VerifyContext,
    YamlConstraintVerifier,
)
from skill_foundry.sandbox.base import BwrapSandbox, UnsafeHostSandbox

SANDBOX = BwrapSandbox() if BwrapSandbox().available() else UnsafeHostSandbox()


def test_yaml_constraints(tmp_path: Path) -> None:
    case = EvalSuite.load(SUITE).cases["gha-release-workflow"]
    v = YamlConstraintVerifier({"id": "y", "file": ".github/workflows/release.yml"})
    r = v.verify(VerifyContext(case.workspace, case.workspace, None, SANDBOX))
    assert not r.passed
    assert (
        "pull_request_target" in r.detail and "not pinned" in r.detail and "permissions" in r.detail
    )


def test_files_verifier_detects_weakened_tests(tmp_path: Path) -> None:
    case = EvalSuite.load(SUITE).cases["pytest-discount-threshold"]
    ws = tmp_path / "ws"
    shutil.copytree(case.workspace, ws)
    v = FilesVerifier({"id": "f", "unchanged": ["tests/test_pricing.py"]})
    assert v.verify(VerifyContext(ws, case.workspace, None, SANDBOX)).passed
    (ws / "tests" / "test_pricing.py").write_text("def test_x():\n    assert True\n")
    assert not v.verify(VerifyContext(ws, case.workspace, None, SANDBOX)).passed


def test_smoke_eval_never_claims_capability(pipeline: dict[str, Any]) -> None:
    f = pipeline["foundry"]
    run_id, rep = f.eval_capability(
        pipeline["version_id"], SUITE, TRACK, runner=FakeScriptedRunner(SANDBOX), sandbox=SANDBOX
    )
    assert rep.status == CapabilityStatus.SMOKE_ONLY
    assert rep.summary["p_value"] is None
    assert any("NOT evidence" in n for n in rep.notes)
    by = {p.case_id: p for p in rep.pairs}
    # the baseline script weakens the visible test: hidden tests + unchanged-file check catch it
    assert set(by["pytest-discount-threshold"].baseline.critical_failed) == {
        "hidden-tests",
        "tests-unchanged",
    }
    assert by["pytest-discount-threshold"].candidate.success
    m = rep.manifest
    assert (
        m["candidate_package_sha256"]
        == f.registry.get_version(pipeline["version_id"]).package_sha256
    )
    assert set(m["task_snapshots"]) == {"pytest-discount-threshold", "pytest-slug-trailing"}
    assert m["runner"]["kind"] == "fake" and "order_seed" in m
    # workspaces are cleaned up; traces kept
    run_root = f.settings.runs_dir / run_id
    assert not list(run_root.rglob("workspace")) and list(run_root.rglob("trace.json"))
    assert (
        f.registry.get_version(pipeline["version_id"]).capability_status
        == CapabilityStatus.SMOKE_ONLY
    )


def _arm(case: str, arm: str, ok: bool, verifiers: list[dict[str, Any]] | None = None) -> ArmResult:
    v = verifiers or [{"verifier_id": "t", "passed": ok, "critical": True}]
    return ArmResult(
        case,
        0,
        arm,
        0,
        ok,
        [] if ok else ["t"],
        v,  # type: ignore[arg-type]
        {
            "tool_calls": 3,
            "failed_actions": 0,
            "duration_s": 1.0,
            "cost_usd": None,
            "skill": {"discoverable": True, "invoked": True, "used": True},
        },
        None if ok else "verifier_failed",
    )


def test_live_statistics_and_regression() -> None:
    small = [PairResult("c", 0, _arm("c", "baseline", False), _arm("c", "candidate", True))]
    assert summarise(small, "live").status == CapabilityStatus.INCONCLUSIVE
    many = [
        PairResult(f"c{i}", 0, _arm("c", "baseline", False), _arm("c", "candidate", True))
        for i in range(12)
    ]
    rep = summarise(many, "live")
    assert rep.status == CapabilityStatus.MEASURED_BENEFIT and rep.summary["p_value"] < 0.05
    reg = [*many, PairResult("r", 0, _arm("r", "baseline", True), _arm("r", "candidate", False))]
    assert summarise(reg, "live").status == CapabilityStatus.REGRESSION
    assert summarise(many, "mock").status == CapabilityStatus.SMOKE_ONLY
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert exact_mcnemar(0, 0) == 1.0 and exact_mcnemar(0, 6) < 0.05


FAKE_CLAUDE = r'''#!/usr/bin/env python3
"""Stand-in for the claude CLI: emits stream-json shaped like the real CLI's output."""
import json, os, sys
argv = sys.argv[1:]
prompt = argv[argv.index("-p") + 1]
skills = []
if "--add-dir" in argv:
    root = argv[argv.index("--add-dir") + 1]
    d = os.path.join(root, ".claude", "skills")
    skills = sorted(os.listdir(d)) if os.path.isdir(d) else []
with open(os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "argv.json"), "w") as fh:
    json.dump({"argv": argv, "env": sorted(os.environ)}, fh)
print(json.dumps({"type": "system", "subtype": "init", "skills": skills,
                  "slash_commands": ["help"] + skills}))
content = [{"type": "tool_use", "name": "Bash", "input": {"command": "python -m pytest -q"}}]
if skills and (prompt.startswith("/") or "AUTO" in os.environ.get("FAKE_MODE", "AUTO")):
    content.insert(0, {"type": "tool_use", "name": "Skill", "input": {"skill": skills[0]}})
    content.append({"type": "tool_use", "name": "Read",
                    "input": {"file_path": root + "/.claude/skills/" + skills[0] + "/SKILL.md"}})
print(json.dumps({"type": "assistant", "message": {"content": content}}))
print(json.dumps({"type": "user", "message": {"content": [
    {"type": "tool_result", "is_error": True, "content": "boom"}]}}))
print(json.dumps({"type": "result", "subtype": "success", "num_turns": 3,
                  "total_cost_usd": 0.0123, "usage": {"input_tokens": 100, "output_tokens": 20}}))
'''


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    d = tmp_path / "fakebin"
    d.mkdir()
    p = d / "claude"
    p.write_text(FAKE_CLAUDE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _task(
    tmp_path: Path, arm: str, files: dict[str, bytes] | None, trigger: str = "auto"
) -> RunTask:
    case = EvalSuite.load(SUITE).cases["pytest-slug-trailing"]
    ws = tmp_path / arm / "ws"
    shutil.copytree(case.workspace, ws)
    (tmp_path / arm / "run").mkdir()
    return RunTask(
        case=case,
        arm=arm,
        workspace=ws,
        run_dir=tmp_path / arm / "run",  # type: ignore[arg-type]
        skill_name="fix-failing-pytest-suite",
        skill_files=files,
        trigger=trigger,
    )  # type: ignore[arg-type]


def test_claude_runner_isolates_arms(
    pipeline: dict[str, Any], tmp_path: Path, fake_claude: Path
) -> None:
    files = pipeline["foundry"].package(pipeline["version_id"])
    runner = ClaudeCodeRunner(
        binary=str(fake_claude),
        kind="mock",
        sandbox=BwrapSandbox(extra_ro=[fake_claude.parent])
        if BwrapSandbox().available()
        else UnsafeHostSandbox(),
    )
    cand = runner.run(_task(tmp_path, "candidate", files))
    assert (
        cand.skill.available and cand.skill.discoverable and cand.skill.invoked and cand.skill.used
    )
    assert cand.cost_usd == 0.0123 and cand.input_tokens == 100 and cand.failed_actions == 1
    rec = json.loads((tmp_path / "candidate" / "run" / "claude-config" / "argv.json").read_text())
    argv = rec["argv"]
    assert "--add-dir" in argv and "--dangerously-skip-permissions" not in argv
    assert "--setting-sources" in argv and "--strict-mcp-config" in argv
    assert "ANTHROPIC_API_KEY" not in rec["env"]  # not passed unless explicitly enabled
    base = runner.run(_task(tmp_path, "baseline", None))
    assert not base.skill.available and base.skill.invoked is False
    rec = json.loads((tmp_path / "baseline" / "run" / "claude-config" / "argv.json").read_text())
    assert "--add-dir" not in rec["argv"]
    explicit = runner.build_argv(
        _task(tmp_path / "x", "candidate", files, "explicit"), "/skillroot"
    )
    assert explicit[explicit.index("-p") + 1].startswith("/fix-failing-pytest-suite ")


OAUTH_CLAUDE = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "9.9.9"; exit 0; fi
printf '{"type":"system","subtype":"init","skills":[],"note":"%s"}\\n' "$CLAUDE_CODE_OAUTH_TOKEN"
printf '{"type":"result","subtype":"success","num_turns":1}\\n'
"""


def test_oauth_token_reaches_the_run_but_never_the_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = "opaque" + "-subscription-" + "credential"  # matches no redaction pattern
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", token)
    binary = tmp_path / "fakebin" / "claude"
    binary.parent.mkdir()
    binary.write_text(OAUTH_CLAUDE)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    sandbox = (
        BwrapSandbox(extra_ro=[binary.parent])
        if BwrapSandbox().available()
        else UnsafeHostSandbox()
    )

    def run(*, passed: bool) -> str:
        arm = tmp_path / ("with" if passed else "without")
        runner = ClaudeCodeRunner(
            binary=str(binary), kind="mock", sandbox=sandbox, pass_oauth_token=passed
        )
        result = runner.run(_task(arm, "baseline", None))
        assert token not in json.dumps(result.__dict__, default=str)
        return (arm / "baseline" / "run" / "trace.jsonl").read_text()

    assert "[REDACTED:credential]" in run(passed=True)
    assert "[REDACTED:credential]" not in run(passed=False)
    assert token not in (tmp_path / "with" / "baseline" / "run" / "trace.jsonl").read_text()


def test_contamination_is_refused(pipeline: dict[str, Any], tmp_path: Path) -> None:
    files = pipeline["foundry"].package(pipeline["version_id"])
    t = _task(tmp_path, "baseline", files)
    with pytest.raises(ContaminationError):
        FakeScriptedRunner(SANDBOX).run(t)
    t2 = _task(tmp_path / "b", "baseline", None)
    (t2.workspace / ".claude" / "skills").mkdir(parents=True)
    with pytest.raises(ContaminationError):
        FakeScriptedRunner(SANDBOX).run(t2)


def test_parse_stream_baseline_contamination_visible() -> None:
    out = "\n".join(
        json.dumps(e)
        for e in [
            {"type": "system", "subtype": "init", "skills": ["fix-failing-pytest-suite"]},
            {"type": "result", "subtype": "error_max_turns"},
        ]
    )
    parsed = parse_stream(out, "fix-failing-pytest-suite")
    assert parsed["skill"]["discoverable"] is True and parsed["status"] == "max_turns"


def test_applicability_detects_contraindications(pipeline: dict[str, Any]) -> None:
    runs = pipeline["foundry"].eval_runs(pipeline["version_id"])
    rep = next(r for r in runs if r["kind"] == "applicability")["report"]
    assert rep["status"] == "pass" and rep["critical_failures"] == []
    pred = {c["case_id"]: c["predicted"] for c in rep["cases"]}
    assert (
        pred["app-pytest-nose-legacy"] == "do_not_use" and pred["app-pytest-jest"] == "do_not_use"
    )
    assert all(s["passed"] for s in rep["safety"])
