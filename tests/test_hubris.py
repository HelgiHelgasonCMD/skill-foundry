"""hubris managed tasks -> experiences -> quarantined candidates, with a fake hubris CLI."""

from __future__ import annotations

import ast
import json
import stat
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

import skill_foundry.autopilot as autopilot_module
from conftest import settings_for
from skill_foundry.autopilot import _lock, run_pass
from skill_foundry.cli.app import app
from skill_foundry.contracts.bundle_v1 import EXT_ACCEPTANCE, assert_valid
from skill_foundry.domain.models import LifecycleState, TrustLevel
from skill_foundry.ingest.hubris import (
    TranscriptAttempts,
    bundle_for,
    read_attempts,
    repo_allowed,
    repo_identity,
    scan_tasks,
    task_to_episode,
)
from skill_foundry.registry import orm
from skill_foundry.service import Foundry

HEAD = "a" * 40
CANARIES = {
    "thinking": "CANARY-THINKING-7f3a",
    "text": "CANARY-TEXT-91bc",
    "blob": "CANARY-BLOB-22de",
    "result": "CANARY-RESULT-5e01",
    "attachment": "CANARY-ATTACH-c4d4",
}
FAKE_SECRET = "ghp_" + "b1" * 18  # assembled at runtime so scanners ignore this file

FAKE_HUBRIS = """#!/usr/bin/env python3
import json, os, sys
root = os.environ["FAKE_HUBRIS_DIR"]
args = [a for a in sys.argv[1:] if a != "--json"]
if args[:2] == ["task", "list"]:
    name = "list"
else:
    name = args[2] + "." + args[1]
sys.stdout.write(open(os.path.join(root, name + ".json")).read())
"""


def _status(
    verdict: str = "succeeded",
    *,
    protect: tuple[str, ...] = ("tests/**",),
    dirty: bool = False,
    hits: tuple[str, ...] = (),
    claim: str | None = "done",
    verification: str | None = "passed",
    exit_code: int | None = 0,
    command: str | None = "uv run pytest -q",
) -> dict[str, Any]:
    git = {
        "baseSha": "c" * 40,
        "headSha": HEAD,
        "commits": 1,
        "changedFiles": ["a.py"],
        "dirty": dirty,
        "protectedHits": list(hits),
    }
    return {
        "id": "task_1",
        "state": "done",
        "prompt": "Fix the failing discount tests",
        "repoRoot": "/nonexistent",
        "outcome": {
            "verdict": verdict,
            "reasons": [],
            "claim": {"status": claim, "summary": "did it", "at": 1} if claim else None,
            "git": git,
        },
        "worker": {"sessionId": "s1", "cwd": "/w", "baseSha": "c" * 40},
        "verification": {
            "id": "verify_1",
            "status": verification,
            "exitCode": exit_code,
            "at": 1793610963574,
        }
        if verification
        else None,
        "verify": {"command": command, "protectedPaths": list(protect)} if command else None,
    }


EVENTS = [
    {"seq": 1, "kind": "task.created", "payload": {}},
    {"seq": 2, "kind": "task.verifying", "payload": {}},
    {"seq": 3, "kind": "task.done", "payload": {}},
]


def _episode(status: dict[str, Any]) -> dict[str, Any] | None:
    return task_to_episode(
        status,
        EVENTS,
        TranscriptAttempts(),
        repo_id="github.com/o/r",
        runtime="python",
        versions={},
    )


def _ingest(foundry: Foundry, tmp_path: Path, episode: dict[str, Any]) -> Any:
    assert_valid(bundle_for([episode]))
    path = tmp_path / "b.json"
    path.write_text(json.dumps(bundle_for([episode])))
    foundry.ingest(path, sync_index=False)
    return foundry.registry.experiences(foundry.ws, include_quarantined=True)[0]


@pytest.mark.parametrize(
    ("status", "outcome", "trust", "linked"),
    [
        (_status(), "verified_success", TrustLevel.VERIFIED, True),
        (_status(protect=()), "verified_success", TrustLevel.RECORDED, False),
        (_status(dirty=True), "verified_success", TrustLevel.RECORDED, False),
        (_status(hits=("tests/a.py",)), "verified_success", TrustLevel.RECORDED, False),
        (_status(command=None), "verified_success", TrustLevel.RECORDED, False),
        (
            _status("failed", verification="failed", exit_code=1),
            "verified_failure",
            TrustLevel.VERIFIED,
            False,
        ),
        (
            _status("inconclusive", verification=None),
            "claimed_success",
            TrustLevel.SELF_REPORTED,
            False,
        ),
        (
            _status("inconclusive", verification=None, claim=None),
            "unknown",
            TrustLevel.SELF_REPORTED,
            False,
        ),
        # A passing check on an otherwise inconclusive task is still only a claim.
        (_status("inconclusive"), "claimed_success", TrustLevel.SELF_REPORTED, False),
    ],
)
def test_outcome_mapping(
    foundry: Foundry,
    tmp_path: Path,
    status: dict[str, Any],
    outcome: str,
    trust: TrustLevel,
    linked: bool,
) -> None:
    ep = _episode(status)
    assert ep is not None
    assert ep["outcome"]["status"] == outcome
    ev = ep["outcome"]["evidence"]
    assert bool(ev and EXT_ACCEPTANCE in ev[0].get("extensions", {})) is linked
    assert _ingest(foundry, tmp_path, ep).trust == trust


@pytest.mark.parametrize("verdict", ["errored", "timed_out", "cancelled"])
def test_verdicts_without_evidence_are_skipped(verdict: str) -> None:
    assert _episode(_status(verdict)) is None


def test_agent_claim_is_never_evidence() -> None:
    ep = _episode(_status("inconclusive", verification=None))
    assert ep is not None and ep["outcome"]["evidence"] == []
    assert "did it" not in json.dumps(ep)


def _record(role: str, *blocks: dict[str, Any], **extra: Any) -> str:
    return json.dumps({"type": role, "message": {"content": list(blocks)}, **extra})


def _tool(tool_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": args}


def _result(tool_id: str, *, error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "is_error": error,
        "content": CANARIES["result"],
    }


def _transcript(path: Path, session_lines: list[str]) -> Path:
    noise = [
        _record("assistant", {"type": "thinking", "thinking": CANARIES["thinking"]}),
        _record("assistant", {"type": "text", "text": CANARIES["text"]}),
        json.dumps(
            {
                "type": "api-request-blob",
                "message": {"content": [{"type": "text", "text": CANARIES["blob"]}]},
            }
        ),
        json.dumps({"type": "attachment", "attachment": CANARIES["attachment"]}),
        "not json at all",
    ]
    path.write_text("\n".join([*noise, *session_lines]) + "\n")
    return path


def test_transcript_allowlist(tmp_path: Path) -> None:
    long_cmd = "echo " + "x" * 300
    path = _transcript(
        tmp_path / "s.jsonl",
        [
            _record("assistant", _tool("t1", "Bash", command="cd /home/u/repo && pytest -x -q")),
            _record("user", _result("t1", error=True)),
            _record("assistant", _tool("t2", "Edit", file_path="/w/src/a.py", old_string="k")),
            _record("user", _result("t2")),
            _record("assistant", _tool("t3", "Read", file_path="/w/src/a.py")),
            _record("assistant", _tool("t4", "Bash", command="cat <<EOF\nhello\nEOF")),
            _record("assistant", _tool("t5", "Bash", command=long_cmd)),
            _record("assistant", _tool("t6", "Write", file_path="/elsewhere/b.py", content="k")),
            _record("assistant", _tool("t7", "Bash", command="git status")),
            _record("assistant", _tool("t8", "Bash", command="subagent"), isSidechain=True),
        ],
    )
    got = read_attempts(path, "/w")
    assert [(a["action_summary"], a["status"]) for a in got.attempts] == [
        ("Ran `pytest -x -q`", "failed"),
        ("Edited a file: src/a.py", "completed"),
        ("Ran `git status`", "unknown"),
    ]
    assert got.omitted == 3  # heredoc, over-long command, edit outside the worktree
    assert all(a["source_event_ids"][0] in got.event_ids for a in got.attempts)
    assert "CANARY" not in json.dumps(got.attempts)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo(path: Path, python: str, remote: str | None = None) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    (path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (path / ".python-version").write_text(python + "\n")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "init")
    if remote:
        _git(path, "remote", "add", "origin", remote)
    return _git(path, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://user:hunter2@github.com/o/r.git", "github.com/o/r"),
        ("git@github.com:o/r.git", "github.com/o/r"),
        ("ssh://git@github.com/o/r", "github.com/o/r"),
    ],
)
def test_repo_identity_drops_credentials(tmp_path: Path, remote: str, expected: str) -> None:
    _repo(tmp_path / "r", "3.12", remote)
    assert repo_identity(str(tmp_path / "r")) == expected


def test_repo_identity_without_remote(tmp_path: Path) -> None:
    assert repo_identity(str(tmp_path / "gone")) == "local/gone"


def test_repo_allowlist_matches_whole_path_segments() -> None:
    assert repo_allowed("github.com/o/r", ["r"])
    assert repo_allowed("github.com/o/r", ["o/r"])
    assert repo_allowed("github.com/o/r", ["github.com/o/r"])
    assert not repo_allowed("github.com/o/regulated-r", ["r"])
    assert not repo_allowed("github.com/o/r", [])


class FakeHubris:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = root / "hubris-data"
        self.root.mkdir()
        self.rows: list[dict[str, Any]] = []
        self.binary = root / "fake-hubris"
        self.binary.write_text(FAKE_HUBRIS)
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("FAKE_HUBRIS_DIR", str(self.root))
        self._flush()

    def _flush(self) -> None:
        (self.root / "list.json").write_text(json.dumps(self.rows))

    def add(
        self,
        task_id: str,
        prompt: str,
        repo_root: Path,
        head: str,
        lines: list[str],
        *,
        verdict: str = "succeeded",
        transcript: bool = True,
    ) -> None:
        status = _status(verdict)
        status.update(id=task_id, prompt=prompt, repoRoot=str(repo_root))
        status["outcome"]["git"]["headSha"] = head
        status["worker"] = {"sessionId": task_id, "cwd": "/w", "baseSha": head}
        path = _transcript(self.root / f"{task_id}.jsonl", lines)
        turn = {"transcriptPath": str(path) if transcript else "/nope/other.jsonl"}
        events = [
            *EVENTS[:1],
            {"seq": 9, "kind": "agent.turn_ended", "payload": turn},
            *EVENTS[1:],
        ]
        (self.root / f"{task_id}.status.json").write_text(json.dumps(status))
        (self.root / f"{task_id}.events.json").write_text(json.dumps(events))
        self.rows.append({"id": task_id, "state": "done", "verdict": verdict, "prompt": prompt})
        self._flush()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHubris:
    return FakeHubris(tmp_path, monkeypatch)


def _steps(*commands: str) -> list[str]:
    out = []
    for i, cmd in enumerate(commands):
        out.append(_record("assistant", _tool(f"t{i}", "Bash", command=cmd)))
        out.append(_record("user", _result(f"t{i}")))
    return out


def _corpus(fake: FakeHubris, tmp_path: Path) -> None:
    """Three independent tasks in three repos across two declared Python versions."""
    tasks = [
        ("alpha", "3.11", "Fix the failing pytest suite in the billing discount module"),
        ("beta", "3.12", "Fix the failing pytest suite in the invoice rounding module"),
        ("gamma", "3.12", "Fix the failing pytest suite in the slug helper module"),
    ]
    for name, py, prompt in tasks:
        root = tmp_path / "repos" / name
        head = _repo(root, py, f"https://github.com/acme/{name}.git")
        fake.add(
            f"task_{name}",
            prompt,
            root,
            head,
            _steps(
                "python -m pytest -x -q",
                "python -m pytest -q -vv",
                "git log -p -n 5",
                "python -m pytest -q",
            ),
        )


ALL_REPOS = ["alpha", "beta", "gamma"]


def _empty_suite(path: Path, tracks: dict[str, Any] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "eval-suite.v1",
                "suite_id": "t",
                "corpus_version": "1",
                "tracks": tracks or {},
                "cases": [],
                "applicability_cases": [],
            }
        )
    )
    return path


@pytest.fixture
def f(tmp_path: Path) -> Iterator[Foundry]:
    fd = Foundry(settings_for(tmp_path / "sf"))
    yield fd
    fd.close()


def _all_bytes(root: Path) -> bytes:
    return b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())


def _batches(f: Foundry) -> list[str]:
    with f.db.tx() as s:
        return list(s.scalars(select(orm.IngestBatch.batch_id)))


def test_import_nothing_unless_a_repo_is_named(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    with pytest.raises(RuntimeError, match="allowlisted"):
        f.import_hubris([], binary=str(fake.binary))
    rep = f.import_hubris(["alpha"], binary=str(fake.binary))
    assert rep["new_episodes"] == ["task_alpha"] and rep["not_allowlisted"] == 2


def test_import_is_idempotent_and_records_environment(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    first = f.import_hubris(["alpha"], binary=str(fake.binary))
    assert first["ingest"]["trust"] == {"verified": 1}
    x = f.registry.experiences(f.ws)[0]
    assert x.environment.repo_id == "github.com/acme/alpha"
    assert x.environment.runtime == "python" and x.environment.versions == {"python": "3.11"}
    assert x.provenance.source == "hubris-task"
    batches = len(_batches(f))
    second = f.import_hubris(["alpha"], binary=str(fake.binary))
    assert second["new_episodes"] == [] and second["already_imported"] == 1
    assert "ingest" not in second
    assert len(_batches(f)) == batches


def test_private_content_never_reaches_storage(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    root = tmp_path / "repos" / "alpha"
    head = _repo(root, "3.12", "https://github.com/acme/alpha.git")
    fake.add(
        "task_secret",
        "Fix the discount tests",
        root,
        head,
        _steps("python -m pytest -q", f"curl -H 'Authorization: {FAKE_SECRET}' example.invalid"),
    )
    f.import_hubris(["alpha"], binary=str(fake.binary))
    out = tmp_path / "export.json"
    f.export_experiences(out)
    f.close()  # flush SQLite's WAL and the index before scanning the files on disk
    stored = _all_bytes(f.settings.home) + out.read_bytes()
    for canary in CANARIES.values():
        assert canary.encode() not in stored
    assert FAKE_SECRET.encode() not in stored
    assert b"Fix the discount tests" in stored  # the scan really covers the stored data


def test_transcript_for_another_session_is_not_read(fake: FakeHubris, tmp_path: Path) -> None:
    root = tmp_path / "repos" / "alpha"
    head = _repo(root, "3.12", "https://github.com/acme/alpha.git")
    fake.add("task_x", "Fix it", root, head, _steps("pytest -q"), transcript=False)
    scan = scan_tasks(["alpha"], set(), str(fake.binary))
    assert scan.missing_transcripts == ["task_x"]
    assert scan.episodes[0]["attempts"] == []


def test_autopilot_stops_at_spec_without_a_suite(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    rep = run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary))
    assert rep["needs_suite"], rep
    assert rep["new_candidates"] == []
    assert f.registry.versions() == []


def test_autopilot_reaches_quarantine_and_stops(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    name = run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary))["needs_suite"][0]
    suite = _empty_suite(
        tmp_path / "suite.json", {name: {"capability_cases": [], "applicability_cases": []}}
    )
    before = {p: _git(p, "status", "--short") for p in (tmp_path / "repos").iterdir()}
    rep = run_pass(f, ALL_REPOS, suite_path=suite, hubris_binary=str(fake.binary))
    assert not rep["failed"], rep["failed"]
    assert [c["ref"] for c in rep["new_candidates"]] == [f"{name}@0.1.0"]
    cv = f.registry.get_version(f"{name}@0.1.0")
    assert cv.state in (LifecycleState.CANDIDATE, LifecycleState.VALIDATED)
    assert (f.settings.quarantine_dir / f"{name}@0.1.0" / name / "SKILL.md").exists()
    assert {p: _git(p, "status", "--short") for p in before} == before
    assert not any((p / ".claude").exists() for p in before)


def test_autopilot_second_pass_changes_nothing(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary))
    clusters = len(f.clusters())
    rep = run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary))
    assert rep["discovered"] is False and rep["import"]["new_episodes"] == []
    assert len(f.clusters()) == clusters


def test_autopilot_cannot_reach_approval_or_publication() -> None:
    tree = ast.parse(Path(autopilot_module.__file__).read_text())
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not used & {"decide", "publish", "rollback", "eval_capability", "approval_for"}


def test_autopilot_skips_when_locked(fake: FakeHubris, tmp_path: Path, f: Foundry) -> None:
    with _lock(f.settings.home / "autopilot.lock") as held:
        assert held
        assert "skipped" in run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary))


def test_autopilot_notifies_once_with_values_in_the_environment(
    fake: FakeHubris, tmp_path: Path, f: Foundry
) -> None:
    _corpus(fake, tmp_path)
    log = tmp_path / "notified.txt"
    cmd = f'echo "$SF_NEEDS_SUITE" >> {log}'
    rep = run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary), notify_command=cmd)
    assert rep["notify_error"] is None
    assert log.read_text().strip() == ",".join(rep["needs_suite"])
    run_pass(f, ALL_REPOS, hubris_binary=str(fake.binary), notify_command=cmd)
    assert len(log.read_text().splitlines()) == 1


def test_cli_import_hubris(
    fake: FakeHubris, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus(fake, tmp_path)
    monkeypatch.setenv("SF_HOME", str(tmp_path / "cli-sf"))
    res = CliRunner().invoke(
        app, ["import", "hubris", "--repo", "alpha", "--hubris-bin", str(fake.binary), "--json"]
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["new_episodes"] == ["task_alpha"]
