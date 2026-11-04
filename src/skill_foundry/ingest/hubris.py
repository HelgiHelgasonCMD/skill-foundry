"""Turn finished hubris managed tasks into canonical experience-bundle.v1 episodes.

hubris runs a task's verification itself and records the exit code and git state, so
its outcome is evidence rather than an agent's claim. The agent's own ``claim`` is
never used as evidence.

Tool calls are not in hubris's events, so attempts come from the session transcript.
That file also holds assistant text, thinking and raw API requests, so the reader
keeps an allowlist: tool name, Bash command and edited file path from ``tool_use``,
plus ``is_error`` from ``tool_result``. Everything else is discarded on read.

The output goes through the normal ingest pipeline, so sanitising, injection
quarantine and trust qualification all still apply.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from skill_foundry.contracts.bundle_v1 import EXT_ACCEPTANCE, make_bundle
from skill_foundry.util import sha256_text, utcnow

SOURCE = "hubris-task"
EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
# hubris verdicts that say nothing about whether the procedure worked.
SKIPPED_VERDICTS = frozenset({"errored", "timed_out", "cancelled"})
MAX_COMMAND_CHARS = 200  # extract_commands ignores longer backtick spans
MAX_SUMMARY_CHARS = 2000
_LEADING_CD = re.compile(r"^cd\s+\S+\s*&&\s*")
RUNTIME_MARKERS = (
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("package.json", "node"),
    ("Cargo.toml", "rust"),
    ("go.mod", "go"),
)
VERSION_FILES = (
    (".python-version", "python"),
    (".node-version", "node"),
    (".nvmrc", "node"),
)


class HubrisError(RuntimeError):
    pass


@dataclass
class TranscriptAttempts:
    attempts: list[dict[str, Any]] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    omitted: int = 0


@dataclass
class HubrisScan:
    episodes: list[dict[str, Any]] = field(default_factory=list)
    already_imported: int = 0
    not_allowlisted: int = 0
    not_finished: int = 0
    skipped_verdicts: dict[str, int] = field(default_factory=dict)
    omitted_commands: int = 0
    missing_transcripts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_episodes": [e["episode_id"] for e in self.episodes],
            "already_imported": self.already_imported,
            "not_allowlisted": self.not_allowlisted,
            "not_finished": self.not_finished,
            "skipped_verdicts": self.skipped_verdicts,
            "omitted_commands": self.omitted_commands,
            "missing_transcripts": self.missing_transcripts,
        }


def _hubris(binary: str, *args: str) -> Any:
    try:
        res = subprocess.run(
            [binary, *args, "--json"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HubrisError(f"could not run {binary}: {exc}") from exc
    if res.returncode != 0:
        raise HubrisError(f"{binary} {' '.join(args)} exited {res.returncode}: {res.stderr[:300]}")
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise HubrisError(f"{binary} {' '.join(args)} did not print JSON") from exc


def _git(root: str, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def repo_identity(repo_root: str) -> str:
    """``github.com/owner/repo`` from the remote, without any embedded credentials."""
    url = _git(repo_root, "remote", "get-url", "origin")
    if not url:
        return f"local/{Path(repo_root).name}"
    scp = re.fullmatch(r"(?:[^@/]+@)?([^:/]+):(?!/)(.+)", url)
    if scp:
        host, path = scp.group(1), scp.group(2)
    else:
        parts = urlsplit(url)
        host, path = parts.hostname or "", parts.path
    path = path.strip("/").removesuffix(".git")
    return f"{host}/{path}" if host else f"local/{Path(path).name}"


def repo_allowed(repo_id: str, allowlist: list[str]) -> bool:
    return any(repo_id == w or repo_id.endswith(f"/{w.strip('/')}") for w in allowlist)


def detect_runtime(repo_root: str, revision: str | None) -> tuple[str | None, dict[str, str]]:
    """Runtime family and declared versions, read from the commit that was tested."""
    if not revision:
        return None, {}
    versions: dict[str, str] = {}
    for path, key in VERSION_FILES:
        text = _git(repo_root, "show", f"{revision}:{path}")
        if text and key not in versions:
            versions[key] = text.split()[0]
    runtime = next(
        (
            family
            for path, family in RUNTIME_MARKERS
            if _git(repo_root, "cat-file", "-e", f"{revision}:{path}") is not None
        ),
        next(iter(versions), None),
    )
    return runtime, versions


def _command_summary(command: str) -> str | None:
    """The command as a backtick span, or None when it cannot be represented safely."""
    if "\n" in command.strip() or "`" in command:
        return None
    cleaned = _LEADING_CD.sub("", " ".join(command.split()))
    if not cleaned or len(cleaned) > MAX_COMMAND_CHARS:
        return None
    return f"Ran `{cleaned}`"


def _edit_summary(file_path: Any, cwd: str | None) -> str | None:
    if not isinstance(file_path, str) or not file_path:
        return None
    p = Path(file_path)
    if p.is_absolute():
        if not cwd or not p.is_relative_to(cwd):
            return None
        p = p.relative_to(cwd)
    return f"Edited a file: {p.as_posix()}"


def read_attempts(path: Path, cwd: str | None) -> TranscriptAttempts:
    """Allowlisted tool activity from a Claude Code session transcript.

    Sub-agent records are skipped. They are not the worker's own procedure.
    """
    calls: dict[str, tuple[str, str | None]] = {}
    errors: dict[str, bool] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("isSidechain"):
                continue
            kind = record.get("type")
            if kind not in ("assistant", "user"):
                continue
            message = record.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if kind == "assistant" and block.get("type") == "tool_use":
                    tool_id, tool, args = block.get("id"), block.get("name"), block.get("input")
                    if isinstance(tool_id, str) and tool_id not in calls:
                        calls[tool_id] = (str(tool), _attempt_summary(tool, args, cwd))
                elif kind == "user" and block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id")
                    if isinstance(tool_id, str):
                        errors[tool_id] = bool(block.get("is_error"))
    out = TranscriptAttempts()
    for tool_id, (tool, summary) in calls.items():
        if tool != "Bash" and tool not in EDIT_TOOLS:
            continue
        if summary is None:
            out.omitted += 1
            continue
        event_id = f"tool:{tool_id}"
        status = (
            "unknown" if tool_id not in errors else "failed" if errors[tool_id] else "completed"
        )
        out.event_ids.append(event_id)
        out.attempts.append(
            {
                "attempt_id": f"a{len(out.attempts) + 1}",
                "action_summary": summary,
                "status": status,
                "result_summary": None,
                "source_event_ids": [event_id],
            }
        )
    return out


def _attempt_summary(tool: Any, args: Any, cwd: str | None) -> str | None:
    if not isinstance(args, dict):
        return None
    if tool == "Bash":
        command = args.get("command")
        return _command_summary(command) if isinstance(command, str) else None
    if tool in EDIT_TOOLS:
        return _edit_summary(args.get("file_path") or args.get("notebook_path"), cwd)
    return None


def _iso(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def _acceptance(status: dict[str, Any], verify: dict[str, Any] | None) -> dict[str, Any] | None:
    """An acceptance link, only when the check could not have been weakened by the agent.

    A passing check counts only if the tree was clean and the files it depends on were
    protected and untouched. Without ``--protect`` the agent could have edited the tests.
    """
    outcome, verification = status["outcome"], status.get("verification")
    git = outcome.get("git") or {}
    command = (verify or {}).get("command")
    if (
        outcome["verdict"] != "succeeded"
        or not verification
        or verification.get("status") != "passed"
        or verification.get("exitCode") != 0
        or not command
        or not (verify or {}).get("protectedPaths")
        or git.get("protectedHits")
        or git.get("dirty")
        or not git.get("headSha")
    ):
        return None
    return {
        "criteria": [f"verify command passes: {command}"],
        "tested_revision": git["headSha"],
        "worktree_fingerprint": None,
    }


def _outcome_status(outcome: dict[str, Any], has_evidence: bool) -> str:
    verdict = outcome["verdict"]
    if verdict == "succeeded" and has_evidence:
        return "verified_success"
    if verdict == "failed" and has_evidence:
        return "verified_failure"
    claim = outcome.get("claim") or {}
    return "claimed_success" if claim.get("status") == "done" else "unknown"


def task_to_episode(
    status: dict[str, Any],
    events: list[dict[str, Any]],
    transcript: TranscriptAttempts,
    *,
    repo_id: str,
    runtime: str | None,
    versions: dict[str, str],
) -> dict[str, Any] | None:
    """Map one finished task. Returns None for verdicts that carry no evidence."""
    outcome = status.get("outcome")
    if not outcome or outcome["verdict"] in SKIPPED_VERDICTS:
        return None
    git = outcome.get("git") or {}
    worker = status.get("worker") or {}
    revision = git.get("headSha") or worker.get("baseSha")
    event_ids = [f"hubris:{e['seq']}" for e in events]
    verify = status.get("verify")
    verification = status.get("verification")
    evidence = []
    if verification and verification.get("status") in ("passed", "failed"):
        command = (verify or {}).get("command")
        shown = _command_summary(command) if isinstance(command, str) else None
        code = verification.get("exitCode")
        what = shown.removeprefix("Ran ") if shown else "the task's verify command"
        item: dict[str, Any] = {
            "evidence_id": verification["id"],
            "kind": "external_check",
            "summary": f"hubris ran {what}: exit {code}",
            "observed_at": _iso(verification["at"]),
            "revision": git.get("headSha"),
            "exit_code": code,
            "source_event_ids": [
                f"hubris:{e['seq']}" for e in events if e["kind"] in ("task.verifying", "task.done")
            ],
        }
        link = _acceptance(status, verify)
        if link:
            item["extensions"] = {EXT_ACCEPTANCE: link}
        evidence.append(item)
    return {
        "episode_id": status["id"],
        "task_summary": status["prompt"].strip()[:MAX_SUMMARY_CHARS],
        "environment": {
            "repo_id": repo_id,
            "revision": revision,
            "versions": versions,
            "runtime": runtime,
            "platform": platform.system().lower(),
        },
        "observations": [],
        "attempts": transcript.attempts,
        "outcome": {"status": _outcome_status(outcome, bool(evidence)), "evidence": evidence},
        "lessons": [],
        "provenance": {
            "source": SOURCE,
            "event_ids": [*event_ids, *transcript.event_ids],
            "redaction_applied": False,
        },
    }


def _transcript_path(events: list[dict[str, Any]], session_id: Any) -> Path | None:
    """The transcript the Stop hook reported, if it really is this session's."""
    for e in reversed(events):
        if e["kind"] != "agent.turn_ended":
            continue
        raw = (e.get("payload") or {}).get("transcriptPath")
        if isinstance(raw, str):
            p = Path(raw)
            if p.name == f"{session_id}.jsonl" and p.is_file():
                return p
    return None


def scan_tasks(allowlist: list[str], known_ids: set[str], binary: str = "hubris") -> HubrisScan:
    """Build episodes for finished tasks in allowlisted repos that are not yet imported."""
    if not allowlist:
        raise HubrisError("no repositories allowlisted; name at least one with --repo")
    scan = HubrisScan()
    for row in _hubris(binary, "task", "list"):
        task_id = row["id"]
        if task_id in known_ids:
            scan.already_imported += 1
            continue
        if row["state"] not in ("done", "cancelled"):
            # `task status` can restart a supervisor, so running tasks are left alone.
            scan.not_finished += 1
            continue
        status = _hubris(binary, "task", "status", task_id)
        repo_root = status["repoRoot"]
        repo_id = repo_identity(repo_root)
        if not repo_allowed(repo_id, allowlist):
            scan.not_allowlisted += 1
            continue
        verdict = (status.get("outcome") or {}).get("verdict", "none")
        if verdict in SKIPPED_VERDICTS or verdict == "none":
            scan.skipped_verdicts[verdict] = scan.skipped_verdicts.get(verdict, 0) + 1
            continue
        events = _hubris(binary, "task", "events", task_id)
        worker = status.get("worker") or {}
        path = _transcript_path(events, worker.get("sessionId"))
        transcript = read_attempts(path, worker.get("cwd")) if path else TranscriptAttempts()
        if path is None:
            scan.missing_transcripts.append(task_id)
        scan.omitted_commands += transcript.omitted
        git = status["outcome"].get("git") or {}
        runtime, versions = detect_runtime(repo_root, git.get("headSha") or worker.get("baseSha"))
        episode = task_to_episode(
            status, events, transcript, repo_id=repo_id, runtime=runtime, versions=versions
        )
        if episode is not None:
            scan.episodes.append(episode)
    return scan


def bundle_for(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    ids = ",".join(sorted(e["episode_id"] for e in episodes))
    return make_bundle(
        f"hubris-{sha256_text(ids)[:16]}", episodes, exported_at=utcnow().isoformat()
    )
