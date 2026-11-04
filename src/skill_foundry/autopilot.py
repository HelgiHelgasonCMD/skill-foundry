"""One unattended pass: new hubris tasks -> quarantined candidates, then stop.

This module deliberately has no route to approval, publication, rollback or live
evaluation. Those stay explicit human steps, and tests assert it stays that way.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from skill_foundry.evaluation.suite import EvalSuite
from skill_foundry.service import Foundry, FoundryError
from skill_foundry.synthesis.leakage import LeakageError
from skill_foundry.synthesis.synthesizers import TemplateSynthesizer
from skill_foundry.util import sha256_json

NOTIFY_TIMEOUT_S = 15


@contextmanager
def _lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"signature": None, "seen_specs": []}
    state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _signature(foundry: Foundry, suite_path: Path | None) -> str:
    """Changes whenever discovery could give a different answer."""
    rows = foundry.registry.experiences(foundry.ws, include_quarantined=True)
    return sha256_json(
        {
            "experiences": sorted([x.experience_id, x.trust.value, x.quarantined] for x in rows),
            "suite": EvalSuite.load(suite_path).content_hash if suite_path else None,
        }
    )


def _candidate(foundry: Foundry, spec: dict[str, Any], suite_path: Path) -> dict[str, Any]:
    name = spec["name"]
    cv = foundry.generate(
        spec["spec_hash"], TemplateSynthesizer(), suite_path=suite_path, actor="autopilot"
    )
    ref = f"{cv.name}@{cv.version}"
    result: dict[str, Any] = {
        "ref": ref,
        "validation": foundry.validate(ref, suite_path=suite_path),
    }
    if name in EvalSuite.load(suite_path).tracks:
        result["applicability"] = foundry.eval_applicability(ref, suite_path, name)["status"]
    else:
        result["applicability"] = "not_run: the suite has no track named after this skill"
    return result


def _notify(
    command: str | None, candidates: list[str], needs_suite: list[str], failed: list[str]
) -> str | None:
    if not command or not (candidates or needs_suite or failed):
        return None
    env = {
        **os.environ,
        "SF_NEW_CANDIDATES": ",".join(candidates),
        "SF_NEEDS_SUITE": ",".join(needs_suite),
        "SF_FAILED": ",".join(failed),
    }
    try:
        res = subprocess.run(
            ["sh", "-c", command],
            env=env,
            capture_output=True,
            timeout=NOTIFY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"notify failed: {exc}"
    return None if res.returncode == 0 else f"notify exited {res.returncode}"


def run_pass(
    foundry: Foundry,
    repos: list[str],
    *,
    suite_path: Path | None = None,
    hubris_binary: str = "hubris",
    notify_command: str | None = None,
) -> dict[str, Any]:
    """Run one idempotent pass. Without a suite it stops at the ProcedureSpec, because
    generating a skill requires held-out tasks to be kept out of synthesis."""
    home = foundry.settings.home
    state_path = home / "autopilot-state.json"
    with _lock(home / "autopilot.lock") as acquired:
        if not acquired:
            return {"skipped": "another pass holds the lock"}
        report: dict[str, Any] = {
            "import": foundry.import_hubris(repos, binary=hubris_binary),
            "index": foundry.index.sync().__dict__,
            "discovered": False,
            "new_candidates": [],
            "needs_suite": [],
            "failed": [],
        }
        state = _load_state(state_path)
        signature = _signature(foundry, suite_path)
        if signature == state["signature"]:
            return report
        found = foundry.discover(suite_path)
        report["discovered"] = True
        report["clusters"] = len(found["clusters"])
        seen = set(state["seen_specs"])
        for spec in found["specs"]:
            versions = foundry.registry.versions(spec["name"])
            if any(v.spec_hash == spec["spec_hash"] for v in versions):
                continue
            if suite_path is None:
                if spec["spec_hash"] not in seen:
                    report["needs_suite"].append(spec["name"])
                    seen.add(spec["spec_hash"])
                continue
            try:
                report["new_candidates"].append(_candidate(foundry, spec, suite_path))
            except (FoundryError, LeakageError) as exc:
                report["failed"].append({"spec": spec["name"], "error": str(exc)})
        report["notify_error"] = _notify(
            notify_command,
            [c["ref"] for c in report["new_candidates"]],
            report["needs_suite"],
            [f["spec"] for f in report["failed"]],
        )
        _save_state(state_path, {"signature": signature, "seen_specs": sorted(seen)})
        return report
