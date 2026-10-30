"""Journaled, crash-recoverable skill installation and exact rollback.

Each step is recorded (fsynced) in a write-ahead journal *before* the next one runs:

    PLANNED -> STAGED -> BACKED_UP -> SWAPPED -> VERIFIED -> COMMITTED
                                                        \\-> ABORTED (recovered)

* STAGED     package written to ``<skills>/.sf-staging-<id>-<name>`` and hash-verified
* BACKED_UP  existing destination copied to ``.sf/installations/<id>/before`` and then
             renamed aside to ``<skills>/.sf-old-<id>-<name>``
* SWAPPED    staging renamed to the destination
* VERIFIED   destination hashes equal the approved package
* COMMITTED  old copy removed; registry updated

Each ``rename`` is atomic on a single POSIX filesystem; the *sequence* is not. There
is a short window between moving the old package aside and moving the new one in
during which the destination is absent. ``recover`` inspects the journal and the
filesystem and rolls forward or back to a consistent state.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from skill_foundry.config import Settings
from skill_foundry.domain.models import LifecycleState
from skill_foundry.registry import orm
from skill_foundry.registry.repo import Registry
from skill_foundry.util import package_hash, sha256_bytes, tree_hashes, utcnow

STATES = ("PLANNED", "STAGED", "BACKED_UP", "SWAPPED", "VERIFIED", "COMMITTED")
TERMINAL = ("COMMITTED", "ABORTED", "ROLLED_BACK")  # ABORTED: recovered to pre-op state


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class Publisher:
    """Where a package is placed inside a target project."""

    name: str
    skills_subdir: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ClaudeProjectPublisher(Publisher):
    name: str = "claude-project"
    skills_subdir: tuple[str, ...] = (".claude", "skills")
    note: str = "Claude Code project-local skills (.claude/skills/<name>/SKILL.md)"


@dataclass(frozen=True)
class AgentsDirPublisher(Publisher):
    """Adapter for agents that read ``.agents/skills``. Discovery and execution
    semantics differ between agents; this only places files, it does not promise
    identical behaviour."""

    name: str = "agents-dir"
    skills_subdir: tuple[str, ...] = (".agents", "skills")
    note: str = "Generic .agents/skills/<name>/ layout; execution semantics vary by agent"


PUBLISHERS: dict[str, Publisher] = {
    "claude-project": ClaudeProjectPublisher(),
    "agents-dir": AgentsDirPublisher(),
}


def _forbidden_roots() -> list[Path]:
    home = Path.home().resolve()
    return [home / ".claude", home / ".agents", home / ".config"]


def safe_destination(target_root: Path, publisher: Publisher, name: str) -> Path:
    if not target_root.exists() or not target_root.is_dir():
        raise PublishError(f"target {target_root} is not an existing directory")
    root = target_root.resolve()
    home = Path.home().resolve()
    if root in (Path("/"), home):
        raise PublishError(f"refusing to publish into {root}: use a project directory")
    for bad in _forbidden_roots():
        if root == bad or root.is_relative_to(bad):
            raise PublishError(f"refusing to write into global agent configuration ({bad})")
    cur = root
    for part in (*publisher.skills_subdir, name):
        cur = cur / part
        if cur.is_symlink():
            raise PublishError(f"{cur} is a symlink; refusing (could escape the target)")
    dest = root.joinpath(*publisher.skills_subdir, name)
    if not dest.resolve().is_relative_to(root):
        raise PublishError("destination escapes the target root")
    for bad in _forbidden_roots():
        if dest.resolve().is_relative_to(bad):
            raise PublishError("destination resolves into global agent configuration")
    return dest


def _check_package_paths(files: dict[str, bytes]) -> None:
    for rel in files:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or rel.startswith("~"):
            raise PublishError(f"package path {rel!r} is not a safe relative path")


@dataclass
class PublishPlan:
    destination: Path
    exists: bool
    managed_by: str | None  # install_id of the Skill Foundry install currently there
    before_hashes: dict[str, str] | None
    changes: list[dict[str, str]]
    diff: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["destination"] = str(self.destination)
        return d


def plan(dest: Path, files: dict[str, bytes], managed_by: str | None) -> PublishPlan:
    exists = dest.exists()
    before = tree_hashes(dest) if exists else None
    changes = []
    diff_parts = []
    old_paths = set(before or {})
    for rel in sorted(old_paths | set(files)):
        new = files.get(rel)
        old_p = dest / rel
        old = old_p.read_bytes() if exists and old_p.is_file() else None
        if old == new:
            continue
        kind = "add" if old is None else "remove" if new is None else "modify"
        changes.append({"path": rel, "change": kind})
        a = (old or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        b = (new or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        diff_parts.append(
            "".join(
                difflib.unified_diff(
                    a,
                    b,
                    fromfile=f"a/{rel}" if old is not None else "/dev/null",
                    tofile=f"b/{rel}" if new is not None else "/dev/null",
                )
            )
        )
    return PublishPlan(dest, exists, managed_by, before, changes, "".join(diff_parts))


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        _fsync_dir(self.path.parent)

    def advance(self, state: str, **extra: Any) -> dict[str, Any]:
        data = self.read()
        data["state"] = state
        data.setdefault("history", []).append({"state": state, "at": utcnow().isoformat()})
        data.update(extra)
        self.write(data)
        return data


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for rel, data in sorted(files.items()):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    for d in sorted({p.parent for p in root.rglob("*")} | {root}, reverse=True):
        _fsync_dir(d)


Fault = Callable[[str], None]


@dataclass
class InstallResult:
    install_id: str
    destination: str
    state: str
    package_sha256: str
    replaced_existing: bool
    backup: str | None
    journal: str
    events: list[str] = field(default_factory=list)


class Installer:
    def __init__(self, registry: Registry, settings: Settings):
        self.registry = registry
        self.settings = settings

    # ------------------------------------------------------------------ helpers
    def _dir(self, install_id: str) -> Path:
        return self.settings.installations_dir / install_id

    def managed_install_at(self, dest: Path) -> str | None:
        """The committed Skill Foundry install whose files are exactly what is at ``dest``."""
        if not dest.exists():
            return None
        current = tree_hashes(dest)
        with self.registry.db.tx() as s:
            rows = s.scalars(
                select(orm.InstallationRow)
                .where(
                    orm.InstallationRow.destination == str(dest),
                    orm.InstallationRow.state == "COMMITTED",
                )
                .order_by(orm.InstallationRow.created_at.desc())
            ).all()
            for r in rows:
                if r.after_hashes == current:
                    return r.install_id
        return None

    def _event(self, install_id: str, action: str, detail: dict[str, Any], actor: str) -> None:
        with self.registry.db.tx() as s:
            s.add(
                orm.InstallEvent(
                    install_id=install_id,
                    action=action,
                    detail=detail,
                    actor=actor,
                    created_at=utcnow(),
                )
            )

    def _set_state(self, install_id: str, state: str) -> None:
        with self.registry.db.tx() as s:
            row = s.get(orm.InstallationRow, install_id)
            if row is not None:
                row.state = state
                row.updated_at = utcnow()

    # ------------------------------------------------------------------ install
    def install(
        self,
        *,
        version_id: str,
        decision_id: str,
        files: dict[str, bytes],
        target_root: Path,
        publisher: Publisher,
        actor: str,
        force_overwrite: bool = False,
        fault: Fault | None = None,
    ) -> InstallResult:
        _check_package_paths(files)
        cv = self.registry.get_version(version_id)
        pkg = package_hash({k: sha256_bytes(v) for k, v in files.items()})
        if pkg != cv.package_sha256:
            raise PublishError("package bytes do not match the registered package hash")
        with self.registry.db.tx() as s:
            dec = s.get(orm.ReviewRow, decision_id)
            if dec is None or dec.decision != "approve" or dec.version_id != version_id:
                raise PublishError("no approval decision for this exact version")
            if dec.package_sha256 != pkg:
                raise PublishError("approval was granted for different package content")
        if cv.state not in (LifecycleState.APPROVED, LifecycleState.ROLLED_BACK):
            raise PublishError(f"version is {cv.state.value}; only approved versions publish")

        dest = safe_destination(target_root, publisher, cv.name)
        managed = self.managed_install_at(dest)
        if dest.exists() and managed is None and not force_overwrite:
            raise PublishError(
                f"{dest} exists and was not installed by Skill Foundry (or was modified since); "
                "refusing to overwrite. Use --force-overwrite (a backup is taken first)."
            )

        install_id = "inst-" + uuid.uuid4().hex[:12]
        idir = self._dir(install_id)
        idir.mkdir(parents=True)
        staging = dest.parent / f".sf-staging-{install_id}-{cv.name}"
        old = dest.parent / f".sf-old-{install_id}-{cv.name}"
        backup = idir / "before"
        expected = {k: sha256_bytes(v) for k, v in files.items()}
        journal = Journal(idir / "journal.json")
        journal.write(
            {
                "install_id": install_id,
                "op": "install",
                "state": "PLANNED",
                "history": [{"state": "PLANNED", "at": utcnow().isoformat()}],
                "version_id": version_id,
                "destination": str(dest),
                "staging": str(staging),
                "old": str(old),
                "backup": str(backup),
                "expected": expected,
                "existed": dest.exists(),
                "previous_install_id": managed,
            }
        )
        before = tree_hashes(dest) if dest.exists() else None
        with self.registry.db.tx() as s:
            s.add(
                orm.InstallationRow(
                    install_id=install_id,
                    version_id=version_id,
                    package_sha256=pkg,
                    decision_id=decision_id,
                    publisher=publisher.name,
                    target_root=str(target_root.resolve()),
                    destination=str(dest),
                    state="PLANNED",
                    before_hashes=before,
                    after_hashes=expected,
                    previous_install_id=managed,
                    actor=actor,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
        step = fault or (lambda _s: None)

        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_tree(staging, files)
        if tree_hashes(staging) != expected:
            raise PublishError("staged package failed hash verification")
        journal.advance("STAGED")
        step("STAGED")
        if dest.exists():
            shutil.copytree(dest, backup, symlinks=True)
            if tree_hashes(backup) != before:
                raise PublishError("backup verification failed")
            os.rename(dest, old)
            _fsync_dir(dest.parent)
        journal.advance("BACKED_UP")
        step("BACKED_UP")
        os.rename(staging, dest)
        _fsync_dir(dest.parent)
        journal.advance("SWAPPED")
        step("SWAPPED")
        return self._finish(install_id, journal, actor, step)

    def _finish(self, install_id: str, journal: Journal, actor: str, step: Fault) -> InstallResult:
        data = journal.read()
        dest, old = Path(data["destination"]), Path(data["old"])
        if tree_hashes(dest) != data["expected"]:
            raise PublishError("installed destination failed hash verification")
        if data["state"] != "VERIFIED":
            journal.advance("VERIFIED")
        step("VERIFIED")
        if old.exists():
            shutil.rmtree(old)
        journal.advance("COMMITTED")
        self._set_state(install_id, "COMMITTED")
        prev = data.get("previous_install_id")
        if prev:
            self._set_state(prev, "SUPERSEDED")
        cv = self.registry.get_version(data["version_id"])
        if cv.state != LifecycleState.PUBLISHED:
            self.registry.transition(
                data["version_id"],
                LifecycleState.PUBLISHED,
                actor,
                f"installed at {dest} ({install_id})",
            )
        self._event(
            install_id,
            "install",
            {"destination": str(dest), "replaced_existing": data["existed"]},
            actor,
        )
        return InstallResult(
            install_id=install_id,
            destination=str(dest),
            state="COMMITTED",
            package_sha256=self.registry.get_version(data["version_id"]).package_sha256,
            replaced_existing=bool(data["existed"]),
            backup=data["backup"] if data["existed"] else None,
            journal=str(journal.path),
        )

    # ------------------------------------------------------------------ recovery
    def pending(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.settings.installations_dir.exists():
            return out
        for j in sorted(self.settings.installations_dir.glob("*/*.json")):
            data = json.loads(j.read_text(encoding="utf-8"))
            if data.get("state") not in (*TERMINAL, "ABORTED"):
                out.append({**data, "journal": str(j)})
        return out

    def recover(self, install_id: str, actor: str = "recover") -> str:
        """Bring an interrupted install to a consistent state. Returns the final state."""
        journal = Journal(self._dir(install_id) / "journal.json")
        data = journal.read()
        if data.get("op") == "rollback":
            raise PublishError("use recover_rollback for rollback journals")
        state = data["state"]
        dest, staging, old = Path(data["destination"]), Path(data["staging"]), Path(data["old"])
        expected = data["expected"]
        if state in TERMINAL:
            return str(state)
        if state in ("SWAPPED", "VERIFIED") or (
            state == "BACKED_UP"
            and not staging.exists()
            and dest.exists()
            and tree_hashes(dest) == expected
        ):
            # New package is in place: roll forward.
            self._finish(install_id, journal, actor, lambda _s: None)
            self._event(install_id, "recover", {"action": "roll_forward", "from": state}, actor)
            return "COMMITTED"
        # Roll back to the pre-install state.
        if staging.exists():
            shutil.rmtree(staging)
        if old.exists():
            if dest.exists():
                shutil.rmtree(dest)
            os.rename(old, dest)
        journal.advance("ABORTED")
        self._set_state(install_id, "ABORTED")
        self._event(install_id, "recover", {"action": "roll_back", "from": state}, actor)
        return "ABORTED"

    def recover_rollback(self, install_id: str, actor: str = "recover") -> str:
        """Undo an interrupted rollback, leaving the installed package in place."""
        idir = self._dir(install_id)
        journal = Journal(idir / "rollback.json")
        data = journal.read()
        if data["state"] in TERMINAL or data["state"] == "ABORTED":
            return str(data["state"])
        dest = Path(data["destination"])
        old = dest.parent / f".sf-rb-old-{install_id}"
        staging = dest.parent / f".sf-rb-staging-{install_id}"
        if data["state"] == "SWAPPED":
            if old.exists():
                shutil.rmtree(old)
            journal.advance("ROLLED_BACK")
            self._set_state(install_id, "ROLLED_BACK")
            self._event(install_id, "recover", {"action": "finish_rollback"}, actor)
            return "ROLLED_BACK"
        if staging.exists():
            shutil.rmtree(staging)
        if old.exists() and not dest.exists():
            os.rename(old, dest)
        journal.advance("ABORTED")
        self._event(install_id, "recover", {"action": "abort_rollback"}, actor)
        return "ABORTED"

    # ------------------------------------------------------------------ rollback
    def rollback(
        self, install_id: str, *, actor: str, policy: str = "refuse", fault: Fault | None = None
    ) -> dict[str, Any]:
        if policy not in ("refuse", "restore-anyway"):
            raise ValueError("policy must be 'refuse' or 'restore-anyway'")
        with self.registry.db.tx() as s:
            row = s.get(orm.InstallationRow, install_id)
            if row is None:
                raise PublishError(f"unknown installation {install_id}")
            if row.state != "COMMITTED":
                raise PublishError(f"installation is {row.state}; only COMMITTED can roll back")
            dest = Path(row.destination)
            after, before = dict(row.after_hashes), row.before_hashes
            version_id, prev = row.version_id, row.previous_install_id
        current = tree_hashes(dest) if dest.exists() else None
        drift = (
            sorted(set((current or {}).items()) ^ set(after.items())) if current != after else []
        )
        idir = self._dir(install_id)
        if drift:
            if policy == "refuse":
                files = sorted({p for p, _ in drift})
                raise PublishError(
                    f"destination changed since installation ({', '.join(files)}); refusing to "
                    "overwrite user changes. Re-run with --recovery restore-anyway to snapshot "
                    "the current state and restore the previous package."
                )
            if dest.exists():
                shutil.copytree(dest, idir / f"drift-{utcnow().strftime('%Y%m%d%H%M%S')}")
        journal = Journal(idir / "rollback.json")
        old = dest.parent / f".sf-rb-old-{install_id}"
        staging = dest.parent / f".sf-rb-staging-{install_id}"
        journal.write(
            {
                "install_id": install_id,
                "op": "rollback",
                "state": "PLANNED",
                "history": [],
                "destination": str(dest),
                "policy": policy,
            }
        )
        step = fault or (lambda _s: None)
        if before is not None:
            shutil.copytree(idir / "before", staging, symlinks=True)
            if tree_hashes(staging) != before:
                raise PublishError("stored snapshot is corrupt; refusing to restore")
        journal.advance("STAGED")
        step("STAGED")
        if dest.exists():
            os.rename(dest, old)
        journal.advance("BACKED_UP")
        step("BACKED_UP")
        if before is not None:
            os.rename(staging, dest)
        _fsync_dir(dest.parent)
        journal.advance("SWAPPED")
        step("SWAPPED")
        restored = tree_hashes(dest) if dest.exists() else None
        if restored != before:
            raise PublishError("restored package does not match the pre-install snapshot")
        if old.exists():
            shutil.rmtree(old)
        journal.advance("ROLLED_BACK")
        self._set_state(install_id, "ROLLED_BACK")
        if prev:
            self._set_state(prev, "COMMITTED")
        self.registry.transition(
            version_id,
            LifecycleState.ROLLED_BACK,
            actor,
            f"rolled back installation {install_id} (policy={policy})",
        )
        self._event(
            install_id,
            "rollback",
            {
                "policy": policy,
                "drift": [p for p, _ in drift],
                "restored_previous": before is not None,
            },
            actor,
        )
        return {
            "install_id": install_id,
            "destination": str(dest),
            "restored": "previous package"
            if before is not None
            else "removed (no prior package existed)",
            "byte_exact": restored == before,
            "policy": policy,
            "drift_detected": bool(drift),
        }

    def history(self, version_id: str | None = None) -> list[dict[str, Any]]:
        q = select(orm.InstallationRow).order_by(orm.InstallationRow.created_at)
        if version_id:
            q = q.where(orm.InstallationRow.version_id == version_id)
        with self.registry.db.tx() as s:
            out = []
            for r in s.scalars(q):
                events = s.scalars(
                    select(orm.InstallEvent)
                    .where(orm.InstallEvent.install_id == r.install_id)
                    .order_by(orm.InstallEvent.id)
                )
                out.append(
                    {
                        "install_id": r.install_id,
                        "version_id": r.version_id,
                        "destination": r.destination,
                        "state": r.state,
                        "publisher": r.publisher,
                        "actor": r.actor,
                        "decision_id": r.decision_id,
                        "previous_install_id": r.previous_install_id,
                        "created_at": r.created_at.isoformat(),
                        "events": [
                            {
                                "action": e.action,
                                "detail": e.detail,
                                "actor": e.actor,
                                "at": e.created_at.isoformat(),
                            }
                            for e in events
                        ],
                    }
                )
            return out
