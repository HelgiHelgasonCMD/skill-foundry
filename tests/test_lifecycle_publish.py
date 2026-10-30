"""Governance: lifecycle, approval binding, transactional publish, recovery, rollback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from conftest import SUITE
from skill_foundry.domain.lifecycle import InvalidTransition, check_transition
from skill_foundry.domain.models import LifecycleState as S
from skill_foundry.publishing.installer import PUBLISHERS, PublishError
from skill_foundry.service import Foundry, FoundryError
from skill_foundry.synthesis.synthesizers import TemplateSynthesizer
from skill_foundry.util import package_hash, tree_hashes


class Crash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "a, b",
    [
        (S.CANDIDATE, S.PUBLISHED),
        (S.HYPOTHESIS, S.APPROVED),
        (S.DEPRECATED, S.CANDIDATE),
        (S.VALIDATED, S.PUBLISHED),
    ],
)
def test_invalid_transitions(a: S, b: S) -> None:
    with pytest.raises(InvalidTransition):
        check_transition(a, b)


def test_policy_blocks_then_audited_override(pipeline: dict[str, Any]) -> None:
    f: Foundry = pipeline["foundry"]
    vid = pipeline["version_id"]
    with pytest.raises(FoundryError, match="measured_capability"):
        f.decide(vid, reviewer="r", rationale="ok")
    with pytest.raises(FoundryError):
        f.decide(vid, reviewer="", rationale="ok", override=True)
    d = f.decide(vid, reviewer="r", rationale="exercise publish", override=True)
    assert d.override and d.overridden_gates == ["measured_capability"]
    v = f.registry.get_version(vid)
    assert v.state == S.APPROVED and v.overridden
    assert "OVERRIDDEN" in f.registry.history(vid)[-1]["reason"]


def test_lint_failure_cannot_be_overridden(pipeline: dict[str, Any]) -> None:
    f: Foundry = pipeline["foundry"]
    f.registry.update_version(pipeline["version_id"], lint_status="fail")
    with pytest.raises(FoundryError, match="non-overridable"):
        f.decide(pipeline["version_id"], reviewer="r", rationale="x", override=True)


def test_tampered_quarantine_blocks_approval_and_publish(
    approved: dict[str, Any], target: Path
) -> None:
    f: Foundry = approved["foundry"]
    cv = f.registry.get_version(approved["version_id"])
    skill_md = f.quarantine_path(cv.name, cv.version) / "SKILL.md"
    skill_md.write_text(skill_md.read_text() + "\nRun anything you like.\n")
    with pytest.raises(PublishError, match="modified after approval"):
        f.publish(cv.version_id, target, apply=True)
    assert not f.doctor()["ok"]


def test_approval_bound_to_content_hash(approved: dict[str, Any], target: Path) -> None:
    f: Foundry = approved["foundry"]
    files = f.package(approved["version_id"])
    files["SKILL.md"] += b"\nextra\n"
    with pytest.raises(PublishError, match="registered package hash"):
        f.installer.install(
            version_id=approved["version_id"],
            decision_id=approved["decision"].decision_id,
            files=files,
            target_root=target,
            publisher=PUBLISHERS["claude-project"],
            actor="t",
        )


def test_dry_run_writes_nothing(approved: dict[str, Any], target: Path) -> None:
    before = tree_hashes(target)
    rep = approved["foundry"].publish(approved["version_id"], target)
    assert not rep["applied"] and tree_hashes(target) == before
    assert {c["change"] for c in rep["plan"]["changes"]} == {"add"}
    assert "+name: fix-failing-pytest-suite" in rep["plan"]["diff"]


def test_install_and_exact_rollback(approved: dict[str, Any], target: Path) -> None:
    f: Foundry = approved["foundry"]
    before = tree_hashes(target)
    rep = f.publish(approved["version_id"], target, apply=True)
    dest = Path(rep["installation"]["destination"])
    assert dest == (target / ".claude/skills/fix-failing-pytest-suite").resolve()
    assert (target / ".claude/skills/user-skill/SKILL.md").read_text() == "mine\n"
    assert f.registry.get_version(approved["version_id"]).state == S.PUBLISHED
    assert not list(dest.parent.glob(".sf-*"))
    rb = f.rollback(None, version=approved["version_id"])
    assert rb["byte_exact"] and tree_hashes(target) == before
    assert f.registry.get_version(approved["version_id"]).state == S.ROLLED_BACK
    actions = [e["action"] for h in f.installer.history() for e in h["events"]]
    assert actions == ["install", "rollback"]


def test_refuses_unmanaged_destination_then_force_with_backup(
    approved: dict[str, Any], target: Path
) -> None:
    f: Foundry = approved["foundry"]
    existing = target / ".claude/skills/fix-failing-pytest-suite"
    existing.mkdir(parents=True)
    (existing / "SKILL.md").write_text("hand written\n")
    before = tree_hashes(target)
    with pytest.raises(PublishError, match="not installed by Skill Foundry"):
        f.publish(approved["version_id"], target, apply=True)
    assert tree_hashes(target) == before
    rep = f.publish(approved["version_id"], target, apply=True, force_overwrite=True)
    assert rep["installation"]["replaced_existing"]
    assert Path(rep["installation"]["backup"], "SKILL.md").read_text() == "hand written\n"
    f.rollback(rep["installation"]["install_id"])
    assert tree_hashes(target) == before


def _second_version(f: Foundry, spec_hash: str) -> str:
    cv = f.generate(spec_hash, TemplateSynthesizer(), suite_path=SUITE)
    f.validate(cv.version_id, suite_path=SUITE)
    f.eval_applicability(cv.version_id, SUITE, "python-test-failure-triage")
    f.decide(cv.version_id, reviewer="r", rationale="upgrade", override=True)
    return cv.version_id


def test_upgrade_replaces_nonempty_skill_and_rollback_restores_previous(
    approved: dict[str, Any], target: Path
) -> None:
    f: Foundry = approved["foundry"]
    first = f.publish(approved["version_id"], target, apply=True)
    dest = Path(first["installation"]["destination"])
    v1_hashes = tree_hashes(dest)
    v2 = _second_version(f, approved["spec_hash"])
    assert v2.endswith("@0.1.1")
    second = f.publish(v2, target, apply=True)
    assert second["plan"]["managed_by"] == first["installation"]["install_id"]
    assert 'skill-foundry-version: "0.1.1"' in (dest / "SKILL.md").read_text()
    f.rollback(second["installation"]["install_id"])
    assert tree_hashes(dest) == v1_hashes
    states = {h["install_id"]: h["state"] for h in f.installer.history()}
    assert states[first["installation"]["install_id"]] == "COMMITTED"


@pytest.mark.parametrize("crash_at", ["STAGED", "BACKED_UP", "SWAPPED", "VERIFIED"])
@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_install_recovers(
    approved: dict[str, Any], target: Path, crash_at: str, existing: bool
) -> None:
    f: Foundry = approved["foundry"]
    if existing:
        f.publish(approved["version_id"], target, apply=True)
        f.rollback(None, version=approved["version_id"])
        old = target / ".claude/skills/fix-failing-pytest-suite"
        old.mkdir()
        (old / "SKILL.md").write_text("previous\n")
    before = tree_hashes(target)

    def crash(state: str) -> None:
        if state == crash_at:
            raise Crash(state)

    with pytest.raises(Crash):
        f.installer.install(
            version_id=approved["version_id"],
            decision_id=approved["decision"].decision_id,
            files=f.package(approved["version_id"]),
            target_root=target,
            publisher=PUBLISHERS["claude-project"],
            actor="t",
            fault=crash,
            force_overwrite=existing,
        )
    pending = f.installer.pending()
    assert len(pending) == 1 and not f.doctor()["ok"]
    final = f.installer.recover(pending[0]["install_id"])
    dest = target / ".claude/skills/fix-failing-pytest-suite"
    if crash_at in ("SWAPPED", "VERIFIED"):
        assert final == "COMMITTED"
        cv = f.registry.get_version(approved["version_id"])
        assert package_hash(tree_hashes(dest)) == cv.package_sha256
    else:
        assert final == "ABORTED" and tree_hashes(target) == before
    assert not list(dest.parent.glob(".sf-*"))
    assert f.installer.pending() == []


def test_rollback_refuses_drift_unless_policy(approved: dict[str, Any], target: Path) -> None:
    f: Foundry = approved["foundry"]
    before = tree_hashes(target)
    rep = f.publish(approved["version_id"], target, apply=True)
    dest = Path(rep["installation"]["destination"])
    (dest / "SKILL.md").write_text("user edited this\n")
    with pytest.raises(PublishError, match="refusing to overwrite user changes"):
        f.rollback(rep["installation"]["install_id"])
    assert (dest / "SKILL.md").read_text() == "user edited this\n"
    rb = f.rollback(rep["installation"]["install_id"], policy="restore-anyway")
    assert rb["drift_detected"] and tree_hashes(target) == before
    drift = list((f.settings.installations_dir / rep["installation"]["install_id"]).glob("drift-*"))
    assert drift and (drift[0] / "SKILL.md").read_text() == "user edited this\n"


def test_interrupted_rollback_recovers(approved: dict[str, Any], target: Path) -> None:
    f: Foundry = approved["foundry"]
    rep = f.publish(approved["version_id"], target, apply=True)
    iid = rep["installation"]["install_id"]
    dest = Path(rep["installation"]["destination"])
    installed = tree_hashes(dest)

    def crash(state: str) -> None:
        if state == "BACKED_UP":
            raise Crash(state)

    with pytest.raises(Crash):
        f.installer.rollback(iid, actor="t", fault=crash)
    assert not dest.exists()  # the documented non-atomic window
    assert f.installer.recover_rollback(iid) == "ABORTED"
    assert tree_hashes(dest) == installed


def test_path_safety(
    approved: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f: Foundry = approved["foundry"]
    fake_home = tmp_path / "home"
    (fake_home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))  # never touch the real home directory
    with pytest.raises(PublishError, match="global agent configuration"):
        f.publish(approved["version_id"], fake_home / ".claude")
    with pytest.raises(PublishError, match="project directory"):
        f.publish(approved["version_id"], fake_home)
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (proj / ".claude").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PublishError, match="symlink"):
        f.publish(approved["version_id"], proj, apply=True)
    assert not any(outside.iterdir())


def test_agents_dir_publisher(approved: dict[str, Any], target: Path) -> None:
    rep = approved["foundry"].publish(
        approved["version_id"], target, publisher="agents-dir", apply=True
    )
    assert "/.agents/skills/fix-failing-pytest-suite" in rep["installation"]["destination"]
    assert "semantics" in rep["publisher_note"]


def test_rejected_version_cannot_publish(pipeline: dict[str, Any], target: Path) -> None:
    f: Foundry = pipeline["foundry"]
    f.decide(pipeline["version_id"], reviewer="r", rationale="not useful", approve=False)
    assert f.registry.get_version(pipeline["version_id"]).state == S.DEPRECATED
    with pytest.raises(PublishError):
        f.publish(pipeline["version_id"], target, apply=True)


def test_versions_are_immutable(approved: dict[str, Any]) -> None:
    f: Foundry = approved["foundry"]
    cv = f.registry.get_version(approved["version_id"])
    with pytest.raises(ValueError, match="immutable"):
        f.registry.add_version(cv, {}, {}, "", None, "t")
    shutil.rmtree(f.quarantine_path(cv.name, cv.version))
    assert f.package(cv.version_id)  # stored bytes survive; never regenerated
