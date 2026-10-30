"""Shared fixtures. The expensive pipeline (ingest -> discover -> generate -> validate)
is built once per session and copied per test, so tests never share mutable state."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from skill_foundry.config import Settings
from skill_foundry.service import Foundry
from skill_foundry.synthesis.synthesizers import TemplateSynthesizer

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BUNDLE = EXAMPLES / "bundles" / "demo-bundle.v1.json"
LATE = EXAMPLES / "bundles" / "late-arrivals.jsonl"
SUITE = EXAMPLES / "eval" / "suite.json"
SKILL = "fix-failing-pytest-suite"
TRACK = "python-test-failure-triage"


def settings_for(home: Path, workspace: str = "demo") -> Settings:
    return Settings(home=home, workspace=workspace, embedding="hashing")


@pytest.fixture(scope="session")
def demo_bundle() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(BUNDLE.read_text())
    return data


@pytest.fixture
def bundle(demo_bundle: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(demo_bundle)


@pytest.fixture
def foundry(tmp_path: Path) -> Iterator[Foundry]:
    f = Foundry(settings_for(tmp_path / "sf"))
    yield f
    f.close()


@pytest.fixture(scope="session")
def pipeline_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    home = tmp_path_factory.mktemp("template") / "sf"
    f = Foundry(settings_for(home))
    f.ingest(BUNDLE)
    d = f.discover(SUITE)
    spec = next(s for s in d["specs"] if s["name"] == SKILL)
    cv = f.generate(spec["spec_hash"], TemplateSynthesizer(), suite_path=SUITE)
    f.validate(cv.version_id, suite_path=SUITE)
    f.eval_applicability(cv.version_id, SUITE, TRACK)
    f.close()
    return {"home": home, "version_id": cv.version_id, "spec_hash": spec["spec_hash"]}


@pytest.fixture
def pipeline(pipeline_template: dict[str, Any], tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A private copy of a validated candidate (state: validated)."""
    home = tmp_path / "sf"
    shutil.copytree(pipeline_template["home"], home)
    f = Foundry(settings_for(home))
    yield {**pipeline_template, "foundry": f, "home": home}
    f.close()


@pytest.fixture
def approved(pipeline: dict[str, Any]) -> dict[str, Any]:
    f: Foundry = pipeline["foundry"]
    dec = f.decide(
        pipeline["version_id"], reviewer="tester", rationale="test override", override=True
    )
    return {**pipeline, "decision": dec}


@pytest.fixture
def target(tmp_path: Path) -> Path:
    t = tmp_path / "project"
    (t / ".claude" / "skills" / "user-skill").mkdir(parents=True)
    (t / "README.md").write_text("project\n")
    (t / ".claude" / "skills" / "user-skill" / "SKILL.md").write_text("mine\n")
    return t
