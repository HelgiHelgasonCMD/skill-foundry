"""Docker sandbox backend (CI job builds docker/verifier.Dockerfile first)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import SUITE
from skill_foundry.evaluation.suite import EvalSuite
from skill_foundry.evaluation.verifiers import PytestVerifier, VerifyContext
from skill_foundry.sandbox.base import DockerSandbox, SandboxPolicy

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not DockerSandbox().available(), reason="docker image not available"),
]

PROBE = """
import json, socket
out = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); out["net"] = True
except OSError:
    out["net"] = False
try:
    open("/inputs/x", "w"); out["ro"] = False
except OSError:
    out["ro"] = True
open("/work/r.json", "w").write(json.dumps(out))
"""


def test_docker_denies_network_and_writes(tmp_path: Path) -> None:
    work, inputs = tmp_path / "w", tmp_path / "i"
    work.mkdir()
    inputs.mkdir()
    res = DockerSandbox().run(
        ["python", "-c", PROBE],
        SandboxPolicy(mounts_rw={"work": work}, mounts_ro={"inputs": inputs}, timeout_s=60),
    )
    assert res.exit_code == 0, res.stderr
    assert json.loads((work / "r.json").read_text()) == {"net": False, "ro": True}


def test_docker_runs_hidden_pytest_verifier() -> None:
    case = EvalSuite.load(SUITE).cases["pytest-slug-trailing"]
    v = PytestVerifier({"id": "h", "tests": "hidden:tests"})
    r = v.verify(VerifyContext(case.workspace, case.workspace, case.hidden, DockerSandbox()))
    assert not r.passed and r.tests_failed  # the unfixed workspace fails the hidden tests
