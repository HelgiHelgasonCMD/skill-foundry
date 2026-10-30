"""Sandbox boundaries (deny by default). These are accident-reduction checks,
not proof of a security boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from skill_foundry.sandbox.base import (
    BwrapSandbox,
    SandboxPolicy,
    SandboxUnavailable,
    make_sandbox,
)

bwrap = pytest.mark.skipif(not BwrapSandbox().available(), reason="bubblewrap unavailable")

PROBE = """
import os, socket, json
out = {"home_listed": os.path.exists(os.path.expanduser("~/.ssh")),
       "env": sorted(os.environ), "real_home": os.path.exists("/root") and False}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); out["net"] = True
except OSError:
    out["net"] = False
try:
    open("/inputs/new.txt", "w"); out["ro"] = False
except OSError:
    out["ro"] = True
open("/work/result.json", "w").write(json.dumps(out))
"""


@bwrap
def test_bwrap_denies_network_home_and_writes(tmp_path: Path) -> None:
    import json

    work, inputs = tmp_path / "w", tmp_path / "i"
    work.mkdir()
    inputs.mkdir()
    pol = SandboxPolicy(mounts_rw={"work": work}, mounts_ro={"inputs": inputs}, timeout_s=30)
    res = BwrapSandbox().run(["python", "-c", PROBE], pol)
    assert res.exit_code == 0, res.stderr
    out = json.loads((work / "result.json").read_text())
    assert out["net"] is False and out["ro"] is True and out["home_listed"] is False
    assert not any(k.startswith(("ANTHROPIC", "AWS", "GITHUB", "SSH_AUTH")) for k in out["env"])
    assert str(Path.home()) not in res.stdout


@bwrap
def test_bwrap_timeout(tmp_path: Path) -> None:
    pol = SandboxPolicy(mounts_rw={"work": tmp_path}, timeout_s=1)
    res = BwrapSandbox().run(["python", "-c", "import time; time.sleep(10)"], pol)
    assert res.timed_out and not res.ok


@bwrap
def test_bwrap_memory_limit(tmp_path: Path) -> None:
    pol = SandboxPolicy(mounts_rw={"work": tmp_path}, memory_mb=256, timeout_s=30)
    res = BwrapSandbox().run(["python", "-c", "x = bytearray(1024 * 1024 * 1024)"], pol)
    assert res.exit_code != 0


def test_unsafe_host_requires_explicit_flag() -> None:
    with pytest.raises(SandboxUnavailable):
        make_sandbox("unsafe-host")
    assert make_sandbox("unsafe-host", allow_unsafe_host=True).name == "unsafe-host"


def test_policy_requires_writable_workdir(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SandboxPolicy(mounts_ro={"work": tmp_path})
