"""CLI surface (JSON output) and the full offline demo (the CI acceptance path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import BUNDLE, SUITE
from skill_foundry.cli.app import app
from skill_foundry.demo import run_demo
from skill_foundry.sandbox.base import BwrapSandbox, DockerSandbox


def _sf(home: Path, *args: str) -> dict:  # type: ignore[type-arg]
    res = CliRunner().invoke(app, ["--home", str(home), "--workspace", "demo", *args, "--json"])
    assert res.exit_code == 0, res.output
    return json.loads(res.output)  # type: ignore[no-any-return]


def test_cli_json_flow(tmp_path: Path) -> None:
    home = tmp_path / "sf"
    assert _sf(home, "ingest", str(BUNDLE))["accepted"] == 25
    assert _sf(home, "index", "status")["healthy"] is True
    d = _sf(home, "discover", "--suite", str(SUITE))
    spec = next(s for s in d["specs"] if s["name"] == "fix-failing-pytest-suite")
    cv = _sf(home, "generate", spec["spec_hash"][:12], "--suite", str(SUITE))
    assert cv["state"] == "candidate"
    v = _sf(home, "validate", cv["version_id"], "--suite", str(SUITE))
    assert v["version"]["state"] == "validated"
    lin = _sf(home, "lineage", cv["version_id"])
    assert lin["steps"] and all(s["support"] for s in lin["steps"])
    plan = _sf(home, "publish", cv["version_id"], "--target", str(tmp_path))
    assert plan["applied"] is False
    res = CliRunner().invoke(
        app,
        ["--home", str(home), "publish", cv["version_id"], "--target", str(tmp_path), "--apply"],
    )
    assert res.exit_code != 0 and "no approval" in res.output.lower()


@pytest.mark.skipif(
    not (BwrapSandbox().available() or DockerSandbox().available()), reason="no sandbox backend"
)
def test_offline_demo_end_to_end(tmp_path: Path) -> None:
    assert run_demo(tmp_path / "demo")


def test_mcp_server_is_read_only(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import asyncio

    from conftest import settings_for
    from skill_foundry.mcp.server import build_server

    server = build_server(settings_for(tmp_path / "sf"))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"list_candidates", "list_hypotheses", "version_lineage", "evaluation_reports"}
    assert not any(w in n for n in names for w in ("approve", "publish", "install", "rollback"))
