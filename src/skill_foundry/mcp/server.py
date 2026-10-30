"""Read-only MCP server (optional extra ``mcp``).

Exposes discovery results, version details/lineage and evaluation reports. There
are deliberately no tools that approve, publish, install or modify anything.
"""

from __future__ import annotations

import json
from typing import Any

from skill_foundry.config import Settings
from skill_foundry.service import Foundry


def _dump(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, indent=1, default=str)


def build_server(settings: Settings) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("install the 'mcp' extra: uv sync --extra mcp") from exc
    foundry = Foundry(settings)
    server = FastMCP("skill-foundry (read-only)")

    @server.tool()
    def list_candidates() -> str:
        """List candidate skill versions with lifecycle and evaluation status."""
        return _dump([v.model_dump(mode="json") for v in foundry.registry.versions()])

    @server.tool()
    def list_hypotheses() -> str:
        """List the latest discovered clusters (skill opportunities)."""
        return _dump([c.model_dump(mode="json") for c in foundry.clusters()])

    @server.tool()
    def version_lineage(version: str) -> str:
        """Evidence lineage for a version (name@x.y.z): why the skill was generated."""
        return _dump(foundry.lineage(version))

    @server.tool()
    def evaluation_reports(version: str) -> str:
        """Stored applicability/capability evaluation reports for a version."""
        return _dump(foundry.eval_runs(version))

    return server


def serve(settings: Settings) -> None:  # pragma: no cover - interactive transport
    build_server(settings).run()
