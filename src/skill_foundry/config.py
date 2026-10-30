"""Runtime settings. All state lives under one directory (default ``./.sf``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    home: Path
    workspace: str = "default"
    embedding: str = "hashing"  # hashing | fastembed
    qdrant_url: str | None = None
    synthesis_log_retention_days: int = 30
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.home / "registry.db"

    @property
    def quarantine_dir(self) -> Path:
        return self.home / "quarantine"

    @property
    def installations_dir(self) -> Path:
        return self.home / "installations"

    @property
    def qdrant_path(self) -> Path:
        return self.home / "qdrant"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @classmethod
    def from_env(cls, home: Path | None = None, workspace: str | None = None) -> Settings:
        h = home or Path(os.environ.get("SF_HOME", ".sf"))
        return cls(
            home=h.resolve(),
            workspace=workspace or os.environ.get("SF_WORKSPACE", "default"),
            embedding=os.environ.get("SF_EMBEDDING", "hashing"),
            qdrant_url=os.environ.get("SF_QDRANT_URL") or None,
        )
