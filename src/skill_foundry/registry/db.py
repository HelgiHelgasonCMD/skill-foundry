"""Engine setup, Alembic migrations, and schema-version guard."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, select, text
from sqlalchemy.orm import Session, sessionmaker

from skill_foundry.registry.orm import SCHEMA_VERSION, Meta

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


class SchemaVersionError(RuntimeError):
    pass


def _on_connect(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


def make_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    event.listen(engine, "connect", _on_connect)
    return engine


def alembic_config(engine: Engine) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", str(engine.url))
    cfg.attributes["engine"] = engine
    return cfg


def migrate(engine: Engine) -> None:
    """Upgrade to head, refusing databases written by a newer Skill Foundry."""
    with engine.connect() as conn:
        has_meta = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='sf_meta'")
        ).first()
        if has_meta:
            row = conn.execute(text("SELECT value FROM sf_meta WHERE key='schema_version'")).first()
            if row and int(row[0]) > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"registry schema v{row[0]} is newer than this Skill Foundry "
                    f"(supports v{SCHEMA_VERSION}); refusing to open"
                )
    command.upgrade(alembic_config(engine), "head")
    with Session(engine) as s, s.begin():
        s.merge(Meta(key="schema_version", value=str(SCHEMA_VERSION)))


def schema_version(engine: Engine) -> int | None:
    with Session(engine) as s:
        row = s.scalar(select(Meta.value).where(Meta.key == "schema_version"))
        return int(row) if row else None


class Database:
    def __init__(self, db_path: Path):
        self.path = db_path
        self.engine = make_engine(db_path)
        migrate(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def tx(self) -> Iterator[Session]:
        """One unit of work: commit on success, roll back on any exception."""
        with self._sessions() as s, s.begin():
            yield s

    def integrity_check(self) -> str:
        with self.engine.connect() as conn:
            return str(conn.execute(text("PRAGMA integrity_check")).scalar())

    def close(self) -> None:
        self.engine.dispose()
