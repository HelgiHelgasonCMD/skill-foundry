"""Alembic environment. Uses the engine passed in by ``registry.db`` when present."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from skill_foundry.registry.orm import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = config.attributes.get("engine") or create_engine(
        config.get_main_option("sqlalchemy.url") or "sqlite://"
    )
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
