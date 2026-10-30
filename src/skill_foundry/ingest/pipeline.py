"""``sf ingest``: bundle/JSONL -> validated, sanitised, qualified, deduplicated experiences.

Canonical rows and their index-outbox entries are written in one transaction, so a
crash can never leave an experience that the index does not know it must process.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from skill_foundry.contracts.bundle_v1 import (
    BundleRejected,
    EpisodeRejection,
    make_bundle,
    validate_bundle,
)
from skill_foundry.contracts.convert import episode_to_experience
from skill_foundry.ingest.qualify import qualify
from skill_foundry.ingest.sanitize import (
    SANITIZER_VERSION,
    LimitExceeded,
    Limits,
    SanitizeReport,
    check_limits,
    sanitize,
)
from skill_foundry.registry import orm
from skill_foundry.registry.repo import Registry, enqueue_index, experience_row
from skill_foundry.util import canonical_json, sha256_bytes, sha256_json, utcnow


@dataclass
class IngestReport:
    batch_id: str
    workspace: str
    source: str
    format: str
    export_id: str | None
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    trust: dict[str, int] = field(default_factory=dict)
    redactions: int = 0
    dropped_fields: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(path: Path, limits: Limits) -> tuple[bytes, str, Any]:
    size = path.stat().st_size
    if size > limits.max_bundle_bytes:
        raise LimitExceeded(f"{path} is {size} bytes (limit {limits.max_bundle_bytes})")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
        return raw, "jsonl", items
    data = json.loads(text)
    if isinstance(data, dict) and "schema_version" in data:
        return raw, "experience-bundle", data
    if isinstance(data, list):
        return raw, "json-list", data
    raise BundleRejected(["unrecognised input: expected a bundle object, a list or JSONL"])


def ingest_file(
    path: Path,
    registry: Registry,
    workspace: str,
    *,
    limits: Limits | None = None,
    actor: str = "cli",
) -> IngestReport:
    lim = limits or Limits()
    raw, fmt, data = _read(path, lim)
    digest = sha256_bytes(raw)
    if fmt != "experience-bundle":
        # Generic input = a list of episodes in the canonical episode shape.
        data = make_bundle(f"generic:{digest[:16]}", data)
    if len(data.get("episodes", [])) > lim.max_episodes:
        raise LimitExceeded(f"more than {lim.max_episodes} episodes")

    parsed = validate_bundle(data)
    report = IngestReport(
        batch_id=f"b-{digest[:16]}-{utcnow().strftime('%Y%m%d%H%M%S%f')}",
        workspace=workspace,
        source=str(path),
        format=fmt,
        export_id=parsed.export_id,
        received=len(parsed.episodes) + len(parsed.rejected),
    )
    for rej in parsed.rejected:
        report.rejected.append(asdict(rej))

    staged = []
    for i, ep in enumerate(parsed.episodes):
        try:
            check_limits(ep, lim, f"$.episodes[{ep['episode_id']}]")
        except LimitExceeded as exc:
            report.rejected.append(asdict(EpisodeRejection(i, ep["episode_id"], [str(exc)])))
            continue
        san = SanitizeReport()
        clean = sanitize(ep, san)
        content_hash = sha256_json(clean)
        x = episode_to_experience(
            clean, workspace=workspace, export_id=parsed.export_id, content_hash=content_hash
        )
        x.provenance = x.provenance.model_copy(
            update={"bundle_sha256": digest, "sanitizer_version": SANITIZER_VERSION}
        )
        x.redactions = san.total_redactions
        x.ingested_at = utcnow()
        if san.injection_hits:
            x.quarantined = True
            x.quarantine_reasons = [f"possible prompt injection at {h}" for h in san.injection_hits]
        x = qualify(x)
        report.redactions += san.total_redactions
        report.dropped_fields += len(san.dropped_fields)
        staged.append(x)

    with registry.db.tx() as s:
        s.add(
            orm.IngestBatch(
                batch_id=report.batch_id,
                workspace=workspace,
                source=str(path),
                source_sha256=digest,
                export_id=parsed.export_id,
                format=fmt,
                stats={},
                rejections=report.rejected,
                created_at=utcnow(),
            )
        )
        s.flush()
        for x in staged:
            dup = s.scalar(
                select(orm.ExperienceRow.experience_id).where(
                    orm.ExperienceRow.workspace == workspace,
                    orm.ExperienceRow.content_hash == x.content_hash,
                )
            )
            if dup is not None:
                report.duplicates += 1
                continue
            existing = s.get(orm.ExperienceRow, x.experience_id)
            if existing is not None:
                report.rejected.append(
                    {
                        "index": None,
                        "episode_id": x.episode_id,
                        "reasons": ["episode_id already ingested with different content"],
                    }
                )
                continue
            s.add(experience_row(x, report.batch_id))
            if not x.quarantined:
                enqueue_index(s, x)
            else:
                report.quarantined.append(
                    {"experience_id": x.experience_id, "reasons": x.quarantine_reasons}
                )
            report.accepted += 1
            report.trust[x.trust.value] = report.trust.get(x.trust.value, 0) + 1
        batch = s.get(orm.IngestBatch, report.batch_id)
        assert batch is not None
        batch.stats = json.loads(
            canonical_json(
                {k: v for k, v in report.to_dict().items() if k not in ("rejected", "quarantined")}
            )
        )
        batch.rejections = report.rejected
    return report
