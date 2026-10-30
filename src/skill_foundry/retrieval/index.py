"""Derived hybrid (dense + BM25) retrieval index over experiences, backed by Qdrant.

SQLite is authoritative; this index is rebuilt from it at any time. Indexing is
driven by the transactional ``index_outbox``: a row is marked ``done`` only after
Qdrant acknowledges the write, so an interrupted or failed sync leaves work
visibly pending rather than silently lost. Point IDs are deterministic
(``uuid5(experience_id)``), so re-processing a row is an idempotent upsert.

Workspace isolation is a *pre-retrieval* constraint: every query carries a
mandatory workspace filter in each prefetch and in the fused query.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sqlalchemy import func, select

from skill_foundry.config import Settings
from skill_foundry.domain.models import Experience
from skill_foundry.registry import orm
from skill_foundry.registry.repo import Registry
from skill_foundry.retrieval.bm25 import TOKENIZER_VERSION, BM25Encoder
from skill_foundry.retrieval.embeddings import EmbeddingProvider
from skill_foundry.util import sha256_json, utcnow

COLLECTION = "sf_experiences"
INDEX_SCHEMA = "experience-index.v1"
MAX_ATTEMPTS = 5
_NS = uuid.UUID("8c5e2a52-3b0f-4b8e-9d7e-5f0c1a9b7d11")


def point_id(experience_id: str) -> str:
    return str(uuid.uuid5(_NS, experience_id))


class IndexNotReady(RuntimeError):
    pass


@dataclass
class IndexHealth:
    workspace: str
    pending: int
    failed: int
    dead: int
    sqlite_count: int
    index_count: int
    config_hash: str
    stored_config_hash: str | None
    reasons: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "healthy": self.healthy}


@dataclass
class SyncResult:
    processed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Hit:
    experience_id: str
    score: float


class ExperienceIndex:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        provider: EmbeddingProvider,
        client: QdrantClient | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.provider = provider
        self.mode = "server" if settings.qdrant_url else "embedded"
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ config
    def config(self) -> dict[str, Any]:
        return {
            "schema": INDEX_SCHEMA,
            "collection": COLLECTION,
            "dense_model": self.provider.model_id,
            "dense_dim": self.provider.dim,
            "dense_semantic": self.provider.semantic,
            "distance": "cosine",
            "sparse": "bm25",
            "tokenizer": TOKENIZER_VERSION,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "fusion": "rrf",
        }

    def config_hash(self) -> str:
        return sha256_json(self.config())

    def _stored_meta(self) -> orm.IndexMetaRow | None:
        with self.registry.db.tx() as s:
            return s.get(orm.IndexMetaRow, COLLECTION)

    def _avgdl(self) -> float:
        meta = self._stored_meta()
        if meta and meta.config.get("avgdl"):
            return float(meta.config["avgdl"])
        return self._corpus_avgdl()

    def _corpus_avgdl(self) -> float:
        with self.registry.db.tx() as s:
            rows = s.scalars(
                select(orm.ExperienceRow.payload).where(orm.ExperienceRow.quarantined.is_(False))
            ).all()
        lens = [BM25Encoder.doc_len(Experience.model_validate(p).text()) for p in rows]
        return sum(lens) / len(lens) if lens else 50.0

    def _ensure_collection(self, avgdl: float) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config={
                    "dense": qm.VectorParams(size=self.provider.dim, distance=qm.Distance.COSINE)
                },
                sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            # Server mode uses a keyword payload index for the mandatory workspace
            # filter. Embedded mode has no payload indexes (filters still apply, by
            # scan), so its performance characteristics differ from the server.
            if self.mode == "server":
                self.client.create_payload_index(
                    COLLECTION, "workspace", field_schema=qm.PayloadSchemaType.KEYWORD
                )
        with self.registry.db.tx() as s:
            meta = s.get(orm.IndexMetaRow, COLLECTION)
            if meta is None:
                s.add(
                    orm.IndexMetaRow(
                        collection=COLLECTION,
                        config={**self.config(), "avgdl": avgdl},
                        config_hash=self.config_hash(),
                        built_at=utcnow(),
                    )
                )

    # ------------------------------------------------------------------ writes
    def _point(self, x: Experience, enc: BM25Encoder) -> qm.PointStruct:
        text = x.text()
        idx, vals = enc.encode_doc(text)
        return qm.PointStruct(
            id=point_id(x.experience_id),
            vector={
                "dense": self.provider.embed([text])[0],
                "bm25": qm.SparseVector(indices=idx, values=vals),
            },
            # Only sanitised identifiers/labels are stored as payload: never raw text.
            payload={
                "workspace": x.workspace,
                "experience_id": x.experience_id,
                "content_hash": x.content_hash,
                "outcome_status": x.outcome_status.value,
                "trust": x.trust.value,
            },
        )

    def sync(self, fault: Callable[[int], None] | None = None) -> SyncResult:
        """Process pending/failed outbox rows. ``fault`` is a test hook."""
        if self._stored_meta() is not None and self._stored_meta_hash() != self.config_hash():
            raise IndexNotReady(
                "index configuration changed (embedding model/tokenizer); run `sf index rebuild`"
            )
        avgdl = self._avgdl()
        self._ensure_collection(avgdl)
        enc = BM25Encoder(avgdl)
        result = SyncResult()
        with self.registry.db.tx() as s:
            rows = s.scalars(
                select(orm.IndexOutbox)
                .where(
                    orm.IndexOutbox.status.in_(("pending", "failed")),
                    orm.IndexOutbox.attempts < MAX_ATTEMPTS,
                )
                .order_by(orm.IndexOutbox.id)
            ).all()
            work = [(r.id, r.experience_id, r.op) for r in rows]
        for n, (row_id, exp_id, op) in enumerate(work):
            try:
                if fault:
                    fault(n)
                if op == "delete":
                    self.client.delete(
                        COLLECTION, points_selector=qm.PointIdsList(points=[point_id(exp_id)])
                    )
                else:
                    x = self.registry.get_experience(exp_id)
                    self.client.upsert(COLLECTION, [self._point(x, enc)], wait=True)
            except Exception as exc:
                self.registry.mark_outbox([row_id], "failed", f"{type(exc).__name__}: {exc}")
                result.failed += 1
                result.errors.append(f"{exp_id}: {exc}")
                continue
            self.registry.mark_outbox([row_id], "done")
            result.processed += 1
        return result

    def _stored_meta_hash(self) -> str | None:
        meta = self._stored_meta()
        return meta.config_hash if meta else None

    def rebuild(self) -> SyncResult:
        """Drop and recreate the index from SQLite (the source of truth)."""
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        avgdl = self._corpus_avgdl()
        with self.registry.db.tx() as s:
            meta = s.get(orm.IndexMetaRow, COLLECTION)
            if meta is not None:
                s.delete(meta)
        self._ensure_collection(avgdl)
        enc = BM25Encoder(avgdl)
        with self.registry.db.tx() as s:
            payloads = s.scalars(
                select(orm.ExperienceRow.payload).where(orm.ExperienceRow.quarantined.is_(False))
            ).all()
        xs = [Experience.model_validate(p) for p in payloads]
        for i in range(0, len(xs), 64):
            self.client.upsert(COLLECTION, [self._point(x, enc) for x in xs[i : i + 64]], wait=True)
        with self.registry.db.tx() as s:
            ids = list(
                s.scalars(select(orm.IndexOutbox.id).where(orm.IndexOutbox.status != "done"))
            )
        if ids:
            self.registry.mark_outbox(ids, "done", "superseded by rebuild")
        return SyncResult(processed=len(xs))

    # ------------------------------------------------------------------ health
    def health(self, workspace: str) -> IndexHealth:
        with self.registry.db.tx() as s:

            def outbox(*conds: Any) -> int:
                return int(
                    s.scalar(
                        select(func.count())
                        .select_from(orm.IndexOutbox)
                        .where(orm.IndexOutbox.workspace == workspace, *conds)
                    )
                    or 0
                )

            pending = outbox(orm.IndexOutbox.status == "pending")
            failed = outbox(
                orm.IndexOutbox.status == "failed", orm.IndexOutbox.attempts < MAX_ATTEMPTS
            )
            dead = outbox(
                orm.IndexOutbox.status == "failed", orm.IndexOutbox.attempts >= MAX_ATTEMPTS
            )
            sqlite_count = int(
                s.scalar(
                    select(func.count())
                    .select_from(orm.ExperienceRow)
                    .where(
                        orm.ExperienceRow.workspace == workspace,
                        orm.ExperienceRow.quarantined.is_(False),
                    )
                )
                or 0
            )
        stored = self._stored_meta_hash()
        index_count = 0
        if self.client.collection_exists(COLLECTION):
            index_count = self.client.count(
                COLLECTION, count_filter=self._ws_filter(workspace), exact=True
            ).count
        h = IndexHealth(
            workspace=workspace,
            pending=pending,
            failed=failed,
            dead=dead,
            sqlite_count=sqlite_count,
            index_count=index_count,
            config_hash=self.config_hash(),
            stored_config_hash=stored,
        )
        if stored is None:
            h.reasons.append("index has never been built")
        elif stored != h.config_hash:
            h.reasons.append("stale: index built with a different embedding/tokenizer config")
        if pending:
            h.reasons.append(f"{pending} outbox row(s) pending")
        if failed:
            h.reasons.append(f"{failed} outbox row(s) failed (will retry)")
        if dead:
            h.reasons.append(f"{dead} outbox row(s) exceeded {MAX_ATTEMPTS} attempts")
        if index_count != sqlite_count:
            h.reasons.append(f"count mismatch: sqlite={sqlite_count} index={index_count}")
        return h

    def require_healthy(self, workspace: str) -> IndexHealth:
        h = self.health(workspace)
        if not h.healthy:
            raise IndexNotReady(
                "retrieval index not ready: "
                + "; ".join(h.reasons)
                + " (run `sf index sync` or `sf index rebuild`, or pass --allow-stale)"
            )
        return h

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _ws_filter(workspace: str, extra: list[qm.Condition] | None = None) -> qm.Filter:
        must: list[qm.Condition] = [
            qm.FieldCondition(key="workspace", match=qm.MatchValue(value=workspace))
        ]
        return qm.Filter(must=must + (extra or []))

    def search(
        self, workspace: str, text: str, limit: int = 10, outcome: str | None = None
    ) -> list[Hit]:
        if not workspace:
            raise ValueError("workspace is mandatory for retrieval")
        extra: list[qm.Condition] = []
        if outcome:
            extra.append(
                qm.FieldCondition(key="outcome_status", match=qm.MatchValue(value=outcome))
            )
        flt = self._ws_filter(workspace, extra)
        idx, vals = BM25Encoder.encode_query(text)
        res = self.client.query_points(
            COLLECTION,
            prefetch=[
                qm.Prefetch(
                    query=self.provider.embed([text])[0], using="dense", limit=limit * 3, filter=flt
                ),
                qm.Prefetch(
                    query=qm.SparseVector(indices=idx, values=vals),
                    using="bm25",
                    limit=limit * 3,
                    filter=flt,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        hits = []
        for p in res.points:
            payload = p.payload or {}
            if payload.get("workspace") != workspace:  # defence in depth
                continue
            hits.append(Hit(str(payload["experience_id"]), float(p.score)))
        return hits

    def dense_vectors(self, experience_ids: list[str]) -> dict[str, list[float]]:
        if not experience_ids:
            return {}
        pts = self.client.retrieve(
            COLLECTION, ids=[point_id(e) for e in experience_ids], with_vectors=["dense"]
        )
        out: dict[str, list[float]] = {}
        for p in pts:
            vec = p.vector
            if isinstance(vec, dict) and "dense" in vec and p.payload:
                out[str(p.payload["experience_id"])] = list(vec["dense"])  # type: ignore[arg-type]
        return out
