"""Derived index: real hybrid behaviour, outbox reliability, idempotency, staleness, rebuild."""

from __future__ import annotations

import pytest

from conftest import BUNDLE, LATE
from skill_foundry.retrieval.bm25 import BM25Encoder, term_id
from skill_foundry.retrieval.embeddings import HashingEmbedder
from skill_foundry.retrieval.index import ExperienceIndex, IndexNotReady
from skill_foundry.service import Foundry


def test_bm25_weights_saturate_and_normalise_length() -> None:
    enc = BM25Encoder(avgdl=10)
    tid = term_id("pytest")

    def weight(text: str) -> float:
        idx, vals = enc.encode_doc(text)
        return dict(zip(idx, vals, strict=True))[tid]

    assert weight("pytest pytest") > weight("pytest pytest " + "filler " * 40)  # length norm
    assert weight("pytest " * 10) < 10 * weight("pytest")  # tf saturation


def test_hybrid_search_ranks_family_first(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    hits = foundry.index.search("demo", "pin third-party actions and restrict token permissions", 5)
    assert hits[0].experience_id.startswith("demo/gha-")
    hits = foundry.index.search("demo", "settings key renamed, startup fails validation", 5)
    assert hits[0].experience_id.startswith("demo/cfg-")


def test_sparse_only_query_uses_idf(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    # 'nose' is rare: BM25 should surface the nose counterexamples
    ids = [h.experience_id for h in foundry.index.search("demo", "nose", 3)]
    assert any("legacy" in i for i in ids)


def test_sync_is_idempotent(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    h1 = foundry.index.health("demo")
    foundry.index.rebuild()
    foundry.index.sync()
    h2 = foundry.index.health("demo")
    assert h1.index_count == h2.index_count == h2.sqlite_count and h2.healthy


def test_failed_indexing_is_visible_and_recoverable(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    foundry.ingest(LATE, sync_index=False)
    assert foundry.index.health("demo").pending == 2

    def boom(n: int) -> None:
        raise ConnectionError("down")

    res = foundry.index.sync(fault=boom)
    assert res.failed == 2 and res.processed == 0
    h = foundry.index.health("demo")
    assert not h.healthy and h.failed == 2
    with pytest.raises(IndexNotReady):
        foundry.discover(None, exploratory=True)
    assert foundry.registry.counts("demo")["experiences"] == 27  # nothing lost in SQLite
    foundry.index.sync()
    assert foundry.index.health("demo").healthy


def test_stale_config_detected(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    foundry.index.close()
    other = ExperienceIndex(foundry.settings, foundry.registry, HashingEmbedder(dim=128))
    try:
        h = other.health("demo")
        assert any("stale" in r for r in h.reasons)
        with pytest.raises(IndexNotReady):
            other.sync()
        other.rebuild()
        assert other.health("demo").healthy
    finally:
        other.close()
        foundry._index = None


def test_payload_contains_no_free_text(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    pts, _ = foundry.index.client.scroll("sf_experiences", limit=100, with_payload=True)
    keys = {k for p in pts for k in (p.payload or {})}
    assert keys == {"workspace", "experience_id", "content_hash", "outcome_status", "trust"}
