"""Dense embedding providers.

``HashingEmbedder`` is deterministic and offline but **lexical, not semantic**: it
hashes word uni/bi-grams into a fixed-width vector. It exists so that tests and the
offline demo are reproducible without downloading a model. ``FastEmbedProvider``
(optional extra ``embeddings``) runs a real CPU sentence-embedding model.
"""

from __future__ import annotations

import itertools
import math
import zlib
from typing import Protocol

from skill_foundry.util import tokens


class EmbeddingProvider(Protocol):
    model_id: str
    dim: int
    semantic: bool

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    semantic = False

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_id = f"hashing-ngram-v1-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = tokens(text)
            grams = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
            for g in grams:
                h = zlib.crc32(g.encode())
                vec[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FastEmbedProvider:
    semantic = True

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "fastembed is not installed; `uv sync --extra embeddings` "
                "or use --embedding hashing"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        import fastembed

        self.model_id = f"fastembed:{model_name}@{fastembed.__version__}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def make_provider(name: str) -> EmbeddingProvider:
    if name == "hashing":
        return HashingEmbedder()
    if name == "fastembed":
        return FastEmbedProvider()
    raise ValueError(f"unknown embedding provider {name!r} (hashing | fastembed)")


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)
