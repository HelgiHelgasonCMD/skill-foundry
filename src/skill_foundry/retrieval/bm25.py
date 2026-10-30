"""BM25 sparse vectors for Qdrant.

Documents carry the BM25 term-frequency component
``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``; Qdrant applies the IDF
component server-side via ``Modifier.IDF``. Queries carry weight 1 per unique term.
Together this is Okapi BM25 over the indexed collection.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

from skill_foundry.util import tokens

TOKENIZER_VERSION = "bm25-tok-v2"


def terms(text: str) -> list[str]:
    """Tokens plus their sub-parts, so 'nose-based' also matches 'nose'."""
    out = []
    for t in tokens(text):
        out.append(t)
        parts = [p for p in re.split(r"[-_.]", t) if p and p != t]
        out.extend(parts)
    return out


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


class BM25Encoder:
    def __init__(self, avgdl: float, k1: float = 1.2, b: float = 0.75):
        self.avgdl = max(avgdl, 1.0)
        self.k1 = k1
        self.b = b

    @staticmethod
    def doc_len(text: str) -> int:
        return len(terms(text))

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = terms(text)
        dl = len(toks)
        tf = Counter(toks)
        weights: dict[int, float] = {}
        for t, f in tf.items():
            w = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            tid = term_id(t)
            weights[tid] = weights.get(tid, 0.0) + w
        idx = sorted(weights)
        return idx, [weights[i] for i in idx]

    @staticmethod
    def encode_query(text: str) -> tuple[list[int], list[float]]:
        idx = sorted({term_id(t) for t in terms(text)})
        return idx, [1.0] * len(idx)
