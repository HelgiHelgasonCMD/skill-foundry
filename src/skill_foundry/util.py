"""Small shared helpers: canonical hashing, tokenisation, time."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "after",
        "before",
        "when",
        "while",
        "which",
        "so",
        "not",
        "no",
        "do",
        "does",
        "did",
        "run",
        "ran",
        "using",
        "use",
        "used",
        "via",
        "our",
        "we",
        "i",
        "you",
        "they",
        "them",
        "their",
        "there",
        "here",
        "up",
        "out",
    ]
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")


def utcnow() -> datetime:
    return datetime.now(UTC)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    out = _TOKEN.findall(text.lower())
    return out if keep_stopwords else [t for t in out if t not in STOPWORDS]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def shingles(text: str, n: int = 3) -> set[str]:
    toks = tokens(text, keep_stopwords=True)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def tree_hashes(root: Path) -> dict[str, str]:
    """Relative POSIX path -> sha256 for every regular file under ``root``."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"symlink not allowed in package tree: {p}")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_bytes(p.read_bytes())
    return out


def package_hash(file_hashes: dict[str, str]) -> str:
    """Hash of a package = hash over its sorted (path, file hash) manifest."""
    return sha256_text("\n".join(f"{k}\0{v}" for k, v in sorted(file_hashes.items())))
