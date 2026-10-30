"""Held-out leakage guard (defence in depth).

The primary guarantee is structural: held-out cases are excluded by task lineage
before synthesis (``mining.lineage.assign_splits``) and the synthesiser input
builder only receives a ProcedureSpec derived from training-split experiences.
This guard additionally refuses any synthesis input or output that shares long
word n-grams with held-out prompts, hidden verifier files or expected outputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from skill_foundry.util import tokens


class LeakageError(RuntimeError):
    pass


def ngrams(text: str, n: int) -> set[str]:
    toks = tokens(text, keep_stopwords=True)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class LeakageGuard:
    forbidden: dict[str, set[str]]  # source label -> n-grams
    n: int = 8

    @classmethod
    def from_texts(cls, items: Iterable[tuple[str, str]], n: int = 8) -> LeakageGuard:
        return cls({label: ngrams(text, n) for label, text in items}, n)

    def findings(self, text: str) -> list[str]:
        grams = ngrams(text, self.n)
        out = []
        for label, forbidden in sorted(self.forbidden.items()):
            hit = grams & forbidden
            if hit:
                out.append(f"{label}: {len(hit)} shared {self.n}-gram(s), e.g. '{sorted(hit)[0]}'")
        return out

    def check(self, text: str, where: str) -> None:
        found = self.findings(text)
        if found:
            raise LeakageError(f"held-out leakage in {where}: " + "; ".join(found))
