"""Evaluation suite loading. Held-out tasks + hidden verifier assets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from skill_foundry.mining.lineage import HeldoutCase
from skill_foundry.resources import EVAL_SUITE_SCHEMA, load_schema
from skill_foundry.util import package_hash, tree_hashes


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    lineage: str
    prompt: str
    workspace: Path
    hidden: Path | None
    verifiers: list[dict[str, Any]]
    limits: dict[str, Any]
    repo_id: str | None = None
    revision: str | None = None
    fake_scripts: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class ApplicabilityCase:
    case_id: str
    task: str
    environment: dict[str, Any]
    expected: str
    reason: str
    signals: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    path: Path
    suite_id: str
    corpus_version: str
    tracks: dict[str, dict[str, Any]]
    cases: dict[str, EvalCase]
    applicability: dict[str, ApplicabilityCase]
    content_hash: str

    @classmethod
    def load(cls, path: Path) -> EvalSuite:
        data = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(load_schema(EVAL_SUITE_SCHEMA)).iter_errors(data),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            raise ValueError(
                "invalid eval suite: "
                + "; ".join(f"{list(e.absolute_path)}: {e.message}" for e in errors[:5])
            )
        base = path.parent
        cases = {}
        for c in data["cases"]:
            cases[c["case_id"]] = EvalCase(
                case_id=c["case_id"],
                lineage=c["lineage"],
                prompt=c["prompt"],
                workspace=(base / c["workspace"]).resolve(),
                hidden=(base / c["hidden"]).resolve() if c.get("hidden") else None,
                verifiers=c["verifiers"],
                limits=c["limits"],
                repo_id=c.get("repo_id"),
                revision=c.get("revision"),
                fake_scripts={
                    k: (base / v).resolve() for k, v in c.get("fake_scripts", {}).items()
                },
            )
        apps = {
            a["case_id"]: ApplicabilityCase(
                case_id=a["case_id"],
                task=a["task"],
                environment=a["environment"],
                expected=a["expected"],
                reason=a["reason"],
                signals=a.get("signals", []),
            )
            for a in data["applicability_cases"]
        }
        for name, t in data["tracks"].items():
            missing = [c for c in t["capability_cases"] if c not in cases]
            missing += [c for c in t["applicability_cases"] if c not in apps]
            if missing:
                raise ValueError(f"track {name} references unknown cases {missing}")
        return cls(
            path=path.resolve(),
            suite_id=data["suite_id"],
            corpus_version=data["corpus_version"],
            tracks=data["tracks"],
            cases=cases,
            applicability=apps,
            content_hash=package_hash(tree_hashes(base)),
        )

    def track(self, name: str) -> tuple[list[EvalCase], list[ApplicabilityCase]]:
        if name not in self.tracks:
            raise KeyError(f"unknown track {name!r}; available: {sorted(self.tracks)}")
        t = self.tracks[name]
        return (
            [self.cases[c] for c in t["capability_cases"]],
            [self.applicability[c] for c in t["applicability_cases"]],
        )

    def heldout_cases(self) -> list[HeldoutCase]:
        out = [
            HeldoutCase(c.case_id, c.prompt, c.lineage, c.repo_id, c.revision)
            for c in self.cases.values()
        ]
        out += [
            HeldoutCase(a.case_id, a.task, None, None, None) for a in self.applicability.values()
        ]
        return out

    def leakage_texts(self) -> list[tuple[str, str]]:
        """Everything that must never reach a synthesiser or a generated skill."""
        items: list[tuple[str, str]] = []
        for c in self.cases.values():
            items.append((f"{c.case_id}:prompt", c.prompt))
            for d in [c.hidden, *c.fake_scripts.values()]:
                if d is None:
                    continue
                files = [d] if d.is_file() else sorted(p for p in d.rglob("*") if p.is_file())
                for f in files:
                    items.append((f"{c.case_id}:{f.name}", f.read_text(errors="replace")))
        return items

    def manifest(self) -> dict[str, str]:
        return {
            "suite_id": self.suite_id,
            "corpus_version": self.corpus_version,
            "content_hash": self.content_hash,
        }
