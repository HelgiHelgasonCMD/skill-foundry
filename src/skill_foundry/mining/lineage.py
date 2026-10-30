"""Task lineage, independence units and train/held-out separation.

Two episodes share a *lineage* when they are (or derive from) the same task:
producer-declared lineage, the same repository snapshot (repo + revision),
near-duplicate task text within one repository, or a lesson that cites both.
Independence is counted in lineages, so three re-runs of one task are one
confirmation, not three.

Held-out evaluation cases are separated *by lineage before synthesis*: any
experience in the same lineage as an evaluation case, from the same task
snapshot, or a near-duplicate of its prompt is excluded from synthesis. Textual
n-gram leakage checks elsewhere are defence in depth, not the primary guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from skill_foundry.domain.models import Experience, IndependenceUnit
from skill_foundry.util import jaccard, sha256_text, shingles

NEAR_DUP_TASK = 0.6  # shingle Jaccard on task summaries within a repository
HELDOUT_NEAR_DUP = 0.5  # shingle Jaccard between a case prompt and an experience task


class _UF:
    def __init__(self, items: list[str]):
        self.p = {i: i for i in items}

    def find(self, a: str) -> str:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.p[hi] = lo


def lineages(xs: list[Experience]) -> dict[str, str]:
    """experience_id -> lineage key (deterministic)."""
    ids = [x.experience_id for x in xs]
    uf = _UF(ids)
    by_decl: dict[str, str] = {}
    by_snapshot: dict[tuple[str, str], str] = {}
    by_episode = {x.episode_id: x.experience_id for x in xs}
    for x in xs:
        if x.task_lineage:
            if x.task_lineage in by_decl:
                uf.union(x.experience_id, by_decl[x.task_lineage])
            by_decl.setdefault(x.task_lineage, x.experience_id)
        env = x.environment
        if env.repo_id and env.revision:
            key = (env.repo_id, env.revision)
            if key in by_snapshot:
                uf.union(x.experience_id, by_snapshot[key])
            by_snapshot.setdefault(key, x.experience_id)
        for lesson in x.lessons:
            for src in lesson.source_episode_ids:
                if src in by_episode:
                    uf.union(x.experience_id, by_episode[src])
    sh = {x.experience_id: shingles(x.task_summary) for x in xs}
    for i, a in enumerate(xs):
        for b in xs[i + 1 :]:
            if (
                a.environment.repo_id
                and a.environment.repo_id == b.environment.repo_id
                and jaccard(sh[a.experience_id], sh[b.experience_id]) >= NEAR_DUP_TASK
            ):
                uf.union(a.experience_id, b.experience_id)
    return {i: "lin-" + sha256_text(uf.find(i))[:12] for i in ids}


def independence_units(xs: list[Experience], lineage_of: dict[str, str]) -> list[IndependenceUnit]:
    groups: dict[str, list[Experience]] = {}
    for x in xs:
        groups.setdefault(lineage_of[x.experience_id], []).append(x)
    units = []
    for lin, members in sorted(groups.items()):
        first = sorted(members, key=lambda m: m.experience_id)[0]
        units.append(
            IndependenceUnit(
                unit_id=lin,
                lineage=lin,
                repo_id=first.environment.repo_id,
                env_fingerprint=first.environment.fingerprint(),
                experience_ids=sorted(m.experience_id for m in members),
            )
        )
    return units


@dataclass(frozen=True)
class HeldoutCase:
    case_id: str
    prompt: str
    lineage: str | None
    repo_id: str | None
    revision: str | None


@dataclass(frozen=True)
class SplitDecision:
    split: str  # train | heldout
    reason: str


def assign_splits(
    xs: list[Experience], cases: list[HeldoutCase], lineage_of: dict[str, str]
) -> dict[str, SplitDecision]:
    out: dict[str, SplitDecision] = {}
    case_lineages = {c.lineage for c in cases if c.lineage}
    case_snapshots = {(c.repo_id, c.revision) for c in cases if c.repo_id and c.revision}
    case_shingles = [(c.case_id, shingles(c.prompt)) for c in cases]
    excluded_lineages: dict[str, str] = {}
    for x in xs:
        reason = None
        if x.task_lineage and x.task_lineage in case_lineages:
            reason = f"declared lineage {x.task_lineage} matches a held-out case"
        elif (x.environment.repo_id, x.environment.revision) in case_snapshots:
            reason = "same repository snapshot as a held-out case"
        else:
            sx = shingles(x.task_summary)
            for cid, cs in case_shingles:
                if jaccard(sx, cs) >= HELDOUT_NEAR_DUP:
                    reason = f"near-duplicate of held-out case {cid}"
                    break
        if reason:
            excluded_lineages[lineage_of[x.experience_id]] = reason
    for x in xs:
        lin = lineage_of[x.experience_id]
        if lin in excluded_lineages:
            out[x.experience_id] = SplitDecision("heldout", excluded_lineages[lin])
        else:
            out[x.experience_id] = SplitDecision("train", "no lineage overlap with held-out cases")
    return out
