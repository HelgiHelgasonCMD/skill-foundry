"""Skill-opportunity discovery: hybrid retrieval + constrained complete-linkage grouping.

Algorithm (small, explicit and testable):

1. For every positive (verified, non-quarantined, training-split) experience, run
   the hybrid BM25 + dense query (RRF) inside its workspace. Only retrieved
   neighbours become candidate pairs.
2. Hard compatibility constraints decide whether a candidate pair may ever be
   grouped: same runtime family, compatible tool set, no major-version conflict.
3. Pair similarity is an explicit weighted mix of dense cosine, task-term Jaccard
   and step-signature Jaccard (all shown to reviewers).
4. Complete linkage: two groups merge only if *every* cross pair is a candidate
   pair with similarity >= threshold. This prevents A~B~C chaining.
5. Each group is checked for cohesion and contradictions, scored with a
   transparent rule-based evidence score, and classified as one-off,
   environment-specific, reusable or insufficient. Similarity alone never makes a
   procedure "reusable": independent lineages, repositories and environments do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from skill_foundry.domain.models import (
    Contradiction,
    EvidenceScore,
    Experience,
    IndependenceUnit,
    RejectedMember,
    Reusability,
    ScoreFactor,
    TaskCluster,
    TrustLevel,
    version_prefix,
)
from skill_foundry.mining.features import step_signatures, task_terms, tools
from skill_foundry.mining.lineage import independence_units
from skill_foundry.retrieval.embeddings import cosine
from skill_foundry.retrieval.index import ExperienceIndex
from skill_foundry.util import jaccard, sha256_json, sha256_text


@dataclass(frozen=True)
class ClusterConfig:
    neighbours: int = 8
    link_threshold: float = 0.45
    w_dense: float = 0.4
    w_task: float = 0.3
    w_steps: float = 0.3
    min_tool_jaccard: float = 0.2
    min_cohesion: float = 0.45
    counterexample_threshold: float = 0.35
    min_independent_units: int = 3
    min_repos: int = 2
    min_envs: int = 2
    exploratory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return sha256_json(self.to_dict())


def incompatibility(a: Experience, b: Experience, cfg: ClusterConfig) -> str | None:
    ra, rb = a.environment.runtime_family(), b.environment.runtime_family()
    if ra and rb and ra != rb:
        return f"runtime {ra} vs {rb}"
    ta, tb = tools(a), tools(b)
    if ta and tb and jaccard(ta, tb) < cfg.min_tool_jaccard:
        return f"tool sets differ ({sorted(ta)} vs {sorted(tb)})"
    va, vb = a.environment.versions, b.environment.versions
    for key in sorted(set(va) & set(vb)):
        if version_prefix(va[key], 1) != version_prefix(vb[key], 1):
            return f"{key} major version {va[key]} vs {vb[key]}"
    return None


@dataclass
class PairInfo:
    sim: float
    dense: float
    task: float
    steps: float


def pair_similarity(
    a: Experience, b: Experience, va: list[float], vb: list[float], cfg: ClusterConfig
) -> PairInfo:
    d = max(0.0, cosine(va, vb))
    t = jaccard(task_terms(a), task_terms(b))
    s = jaccard(step_signatures(a.attempts), step_signatures(b.attempts))
    return PairInfo(cfg.w_dense * d + cfg.w_task * t + cfg.w_steps * s, d, t, s)


def _key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def complete_linkage(ids: list[str], sims: dict[str, float], threshold: float) -> list[list[str]]:
    groups = [[i] for i in sorted(ids)]

    def link(g1: list[str], g2: list[str]) -> float:
        worst = 1.0
        for a in g1:
            for b in g2:
                s = sims.get(_key(a, b))
                if s is None:
                    return -1.0  # not a candidate pair (or incompatible): never merge
                worst = min(worst, s)
        return worst

    while True:
        best = (-1.0, -1, -1)
        for i, j in combinations(range(len(groups)), 2):
            score = link(groups[i], groups[j])
            if score > best[0]:
                best = (score, i, j)
        if best[0] < threshold:
            break
        _, i, j = best
        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]
    return sorted(groups, key=lambda g: (-len(g), g))


def evidence_score(
    members: list[Experience],
    units: list[IndependenceUnit],
    counterexamples: list[Experience],
    contradictions: list[Contradiction],
    cfg: ClusterConfig,
) -> EvidenceScore:
    repos = {u.repo_id for u in units if u.repo_id}
    envs = {u.env_fingerprint for u in units}
    kinds = {e.kind for m in members for e in m.evidence if e.trust == TrustLevel.VERIFIED}
    verified = [m for m in members if m.trust == TrustLevel.VERIFIED]
    raw = [
        (
            "independent_lineages",
            min(len(units) / 5, 1.0),
            0.35,
            f"{len(units)} independent task lineage(s) (saturates at 5)",
        ),
        (
            "distinct_repositories",
            min(len(repos) / 3, 1.0),
            0.15,
            f"{len(repos)} distinct repositories (saturates at 3)",
        ),
        (
            "distinct_environments",
            min(len(envs) / 3, 1.0),
            0.10,
            f"{len(envs)} distinct environment fingerprints (saturates at 3)",
        ),
        (
            "verified_ratio",
            len(verified) / max(len(members), 1),
            0.15,
            f"{len(verified)}/{len(members)} members acceptance-verified",
        ),
        (
            "check_kinds",
            min(len(kinds) / 2, 1.0),
            0.05,
            f"verified check kinds: {sorted(kinds) or 'none'}",
        ),
        (
            "counterexamples_known",
            1.0 if counterexamples else 0.5,
            0.10,
            f"{len(counterexamples)} counterexample(s) available to bound applicability"
            if counterexamples
            else "no counterexamples known: applicability boundary is untested",
        ),
        (
            "single_lineage_dominance",
            1.0 - max((len(u.experience_ids) for u in units), default=0) / max(len(members), 1)
            if len(units) > 1
            else 0.0,
            0.10,
            "share of members outside the largest lineage (re-runs do not add confidence)",
        ),
    ]
    factors = [
        ScoreFactor(
            name=n, value=round(v, 3), weight=w, contribution=round(v * w, 3), explanation=e
        )
        for n, v, w, e in raw
    ]
    penalty = 0.25 * len(contradictions)
    if contradictions:
        factors.append(
            ScoreFactor(
                name="contradictions",
                value=float(len(contradictions)),
                weight=-0.25,
                contribution=-round(penalty, 3),
                explanation=f"{len(contradictions)} contradiction(s) found",
            )
        )
    total = round(max(0.0, sum(f.contribution for f in factors)), 3)
    if len(units) < cfg.min_independent_units:
        label = "insufficient"
    elif total >= 0.75:
        label = "strong"
    elif total >= 0.55:
        label = "moderate"
    else:
        label = "weak"
    return EvidenceScore(total=total, label=label, factors=factors)


def detect_contradictions(
    members: list[Experience], counterexamples: list[Experience]
) -> list[Contradiction]:
    out: list[Contradiction] = []
    success_steps: dict[str, set[str]] = {}
    for m in members:
        for sig in step_signatures(m.attempts):
            success_steps.setdefault(sig.split(" #")[0], set()).add(m.experience_id)
    member_envs = {m.environment.fingerprint() for m in members}
    for c in counterexamples:
        if c.environment.fingerprint() not in member_envs:
            continue  # different environment: a contraindication, not a contradiction
        for sig in step_signatures(c.attempts, status="failed"):
            base = sig.split(" #")[0]
            if base in success_steps:
                out.append(
                    Contradiction(
                        kind="same_step_failed_in_same_environment",
                        detail=f"step '{base}' succeeded in members but failed in "
                        f"{c.experience_id} "
                        "with an identical environment fingerprint",
                        experience_ids=sorted(success_steps[base] | {c.experience_id}),
                    )
                )
    for a, b in combinations(members, 2):
        if jaccard(step_signatures(a.attempts), step_signatures(b.attempts)) < 0.15:
            out.append(
                Contradiction(
                    kind="divergent_procedures",
                    detail="two members solved the task with almost disjoint steps",
                    experience_ids=sorted([a.experience_id, b.experience_id]),
                )
            )
    return out


def classify(
    units: list[IndependenceUnit], contradictions: list[Contradiction], cfg: ClusterConfig
) -> tuple[Reusability, list[str]]:
    repos = {u.repo_id for u in units if u.repo_id}
    envs = {u.env_fingerprint for u in units}
    why = [f"{len(units)} independent lineage(s), {len(repos)} repo(s), {len(envs)} env(s)"]
    if len(units) == 1:
        return Reusability.ONE_OFF, [*why, "a single task lineage cannot show reuse"]
    if len(units) < cfg.min_independent_units:
        return Reusability.INSUFFICIENT, [
            *why,
            f"needs >= {cfg.min_independent_units} independent lineages",
        ]
    if len(envs) < cfg.min_envs:
        return Reusability.ENVIRONMENT_SPECIFIC, [*why, "all evidence from one environment"]
    if len(repos) < cfg.min_repos:
        return Reusability.ENVIRONMENT_SPECIFIC, [*why, "all evidence from one repository"]
    if contradictions:
        return Reusability.INSUFFICIENT, [*why, "unresolved contradictions"]
    return Reusability.REUSABLE, [*why, "meets independence thresholds"]


def discover(
    workspace: str,
    train: list[Experience],
    index: ExperienceIndex,
    lineage_of: dict[str, str],
    cfg: ClusterConfig,
) -> list[TaskCluster]:
    positives = {x.experience_id: x for x in train if x.is_positive}
    counter = {x.experience_id: x for x in train if x.is_counterexample}
    others = {
        x.experience_id: x
        for x in train
        if x.experience_id not in positives and x.experience_id not in counter
    }
    vecs = index.dense_vectors(sorted(positives) + sorted(counter))

    sims: dict[str, float] = {}
    pairinfo: dict[str, PairInfo] = {}
    near_rejects: dict[str, dict[str, str]] = {pid: {} for pid in positives}
    for pid, x in positives.items():
        for hit in index.search(workspace, x.text(), limit=cfg.neighbours):
            nid = hit.experience_id
            if nid == pid:
                continue
            if nid in others:
                o = others[nid]
                near_rejects[pid][nid] = (
                    f"trust={o.trust.value}: {'; '.join(o.trust_reasons[:2])}"
                    if o.trust != TrustLevel.VERIFIED
                    else f"outcome={o.outcome_status.value}"
                )
                continue
            if nid not in positives:
                continue
            k = _key(pid, nid)
            if k in sims or k in pairinfo:
                continue
            y = positives[nid]
            why = incompatibility(x, y, cfg)
            if why:
                near_rejects[pid][nid] = f"incompatible: {why}"
                continue
            if pid not in vecs or nid not in vecs:
                continue
            info = pair_similarity(x, y, vecs[pid], vecs[nid], cfg)
            pairinfo[k] = info
            sims[k] = info.sim

    groups = complete_linkage(sorted(positives), sims, cfg.link_threshold)
    clusters = []
    for g in groups:
        members = [positives[i] for i in g]
        member_set = set(g)
        units = independence_units(members, lineage_of)
        pw = {k: round(v, 3) for k, v in sims.items() if set(k.split("|")) <= member_set}
        vals = list(pw.values())
        min_pw = min(vals) if vals else 1.0
        mean_pw = sum(vals) / len(vals) if vals else 1.0
        rep = max(g, key=lambda i: (sum(sims.get(_key(i, j), 0) for j in g if j != i), i))

        rejected: dict[str, str] = {}
        for i in g:
            for nid, reason in near_rejects[i].items():
                if nid not in member_set:
                    rejected.setdefault(nid, reason)
            for j in positives:
                if j in member_set:
                    continue
                k = _key(i, j)
                # Only near-misses are worth a reviewer's attention.
                if k in sims and j not in rejected and sims[k] >= cfg.link_threshold / 2:
                    worst = min((sims.get(_key(j, m), -1.0) for m in g), default=-1.0)
                    if worst < cfg.link_threshold:
                        rejected[j] = (
                            f"complete-linkage: similarity to {i.split('/')[-1]} is {sims[k]:.2f} "
                            f"but its weakest link to this group is "
                            f"{'none' if worst < 0 else f'{worst:.2f}'} (< {cfg.link_threshold})"
                        )
        outliers = sorted(
            i
            for i in g
            if len(g) > 2 and min(sims.get(_key(i, j), 0.0) for j in g if j != i) < cfg.min_cohesion
        )

        cx: list[Experience] = []
        if rep in vecs:
            for cid, c in counter.items():
                if cid not in vecs:
                    continue
                cx_score = cfg.w_dense * max(
                    0.0, cosine(vecs[rep], vecs[cid])
                ) + cfg.w_task * jaccard(task_terms(positives[rep]), task_terms(c))
                if cx_score >= cfg.counterexample_threshold * (cfg.w_dense + cfg.w_task):
                    cx.append(c)
        contradictions = detect_contradictions(members, cx)
        score = evidence_score(members, units, cx, contradictions, cfg)
        cls, reasons = classify(units, contradictions, cfg)
        if min_pw < cfg.min_cohesion and len(g) > 1:
            reasons.append(f"cohesion {min_pw:.2f} below {cfg.min_cohesion}")
            cls = Reusability.INSUFFICIENT
        eligible = cls == Reusability.REUSABLE and score.label in ("moderate", "strong")
        if cfg.exploratory and not eligible and len(units) >= 1:
            eligible = True
            reasons.append("EXPLORATORY: thresholds relaxed; not evidence of reusability")
        clusters.append(
            TaskCluster(
                cluster_id="c-" + sha256_text("|".join(g))[:12],
                workspace=workspace,
                member_ids=g,
                representative_id=rep,
                min_pairwise=round(min_pw, 3),
                mean_pairwise=round(mean_pw, 3),
                pairwise=pw,
                outliers=outliers,
                rejected=[
                    RejectedMember(experience_id=k, reason=v) for k, v in sorted(rejected.items())
                ],
                counterexample_ids=sorted(c.experience_id for c in cx),
                contradictions=contradictions,
                units=units,
                classification=cls,
                classification_reasons=reasons,
                evidence_score=score,
                eligible_for_synthesis=eligible,
                exploratory=cfg.exploratory and cls != Reusability.REUSABLE,
                config_hash=cfg.hash(),
            )
        )
    return clusters
