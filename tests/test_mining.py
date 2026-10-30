"""Discovery: anti-chaining, independence, held-out separation, classification."""

from __future__ import annotations

from typing import Any

from conftest import BUNDLE, SUITE
from skill_foundry.domain.models import Reusability
from skill_foundry.evaluation.suite import EvalSuite
from skill_foundry.mining.cluster import complete_linkage
from skill_foundry.mining.features import normalize_command, step_signature
from skill_foundry.mining.lineage import assign_splits, lineages
from skill_foundry.service import Foundry
from skill_foundry.util import jaccard, shingles


def test_complete_linkage_prevents_chaining() -> None:
    sims = {"a|b": 0.9, "b|c": 0.9}  # a and c are not similar at all
    groups = complete_linkage(["a", "b", "c"], sims, 0.5)
    assert ["a", "b", "c"] not in groups
    assert len(groups) == 2


def test_complete_linkage_merges_cliques() -> None:
    sims = {"a|b": 0.9, "a|c": 0.8, "b|c": 0.7, "c|d": 0.6}
    assert complete_linkage(["a", "b", "c", "d"], sims, 0.5)[0] == ["a", "b", "c"]


def test_command_normalisation() -> None:
    assert normalize_command("git log -p -n 5 -- billing/invoice.py") == "git log -p -n 5 -- <arg>"
    assert step_signature("Ran `actionlint .github/workflows/x.yml`") == "actionlint <arg>"
    assert step_signature("Fixed root cause: very specific detail") == "fixed root cause"


def test_discovery_on_corpus(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    d = foundry.discover(SUITE)
    by_member = {m: c for c in d["clusters"] for m in c.member_ids}
    pt = by_member["demo/pt-billing-1"]
    # near-duplicate re-runs collapse into one independence unit
    assert {"demo/pt-billing-2", "demo/pt-billing-3"} <= set(pt.member_ids)
    assert len(pt.units) == len(pt.member_ids) - 2
    assert pt.classification == Reusability.REUSABLE and pt.eligible_for_synthesis
    assert set(pt.counterexample_ids) == {"demo/pt-legacy-1", "demo/pt-legacy-2"}
    # families never merge; the bridge episode does not chain them
    assert by_member["demo/gha-web-1"].cluster_id != pt.cluster_id
    assert by_member["demo/cfg-orders-1"].cluster_id != pt.cluster_id
    assert by_member["demo/bridge-1"].member_ids == ["demo/bridge-1"]
    assert by_member["demo/bridge-1"].classification == Reusability.ONE_OFF
    assert by_member["demo/oneoff-1"].classification == Reusability.ONE_OFF
    # untrusted / quarantined experiences never become members
    for bad in ("demo/pt-notify-1", "demo/pt-search-1", "demo/pt-maps-1", "demo/gha-inj-1"):
        assert bad not in by_member
    # score is explainable
    assert {f.name for f in pt.evidence_score.factors} >= {"independent_lineages", "verified_ratio"}


def test_heldout_lineage_excluded_before_synthesis(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    d = foundry.discover(SUITE)
    assert "demo/pt-pricing-1" in d["heldout_excluded"]
    members = {m for c in d["clusters"] for m in c.member_ids}
    assert "demo/pt-pricing-1" not in members


def test_no_lineage_crosses_the_split(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    xs = foundry.registry.experiences("demo")
    lin = lineages(xs)
    suite = EvalSuite.load(SUITE)
    splits = assign_splits(xs, suite.heldout_cases(), lin)
    by_lineage: dict[str, set[str]] = {}
    for x in xs:
        by_lineage.setdefault(lin[x.experience_id], set()).add(splits[x.experience_id].split)
    assert all(len(v) == 1 for v in by_lineage.values())
    train = [x for x in xs if splits[x.experience_id].split == "train"]
    for c in suite.cases.values():
        for x in train:
            assert jaccard(shingles(x.task_summary), shingles(c.prompt)) < 0.5


def test_exploratory_mode_is_marked(foundry: Foundry) -> None:
    foundry.ingest(BUNDLE)
    d: dict[str, Any] = foundry.discover(SUITE, exploratory=True)
    one = next(c for c in d["clusters"] if c.member_ids == ["demo/oneoff-1"])
    assert one.eligible_for_synthesis and one.exploratory
    assert any("EXPLORATORY" in r for r in one.classification_reasons)
