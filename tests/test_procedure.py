"""ProcedureSpec inference: provenance per step, determinism, nothing invented."""

from __future__ import annotations

from typing import Any

from skill_foundry.procedure.diff import spec_diff
from skill_foundry.procedure.infer import infer_spec, spec_hash


def test_every_step_has_support(pipeline: dict[str, Any]) -> None:
    spec = pipeline["foundry"].spec(pipeline["spec_hash"])
    assert spec.steps
    for s in [*spec.steps, *(st for b in spec.branches for st in b.steps)]:
        assert s.support and s.support_units >= 2 and s.origin == "observed_inference"
    for v in spec.verification:
        assert v.evidence_refs
    assert all(c.support for c in spec.contraindications)
    # anything excluded carries its reason and evidence
    assert all(e.reason and e.support for e in spec.excluded_steps)
    # producer lessons never become steps
    assert all("Reproduce with python -m pytest, isolate" not in s.action for s in spec.steps)


def test_spec_is_deterministic(pipeline: dict[str, Any]) -> None:
    f = pipeline["foundry"]
    spec = f.spec(pipeline["spec_hash"])
    cluster = f.registry.get_cluster(spec.cluster_id)
    xs = {x.experience_id: x for x in f.registry.experiences("demo")}
    again = infer_spec(
        cluster, [xs[i] for i in cluster.member_ids], [xs[i] for i in cluster.counterexample_ids]
    )
    assert spec_hash(again) == pipeline["spec_hash"]
    assert spec_diff(spec, again) == {"initial": False}


def test_branch_condition_from_observation(pipeline: dict[str, Any]) -> None:
    spec = pipeline["foundry"].spec(pipeline["spec_hash"])
    b = spec.branches[0]
    assert b.condition_basis == "observation" and "VersionMismatch" in b.condition


def test_recovery_and_failed_approaches(pipeline: dict[str, Any]) -> None:
    spec = pipeline["foundry"].spec(pipeline["spec_hash"])
    assert spec.recovery and spec.recovery[0].support == ["demo/pt-geo-1"]
    assert any("expected value in the test" in f.text for f in spec.failed_approaches)
