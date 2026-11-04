"""Application service: the use cases behind the CLI, the demo and the MCP server.

Every state-changing operation is explicit. Nothing here installs, approves or
publishes implicitly; ``publish`` without ``apply=True`` only plans.
"""

from __future__ import annotations

import difflib
import json
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select

from skill_foundry import __version__
from skill_foundry.config import Settings
from skill_foundry.contracts.bundle_v1 import assert_valid, make_bundle
from skill_foundry.contracts.convert import experience_to_episode
from skill_foundry.domain.lifecycle import InvalidTransition
from skill_foundry.domain.models import (
    ArtifactManifest,
    CandidateVersion,
    CapabilityStatus,
    CheckStatus,
    Experience,
    FileEntry,
    LifecycleState,
    ProcedureSpec,
    ReviewDecision,
    TaskCluster,
)
from skill_foundry.domain.policy import NON_OVERRIDABLE, PromotionPolicy
from skill_foundry.evaluation.applicability import evaluate_applicability
from skill_foundry.evaluation.harness import CapabilityReport, run_capability
from skill_foundry.evaluation.runners import AgentRunner
from skill_foundry.evaluation.suite import EvalSuite
from skill_foundry.ingest.hubris import bundle_for, scan_tasks
from skill_foundry.ingest.labels import apply_label
from skill_foundry.ingest.pipeline import IngestReport, ingest_file
from skill_foundry.ingest.sanitize import redact_text
from skill_foundry.mining.cluster import ClusterConfig, discover
from skill_foundry.mining.lineage import assign_splits, lineages
from skill_foundry.procedure.diff import changelog_text, spec_diff
from skill_foundry.procedure.infer import infer_spec, spec_hash
from skill_foundry.publishing.installer import (
    PUBLISHERS,
    Installer,
    PublishError,
    plan,
    safe_destination,
)
from skill_foundry.registry import orm
from skill_foundry.registry.db import Database
from skill_foundry.registry.repo import NotFound, Registry
from skill_foundry.retrieval.embeddings import make_provider
from skill_foundry.retrieval.index import ExperienceIndex
from skill_foundry.sandbox.base import Sandbox
from skill_foundry.synthesis.leakage import LeakageGuard
from skill_foundry.synthesis.render import TEMPLATE_VERSION, render_package
from skill_foundry.synthesis.synthesizers import SkillSynthesizer, spec_payload
from skill_foundry.util import package_hash, sha256_bytes, sha256_text, tree_hashes, utcnow
from skill_foundry.validation.lint import lint_skill_dir
from skill_foundry.validation.spec import validate_skill_dir


class FoundryError(RuntimeError):
    pass


def _bump(prev: str | None, minor: bool) -> str:
    if prev is None:
        return "0.1.0"
    major, mi, patch = (int(x) for x in prev.split("."))
    return f"{major}.{mi + 1}.0" if minor else f"{major}.{mi}.{patch + 1}"


class Foundry:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.home.mkdir(parents=True, exist_ok=True)
        self.db = Database(settings.db_path)
        self.registry = Registry(self.db)
        self._index: ExperienceIndex | None = None
        self.installer = Installer(self.registry, settings)

    @property
    def index(self) -> ExperienceIndex:
        if self._index is None:
            self._index = ExperienceIndex(
                self.settings, self.registry, make_provider(self.settings.embedding)
            )
        return self._index

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
        self.db.close()

    @property
    def ws(self) -> str:
        return self.settings.workspace

    # ------------------------------------------------------------------ ingest
    def ingest(self, path: Path, *, sync_index: bool = True) -> IngestReport:
        report = ingest_file(path, self.registry, self.ws)
        if sync_index:
            self.index.sync()
        return report

    def import_hubris(self, repos: list[str], *, binary: str = "hubris") -> dict[str, Any]:
        """Ingest finished hubris managed tasks from explicitly allowlisted repositories."""
        known = {x.episode_id for x in self.registry.experiences(self.ws, include_quarantined=True)}
        scan = scan_tasks(repos, known, binary)
        out: dict[str, Any] = scan.to_dict()
        if not scan.episodes:
            return out
        bundle = bundle_for(scan.episodes)
        # The bundle is not sanitised until ingest, so it must not outlive this call.
        self.settings.home.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.settings.home) as tmp:
            path = Path(tmp) / f"{bundle['export_id']}.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            out["ingest"] = self.ingest(path).to_dict()
        return out

    def label(
        self, experience_id: str, key: str, value: str, author: str, reason: str
    ) -> Experience:
        x = apply_label(self.registry, experience_id, key, value, author, reason)
        self.index.sync()
        return x

    # ------------------------------------------------------------------ discover
    def discover(
        self,
        suite_path: Path | None,
        *,
        exploratory: bool = False,
        allow_stale: bool = False,
        cluster_cfg: ClusterConfig | None = None,
    ) -> dict[str, Any]:
        if not allow_stale:
            self.index.require_healthy(self.ws)
        xs = self.registry.experiences(self.ws)
        lin = lineages(xs)
        suite = EvalSuite.load(suite_path) if suite_path else None
        splits = assign_splits(xs, suite.heldout_cases() if suite else [], lin)
        train = [x for x in xs if splits[x.experience_id].split == "train"]
        heldout = {k: v.reason for k, v in splits.items() if v.split == "heldout"}
        cfg = cluster_cfg or ClusterConfig(exploratory=exploratory)
        clusters = discover(self.ws, train, self.index, lin, cfg)
        run_id = "d-" + sha256_text(utcnow().isoformat())[:12]
        self.registry.save_clusters(run_id, clusters)
        specs = []
        by_id = {x.experience_id: x for x in xs}
        for c in clusters:
            if not c.eligible_for_synthesis:
                continue
            spec = infer_spec(
                c, [by_id[i] for i in c.member_ids], [by_id[i] for i in c.counterexample_ids]
            )
            h = spec_hash(spec)
            self.registry.save_spec(spec, h)
            specs.append(
                {
                    "spec_hash": h,
                    "name": spec.name,
                    "cluster_id": c.cluster_id,
                    "steps": len(spec.steps),
                    "branches": len(spec.branches),
                }
            )
        return {
            "run_id": run_id,
            "experiences": len(xs),
            "train": len(train),
            "heldout_excluded": heldout,
            "suite": suite.manifest() if suite else None,
            "cluster_config": cfg.to_dict(),
            "index": {**self.index.config(), "mode": self.index.mode},
            "clusters": clusters,
            "specs": specs,
        }

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        spec_ref: str,
        synthesizer: SkillSynthesizer,
        *,
        suite_path: Path | None,
        actor: str = "cli",
    ) -> CandidateVersion:
        spec = self.registry.get_spec(spec_ref)
        s_hash = spec_hash(spec)
        guard = None
        suite = EvalSuite.load(suite_path) if suite_path else None
        if suite is not None:
            guard = LeakageGuard.from_texts(suite.leakage_texts())
            guard.check(spec_payload(spec), "synthesis input (procedure spec)")
        elif not spec.exploratory:
            raise FoundryError(
                "an evaluation suite is required so held-out tasks can be kept "
                "out of synthesis (pass --suite)"
            )
        out = synthesizer.synthesize(spec)
        if guard is not None:
            guard.check(out.raw_output, "synthesiser output")

        existing = self.registry.versions(spec.name)
        parent = existing[-1] if existing else None
        version = _bump(
            parent.version if parent else None, minor=parent is None or parent.spec_hash != s_hash
        )
        xs = self.registry.experiences(self.ws, ids=spec.source_experience_ids)
        lin_all = lineages(self.registry.experiences(self.ws))
        lineage = {e: lin_all[e] for e in spec.source_experience_ids if e in lin_all}
        files = render_package(
            spec,
            version=version,
            wording=out.wording,
            description=out.description,
            suggestions=out.suggestions,
            lineage=lineage,
        )
        if guard is not None:
            for rel, data in files.items():
                guard.check(data.decode(), f"rendered {rel}")
        file_hashes = {k: sha256_bytes(v) for k, v in files.items()}
        pkg = package_hash(file_hashes)
        qdir = self.quarantine_path(spec.name, version)
        if qdir.exists():
            raise FoundryError(f"quarantine directory {qdir} already exists")
        for rel, data in files.items():
            p = qdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if package_hash(tree_hashes(qdir)) != pkg:
            raise FoundryError("quarantine write verification failed")

        cluster = self.registry.get_cluster(spec.cluster_id)
        manifest = ArtifactManifest(
            skill_name=spec.name,
            version=version,
            package_sha256=pkg,
            files=[
                FileEntry(path=k, sha256=v, size=len(files[k]))
                for k, v in sorted(file_hashes.items())
            ],
            spec_hash=s_hash,
            spec_version=spec.spec_version,
            evidence_snapshot={x.experience_id: x.content_hash for x in xs},
            synthesizer={
                "provider": out.provider,
                "model": out.model,
                "config": out.config,
                "usage": out.usage,
            },
            template_version=TEMPLATE_VERSION,
            retrieval_config={**self.index.config(), "mode": self.index.mode},
            clustering_config={
                "config_hash": cluster.config_hash,
                "inference_config_hash": spec.inference_config_hash,
                "cluster_id": cluster.cluster_id,
            },
            eval_suite=suite.manifest() if suite else None,
            created_at=utcnow(),
            skill_foundry_version=__version__,
        )
        diff = spec_diff(self.registry.get_spec(parent.spec_hash) if parent else None, spec)
        cv = CandidateVersion(
            version_id=f"{spec.name}@{version}",
            name=spec.name,
            version=version,
            package_sha256=pkg,
            spec_hash=s_hash,
            state=LifecycleState.CANDIDATE,
            evidence_strength=spec.evidence_score.label,
            review_items=out.review_items,
            created_at=utcnow(),
        )
        self.registry.add_version(
            cv,
            files,
            manifest.model_dump(mode="json"),
            changelog_text(diff, version),
            parent.version_id if parent else None,
            actor,
        )
        with self.db.tx() as s:
            s.add(
                orm.SynthesisLog(
                    version_id=cv.version_id,
                    provider=out.provider,
                    model=out.model,
                    config=out.config,
                    input_sha256=sha256_text(out.raw_input),
                    output_sha256=sha256_text(out.raw_output),
                    input_redacted=redact_text(out.raw_input),
                    output_redacted=redact_text(out.raw_output),
                    created_at=utcnow(),
                    expires_at=utcnow()
                    + timedelta(days=self.settings.synthesis_log_retention_days),
                )
            )
        return cv

    def quarantine_path(self, name: str, version: str) -> Path:
        # The leaf directory must be named exactly like the skill (spec requirement).
        return self.settings.quarantine_dir / f"{name}@{version}" / name

    def package(self, version_id: str) -> dict[str, bytes]:
        return self.registry.package_files(version_id)

    def verify_quarantine(self, cv: CandidateVersion) -> bool:
        q = self.quarantine_path(cv.name, cv.version)
        return q.exists() and package_hash(tree_hashes(q)) == cv.package_sha256

    # ------------------------------------------------------------------ validate
    def validate(
        self, ref: str, *, suite_path: Path | None = None, actor: str = "cli"
    ) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        if not self.verify_quarantine(cv):
            raise FoundryError("quarantined package differs from the registered artifact")
        qdir = self.quarantine_path(cv.name, cv.version)
        spec = self.registry.get_spec(cv.spec_hash)
        guard = (
            LeakageGuard.from_texts(EvalSuite.load(suite_path).leakage_texts())
            if suite_path
            else None
        )
        training_specific = []
        for x in self.registry.experiences(self.ws, ids=spec.source_experience_ids):
            training_specific += [
                a.action_summary.split(":", 1)[1] for a in x.attempts if ":" in a.action_summary
            ]
        spec_r = validate_skill_dir(qdir)
        lint_r = lint_skill_dir(qdir, spec, leakage=guard, training_texts=training_specific)
        spec_status = CheckStatus.PASS if spec_r.passed else CheckStatus.FAIL
        warns = any(i.severity == "warning" for i in lint_r.issues)
        lint_status = (
            CheckStatus.FAIL
            if not lint_r.passed
            else CheckStatus.WARN
            if warns
            else CheckStatus.PASS
        )
        self.registry.add_check(
            cv.version_id, "spec", spec_status.value, spec_r.model_dump(mode="json")
        )
        self.registry.add_check(
            cv.version_id, "lint", lint_status.value, lint_r.model_dump(mode="json")
        )
        cv = self.registry.update_version(
            cv.version_id, spec_compliance=spec_status, lint_status=lint_status
        )
        if spec_r.passed and lint_r.passed and cv.state == LifecycleState.CANDIDATE:
            cv = self.registry.transition(
                cv.version_id,
                LifecycleState.VALIDATED,
                actor,
                "spec validation and content lint passed",
            )
        return {
            "version": cv,
            "spec": spec_r,
            "lint": lint_r,
            "note": "A valid SKILL.md is not evidence that the skill helps an agent.",
        }

    # ------------------------------------------------------------------ evaluate
    def eval_applicability(self, ref: str, suite_path: Path, track: str) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        suite = EvalSuite.load(suite_path)
        _, app_cases = suite.track(track)
        files = self.package(cv.version_id)
        spec = self.registry.get_spec(cv.spec_hash)
        quarantined = [
            x for x in self.registry.experiences(self.ws, include_quarantined=True) if x.quarantined
        ]
        rep = evaluate_applicability(files["SKILL.md"].decode(), spec, app_cases, quarantined)
        status = CheckStatus(rep["status"])
        run_id = "e-" + sha256_text(f"{cv.version_id}{utcnow().isoformat()}")[:12]
        manifest = {
            "suite": suite.manifest(),
            "track": track,
            "candidate_package_sha256": cv.package_sha256,
            "runner_kind": "static",
        }
        with self.db.tx() as s:
            s.add(
                orm.EvalRunRow(
                    run_id=run_id,
                    version_id=cv.version_id,
                    kind="applicability",
                    runner_kind="static",
                    status=status.value,
                    report=rep,
                    manifest=manifest,
                    created_at=utcnow(),
                )
            )
        self.registry.update_version(cv.version_id, applicability_status=status)
        return {"run_id": run_id, **rep}

    def eval_capability(
        self,
        ref: str,
        suite_path: Path,
        track: str,
        *,
        runner: AgentRunner,
        sandbox: Sandbox,
        repeats: int = 1,
        seed: int = 0,
        trigger: Literal["auto", "explicit"] = "auto",
    ) -> tuple[str, CapabilityReport]:
        cv = self.registry.get_version(ref)
        suite = EvalSuite.load(suite_path)
        files = self.package(cv.version_id)
        run_id = "e-" + sha256_text(f"{cv.version_id}{utcnow().isoformat()}")[:12]
        run_root = self.settings.runs_dir / run_id
        report = run_capability(
            suite,
            track,
            runner=runner,
            sandbox=sandbox,
            run_root=run_root,
            skill_name=cv.name,
            skill_files=files,
            repeats=repeats,
            seed=seed,
            trigger=trigger,
        )
        with self.db.tx() as s:
            s.add(
                orm.EvalRunRow(
                    run_id=run_id,
                    version_id=cv.version_id,
                    kind="capability",
                    runner_kind=runner.kind,
                    status=report.status.value,
                    report=report.to_dict(),
                    manifest=report.manifest,
                    created_at=utcnow(),
                )
            )
        # A fake/mock run never replaces a live measurement.
        current = cv.capability_status
        if runner.kind == "live" or current in (
            CapabilityStatus.NOT_MEASURED,
            CapabilityStatus.SMOKE_ONLY,
        ):
            self.registry.update_version(cv.version_id, capability_status=report.status)
        return run_id, report

    def eval_runs(self, ref: str) -> list[dict[str, Any]]:
        cv = self.registry.get_version(ref)
        with self.db.tx() as s:
            rows = s.scalars(
                select(orm.EvalRunRow)
                .where(orm.EvalRunRow.version_id == cv.version_id)
                .order_by(orm.EvalRunRow.created_at)
            )
            return [
                {
                    "run_id": r.run_id,
                    "kind": r.kind,
                    "runner_kind": r.runner_kind,
                    "status": r.status,
                    "report": r.report,
                    "manifest": r.manifest,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------ review
    def review(self, ref: str, acknowledged: list[str] | None = None) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        policy = PromotionPolicy().evaluate(cv, set(acknowledged or []))
        files = self.package(cv.version_id)
        parent = None
        with self.db.tx() as s:
            row = s.get(orm.VersionRow, cv.version_id)
            parent_id = row.parent_version_id if row else None
        if parent_id:
            parent = self.package(parent_id)
        diff = package_diff(parent or {}, files, "a" if parent else "/dev/null")
        return {
            "version": cv,
            "policy": policy,
            "changelog": self.registry.changelog(cv.version_id),
            "parent": parent_id,
            "diff": diff,
            "quarantine_intact": self.verify_quarantine(cv),
            "history": self.registry.history(cv.version_id),
            "evals": [{k: v for k, v in e.items() if k != "report"} for e in self.eval_runs(ref)],
        }

    def decide(
        self,
        ref: str,
        *,
        reviewer: str,
        rationale: str,
        approve: bool = True,
        override: bool = False,
        acknowledge: list[str] | None = None,
    ) -> ReviewDecision:
        if not reviewer.strip() or not rationale.strip():
            raise FoundryError("a named reviewer and a rationale are required")
        cv = self.registry.get_version(ref)
        if not self.verify_quarantine(cv):
            raise FoundryError("quarantined package was modified after generation; refusing")
        pol = PromotionPolicy().evaluate(cv, set(acknowledge or []))
        failed = [g.gate for g in pol.failed]
        if approve:
            hard = [g for g in failed if g in NON_OVERRIDABLE]
            if hard:
                raise FoundryError(f"cannot approve: non-overridable gates failed: {hard}")
            if failed and not override:
                raise FoundryError(
                    "promotion blocked by policy: "
                    + "; ".join(f"{g.gate} ({g.detail})" for g in pol.failed)
                    + ". Pass --override with a rationale to approve anyway (recorded)."
                )
            if cv.state != LifecycleState.VALIDATED:
                raise InvalidTransition(f"{cv.version_id} is {cv.state.value}; must be validated")
        decision = ReviewDecision(
            decision_id="rev-" + sha256_text(f"{cv.version_id}{utcnow().isoformat()}")[:12],
            version_id=cv.version_id,
            package_sha256=cv.package_sha256,
            reviewer=reviewer,
            decision="approve" if approve else "reject",
            rationale=rationale,
            override=bool(failed) and approve,
            overridden_gates=failed if approve else [],
            acknowledged_items=sorted(acknowledge or []),
            created_at=utcnow(),
        )
        with self.db.tx() as s:
            s.add(orm.ReviewRow(**decision.model_dump(), policy_result=pol.model_dump()))
        if approve:
            self.registry.transition(
                cv.version_id,
                LifecycleState.APPROVED,
                reviewer,
                ("OVERRIDDEN (not empirically proven): " if decision.override else "") + rationale,
            )
            if decision.override:
                self.registry.update_version(cv.version_id, overridden=True)
        else:
            self.registry.transition(
                cv.version_id, LifecycleState.DEPRECATED, reviewer, f"rejected: {rationale}"
            )
        return decision

    def approval_for(self, version_id: str, package_sha256: str) -> str:
        with self.db.tx() as s:
            row = s.scalars(
                select(orm.ReviewRow)
                .where(
                    orm.ReviewRow.version_id == version_id,
                    orm.ReviewRow.decision == "approve",
                    orm.ReviewRow.package_sha256 == package_sha256,
                )
                .order_by(orm.ReviewRow.created_at.desc())
            ).first()
            if row is None:
                raise PublishError(f"no approval bound to {version_id} with this content hash")
            return row.decision_id

    # ------------------------------------------------------------------ publish
    def publish(
        self,
        ref: str,
        target: Path,
        *,
        publisher: str = "claude-project",
        apply: bool = False,
        force_overwrite: bool = False,
        actor: str = "cli",
    ) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        pub = PUBLISHERS[publisher]
        files = self.package(cv.version_id)
        dest = safe_destination(target, pub, cv.name)
        p = plan(dest, files, self.installer.managed_install_at(dest))
        result: dict[str, Any] = {
            "version_id": cv.version_id,
            "package_sha256": cv.package_sha256,
            "publisher": pub.name,
            "publisher_note": pub.note,
            "plan": p.to_dict(),
            "applied": False,
            "overridden": cv.overridden,
        }
        if not apply:
            result["note"] = "DRY RUN: nothing was written. Re-run with --apply to install."
            return result
        if not self.verify_quarantine(cv):
            raise PublishError("quarantined package was modified after approval; refusing")
        decision_id = self.approval_for(cv.version_id, cv.package_sha256)
        inst = self.installer.install(
            version_id=cv.version_id,
            decision_id=decision_id,
            files=files,
            target_root=target,
            publisher=pub,
            actor=actor,
            force_overwrite=force_overwrite,
        )
        result["applied"] = True
        result["installation"] = inst.__dict__
        return result

    def rollback(
        self,
        install_id: str | None,
        *,
        version: str | None = None,
        policy: str = "refuse",
        actor: str = "cli",
    ) -> dict[str, Any]:
        if install_id is None:
            if version is None:
                raise FoundryError("pass an installation id or --version")
            cv = self.registry.get_version(version)
            live = [h for h in self.installer.history(cv.version_id) if h["state"] == "COMMITTED"]
            if not live:
                raise FoundryError(f"{cv.version_id} has no active installation")
            install_id = live[-1]["install_id"]
        return self.installer.rollback(install_id, actor=actor, policy=policy)

    # ------------------------------------------------------------------ inspect
    def lineage(self, ref: str) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        spec = self.registry.get_spec(cv.spec_hash)
        cluster = self.registry.get_cluster(spec.cluster_id)
        manifest = self.registry.manifest(cv.version_id)
        xs = {
            x.experience_id: x
            for x in self.registry.experiences(
                self.ws,
                include_quarantined=True,
                ids=[*spec.source_experience_ids, *spec.counterexample_ids],
            )
        }

        def ev(eid: str) -> dict[str, Any]:
            x = xs.get(eid)
            if x is None:
                return {"experience_id": eid, "missing": True}
            snap = manifest["evidence_snapshot"].get(eid)
            return {
                "experience_id": eid,
                "trust": x.trust.value,
                "outcome": x.outcome_status.value,
                "repo": x.environment.repo_id,
                "revision": x.environment.revision,
                "content_hash_matches_snapshot": snap == x.content_hash if snap else None,
                "source": x.provenance.source,
                "bundle": x.provenance.bundle_export_id,
                "evidence": [
                    {
                        "id": e.evidence_id,
                        "kind": e.kind,
                        "trust": e.trust.value,
                        "exit_code": e.exit_code,
                        "events": e.source_event_ids,
                        "acceptance": e.acceptance.criteria if e.acceptance else None,
                    }
                    for e in x.evidence
                ],
            }

        steps = [
            {
                "step": s.step_id,
                "action": s.action,
                "origin": s.origin,
                "lineages": s.support_units,
                "support": [ev(e) for e in s.support],
            }
            for s in spec.steps
        ]
        steps += [
            {
                "step": s.step_id,
                "action": f"[{b.condition}] {s.action}",
                "origin": s.origin,
                "lineages": s.support_units,
                "support": [ev(e) for e in s.support],
            }
            for b in spec.branches
            for s in b.steps
        ]
        with self.db.tx() as s:
            reviews = [
                {
                    "decision": r.decision,
                    "reviewer": r.reviewer,
                    "override": r.override,
                    "gates": r.overridden_gates,
                    "rationale": r.rationale,
                    "package_sha256": r.package_sha256,
                    "at": r.created_at.isoformat(),
                }
                for r in s.scalars(
                    select(orm.ReviewRow).where(orm.ReviewRow.version_id == cv.version_id)
                )
            ]
        return {
            "version": cv.model_dump(mode="json"),
            "cluster": {
                "cluster_id": cluster.cluster_id,
                "classification": cluster.classification.value,
                "units": len(cluster.units),
                "evidence_score": cluster.evidence_score.model_dump(),
            },
            "spec_hash": cv.spec_hash,
            "steps": steps,
            "verification": [v.model_dump() for v in spec.verification],
            "contraindications": [c.model_dump() for c in spec.contraindications],
            "counterexamples": [ev(e) for e in spec.counterexample_ids],
            "excluded_steps": [e.model_dump() for e in spec.excluded_steps],
            "evaluations": [
                {k: v for k, v in e.items() if k != "report"} for e in self.eval_runs(ref)
            ],
            "reviews": reviews,
            "installations": self.installer.history(cv.version_id),
            "manifest": manifest,
        }

    def version_diff(self, a: str, b: str) -> str:
        va, vb = self.registry.get_version(a), self.registry.get_version(b)
        return package_diff(self.package(va.version_id), self.package(vb.version_id))

    def export_experiences(self, out: Path) -> dict[str, Any]:
        xs = self.registry.experiences(self.ws)
        bundle = make_bundle(
            f"skill-foundry-export:{self.ws}:{utcnow().strftime('%Y%m%d')}",
            [experience_to_episode(x) for x in xs],
            exported_at=utcnow().isoformat(),
        )
        assert_valid(bundle)
        out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        return {"path": str(out), "episodes": len(xs), "schema": "experience-bundle.v1"}

    def export_version(self, ref: str, out: Path) -> dict[str, Any]:
        cv = self.registry.get_version(ref)
        dest = out / cv.name
        if dest.exists():
            raise FoundryError(f"{dest} exists; refusing to overwrite")
        for rel, data in self.package(cv.version_id).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (out / f"{cv.name}.lineage.json").write_text(
            json.dumps(self.lineage(ref), indent=2, default=str), encoding="utf-8"
        )
        return {
            "path": str(dest),
            "version": cv.version_id,
            "state": cv.state.value,
            "overridden": cv.overridden,
        }

    # ------------------------------------------------------------------ doctor
    def doctor(self) -> dict[str, Any]:
        from skill_foundry.sandbox.base import BwrapSandbox, DockerSandbox

        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, detail: str, required: bool = True) -> None:
            checks.append({"check": name, "ok": ok, "detail": detail, "required": required})

        add("python", sys.version_info >= (3, 12), sys.version.split()[0])
        add("registry_schema", True, f"schema v{self.db_schema()} ({self.settings.db_path})")
        integrity = self.db.integrity_check()
        add("registry_integrity", integrity == "ok", integrity)
        h = self.index.health(self.ws)
        add(
            "index_health",
            h.healthy,
            "; ".join(h.reasons)
            or f"{h.index_count} points match {h.sqlite_count} experiences ({self.index.mode})",
        )
        add(
            "embedding_provider",
            True,
            f"{self.index.provider.model_id} (semantic={self.index.provider.semantic})",
            required=False,
        )
        pend = self.installer.pending()
        add(
            "install_journals",
            not pend,
            "no interrupted installs"
            if not pend
            else f"interrupted: {[p['install_id'] for p in pend]} "
            "(run `sf publish --recover <id>`)",
        )
        bad_q = [v.version_id for v in self.registry.versions() if not self.verify_quarantine(v)]
        add(
            "quarantine_integrity",
            not bad_q,
            "all quarantined packages match the registry"
            if not bad_q
            else f"modified or missing: {bad_q}",
        )
        add("sandbox_bwrap", BwrapSandbox().available(), "bubblewrap usable", required=False)
        add(
            "sandbox_docker",
            DockerSandbox().available(),
            "docker + skill-foundry-verifier image",
            required=False,
        )
        add(
            "claude_cli",
            shutil.which("claude") is not None,
            "only used with --runner claude-code (opt-in, live)",
            required=False,
        )
        purged = self.purge_expired_logs()
        add("synthesis_log_retention", True, f"purged {purged} expired log bodies", required=False)
        return {"ok": all(c["ok"] for c in checks if c["required"]), "checks": checks}

    def db_schema(self) -> int | None:
        from skill_foundry.registry.db import schema_version

        return schema_version(self.db.engine)

    def purge_expired_logs(self) -> int:
        n = 0
        with self.db.tx() as s:
            for row in s.scalars(
                select(orm.SynthesisLog).where(
                    orm.SynthesisLog.expires_at < utcnow(),
                    orm.SynthesisLog.input_redacted.is_not(None),
                )
            ):
                row.input_redacted = row.output_redacted = None
                n += 1
        return n

    def clusters(self) -> list[TaskCluster]:
        return self.registry.clusters(self.ws)

    def spec(self, ref: str) -> ProcedureSpec:
        return self.registry.get_spec(ref)


def package_diff(old: dict[str, bytes], new: dict[str, bytes], old_label: str = "a") -> str:
    parts = []
    for rel in sorted(set(old) | set(new)):
        a = old.get(rel, b"").decode(errors="replace").splitlines(keepends=True)
        b = new.get(rel, b"").decode(errors="replace").splitlines(keepends=True)
        if a == b:
            continue
        parts.append(
            "".join(
                difflib.unified_diff(
                    a,
                    b,
                    fromfile=f"{old_label}/{rel}" if rel in old else "/dev/null",
                    tofile=f"b/{rel}" if rel in new else "/dev/null",
                )
            )
        )
    return "".join(parts)


__all__ = ["Foundry", "FoundryError", "NotFound", "package_diff"]
