"""`sf demo`: the complete offline pipeline against a throwaway target project.

Everything runs locally and deterministically with the bundled synthetic corpus,
the deterministic template renderer and the scripted fake runner. The demo proves
pipeline correctness. It does not, and cannot, show that a skill improves a real
agent: that needs the live benchmark (docs/live-benchmark.md).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from skill_foundry.config import Settings
from skill_foundry.evaluation.runners import FakeScriptedRunner
from skill_foundry.publishing.installer import PUBLISHERS, PublishError
from skill_foundry.retrieval.index import IndexNotReady
from skill_foundry.sandbox.base import SandboxUnavailable, make_sandbox
from skill_foundry.service import Foundry, FoundryError
from skill_foundry.synthesis.synthesizers import TemplateSynthesizer
from skill_foundry.util import package_hash, tree_hashes

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TRACK = "python-test-failure-triage"
SKILL = "fix-failing-pytest-suite"


class _Crash(RuntimeError):
    pass


def _target_project(root: Path) -> Path:
    t = root / "throwaway-project"
    (t / ".claude" / "skills" / "my-own-skill").mkdir(parents=True)
    (t / "README.md").write_text("# Throwaway project for the Skill Foundry demo\n")
    (t / ".claude" / "skills" / "my-own-skill" / "SKILL.md").write_text(
        "---\nname: my-own-skill\ndescription: A user's hand-written skill. Use when testing "
        "that Skill Foundry preserves user files.\n---\n\n# Mine\n"
    )
    return t


def run_demo(
    workdir: Path | None, *, unsafe_host: bool = False, console: Console | None = None
) -> bool:
    con = console or Console(highlight=False)
    if not EXAMPLES.is_dir():
        con.print("[red]examples/ not found: run the demo from a source checkout[/red]")
        return False
    root = (workdir or Path(tempfile.mkdtemp(prefix="sf-demo-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        con.print(f"[red]{root} is not empty; the demo needs an empty directory[/red]")
        return False
    bundle = EXAMPLES / "bundles" / "demo-bundle.v1.json"
    late = EXAMPLES / "bundles" / "late-arrivals.jsonl"
    suite = EXAMPLES / "eval" / "suite.json"
    target = _target_project(root)
    target_before = tree_hashes(target)
    settings = Settings(home=root / "sf-home", workspace="demo", embedding="hashing")
    f = Foundry(settings)
    results: list[tuple[str, bool, str]] = []

    def step(n: int, title: str) -> None:
        con.rule(f"[bold cyan]{n}. {title}")

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        con.print(("[green]PASS[/green] " if ok else "[red]FAIL[/red] ") + escape(detail))

    try:
        con.print(
            Panel(
                f"state: {settings.home}\nthrowaway target: {target}\n"
                "embedding: hashing (deterministic, lexical; not semantic)\n"
                "synthesiser: template (renders the ProcedureSpec; no model)\n"
                "runner: fake-scripted (smoke test; NOT agent evidence)",
                title="Skill Foundry offline demo",
            )
        )

        step(1, "Import a canonical experience-bundle.v1")
        rep = f.ingest(bundle)
        con.print(
            f"received {rep.received}, accepted {rep.accepted}, rejected "
            f"{len(rep.rejected)}, quarantined {len(rep.quarantined)}"
        )
        for q in rep.quarantined:
            con.print(
                f"  [yellow]quarantined[/yellow] {q['experience_id']}: {escape(q['reasons'][0])}"
            )
        record(
            "import",
            rep.accepted == rep.received and len(rep.quarantined) == 1,
            "schema-valid bundle imported; injection-bearing episode quarantined",
        )

        step(2, "Evidence qualification and deduplication")
        t = Table("experience", "outcome", "trust", "why")
        for x in f.registry.experiences("demo"):
            if x.trust.value != "verified" or x.outcome_status.value != "verified_success":
                t.add_row(
                    x.episode_id,
                    x.outcome_status.value,
                    x.trust.value,
                    escape(" ".join(x.trust_reasons)[:110]),
                )
        con.print(t)
        again = f.ingest(bundle)
        record(
            "dedup",
            again.duplicates == rep.accepted and again.accepted == 0,
            f"re-import: {again.duplicates} duplicates, {again.accepted} new",
        )

        step(3, "Hybrid retrieval (BM25 + dense, RRF) and constrained grouping")
        d = f.discover(suite)
        for exp, why in d["heldout_excluded"].items():
            con.print(f"  [magenta]held out of synthesis[/magenta] {exp}: {escape(why)}")
        t = Table("cluster", "members", "lineages", "classification", "score", "eligible")
        for c in d["clusters"]:
            t.add_row(
                c.cluster_id,
                ", ".join(m.split("/")[-1] for m in c.member_ids),
                str(len(c.units)),
                c.classification.value,
                f"{c.evidence_score.total} ({c.evidence_score.label})",
                "yes" if c.eligible_for_synthesis else "no",
            )
        con.print(t)
        spec_ref = next(s for s in d["specs"] if s["name"] == SKILL)
        cluster = f.registry.get_cluster(spec_ref["cluster_id"])
        bridge_alone = any(c.member_ids == ["demo/bridge-1"] for c in d["clusters"])
        record(
            "grouping",
            bridge_alone and "demo/pt-pricing-1" in d["heldout_excluded"],
            "bridge episode not chained into a family; held-out lineage excluded",
        )

        step(4, "Inspectable discovery of a reusable procedure")
        con.print(
            f"cluster {cluster.cluster_id}: {cluster.classification.value}; "
            f"representative {cluster.representative_id}"
        )
        for u in cluster.units:
            con.print(
                f"  lineage {u.unit_id}: {', '.join(e.split('/')[-1] for e in u.experience_ids)}"
            )
        for r in cluster.rejected[:6]:
            con.print(f"  [yellow]rejected[/yellow] {r.experience_id}: {escape(r.reason[:100])}")
        for cx in cluster.counterexample_ids:
            con.print(f"  [red]counterexample[/red] {cx}")
        for fac in cluster.evidence_score.factors:
            con.print(f"  score {fac.name}: {fac.contribution} ({escape(fac.explanation)})")
        record(
            "discovery",
            len(cluster.units) < len(cluster.member_ids),
            f"{len(cluster.member_ids)} episodes counted as {len(cluster.units)} independent "
            "lineages (near-duplicates collapsed)",
        )

        step(5, "ProcedureSpec with per-step evidence references")
        spec = f.spec(spec_ref["spec_hash"])
        for s in spec.steps:
            con.print(
                f"  {s.step_id}. {escape(s.action)} [dim]<- {s.support_units}/"
                f"{len(cluster.units)} lineages[/dim]"
            )
        for b in spec.branches:
            con.print(f"  {b.branch_id}. if {escape(b.condition)} -> {escape(b.steps[0].action)}")
        for c in spec.contraindications:
            con.print(f"  do not use ({c.kind}): {escape(c.text)}")
        record(
            "spec",
            all(s.support for s in spec.steps),
            "every step cites supporting experiences; nothing is model-generated",
        )

        step(6, "Generate a compliant candidate skill in quarantine")
        cv = f.generate(spec_ref["spec_hash"], TemplateSynthesizer(), suite_path=suite)
        q = f.quarantine_path(cv.name, cv.version)
        con.print(f"{cv.version_id} -> {q}")
        con.print(Panel(escape((q / "SKILL.md").read_text()[:2500]), title="SKILL.md (head)"))
        record(
            "generate",
            not str(q).startswith(str(target)) and ".claude" not in q.parts,
            "candidate written to quarantine, not to any agent-discovered directory",
        )

        step(7, "Independent validation, security lint and applicability checks")
        v = f.validate(cv.version_id, suite_path=suite)
        for r in (v["spec"], v["lint"]):
            con.print(
                f"  {r.kind}: {'pass' if r.passed else 'FAIL'}"
                + (f" (+ {r.reference_validator})" if r.reference_validator else "")
                + "".join(f"\n    {i.severity}: {escape(i.message)}" for i in r.issues)
            )
        app_rep = f.eval_applicability(cv.version_id, suite, TRACK)
        for c in app_rep["cases"]:
            con.print(
                f"  applicability {c['case_id']}: expected {c['expected']}, got "
                f"{c['predicted']} ({escape(c['why'])})"
            )
        record(
            "validate",
            v["spec"].passed and v["lint"].passed and app_rep["status"] == "pass",
            f"spec+lint pass; applicability {app_rep['accuracy']} (static checks)",
        )

        step(8, "Sandboxed deterministic evaluation (SMOKE TEST)")
        try:
            sb = make_sandbox("auto")
        except SandboxUnavailable as exc:
            if not unsafe_host:
                con.print(f"[yellow]{exc}[/yellow]")
                record(
                    "smoke-eval",
                    False,
                    "no sandbox available (pass --unsafe-host to run trusted fixtures on the host)",
                )
                raise
            sb = make_sandbox("unsafe-host", allow_unsafe_host=True)
        run_id, cap = f.eval_capability(
            cv.version_id, suite, TRACK, runner=FakeScriptedRunner(sb), sandbox=sb
        )
        from skill_foundry.cli.app import print_capability

        print_capability(run_id, cap.to_dict())
        record(
            "smoke-eval",
            cap.status.value == "smoke_only",
            f"paired runs in {sb.name}; result labelled smoke_only (not capability evidence)",
        )

        step(9, "Review and immutable approval")
        rv = f.review(cv.version_id)
        for g in rv["policy"].gates:
            con.print(f"  gate {g.gate}: {'ok' if g.passed else 'BLOCKED'} ({escape(g.detail)})")
        blocked = False
        try:
            f.decide(cv.version_id, reviewer="demo-reviewer", rationale="looks good")
        except FoundryError as exc:
            blocked = True
            con.print(f"  [yellow]plain approval refused:[/yellow] {escape(str(exc)[:160])}")
        dec = f.decide(
            cv.version_id,
            reviewer="demo-reviewer",
            override=True,
            rationale="Demo only: approving without live evidence to exercise "
            "publication. Not empirically proven.",
        )
        con.print(
            f"  decision {dec.decision_id}: OVERRIDDEN gates {dec.overridden_gates}; "
            f"bound to package {dec.package_sha256[:16]}"
        )
        record(
            "approval",
            blocked and dec.override,
            "policy blocked promotion without live evidence; explicit audited override recorded",
        )

        step(10, "Publication dry run: exact destination and diff")
        dry = f.publish(cv.version_id, target)
        con.print(f"  destination {dry['plan']['destination']}")
        for c in dry["plan"]["changes"]:
            con.print(f"    {c['change']} {c['path']}")
        con.print(escape(dry["plan"]["diff"][:1200]))
        record(
            "dry-run",
            not dry["applied"] and tree_hashes(target) == target_before,
            "dry run wrote nothing",
        )

        step(11, "Explicit installation into the throwaway project")
        pub = f.publish(cv.version_id, target, apply=True, actor="demo-reviewer")
        inst = pub["installation"]
        installed = Path(inst["destination"])
        user_ok = (target / ".claude/skills/my-own-skill/SKILL.md").exists()
        record(
            "install",
            installed.joinpath("SKILL.md").exists()
            and user_ok
            and package_hash(tree_hashes(installed)) == cv.package_sha256,
            f"installed {inst['install_id']}; hash verified; user's own skill preserved",
        )

        step(12, "Byte-exact rollback")
        rb = f.rollback(inst["install_id"], actor="demo-reviewer")
        record(
            "rollback",
            rb["byte_exact"] and tree_hashes(target) == target_before,
            "target project is byte-identical to its pre-install state",
        )

        step(13, "Registry and index recovery checks")
        calls = {"n": 0}

        def flaky(n: int) -> None:
            calls["n"] += 1
            if n == 0:
                raise ConnectionError("simulated vector store outage")

        f.ingest(late, sync_index=False)
        res = f.index.sync(fault=flaky)
        h = f.index.health("demo")
        con.print(
            f"  after failed sync: processed={res.processed} failed={res.failed}; "
            f"health: {escape('; '.join(h.reasons))}"
        )
        refused = False
        try:
            f.discover(suite)
        except IndexNotReady as exc:
            refused = True
            con.print(f"  [yellow]discovery refused:[/yellow] {escape(str(exc)[:120])}")
        f.index.sync()
        f.index.rebuild()
        h2 = f.index.health("demo")
        con.print(
            f"  after retry + rebuild: healthy={h2.healthy} ({h2.index_count} points / "
            f"{h2.sqlite_count} experiences)"
        )
        x = f.registry.get_experience("demo/cfg-billing-1")
        redacted = "REDACTED" in " ".join(o.summary for o in x.observations)

        def crash(state: str) -> None:
            if state == "BACKED_UP":
                raise _Crash("simulated crash mid-install")

        try:
            f.installer.install(
                version_id=cv.version_id,
                decision_id=dec.decision_id,
                files=f.package(cv.version_id),
                target_root=target,
                publisher=PUBLISHERS["claude-project"],
                actor="demo",
                fault=crash,
            )
        except _Crash:
            pending = f.installer.pending()
            con.print(
                f"  interrupted install journals: {[p['install_id'] for p in pending]} "
                f"(state {pending[0]['state']})"
            )
            final = f.installer.recover(pending[0]["install_id"])
            con.print(f"  recovered -> {final}")
        doc = f.doctor()
        record(
            "recovery",
            refused
            and h2.healthy
            and redacted
            and doc["ok"]
            and tree_hashes(target) == target_before,
            "stale index refused then repaired; secret redacted at ingest; interrupted "
            "install recovered; doctor clean",
        )
    except (FoundryError, PublishError, SandboxUnavailable, IndexNotReady) as exc:
        con.print(f"[red]demo stopped: {escape(str(exc))}[/red]")
    finally:
        f.close()

    con.rule("[bold]Summary")
    t = Table("check", "result", "detail")
    for name, ok, detail in results:
        t.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", escape(detail))
    con.print(t)
    con.print(
        Panel(
            "Demonstrated: pipeline correctness with deterministic fixtures.\n"
            "NOT demonstrated: that this skill improves a real coding agent. Capability "
            "improvement is NOT MEASURED; the smoke eval replays scripted actions. See "
            "docs/live-benchmark.md.",
            title="What this shows",
        )
    )
    ok = len(results) == 13 and all(r[1] for r in results)
    con.print(f"demo state kept in {root}")
    return ok
