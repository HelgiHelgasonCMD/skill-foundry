"""`sf` command-line interface. Every command supports ``--json`` for automation."""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from skill_foundry.config import Settings
from skill_foundry.service import Foundry

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=(
        "Skill Foundry: turn verified agent experience into evidence-backed, evaluated, governed "
        "Agent Skills. Generation, evaluation, approval and publication are separate "
        "explicit steps."
    ),
)
index_app = typer.Typer(no_args_is_help=True, help="Derived retrieval index (Qdrant).")
hyp_app = typer.Typer(no_args_is_help=True, help="Discovered clusters / skill hypotheses.")
spec_app = typer.Typer(no_args_is_help=True, help="ProcedureSpec intermediate representation.")
eval_app = typer.Typer(no_args_is_help=True, help="Applicability and capability evaluation.")
cand_app = typer.Typer(no_args_is_help=True, help="Candidate skill versions.")
export_app = typer.Typer(no_args_is_help=True, help="Export skills or experiences.")
fb_app = typer.Typer(no_args_is_help=True, help="Post-publication outcome feedback.")
import_app = typer.Typer(no_args_is_help=True, help="Import experience from other tools.")
app.add_typer(index_app, name="index")
app.add_typer(hyp_app, name="hypothesis")
app.add_typer(spec_app, name="spec")
app.add_typer(eval_app, name="eval")
app.add_typer(cand_app, name="candidates")
app.add_typer(export_app, name="export")
app.add_typer(fb_app, name="feedback")
app.add_typer(import_app, name="import")

console = Console(highlight=False, soft_wrap=True)
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")]
SuiteOpt = Annotated[Path | None, typer.Option("--suite", help="Evaluation suite JSON.")]
_state: dict[str, Any] = {}


@app.callback()
def _main(
    home: Annotated[
        Path | None, typer.Option(help="State directory (default ./.sf or $SF_HOME).")
    ] = None,
    workspace: Annotated[str | None, typer.Option(help="Workspace / project boundary.")] = None,
    embedding: Annotated[str | None, typer.Option(help="hashing | fastembed")] = None,
) -> None:
    s = Settings.from_env(home, workspace)
    if embedding:
        s = Settings(
            home=s.home, workspace=s.workspace, embedding=embedding, qdrant_url=s.qdrant_url
        )
    _state["settings"] = s


def foundry() -> Foundry:
    if "foundry" not in _state:
        _state["foundry"] = Foundry(_state["settings"])
        # Close before interpreter shutdown, or embedded Qdrant prints a traceback to stderr.
        atexit.register(_state["foundry"].close)
    f: Foundry = _state["foundry"]
    return f


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def emit(obj: Any, as_json: bool) -> bool:
    """Print JSON and return True when --json, else return False for human output."""
    if as_json:
        sys.stdout.write(json.dumps(_jsonable(obj), indent=2, default=str) + "\n")
    return as_json


def fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]error:[/red] {escape(msg)}")
    raise typer.Exit(code)


def _run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError, LookupError, OSError) as exc:
        fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------- basics
@app.command()
def doctor(as_json: JsonOpt = False) -> None:
    """Check environment, registry, index, journals and quarantine integrity."""
    rep = _run(foundry().doctor)
    if not emit(rep, as_json):
        t = Table("check", "ok", "detail")
        for c in rep["checks"]:
            mark = (
                "[green]ok[/green]"
                if c["ok"]
                else ("[red]FAIL[/red]" if c["required"] else "[yellow]n/a[/yellow]")
            )
            t.add_row(c["check"], mark, escape(c["detail"]))
        console.print(t)
    raise typer.Exit(0 if rep["ok"] else 1)


@app.command()
def ingest(path: Path, as_json: JsonOpt = False) -> None:
    """Import an experience-bundle.v1 file or generic JSON/JSONL episodes."""
    rep = _run(foundry().ingest, path)
    if not emit(rep, as_json):
        console.print(
            f"received {rep.received}, accepted {rep.accepted}, duplicates "
            f"{rep.duplicates}, rejected {len(rep.rejected)}, quarantined "
            f"{len(rep.quarantined)}, redactions {rep.redactions}"
        )
        console.print(f"trust: {rep.trust}")
        for r in rep.rejected:
            console.print(
                f"  [red]rejected[/red] {r['episode_id']}: {escape('; '.join(r['reasons']))}"
            )
        for q in rep.quarantined:
            console.print(
                f"  [yellow]quarantined[/yellow] {q['experience_id']}: {escape(q['reasons'][0])}"
            )


@app.command()
def label(
    experience_id: str,
    key: Annotated[
        str, typer.Option(help="outcome | attest-acceptance | quarantine | lineage | note")
    ],
    value: Annotated[str, typer.Option()],
    author: Annotated[str, typer.Option()],
    reason: Annotated[str, typer.Option()],
    as_json: JsonOpt = False,
) -> None:
    """Human correction/label of an experience (audited)."""
    x = _run(foundry().label, experience_id, key, value, author, reason)
    if not emit(x, as_json):
        console.print(f"{x.experience_id}: trust={x.trust.value} quarantined={x.quarantined}")


@app.command()
def experiences(
    as_json: JsonOpt = False,
    all_: Annotated[bool, typer.Option("--all", help="Include quarantined.")] = False,
) -> None:
    """List experiences in the workspace with their trust level."""
    xs = foundry().registry.experiences(foundry().ws, include_quarantined=all_)
    if not emit(
        [
            {
                "id": x.experience_id,
                "trust": x.trust.value,
                "outcome": x.outcome_status.value,
                "quarantined": x.quarantined,
                "reasons": x.trust_reasons,
            }
            for x in xs
        ],
        as_json,
    ):
        t = Table("experience", "outcome", "trust", "notes")
        for x in xs:
            t.add_row(
                x.experience_id,
                x.outcome_status.value,
                x.trust.value,
                escape("QUARANTINED" if x.quarantined else (x.trust_reasons or [""])[0]),
            )
        console.print(t)


# ---------------------------------------------------------------------------- index
@index_app.command("status")
def index_status(as_json: JsonOpt = False) -> None:
    h = foundry().index.health(foundry().ws)
    if not emit(h, as_json):
        console.print(
            f"healthy={h.healthy} sqlite={h.sqlite_count} index={h.index_count} "
            f"pending={h.pending} failed={h.failed} dead={h.dead}"
        )
        for r in h.reasons:
            console.print(f"  - {escape(r)}")


@index_app.command("sync")
def index_sync(as_json: JsonOpt = False) -> None:
    """Process pending index-outbox rows (idempotent)."""
    r = _run(foundry().index.sync)
    if not emit(r.__dict__, as_json):
        console.print(f"processed={r.processed} failed={r.failed}")


@index_app.command("rebuild")
def index_rebuild(as_json: JsonOpt = False) -> None:
    """Drop and rebuild the index from SQLite (the source of truth)."""
    r = _run(foundry().index.rebuild)
    if not emit(r.__dict__, as_json):
        console.print(f"rebuilt: {r.processed} experiences indexed")


# ---------------------------------------------------------------------------- hubris
RepoOpt = Annotated[
    list[str],
    typer.Option(
        "--repo",
        help="Repository to import from: name, owner/name or full id. Repeatable. "
        "Nothing is imported unless named.",
    ),
]
HubrisBinOpt = Annotated[str, typer.Option("--hubris-bin", help="The hubris executable.")]


@import_app.command("hubris")
def import_hubris_cmd(
    repo: RepoOpt, hubris_bin: HubrisBinOpt = "hubris", as_json: JsonOpt = False
) -> None:
    """Ingest finished hubris managed tasks from the named repositories."""
    rep = _run(foundry().import_hubris, repo, binary=hubris_bin)
    if not emit(rep, as_json):
        console.print(f"new episodes: {len(rep['new_episodes'])}")
        for key in ("already_imported", "not_allowlisted", "not_finished", "omitted_commands"):
            console.print(f"  {key}: {rep[key]}")
        console.print(f"  skipped_verdicts: {rep['skipped_verdicts']}")
        if rep["missing_transcripts"]:
            console.print(f"  [yellow]no transcript[/yellow]: {rep['missing_transcripts']}")
        if "ingest" in rep:
            console.print(f"  trust: {rep['ingest']['trust']}")


@app.command("autopilot")
def autopilot_cmd(
    repo: RepoOpt,
    suite: SuiteOpt = None,
    hubris_bin: HubrisBinOpt = "hubris",
    as_json: JsonOpt = False,
) -> None:
    """One unattended pass: import, index, discover, generate into quarantine, validate.

    Never approves, publishes or runs a live evaluation. Without --suite it stops at the
    ProcedureSpec. Runs SF_NOTIFY_CMD when something new needs you.
    """
    from skill_foundry.autopilot import run_pass

    rep = _run(
        run_pass,
        foundry(),
        repo,
        suite_path=suite,
        hubris_binary=hubris_bin,
        notify_command=os.environ.get("SF_NOTIFY_CMD"),
    )
    if not emit(rep, as_json):
        if "skipped" in rep:
            console.print(f"skipped: {rep['skipped']}")
            return
        console.print(
            f"imported {len(rep['import']['new_episodes'])} task(s); "
            f"discovery {'ran' if rep['discovered'] else 'not needed'}"
        )
        for c in rep["new_candidates"]:
            console.print(f"  candidate in quarantine: {c['ref']} (run `sf review {c['ref']}`)")
        for name in rep["needs_suite"]:
            console.print(f"  spec ready, needs an evaluation suite to generate: {name}")
        for f in rep["failed"]:
            console.print(f"  [red]failed[/red] {f['spec']}: {escape(f['error'])}")


# ---------------------------------------------------------------------------- discover
@app.command("discover")
def discover_cmd(
    suite: SuiteOpt = None,
    exploratory: Annotated[bool, typer.Option(help="Relax evidence thresholds (marked).")] = False,
    allow_stale: Annotated[
        bool, typer.Option(help="Run even if the index is not healthy.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Cluster verified experiences and infer ProcedureSpecs for eligible clusters."""
    rep = _run(foundry().discover, suite, exploratory=exploratory, allow_stale=allow_stale)
    if not emit(rep, as_json):
        print_clusters(rep["clusters"])
        for exp, why in rep["heldout_excluded"].items():
            console.print(f"  [magenta]held-out[/magenta] {exp}: {escape(why)}")
        for s in rep["specs"]:
            console.print(
                f"spec {s['spec_hash'][:12]} name={s['name']} steps={s['steps']} "
                f"branches={s['branches']}"
            )


def print_clusters(clusters: list[Any]) -> None:
    t = Table("cluster", "members", "lineages", "class", "score", "eligible", "min sim")
    for c in clusters:
        t.add_row(
            c.cluster_id,
            ", ".join(m.split("/")[-1] for m in c.member_ids),
            str(len(c.units)),
            c.classification.value,
            f"{c.evidence_score.total} {c.evidence_score.label}",
            "yes" if c.eligible_for_synthesis else "no",
            str(c.min_pairwise),
        )
    console.print(t)


@hyp_app.command("list")
def hyp_list(as_json: JsonOpt = False) -> None:
    cs = foundry().clusters()
    if not emit(cs, as_json):
        print_clusters(cs)


@hyp_app.command("show")
def hyp_show(cluster_id: str, as_json: JsonOpt = False) -> None:
    """Representatives, outliers, rejected members, counterexamples, score factors."""
    c = _run(foundry().registry.get_cluster, cluster_id)
    if emit(c, as_json):
        return
    console.print(
        f"[bold]{c.cluster_id}[/bold] {c.classification.value} "
        f"(eligible={c.eligible_for_synthesis})"
    )
    for r in c.classification_reasons:
        console.print(f"  reason: {escape(r)}")
    console.print(
        f"representative: {c.representative_id}; min/mean pairwise "
        f"{c.min_pairwise}/{c.mean_pairwise}"
    )
    for u in c.units:
        console.print(
            f"  lineage {u.unit_id} ({u.repo_id}, {u.env_fingerprint}): "
            f"{', '.join(u.experience_ids)}"
        )
    for r in c.rejected:
        console.print(f"  [yellow]rejected[/yellow] {r.experience_id}: {escape(r.reason)}")
    for x in c.counterexample_ids:
        console.print(f"  [red]counterexample[/red] {x}")
    for k in c.contradictions:
        console.print(f"  [red]contradiction[/red] {k.kind}: {escape(k.detail)}")
    t = Table("factor", "value", "weight", "contribution", "explanation")
    for f in c.evidence_score.factors:
        t.add_row(f.name, str(f.value), str(f.weight), str(f.contribution), escape(f.explanation))
    console.print(t)
    console.print(f"evidence score: {c.evidence_score.total} ({c.evidence_score.label})")


@spec_app.command("show")
def spec_show(ref: str, as_json: JsonOpt = False) -> None:
    s = _run(foundry().spec, ref)
    if emit(s, as_json):
        return
    print_spec(s)


def print_spec(s: Any) -> None:
    console.print(f"[bold]{s.name}[/bold]: {escape(s.trigger)}")
    for st in s.steps:
        console.print(
            f"  {st.step_id}. {escape(st.action)}  [dim]({st.support_units} lineages: "
            f"{', '.join(e.split('/')[-1] for e in st.support)})[/dim]"
        )
    for b in s.branches:
        console.print(
            f"  {b.branch_id} [{b.condition_basis}] {escape(b.condition)} -> "
            f"{escape(b.steps[0].action)}"
        )
    for v in s.verification:
        console.print(f"  verify: {escape(v.description)} ({len(v.evidence_refs)} evidence refs)")
    for r in s.recovery:
        console.print(f"  recover: {escape(r.when)} -> {escape(r.action)}")
    for c in s.contraindications:
        console.print(f"  do not use [{c.kind}]: {escape(c.text)}")
    for e in s.excluded_steps:
        console.print(f"  [dim]excluded: {escape(e.action)} ({escape(e.reason)})[/dim]")


@spec_app.command("diff")
def spec_diff_cmd(a: str, b: str, as_json: JsonOpt = False) -> None:
    from skill_foundry.procedure.diff import spec_diff

    d = spec_diff(foundry().spec(a), foundry().spec(b))
    if not emit(d, as_json):
        for k, v in d.items():
            console.print(f"{k}: {escape(str(v))}")


# ---------------------------------------------------------------------------- generate
@app.command()
def generate(
    spec_ref: str,
    suite: SuiteOpt = None,
    synthesizer: Annotated[str, typer.Option(help="template | anthropic")] = "template",
    model: Annotated[str | None, typer.Option(help="Model for --synthesizer anthropic.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Render a candidate skill into quarantine (never into an agent directory)."""
    from skill_foundry.synthesis.synthesizers import make_synthesizer

    synth = _run(make_synthesizer, synthesizer, model)
    cv = _run(foundry().generate, spec_ref, synth, suite_path=suite)
    if not emit(cv, as_json):
        q = foundry().quarantine_path(cv.name, cv.version)
        console.print(f"generated {cv.version_id} ({cv.state.value}) in quarantine: {q}")
        console.print(f"package sha256 {cv.package_sha256}")
        for i in cv.review_items:
            console.print(f"  [yellow]review item[/yellow] {i.item_id}: {escape(i.detail)}")


@app.command()
def validate(ref: str, suite: SuiteOpt = None, as_json: JsonOpt = False) -> None:
    """Spec validation (incl. reference validator) + content/security lint."""
    rep = _run(foundry().validate, ref, suite_path=suite)
    if emit(rep, as_json):
        return
    for r in (rep["spec"], rep["lint"]):
        console.print(
            f"{r.kind}: {'[green]pass[/green]' if r.passed else '[red]FAIL[/red]'}"
            + (f" (reference: {r.reference_validator})" if r.reference_validator else "")
        )
        for i in r.issues:
            console.print(f"  {i.severity} {i.code}: {escape(i.message)}")
    console.print(f"state: {rep['version'].state.value}. {rep['note']}")


# ---------------------------------------------------------------------------- eval
@eval_app.command("applicability")
def eval_applicability(
    ref: str,
    suite: Annotated[Path, typer.Option()],
    track: Annotated[str, typer.Option()],
    as_json: JsonOpt = False,
) -> None:
    """Static trigger/contraindication/safety checks against held-out task descriptions."""
    rep = _run(foundry().eval_applicability, ref, suite, track)
    if emit(rep, as_json):
        return
    t = Table("case", "expected", "predicted", "ok", "why")
    for c in rep["cases"]:
        t.add_row(
            c["case_id"],
            c["expected"],
            c["predicted"],
            "yes" if c["passed"] else "[red]NO[/red]",
            escape(c["why"]),
        )
    console.print(t)
    for s in rep["safety"]:
        console.print(
            f"  safety {s['check']}: {'ok' if s['passed'] else 'FAIL'} ({escape(s['detail'])})"
        )
    console.print(f"status: {rep['status']} accuracy={rep['accuracy']}  [dim]{rep['method']}[/dim]")


@eval_app.command("capability")
def eval_capability(
    ref: str,
    suite: Annotated[Path, typer.Option()],
    track: Annotated[str, typer.Option()],
    runner: Annotated[str, typer.Option(help="fake | claude-code")] = "fake",
    sandbox: Annotated[str, typer.Option(help="auto | bwrap | docker | unsafe-host")] = "auto",
    unsafe_host: Annotated[
        bool,
        typer.Option("--unsafe-host", help="Allow running trusted fixtures directly on the host."),
    ] = False,
    repeats: int = 1,
    seed: int = 0,
    trigger: Annotated[str, typer.Option(help="auto | explicit (skill invocation)")] = "auto",
    model: Annotated[str | None, typer.Option(help="Model for --runner claude-code.")] = None,
    live_opt_in: Annotated[
        bool,
        typer.Option(
            "--i-understand-this-runs-a-live-agent",
            help="Required for --runner claude-code: runs the real CLI and uses your quota or "
            "API credit.",
        ),
    ] = False,
    pass_api_key: Annotated[
        bool, typer.Option(help="Pass ANTHROPIC_API_KEY into the sandbox.")
    ] = False,
    pass_oauth_token: Annotated[
        bool,
        typer.Option(
            help="Pass CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) into the sandbox, "
            "to run on a Claude subscription."
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Paired baseline-vs-candidate runs with independent verifiers."""
    from skill_foundry.evaluation.runners import AgentRunner, ClaudeCodeRunner, FakeScriptedRunner
    from skill_foundry.sandbox.base import make_sandbox

    sb = _run(make_sandbox, sandbox, allow_unsafe_host=unsafe_host)
    r: AgentRunner
    if runner == "fake":
        r = FakeScriptedRunner(sb)
    elif runner == "claude-code":
        if not live_opt_in:
            fail(
                "--runner claude-code executes the real Claude Code CLI; pass "
                "--i-understand-this-runs-a-live-agent to opt in"
            )
        if pass_oauth_token and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            fail(
                "--pass-oauth-token: CLAUDE_CODE_OAUTH_TOKEN is not set. "
                "Run `claude setup-token` and export the token it prints."
            )
        r = _run(
            ClaudeCodeRunner,
            model=model,
            pass_api_key=pass_api_key,
            pass_oauth_token=pass_oauth_token,
        )
    else:
        fail(f"unknown runner {runner}")
        return
    run_id, rep = _run(
        foundry().eval_capability,
        ref,
        suite,
        track,
        runner=r,
        sandbox=sb,
        repeats=repeats,
        seed=seed,
        trigger=trigger,
    )
    if not emit({"run_id": run_id, **rep.to_dict()}, as_json):
        print_capability(run_id, rep.to_dict())


def print_capability(run_id: str, rep: dict[str, Any]) -> None:
    t = Table(
        "case",
        "rep",
        "baseline",
        "candidate",
        "baseline failure",
        "candidate failure",
        title=f"capability run {run_id} (runner_kind={rep['runner_kind']})",
    )
    for p in rep["pairs"]:
        t.add_row(
            p["case_id"],
            str(p["repeat"]),
            "[green]pass[/green]" if p["baseline"]["success"] else "[red]fail[/red]",
            "[green]pass[/green]" if p["candidate"]["success"] else "[red]fail[/red]",
            escape(", ".join(p["baseline"]["critical_failed"]) or "-"),
            escape(", ".join(p["candidate"]["critical_failed"]) or "-"),
        )
    console.print(t)
    s = rep["summary"]
    for arm in ("baseline", "candidate"):
        a = s[arm]
        console.print(
            f"  {arm}: {a['successes']}/{s['pairs']} (Wilson95 {a['wilson95']}), "
            f"tool calls {a['mean_tool_calls']}, failed actions "
            f"{a['mean_failed_actions']}, time {a['mean_duration_s']}s, cost "
            f"{a['mean_cost_usd'] if a['mean_cost_usd'] is not None else 'n/a'}"
        )
    console.print(f"  discordant pairs: {s['discordant']}; p-value: {s['p_value']}")
    console.print(f"  status: [bold]{rep['status']}[/bold]")
    for n in rep["notes"]:
        console.print(f"  [yellow]note:[/yellow] {escape(n)}")


@app.command()
def compare(run_id: str, as_json: JsonOpt = False) -> None:
    """Show a stored capability run side by side."""
    from sqlalchemy import select

    from skill_foundry.registry import orm

    with foundry().db.tx() as s:
        row = s.scalars(select(orm.EvalRunRow).where(orm.EvalRunRow.run_id == run_id)).first()
        if row is None:
            fail(f"no eval run {run_id}")
            return
        rep = dict(row.report)
        kind = row.kind
    if emit(rep, as_json):
        return
    if kind == "capability":
        print_capability(run_id, rep)
    else:
        console.print_json(json.dumps(rep, default=str))


# ---------------------------------------------------------------------------- candidates
@cand_app.command("list")
def candidates_list(as_json: JsonOpt = False) -> None:
    vs = foundry().registry.versions()
    if not emit(vs, as_json):
        print_versions(vs)


def print_versions(vs: list[Any]) -> None:
    t = Table(
        "version", "state", "spec", "lint", "applicability", "capability", "evidence", "override"
    )
    for v in vs:
        t.add_row(
            v.version_id,
            v.state.value,
            v.spec_compliance.value,
            v.lint_status.value,
            v.applicability_status.value,
            v.capability_status.value,
            v.evidence_strength,
            "[red]OVERRIDDEN[/red]" if v.overridden else "",
        )
    console.print(t)


@app.command()
def versions(
    name: Annotated[str | None, typer.Argument()] = None, as_json: JsonOpt = False
) -> None:
    """Versions with lifecycle state and separately tracked validation/evaluation status."""
    vs = foundry().registry.versions(name)
    if not emit(vs, as_json):
        print_versions(vs)


@app.command()
def diff(a: str, b: str) -> None:
    """Unified diff between two stored candidate packages."""
    sys.stdout.write(_run(foundry().version_diff, a, b) or "(identical)\n")


@app.command()
def lineage(ref: str, as_json: JsonOpt = False) -> None:
    """Why this skill exists: steps -> experiences -> evidence -> source events."""
    rep = _run(foundry().lineage, ref)
    if emit(rep, as_json):
        return
    v = rep["version"]
    console.print(
        f"[bold]{v['version_id']}[/bold] state={v['state']} "
        f"package={v['package_sha256'][:16]} spec={rep['spec_hash'][:16]}"
    )
    c = rep["cluster"]
    console.print(
        f"cluster {c['cluster_id']} ({c['classification']}, {c['units']} lineages, "
        f"score {c['evidence_score']['total']} {c['evidence_score']['label']})"
    )
    for st in rep["steps"]:
        console.print(
            f"  step {st['step']} [{st['origin']}] {escape(st['action'])} "
            f"<- {st['lineages']} lineages"
        )
        for e in st["support"]:
            evs = ", ".join(
                f"{x['id']}({x['kind']},{x['trust']},events={x['events']})"
                for x in e.get("evidence", [])
            )
            console.print(
                f"     {e['experience_id']} trust={e.get('trust')} "
                f"snapshot_ok={e.get('content_hash_matches_snapshot')} :: {evs}"
            )
    for x in rep["counterexamples"]:
        console.print(f"  counterexample {x['experience_id']} ({x.get('outcome')})")
    for e in rep["evaluations"]:
        console.print(
            f"  eval {e['run_id']} {e['kind']} runner={e['runner_kind']} -> {e['status']}"
        )
    for r in rep["reviews"]:
        console.print(
            f"  review {r['decision']} by {r['reviewer']} override={r['override']} "
            f"gates={r['gates']}: {escape(r['rationale'])}"
        )
    for i in rep["installations"]:
        console.print(f"  install {i['install_id']} {i['state']} -> {i['destination']}")


@app.command()
def review(
    ref: str,
    ack: Annotated[list[str] | None, typer.Option(help="Review item ids to acknowledge.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Show diff vs parent version, policy gates, evaluations and history."""
    rep = _run(foundry().review, ref, ack)
    if emit(rep, as_json):
        return
    v = rep["version"]
    console.print(
        f"[bold]{v.version_id}[/bold] state={v.state.value} quarantine_intact="
        f"{rep['quarantine_intact']} parent={rep['parent']}"
    )
    console.print(f"changelog: {escape(rep['changelog'])}")
    t = Table("gate", "passed", "detail", title="promotion policy")
    for g in rep["policy"].gates:
        t.add_row(g.gate, "[green]yes[/green]" if g.passed else "[red]no[/red]", escape(g.detail))
    console.print(t)
    console.print(f"policy allows promotion: {rep['policy'].allowed}")
    for i in v.review_items:
        console.print(f"  review item {i.item_id}: {escape(i.detail)}")
    console.print("[dim]--- package diff ---[/dim]")
    sys.stdout.write(rep["diff"][:20000])


@app.command()
def approve(
    ref: str,
    reviewer: Annotated[str, typer.Option()],
    rationale: Annotated[str, typer.Option()],
    override: Annotated[
        bool,
        typer.Option(
            help="Approve despite failed overridable gates (recorded; shown as OVERRIDDEN)."
        ),
    ] = False,
    ack: Annotated[list[str] | None, typer.Option(help="Acknowledge review items.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Record a human approval bound to the exact package hash."""
    d = _run(
        foundry().decide,
        ref,
        reviewer=reviewer,
        rationale=rationale,
        approve=True,
        override=override,
        acknowledge=ack,
    )
    if not emit(d, as_json):
        console.print(
            f"approved {d.version_id} (decision {d.decision_id}, package "
            f"{d.package_sha256[:16]})"
            + (f" [red]OVERRIDDEN gates: {d.overridden_gates}[/red]" if d.override else "")
        )


@app.command()
def reject(
    ref: str,
    reviewer: Annotated[str, typer.Option()],
    rationale: Annotated[str, typer.Option()],
    as_json: JsonOpt = False,
) -> None:
    d = _run(foundry().decide, ref, reviewer=reviewer, rationale=rationale, approve=False)
    if not emit(d, as_json):
        console.print(f"rejected {d.version_id}")


@app.command()
def publish(
    ref: Annotated[str | None, typer.Argument()] = None,
    target: Annotated[Path | None, typer.Option(help="Project directory to install into.")] = None,
    publisher: Annotated[str, typer.Option(help="claude-project | agents-dir")] = "claude-project",
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually install (default: dry run).")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Explicit dry run (default).")] = False,
    force_overwrite: Annotated[
        bool, typer.Option(help="Replace a non-Skill-Foundry directory (backup taken).")
    ] = False,
    recover: Annotated[str | None, typer.Option(help="Recover an interrupted install id.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Show the exact diff and destination; with --apply, install via a recoverable journal."""
    if recover:
        state = _run(foundry().installer.recover, recover)
        if not emit({"install_id": recover, "state": state}, as_json):
            console.print(f"{recover}: {state}")
        return
    if ref is None or target is None:
        fail("publish needs a version and --target")
        return
    if apply and dry_run:
        fail("--apply and --dry-run are mutually exclusive")
    rep = _run(
        foundry().publish,
        ref,
        target,
        publisher=publisher,
        apply=apply,
        force_overwrite=force_overwrite,
    )
    if emit(rep, as_json):
        return
    p = rep["plan"]
    console.print(
        f"destination: {p['destination']} (exists={p['exists']}, managed_by={p['managed_by']})"
    )
    console.print(f"publisher: {rep['publisher']}: {rep['publisher_note']}")
    for c in p["changes"]:
        console.print(f"  {c['change']:7} {c['path']}")
    sys.stdout.write(p["diff"][:20000])
    if rep["overridden"]:
        console.print("[red]this version was approved by OVERRIDE (not empirically proven)[/red]")
    if rep["applied"]:
        console.print(f"[green]installed[/green] {rep['installation']['install_id']}")
    else:
        console.print(f"[yellow]{rep['note']}[/yellow]")


@app.command()
def rollback(
    install_id: Annotated[str | None, typer.Argument()] = None,
    version: Annotated[
        str | None, typer.Option(help="Roll back this version's active install.")
    ] = None,
    recovery: Annotated[str, typer.Option(help="refuse | restore-anyway")] = "refuse",
    as_json: JsonOpt = False,
) -> None:
    """Restore exactly what was at the destination before the installation."""
    rep = _run(foundry().rollback, install_id, version=version, policy=recovery)
    if not emit(rep, as_json):
        console.print(
            f"rolled back {rep['install_id']}: {rep['restored']} (byte_exact={rep['byte_exact']})"
        )


@app.command()
def installations(
    ref: Annotated[str | None, typer.Argument()] = None, as_json: JsonOpt = False
) -> None:
    """Installation and rollback history (independent of candidate versions)."""
    vid = foundry().registry.get_version(ref).version_id if ref else None
    hist = foundry().installer.history(vid)
    if not emit(hist, as_json):
        for h in hist:
            console.print(f"{h['install_id']} {h['version_id']} {h['state']} -> {h['destination']}")
            for e in h["events"]:
                console.print(f"    {e['at']} {e['action']} by {e['actor']}: {e['detail']}")


# ---------------------------------------------------------------------------- export
LicenseAck = Annotated[
    bool,
    typer.Option(
        "--acknowledge-source-licensing",
        help="Confirm you may redistribute content derived from the source experiences.",
    ),
]


@export_app.command("skill")
def export_skill(
    ref: str,
    out: Annotated[Path, typer.Option()],
    ack: LicenseAck = False,
    as_json: JsonOpt = False,
) -> None:
    """Export a stored package + lineage. Generated skills may embed project-specific or
    proprietary details from source experiences: review licensing before sharing."""
    if not ack:
        fail(
            "exported skills can contain details derived from private/proprietary source "
            "experiences; pass --acknowledge-source-licensing after reviewing them"
        )
    out.mkdir(parents=True, exist_ok=True)
    rep = _run(foundry().export_version, ref, out)
    if not emit(rep, as_json):
        console.print(f"exported {rep['version']} ({rep['state']}) to {rep['path']}")


@export_app.command("experiences")
def export_experiences(
    out: Annotated[Path, typer.Option()], ack: LicenseAck = False, as_json: JsonOpt = False
) -> None:
    """Export sanitised experiences as a canonical experience-bundle.v1 (no transcripts)."""
    if not ack:
        fail("pass --acknowledge-source-licensing to export experience content")
    rep = _run(foundry().export_experiences, out)
    if not emit(rep, as_json):
        console.print(f"wrote {rep['episodes']} episodes to {rep['path']}")


# ---------------------------------------------------------------------------- feedback
@fb_app.command("ingest")
def feedback_ingest(path: Path, as_json: JsonOpt = False) -> None:
    from skill_foundry.monitoring.feedback import ingest_feedback

    rep = _run(ingest_feedback, path, foundry().registry)
    if not emit(rep, as_json):
        console.print(rep)


@fb_app.command("status")
def feedback_status(ref: str, as_json: JsonOpt = False) -> None:
    from skill_foundry.monitoring.feedback import monitor

    rep = _run(monitor, foundry().registry, foundry().registry.get_version(ref).version_id)
    if not emit(rep, as_json):
        console.print(rep)


@app.command("propose-revision")
def propose_revision_cmd(ref: str, as_json: JsonOpt = False) -> None:
    """Propose (not generate, not publish) a revised spec from attributed failures."""
    from skill_foundry.monitoring.feedback import propose_revision

    rep = _run(propose_revision, foundry().registry, foundry().registry.get_version(ref).version_id)
    if not emit(rep, as_json):
        console.print(rep)


# ---------------------------------------------------------------------------- demo / mcp
@app.command()
def demo(
    workdir: Annotated[
        Path | None,
        typer.Option(help="Empty directory for demo state and the throwaway target project."),
    ] = None,
    unsafe_host: Annotated[
        bool,
        typer.Option(
            "--unsafe-host", help="Allow host execution of trusted fixtures if no sandbox exists."
        ),
    ] = False,
) -> None:
    """Run the full offline pipeline against a throwaway project."""
    from skill_foundry.demo import run_demo

    ok = run_demo(workdir, unsafe_host=unsafe_host, console=console)
    raise typer.Exit(0 if ok else 1)


@app.command()
def mcp() -> None:
    """Serve a read-only MCP interface (stdio). No approve/publish tools exist."""
    from skill_foundry.mcp.server import serve

    serve(_state["settings"])


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    app()


if __name__ == "__main__":
    main()
