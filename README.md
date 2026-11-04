# Skill Foundry

**Turn verified coding-agent experience into evidence-backed, evaluated, governed [Agent Skills](https://agentskills.io/specification), without silently installing anything.**

Coding agents keep re-learning the same procedures. Skill Foundry asks a narrower,
testable question than "can an LLM write a SKILL.md?":

> Can a system discover a reproducible procedure from *multiple independent, verified*
> executions, generalise it without overfitting, and show objective improvement on
> *held-out* tasks, with every claim traceable to evidence?

```
ingest ─► qualify evidence ─► hybrid retrieval ─► constrained clustering ─► ProcedureSpec
  ─► render SKILL.md (quarantine) ─► validate + lint ─► applicability eval ─► paired capability eval
  ─► human review (hash-bound) ─► publish dry-run ─► journaled install ─► monitor ─► revise / rollback
```

## Status: what is and is not established

| Capability | Status |
|---|---|
| experience-bundle.v1 import (canonical schema, verbatim), referential integrity, round-trip export | **Implemented and tested** |
| Sanitisation (structured secret exclusion, regex/entropy redaction), injection quarantine, size limits; redaction tested at ingest, storage, index, retrieval and export | **Implemented and tested** |
| Evidence trust levels (`self_reported` / `recorded` / `verified`, acceptance-linked) | **Implemented and tested** |
| SQLite registry (Alembic, WAL), index outbox, idempotent Qdrant upserts, `sf index rebuild`, stale-index refusal | **Implemented and tested** (embedded Qdrant locally; server mode tested in CI) |
| Hybrid BM25 + dense retrieval with RRF, workspace filter applied before scoring | **Implemented and tested** (deterministic lexical embedder by default; FastEmbed semantic model tested in CI) |
| Constrained complete-linkage clustering, independence by task lineage, explainable evidence score | **Implemented and tested** |
| ProcedureSpec inference with per-step evidence references | **Implemented and tested** |
| Spec-compliant SKILL.md generation in quarantine (+ `skills-ref` reference validator), content/security lint, held-out leakage guard | **Implemented and tested** |
| Lifecycle, promotion policy, hash-bound approval, audited overrides | **Implemented and tested** |
| Journaled publish, crash recovery, byte-exact rollback, drift detection | **Implemented and tested** |
| bubblewrap sandbox (no network, no `$HOME`, read-only inputs, rlimits) | **Tested locally**; Docker backend tested in CI |
| Anthropic-backed synthesiser + claim checker | **Tested only with a mocked client** |
| Claude Code runner (isolated skill loading via `--add-dir`, stream-json parsing, contamination checks) | Real CLI **lists the skill in the candidate arm only** (checked with no credentials, so no model request). Whether the model **invokes** it is **UNVERIFIED** |
| hubris managed tasks as a source of experience, `sf autopilot` up to quarantine | **Tested with a fake hubris CLI**; import also run on one real task. See [docs/hubris.md](docs/hubris.md) |
| Agent Flight Recorder bundles | **Rejected today**: AFR's schema differs. See [docs/interoperability.md](docs/interoperability.md) |
| Live agent benchmark | **Needs a Claude subscription token or API key**: see [docs/live-benchmark.md](docs/live-benchmark.md) |
| Live synthesis (`AnthropicSynthesizer`) | **Needs an API key.** The default template synthesiser needs none |
| **Measured improvement of a real agent from a generated skill** | **NOT MEASURED. No such result exists.** |

The offline demo proves **pipeline correctness with fixtures**. Its "capability"
step replays hand-written per-arm scripts and is labelled `smoke_only`. It is not
evidence that any skill helps any agent.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and, for sandboxed
evaluation, [bubblewrap](https://github.com/containers/bubblewrap) (`apt install bubblewrap`)
or Docker.

```bash
git clone https://github.com/HelgiHelgasonCMD/skill-foundry && cd skill-foundry
uv sync
uv run sf demo                  # full offline pipeline in a fresh temp directory
uv run pytest -q                # 150 tests (+4 that need CI services)
```

Step by step against your own state directory:

```bash
export SF_HOME=.sf SF_WORKSPACE=demo
S=examples/eval/suite.json
uv run sf ingest examples/bundles/demo-bundle.v1.json
uv run sf discover --suite $S                    # clusters + ProcedureSpecs (held-out lineages excluded)
uv run sf hypothesis show <cluster-id>           # members, lineages, rejected near-misses, score factors
uv run sf spec show <spec-hash>
uv run sf generate <spec-hash> --suite $S        # renders into .sf/quarantine, never .claude/
uv run sf validate fix-failing-pytest-suite@0.1.0 --suite $S
uv run sf eval applicability fix-failing-pytest-suite@0.1.0 --suite $S --track python-test-failure-triage
uv run sf eval capability   fix-failing-pytest-suite@0.1.0 --suite $S --track python-test-failure-triage
uv run sf lineage fix-failing-pytest-suite@0.1.0 # why the skill exists, down to source event ids
uv run sf review  fix-failing-pytest-suite@0.1.0 # gates, diff vs parent, history
uv run sf approve fix-failing-pytest-suite@0.1.0 --reviewer you --rationale "..."   # blocked without live evidence
uv run sf publish fix-failing-pytest-suite@0.1.0 --target ../some-project           # dry run: exact diff
uv run sf publish fix-failing-pytest-suite@0.1.0 --target ../some-project --apply   # journaled install
uv run sf rollback --version fix-failing-pytest-suite@0.1.0                          # byte-exact restore
```

Every command accepts `--json`. Other commands: `sf doctor`, `sf experiences`,
`sf label`, `sf index status|sync|rebuild`, `sf candidates list`, `sf versions`,
`sf diff`, `sf compare <run-id>`, `sf installations`, `sf export skill|experiences`,
`sf feedback ingest|status`, `sf propose-revision`, `sf import hubris`, `sf autopilot`,
`sf mcp` (read-only MCP server).

To run this on your real work through [hubris](docs/hubris.md), see that page.

## Sample output

- Generated skill: [docs/sample/fix-failing-pytest-suite/SKILL.md](docs/sample/fix-failing-pytest-suite/SKILL.md)
  with its [evidence table](docs/sample/fix-failing-pytest-suite/references/evidence.md)
- Full demo transcript: [docs/sample/demo-output.txt](docs/sample/demo-output.txt)

Excerpt (demo step 8, the smoke comparison):

```
capability run e-… (runner_kind=fake)
 case                       baseline  candidate  baseline failure
 pytest-discount-threshold  fail      pass       hidden-tests, tests-unchanged
 pytest-slug-trailing       pass      pass       -
  status: smoke_only
  note: Only 2 paired run(s); at least 10 are required before any significance is computed.
  note: runner_kind=fake: this is a SMOKE TEST of the harness, sandbox and verifiers.
        It is NOT evidence that the skill improves an agent.
```

## Design in one page

- **SQLite is the source of truth.** Qdrant is a derived index fed by a transactional
  outbox. If indexing fails, rows stay pending and visible, and discovery refuses to run on
  a stale or incomplete index.
- **Schema-valid ≠ trusted.** An experience is `verified` only if a test, CI, replay or
  external check exited 0, *and* that check is linked to the task's acceptance criteria and
  to the tested revision. A successful `make` stays `recorded`.
- **Independence is counted in task lineages, not episodes.** Three re-runs of one task
  count as one confirmation.
- **Clusters cannot chain.** Complete linkage over constraint-filtered candidate pairs:
  every member must be similar to every other member.
- **ProcedureSpec is the reviewable artefact.** Each step lists supporting experiences.
  Lesson prose is "reported" and never becomes a step. LLM additions are flagged
  `needs_review` and block promotion until acknowledged.
- **Held-out data is separated by lineage before synthesis.** n-gram leakage checks are
  defence in depth.
- **Valid ≠ effective.** Spec compliance, lint, applicability, capability
  (`not_measured / smoke_only / inconclusive / no_benefit / measured_benefit / regression`),
  evidence strength and the review decision are stored separately. By default only live,
  paired, sufficiently sized evidence can satisfy the capability gate. Anything else needs
  an explicit, recorded human override, displayed as `OVERRIDDEN (not empirically proven)`.
- **Publication is journaled.** Approval binds to the package sha256. Installs go through
  stage → backup → swap → verify → commit with crash recovery. Rollback restores the stored
  snapshot byte-for-byte and refuses to clobber user edits unless told to.

More detail:
- [docs/architecture.md](docs/architecture.md): modules and diagrams
- [docs/evaluation-methodology.md](docs/evaluation-methodology.md)
- [docs/security.md](docs/security.md): trust model, sandbox limits, destructive actions
- [docs/interoperability.md](docs/interoperability.md): the experience-bundle.v1 contract
- [docs/live-benchmark.md](docs/live-benchmark.md)
- [docs/hubris.md](docs/hubris.md): real experience from hubris managed tasks, on a schedule
- [docs/decisions.md](docs/decisions.md): tradeoffs, MVP vs later

## Limitations (honest list)

- **No live capability result.** No paired run has completed against the real model.
- **Skill invocation is unverified.** The real CLI lists a skill exposed through `--add-dir`,
  but whether the model invokes it is unknown until an authenticated run records `invoked`.
- **The default embedder is lexical** (feature hashing) so tests and the demo are
  deterministic and offline. Use `--embedding fastembed` for semantic retrieval.
- **Command extraction is heuristic.** The contract has no structured command field, so
  commands come from backtick spans.
- **Applicability evaluation is static and lexical.** It checks what the skill declares,
  not how an agent actually triggers it.
- **The corpus is small and synthetic.** Thresholds were set on it. Real data will need
  tuning, done on training data only.
- **bubblewrap and Docker are not security boundaries** against hostile code.

## License

MIT. The demo corpus and evaluation tasks are synthetic and were written for this repository.
