# Evaluation methodology

Skill Foundry separates three kinds of evaluation. **Only kind C can show that a skill
improves an agent.**

## A. Pipeline correctness

This is the pytest suite (`uv run pytest`) plus `sf demo`. It covers:
- schema parsing, referential integrity and round trips
- redaction at every boundary
- retrieval and clustering behaviour, ProcedureSpec provenance
- spec validation, lint and leakage
- sandbox limits, verifiers, the mocked Claude runner
- lifecycle and policy
- journaled publish, crash recovery and rollback

None of it is stored as evidence about a skill.

## B. Applicability and safety (`sf eval applicability`)

These checks are deterministic and run against held-out task *descriptions* with known
expected answers (`use` / `do_not_use`):

- **Trigger coverage:** lexical coverage of the skill's trigger terms and keywords.
- **Contraindications:** the task matches a contraindication taken from a verified
  counterexample or a reported lesson.
- **Unsupported environments:** the task's runtime family was never evidenced, or its
  fingerprint is one where the procedure was observed to fail.
- **Safety:**
  - quarantined (injection-bearing) records do not propagate into the skill (n-gram check)
  - counterexamples surface as "When NOT to use"
  - no `allowed-tools`

A wrong `use` on an expected `do_not_use` case is **critical**. This is a static check of
what the skill *declares*. It does not measure live triggering.

## C. Capability (`sf eval capability`)

Paired baseline-vs-candidate runs.

| control | how |
|---|---|
| same starting state | each arm gets a fresh copy of the same task snapshot; the snapshot hash is recorded |
| same conditions | identical limits (timeout, turns, budget), runner configuration, model, tool allowlist and verifiers |
| order effects | arm order randomised per pair from a recorded seed |
| repeats | `--repeats N` for nondeterministic agents |
| hidden answers | hidden tests live outside the workspace and are mounted read-only only for the verifier step. They are excluded from synthesis by lineage and checked again by n-gram guard |
| contamination | fresh workspace with no `.claude/skills`, private empty `CLAUDE_CONFIG_DIR`, `--setting-sources project`, `--strict-mcp-config`, no session persistence. The baseline must not see or invoke the skill (enforced) |
| skill exposure | candidate arm only: exact package (hash-checked) under `<run>/skillroot/.claude/skills/<name>`, passed with `--add-dir` |
| skill observation | from stream-json, recorded separately: `discoverable` (listed at init), `invoked` (Skill tool use), `used` (read skill resources after invoking) |
| trigger modes | `--trigger auto` (plain prompt) vs `--trigger explicit` (`/<skill> prompt`), reported separately |
| judging | independent verifiers only (hidden pytest, unchanged-file checks, YAML constraints, commands). The agent's own claim of success is never used |
| cleanup | workspace, skill root and config deleted after each arm; redacted traces kept |

Each pair records success, critical verifier failures, tool calls, failed actions,
wall time, and tokens/cost **only when the runner reports them**.

### Reporting and statistics

- Success counts per arm with Wilson 95% intervals.
- Discordant pairs: baseline-only versus candidate-only successes.
- Exact McNemar (two-sided binomial on discordant pairs) **only if there are at least 10
  pairs**. Otherwise `p_value: null` with an explicit "insufficient sample" note.
- Critical regression: a critical verifier passes without the skill and fails with it.
- Reproducibility manifest:
  - runner and CLI version, model, sandbox
  - candidate package sha256
  - suite id, corpus version and content hash
  - task snapshot hashes
  - verifier version, limits, seed, trigger mode

`capability_status` is derived as follows:

| runner_kind | result → status |
|---|---|
| `fake` / `mock` | always `smoke_only`: never evidence of benefit |
| `live` | critical regression → `regression`; fewer than 10 pairs → `inconclusive`; more candidate-only successes with p < 0.05 → `measured_benefit`; no advantage → `no_benefit`; otherwise `inconclusive` |

If the candidate skill was never observed being invoked in a live run, the report says
so. Any difference then cannot be attributed to the skill.

## Promotion policy

The default policy blocks approval unless all of these hold:
- spec compliance passes
- lint passes
- applicability passes
- there is no regression
- capability is `measured_benefit`
- evidence strength is moderate or strong
- every review item is acknowledged

A named human may override any gate except spec compliance and lint, with a rationale.
The decision records the overridden gates, and the version is shown as
`OVERRIDDEN (not empirically proven)` from then on. The demo uses this path openly.

## Train / held-out separation

1. Lineages are computed first. They come from declared lineage, the same repo and
   revision, near-duplicate task text in one repo, and shared lessons.
2. Any lineage that touches a held-out case is excluded from synthesis. The match can be
   by declared lineage, by the same snapshot, or by a near-duplicate prompt.
3. Thresholds and rules are constants set on training data. Held-out cases are never used
   to tune them.
4. n-gram leakage checks run on synthesis input, synthesis output and rendered files, as
   defence in depth.
