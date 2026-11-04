# Running Skill Foundry on your work with hubris

Do your work as hubris managed tasks. A scheduled `sf autopilot` pass turns finished tasks
into candidates in quarantine and tells you. Approval and merging stay with you.

```
you:   hubris task submit "<task>" --verify "uv run pytest -q" --protect "tests/**"
auto:  sf autopilot: import → index → discover → generate → validate → applicability
       └─ stops at quarantine and runs SF_NOTIFY_CMD
you:   sf review → (optional) live eval → sf approve
you:   publish in a worktree → PR → merge     ← new hubris worktrees now have the skill
```

## Which work counts

hubris runs the check itself, so its result is evidence. The agent saying "done" is only a claim.

| hubris result | stored as | trust |
|---|---|---|
| `succeeded`, clean tree, `--protect` set, no protected file touched | `verified_success` | `verified` |
| `succeeded` without `--protect`, or with a dirty tree | `verified_success` | `recorded` (the agent could have edited the tests) |
| `failed` | `verified_failure` | `verified` counterexample |
| `inconclusive` | `claimed_success` or `unknown` | `self_reported` |
| `errored`, `timed_out`, `cancelled` | skipped | |

Only `verified` tasks can support a skill, so **use `--verify` and `--protect` together**. Plain
`hubris launch` sessions have no verification and are not imported. `verified` also requires a
hubris version whose `task status --json` includes the `verify` spec; older versions give `recorded`.

The check is only as strong as its command. `--verify "test -f hello.txt"` is taken at face value.

**Expect a wait.** A skill counts as reusable only with at least 3 independent tasks across at least
2 repositories and 2 environments. On one machine, "environment" means the runtime and the version
declared in `.python-version`, `.node-version` or `.nvmrc`. Repositories that declare no differing
version count as one environment, so their procedures stay `environment_specific` and no skill is
generated.

## What is read from your sessions

Commands, edited file paths and failure flags. From each transcript it keeps only:

- tool name, Bash command and edited file path from `tool_use`;
- `is_error` from `tool_result`.

Thinking, assistant text, tool output, attachments and API request records are discarded on read.
Sub-agent records are skipped. Multi-line and very long commands are omitted and counted. What
remains, plus the task prompt and verify command, goes through normal sanitising, redaction and
injection screening. Secrets are detected by pattern, so a secret in an unusual format can survive
in a stored command. Review what you allowlist.

**Nothing is imported unless you name the repository.** Skills that span repositories need them in
one workspace, so the `--repo` list is your privacy boundary: leave sensitive repositories out.

## Set it up

From your Skill Foundry checkout, one pass by hand:

```bash
export SF_HOME=~/.local/share/skill-foundry SF_WORKSPACE=mine
uv run sf import hubris --repo my-api --repo my-web
uv run sf autopilot --repo my-api --repo my-web
```

Then on a schedule, with `crontab -e`:

```cron
*/30 * * * * cd /path/to/skill-foundry && SF_HOME=$HOME/.local/share/skill-foundry SF_WORKSPACE=mine SF_NOTIFY_CMD='notify-send "Skill Foundry" "$SF_NEW_CANDIDATES $SF_NEEDS_SUITE $SF_FAILED"' uv run sf autopilot --repo my-api --repo my-web --suite /path/to/suite.json >> $HOME/.local/share/skill-foundry/autopilot.log 2>&1
```

Cron's `PATH` is minimal, so `uv` and `hubris` may need full paths (`--hubris-bin` sets the latter).
Under WSL without `notify-send`, use any command you like. Values arrive in the environment:
`SF_NEW_CANDIDATES`, `SF_NEEDS_SUITE` and `SF_FAILED`.

A pass does nothing if another is running, and does no discovery when nothing has changed. Embedded
Qdrant allows one process at a time, so an `sf` command you run during a pass can fail with a lock
error: run it again. It never approves, publishes or runs a live evaluation, so it can't spend your
quota.

### The evaluation suite

**Without `--suite`, a pass stops at the ProcedureSpec** and reports `needs_suite`. A suite holds the
held-out tasks that must be kept out of synthesis, and `sf generate` refuses to run without one.

There is no way yet to build a suite from your hubris tasks. To reach quarantine today, write a suite
with no held-out cases:

```json
{"schema_version": "eval-suite.v1", "suite_id": "mine", "corpus_version": "1",
 "tracks": {}, "cases": [], "applicability_cases": []}
```

That is honest only as far as it goes: nothing is held out, so a skill built this way has no
capability evidence, and approving it requires a recorded override. Add a track named after a skill
to get applicability checks for it.

## Approve and publish

```bash
uv run sf review <skill>@<version>
uv run sf approve <skill>@<version> --reviewer you --rationale "..."   # blocked without live evidence
```

`sf publish` writes files into one checkout, but hubris cuts new worktrees from the default branch,
so a skill must be merged to reach them:

```bash
git -C my-api worktree add .claude/worktrees/skill-x -b skill/x
uv run sf publish <skill>@<version> --target my-api/.claude/worktrees/skill-x          # dry run
uv run sf publish <skill>@<version> --target my-api/.claude/worktrees/skill-x --apply
git -C my-api/.claude/worktrees/skill-x add -f .claude/skills   # -f: many repos ignore .claude/
git -C my-api/.claude/worktrees/skill-x commit -m "Add skill x"
# push, open a PR, review, merge
```

**Treat the merge as a privileged step.** hubris workers run unattended in auto permission mode, so
a merged skill steers agents nobody is watching. Read the SKILL.md in the PR.

A version can be published to one place at a time: publishing it into a second repository is refused
as `version is published`. To use a skill in several repositories, copy the merged directory yourself
or publish a new version for each.

Once merged, undo it with `git revert`. `sf rollback` only restores the one checkout it installed
into. `~/.claude/skills` stays refused.

## Not built yet

- Feeding later task outcomes into `sf feedback` automatically.
- Building held-out evaluation suites from hubris tasks. Until then there is no route to
  `measured_benefit` on real work.
