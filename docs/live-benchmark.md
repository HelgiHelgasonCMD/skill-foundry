# Live benchmark (optional, not yet run)

**Status: not run. There is no measured capability result for any skill in this repository.**

The harness, the Claude Code adapter and its isolation are implemented, but no paired
run has ever completed against the real model.

## What has been checked against the real CLI

Claude Code 2.1.318 in bubblewrap, with no credentials, so no model request was made:

| check | result |
|---|---|
| Candidate arm: the skill appears in the `init` event, exposed only through `--add-dir` | **Yes** on both cases |
| Baseline arm: the skill appears | **No** on both cases |
| Skills from `~/.claude/skills` leak into either arm | **No** |
| The private `CLAUDE_CONFIG_DIR` hides the host login | **Yes**: "Not logged in" |
| Failed runs count as evidence against the skill | **No**: status `inconclusive` |

**Still unverified:** whether the model actually invokes the skill (`invoked`, `used`), and
any effect on results. Those need an authenticated run.

## Run it on a Claude subscription

No API key is needed. In your own terminal, so the token stays out of any agent session:

```bash
claude setup-token                      # opens a browser, prints a one-year token, saves nothing
export CLAUDE_CODE_OAUTH_TOKEN=<token>
```

The token can only make model requests. Revoke it from your Claude account settings when you
stop using it.

Prerequisites: the `claude` CLI on `PATH`, plus bubblewrap.

```bash
export SF_HOME=.sf-live SF_WORKSPACE=demo
S=examples/eval/suite.json
uv run sf ingest examples/bundles/demo-bundle.v1.json
uv run sf discover --suite $S
uv run sf generate <spec-hash> --suite $S
uv run sf validate fix-failing-pytest-suite@0.1.0 --suite $S

# Start with one repeat: does the model use the skill at all?
uv run sf eval capability fix-failing-pytest-suite@0.1.0 --suite $S \
    --track python-test-failure-triage --runner claude-code \
    --i-understand-this-runs-a-live-agent --pass-oauth-token \
    --model <model-id> --repeats 1 --seed 1 --trigger auto
# repeat with --trigger explicit to separate triggering from usefulness
uv run sf compare <run-id>
```

With an API key instead, use `--pass-api-key` and `ANTHROPIC_API_KEY`.

What to expect:

- **It uses your subscription quota,** which your other Claude Code sessions share. Each repeat
  runs both arms of every case in the track.
- **`max_budget_usd` is unverified on a subscription,** and any reported cost is not what you pay.
  Rely on `--repeats` and each case's `timeout_s` as the real limits until a run shows otherwise.
- **The token is visible to the agent under test,** inside a network-enabled sandbox. It is passed
  to bubblewrap on its command line, so other processes running as your user can see it during a
  run. It is scrubbed from traces.
- **If every run reports `crash` with "Not logged in" in `trace.jsonl`,** the token did not reach
  the run: check the export and `--pass-oauth-token`.
- **Never add `--bare`:** bare mode does not read this token.

## Reading the result

- The bundled tracks have 1 to 2 tasks, so 5 repeats give at most 10 pairs. That is just
  enough to compute a p-value, but nowhere near enough for a general claim. Add tasks
  (with hidden verifiers) before drawing conclusions.
- Check `skill_observed.invoked`. If the skill was never invoked, differences cannot be
  attributed to it.
- Results only count for the recorded model, CLI version, tool allowlist and tasks (see
  the run manifest).
- Publish the raw report JSON alongside any claim.
