# Live benchmark (optional, not yet run)

**Status: not run. There is no measured capability result for any skill in this repository.**

The harness, the Claude Code adapter and its isolation are implemented. They are tested
only against a mock `claude` executable. Whether the real CLI discovers a skill passed
through `--add-dir` under this isolation (empty `CLAUDE_CONFIG_DIR`,
`--setting-sources project`) is **unverified**. The first live run will record it: see
`skill_observed` in the report.

## Prerequisites

- Claude Code CLI on `PATH` (`claude --version`), plus bubblewrap.
- `ANTHROPIC_API_KEY` in the environment. OAuth/keychain logins are not visible inside
  the sandbox by design.
- A budget you accept. Every case sets `max_budget_usd`, and the CLI enforces it per run.

## Run

```bash
export SF_HOME=.sf-live SF_WORKSPACE=demo
S=examples/eval/suite.json
uv run sf ingest examples/bundles/demo-bundle.v1.json
uv run sf discover --suite $S
uv run sf generate <spec-hash> --suite $S
uv run sf validate fix-failing-pytest-suite@0.1.0 --suite $S
uv run sf eval capability fix-failing-pytest-suite@0.1.0 --suite $S \
    --track python-test-failure-triage --runner claude-code \
    --i-understand-this-runs-a-live-agent --pass-api-key \
    --model <model-id> --repeats 5 --seed 1 --trigger auto
# repeat with --trigger explicit to separate triggering from usefulness
uv run sf compare <run-id>
```

## Reading the result

- The bundled tracks have 1 to 2 tasks, so 5 repeats give at most 10 pairs. That is just
  enough to compute a p-value, but nowhere near enough for a general claim. Add tasks
  (with hidden verifiers) before drawing conclusions.
- Check `skill_observed.invoked`. If the skill was never invoked, differences cannot be
  attributed to it.
- Results only count for the recorded model, CLI version, tool allowlist and tasks (see
  the run manifest).
- Publish the raw report JSON alongside any claim.
