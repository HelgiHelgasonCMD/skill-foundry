# Security and trust model

## Trust boundaries

| input | treated as | consequences |
|---|---|---|
| imported experiences | **untrusted data** | size/shape limits; secret-named fields and hidden-reasoning fields dropped; regex + entropy redaction; injection heuristics quarantine the record (excluded from indexing and synthesis) |
| producer lessons | *reported* claims | never become procedure steps; only preconditions/contraindications, labelled `reported` |
| schema validity | shape only | does not raise trust; `verified` requires an acceptance-linked objective check on the tested revision |
| LLM synthesiser output | untrusted suggestion | prompt marks spec data as untrusted; unsupported commands/claims are rejected or flagged; flags block promotion until acknowledged |
| generated skills | **untrusted instructions** | written only to quarantine; never `allowed-tools`; `scripts/` refused unless explicitly approved; header states they grant no authority |
| verified source evidence | evidence | never grants execution permissions to a skill |

Redaction is a safety net, not a guarantee. Review exported skills before sharing:
they can carry project-specific or proprietary details. `sf export` requires
`--acknowledge-source-licensing`.

## Sandbox

`SandboxPolicy` is deny by default.
- **Mounts:** one writable scratch dir and explicit read-only inputs.
- **Withheld:** no network, no `$HOME`, no credentials, no SSH agent, no Docker socket.
- **Environment:** empty, plus an allowlist.
- **Limits:** wall-clock timeout with process-group kill, plus rlimits on address space,
  CPU, file size, open files and core dumps.

Backends:

- **bubblewrap**: user namespaces with `--unshare-all` and `--cap-drop ALL`.
  - `/usr` is read-only.
  - The interpreter prefix and virtualenv are bound read-only so Python verifiers can run.
    The home directory itself is not mounted.
- **Docker**:
  - `--network none --read-only --cap-drop ALL --security-opt no-new-privileges`
  - `--pids-limit` and `--memory`, running as a non-root user
- **unsafe-host**: only with an explicit `--unsafe-host`, for trusted fixtures.

**Neither bubblewrap nor Docker is a security boundary against deliberately hostile
code.** Kernel attack surface, resource exhaustion and side channels remain. Both
reduce the damage from accidents in *trusted fixture* code. Do not run untrusted
generated code with them and assume safety.

Both the agent run and the verifiers go through the sandbox. Containing only the
verifier would not make a run safe if the agent itself ran unrestricted on the host.

### Live Claude Code runs

The CLI needs network access and authentication. The live runner:
- runs in bubblewrap with networking enabled. bwrap cannot restrict destinations, so this
  boundary is **weaker** than the offline one.
- uses a private empty `CLAUDE_CONFIG_DIR`, so host credentials and user skills are not
  visible.
- passes `ANTHROPIC_API_KEY` only with `--pass-api-key`. OAuth/keychain logins are not
  available inside the sandbox.
- restricts the agent to an allowlist (`Read Edit Write Glob Grep Skill` and specific
  `Bash(...)` patterns) and never uses `--dangerously-skip-permissions`.
- requires `--i-understand-this-runs-a-live-agent`.

## Destructive or outward-facing actions

| action | safeguard |
|---|---|
| `sf publish --apply` | approval bound to package hash; quarantine re-hashed; refuses global agent dirs (`~/.claude`, `~/.agents`, `~/.config`), `$HOME`, `/`, symlinks on the path; refuses unmanaged existing destinations unless `--force-overwrite` (snapshot taken first); journaled with recovery |
| `sf rollback` | verifies the destination still matches what was installed; refuses on drift unless `--recovery restore-anyway` (drifted state snapshotted first); restores stored bytes |
| `sf index rebuild` | drops only the derived index; SQLite untouched |
| `sf export` | requires licensing acknowledgement |
| model / agent execution | opt-in flags only; nothing runs in the background; no auto-update, auto-approve or auto-publish anywhere (including MCP, which is read-only) |

Tests and the demo only write to temporary directories. The path-safety test fakes
`$HOME`.
