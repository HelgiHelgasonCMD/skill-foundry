# Interoperability: experience-bundle.v1

`schemas/experience-bundle-v1.schema.json` is the **canonical cross-repository contract**
shared with Agent Flight Recorder (AFR). It was supplied with the project
specification and is stored **byte-for-byte**:

```
sha256 6d649ed25c144a64c1be77fe696500b0f8865763cc71fc107ae2bee54298fc36
```

`tests/test_contract.py::test_canonical_schema_is_pinned_verbatim` fails if the file changes.
Skill Foundry has no runtime, package, database or API dependency on AFR. The JSON
file is the only coupling.

## Shape (normative: see the schema)

- **Bundle:** `schema_version` (const `experience-bundle.v1`), `export_id`, optional
  `exported_at`, `episodes[]`, optional `extensions`.
- **Episode:** `episode_id`, `task_summary`, `environment`, `observations[]`,
  `attempts[]`, `outcome`, `lessons[]`, `provenance`, optional `extensions`.
  - `environment`: `repo_id`, `revision`, `versions`, optional `runtime` and
    `platform`. `repo_id` and `revision` may be null.
  - `outcome.status`: `unknown | claimed_success | verified_success | verified_failure`.
    Verified statuses need at least one evidence item.
  - `evidence`: `evidence_id`, `kind` (`test|ci|command|replay|external_check|other`),
    `summary`, `observed_at`, optional `revision`, `exit_code`, `source_event_ids`.
  - `lessons`: `problem`, `observed_symptoms`, `failed_approaches`,
    `successful_approach` (nullable), `preconditions`, `contraindications`,
    `versions`, `verification_evidence_ids`, `source_episode_ids`,
    `confidence_basis`.
  - `provenance`: `source`, `event_ids`, `redaction_applied`.
- `additionalProperties: false` applies everywhere except inside `extensions` objects.

## Checks beyond JSON Schema (Skill Foundry side)

| check | failure → |
|---|---|
| unknown `schema_version` | whole bundle rejected (`UnsupportedVersion`) |
| envelope schema error | whole bundle rejected |
| episode schema error | that episode rejected with reasons; others imported |
| duplicate episode / observation / attempt / evidence / lesson ids | episode rejected |
| any `source_event_ids` not in the episode's `provenance.event_ids` | episode rejected |
| lesson `verification_evidence_ids` not found in the bundle | episode rejected |
| lesson `source_episode_ids` not in the bundle | episode rejected |
| same `episode_id` re-imported with different content | rejected (no silent overwrite) |

## Trust is not part of the contract

The schema validates shape. Skill Foundry assigns trust itself (see the README):

- `claimed_success` / `unknown` → `self_reported`
- `verified_success` → `verified` only if some evidence item meets all of these:
  - is `test|ci|replay|external_check`
  - has `exit_code == 0`
  - is acceptance-linked
  - its tested revision equals `environment.revision`

  Otherwise → `recorded`.

## Optional extensions read by Skill Foundry

These are namespaced and optional. Producers that omit them are fully supported, but their
successes will qualify as `recorded` until a human attests them with
`sf label --key attest-acceptance`.

| location | key | value |
|---|---|---|
| `evidence.extensions` | `skill_foundry.acceptance` | `{"criteria": [str], "tested_revision": str, "worktree_fingerprint": str?}` |
| `environment.extensions` | `skill_foundry.worktree_fingerprint` | hash of the tested working tree (uncommitted changes included) |
| `episode.extensions` | `skill_foundry.task_lineage` | stable id shared by re-runs / derived tasks |

## Export

`sf export experiences --out f.json --acknowledge-source-licensing` writes a
schema-valid bundle containing only sanitised fields. Quarantined records and raw
transcripts are never exported. A round-trip test asserts schema validity and equality
of canonical fields.

## Other schemas (Skill Foundry-owned)

- `schemas/eval-suite-v1.schema.json`: held-out tasks, verifiers, applicability cases.
- `schemas/outcome-feedback-v1.schema.json`: post-publication outcomes. An event is
  attributed to a version only when it carries that exact `package_sha256`, and is not
  attributed if `skill_invoked: false`.
