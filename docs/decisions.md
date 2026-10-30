# Decisions, tradeoffs and scope

| decision | why | cost |
|---|---|---|
| Complete linkage over constraint-filtered hybrid-kNN pairs (no HDBSCAN / connected components) | no chaining; every merge is explainable from listed pair scores; trivial at this scale | O(n²) per workspace; may over-split (visible via rejected near-misses) |
| Independence counted in task lineages | three re-runs are one confirmation | lineage heuristics (snapshot, near-dup text, lessons) can under- or over-merge; producers can declare lineage explicitly |
| Typed ProcedureSpec between evidence and Markdown | reviewable and diffable; provider-independent; deterministic validation | rendering is less fluent than free LLM prose |
| Deterministic template renderer by default; LLM optional | reproducible offline pipeline; no false impression of autonomous synthesis | wording quality is limited to what the evidence says |
| SQLite authoritative, Qdrant derived via outbox | correctness never depends on the vector store; rebuildable | an extra sync step; discovery refuses on a stale index |
| Hashing embedder default, FastEmbed optional | deterministic tests/demo without downloads | lexical similarity only by default (documented, reported by `sf doctor`) |
| Default policy requires live `measured_benefit` | a valid SKILL.md is not an effective skill | nothing is promotable without a live benchmark unless a human overrides (recorded) |
| Journaled multi-step install (not "atomic directory replace") | `rename` cannot atomically replace a non-empty directory; the journal makes every crash point recoverable | brief window where the destination is absent; documented |
| bubblewrap first, Docker in CI | bwrap works without a daemon; Docker is common in CI | neither is a hostile-code boundary |

## MVP (implemented)

The full offline vertical slice:
- import, qualification, retrieval, discovery
- ProcedureSpec, generation, validation
- applicability and smoke evaluation
- review, publish, rollback, recovery
- feedback attribution and revision proposals
- read-only MCP

## Later

- Run and publish a live benchmark with more held-out tasks per track.
- An LLM-assisted step-alignment fallback when commands are not in backticks. It would
  stay flagged for review.
- More publisher adapters once their discovery semantics are documented.
- Statistical power planning (pairs needed for a target effect size).
- Optional FastAPI read-only service.
