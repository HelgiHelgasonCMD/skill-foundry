# Evidence for `fix-failing-pytest-suite`

Every procedure step below lists the experiences that support it. Experience IDs refer to records in the Skill Foundry registry (`sf lineage` shows the full chain down to evidence items and source events).

- Cluster: `c-06da7d578607`
- Evidence score: **0.932 (strong)**

| factor | value | weight | contribution | explanation |
|---|---|---|---|---|
| independent_lineages | 1.0 | 0.35 | 0.35 | 5 independent task lineage(s) (saturates at 5) |
| distinct_repositories | 1.0 | 0.15 | 0.15 | 5 distinct repositories (saturates at 3) |
| distinct_environments | 1.0 | 0.1 | 0.1 | 5 distinct environment fingerprints (saturates at 3) |
| verified_ratio | 1.0 | 0.15 | 0.15 | 7/7 members acceptance-verified |
| check_kinds | 0.5 | 0.05 | 0.025 | verified check kinds: ['test'] |
| counterexamples_known | 1.0 | 0.1 | 0.1 | 2 counterexample(s) available to bound applicability |
| single_lineage_dominance | 0.571 | 0.1 | 0.057 | share of members outside the largest lineage (re-runs do not add confidence) |

## Steps

| step | action | lineages | supporting experiences |
|---|---|---|---|
| s1 | Reproduce the failure with `python -m pytest -x -q` to stop at the first failing test | 5 | `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`, `demo/pt-geo-1`, `demo/pt-inventory-1`, `demo/pt-ledger-1`, `demo/pt-textkit-1` |
| s2 | Re-run only the failing test with `python -m pytest <test-node-id> -vv` to read the full assertion diff | 5 | `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`, `demo/pt-geo-1`, `demo/pt-inventory-1`, `demo/pt-ledger-1`, `demo/pt-textkit-1` |
| s3 | Inspect recent changes to the module under test with `git log -p -n 5 -- <module-path>` | 5 | `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`, `demo/pt-geo-1`, `demo/pt-inventory-1`, `demo/pt-ledger-1`, `demo/pt-textkit-1` |
| s4 | Fix the root cause in the source module (not the test) | 5 | `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`, `demo/pt-geo-1`, `demo/pt-inventory-1`, `demo/pt-ledger-1`, `demo/pt-textkit-1` |
| s5 | Run the whole suite with `python -m pytest -q` to confirm the fix and check for regressions | 5 | `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`, `demo/pt-geo-1`, `demo/pt-inventory-1`, `demo/pt-ledger-1`, `demo/pt-textkit-1` |
| b1.1 | when you observe environment (VersionMismatch), e.g. "The failure only reproduced inside the project virtualenv; dependency versions differed from the lockfile": Compare installed packages against the lockfile with `python -m pip freeze` before changing code | 2 | `demo/pt-ledger-1`, `demo/pt-textkit-1` |

## Verification evidence

- Run `python -m pytest -q` and require exit code 0: `demo/pt-billing-1#ev1`, `demo/pt-billing-2#ev1`, `demo/pt-billing-3#ev1`, `demo/pt-geo-1#ev1`, `demo/pt-inventory-1#ev1`, `demo/pt-ledger-1#ev1`, `demo/pt-textkit-1#ev1`

## Counterexamples

- `demo/pt-legacy-1`
- `demo/pt-legacy-2`

## Excluded (single-lineage) steps

- none

## Independent lineages

- `lin-0a3fecbc7368`: `demo/pt-inventory-1`
- `lin-0a81f646a5be`: `demo/pt-ledger-1`
- `lin-2da05f7f928a`: `demo/pt-textkit-1`
- `lin-98a8ccdacca1`: `demo/pt-geo-1`
- `lin-a8ea4c0562c5`: `demo/pt-billing-1`, `demo/pt-billing-2`, `demo/pt-billing-3`
