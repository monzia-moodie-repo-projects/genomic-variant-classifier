# `scripts/forensics/` — spent investigative tooling

**Created 2026-07-12.** 68 one-off scripts, moved here from `scripts/`.

## What these are

Each of these produced a **finding**, not a feature. They are the probes, diagnostics,
audits and verifiers written during the cohort-v2 / cohort-v3 investigations, the
alleleless-variant recovery, the git-history rewrite, and the out-of-fold / split audits.
They ran once, answered a question, and the answer was written into a document.

They are archived rather than deleted, deliberately:

- **They are provenance.** Several are cited by name in committed evidence documents —
  notably `docs/status/ALLELELESS_PROVENANCE_2026-07-09_FINAL.md`, which references
  `diagnose_alleleless_keys_v3.py`, `diagnose_collision_groups_v2.py`,
  `probe_resolve_vs_sourceid.py`, `probe_sid_in_vcf.py`, `probe_snv_alleleless.py`,
  `verify_v3_against_v2.py` and `audit_recovery_collapse.py`. **Those all live HERE now.**
  A scientific result whose tooling has been deleted is not reproducible.
- **Deleting untracked files is irreversible.** They were never committed, so `rm` would
  have destroyed them with no git history to recover from. Archiving costs a few hundred
  kilobytes of text and loses nothing.

## What is NOT here, and why

`scripts/` retains everything **reachable from `tests/` or `src/`**, computed as a
**transitive closure**, not a one-hop reference count. That distinction mattered:
`scripts/diagnose_identity_join.py` is imported by no test *directly*, but by
`scripts/probe_identity_first_recovery.py`, which a test *does* import. A naive
"is it referenced by a test?" check would have archived it and turned the suite red.

The classification is reproducible:

```
python scripts/audit_untracked_hygiene.py
```

It buckets every file into TEST-MODULE / REQUIRED / DOC-ONLY / ORPHAN and prints the
evidence for each. Do not disposition files by eye.

## Status

**Frozen.** Nothing here is maintained, imported by the package, or run by the test suite.
If you find yourself editing a file in this directory, it probably belongs back in
`scripts/` — move it, and add a test.

## Why it happened

These 68 files sat **untracked** in the working tree for weeks. On 2026-07-12 the G1
pre-flight gate (`scripts/Run_Preflight_Local.ps1`) failed with *"working tree has
uncommitted changes"* and **blocked a Run-17 launch** — which is exactly what that check
exists to do. Undispositioned files are not free; they eventually stand between you and a
run.
