# SESSION 2026-08-21 to 2026-08-22 -- authority becomes typed

**Author: Monzia Moodie**
**Commits:** `b115bab`, `1c50680`, `69ba5f6` (three, all pushed)
**Ratchet:** 5213 -> 5213 (unchanged; no test was added or removed)
**Preceding head:** `084ece5`
**Ending head:** `69ba5f6`

> **Record status:** pre-archive-migration. Future archive class: session
> notebook. Migration required. Written in the established `docs/sessions/`
> location because three commits were published with no record and the archive
> foundation does not yet exist; D2b will ingest it without rewriting its
> historical meaning.

---

## 0. What this session did, and did not do

It closed `RUNNER-GATE-METADATA-ORDER-1` at both ends, accepted two
architecture decision records, and produced a documented governance
architecture for the project's knowledge surfaces. It touched none of the five
roadmap deliverables, no Run 17 work, and no scientific code.

Seven generations of ruling were received and adopted in sequence. Every one is
recorded in section 8 with its digest, because the file name is a rolling
convention rather than a single document and the sequence is the record.

---

## 1. The three commits

| commit | unit | plan digest | gate | wall clock |
|---|---|---|---|---|
| `b115bab` | D0a -- ADR-0001 | `41f20979bdefe669` | 5198 passed, 15 skipped, 0 failed, 33 warnings | 998.13s |
| `1c50680` | D0b -- ADR-0002 | `4d545b8cfad35174` | 5198 passed, 15 skipped, 0 failed, 33 warnings | 1010.66s |
| `69ba5f6` | D3a -- RUNNER-GATE-METADATA-ORDER-1 | `e264e3ca19702caa` | 5198 passed, 15 skipped, 0 failed, 33 warnings | 954.89s |

Total gate time 2,963.68 seconds, a 5.8 per cent spread. Every run measured
`tests/` collecting 5213, matching the ratchet, and every run accounted for
every collected case: 5198 + 15 + 0 = 5213.

### Validation, stated precisely

`b115bab` and `1c50680` each passed an independent LOCAL transactional
acceptance gate before publication, inside their own transaction, with rollback
still available. They were NOT independently exercised as separate remote
continuous-integration revisions: a single `git push` delivered all three
commits, and the post-push lineage covered the publication sequence at its final
head. Continuous-integration run `32540357447` completed in 17 minutes 56
seconds and PASSED.

The distinction matters and must not be collapsed in either direction. The final
tree is verified on both interpreters. Each intermediate commit is verified
locally on one.

---

## 2. RUNNER-GATE-METADATA-ORDER-1, closed at both ends

`tests/EXPECTED_SUITE_SIZE` carried, for the 5207 -> 5213 entry at line 6977,
an acceptance line reading `0 passed, 0 skipped, 0 failed`. The transaction
proof record for `f125187` reports the actual gate as 4978 passed, 10 skipped,
0 failed.

The cause was ordering, not transcription. `build_plan()` was called at line 362
with literal zeroes; `_run_suite()` ran at line 388. The postimage was fixed
before the results existed. The false line was therefore the deterministic
output of the architecture rather than a typing error.

**The repair.** The false line is RETAINED with a superseding correction beside
it, because a correction belongs beside a record and never inside it. Fifty-two
other acceptance lines in the same 6,979-line file are true historical records
and were untouched. Seven acceptance lines in that file match no regular shape,
so the edit was a single exact-literal replacement proven to occur exactly once
rather than a parse.

The producer was corrected in the same commit: the acceptance placeholder left
`RATCHET_ENTRY`, and `passed` and `skipped` were removed from `build_plan`,
since a parameter existing only to receive zero is a defect waiting to be
re-enabled.

Byte deltas, computed independently of the installer and agreeing exactly:
ratchet +976, producer +164.

---

## 3. What failed, and why

### An installer was withdrawn before use: INSTALLER-POSTCOMMIT-ROLLBACK-1

The first D0/D3 installer described itself as reversible and restored
timestamped backups on any exception. Its exception handler stayed armed across
two operations that follow a successful `git commit` -- deleting the temporary
message file, and removing its own `.bak_` files. A failure in either would have
restored pre-commit content while HEAD had already advanced: transactional split
brain.

Traced precisely: the final status check is already safe, because
`backups.clear()` precedes it and the handler then iterates an empty dictionary.
The genuine window is exactly two operations wide. A two-operation window that
produces an unrecoverable inconsistency is still a defect.

The deeper fault was architectural. The repository already owns a crash-safe
transaction primitive with write-ahead ordering, fsynced preimages, a persisted
manifest and reconstruction after process death. The withdrawn installer
hand-rolled a backup scheme instead -- in the same commit that installs a record
about one semantic concept having one typed owner.

The replacement uses `RepositoryTransaction` and separates filesystem
installation from git publication. After `tx.commit()` the journal is destroyed,
so a post-commit content restore is structurally impossible rather than merely
avoided. A publication failure yields `INSTALL_APPLIED_PUBLICATION_PENDING`,
which is honest and recoverable.

### A guard that invalidated its own sequence: INSTALLER-HEAD-PIN-BLOCKS-SEQUENCE-1

The replacement installer runs three units in sequence, each advancing HEAD, and
pinned HEAD to a single constant. Unit one therefore invalidated units two and
three by succeeding. The guard fired correctly against its own stated rule; the
rule was wrong.

HEAD equality was never the real invariant. What protects each unit is the
per-file digest pin. The correct invariant is ANCESTRY: the baseline must be an
ancestor of HEAD, which permits our own commits to accumulate while refusing a
divergent line. Version 2 also reports every intervening commit by name and
refuses if any of them touched a file the unit patches.

### Two self-audit findings, repaired before delivery

`PublicationPending` was defined after `main()`. Python resolves names at call
time so it worked, but "it happens to work" is not a standard. And
`adr_dir_pre_existed` was assigned only on the apply path, so the `finally`
block relied on catching a `NameError` on every dry run -- catching an
initialisation gap to keep going is exactly the silent tolerance this project
removes.

### An invocation that depended on shell state

A delivered command block used a relative script path after a `cd` into the
repository, and a shell variable set several commands earlier that no longer
held the virtual-environment interpreter. The project's own standing rule
already forbids both. The script itself was location-independent; only the
invocation was not.

---

## 4. Findings register at session end

### Confirmed

| identifier | one line |
|---|---|
| `RUNNER-GATE-METADATA-ORDER-1` | acceptance metadata rendered before the gate ran; CLOSED by `69ba5f6` |
| `INSTALLER-POSTCOMMIT-ROLLBACK-1` | content restore reachable after a successful commit; installer withdrawn |
| `INSTALLER-HEAD-PIN-BLOCKS-SEQUENCE-1` | a sequential installer pinned to one HEAD invalidates itself |
| `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1` | `_git` returns None on failure and both clean-tree and head-unmoved assertions return early, silently |
| `RESOURCE-HANDLE-LEAK-1` | 869 ResourceWarning occurrences across 51 sites; four in shipped code |
| `KAN-REPAIR-DUAL-AUTHORITY-1` | an in-process repair and a site-packages patch script both live |
| `KAN-IMPORT-SIDE-EFFECT-1` | module-level execution mutates installed-package globals at import |
| `METRICSTATUS-NAME-COLLISION-1` | two different `MetricStatus` enumerations, nine members and two |
| `PHASE-LIST-MEMBERSHIP-OVERLAP-1` | `esm2_delta_norm` is in the guarded contract and in a phase list |
| `ROADMAP-STALE-1` | last delta 2026-08-08; symptom |
| `ROADMAP-ROLE-OVERLOAD-1` | 7,019 lines, 324 headings, at least seven roles; root cause |
| `STATE-STORE-OWNERSHIP-1` | state identity encoded by path convention rather than a registry |
| `AGENT-LIVENESS-SEMANTICS-1` | a liveness gate whose default mode cannot fail on staleness |
| `AGENT-RUNTIME-TIMESTAMP-SEMANTICS-1` | 142 timestamp fields, 50 distinct values; not per-agent execution times |
| `ATTESTATION-SCHEMA-DRIFT-1` | three artifacts declare schema version 1 with two `acceptance` shapes |
| `ATTESTATION-NOT-PRESERVED-1` | three commit messages point at files outside version control |
| `GATE-WARNING-TAXONOMY-MISLEADING-1` | occurrences and sites reported as if comparable |
| `GATE-ENVIRONMENT-SPLIT-1` | local 5198/15/0 against continuous integration 5199/13/1; both sum to 5213 |
| `INVOCATION-DEPENDS-ON-SHELL-STATE-1` | a delivered command block depended on working directory and a stale variable |

`ATTESTATION-SCHEMA-DRIFT-1` was observed, classified, and deliberately NOT
repaired in this unit. It is carried forward. Repairing it inside the historical
narrative would destroy the causality this record exists to preserve.

### Refuted

| identifier | why |
|---|---|
| `GATE-UNDEFINED-METRIC-WARNING-1` | all three occurrences arise inside tests that deliberately construct degenerate single-class cohorts to prove the `UNDEFINED` handling works; no production path emits one |
| `TRANSACTION-ABANDONED-NON-TERMINAL-1` | `ABANDONED` means execution gave up while recovery work remains owed; its declared transition target is `ROLLING_BACK`. Treating it as terminal would conflate process-lifecycle with repository-obligation terminality |

### Undetermined

`PHYLOP-SOURCE-COLLISION-1` -- claimed as current in strategy prose; the
repository records named repairs at `PHYLOP-SOURCE-OWNERSHIP-1`,
`PHYLOPBACKEND-1` and `PHYLOPSWALLOW-1`, measured 2026-08-12. Probably stale.
Not promoted without a reproduction.

`AGENT-FLEET-STALE-1` -- cannot be established until timestamp semantics are
repaired. Twenty of twenty-two last-run values fall inside a 37-second window on
2026-06-20, several sharing a value to six decimal places.

---

## 5. Warnings, triaged into three classes

The gate reported 33 warnings on every run. A `-W always -r wa` capture found
922 occurrences across 874 parsed site lines -- the difference being CPython's
own default filter, which ignores `ResourceWarning` unless development mode is
on. `ResourceWarning` alone accounts for 869 of the 889 difference.

An earlier attribution of the gap to four test modules carrying
`warnings.filterwarnings("ignore")` was WRONG and is corrected here. Those four
modules are still a real defect -- a module-level, category-less, process-wide
ignore executed at collection -- but they are not the explanation.

| class | count | disposition |
|---|---|---|
| semantic scientific | 3 UndefinedMetricWarning | REFUTED as a defect; they are evidence of correct `UNDEFINED` handling |
| correctness | 2 UserWarning | one degenerate-cohort test, one documented LightGBM feature-name warning |
| resource hygiene | 869 ResourceWarning | `RESOURCE-HANDLE-LEAK-1`, four shipped sites |

`RESOURCE-HANDLE-LEAK-1` is not cosmetic. On Windows an open handle blocks
deletion and rename, and `RepositoryTransaction.rollback` performs both. A
leaked handle is a latent cause of the `RECOVERY_REQUIRED` condition the
primitive exists to make rare -- and two of the leaking tests walk the very tree
an installer mutates.

---

## 6. What was measured, and what it cost

Five read-only probes were built and run, none of which wrote into the
repository.

| probe | closed |
|---|---|
| repository census | 111 checks: identity, ratchet, both collection scopes, hygiene, source-contract census, feature contract, text attributes, live ignore ownership, agent gate, documentation freshness |
| closure census | exact D3 source spans, `require_clean_tree`'s true shape, `.gvc-state`, the artifact-root collision, `incomplete_transactions` verbatim, all 324 roadmap headings, the seven unparsed acceptance lines, the phase-list intersection, the import banner's emitting site |
| transaction application-programming-interface census | `tx.create()` creates parents; `tx.commit()` makes no git commit; `InstallPlan` has no positivity constraint |
| warnings and attestations | the warning taxonomy and the attestation field-set reconciliation |
| README and documentation census | the README numbered line by line, `test_readme_claims.py` in full, every reference to the README across the tracked corpus |

Three claims in inherited documentation were corrected against source: the agent
roster is 22 and not 21; the feature-count constant is
`EXPECTED_TABULAR_FEATURE_COUNT`; and the roadmap's most recent open-item
heading reads FIFTY-FOUR items, not the seventeen plus ten carried forward that
memory recorded.

The README was reconstructed byte-for-byte from the probe listing and verified
against its digest, `eb81eb0f8aadf719`, 16,445 bytes -- so every anchor for the
forthcoming edit is proven rather than transcribed.

---

## 7. The knowledge architecture, adopted

The session's largest outcome is not a commit. It is a governing architecture
for where facts live, adopted across generations 6 and 7 of the ruling.

```
README      public scientific identity          low mutability
ROADMAP     current programme state + future    very high
ARCHIVE     development and scientific memory   append-dominant
ADRs        normative decisions                 controlled
registries  structured executable facts         transactional
attestations measured execution evidence        transactional
Git         committed byte history              immutable
```

Placement rule: **who needs this fact, for what decision, over what time
horizon?**

Five laws now govern:

- one semantic concept, one typed owner;
- derived presentation is not the source of truth;
- direct evidence outranks arithmetic reconstruction;
- no assertion may be retired until its invariant has another proven owner
  (`INVARIANT-HANDOFF-1`);
- move guarantees before moving presentation.

Two measured constraints shape the migration ahead. Retiring
`test_readme_claims.py` -- ten collected tests, zero parametrized -- and
installing five public-contract tests is a net minus five, and no installer in
the repository can execute a decreasing unit today, because `build_plan()`
refuses when the delta is not positive. And the suite ratchet must be demoted
conceptually to accidental test-loss detection: it was never a measure of
assurance, and a change that reduces test count can increase it.

An experiment settled how the roadmap archival move must be proven. `git mv`
preserves the blob object identifier exactly, because blobs are
content-addressed -- verified: `f4e1a7a297bb496202e87c8076a2770747db47d0` at
both paths, before and after commit. But git stores no rename entity; the commit
object records a tree and a parent. `git log --follow` is similarity detection,
and it was broken deliberately by renaming with a full rewrite, at which point
history stopped at one commit. So the archival proof asserts blob-identifier
equality and does not rely on rename recognition.

---

## 8. The ruling sequence

`decision.txt` is a rolling name; the newest file on disk is the current ruling.
Seven generations were received, each superseding the last:

| # | date | subject | sha256 | lines |
|---:|---|---|---|---:|
| 1 | 2026-08-20 | transactional installation architecture | `7bb10a21c4064d28` | 560 |
| 2 | 2026-08-21 | integrated conflict rulings, R1-A3 | `114a7f603bc0a9dc` | 1914 |
| 3 | 2026-08-21 | journal placement and roadmap structure | `8e24bfa7aecc5c4b` | 1064 |
| 4 | 2026-08-21 | closure-census rulings | `5c5ad449dc8eb6bf` | 1550 |
| 5 | 2026-08-21 | installer review; withdrawal | `e51771f80c3d38c8` | 1601 |
| 6 | 2026-08-22 | project knowledge architecture | `e1199e564a09b914` | 1159 |
| 7 | 2026-08-22 | authority-migration protocol | `8eb4377dce942d23` | 1031 |

Generation 2's digest is pinned in `HANDOFF_2026-08-21_session-close.md` section
8.2 under the name `decision.txt`. None of these are yet preserved in the
repository; that is D1, and it remains open.

---

## 9. Ending state

```
HEAD                 69ba5f6 == origin/main, ahead/behind 0 0
ratchet              5213
tests/ collects      5213
tests/unit collects  4988
working tree         clean, including untracked
detritus             none
transaction journals 0
architecture records docs/architecture/decisions/  2 files
continuous integration  32540357447 PASSED, 17m56s, at 69ba5f6
```

## 10. Next intended action

Read-only closure census for the invariant-ownership and path-reference
questions, then ADR-0003 establishing authority domains, non-authorities,
archive semantics, `SuiteTransition`, `INVARIANT-HANDOFF-1`, canonical
directories, and typed freshness. Invariant relocation precedes every
presentation move.

## 11. Remaining uncertainty

Whether any test outside `test_readme_claims.py` binds the base-model roster or
the agent registry is UNMEASURED. If none does, deleting those assertions would
remove live invariants rather than relocate them -- and roadmap 6.6a is the
defect in which a thirteen-model ensemble silently became twelve.
