# SESSION 2026-08-22 to 2026-08-23 -- a state model acquires directories

**Author: Monzia Moodie**
**Commits:** `f567381`, `584c3fb`, `29f2969`, `57494e3`
**Ratchet:** 5303 -> 5314 -> 5352
**Preceding head:** `0999af0`
**Ending head:** `57494e3`

> **Record status:** pre-archive-migration. Future archive class: session
> notebook. Migration required.
>
> Fourth record covering 2026-08-22, extending across midnight into 08-23. The
> three earlier records cover `084ece5`..`69ba5f6`, `31c279a`..`f62f40d` and
> `a60f18f`..`88e844e` and are true records of what they cover. The archive is
> append-dominant.

---

## 0. What this session did

It ruled that machine records are not documentation, expressed the new plane's
policy before writing a byte into it, then found and repaired a defect in the
transaction primitive that had been invisible to every check the repository had.

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `f567381` | ADR-0004 | NEUTRAL | 5303 | 5288 passed, 15 skipped, 902.72s |
| `584c3fb` | records boundary | NEUTRAL | 5303 | 5288 passed, 15 skipped, 914.24s |
| `29f2969` | topology repair | ADDITION +11 | 5303 -> 5314 | 5299 passed, 15 skipped, 850.47s |
| `57494e3` | record ontology | ADDITION +38 | 5314 -> 5352 | 5337 passed, 15 skipped, 1180.17s |

---

## 1. TRANSACTION-STATE-MODEL-INCOMPLETE-1

`create()` reaches `_write_durable`, which runs
`path.parent.mkdir(parents=True, exist_ok=True)`. `_restore_target` unlinks a
created file and RETURNS -- no directory handling existed anywhere. So a
rolled-back transaction left behind every directory it had made.

**It was invisible to everything.** Measured immediately after the first
occurrence: `Get-ChildItem -Force -Recurse` returned nothing, and
`git status --porcelain=v2 --untracked-files=all` -- git's strongest untracked
check -- returned nothing, because git does not represent empty directories.
`iter_repository_detritus` looks for backup-shaped FILES.

It also contradicted the module's own stated invariant, *"failure -- the
repository is byte-identical to how it was found"*. An empty directory has no
bytes, so the letter survived and the intent did not.

### Why the suite could not have caught it

The tests modelled repository state as `S = (F, J)` -- selected file bytes and
journal state -- while the contract claims `S = (F, D, T, G, J)`. Directory
topology was never in the model, so the residue was not merely unnoticed, it was
**unrepresentable**. Two tests looked as though they should have caught it:

- `test_a_committed_transaction_leaves_no_artefact` filters `.bak`, `.orig`,
  `.rej`, `.tmp` in a NAME. A directory called `repository_records` matches
  none. It is also the COMMIT path, and the fixture creates `src/` itself, so no
  new directory ever arises in it.
- `test_a_failed_transaction_restores_the_repository_exactly` asserts two file
  states and the journal. No directory assertion, no `rglob`, no topology
  comparison at all.

`TEST-CONTRACT-OVERCLAIM-1`: both names are stronger than their predicates.
`leaves_no_artefact` means `leaves_no_backup_shaped_file`;
`restores_the_repository_exactly` means `restores_file_bytes_and_existence`.

**The project had already learned this lesson elsewhere.** `tests/conftest.py`
carries a July data-pollution guard whose own words are *"none of this appears
in `git status`, because `data/raw/` is gitignored -- the tool that would have
caught it was blindfolded"* and *"a finding in a document is a comment; a finding
that fails a test is a gate."* That guard is `S = (D)` for one directory. The
transaction tests did not inherit it.

### Falsified before repair

Every case was measured against the live module at `584c3fb` BEFORE a line of
the repair was written:

```
parent already exists         topology restored          (control, clean)
one missing ancestor          src/pkg survived
three missing ancestors       src/a, src/a/b, src/a/b/c survived
two targets, one new parent   src/pkg survived, ONCE
fresh-process recovery        src/pkg survived
```

throughout which the existing suite's own assertions PASSED. That is what makes
it a coverage gap rather than a contradiction. The fresh-process case is
decisive: `recover_transaction` reads the manifest alone, so if the directory is
not recorded there, recovery cannot remove it either -- the gap was in the
DURABLE model.

### The repair

Directory-creation INTENTS are recorded in the manifest BEFORE the mutation.
Levels are materialized individually rather than through an opaque recursive
`mkdir`, so each has one recorded intent and one mutation. Rollback is
two-phase: targets, then topology deepest-first via `rmdir`, never recursive
deletion, through the SAME free-standing helper `recover_transaction` uses.

Foreign content is never deleted **and** never reported as a clean rollback.
*"Do not destroy someone else's state"* and *"the pre-state was restored"* are
separate predicates; a safe failure is still a failure, so it becomes
`RECOVERY_REQUIRED`, which is retryable.

### Proven downstream, by code that knows nothing of it

The ontology installer refused at `584c3fb` with *"the created package directory
survived the rollback"* -- a guard written for exactly this. Rebased onto
`29f2969` and re-run unchanged in every other respect, it reported
**`tree clean, no journal, package directory removed -- verified`**, in the dry
run and again in the apply. That is independent evidence, not self-testimony.

---

## 2. The records plane, expressed before it was used

`ADR-0004` at `f567381` ruled that **machine records are not documentation**.
Measured: twenty-six evidence documents committed across SIX directories, two of
which had no plane at all, while eleven install attestations lived outside
version control entirely. `docs/archive/`, assigned DEVELOPMENT_NOTEBOOK, held
three files -- a stranded git worktree recovery artifact.

`584c3fb` then expressed the plane's policy **before its first byte**, because
measurement inverted the planned order:

- `RECORDS-EOL-NORMALIZATION-1`: preserved artifact bytes resolved to
  `text: set, eol: lf` and would have been normalised on checkout.
- `RECORDS-CONTAINER-INCLUSION-1`: `.dockerignore` named no `records/` path, so
  the plane would have shipped inside container images.

Writing preserved bytes under a policy that would normalise them, intending to
fix the policy afterwards, would mean the first artifacts in the archive were
protected by nothing.

**The guard is the effect, not the text.** Fifteen attribute resolutions were
queried from git inside the transaction -- ten confirming the new policy, five
confirming nothing else moved. An earlier draft ordered the `-text` rule FIRST,
where the general `records/**/*.json` rule below it would have won and
reinstated the very normalisation it forbids. Reading the file caught it;
asserting the effect would have caught it independently.

---

## 3. Six defects in my own instruments

This is the largest count in any session record so far, and every one was found
by review or by a failing run rather than by the instrument itself.

| identifier | what it was |
|---|---|
| `PROBE-DIRECTIONAL-LABEL-INVERSION-1` | a falsification probe whose `diff()` was called `after.diff(before)` while its body read `other - self`. Both labels inverted, the verdict read a permanently empty key, and it printed **"FALSIFICATION DID NOT BEHAVE AS PREDICTED"** while the evidence three lines above showed the defect exactly. Its control passed for the WRONG REASON. |
| `PROBE-CLASSIFIER-COARSE-1` | reported `ArchiveRootState: LEGACY_MIXED` from a structural rule; the contents were homogeneous recovery evidence. Structure rendered as semantics. |
| `PROBE-OVERREFUSAL-1` | answered NOT PROVABLE because its filter matched *"mentions the changelog"* rather than *"derives identities from content"*. A refusal is a claim, and a claim is checkable. |
| `PROBE-PACKAGING-SCOPE-INCOMPLETE-1` | matched packaging keywords and therefore never looked at `[build-system]`, leaving the source-distribution conclusion resting on an unmeasured premise. |
| `CLAIM-WITHOUT-MEASUREMENT-1` | I asserted `import platform` was missing and called it a defect I would not ship past. It was at line 76. The audit was right and I overrode it on intuition. |
| `ORDINAL-CLAIM-UNMEASURED-1` | installers print *"the Nth version-2 document"* from a hardcoded ordinal. `29f2969` printed "sixth" and `57494e3` printed "fifth" -- the fifth landed after the sixth, because the ontology unit was authored before the topology repair intervened. Harmless in substance; the string never enters the attestation. A count asserted rather than measured. |

To which must be added a discipline failure that was not an instrument at all:
I measured an uploaded transcript as byte-identical to a previous one and used
that as grounds **not to read it**, then read four fragments of 1,244 lines. In
those unread lines were a fixture running five git commands with no
`check=True`, two parametrizations making the collected count 49 rather than 38,
three autouse fixtures every new test inherits, and the July guard quoted above
that had already reached this session's central conclusion.

A digest establishes that two files are identical. It establishes nothing about
whether either has been understood.

---

## 4. Two findings registered and deliberately deferred

`SUITE-TRANSITION-KIND-INCOMPLETE-1`, verified by execution: a pure rename
produces both added and removed identities with a count delta of zero, and is
expressible only as `DELIBERATE_RETIREMENT` -- `NEUTRAL` refuses at verify,
`ADDITION` at construction. That would record a retirement where nothing was
retired. `IDENTITY_REPLACEMENT` is the missing kind.

This is why `TEST-CONTRACT-OVERCLAIM-1` was repaired by **strengthening rather
than renaming**: strengthening makes the original names true, removes the
overclaim more completely than a rename would, and changes zero node identities,
so no primitive change is smuggled into a repair unit.

`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`: ADR-0002 ruled a typed certification
scope with fail-closed exclusions. Measured 2026-08-22: **zero code references**
anywhere in `src/`, `tests/` or `scripts/`. The topology oracle is therefore
test-local and says so in a comment, rather than quietly becoming a second
exclusion authority.

---

## 5. Findings

### Closed

`TRANSACTION-STATE-MODEL-INCOMPLETE-1`, `TRANSACTION-CREATE-DIRECTORY-RESIDUE-1`,
`TEST-CONTRACT-OVERCLAIM-1`, `TXTEST-FIXTURE-UNCHECKED-GIT-1`,
`RECORDS-EOL-NORMALIZATION-1`, `RECORDS-CONTAINER-INCLUSION-1`,
`PROBE-DIRECTIONAL-LABEL-INVERSION-1`, `PROBE-PACKAGING-SCOPE-INCOMPLETE-1`.

### Open

`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`ORDINAL-CLAIM-UNMEASURED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `ATTESTATION-NOT-PRESERVED-1` (now seventeen
documents outside version control), `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`,
`RESOURCE-HANDLE-LEAK-1`, `ROADMAP-STALE-1`, `ROADMAP-ROLE-OVERLOAD-1`,
`DOWNLOADS-SHADOWS-TOP-LEVEL-MODULES-1`, `METRICSTATUS-NAME-COLLISION-1`,
`KAN-REPAIR-DUAL-AUTHORITY-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`PHASE-LIST-MEMBERSHIP-OVERLAP-1`, `STATE-STORE-OWNERSHIP-1`,
`AGENT-LIVENESS-SEMANTICS-1`, `AGENT-RUNTIME-TIMESTAMP-SEMANTICS-1`,
`GATE-ENVIRONMENT-SPLIT-1`, `PROBE-OVERREFUSAL-1`, `PROBE-CLASSIFIER-COARSE-1`,
`CLAIM-WITHOUT-MEASUREMENT-1`, `PAYLOAD-DELIVERY-UNVERIFIED-1`.

### Not a defect -- a property

`INSTALLER-BASELINE-COLLISION-1`. Every ratchet-moving unit invalidates every
other pending ratchet-moving unit's baseline, because two units cannot both
render the counter from *one measured count* if neither has seen the other. The
ontology installer refused on it and was rebased by changing two constants.
Observed twice. Pending units must be rebased in dependency order.

---

## 6. Ending state

```
HEAD                    57494e3
ratchet                 5352
suite identity digest   70a3b350199cf2ec
gate                    5337 passed, 15 skipped, 0 failed
working tree            clean, including untracked
detritus                none
transaction journals    0
continuous integration  green through 29f2969; 57494e3 in flight
```

Suite identity chain, each link verified from both sides:

```
5213  75fd25f457dfa55d
5222  29978734b1cf6b8a
5237  972c352bd2b7ca08
5314  3e9ebd785ff757f3
5352  70a3b350199cf2ec
```

## 7. Next intended action

The typed installation-attestation archive manifest, then the eleven historical
attestations preserved verbatim under a boundary policy that now protects them,
then D2c as the atomic ROADMAP authority succession.

## 8. Remaining uncertainty

Whether the continuous-integration alert workflow can alert. Its condition was
read and its dispatch path is exercisable, but no run has failed in the visible
window, so the `workflow_run` failure branch remains unexecuted against a real
event payload.
