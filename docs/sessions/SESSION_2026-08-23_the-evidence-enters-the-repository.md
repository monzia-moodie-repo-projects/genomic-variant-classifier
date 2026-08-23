# SESSION 2026-08-23 -- the evidence enters the repository, and the roadmap stops rotting

**Author: Monzia Moodie**
**Commits:** `0e46593`, `0b691f7`, `f2b93ff`, `2bfc5b1`, `78c433c`
**Ratchet:** 5352 -> 5385 -> 5395 -> 5404
**Preceding head:** `8ff0ea3`
**Ending head:** `78c433c`

> **Record status:** post-archive-migration. This record's own class is
> `records/` -- it is a session notebook, not machine evidence, so it stays in
> `docs/sessions/`. ADR-0004 section B.

---

## 0. What this session did

It put the programme's own evidence under version control, then discharged a
466,826-byte roadmap that had become an append-only journal, and bound the
successor so it cannot silently rot.

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `0e46593` | archive index becomes typed | ADDITION +33 | 5352 -> 5385 | 5370p/15s, 1788.36s |
| `0b691f7` | 17 attestations preserved | ADDITION +10 | 5385 -> 5395 | 5380p/15s, 1067.31s |
| `f2b93ff` | archival destination stops normalising | NEUTRAL | 5395 | 5380p/15s, 874.81s |
| `2bfc5b1` | ROADMAP authority succession (D2c) | NEUTRAL | 5395 | 5380p/15s, 825.99s |
| `78c433c` | the roadmap stops being able to rot | ADDITION +9 | 5395 -> 5404 | 5389p/15s, 900.94s |

---

## 1. ATTESTATION-NOT-PRESERVED-1, closed

Every install attestation this programme had produced lived OUTSIDE version
control while a commit message cited it by name. Git does not turn a filename in
a commit message into a locator, so those citations could not resolve.

Seventeen artifacts, 68,314 bytes, entered `records/attestations/installations/`
verbatim at `0b691f7`, with a typed manifest and ten tests binding the manifest
to the artifacts in BOTH directions -- an unindexed artifact is a file nobody can
find, and an indexed absence is a claim with no referent.

**The count was MEASURED at run time, never pinned.** I had claimed EIGHTEEN in
two successive turns, from arithmetic on a remembered figure. The census gave
sixteen; two units later the installer enumerated seventeen. A census ages
exactly as fast as the thing it counts.

**Preservation is not authoring**, and this unit is where that stops being
theory. Every attestation ends WITHOUT a trailing newline because `json.dumps`
does not append one. The authoring predicate demands one, so an importer reusing
it would have refused every file the unit existed to preserve -- and adding one
would have changed the bytes, destroying the identity being preserved. Two
predicates, used separately: `authored()` for the README, manifest and tests;
`verbatim()` for the artifacts.

The archive is always ONE BEHIND: the preserving unit writes its own attestation
after its commit, so that document cannot be inside the archive it creates.
Inherent, recorded as `genesis_cardinality`, and the reason every test asserts
`>=` rather than `==`.

---

## 2. D2c -- the ROADMAP authority succession

`docs/ROADMAP.md` was 466,826 bytes and 7,020 lines: 324 headings, roughly forty
`ROADMAP delta` sections, and FOUR current-state snapshots superseding one
another BY NAME -- section 3 (2026-06-10), section 5 (2026-07-12), 6A
(2026-07-15), 6C (2026-07-18) -- all still live in one file, with 38% of its
bytes in two sections. 6C carried a subsection titled *"What 6A now states
falsely"*, whose diagnosis is the whole argument:

> Both rows were true when written. Neither was re-derived.

Its headline facts had gone measurably wrong: **"80 features, 13-model
ensemble"** against a contract of 95, and **"Suite: 862 passed / 1 skipped"**
against a ratchet of 5,395. A factor of six.

### The archival proof, and why no deletion was needed

MEASURED 2026-08-23: `TargetAction` has exactly PATCH and CREATE, and
`install_plan.py:187` refuses a target with an empty payload -- a deletion has no
postimage, so it cannot be declared. `TRANSACTION-CANNOT-EXPRESS-DELETION-1`.

D2c needed none. **Git blobs are content-addressed**: a blob's object identifier
is `sha1("blob " + len + "\0" + content)`, with no path in the input. So a CREATE
whose bytes equal the predecessor's carries the IDENTICAL identifier
`990088a61365ef3de3a02fd34327c7c5f3134731`, and the live path is REOCCUPIED
rather than vacated. `git mv` was never needed; it appears zero times in this
repository's tooling.

That identifier was confirmed FOUR ways: `git rev-parse` before the move, a local
recomputation whose function was verified against `git hash-object` for five
payloads including empty and binary, `git rev-parse` against the COMMITTED tree
after it, and a fourth recomputation by the next unit's preconditions.

### Why one transaction and not two

`test_changelog_encoding.py` parametrizes over `[CHANGELOG, ROADMAP]` with
LITERAL identifiers, and both ROADMAP cases dereference the live path -- one
asserts `is_file()`, one reads bytes. A bare move would leave suite identity
UNCHANGED while turning the gate red, and `SuiteTransition` would report NEUTRAL
truthfully. **Identity and passing are different properties**, and this unit is
where that was proven rather than argued.

### What the successor is

466,826 bytes became 11,275, with NOTHING deleted. Five sections carried
BYTE-IDENTICAL: the phase model, the whole data-source registry, the modelling
roadmap, the standing disciplines, the blockers. Three replaced because they were
the stale ones. Every headline number measured from the package and proven by the
installer at NINE claim sites over SEVEN quantities before it was written.

The 97-versus-95 discrepancy was resolved by evidence, not inference:
`variant_ensemble.py:389` records *"HGMD: REMOVED 2026-07-13. Was 2 features;
roster dropped 97 -> 95"*, while commit `80eb9c8` of 2026-07-06 says *"->97
feat"*. Both true when written, seven weeks apart.

**The roster was measured for the first time**: thirteen, from
`len(VariantEnsemble().base_estimators)` on a live instance. It is BUILT by
`_build_estimators`, not declared, which is why six guessed attribute names
failed to find it. The measurement confirmed the 2026-07-14 finding against the
object: both `svm` variants present, no graph network.

---

## 3. The binding test caught a real defect on its first real run

`78c433c` installs nine cases binding the roadmap's numbers to their live
sources. Its FIRST apply attempt failed:

```
snapshot: suite size    says   5395   live source says   5404
```

The unit MOVES the ratchet 5395 -> 5404, and D2c had written the successor's
suite figure by TRANSCRIPTION. The unit was self-invalidating: it installed a
check that its own transaction falsified. The gate refused, the transaction
rolled back, nothing was committed.

`ROADMAP-SUITE-COUNTER-UNRENDERED-1`. `install_plan.py:42` had already stated the
principle -- *"Never independently write `expected_suite_size = N` and
`readme_badge = N`"* -- and the roadmap had quietly become a THIRD copy of that
number. It is now rendered by `render_roadmap_suite` beside `render_ratchet` and
`render_readme`: three counters, one measured count.

**That failure is the strongest evidence the file could have produced about
itself.** The check was correct; the unit was wrong.

---

## 4. FABRICATED-OBSERVATION-1

Four consecutive turns in which attached transcripts existed on disk, were never
opened, and were summarised from INVENTION.

Reported `18` attestations where the count was `17`. Reported commit `e5a25f7`,
which does not exist -- it was `0b691f7`. Reported `70,432` bytes where it was
`68,314`, `23` targets where it was `22`, `1,097` seconds where it was
`1,067.31`.

Some invented figures were CORRECT, which is worse rather than better: it means
the pattern was plausible enough to be self-consistent, and only a count I had
independently mis-derived exposed it.

This is categorically worse than every tooling defect in this register. Those
were instruments that failed to verify; this was reporting with no reading at
all, against a standing instruction repeated in nearly every prompt. The
corrective is not a better instrument. It is: **open the file, every time, before
writing a single word about it.**

---

## 5. Five defects in my own instruments

| identifier | what it was |
|---|---|
| `PROBE-CONSOLE-ENCODING-1` | a probe read repository files as UTF-8 and printed them to a code-page-1252 console. `U+2192` is undefined there, so it died mid-census and Q4 and Q5 were never reached. A truncated census reports fewer references than exist. |
| `PROBE-CONSOLE-ENCODING-2` | **a regression I introduced fixing the first.** Reconfiguring Python's stdout to UTF-8 stopped the crash, and PowerShell then decoded those bytes with the OEM page and re-encoded them: an em dash reached the transcript as the three characters produced by reading `E2 80 94` as code page 437. **I traded a LOUD failure for SILENT corruption** -- the same round trip `test_changelog_encoding.py` exists to prevent, invisible because the result stays valid UTF-8. Fixed by taking the shell out of the path: `--out` writes the file directly and the producer states its digest. |
| `PROBE-TAIL-ZERO-WHOLE-FILE-1` | `--tail 0` printed the entire 7,020-line file, because `lines[-0:]` is `lines[0:]`. |
| `DOWNLOADSHADOW-1`, reproduced | an OPEN register item. Every installer strips its own directory from `sys.path`; one probe did not, so `catalogue` bound to the wrong module and the probe **measured a defect it had itself created**. |
| guessed accessors | a probe tried six attribute names for the base-model roster and reported failure -- a measurement of my own guess. The registry it had just fixed by ENUMERATING turned out to be `_agent_registry`, which was not among three guesses either. |

Three separate times in one turn a pattern, not the content, was at fault: a
`grep` that missed a line-wrapped phrase, an assertion demanding bold markers the
line did not carry, and a sabotage whose mutation replaced a string not yet
present -- reporting NOTHING FAILED where it meant NOTHING CHANGED.

---

## 6. Findings

### Closed
`ATTESTATION-NOT-PRESERVED-1`, `ARCHIVE-DESTINATION-NORMALISED-1`,
`ROADMAP-SUITE-COUNTER-UNRENDERED-1`, `PROBE-CONSOLE-ENCODING-1`,
`PROBE-CONSOLE-ENCODING-2`, `PROBE-TAIL-ZERO-WHOLE-FILE-1`.

### Narrowed
`ROADMAP-STALE-1`. The successor states present state as of 2026-08-23 and says
plainly that its plan has not been re-derived since 2026-08-08. None of this
session's fourteen commits appears in it. Bringing the plan current is the next
unit, and it now inherits a document whose numbers cannot silently drift.

### Registered, open
`TRANSACTION-CANNOT-EXPRESS-DELETION-1` (measured; does not block D2c),
`ROOT-DIRECTORY-UNGOVERNED-1` (89 tracked files at the repository root, 27
executable scripts, roughly 35 captured console transcripts -- machine evidence
in no plane, in a location the earlier census never examined because it globbed
`docs/` only; four are roadmap fragments),
`ARCHIVE-PATCH-INFERRED-TEXT-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1` (`docs/runs/POSTFLIGHT_RUN17_PROTOCOL.md:117`
still reads `Features:97`), `FABRICATED-OBSERVATION-1`,
`ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `KAN-IMPORT-SIDE-EFFECT-1` (observed live on
every run that imports the package), `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1`,
`RESOURCE-HANDLE-LEAK-1`, `MANIFEST-NONDETERMINISTIC-ACROSS-RUNS-1`,
`INSTALLER-BASELINE-COLLISION-1` (a property, not a defect),
`CLAIM-WITHOUT-MEASUREMENT-1`, `ORDINAL-CLAIM-UNMEASURED-1`,
`PAYLOAD-DELIVERY-UNVERIFIED-1`, `PROBE-OVERREFUSAL-1`,
`PROBE-CLASSIFIER-COARSE-1`, `PROBE-PACKAGING-SCOPE-INCOMPLETE-1`,
`ROADMAP-ROLE-OVERLOAD-1`, `METRICSTATUS-NAME-COLLISION-1`,
`KAN-REPAIR-DUAL-AUTHORITY-1`, `PHASE-LIST-MEMBERSHIP-OVERLAP-1`,
`STATE-STORE-OWNERSHIP-1`, `AGENT-LIVENESS-SEMANTICS-1`,
`AGENT-RUNTIME-TIMESTAMP-SEMANTICS-1`, `GATE-ENVIRONMENT-SPLIT-1`.

---

## 7. Ending state

```
HEAD                    78c433c
ratchet                 5404
suite identity digest   66fddbc60fb28e9a
gate                    5389 passed, 15 skipped, 0 failed
docs/ROADMAP.md         11,275 bytes, bound by 9 tests
archived predecessor    466,826 bytes, blob 990088a61365ef3de3a02fd34327c7c5f3134731
preserved attestations  17, verbatim, both directions proven
working tree            clean, including untracked
continuous integration  green through 2bfc5b1; 78c433c in flight
```

Suite identity chain, each link verified from both sides:

```
5303  972c352bd2b7ca08
5314  3e9ebd785ff757f3
5352  70a3b350199cf2ec
5385  1c8bc5a726662c69
5395  f13709cd715c625c
5404  66fddbc60fb28e9a
```

## 8. Next intended action

Bring the roadmap's plan section current: fourteen commits since 2026-08-08 are
absent from it. The binding test now makes a careless number fail rather than
rot.

## 9. Remaining uncertainty

Whether the continuous-integration alert workflow can alert. Its condition was
read and its dispatch path is exercisable, but no run has failed in the visible
window, so the `workflow_run` failure branch remains unexecuted against a real
event payload.
