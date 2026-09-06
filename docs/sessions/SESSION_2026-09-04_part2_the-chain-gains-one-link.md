# SESSION 2026-09-04 part 2 -- the chain gains one link, and three censuses were wrong

**Author: Monzia Moodie**
**Commits:** `1fcb1c7`, `acd1561`, `2c94ae3`, `c18a1df`, `f22edc5`, `6782617`, `85d0247`, `bc8b6ce`
**Ratchet:** 6136 -> 6237
**Preceding head:** `f9b4075` (the 2026-09-04 part-1 session record)
**Ending head:** `bc8b6ce`, pushed, `+0 -0`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `1fcb1c7` | D-ROADMAP-REPAIR | NEUTRAL | 6136 | 898.1s |
| `acd1561` | D-CORRECTION-2 | NEUTRAL | 6136 | 6121p/15s/33w |
| `2c94ae3` | D-CLAUDE-MD-BINDING | ADDITION +3 | 6136 -> 6139 | 6124p/15s/33w |
| `c18a1df` | D-CORRECTION-3 | NEUTRAL | 6139 | 6124p/15s/33w |
| `f22edc5` | D-ADR-0005 | NEUTRAL | 6139 | 6124p/15s/33w |
| `6782617` | D-P0B1B | ADDITION +39 | 6139 -> 6178 | 6163p/15s/33w |
| `85d0247` | D-P0B-MEASUREMENT | NEUTRAL | 6178 | 6163p/15s/33w |
| `bc8b6ce` | D-P1I | ADDITION +59 | 6178 -> 6237 | 6222p/15s/33w |

Eight units. The arc runs from a roadmap repair, through the governance work
the adopted plan calls P0-B, to the first link of the production chain the
plan calls P1.

---

## 1. `2c94ae3` -- the operating manual stops keeping a stale copy

`CLAUDE.md` said `EXPECTED_TABULAR_FEATURE_COUNT` is 97. MEASURED by parse tree
over every tracked Python file: the constant is 95 and `TABULAR_FEATURES` holds
95 literal elements. They agree; the document was stale by two, for seven
weeks.

The document that instructs *never work from a remembered state* was keeping
one. `tests/unit/test_claude_md_claims.py` now binds it, so the count is
rendered in four places from one measured source: the ratchet, the README
badge, the roadmap row, and the operating manual.

Sabotage-tested six ways before shipping. The matrix re-encountered
`SABOTAGE-HARNESS-STALE-BYTECODE-1`: `range(95)` and `range(94)` produce
same-size files, and Python invalidates bytecode on modification time and size,
so a stale `.pyc` was served when two writes landed within one clock tick.
`BYTECODE-GUARD-PREVENTS-WRITING-NOT-READING-1` -- `-B` and
`PYTHONDONTWRITEBYTECODE` prevent writing, not reading. Only clearing
`__pycache__` works.

---

## 2. `acd1561` and `c18a1df` -- a correction, and a correction to it

`D-CORRECTION-2` withdrew `RUNNER-GATE-METADATA-ORDER-1` from a still-open list
because it had CLOSED at `69ba5f6` fourteen days earlier, and registered
`STALE-BACKLOG-CARRIED-A-CLOSED-FINDING-1` over it.

**The same document carried a closed finding as open, in the next paragraph.**
It stated that `audit_data_tree.py` has *ten mentions and zero invocations* --
true on 2026-08-30, false when written. `AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`
had CLOSED at `fd6cd4e` on 2026-08-30, five days earlier.
`STALE-CLAIM-REPEATED-IN-THE-CORRECTION-ABOUT-STALE-CLAIMS-1`.

`c18a1df` withdraws it, with the closure proven at three levels BY EXECUTION:
`tests/unit/test_data_tree_gate.py`, 14 tests, ALL PASSED, ZERO SKIPPED, in
15.43 seconds. EXISTENCE -- `audit_rows` and `audit_tree` callable. INVOCATION
-- `run_all` drove a monkeypatched gate and the spy recorded the call.
CONSEQUENCE -- the sentinel row returned through `run_all`'s rows, and
`return_code == 2` holds exactly when a FAIL row is present, both from one
`AuditReport`.

The verbose run existed to check the SKIP PATHS. Neither fired. A skipped test
reads as green, and that guard's own text concedes it *must not skip on the
development machine or in continuous integration*.

**Why the original measurement was wrong, which is the reusable part.** The
wiring loads by path through `importlib.util.spec_from_file_location`, so a
scan for `import audit_data_tree` reports ZERO while the gate is demonstrably
called. The same scan counted 31 "invocations" of `preflight_data_guard`, EVERY
ONE A LINE OF MARKDOWN PROSE. Wrong in both directions from one static search.

`c18a1df` also restates fifteen findings from *coherent* to UNDETERMINED BY
AUTHORITY. MEASURED across all branches by fixed-string match: the auditor and
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` are named by five commits each
and are the only two with a `feat` commit among them; fourteen entries are
named ONLY by `docs` commits; THREE are named by NO COMMIT ON ANY BRANCH.

---

## 3. `f22edc5` and `6782617` -- the Observation role acquires a typed owner

ADR-0001 named four machine-readable artifact types on 2026-08-21: Observation,
Finding, Decision, Attestation. MEASURED 2026-09-05: TWO of the four had a
typed owner. `Attestation` has `install_attestation.py` at schema version 3;
`Decision` has `docs/architecture/decisions/` and `test_adr_contract.py`.
`Observation` and `Finding` had none, and ADR-0001's own decision-sequence
preservation manifest returns ZERO files.
`ADR-0001-DECISION-MANIFEST-IS-UNBUILT-1`, registered against ADR-0001.

`f22edc5` is ADR-0005, which declares the owner. `6782617` builds it:
`repository_measurement`, six modules, standard library only, no re-exports --
MEASURED, not preferred: `transactions` 0 re-export lines,
`repository_records` 0, `paths` 0.

Thirty-nine tests including the four sabotage cases the adopted plan requires,
and an isolation guard parsed with an Abstract Syntax Tree and watched failing
in five directions. The guard uses `ast.walk` so a DEFERRED import inside a
function is seen -- a module-scope-only walker would let one indented line
bypass every check -- and it STATES WHAT IT CANNOT DO rather than claiming to
catch dynamic imports.

---

## 4. `85d0247` -- four instruments learn to state their own limits

All four designated probes now emit the wire schema and none imports the
checkout it measures. The canonical parser accepted and round-tripped each.

**The corpus-identity proof.** Two independently written probes declaring
`**/*.md` at `67826176` produced the SAME membership digest
`624bf7ad2f2db4cf`. Equal counts prove nothing; equal digests prove the same
356 files, which is what lets one probe's `does_not_prove` cite the other's
measurement. Recomputed a third time from the domain-separation rule alone.

**One defect, three times, in four instruments.** `Probe_SectionConvention:208`,
`Probe_StillOpenLedger:277` and `Probe_FindingRegister:413` each read
`data = blobs.get(oid); if None: continue` -- a declared corpus member never
read, counted NOWHERE. All four were written the same day from the same shape;
the one that got it right was written LAST. The repair was FORCED, not chosen:
`AnalysisCoverage` refuses `succeeded + failed != attempted`.

**P0-B.4.** Four candidate definitions of "authoritative probe", all failing:
67 files named like a probe across FIFTEEN locations with four not Python;
`scripts/forensics` holding 70 files under SIXTEEN leading verbs; `git grep`
exiting 1 on every registry name; one probe in continuous integration that is
none of the four. `AUTHORITATIVE-PROBE-HAS-NO-DEFINED-POPULATION-1`.

---

## 5. `bc8b6ce` -- evidence that was written comes back

MEASURED 2026-09-06 at `85d0247` by parse tree over all 1,072 tracked Python
files, zero unread and zero parse failures: the seven source-kernel types have
ZERO construction sites under `src/`. Five carried `as_record`; NOTHING
reconstructed an object from one. The `persistence -> reload` link did not
exist.

Thirteen methods added to `provenance/source.py`, one to
`provenance/coordinate.py`. Class counts unchanged at ten and three.

**Reload is not a back door.** Every `from_record` returns `cls(...)`, so
`__post_init__` runs. Nine tests prove a hand-edited record cannot construct
what fresh construction refuses. Replacing `return cls.of(...)` with
`cls.__new__(cls)` FAILS FOURTEEN TESTS; removing the schema-version,
duplicate-role and coordinate checks fails EXACTLY ONE each, which proves each
negative case fires on the invariant it names.

**No digest moved**, proven by building the original and modified packages side
by side and driving both over one manifest. Field sets identical for all eight
dataclasses; `_RECORD_KEYS` is a dataclass field NOWHERE because it carries no
annotation. The counterfactual was measured: an annotated `_KEYS` DOES become a
field and would have entered every digest.

All thirteen frozen v4 cases reload, re-render byte-identically, reproduce
their v4 digests, and every v5 digest differs.

---

## 6. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | `FABRICATED-DIGEST-4`: extended 16 observed characters with 48 invented | a constant-tracing check before the dry run |
| 2 | `FABRICATED-DIGEST-5`: extended 32 with 32, in the installer whose docstring warned about the first | the same check, one unit later |
| 3 | `INVENTED-A-CONSTRUCTOR-ARGUMENT-1`: `SuiteTransition(expected_delta=3)` from a sibling's signature | the dry run crashed at section 4 |
| 4 | Invented the labels "P1-A through P1-E"; the plan defines no such sequence | grepping the plan for `P1` |
| 5 | Said "six source-kernel types"; there are SEVEN | reading `source.py` |
| 6 | `SILENT-NO-OP-REPLACEMENT-1`: two `.replace()` calls changed nothing, unasserted | a post-edit check |
| 7 | `A-PATHSPEC-THAT-MATCHES-NOTHING-IS-SILENT-1`, twice; the second in a command written after registering it | a copy that created an EMPTY directory |
| 8 | `TWO-CHANGES-ONE-DESCRIPTION-1`: a `Path`-recognition widening rode along with a `spec_from_file_location` fix | comparing two census outputs |
| 9 | A check that CANNOT PASS: `evidence.added_nodeids != ids`, tuple against frozenset | the dry run refused |
| 10 | `CONSTRUCTION-AND-IMPORT-CENSUS-MISSES-REFLECTION-1` | the acceptance gate, inside a transaction |
| 11 | Predicted a byte count (`485,191`) instead of computing it; the answer was `485,188` | the dry run |
| 12 | Printed `(was 76)` for a node count I never measured; it was 68 | reading the earlier collection |
| 13 | A stale count in my own installer header: "two REPLACEMENTS" when three | a derived check against the call sites |

**ERROR 10 IS THE EXPENSIVE ONE, AND THE MOST INSTRUCTIVE.** The first
`--apply` of `D-P1I` FAILED its gate on four cases of
`test_the_SEMANTIC_projection_is_unchanged`. Nothing was committed, the
transaction rolled back, and both kernel files were verified restored by
digest.

My census asked WHO CONSTRUCTS these types and WHO IMPORTS them. That module
does neither: it unpickles fixtures and reflects with `hasattr`. A consumer
that never names a type in an import or a call is invisible to both questions,
so the six-module list was a FLOOR, not a census.

The repair then found that FIVE of sixteen corpus entries lack `as_record`, not
the four that failed -- the fifth being a `TransformationIdentity` this change
does not touch. An allowlist built from the four assertion messages would have
been right BY LUCK.

**ERRORS 1 AND 2 ARE ONE FAILURE.** `FABRICATED-DIGEST-4` was caught, its
warning written into the next installer's docstring, and then `-5` was
committed in that same file for a different value. The rule was applied to the
REMEMBERED INSTANCE rather than to the CLASS. `require_full_sha256` and
`_validate_pins()` now make it mechanical, and the structural cause is mine: I
truncate digests in my own verification output and then reuse them.

**ERRORS 6, 8 AND 9 ARE ONE FAMILY.** Each is a second copy of something that
already had one authority. The repair in all three cases was DELETION, not
correction: `_RT`, the widened check, and the unsatisfiable comparison. The
installer now derives the added set as `after.nodeids - before.nodeids`, from
the same snapshots `verify()` compares, so the declaration cannot disagree with
the observation.

**TEN TIMES A CHECK OF MINE WAS THE DEFECTIVE PARTY**, not the code -- banning
a string that survived only in a comment, comparing the wrong dictionary entry,
extracting a code block from the wrong line, exec'ing a module without
`__file__`. Each was found by reading the failure rather than adjusting the
code to satisfy it.

---

## 7. Findings

### Registered
`STALE-CLAIM-REPEATED-IN-THE-CORRECTION-ABOUT-STALE-CLAIMS-1`.
`ADR-0001-DECISION-MANIFEST-IS-UNBUILT-1`, against ADR-0001, not against
ADR-0005.
`AUTHORITATIVE-PROBE-HAS-NO-DEFINED-POPULATION-1`.
`CONSTRUCTION-AND-IMPORT-CENSUS-MISSES-REFLECTION-1`.
`A-PATHSPEC-THAT-MATCHES-NOTHING-IS-SILENT-1`.
`SILENT-NO-OP-REPLACEMENT-1`.
`TWO-CHANGES-ONE-DESCRIPTION-1`.
`FULL-DIGEST-GUARD-IS-MECHANICAL-1`, the repair for the two fabrications.
`ADR-CONTRACT-DOCSTRING-COUNTS-STALE-1`: that guard says *five of the twelve
tests*; enumerated from its parse tree, FIFTEEN tests and SIX negative
controls, and its fixture message says three records are accepted when four
are. Reported, not repaired.

### Closed
`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`, at `fd6cd4e` on 2026-08-30, proven here
at three levels by execution.
`CLAUDE-MD-FEATURE-COUNT-IS-STALE-1`, at `2c94ae3`.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` -- MEASURED open at `85d0247`:
zero production construction sites across 1,072 tracked Python files. `bc8b6ce`
closes ONE of three absent links. The producer and the observable downstream
behaviour remain absent, and the adopted plan names *tests instantiate it* as
explicitly insufficient -- which is exactly what the 59 new tests are.

The remaining entries of the carried list are NOT re-listed here. `c18a1df`
restated fifteen of them from *coherent* to UNDETERMINED BY AUTHORITY, and
re-carrying them as though their state were known would reproduce
`LIST-CARRIED-FORWARD-DRIFTS-BOTH-WAYS-1` on the day it was corrected. Their
status is recorded in
`docs/sessions/CORRECTION_2026-09-04_part3_a-closed-finding-repeated.md`, with
the commit census that produced it.

### Deferred
`Finding` remains without a typed owner, deliberately. 886 identifiers are
named by records and absent from any register; whether they share one lifecycle
is unmeasured, and a monolithic register that falsely normalised heterogeneous
objects would be worse than narrative records.

---

## 8. Ending state

```
HEAD     bc8b6ce, pushed, +0 -0
ratchet  6237, and the README badge and roadmap row agree
gate     6222 passed, 15 skipped, 0 failed, 33 warnings
```

Fifteen gates ran on 2026-09-04, and the warning total held at 33 in every one.
Durations: minimum 842.9 seconds, maximum 1355.5 seconds. NO VALIDATED
PREDICTIVE MODEL EXISTS. I quoted a predictive band three times and was wrong
in both directions -- undershooting the floor twice and overshooting the
ceiling twice -- so the honest statement is the observation set and nothing
more. `GATE-TIMING-NOISE-EXCEEDS-TREND-1` stands, strengthened.

`docs/CHANGELOG.md` grew 632,191 -> 703,201 bytes across thirteen prepends.

## 9. Next intended action

Re-run `Probe_SessionRecordCoverage_2026-08-28.py` to confirm twelve of twelve
WORK commits named, `bc8b6ce` among them.

Then P1 continues. The chain needs a PRODUCER: something under `src/` that
reads a real source artifact, computes its digest, and builds a
`SourceEvidenceManifest`. Then a wiring point and an observable downstream
consequence, which is the only step that closes the finding.

The stray repository at `C:\Users\monzi\.git` remains REGISTERED and NOT
RESOLVED: 849 unreachable objects, 200,027,701 bytes of content whose path
names are unrecoverable, and a `pyproject.toml` beside it. Removal is
destructive and is not mine to perform.
