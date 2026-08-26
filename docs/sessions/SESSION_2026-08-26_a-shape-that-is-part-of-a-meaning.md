# SESSION 2026-08-26 -- a shape that is part of a meaning

**Author: Monzia Moodie**
**Commit:** `d73f526`
**Ratchet:** 5557 -> 5573
**Preceding head:** `9bc8da0`
**Ending head:** `d73f526`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `d73f526` | ATTESTATION-V3-TYPING | ADDITION +16 | 5557 -> 5573 | 5558p/15s, 1333.0s |

`ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1` closes.

---

## 1. The defect, and the audit that sized it

Version 2 enforced CROSS-FIELD consistency -- counter deltas against identity
deltas, gate totals against collection counts, kind against the observed sets --
and almost nothing about PRIMITIVE TYPES. A digest could be any string, a
timestamp any string, a count any value.

The measurement that mattered was not "would typing be nice" but "what does the
corpus actually look like". Applying the typing
`install_attestation_reconstruction.py` had used since 2026-08-25:

    version-2 documents preserved      8
    their ONLY typing failure          repository.pre_head, post_head
    every other typed field            ALREADY CONFORMED

So the corpus was already fully typed but for ONE FIELD PAIR -- and that pair
was the one that could not be tightened without changing every producer.
MEASURED across the delivered installers: **102 `rev-parse --short` call sites
and ZERO full ones.** Not one installer had ever captured a full object
identifier.

That single measurement decided the unit's shape. Without it I would have been
choosing between "add a regular expression" and "change every installer"
without knowing which I was proposing.

---

## 2. Record both, and bind them

`pre_head` keeps the abbreviation git prints and every historical attestation
carries. `pre_head_oid` carries the full forty characters.

**AND THE TWO ARE CHECKED AGAINST EACH OTHER.** Recording both is worth nothing
if they may disagree -- two independent fields would simply double the surface
for a wrong value. The abbreviation must be a PREFIX of the full identifier,
which is the only relationship that makes the pair evidence rather than
decoration.

A pending install records null for BOTH post identifiers, and one of each is
refused as "a state that cannot exist": half-committed.

### It resolved a real inconsistency

The same repository was typing one concept two ways: `install_attestation`
accepting seven characters while `install_attestation_reconstruction` demanded
forty. They now agree, and version 3 follows the reconstruction's
`commit_oid`-plus-abbreviation pattern.

### Versions 1 and 2 are not migrated

Nine version-1 and eight version-2 documents are preserved. They stay exactly
as emitted, and `validate` refuses to judge them -- the principle the module has
stated for version 1 since 2026-08-22, now applying to two versions.

MEASURED: `test_attestation_archive.py` does NOT import `install_attestation`
at all. It guards the preserved corpus by BYTES and DIGESTS, never by schema,
so this version boundary cannot disturb the documents it refuses to judge.

---

## 3. Both controls fired, in opposite directions

    OLD tests against the NEW module    20 red -- every v2 payload rejected
    NEW tests against the OLD module    32 red, 23 of them the new cases

The first is the migration's blast radius, measured rather than estimated. The
second proves the sixteen new cases are not assertions that would pass either
way.

Eight version-3 rules sabotaged, eight detected. Two are worth noting for the
opposite reason to usual: removing the plan-digest typing turns **41** cases red
and removing target-digest typing turns **29**. Those rules are load-bearing
across the whole file rather than guarded by a single case -- nobody deletes
them by accident.

---

## 4. The installer was the first producer

It calls `git rev-parse HEAD` without `--short` -- the 103rd call site and the
first full one -- and verifies the abbreviation is a prefix before anything else
runs. It reloads the patched module and refuses unless `SCHEMA_VERSION == 3`,
because the module imported at start-up still declares 2 and would produce a
document its own successor rejects AFTER the commit.

The prevalidation uses a SYNTHETIC FORTY-CHARACTER head. The seven-character
placeholder every prior installer used would now be refused for the right
reason and prove nothing about the real document.

The attestation for `d73f526` was audited independently against the typing it
introduces: eighteen checks, all passing, including both prefix relationships.

---

## 5. Errors made, and two corrections to a COMMITTED record

| # | error | how it surfaced |
|---|---|---|
| 1 | My probe filtered on `schema` and not `schema_version`, auditing nine version-1 documents against version-3 typing | their `NoneType` refusals were an artefact of the filter, not a finding |
| 2 | The session record at `9bc8da0` states "eighteen preserved documents were judged against version 2" | MEASURED: EIGHT were. Nine are version 1, which the module explicitly refuses to judge, and one is a reconstruction under a different schema |
| 3 | I stated `CHANGELOG_ENTRY.md` was 3,224 bytes | it is 3,168. The number was typed, not computed, one line below two that were |
| 4 | My first `declared_test_identities` helper refused `ids=REQUIRED_KEYS` | a module-level list of twelve strings is stable and declarable. A predicate that refuses a correct case is a bug, not strictness |

### The corrections, stated plainly

**`PROBE-VERSION-CONFLATION-1`** and **`FIGURE-STATED-WITHOUT-MEASUREMENT-1`**.
Errors 1 and 2 are one mistake made twice: a corpus counted before it was
separated, and then a directory count attached to a claim about validation.
Error 3 is the same family, one day earlier.

The record at `9bc8da0` is NOT amended. It is pinned by digest, preserved in
the records plane, and correcting it in place would make it a record of what I
wish I had written. METHODS M1 established the alternative on 2026-08-24: a
document that erases its own former claims cannot be audited. The correction
lives here, and the wrong figure stays where it was published.

### A hypothesis withdrawn on evidence

Two turns before this record I noted that both sub-band continuous-integration
timings fell on documentation-only commits, and named it "a hypothesis, not yet
a finding". A third documentation-only run came in at 1058s -- the LONGEST of
the five compared:

    documentation-only   836, 1032, 1058 seconds
    code-changing        1044, 1045 seconds

The groups overlap. REFUTED, and withdrawn. Nothing is registered, because
withdrawing a hypothesis on evidence is the whole value of having stated it as
one rather than as a finding.

---

## 6. Findings

### Closed
`ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1`.

### Repaired in tooling, not yet installed
`PAYLOAD-FILENAME-CASE-UNCHECKED-1` and `INSTALLER-PARAMETRIZE-COUNT-REGEX-1`.
`installer_helpers.py` resolves payloads by exact name -- reporting a case-only
mismatch AS a case-only mismatch -- and derives declared node identities by
PARSING, reproducing both real payloads' counts (15 and 57) without running
pytest. Adopted by the next installer.

### Registered
- `PROBE-VERSION-CONFLATION-1` -- a probe counted a corpus before separating it
  by schema version.
- `FIGURE-STATED-WITHOUT-MEASUREMENT-1` -- two figures stated without being
  computed, one of them in a committed record.

### Still open, unchanged
`ATTESTATION-OPTIONAL-SUBSCHEMA-UNOWNED-1`, `DRIFT-1` beyond phase 0,
`METHODS-HISTORICAL-CONFIGURATION-UNATTRIBUTED-1`,
`WORKFLOW-PIN-NODEID-EMBEDS-LINE-NUMBER-1`,
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`, `ATTRIBUTION-STILL-REFUSES-1`,
`TEMPORALCITE-1`, `BASELINE1-SCOPE-INCOMPLETE-1`,
`DRIFT-WORKFLOW-INVOCATION-UNBOUND-1`, `README-PROFILE-PATH-UNVERIFIED-1`,
`PATCHER-SUPERSEDED-STILL-TRACKED-1`, `SCRIPTS-ARE-CRLF-1`,
`ROOT-DIRECTORY-UNGOVERNED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `TRANSACTION-CANNOT-EXPRESS-DELETION-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`DOWNLOADSHADOW-1`, `FABRICATED-OBSERVATION-1`, `DOCLOC-1`,
`GATE-DURATION-INCREASED-1`, `REDIRECT-2>&1-LOSES-OUTPUT-1`,
`PROBE-GLOB-TOO-SHALLOW-1`, `PROBE-CLASSIFIER-COARSE-2`, `PROBE-PATH-ASSUMED-1`,
`PROBE-STATUS-COUNTER-CONFLATES-CLASSES-1`.

---

## 7. Ending state

```
HEAD                    d73f526  (pushed; 9bc8da0..d73f526)
ratchet                 5573
gate                    5558 passed, 15 skipped, 0 failed, 0 errors
attestations            18 preserved (9 v1, 8 v2, 1 reconstruction) + v3 loose
working tree            clean, including untracked
continuous integration  sixteen consecutive push runs green
```

### GATE-DURATION-INCREASED-1, eighth observation

```
LOCAL   892  901  908 | 1403 1354 1364 1570 1400 1410 1305 | 1089 | 1333
```

1333.0s sits INSIDE the 1305-1570 band. The 1089s reading remains a single
point fitting neither band, and no claim is made from it.

### The chain of identity digests

`d73f526`'s before-digest is byte-identical to what `9bc8da0` recorded as its
after-digest, which matched `53d6034`'s, which matched `1ea45de`'s. Four
independently written attestations agree on the suite's identity across three
commit boundaries.

## 8. Next intended action

`ATTESTATION-OPTIONAL-SUBSCHEMA-UNOWNED-1`. `amendments` and
`invariant_migrations` are permitted at the top level without their CONTENTS
being validated -- a closed top-level vocabulary with open nested ones. MEASURED
2026-08-25: exactly one preserved document uses `amendments`, with a
per-artifact shape of `{artifact, finding, kind, preimage_sha256,
postimage_sha256}`.

One document is one worked example, not a schema. Whether that shape is
authoritative or incidental must be measured before it is typed.
