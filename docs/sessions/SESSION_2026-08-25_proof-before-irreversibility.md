# SESSION 2026-08-24 to 2026-08-25 -- proof must precede irreversibility

**Author: Monzia Moodie**
**Commits:** `7cc213d`, `2e7e435`, `abcb22e`, `47646ef`
**Ratchet:** 5436 -> 5443 -> 5479 -> 5524
**Preceding head:** `6a6ce47`
**Ending head:** `47646ef`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `7cc213d` | README-1 -- command runnability | ADDITION +1 | 5435 -> 5436 | 5421p/15s, 1402.87s |
| `2e7e435` | METHODS M1 -- architecture currency | ADDITION +7 | 5436 -> 5443 | 5428p/15s, 1354.32s |
| `abcb22e` | DRIFT-1 P0 -- readiness | DELIBERATE_RETIREMENT 52/16 | 5443 -> 5479 | 5464p/15s, 1364.02s |
| `47646ef` | P0-R -- evidence reconstruction | DELIBERATE_RETIREMENT 46/1 | 5479 -> 5524 | 5509p/15s, 1570.29s |

Four findings close: `README-1`, `METHODS-CURRENT-ARCHITECTURE-STALE-1`, `DRIFT-1`
phase 0, and `PROOF-AFTER-IRREVERSIBILITY-1`.

---

## 1. The defect that defined the session

`abcb22e` applied eight targets, proved its suite transition by node identity,
passed a 5,479-case acceptance gate, committed -- and THEN refused:

    ATTESTATION INVALID AFTER A SUCCESSFUL COMMIT:
        a deliberate retirement requires a justification

**Publication schema validation sat downstream of the irreversible commit.** A
defect discoverable before it was discovered after, and the repository crossed
the boundary without its publication evidence. `abcb22e`'s message cites a
filename that did not exist.

### The omission was not carelessness

MEASURED: `TransitionEvidence` carries `kind`, counts, digests and OBSERVED
identities. `SuiteTransition` owns the EXPECTED identities and the
`justification`. The installer hand-built the record mostly from the evidence
object, which has **no structural route to `justification` at all**. The field
was not forgotten -- it was unreachable from what was being serialised.

Three vocabularies were being joined by hand, and they genuinely differ:

    PlannedTarget.as_record        relpath, action, sha256, size, reason
    attestation target            path, action, post_sha256, post_size

    TransitionEvidence.as_record  kind, before{count,digest}, after{...},
                                  added_nodeids, removed_nodeids
    attestation suite_transition  nine keys, expected AND observed separately,
                                  plus justification

`_exact_keys` refuses unknown keys as hard as missing ones, so neither
`as_record` output can be used directly. Every installer translated by hand.

---

## 2. P0-R: what can be established later, and what cannot

`gvc.install-attestation` v2 requires `started_at`. MEASURED 2026-08-25, that
value is **unrecoverable within a 1,434-second window**: the installer samples
its clock after the heavy package imports, the apply log's creation time bounds
it only from below, and nothing closes the upper end.

`validate()` would ACCEPT an invented value, because it checks PRESENCE, not
semantic validity. **A schema accepting a false value does not make that value
evidence.**

So a new artifact class, `gvc.install-attestation-reconstruction` v1:

    finished_at   DERIVED_EXACT   2026-08-25T00:24:52Z, by interval squeeze.
                  The committer date of abcb22e and the apply log's last write
                  fall in the SAME SECOND, and the sample happens between them.
                  Derived, not observed -- and the record says which.
    started_at    BOUNDED         [00:00:58Z, 00:24:52Z]. No point estimate.
    plan_digest   UNRECOVERABLE   16 of 64 characters were printed and the plan
                  no longer exists. A prefix is not a digest.

The invented point estimate is **unconstructible**: BOUNDED may not carry a
value, UNRECOVERABLE may not carry data or witnesses, DERIVED_EXACT must state
its resolution and derivation, and `reconstruction_status` is DERIVED from the
fields rather than declared. Status: PARTIAL, for two independent reasons.

### Not an amendment

MEASURED: exactly one preserved attestation uses `amendments`, and its shape is
per-artifact mutation -- `{artifact, finding, kind, preimage_sha256,
postimage_sha256}`. A document never emitted has no preimage and no postimage.
The shape cannot express this case, independently of the missing `started_at`.

---

## 3. The archive held a closed-world assumption, and the first authored record found it

`ArchiveManifest` says the archive may grow. Two binding tests quietly said it
may grow only with objects identical in lifecycle to the seventeen it was born
with:

    test_attestation_archive.py:113   ARTIFACTS.iterdir() -- hardcoded to one
                                      subdirectory, and NOT recursive
    test_attestation_archive.py:128   loops EVERY entry requiring NO trailing
                                      newline -- while the AUTHORING policy
                                      REQUIRES one

Both verified in source before being repaired, and both repaired **semantically**
-- keyed on `ArchiveEntry.reconstructs_missing_artifact`, never on a filename.
ADR-0004 is explicit that provenance is a fact about the artifact and encoding
it in a path loses it the moment the artifact moves.

The reconciliation is now recursive over the role root, which also **refuses an
unindexed record appearing in a third subtree later** -- something neither the
old form nor a two-directory enumeration would do. A complementary positive
assertion proves reconstructions obey the authoring policy, because an
exemption is not a contract. And a vacuity guard refuses the case where every
entry is a reconstruction and the preservation loop examines nothing.

A third assumption was investigated and REFUTED: I inferred `.gitattributes`
left `reconstructions/` unprotected against end-of-line normalisation. `git
check-attr` resolves that path to `text: set, eol: lf` already. Asking git beat
inferring precedence, and no change was needed.

---

## 4. Errors made, and what each cost

| # | error | how it surfaced |
|---|---|---|
| 1 | Declared ADDITION of 36 when 16 pin identities were renamed | the primitive: "a count of +36 cannot distinguish these" |
| 2 | Fabricated two digests, real 16-character prefix plus 48 invented | caught by a coincidental neighbouring edit |
| 3 | Line-feed-only reconstruction of a CRLF file | 740-byte deficit, exactly one per line |
| 4 | Demanded a trailing newline from a file that has none | the installer refused its own correct payload |
| 5 | Anchor appended a newline it already carried | derivation came out ONE BYTE long |
| 6 | `textwrap.wrap` fused "are retired" into "areretired" | verbatim comparison; reads fine at a glance |
| 7 | Hand-built the record my own guard forbids | my own audit, in the unit repairing it |
| 8 | Executed a module before registering it in `sys.modules` | shipped, after writing the reason down twice |
| 9 | Called `as_attestation_record` on the PREIMAGE class, twice | AttributeError after the measurement transaction |
| 10 | Declared 44 identities, ignoring two added and one renamed away | the primitive: "a count of +45 cannot distinguish these" |
| 11 | Docstring claimed removal the code did not perform | reading the helper in full |
| 12 | Probe glob `*.json` at one level, missing 17 artifacts one down | the recursive listing |
| 13 | Classifier called every pytest-default id "positional risk" | 234 flagged; the refusal itself measured ONE |

**Three of these are the same error**: 1, 10 -- a transition counted by hand
rather than derived. **Three more are one error**: 3, 4, 5 -- every property of
an existing file is a property to preserve unless it is the one being repaired.
**And 7, 8, 9 are one rule** now written into the installer: *no method this
unit adds may be called on a class imported from the repository, and no
installer may author a record that has an owner.*

### The general lesson

Writing an explanation into a comment did not prevent recurrence -- defect 8
recurred twice after the reason was recorded. What prevented it was a single
function that cannot be called the wrong way. **A rule enforced by structure
outperforms a rule enforced by memory**, including my own.

---

## 5. REDIRECT-2>&1-LOSES-OUTPUT-1

MEASURED 2026-08-25. The same installer, same repository state, same exit code:

    > file 2>&1     exit 2       0 bytes
    *> file         exit 2   4,231 bytes

Exit 2 is the REFUSED path, which prints a diagnosis before returning, so
output existed in both runs. I diagnosed the empty file three ways and TESTED
the buffering hypothesis, which refuted itself -- 220 bytes appeared mid-run
with and without `-u`.

**STANDING CONSEQUENCE:** every installer and probe invocation uses `*>`, which
redirects all six PowerShell streams. A zero-byte transcript is otherwise
indistinguishable from a process that never ran.

---

## 6. Findings

### Closed
`README-1`, `METHODS-CURRENT-ARCHITECTURE-STALE-1`, `DRIFT-1` phase 0,
`PROOF-AFTER-IRREVERSIBILITY-1`.

### Registered
- `PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1` -- MEASURED across
  TWENTY-TWO installers: every `except PublicationPending` handler writes
  `json.dumps` directly, never through `AttestationDocument`. Fixed by
  centralising publication, not by editing twenty-two scripts.
- `ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1` -- v2 validates cross-field
  consistency but not primitive types: seven-character commit identifiers,
  unvalidated timestamps, no digest-shape check. V3 work; v2 must not become a
  moving target, since preserved documents were judged against it as written.
- `ATTESTATION-OPTIONAL-SUBSCHEMA-UNOWNED-1` -- `amendments` and
  `invariant_migrations` are permitted without their contents being validated:
  a closed top-level vocabulary with open nested ones.
- `REDIRECT-2>&1-LOSES-OUTPUT-1`, `INSTALLATION-ARCHIVE-BINDING-HARDCODES-ARTIFACTS-DIR-1`
  (repaired), `PROBE-GLOB-TOO-SHALLOW-1`, `PROBE-CLASSIFIER-COARSE-2`,
  `WORKFLOW-PIN-NODEID-EMBEDS-LINE-NUMBER-1`, `DRIFT-ALERT-BODY-STALE-1`
  (repaired in abcb22e), `DRIFT-WORKFLOW-INVOCATION-UNBOUND-1`,
  `CONTINUAL-TRAINER-UNTESTED-1`, `README-PROFILE-PATH-UNVERIFIED-1`,
  `PATCHER-SUPERSEDED-STILL-TRACKED-1`, `SCRIPTS-ARE-CRLF-1`,
  `FABRICATED-DIGEST-3`, `GATE-DURATION-INCREASED-1`.

### Refuted
`INSTALLATION-ARCHIVE-CRLF-GUARD-UNPROTECTED-1` -- withdrawn on evidence, not
merely dropped. `git check-attr` resolves the reconstruction path to
`eol: lf` already.

### Open, unchanged
`CONTINUAL-FEATURE-DRIFT-FAILURE-AS-NO-DRIFT-1`, `DRIFT-1` beyond phase 0,
`METHODS-HISTORICAL-CONFIGURATION-UNATTRIBUTED-1`,
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1`, `ATTRIBUTION-STILL-REFUSES-1`,
`TEMPORALCITE-1`, `BASELINE1-SCOPE-INCOMPLETE-1`, `PROBE-SCOPE-BLIND-AUDIT-1`,
`PAYLOAD-DELIVERY-STALE-NAME-1`, `RUN-PLANNING-DOCS-UNMARKED-1`,
`ROOT-DIRECTORY-UNGOVERNED-1`, `EVIDENCE-DISPOSITION-INCONSISTENT-1`,
`ONTOLOGY-ZERO-LENGTH-REFUSAL-1`, `SUITE-TRANSITION-KIND-INCOMPLETE-1`,
`CERTIFICATION-SURFACE-UNIMPLEMENTED-1`, `TRANSACTION-CANNOT-EXPRESS-DELETION-1`,
`ARCHIVE-SEMANTIC-COLLISION-1`, `AF-FIX-WORK-TRACKED-1`,
`POSTFLIGHT-FEATURE-COUNT-STALE-1`, `KAN-IMPORT-SIDE-EFFECT-1`,
`DOWNLOADSHADOW-1`, `FABRICATED-OBSERVATION-1`, `DOCLOC-1`.

---

## 7. Ending state

```
HEAD                    47646ef  (pushed; 2e7e435..47646ef, 45 objects)
ratchet                 5524
gate                    5509 passed, 15 skipped, 0 failed
archive                 18 entries, genesis 17
records plane           artifacts/ (17 preserved) + reconstructions/ (1 authored)
attestation             install-attestation-P0R-...-20260825T055941Z.json, v2, PUBLISHED
working tree            clean, including untracked
continuous integration  green through 2e7e435; 47646ef in flight
```

`GATE-DURATION-INCREASED-1`: four consecutive local gates at 1354-1570s against
an earlier 891-913s band, while hosted continuous integration held 16-18
minutes throughout. The machine, not the suite.

## 8. Next intended action

`CONTINUAL-FEATURE-DRIFT-FAILURE-AS-NO-DRIFT-1`, verified at
`continual_trainer.py:365-367, 390-392, 411`: a bare `except Exception` sets
`drift_report = None`, line 391 collapses None to False, and line 411 renders
"No significant drift detected." Same collapse as the monitor's repaired
`return 0`, third location -- and worse, because `should_retrain` is an OR of
two triggers, so a swallowed feature-drift failure silently reduces a
two-signal decision to one while the record asserts the negative.

It is the last measured fail-open scientific defect still open.
