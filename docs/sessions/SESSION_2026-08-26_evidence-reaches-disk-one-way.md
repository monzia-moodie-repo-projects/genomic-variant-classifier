# SESSION 2026-08-26 -- evidence reaches disk one way

**Author: Monzia Moodie**
**Commit:** `53d6034`
**Ratchet:** 5542 -> 5557
**Preceding head:** `66426c7`
**Ending head:** `53d6034`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `53d6034` | PUBLICATION-BOUNDARY | ADDITION +15 | 5542 -> 5557 | 5542p/15s, 1304.80s |

`PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1` closes.

---

## 1. The defect: two paths, one validator

`AttestationDocument` validated and serialised but did not WRITE. So every
caller opened a file itself, and the two paths diverged:

    success   payload -> AttestationDocument -> to_json -> caller writes
    pending   payload -> json.dumps -------------------> caller writes

MEASURED 2026-08-26 across **thirty-three** delivered installers: every single
`except PublicationPending` handler took the second path.

**NOT twenty-two.** The figure carried into this session was a stale census from
2026-08-25 and understated the true count by half. It also includes four
installers written the day before by an author who had just applied the
opposing rule -- *an installer may request publication; it may not implement
it* -- to the transition record and the target record IN THOSE SAME FILES.

### The pending state was always validatable

Read at source, not recalled. `InstallStatus` declares
INSTALL_APPLIED_PUBLICATION_PENDING. `validate` line 148 requires
`publication_error` present exactly when the status is pending. Line 221
refuses a nonzero gate return code ONLY when PUBLISHED. And `_exact_keys`
constrains the KEY SET of `repository`, never the values, so `post_head` may be
null.

The schema anticipated this state from the start. **The pending path simply
never used it.** So this unit routes an existing state through an existing
validator; it invents no shape. Had the answer gone the other way, the unit
would have been "give the pending state a shape that CAN be validated" -- a
materially different piece of work, and the reason the question was measured
before anything was authored.

---

## 2. The repair

`publish(document, destination)` in the module that owns validation. There is
no second function for pending documents: a pending attestation reaches the
validator by being CONSTRUCTED, like any other.

Three refusals, none of them paranoia:

- **a raw dict** -- a dict has not been validated, and the TYPE is what proves
  it was. Accepting a payload here would restore the bypass in one argument.
- **an existing destination** -- evidence is written once; overwriting would
  destroy the record of what an earlier install claimed.
- **a missing parent directory** -- creating one to store evidence hides a
  misconfigured path until an audit cannot find the artifact.

And it **re-parses its own output**, proving the BYTES validate rather than
merely the object that produced them.

### The re-parse nearly failed this repository's own standard

A sabotage matrix showed it could not be made to fire. By the standard
`suite_transition.py` set when it DELETED three unreachable checks -- "defence
in depth that cannot fire is not defence" -- it had to become demonstrable or
be removed.

Measurement found the reachable case: **`AttestationDocument` is a FROZEN
dataclass whose `payload` is a MUTABLE dict.** A document that validated at
construction can be altered afterwards, and `to_json` renders the alteration
faithfully. A test now mutates a validated document, proves `publish` refuses,
and proves no partial artifact is left behind.

### The newline guard is platform-asymmetric, and says so

MEASURED: on Linux `os.linesep` is a line feed, so `newline=None` and
`newline="\\n"` produce IDENTICAL bytes and no test there can distinguish them.
On Windows -- where every attestation to date was written -- `newline=None`
emits CRLF and the assertion fires.

A sabotage of that argument therefore reports NOTHING FAILED in a Linux
container. That is a limit of one environment, not a toothless guard, and the
test records the distinction rather than quietly claiming detection.

---

## 3. Why thirty-three scripts were not the target

Because the thirty-fourth would repeat it. Those files live in a downloads
directory, are historical artifacts of already-committed units, and editing
them changes nothing in the repository.

The durable fix is a boundary IN THE PACKAGE plus a static guard refusing any
module that serialises evidence outside it -- following the pattern
`test_attestation_projection.py` proved on 2026-08-25 can find a PLANTED
offender. The guard is PARSED, not grepped, because this test file's own
docstring names `json.dumps` several times.

Two further cases keep the guard honest: one proves the search finds a planted
offender, and one proves the OWNER still serialises, so its exemption cannot
become decorative.

**The static guard ran against the real package and passed.** Across 197
`json.dump`/`dumps` call sites measured by the probe, no module outside the
owner serialises an attestation. That was a prediction before the gate; it is a
measurement after it.

---

## 4. This installer was the first consumer

Both publication paths call `att.publish()`, and both **reload the module from
disk first** -- the one imported at start-up is the PREIMAGE and has no
`publish()`. That is the rule learned three times on 2026-08-25: no method a
unit adds may be called on a class imported from the repository.

The pending handler is the point. If publication fails there, it prints the
failure and returns 5 **rather than writing the evidence by any other means**.
A unit that installed a publication boundary and then bypassed it in its own
pending path would be the defect committed inside its own repair -- which has
happened twice in this programme.

The attestation for `53d6034` was written by `publish()`. The boundary's first
real use is its own installation record.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Carried a stale census of "twenty-two installers" into a finding | the probe measured THIRTY-THREE. A count six days old, understated by half |
| 2 | A probe's status counter conflated two artifact classes | reported `<none>` for the reconstruction, which correctly carries `reconstruction_status` |
| 3 | Called 1304.80s evidence that GATE-DURATION-INCREASED-1 "weakens" | the bands do not overlap: 892-908s pre-shift, 1305-1570s post. Over-read one datum as a reversal |
| 4 | Compared pytest's internal timing against the wall-clock band | 1289.61s and 1304.80s are two different measurements; I mixed them |

Errors 3 and 4 are one error: **reading a number without checking which
quantity it measures.** Both were caught by tabulating the series rather than
recalling it.

---

## 6. Findings

### Closed
`PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1`.

### Registered
- `PROBE-STATUS-COUNTER-CONFLATES-CLASSES-1` -- a probe counted `status` across
  two artifact classes and reported a well-formed reconstruction as `<none>`.
  Informational.
- `PAYLOAD-FILENAME-CASE-UNCHECKED-1` -- installers match payload names exactly
  and Windows forgives case; a case-sensitive filesystem would refuse.
- `INSTALLER-PARAMETRIZE-COUNT-REGEX-1` -- an informational count printed 0
  where the payload declares 1. The regex matches `ids=(`; the payload uses
  `ids=[`. Gates nothing.

### Still open, unchanged
`ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1`,
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
`PROBE-GLOB-TOO-SHALLOW-1`, `PROBE-CLASSIFIER-COARSE-2`, `PROBE-PATH-ASSUMED-1`.

---

## 7. Ending state

```
HEAD                    53d6034  (pushed; 66426c7..53d6034)
ratchet                 5557
gate                    5542 passed, 15 skipped, 0 failed, 0 errors
working tree            clean, including untracked
continuous integration  fourteen consecutive push runs green
```

### GATE-DURATION-INCREASED-1, tabulated rather than recalled

```
LOCAL   892  901  908  | 1403  1354  1364  1570  1400  1410  1305
HOSTED  13 runs, 836-1085s, median 1039s
```

Three runs at 892-908s, then SEVEN at 1305-1570s. The bands do not overlap and
the lowest post-shift value is 44% above the highest pre-shift one. Hosted
timing held 979-1085s throughout with a single 836s outlier -- a
documentation-only commit. One observation, and no claim made from it.

### The chain of identity digests

`53d6034`'s before-digest is byte-identical to what `66426c7` recorded as its
after-digest, which was itself identical to `1ea45de`'s. Three independently
written attestations agree on the suite's identity across two commit
boundaries. That is what makes the ratchet a chain rather than a series of
unrelated counts.

## 8. Next intended action

`ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1`. Version 2 enforces cross-field
consistency but not primitive types: seven-character commit identifiers are
accepted where the reconstruction schema demands forty, timestamps are
unvalidated strings, and no digest-shape check exists.

Version 2 must NOT become a moving target -- eighteen preserved documents were
judged against it as written. So this is a version 3 with a migration boundary,
not an amendment, and the reconstruction schema built on 2026-08-25 is the
worked example of the typing it needs.
