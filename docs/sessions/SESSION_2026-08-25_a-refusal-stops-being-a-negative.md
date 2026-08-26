# SESSION 2026-08-25 -- a refusal stops being a negative result

**Author: Monzia Moodie**
**Commits:** `9125400`, `1ea45de`
**Ratchet:** 5524 -> 5524 (NEUTRAL) -> 5542
**Preceding head:** `47646ef`
**Ending head:** `1ea45de`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `9125400` | D-SESSION-10 -- the record of four commits | NEUTRAL | 5524 | 5509p/15s |
| `1ea45de` | CONTINUAL-1 -- drift honesty | ADDITION +18 | 5524 -> 5542 | 5527p/15s, 1399.85s |

`CONTINUAL-FEATURE-DRIFT-FAILURE-AS-NO-DRIFT-1` closes -- the last measured
fail-open scientific defect. `DETECTOR-CONTRACT-COMMENT-STALE-1` and
`STALE-NUMBER-GUARD-CANNOT-SEE-HISTORY-1` close with it.

---

## 1. The defect: a deliberate refusal, inverted

`DriftDetector.check` REFUSES rather than degrading. Measured at
`monitoring/drift_detector.py`:

    393  if missing:
    394      raise KeyError("... Refusing to report partial coverage as a
                            completed drift check.")
    415  if new_arr.shape[1] != self.n_features:
    416      raise ValueError("... would silently pair up the WRONG features.")

`training/continual_trainer.py` caught those refusals with a bare
`except Exception`, logged a WARNING, set `drift_report = None`, and:

    390  feature_drift_triggered = (drift_report is not None and
                                    drift_report.action_required)
    395  should_retrain = feature_drift_triggered or label_drift_triggered
    411  "reason": ... else "No significant drift detected."

**THE ASSESSMENT LAYER'S REFUSAL WAS INVERTED INTO THE EXACT CLAIM IT REFUSED
TO MAKE** -- and written to `decision_<release>.json` at line 448, a durable
artifact, not merely a log line.

### Three aggravations the full read revealed

1. **The `try` spanned an IMPORT.** `from ...api.pipeline import
   engineer_features` sat inside it, so an ImportError -- a deployment fault
   unrelated to drift -- arrived as a scientific negative.
2. **`logger.warning`, not `error`.** The sole trace of a failed scientific
   measurement sat below the level most configurations surface.
3. **`should_retrain` is an OR.** A swallowed feature-drift failure silently
   reduced a two-signal decision to ONE while the record asserted the negative.

### It was the EXPECTED path, not an edge case

`drift_detector.py` records that the Run-15 reference carries 78 features
against a tabular contract of 95. A KeyError is what a live run produces --
and the comment adds that ignoring the difference "is precisely how this
subsystem died the first time."

---

## 2. The fix is this repository's own shape

Not borrowed from `DriftReadinessReason`. That vocabulary answers "why may an
assessment not PROCEED"; this assessment proceeded and raised, and
`drift_readiness.py` states the governing rule itself:

> No layer may author a fact owned by a downstream layer.

The correct precedent was one layer DOWN, in the type that owns the fact:
`DriftReport.joint_tests_run` / `joint_tests_reason`, which exists because "if
a profile-driven run quietly substituted a benign p-value there, that
escalation would be permanently disarmed WHILE APPEARING TO WORK."

`decision` now carries `feature_drift_checked` and
`feature_drift_not_checked_reason`. The not-checked entry is appended FIRST, so
the `else` branch -- "No significant drift detected." -- is unreachable when the
check did not run, while remaining reachable when it ran and found nothing. The
`reason` expression itself is BYTE-IDENTICAL to the preimage's: the existing
structure was already correct and simply had nothing to say.

---

## 3. The tests forced a design change, and that is their value

`CONTINUAL-TRAINER-UNTESTED-1` confirmed by enumeration: of three test files
mentioning the trainer, one cites it in a docstring, one lists it as a
FORBIDDEN import in a layering assertion, and one states in prose that it "has
no test coverage." Three independent documents, one gap.

**The first draft TRANSCRIBED the decision expression into the test.** A
sabotage matrix showed the weakness exactly: deleting the not-checked branch
from the module left every behavioural test GREEN, because none of them ran
module code. Only the transcription pin fired.

A guard that cannot observe the thing it guards is the shape this repository
keeps finding, and a transcription is that shape in test form. So the decision
was hoisted into `render_retraining_decision`, a pure keyword-only module-level
function, and the tests CALL it. The same sabotage now turns FOUR cases red.

Two structural guards close the loophole: `run` must delegate to it, and the
honest-negative string may exist at exactly ONE site.

**Ten guards sabotaged, ten detected** -- after replacing TWO INVALID mutations:
one removed an import rather than moving it, one kept the call it claimed to
delete. Distinguishing "my sabotage did nothing" from "nothing checks this" is
now the routine step it should always have been.

Against the preimage the tests raise a COLLECTION ERROR rather than assertion
failures, because `render_retraining_decision` does not exist there. A stronger
form of "the preimage fails", and a DIFFERENT one -- reporting it as "four red"
would have been wrong.

---

## 4. Two findings found while repairing, and closed in the same unit

**DETECTOR-CONTRACT-COMMENT-STALE-1.** `drift_detector.py` cited a 97-feature
contract. MEASURED: `EXPECTED_TABULAR_FEATURE_COUNT` is **95**, defined once at
`models/variant_ensemble.py:193`, where `TABULAR_FEATURES` holds exactly 95
entries. And 97 was ITSELF a figure a preflight gate had already corrected --
`SESSION_2026-08-02_pre1-preflight-contract-gate.md` records "97-feature
contract (88 + 3 + 6)" being replaced by "95, 86 + 3 + 6". The superseded
number survived because NOTHING BINDS A COMMENT TO A CONSTANT.

The arithmetic moved with it: 97 - 78 = 19, but 95 - 78 = 17. Correcting the
contract alone would have replaced one stale number with an inconsistent pair.

**STALE-NUMBER-GUARD-CANNOT-SEE-HISTORY-1**, found while writing that repair.
The first guard rejected ANY line containing "97" and "feature" -- which would
have forbidden the sentence recording the correction, contradicting what
METHODS M1 established a day earlier: "a document that erases its own former
claims cannot be audited." The guard now distinguishes a CLAIM from a RECORD,
and `test_that_guard_accepts_a_HISTORICAL_mention` proves the permissive
direction, because a stale-number rule that cannot be satisfied by an honest
correction pushes the next author to DELETE the record instead of marking it.

**Then I reproduced the identical defect in the installer**, within the hour,
and it refused: "the postimage still says 'ignore 19'" -- rejecting its own
history sentence. Third instance today of a fix applied in one place and
reintroduced in the next, after the `sys.modules` registration and the
preimage-class method calls.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Probe hardcoded `monitoring/continual_trainer.py` from a day-old memory | the probe reported ABSENT -- correctly. The module is under `training/` |
| 2 | Nearly asserted "ContinualLearner has no callers" from a census that never searched that name | caught before stating it. Zero hits from a query never issued is not absence |
| 3 | Installer demanded a trailing newline from a file that has none | REFUSED its own correct payload -- fourth occurrence of one rule |
| 4 | Installer's stale-number guard could not see history | REFUSED its own correct payload, second time in one unit |
| 5 | Test transcribed the decision instead of executing it | sabotage: module mutated, every behavioural test still green |
| 6 | Two invalid sabotages reported NOTHING FAILED | both were mutations that changed nothing |
| 7 | Placeholder `SuiteTransition(expected_added_nodeids=None)` | TypeError, caught by exercising the real primitive before shipping |
| 8 | Four substring checks flagged narration as code | located each; all were docstrings |
| 9 | `INSTALLER-PARAMETRIZE-COUNT-REGEX-1` -- reported 0 parametrize decorators where there is 1 | the regex matches `ids=(` and the payload uses `ids=[`. Informational only; gates nothing |

**Errors 3 and 4 are the same error**: an installer imposing a convention the
target does not hold. Both refused loudly and both cost one round trip.

**The general lesson, again:** a rule enforced by structure outperforms a rule
enforced by memory. I fixed the history-blind guard in the test file, wrote a
test proving both directions, and then wrote the same broken rule into the
installer.

---

## 6. Findings

### Closed
`CONTINUAL-FEATURE-DRIFT-FAILURE-AS-NO-DRIFT-1`,
`CONTINUAL-TRAINER-UNTESTED-1` (confirmed, then closed by the first 18 tests),
`DETECTOR-CONTRACT-COMMENT-STALE-1`,
`STALE-NUMBER-GUARD-CANNOT-SEE-HISTORY-1`.

### Registered
- `INSTALLER-PARAMETRIZE-COUNT-REGEX-1` -- a false count printed into a
  transcript that becomes part of the record. Gates nothing.
- `PROBE-PATH-ASSUMED-1` -- a probe hardcoded a module path from recollection.
  Repaired: the path is now DERIVED by searching the tracked file list, and the
  probe refuses if the search does not resolve to exactly one file.

### Still open, unchanged
`PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1` (twenty-two installers),
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
`PROBE-GLOB-TOO-SHALLOW-1`, `PROBE-CLASSIFIER-COARSE-2`.

---

## 7. Ending state

```
HEAD                    1ea45de  (pushed; 9125400..1ea45de)
ratchet                 5542
gate                    5527 passed, 15 skipped, 0 failed, 0 errors
working tree            clean, including untracked
continuous integration  twelve consecutive push runs green; 1ea45de in flight
```

`GATE-DURATION-INCREASED-1`: 1399.85s, a fifth consecutive local gate in the
1354-1570s band against an earlier 891-913s one, while hosted continuous
integration held 16-18 minutes throughout. The machine, not the suite.

The `drift_detector.py` derivation produced the SAME digest
(`35ef4ce58858f284`, 40,708 bytes) in the dry run and the apply -- so an
unpinned postimage is not an unverified one when the preimage is pinned, the
anchor is unique, and the replacement is deterministic.

## 8. Next intended action

`PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1`. MEASURED across TWENTY-TWO
installers: every `except PublicationPending` handler writes `json.dumps`
directly to disk, never through `AttestationDocument`. Two publication paths
exist, and only one is validated.

The fix is to abolish installer-owned publication, not to patch twenty-two
scripts -- an installer may REQUEST publication; it may not IMPLEMENT it.
