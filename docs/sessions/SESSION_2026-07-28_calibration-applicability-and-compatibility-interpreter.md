# SESSION 2026-07-28 — calibration applicability correction and the compatibility interpreter (commit 3b-1a)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `b6bf19f`, ratchet 3455
**Roadmap position:** Tier 1 item 6, commit 3b-1a of 3b-1a / 3b-1b / 3b-2
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. How this commit came to exist

Commit 3b was to make the registry authoritative and retire the evaluator's
computation in one step. Before granting the projection authority, it was run in
SHADOW against the frozen legacy oracle, and the two disagreed on six
field-cohort pairs. Had the authority switch happened first, those six would have
appeared as moved values in a commit whose diff was meant to be mostly deletion,
with six plausible causes between wiring, execution, substitution, rounding,
report construction and removal.

The six turned out to be TWO scientifically different questions, not one.

---

## 2. Calibration and discrimination are different estimands

`_requires_calibration_support` required both reference classes. Its docstring
reasoned that a single-class calibration figure is "computable but scientifically
empty" — correctly identifying the value, then dismissing it.

That conflated two estimands. DISCRIMINATION asks whether predictions rank one
class against another; a single-class cohort cannot support it. CALIBRATION asks
whether predicted probabilities match observed event frequencies; a single-class
cohort can.

Measured on an all-negative cohort with every prediction at 0.10:

    occupied bins      1
    observed frequency 0.00
    mean prediction    0.10
    absolute gap       0.10

That number is not empty. It measures systematic OVERPREDICTION of the event
probability in that population. The mirror case, all-positive at 0.90, measures
underprediction.

What a single-class cohort genuinely limits is INTERPRETATION. Those limits now
live in metadata and certification policy rather than in an applicability rule
that declared the arithmetic undefined.

### 2.1 The three axes stayed separate, and the machinery was already there

    status                 ok        the value is correct
    reference_class_support single_class   the structure is recorded, neutrally
    certification_eligible  False     the claim is not admissible
    certification_blocked_by single_class_cohort

The certification block comes from the PRE-EXISTING `_certification_eligibility`
policy, not from anything added here. The applicability rule had been doing
certification's job. A test pins that the blocker derives from cohort facts and
not from the presence of the diagnostic token, so a future rename cannot silently
unblock certification.

---

## 3. Three defects found while implementing

### 3.1 Applicable-verdict metadata was silently discarded

`Applicability` permits metadata on an APPLICABLE verdict. `compute()` merged it
only on the refusal path. The type allowed the structure and nothing consumed it,
so the new calibration diagnostic was accepted and dropped.

Fixed, and hardened beyond the fix: registry-owned keys are now REJECTED on
collision rather than shadowed by merge order. Shadowing would also have
prevented an overwrite, but silently — a descriptor author who believed they were
setting the population scope would receive no signal. The protected set is
DERIVED from what `ctx.support()` supplies rather than hand-listed, so a key
added there is protected the moment it exists.

That derivation immediately caught a collision in this very commit: the
calibration verdict restated `n_classes_observed`, which support already
provides.

### 3.2 The occupancy theorem was necessary but not sufficient

"At least one occupied bin" was implemented as a representation invariant inside
`CalibrationBins` rather than as an applicability condition — it is a THEOREM of
the conditions applicability already checks, and an unreachable applicability
branch would make INSUFFICIENT_SUPPORT mean two different things.

A deliberate break then showed occupancy alone was not enough. Mapping 1.0 to bin
10 of a ten-bin table produced an OCCUPIED, entirely plausible table: expected
0.375, maximum 0.5, status OK, no exception. The occupancy check asks "did
anything land in a bin?"; only a RANGE check asks "did everything land in a VALID
bin?". Both invariants are now enforced, range first, and a violation is FAILED —
an implementation defect — never INSUFFICIENT_SUPPORT, which would blame the data
for a fault in the code.

### 3.3 A patch written against code that does not exist

`compute()` was patched against `**ctx.support(), **eff}` — text from the
WITHDRAWN first commit 3a, whose `_effective_support` work never landed. Only the
single-match assertion caught it. Recorded because it is the third time in this
session a mechanical check has protected against recollection.

---

## 4. The compatibility layer is an interpreter, and was untested

The first sabotage matrix ran 11 mutations and SURVIVED SIX.

The finding is not "six undetected mutations". It is **one architectural blind
spot producing six surviving mutations**: `legacy_projection.py` had no dedicated
test module and borrowed coverage from the calibration suite, which proved the
resolver was not called for calibration and nothing about the resolver's own
contract, the authorisation rules, or the per-field rounding.

That distinction is predictive. It says a dedicated module should collapse the
survivor count — and it did, to two, both of which were defects in the MUTATIONS
rather than further gaps in the code.

### 4.1 What the module became

Not a helper. An interpreter over a declarative policy, so it is tested on
DECISION PATHS rather than outputs:

    UndefinedProjectionRule   a CLOSED vocabulary -- NONE, ZERO
    ProjectionDecision        which rule fired, was it authorised, which source,
                              was rounding applied
    DECISION_MATRIX           the legal state space declared ONCE, tests
                              generated from it

`ProjectionDecision` exists for one specific reason: two different rules
legitimately produce the same legacy scalar.

    constant_classifier.f1     = 0.0   a MEASUREMENT
    degenerate_all_negative.f1 = 0.0   a SUBSTITUTION for canonical UNDEFINED

Identical numbers, opposite meanings. No assertion comparing values can separate
them.

### 4.2 Policy completeness

Every rule member must be reachable from some policy, every policy must name
exactly one member, every member must appear in the decision matrix, and every
policy field must be exercised. That last test FAILED on its first run — three
policy fields had no matrix row — which is exactly what it is for.

---

## 5. Verification

### 5.1 Two oracles, checked independently, correctly disagreeing

    LEGACY REPORT ORACLE   480 values   ZERO movements
    TYPED REGISTRY ORACLE  384 values   10 DECLARED movements

The typed oracle moves because this commit reverses a scientific judgement. The
movements are declared BY IDENTITY, not by count — one metric,
`expected_calibration_error`, across two fixtures, five fields each:

    status                  insufficient_support -> ok
    reason                  calibration_requires_class_support -> None
    value                   NaN -> 0.5
    certification_eligible  None -> False
    metadata                gains reference_class_support and
                            certification_blocked_by

`certification_eligible` moving from None to False is itself informative: the
refusal previously never reached the certification path, so no verdict was
recorded. The metric is now applicable, certification is evaluated, and it is
blocked.

**The fixture was NOT regenerated.** Regenerating it would destroy the only
record of what the registry produced before the correction. It is no longer
merely a regression oracle; it is documentation, and a future reviewer can answer
"what exactly changed when calibration applicability was corrected" without
reconstructing history.

### 5.2 Shadow comparison

    before 3b-1a   6 mismatches
    after  3b-1a   exactly 2, and exactly the right two:
                       degenerate_all_negative/auprc
                       degenerate_all_positive/auprc

Asserted by IDENTITY, not count — the wrong four could have vanished.

### 5.3 Sabotage matrix

Thirteen mutations, **thirteen detected, zero undetected**.

| break | detected | tests |
|---|---|---|
| B1 the both-classes calibration refusal is RESTORED | yes | 6 |
| B2 the neutral diagnostic is dropped | yes | 3 |
| B3 applicable-verdict metadata is discarded again | yes | 3 |
| B4 protected-key collisions are shadowed, not rejected | yes | 1 |
| B5 the occupancy invariant is removed | yes | 1 |
| B6 the range invariant is removed | yes | 2 |
| B7 every result is routed through the resolver, OK included | yes | 5 |
| B8 the resolver accepts an OK result | yes | 1 |
| B9 a substitute fires for a non-UNDEFINED refusal | yes | 3 |
| B10 a substitute fires for an unauthorised reason | yes | 3 |
| B11 prevalence rounding drops from 4 decimals to 5 | yes | 2 |
| B12 the ZERO rule is SWAPPED onto auroc | yes | 1 |
| B13 mcc and f1 authorised reasons are SWAPPED | yes | 3 |

The two SWAP mutations are the most valuable additions. They are structurally
valid edits of the kind maintenance produces, and without a decision matrix
asserting which rule fires for which reason they would produce identical numbers
on every cohort in either oracle.

### 5.4 Every survivor mapped to a missing execution path

| survivor | missing path | test that now covers it |
|---|---|---|
| B7, B8, B9, B10, B11 | the interpreter had no dedicated test module | `test_legacy_projection.py`, 48 tests from the decision matrix |
| B4 | the protected-key check was tested on a SYNTHETIC descriptor | rebuilt through `compute()` on a REGISTERED descriptor, asserting SURVIVAL of every registry-owned key |

And two gaps the rebuilt mutations exposed. **B9 revealed a hole in the matrix
itself**: every non-UNDEFINED row carried an unauthorised reason, so removing the
status condition changed nothing. Authorisation must be CONJUNCTIVE — a cohort
that was merely too small must not receive a substitute authorised for an
undefined mathematical form, even when the reason string matches. Three rows
added. **B7 was malformed**: a ternary semantically identical to the structural
branch, routing no OK result anywhere new.

---

## 6. Files

    src/genomic_variant_classifier/evaluation/legacy_projection.py   NEW
    src/genomic_variant_classifier/evaluation/registry.py            applicability reversal, protected keys
    src/genomic_variant_classifier/evaluation/metrics.py             two representation invariants
    src/genomic_variant_classifier/evaluation/capabilities.py        reference_class_support
    tests/unit/test_legacy_projection.py                             NEW, 48 tests
    tests/unit/test_calibration_binning_convention.py                33 -> 43
    tests/unit/test_metric_registry.py                               44 -> 48
    tests/unit/test_registry_vocabulary_completion.py                declared movement set

Ratchet 3455 -> 3517 (+62), measured by `pytest --collect-only`.

---

## 7. Next

**3b-1b** adds `SINGLE_CLASS_AUPRC` to the closed rule vocabulary: typed AUPRC
stays UNDEFINED with reason `binary_class_support_required`, while the legacy
scalar is DERIVED from the registered prevalence — 0.0 to 0.0, 1.0 to 1.0, never
a table constant — failing closed if prevalence is not OK or is mixed. Shadow
reaches zero.

**3b-2** switches authority and retires the evaluator's computation. Its diff is
almost entirely subtractive, which is the best possible structure for safely
removing duplicated scientific computation.

---

*Written 2026-07-28.*
