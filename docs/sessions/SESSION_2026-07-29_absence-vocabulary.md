# SESSION 2026-07-29 — the absence vocabulary (CI-u-2)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `2a1e7f6`, ratchet 3630
**Roadmap position:** CI-u-2, the vocabulary stage
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The defect

`dump_strict_json` refuses a non-finite number, correctly. But the flat report
surface had no way to say a value was ABSENT, so the whole file was rejected
rather than the one field being recorded as missing.

Measured 2026-07-29 at `2a1e7f6`:

    healthy               PERSISTS
    all-negative          REFUSED   auroc, tpr_curve[0], tpr_curve[1]
    all-positive          REFUSED   auroc, fpr_curve[0], fpr_curve[1]
    constant classifier   PERSISTS
    non-finite input      REFUSED   auroc, auprc, mcc, f1, brier_score

**Three of five cohorts produced reports that could not be written at all.** A
scientifically valid evaluation over a degenerate cohort had no artifact.

`constant_classifier` persisting is the control: it has both classes, so `auroc`
is defined even though the classifier is useless. The refusal tracks genuine
undefinedness rather than poor performance.

## 2. Bare null is not enough

A `null` says a value is missing and nothing about why. A reader cannot separate
"mathematically undefined because only one class is present" from "the input was
refused" from "a bug produced nothing" -- and those demand different responses.

    UNDEFINED_ON_COHORT       a property of the DATA; a legitimate finding
    WITHHELD_BY_INPUT_GATE    a property of the MODEL OUTPUT; a defect to fix
    INSUFFICIENT_SUPPORT      the cohort cannot support the estimand
    NOT_APPLICABLE            the quantity does not apply here

Reporting the first two identically tells a reader to investigate the wrong
thing.

## 3. Curve-level absence, decided by measurement

**No curve in any degenerate cohort mixes valid and non-finite entries.**

    all-negative   tpr_curve all-bad 2/2; every other curve clean
    all-positive   fpr_curve all-bad 2/2; every other curve clean
    non-finite     all curves EMPTY -- withheld upstream by CI-t's input gates

Element-level absence would be a representation for a state that CANNOT OCCUR.
The choice is measured, not aesthetic — and a test pins the premise, so if it
ever stops holding the design is revisited rather than quietly extended.

**Absence is per-curve, not per-report.** On an all-negative cohort `tpr_curve` is
absent while `fpr_curve`, `precision_curve` and `recall_curve` remain valid.
Marking the report's curves absent wholesale would discard three usable arrays.

**And the non-finite case is a third state.** Those curves are EMPTY rather than
poisoned, because CI-t withheld them upstream — absent because refused, not
absent because undefined. `n_expected` is what distinguishes a withheld curve
over two hundred rows from an empty curve over an empty cohort.

## 4. Two structures, not one

`FieldAbsence` carries `cause`, `reason`, `detail`. `CurveAbsence` carries
`cause`, `reason`, `n_expected`. A scalar and an array are absent in different
ways, and one map with mixed semantics would invite a reader to treat them alike.

## 5. Verification

Regression `FAILED` list byte-identical at 40. The frozen report oracle moves
only `schema_version`, commit 3b-2's declared field.

**Sabotage: seven mutations, seven detected, zero undetected**, clean on the
first pass.

| break | detected |
|---|---|
| B1 a free-string cause is accepted | yes |
| B2 a legitimate 0.0 is treated as absent | yes |
| B3 a partial curve is written as complete | yes |
| B4 an empty curve is treated as present | yes |
| B5 n_expected is dropped from the curve record | yes |
| B6 the two record types share one shape | yes |
| B7 every value reports as absent | yes |

B2 and B7 matter most: both would DESTROY real measurements rather than merely
fail to record absence. A detector that swallowed a legitimate `0.0` would be
worse than the defect it was written to fix.

## 6. What this commit deliberately does NOT do

It does not wire the vocabulary into `EvaluationReport`. That is u-3, and the
boundary is deliberate: wiring changes what a persisted artifact CONTAINS, which
is a schema change with a read side.

It also requires something this commit cannot supply. **The cause is only
knowable where the refusal happened** — the input gates know they withheld, the
registry knows the cohort was single-class. Reconstructing it later from a `NaN`
would be exactly the inference this vocabulary exists to replace, so u-3 must
thread the gate verdicts through rather than guess at serialisation time.

## 7. Files

    src/genomic_variant_classifier/evaluation/absence.py   NEW
    tests/unit/test_absence_vocabulary.py                  NEW, 25 tests
    docs/CARRIED_ITEMS.md                                  u-2 recorded

Ratchet 3630 -> 3655 (+25), measured.

---

*Written 2026-07-29.*
