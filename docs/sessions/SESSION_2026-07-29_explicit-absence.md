# SESSION 2026-07-29 — explicit absence in the artifact (CI-u-3)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `594a6af`, ratchet 3655
**Roadmap position:** CI-u-3, the wiring stage — **CI-u complete**
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The frozen oracle, and what changed

Captured on the untouched tree before any edit:

    cohort                persists  evidence
    healthy               YES       03b67304e8e76f17a2d56309   15,511 bytes
    all-negative          NO        auroc, tpr_curve[0], tpr_curve[1]
    all-positive          NO        auroc, fpr_curve[0], fpr_curve[1]
    constant classifier   YES       0d195a97bf43908b205f1989    6,568 bytes
    non-finite input      NO        auprc, auroc, brier_score, calibration_ece,
                                    calibration_mce, f1, mcc

**Three of five could not be written at all.** After this commit all five persist,
and the legacy report oracle moves only `schema_version`.

The non-finite refusal is **seven** fields, not the five I had been describing —
`calibration_ece` and `calibration_mce` refuse as well. I had carried across the
fact that CI-t withholds the calibration CURVES without checking the scalars.

## 2. The cause is threaded, never inferred

    all-negative / all-positive   auroc absent, UNDEFINED_ON_COHORT
                                  only the one undefined curve marked
    non-finite input              7 scalars absent, WITHHELD_BY_INPUT_GATE
                                  all 6 curves absent
    healthy                       nothing absent

The `NaN` is identical in every case. Only the gate verdict separates a property
of the DATA — a legitimate finding — from a property of the MODEL OUTPUT, which
is a defect to investigate. `label_check`, `probability_check` and
`ranking_check` are CI-t's verdicts, already computed; no inference is performed
at serialisation time.

## 3. Four defects of my own, each found by measurement

**The invariant was VACUOUS as first wired.** `to_serializable` nulled every
declared-absent field and THEN asserted that declared-absent fields were null.
The code had just made the assertion true. A sabotage deleting the call entirely
survived, because no payload could reject it. It now checks the REPORT before
normalising, which is falsifiable — and a fabricated absence entry fails it.

**The scalar predicate tested `is None`** when the report's own representation
uses `NaN`. **The curve predicate tested emptiness** when an absent curve on the
report is `[nan, nan]`, not `[]`.

**The completeness half over-reached.** It demanded an absence record for every
empty curve and fired on legitimately-constructed reports that simply have no
curves. That conflates a NULL SCALAR — a value that went missing and must say why
— with an EMPTY COLLECTION, which is a perfectly good value meaning "no points".
Only the scalar half is completeness; the curve half is consistency only.

**And I mis-stated the acceptance criterion**, claiming both healthy digests must
be byte-identical. That is impossible across a schema bump. The correct criterion
is that NO MEASURED VALUE MOVES, and that holds: only the two new keys and the
version number differ.

## 4. The register caught a real discharge

`CI-u is listed OPEN but its condition no longer holds.` The predicate written in
u-2 tries to serialise a single-class report and expects failure; it now
succeeds. **The register detected a status change made in code before the
document caught up** — which is exactly what it exists to do, and the first time
it has fired on a genuine discharge rather than a synthetic one.

Its predicate is now INVERTED rather than deleted: if a report ever becomes
unpersistable again, that fails as a regression instead of silently reverting.

## 5. Verification

Regression `FAILED` list byte-identical at 40. Legacy report oracle moves only
`schema_version`.

**Sabotage: nine mutations, nine detected, zero undetected, zero anchor misses.**

| break | detected |
|---|---|
| B1 the biconditional is never called | yes |
| B2 the null-side check is neutered | yes |
| B3 the orphan-side check is neutered | yes |
| B4 every cause becomes undefined_on_cohort | yes |
| B5 absent scalars are not nulled | yes |
| B6 absent curves are not emptied | yes |
| B7 the field-absence map is withheld | yes |
| B8 every curve is marked absent | yes |
| B9 a non-finite scalar counts as finite | yes |

Two earlier rounds were discarded rather than accepted: one had four ANCHOR
MISSES after the code was rewritten beneath the mutations, and B7 was initially
a no-op because `{} or {...}` evaluates to the dictionary.

## 6. Files

    src/genomic_variant_classifier/evaluation/evaluator.py   absence maps, schema 4
    tests/unit/test_explicit_absence.py                      NEW, 27 tests
    tests/unit/test_carried_item_register.py                 CI-u discharged
    tests/unit/test_typed_report_surface.py                  schema 4
    tests/unit/test_bootstrap_reconciliation.py              schema 4
    tests/unit/test_computation_path_guards.py               schema 4
    docs/CARRIED_ITEMS.md                                    CI-u discharged

Ratchet 3655 -> 3682 (+27), measured.

---

*Written 2026-07-29.*
