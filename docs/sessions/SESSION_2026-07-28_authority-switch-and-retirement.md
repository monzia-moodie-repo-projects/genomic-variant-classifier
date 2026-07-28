# SESSION 2026-07-28 — the authority switch and evaluator retirement (commit 3b-2)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `2579234`, ratchet 3529
**Roadmap position:** Tier 1 item 6, commit 3b-2 — **the last**
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What this commit does

The registry becomes the only computation path. `ClinicalEvaluator.evaluate` no
longer computes any metric: every flat scalar in the report is a projection of a
typed registry result, and the typed results are carried alongside them at schema
version 3.

Deleted from the report path:

    roc_auc_score(y, p)                     the area under the receiver
                                            operating characteristic curve
    average_precision_score(y, p)           the area under the precision-recall
                                            curve
    matthews_corrcoef(y, (p >= 0.5))        at a threshold visible nowhere
    f1_score(y, (p >= 0.5))                 at the same hidden threshold
    float(np.mean((p - y) ** 2))            an inline Brier expression
    self._calibration_error(y, p)           a private calibration loop
    the method _calibration_error itself

Verified structurally: none of those seven signatures survives in `evaluate()`.

### 1.1 The acceptance criterion

    480 report field values compared
     10 movements, ALL schema_version 2 -> 3, one per cohort
    470 values byte-identical

Not one measured number changed when the report stopped computing them. The
schema version moving is the report stating what it now contains, which is the
one thing a schema version is for, and it is declared BY IDENTITY rather than by
count.

The scikit-learn warning count fell from nine to three. The six that disappeared
— including *"F-score is ill-defined and being set to 0.0"* — were raised by code
that no longer exists.

---

## 2. Why this commit could be almost entirely subtractive

Because the equivalence was proved BEFORE authority transferred. The shadow phase
ran the projection against the frozen legacy oracle across three commits:

    3b-1a   6 mismatches -> 2   calibration applicability corrected
    3b-1b   2 -> 0              the derived single-class AUPRC rule

By the time `project_legacy_fields` was made authoritative, it had already been
shown to reproduce every legacy scalar exactly. Had the switch come first, the
six disagreements would have surfaced as moved values in a mostly-deletion diff
with six plausible causes between wiring, execution, substitution, rounding,
construction and removal.

---

## 3. The guards that keep the retirement retired

Deletion does not keep code deleted. A future edit adding
`f1_score(y, p >= 0.5)` back would be small, plausible, and would produce a number
indistinguishable from the projected one on almost every cohort.

Two independent mechanisms, because each catches what the other cannot:

    the abstract-syntax-tree guard   duplication that is WRITTEN
    the counting wrappers            duplication that is EXECUTED

Static analysis cannot see a kernel reached through a dynamic lookup. Counting
cannot see dead code a future edit will wake.

The static guard is NARROWED to the report-construction path. It does not ban
thresholding across the module: `_find_operating_point` legitimately sweeps
thresholds, and a blanket rule would either fail on it or be weakened until it
caught nothing.

Carried item (o) is discharged. Written earlier it would have tested the intended
ARCHITECTURE against an implementation that contradicted it — a test guaranteed
to fail for a reason nobody intended to fix.

### 3.1 Composition is not duplication

The first counting guard asserted that NO kernel is invoked twice. It failed on
`auprc`, and the failure was correct: `auprc_gain` is a registered metric defined
as `auprc - no_skill_auprc`, so it calls `auprc` by construction. Two
invocations, one per registered metric that needs the quantity.

The guard now declares an explicit composition budget, so a NEW duplicate still
fails while legitimate composition passes — and asserts the budget is fully
consumed, so an allowance that stops describing the registry is caught rather
than left as a blanket licence.

---

## 4. Four defects found while implementing

### 4.1 A scikit-learn-free import chain, broken and restored

`registry.py` defers its `metrics` import inside every predicate and adapter
DELIBERATELY, so the registry imports without scikit-learn. But commit 2b-2's
threshold adapters were built by factories invoked at MODULE SCOPE, and those
factories performed the very import the pattern exists to defer.

Latent while nothing imported `registry` at module level. A real defect the
moment `evaluator` did. Both paths now bind by NAME and resolve at call time.

**My own check said the package imported cleanly.** It did not: `sklearn` was
already loaded in that process, so my blocker only intercepted new imports. The
landed test uses a clean subprocess, which is why it was right and I was not.
Third time this session a harness of mine was weaker than the test it stood in
for.

### 4.2 The typed surface was computed and discarded

Commit 3a introduced schema version 3 as a CAPABILITY and stated that `evaluate`
would emit it once the report became a projection. The report became a
projection — and `metric_results` was never populated, the schema never advanced.
Every call computed the typed results and threw them away.

Found only by chasing a surviving mutation. No guard asserted on the report's
typed surface, so nothing noticed.

### 4.3 A declaration that did not govern its behaviour

The calibration adapters read the module constant `_CALIBRATION_PARAMETERS["n_bins"]`
rather than their own declared `parameters`, so a descriptor could DECLARE twenty
bins and COMPUTE with ten. The threshold metrics were never exposed to this:
commit 2b-2 bound them to one shared `ThresholdParameters` object asserted by
identity. Calibration is now bound the same way — the declaration IS the
parameter, not a description of one.

### 4.4 A miscount of my own

I reported twelve failing tests as "eleven, plus one to diagnose". There was no
mysterious fifteenth; twelve referenced the retired `_calibration_error` and
three were the import guards. Recorded because an unexplained residual is exactly
the kind of thing that should never be left standing.

---

## 5. Tests rewritten, never deleted

Twelve tests referenced `ClinicalEvaluator._calibration_error`. Deleting them
would have discarded the only coverage of the interval convention at every
interior edge — the defect that survived seventeen days.

    the binning tests    now compare the kernel against an INDEPENDENT reference
                         written from the CONVENTION rather than from the code
    the evaluator tests  now assert on the REPORT FIELD, exercising the whole
                         path: registry computation, projection, per-field
                         rounding

That last change surfaced something worth keeping. The assertion first failed at
`0.11483` against `0.11482593...`: moving to the report surface correctly brought
the five-decimal rounding contract into scope.

Three further tests were updated for the declared movement — commit 3a's
`test_no_report_field_moved` and `test_evaluate_still_emits_the_historical_schema_version`,
which pinned 3a's deliberate incompleteness, and two bootstrap-reconciliation
tests pinning the pre-3b-2 schema.

---

## 6. Attribution governs admissibility, never arithmetic

`evaluate` gains an optional `source_id`. Without one the population is
unattributed: no fingerprint, comparison returns UNKNOWN, and
`certification_eligible` is False with `certification_blocked_by =
unattributed_population`.

Measured on the same cohort:

    unattributed   auroc 0.98837   certifiable False   fingerprint None
    attributed     auroc 0.98837   certifiable True    fingerprint sha256:05cd8ac0...

The number is identical. What differs is whether it can support a certified
claim, because a certified claim asserts something about a NAMED set of rows.
This was verified not to disturb the frozen oracle before it was added.

---

## 7. Verification

### 7.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list of 40, all sandbox dependency gaps. Two new failures appeared mid-work and
were resolved, not tolerated.

### 7.2 Sabotage matrix

Eight mutations, **eight detected, zero undetected**.

| break | detected | tests |
|---|---|---|
| B1 the report recomputes F1 directly at a literal 0.5 | yes | 2 |
| B2 the report recomputes the receiver-operating-characteristic area directly | yes | 1 |
| B3 the private calibration loop is reinstated | yes | 1 |
| B4 the projection computes instead of translating | yes | 2 |
| B5 the two calibration errors declare different bin counts | yes | 1 |
| B6 a second threshold is applied during report construction | yes | 4 |
| B7 unattributed populations become certifiable again | yes | 4 |
| B8 `evaluate` fabricates a source identity | yes | 3 |

### 7.3 The first run left two, and both were instructive

**B8 was a real gap that concealed a real defect.** Every guard constructed its
population directly, so nothing observed what `evaluate` did with attribution.
Closing it exposed §4.2 — the typed surface being discarded. The same lesson as
the protected-key guard in commit 3b-1a: test the real execution graph, not a
synthetic stand-in.

**B5 was a matrix-scope error**, not a coverage gap: the suite containing the
test that catches it was not in the run list. Widening the scope exposed §4.3 —
the declaration that did not govern its behaviour.

---

## 8. Files

    src/genomic_variant_classifier/evaluation/evaluator.py   retirement, projection, source_id
    src/genomic_variant_classifier/evaluation/registry.py    deferred imports, certification, bound adapter
    tests/unit/test_computation_path_guards.py               NEW, 16 tests
    tests/unit/test_calibration_binning_convention.py        independent reference
    tests/unit/test_calibration_implementations_agree.py     report-surface assertions
    tests/unit/test_typed_report_surface.py                  declared movement set
    tests/unit/test_bootstrap_reconciliation.py              schema assertions

Ratchet 3529 -> 3545 (+16), measured by `pytest --collect-only`.

---

## 9. Tier 1 item 6 is complete

Fourteen commits, each carrying one independently falsifiable change:

    d3851a3  MetricResult to the vocabulary layer
    a6df4ef  the typed immutable registry
    974d426  controlled metadata vocabulary
    b22012a  fail-closed prediction-input contract
    951fd82  the EvaluationPopulation contract
    683b514  the calibration binning convention
    2c4aa9e  canonical metric descriptor vocabulary
    15ad3f0  descriptor immutability audit
    132bcc2  typed report surface and schema version 3
    b6bf19f  population attribution
    6029d74  calibration applicability and the compatibility interpreter
    2579234  the derived single-class AUPRC rule
    (this)   the authority switch and evaluator retirement

One computation path. One binning. One projection. Every scalar in the report is
a derived view of a typed result that carries its own status, reason,
applicability verdict, population and certification eligibility.

---

*Written 2026-07-28.*
