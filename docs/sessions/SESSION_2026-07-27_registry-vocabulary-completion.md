# SESSION 2026-07-27 — registry vocabulary completion (commit 2b-2)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `683b514`, ratchet 3354
**Roadmap position:** Tier 1 item 6, commit 2b-2
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What this commit is

Not "more descriptors". What is completed is the VOCABULARY every descriptor
speaks, so later additions cannot produce a second dialect in which some
descriptors declare their classification, parameters and provenance while others
leave them implicit.

    ResultKind               what kind of quantity a descriptor produces
    ThresholdParameters      the typed, canonical threshold declaration
    parameters               its immutable, JSON-validated serialisation
    REGISTRY_SCHEMA_VERSION  1 -> 2, enforced at IMPORT for every descriptor
    four new descriptors     maximum calibration error, Matthews correlation
                             coefficient, F1, prevalence

### 1.1 Three layers, and this commit touches only the first

    Layer 1  metric semantics        descriptor, kernel, threshold provenance
    Layer 2  registry orchestration  execution, applicability, certification
    Layer 3  report projection       compatibility report, legacy flat fields

2b-2 completes Layer 1. `ClinicalEvaluator` keeps its own threshold computation
until commit 3 turns Layer 3 into a pure projection. That divergence is
DELIBERATE, which is why the registry's rules are called CANONICAL throughout:
a reader must not mistake a temporary difference for an accident.

The Abstract Syntax Tree guard `test_evaluator_does_not_compute_threshold_metrics_directly`
is NOT written here. Writing it before the evaluator is intentionally retired
would test today's intended architecture rather than today's intended
implementation. It is carried to commit 3.

---

## 2. The acceptance criterion, and why the snapshot came first

The criterion is not "the new descriptors produce the expected values". It is:

> every result that already existed is byte-identical afterwards.

A baseline captured from a frozen implementation and expectations written by the
author of the change are different scientific standards. The snapshot says
"prove nothing moved relative to a frozen implementation". A handwritten
expectation says "prove the implementation agrees with what its author
expected". Only the first can detect a movement the author did not anticipate,
which is the only kind that matters.

The snapshot was therefore captured on the 2b-1 tree BEFORE a line of 2b-2 was
written, and committed as `tests/fixtures/registry_snapshot_2b1.json`:

    8 cohorts, 48 results, 6 metrics
    status coverage: ok 31, undefined 6, insufficient_support 2,
                     failed 6, not_applicable 3

**Result: 48 pre-existing results x 8 fields = 384 comparisons, ZERO
movements.** Status, value with NaN semantics, reason, certification
eligibility, support, population scope, population fingerprint and the whole
metadata mapping. No carve-outs and no expected-change list — a test carrying an
exemption is weaker than one that cannot.

Exactly four names were added and none removed.

---

## 3. Design decisions that were not free choices

### 3.1 `ResultKind` on the descriptor, never in result metadata

Placing it in metadata would perturb every already-serialised result and force
the acceptance test to carry an exemption. It joins the serialised surface at
schema version 3, deliberately. It is CLASSIFICATION, not dispatch: it does not
determine applicability, certification or required inputs, because a
classification that quietly drove behaviour would be a second control path.

### 3.2 A degenerate confusion margin is caught by APPLICABILITY, not by a NaN

`compute` already rules, deliberately, that an APPLICABLE metric returning a
non-finite value is FAILED — *"an implementation defect, not a property of the
cohort. Calling it UNDEFINED would blame the data."*

So the specified route — kernel returns NaN, interpretation maps it to
UNDEFINED — could not work: it would have produced FAILED. A vanishing
confusion-matrix margin IS a property of the cohort, so it is recognised BEFORE
dispatch by the applicability predicate, giving UNDEFINED and honouring both the
ruling and the landed policy.

### 3.3 One `ThresholdParameters` object, shared by identity

The descriptor's serialised mapping, its kernel adapter and its applicability
predicate all reference the SAME instance, and `_validate_registry` asserts that
by identity at import. Three copies of a threshold that merely happen to be
equal today is exactly how a threshold comes to differ tomorrow.

The typed object is the semantics; `to_mapping()` is its serialisation. Code
reads `descriptor.threshold_parameters.threshold` — checkable and refactorable —
rather than `descriptor.parameters["decision_threshold"]`, which silently
returns nothing useful when misspelled.

### 3.4 The operator is provenance, not pedantry

`>=` and `>` differ exactly at `prob == threshold`. With the conventional 0.5
that is the value a maximally uncertain model emits and the value a two-model
average produces whenever the pair disagrees. A threshold without its operator
is incomplete provenance.

### 3.5 Zero denominators are UNDEFINED, and scikit-learn agrees

scikit-learn returns 0.0 for both the Matthews correlation coefficient and F1 on
degenerate cohorts — and raises `UndefinedMetricWarning` while doing so. **Its
own warning is the evidence that the 0.0 is a fabrication.** Reporting it as
observed performance would make a constant classifier that never discriminated
indistinguishable from one that discriminated and found no correlation.

Where scikit-learn is defined, the kernels agree with it bit-for-bit
(`0.932036691576133`).

---

## 4. A universal quantifier that was true only by accident

Two tests from commit 2a asserted that EVERY registered metric refuses on a
cohort with non-finite probabilities. That was true only because every
registered metric happened to consume predictions.

`prevalence` reads reference labels alone. A cohort whose model output is
corrupt still has a perfectly well-defined prevalence, and refusing it would
report a defect in the predictions as a defect in the cohort. Both tests are now
SCOPED to `ResultKind.PREDICTION_METRIC` — which is precisely the distinction
`ResultKind` exists to make expressible — with an assertion that the scoping did
not empty them, and an explicit assertion that prevalence survives.

---

## 5. Verification

### 5.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list: 40, all sandbox dependency gaps. No test was lost; the one name absent from
the baseline is a deliberate rename made while scoping.

### 5.2 Sabotage matrix

Twelve breaks applied, **twelve detected, zero undetected**.

| break | detected |
|---|---|
| B1 Matthews threshold changed to 0.51 | yes |
| B2 F1 operator changed to `>` | yes |
| B3 adapter hardcodes a threshold instead of reading the declaration | yes |
| B4 maximum calibration error computed with a second binning loop | yes |
| B5 prevalence given a probability requirement | yes, at IMPORT |
| B6 prevalence marked a prediction metric | yes |
| B7 descriptor parameters left mutable | yes, at IMPORT |
| B8 threshold source provenance removed | yes, at IMPORT |
| B9 undefined F1 reported as 0.0 | yes |
| B10 one descriptor omitted from the report set | yes, at IMPORT |
| B11 degenerate margin no longer refused | yes |
| B12 expected and maximum calibration error declare different bin counts | yes |

Four of those fire at IMPORT rather than as test failures, because
`_validate_registry` refuses the declaration outright. That is the strongest
detection available: a malformed registry fails the import rather than the run.

### 5.3 THE FIRST RUN LEFT THREE UNDETECTED

**B4 was a real gap, and it was the seventeen-day defect reproduced inside the
test written to prevent it.** `test_the_maximum_calibration_error_reads_the_shared_table`
used a random continuous cohort, where no probability sits exactly on an interior
decade edge, so a second binning loop using the SUPERSEDED left-open convention
produced the identical answer. The test now uses an interior-edge cohort and
first proves that cohort separates the two conventions.

**B9 was a real gap.** The registry refuses degenerate cohorts through
applicability, before dispatch, so no registry-level test ever reached the
kernel's zero-denominator branch and replacing its NaN with 0.0 broke nothing.
The kernels are public functions with their own contract and are now tested
directly.

**B3 was a malformed break** — hardcoding 0.5 when the declared threshold is 0.5
is a no-op. Rebuilt to hardcode a differing value, and then detected.

---

## 6. Files

    src/genomic_variant_classifier/evaluation/registry.py   vocabulary, 10 descriptors, schema v2
    src/genomic_variant_classifier/evaluation/metrics.py    3 kernels + threshold applier
    tests/fixtures/registry_snapshot_2b1.json               NEW, the frozen baseline
    tests/unit/test_registry_vocabulary_completion.py       NEW, 58 tests
    tests/unit/test_metric_registry.py                      schema fields on ad-hoc descriptors
    tests/unit/test_prediction_input_contract.py            scoped to prediction metrics, 27 -> 30

Ratchet 3354 -> 3415 (+61), measured by `pytest --collect-only`.

---

*Written 2026-07-27.*
