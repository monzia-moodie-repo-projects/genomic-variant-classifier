# SESSION 2026-07-29 — the metric registry, commit 1: the catalogue and the confusion family

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `c02a2f2`, ratchet 3711
**Roadmap position:** Priority 1 of `project_metrics.txt` — the canonical metric registry
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. Why this session exists, stated plainly

The handoff of 2026-07-20, at line 454, asked the next session to decide EXPLICITLY
whether to continue the metric registry or go straight at the five deliverables of
its Part One, and warned that the decision "should not be made implicitly twice."

**It was made implicitly a second time.** Three sessions — 28 July, 29 July, and
the first part of this one — went into repairing the metric SURFACE. Sixteen
commits, suite 3,247 to 3,711. None of it was the `METRIC_REGISTRY` of Priority 1.

Asked explicitly this time, Monzia chose the metric registry. This is its commit 1.

## 2. What was measured first

    METRIC_REGISTRY under src/                          0 occurrences
    metrics in the live registry                        10
    of the fifteen the handoff names as missing         13 still missing

And a correction to a figure I had been repeating: the JEPA disk blocker is GONE.

    measured 2026-07-20     10.91 GB free   against ~14.7 GB needed
    measured 2026-07-29     56.01 GB free   a surplus of about 41 GB

Nine days of a stale number quoted as law. One command settled it.

## 3. The catalogue -- absence made visible

`evaluation/catalogue.py`. `project_metrics.txt` specifies sixteen panels; two are
present. The other fourteen were absent AND INVISIBLE: nothing in the code
recorded that they had been specified, so a reader saw ten metrics and no sign
that thirteen more had been asked for.

A missing metric and a metric nobody ever specified look identical. Only one is a
gap. The catalogue now registers every specified metric with a written formula,
value range, direction and implementation status — so an unbuilt metric is a
REGISTERED ABSENCE rather than a silent one. It is the same principle as the
artifact absence vocabulary, applied one level up to the metric catalogue.

**23 specified, 17 built, 6 absent.** The count began this commit at 13.

`direction` is load-bearing rather than decorative: a dashboard sorting by value
cannot infer it from a name, and a Brier score sorted as though higher were better
ranks the worst model first. `brier_resolution` is the one Brier component where
higher IS better, because the decomposition is
`brier = reliability - resolution + uncertainty`.

## 4. The confusion family -- seven of the thirteen

Hand-computed from a fixture with TP=3, FN=1, FP=1, TN=5, then checked:

    sensitivity                 3/4                      0.750000
    specificity                 5/6                      0.833333
    positive predictive value   3/4                      0.750000
    negative predictive value   5/6                      0.833333
    balanced accuracy           (0.75 + 0.8333)/2        0.791667
    positive likelihood ratio   0.75 / (1 - 0.8333)      4.500000
    negative likelihood ratio   (1 - 0.75) / 0.8333      0.300000

Every one matches. All follow the module's established discipline: fail closed on
non-finite probabilities, and **NaN rather than zero** when a margin is empty,
because scikit-learn's 0.0 is indistinguishable from a classifier that was
measured and scored nothing.

**Both the predictive values AND the likelihood ratios are present, deliberately.**
The predictive values depend on prevalence and do not transfer between cohorts; the
likelihood ratios do not depend on it and do. A clinical report needs both — the
first to say what a result means in THIS cohort, the second to say what the test is
worth anywhere. Reporting only one is a common and consequential omission.

They are registered but NOT in the flat report surface. Adding them would move the
frozen 480-value oracle, which is a separate declared change — the same staging
that kept commit 3a's schema introduction apart from 3b-2's authority switch.

## 5. The registry's own validators caught me twice

Both at import time, and both correctly.

**Identity, not equality.** Every applicability predicate must share the SAME
`ThresholdParameters` object as its kernel. My label-only predicates carried none,
and the validator refused: "two thresholds that merely happen to be equal today
will not stay equal." They now take the object explicitly, which is what makes the
identity assertion meaningful rather than vacuous.

**The report surface is protected.** `REPORT_METRIC_NAMES` must match the
descriptors marked for inclusion, and my seven defaulted to `True` — which would
have moved the frozen oracle silently. The validator refused that too.

## 6. Four defects of my own

**Balanced accuracy refused on a perfect classifier.** I gave it the
likelihood-ratio predicate purely to satisfy the identity validator, and it
inherited a refusal at specificity exactly 1.0 — where balanced accuracy is plainly
`(1 + 1) / 2 = 1.0`. That is the same over-restriction corrected in commit 3b-1a,
repeated. Borrowing a predicate because it typechecks is how a metric acquires a
restriction nobody intended.

**Two invented names.** `ThresholdOperator.GREATER_EQUAL` (it is
`GREATER_OR_EQUAL`) and a `rationale=` field (it is `source: ThresholdSource`).
Both from constructing a plausible name instead of reading the class.

**Two catalogue display names disagreed with the registry** — `f1` is "F1 score,
positive class" and `auprc_gain` is "Precision-recall gain over the no-skill
floor". A test caught both, and I took the registry's wording rather than
defending mine.

**And the kernels had no tests.** I hand-verified them in a throwaway probe and
never committed that verification. The sabotage matrix then found three surviving
mutations, all numerical and all dangerous — including the positive likelihood
ratio dividing by specificity rather than one-minus-specificity, which is the exact
misstatement the kernel's own docstring warns against. **A warning in a docstring
is not a check, and an interactive verification that is not committed protects
nothing.** `test_confusion_family_kernels.py` now pins every value by hand
computation and every degenerate case.

## 7. Two landed tests updated deliberately

`test_a_healthy_cohort_is_certification_eligible` asserted that EVERY registered
metric is OK on a healthy cohort — true of ten, false of seventeen. The fixture's
classifier is perfectly specific, so the positive likelihood ratio is genuinely
unbounded. That is a property of the CLASSIFIER, not the cohort. The exclusion is
NAMED, and a second test asserts those two refuse for that stated reason so the
exclusion cannot hide an unrelated failure.

`test_exactly_four_result_names_were_added` became
`test_exactly_the_expected_result_names_were_added`, enumerating all eleven rather
than counting — so a metric appearing or vanishing is named rather than reduced to
an integer a reader cannot check.

## 8. Verification

Regression `FAILED` list byte-identical at 40. **Frozen report oracle moves only
`schema_version`**, commit 3b-2's already-declared field — the report surface is
untouched.

**Sabotage: nine mutations, nine detected, zero undetected, zero anchor misses.**
The first run left three, all numerical, all closed by the kernel tests.

Ratchet 3711 -> 3851 (+140), MEASURED BY FULL COLLECTION.

**The first attempt aborted at the delta gate and rolled back cleanly.** I had
COMPUTED the delta by summing the four modules I edited -- 96 + 29 + 7 + 1 = 133 --
rather than MEASURING full collection, which the standing rule forbids precisely
because a computed sum misses modules that grow without being edited.

The missing seven were `tests/unit/test_prediction_input_contract.py`, which
parametrises over every REGISTERED metric to assert the fail-closed input
contract. Seven new metrics added seven cases automatically:

    test_metric_catalogue                 0 -> 96   (+96)
    test_confusion_family_kernels         0 -> 29   (+29)
    test_registry_vocabulary_completion  62 -> 69   (+7)
    test_prediction_input_contract       30 -> 37   (+7)   <- the miss
    test_metric_registry                 48 -> 49   (+1)
                                                    ----
                                                    +140

That the input contract picked them up unprompted is a good result: the seven
kernels are covered by the fail-closed guard without my writing anything. The
defect was in my accounting, not in the coverage.

## 9. What remains, and what is still needed

Six specified metrics absent: `partial_auroc`, `integrated_calibration_index`,
`adaptive_expected_calibration_error`, and the three Brier components. The Brier
decomposition builds on `CalibrationBins`, which exists; the other three need new
machinery.

**`project_metrics.txt` is still required** for the panel letters. It is 34,678
bytes, was uploaded on 2026-07-20, and that session is not in the transcript store.
Every catalogue entry carries `panel=None` and a test asserts that NONE is
assigned, so a guessed letter cannot slip in. When the document is supplied, that
test is the one to change.

Also still absent from the handoff's registry scope: `OperatingPointMetrics` with
threshold provenance, the clinical panels (decision-curve net benefit,
selective-prediction risk-at-coverage), the typed `BinaryEvaluationReport` that
refuses to serialise without prevalence, sample count and split identity, and a
living glossary generated FROM the registry and pinned by a test.

## 10. Files

    src/genomic_variant_classifier/evaluation/catalogue.py   NEW, the declared catalogue
    src/genomic_variant_classifier/evaluation/metrics.py     seven kernels
    src/genomic_variant_classifier/evaluation/registry.py    seven descriptors, four predicates
    tests/unit/test_metric_catalogue.py                      NEW, 96 tests
    tests/unit/test_confusion_family_kernels.py              NEW, 29 tests
    tests/unit/test_metric_registry.py                       48 -> 49
    tests/unit/test_registry_vocabulary_completion.py        62 -> 69

---

*Written 2026-07-29.*
