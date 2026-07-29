# SESSION 2026-07-28 — the shared-population model comparison (CI-q)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `957e33c`, ratchet 3594
**Roadmap position:** CI-q, unblocked by CI-t
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What was wrong

`compare_models` scored several models against one shared `y_true` and produced a
complete, ordered table. **Nothing in that table could demonstrate the models had
seen the same rows.** Both results were unattributed, so `compare_membership`
returned UNKNOWN, and the artifact asserted a ranking whose premise it could not
support.

For a model comparison this is not a refinement. Same-population is the ENTIRE
PREMISE: a ranking of models scored over different cohorts is not a ranking of
models.

And with one corrupt model, measured before this commit:

    good     0.99937
    fair     0.74253
    corrupt  NaN

A ranking was presented. The corrupt model sorted last on a NaN comparison, and a
reader could not distinguish "evaluated and worst" from "never evaluated".

## 2. What was built

**One population, handed over by object.** `compare_models` constructs a single
`EvaluationPopulation` and passes THE SAME OBJECT to every model through a new
`population=` parameter on `evaluate`. Intra-call sameness is proved by
construction, not inferred from equal fingerprints.

**Admissibility before ordering.** The ranking is refused entirely when any
submitted model lacks a valid value for the ranking metric. Not filtered: a
ranking that silently excludes a submitted model is not a ranking of the models
submitted. And no sort runs at all, because sorting with a NaN present places it
last, which visually implies "worst" rather than "not evaluated".

**Admissibility reads the TYPED result.** Measured: `format_ci` renders all four
interval states identically and the certification Boolean is False in all four,
so neither is evidence about the model.

**Two claims kept apart.** `SHARED_BY_CONSTRUCTION` needs no identity;
`VERIFIED_BY_FINGERPRINT` requires an attributed cohort. `compare_membership` is
untouched, because UNKNOWN remains the correct answer to the question it asks and
teaching it otherwise would destroy its only honest answer.

**Three certification axes, never collapsed.** An unattributed shared comparison
is `(True, False, False)`: internally valid, externally unreproducible. One
Boolean would report it as invalid, which is false.

**A versioned metadata sidecar.** Comparison-level facts describe the COMPARISON,
not any model in it; duplicating them per row invites a reader to believe they
could differ between rows. The eleven legacy columns are preserved on grounds of
CHURN — measured, the artifact has no consumers at all.

## 3. The sabotage found that the central claim was false

**B5: the shared population was built and never handed over.** Each model still
constructed its own; fingerprints matched only because the same `source_id` was
passed to each — equal by coincidence, not shared by construction.

My own test counted how many populations were built with the comparison scope and
missed it entirely. The ruling had said to prove this **by identity or a
construction token, not by equal fingerprints**, and I had not. The test now
records the object passed to each evaluation and asserts `received[0] is
received[1]`.

When the fix changed the call site, B5's anchor stopped matching and the matrix
reported ANCHOR-MISS. **An anchor miss is not a detection**, so the mutation was
rebuilt against the real code before the matrix was accepted.

## 4. The register's own predicate was defective

I predicted the register would catch the CI-q divergence. **It did not**, and
that is worse than being caught.

`_condition_q` scanned every file in `src/` for the text `.evaluate(` and asked
whether `source_id` appeared nearby. It matched SIX places, of which exactly ONE
was a real call:

    evaluator.py:28     a docstring example
    evaluator.py:1518   the real call site        <- the only true positive
    registry.py:57,64   prose in a module docstring
    canonical.py:8      prose in a module docstring
    gnn.py:460          self.evaluate(val_dataset) -- a different function

A docstring will never pass `source_id`, so the item would have reported OPEN
forever regardless of the code.

**This is the fourth malformed probe of this session and the same shape as all
the others: a text search over a superset of what the question asked.** It is the
most serious instance, because it was written into the REGISTER — the mechanism
built to stop status drifting away from code. A register whose predicates are
text searches is a register that drifts silently.

The predicate now parses `compare_models` and asks the question the item actually
poses: does it hand a shared population to each evaluation? Verified to
discriminate in both directions — removing the hand-over re-opens the item as a
failure.


## 4a. A defect the full suite could not see

The first package installed cleanly, ran **3610 tests green**, and then **broke
`git add -A`**:

    ?? nul.metadata.json
    error: open("nul.metadata.json"): No such file or directory
    fatal: adding files failed

`test_evaluator_phase5` passes `output_csv=os.devnull`, which on Windows is
`nul` -- a RESERVED DEVICE NAME with no suffix. `Path("nul").with_suffix(
".metadata.json")` therefore produced `nul.metadata.json` in the repository root:
an entry that appears in a directory listing and CANNOT BE OPENED. Git saw it and
could not index it.

**Nothing in the suite could have caught this.** A test that writes to the null
device never reads back what it wrote, so the sidecar's existence was never
asserted and its unopenability never surfaced. Version control caught it, one
layer outside the tests.

`write_csv` now writes no sidecar for a null device -- a metadata file beside a
discarded table is meaningless anyway -- and the detector is verified not to
over-match: `nulls.csv` and `annulment.csv` are untouched, because a fix that
swallowed a real artifact would be worse than the defect it repaired.

## 5. Verification

Regression `FAILED` list byte-identical at 40. The frozen report oracle moves
only `schema_version`, commit 3b-2's declared field.

**Sabotage: ten mutations, ten detected, zero undetected.**

| break | detected |
|---|---|
| B1 sort_values is restored unconditionally | yes |
| B2 the corrupt model is filtered out and the rest ranked | yes |
| B3 admissibility reads the certification Boolean instead | yes |
| B4 an unattributed comparison is given a fingerprint | yes |
| B5 each model builds its own population again | yes |
| B6 the relation is always SHARED_BY_CONSTRUCTION | yes |
| B7 certification ignores attribution | yes |
| B8 the three axes are collapsed in the invariant | yes |
| B9 a refused ranking need not name a blocker | yes |
| B10 the population key becomes a source identity | yes |

## 6. Files

    src/genomic_variant_classifier/evaluation/model_comparison.py  NEW
    src/genomic_variant_classifier/evaluation/evaluator.py         shared population, admissibility
    src/genomic_variant_classifier/evaluation/__init__.py          exports
    tests/unit/test_model_comparison.py                            NEW, 33 tests
    tests/unit/test_evaluator_phase5.py                            ModelComparison surface
    tests/unit/test_carried_item_register.py                       CI-q predicate rewritten
    docs/CARRIED_ITEMS.md                                          CI-q discharged

Ratchet 3594 -> 3627 (+33), measured.

---

*Written 2026-07-28.*
