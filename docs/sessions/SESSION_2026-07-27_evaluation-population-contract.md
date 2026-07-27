# SESSION 2026-07-27 — the evaluation population contract (commit 2a-1)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `b22012a`, ratchet 3273
**Roadmap position:** Tier 1 item 6, commit 2a-1 — the population contract
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What this commit completes

Ruled 2026-07-27:

> No numerical kernel may select, filter, normalise or redefine its evaluation
> population. Population construction is an explicit upstream operation, and
> every result must describe exactly that population.

Commit 2a enforced that for PREDICTIONS: non-finite scores and probabilities now
fail closed. It deliberately left label eligibility standing, because withheld
labels are first-class in this project and selecting on them is a legitimate
population decision that could not simply be deleted. It was parked behind a
named transitional selector, `metrics.select_finite_reference_labels`, so this
commit would have one precise deletion target rather than an anonymous clause.

That selector is now retired. Label eligibility is an explicit, recorded
restriction of an `EvaluationPopulation`.

---

## 2. `EvaluationPopulation`

A frozen, immutable claim about which rows a number describes. Narrowing is the
only operation it offers: there is no widen, reorder, duplicate, relabel or
repair, because each would break the claim.

    attempted_cohort(n=1000) -> label_eligible(n=980, -20 reference_label_withheld)

### 2.1 Addressing model

`indices` are absolute positions into the source frame, never into a parent, so
`take` is one fancy-index with no chain to walk and a population five narrowings
deep still states plainly which original rows it covers. The parent link is
provenance, not address translation.

### 2.2 Invariants, all raising, none warning

| invariant | why it is load-bearing |
|---|---|
| may only STRICTLY narrow | an unchanged population must not acquire artificial lineage claiming a restriction that never occurred |
| child membership is a genuine SUBSET | smaller, ordered, unique and in range is not enough: parent `[0,2,4,6]` with child `[1,3]` satisfies all four and re-admits removed rows |
| indices strictly increasing | a population is a SET; permitting an order invites code to depend on it |
| no duplicates | a duplicated row is counted twice in every metric |
| reason required exactly when a parent exists | a narrowing that cannot say why is the defect the class prevents |
| root must contain every source row | a partial population must be derived, so its reason and parent are recorded |
| child inherits `source_id` and `n_source` unchanged | a narrowing cannot change the frame it is measured against |
| mask boolean | an integer mask, even one holding only 0 and 1, would be read as POSITIONS |
| mask length equals `self.n` | source-length and population-length masks are interchangeable whenever a population is complete — every fixture, almost no production case |
| indices integral, rejected before casting | `np.array([1.7], dtype=np.int64)` silently yields `[1]` |
| `take` rejects an already-projected array | double projection yields a shorter array that still looks plausible |
| owned copy, then read-only | setting the write flag on a VIEW leaves the writable base reachable by the caller |

### 2.3 Membership fingerprint

`sha256(source_id || n_source || indices)`, memoised on first access. It reaches
the defect cardinality cannot: two equal-sized but DIFFERENT subsets. `n = 980`
beside `n = 980` says nothing about whether the same 980 rows were used.

Measured: two disjoint 500-row subsets of one frame have equal `n` and can carry
an identical scope, and their fingerprints differ. Renaming a population leaves
the fingerprint unchanged, because it measures rows and not labels.

---

## 3. `population_source_id`

Absolute indices are meaningful only relative to a NAMED source. Two populations
over different frames can carry identical indices, `n_source` and `scope` and
describe entirely different rows.

`CanonicalVariantTable.population_projection(partition)` returns a
`CanonicalPopulationProjection` whose identity derives from the cohort version,
the selected partition, and the ORDERED `variant_id` sequence.

### 3.1 Why not `partition + cohort_version`

Those identify a CATEGORY of population, not an exact frame. Two tables can share
both while differing in row membership, order, filtered variant set, or corrected
data under one human-readable version. Different frames would then produce
identical membership fingerprints whenever their absolute indices coincided — and
they coincide constantly, because `full()` always yields `arange(n)`.

Measured on a two-partition table: the `test` and `cal` projections both occupy
indices `[0, 1]` within their own frames, and receive different identities.

### 3.2 Discriminations verified

| case | outcome |
|---|---|
| same variants, order, partition, version | same identity |
| different variants, same count and positions | different |
| same variants, different ORDER | different |
| different `cohort_version` | different |
| `["ab","c"]` versus `["a","bc"]` | different — length-prefixed |
| partition literally named `__all__` versus the all-rows projection | different — distinct namespaces |
| same variants, different model scores | SAME |

The last is the one easy to get wrong in the safe-looking direction: binding
predictions into the identity would break paired model comparison.

### 3.3 Verified against real structure, not assumed

`_select` preserves the original pandas index while `arrays()` calls `.to_numpy()`,
so the projected arrays are contiguous. That is what makes addressing the
PARTITION PROJECTION correct rather than merely preferable: `n_source` is the
attempted metric population before any label restriction, `take()` consumes
exactly what `arrays()` produces, and no context can address a row in another
partition. `source_indices` records the mapping back to the table for provenance
without being needed for projection.

`population_projection` deliberately does NOT require a score column: identity is
about which variants are evaluated, not what a model predicted, so a population
can be named before it is scored. The digest is memoised per partition rather
than computed at construction, because hashing the ordered `variant_id` sequence
is O(n) against a roughly 1.5-million-variant cohort and most callers never build
a population.

---

## 4. `MetricContext` wiring

`population` is REQUIRED, and the standalone `population_scope` field is REMOVED
and derived from `population.scope`. Two sources of truth for one fact eventually
disagree.

Twelve construction sites were measured, all in tests: no production code builds
a context yet, because the registry is not wired into the evaluator until commit
3. Twelve mechanical edits is not a reason to accept an optional field that would
quietly become permanent.

Arrays in a context are ALREADY PROJECTED and are validated against
`population.n`, not `n_source`. `support()` now reports
`population_fingerprint` beside `population_scope`.

The two cases stay distinct because they answer different questions:

    non-finite probabilities : population=attempted_cohort, n_observations=1000, FAILED
    withheld labels          : population=label_eligible,  n_observations=980,  ok

---

## 5. What was deliberately NOT done

`cohort_version` validation was NOT tightened. Ruled out of this commit because
it would combine exact population identity, provenance-policy strength and
certification admissibility in one change, and force twenty fixture edits
unrelated to the population abstraction.

**Audit, 2026-07-27:**

    cohort_version call sites audited:
        generic "v2": 20
        "v2-xyz": 1
        "v2-abc": 1
        "v1": 1

    Current mitigation:
        population_source_id also hashes the ordered variant_id sequence and
        partition, so distinct row sets remain distinguishable.

    Residual ambiguity:
        identical variants in identical order and partition, evaluated under
        different label/adjudication policies but the same generic
        cohort_version, produce the same population_source_id.

That is a dataset-policy provenance defect, not a row-membership defect, and
belongs with the broader identity model — `dataset_identity`,
`cohort_policy_version`, `partition_identity` — in its own commit.

`metrics.evaluate` remains UNCHANGED and non-certifiable. Its label mask survives
in `clean_arrays` for that path alone.

---

## 6. Verification

### 6.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list before and after: 40 failures, all sandbox dependency gaps (`pyarrow`,
`xgboost`) that are green in continuous integration. Baseline taken in a separate
pristine clone, never by reverting the working tree.

### 6.2 Sabotage matrix

Fourteen breaks applied, **fourteen detected, zero undetected**, green after
restore.

| break | detected |
|---|---|
| B1 strict-narrowing guard removed | yes |
| B2 subset proof removed | yes |
| B3 fingerprint ignores source identity | yes |
| B4 fingerprint ignores membership | yes |
| B5 indices stored as a view, not an owned copy | yes |
| B6 non-integral indices silently truncated | yes |
| B7 integer restriction mask accepted | yes |
| B8 a root may be partial | yes |
| B9 source identity drops the ordered variant ids | yes |
| B10 length-prefixing dropped | yes |
| B11 all-rows projection reuses the named-partition namespace | yes |
| B12 context stops validating arrays against the population | yes |
| B13 `support()` stops reporting the fingerprint | yes |
| B14 the retired label selector is reinstated | yes |

### 6.3 The first run left two undetected

**B12 was a real gap.** The array-length guard had been verified interactively
but never written as a test, so disabling it broke nothing. Closed by
`test_a_context_refuses_arrays_that_do_not_match_its_population`.

**B11 was a malformed break.** Replacing the conditional namespace still encoded
`None` as `""` and `"__all__"` as `"__all__"`, so no collision was created. Rebuilt
to encode `None` as the literal `"__all__"` under the same namespace, and then
detected.

### 6.4 THE MEASURED COLLECTION DELTA CAUGHT AN ACCIDENTAL DELETION

The most important finding of this session, and it was found by arithmetic, not
by review.

`test_prediction_input_contract.py` collected 25 tests before the commit and 19
after — yet the edit accounted for only three removals and five additions, which
should have given 27.

The tripwire retirement replaced everything from its anchor to the END OF THE
FILE. Three test functions appended AFTER that anchor during commit 2a were
destroyed with it, costing eight test cases:

    test_every_metric_refuses_at_the_gate_not_by_raising          (6, parametrised)
    test_a_probability_metric_is_unaffected_by_nonfinite_scores_at_the_gate
    test_the_probability_range_guard_runs_before_the_finiteness_assertion

The first is the test that closed the B4 gap in COMMIT 2a's OWN sabotage matrix —
the one asserting that a refusal comes from the gate rather than from a kernel
raising, a distinction status alone cannot make. Losing it would have silently
reopened a gap that a sabotage matrix had already found once.

All three were restored verbatim, and the module now collects 27.

**A computed ratchet would have recorded the number the edit intended and lost
these silently.** This is why ratchet moves are measured.

---

## 7. Files

    src/genomic_variant_classifier/evaluation/population.py    NEW
    src/genomic_variant_classifier/evaluation/canonical.py     projection + identity
    src/genomic_variant_classifier/evaluation/registry.py      context carries the population
    src/genomic_variant_classifier/evaluation/capabilities.py  POPULATION_FINGERPRINT
    src/genomic_variant_classifier/evaluation/metrics.py       selector retired
    tests/unit/test_evaluation_population.py                   NEW, 41 tests
    tests/unit/test_prediction_input_contract.py               25 -> 27
    tests/unit/test_metric_registry.py                         42 -> 44
    tests/unit/test_metric_metadata_vocabulary.py              population wiring

Ratchet 3273 -> 3318 (+45), measured by `pytest --collect-only`.

---

*Written 2026-07-27.*
