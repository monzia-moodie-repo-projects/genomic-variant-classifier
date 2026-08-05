# SESSION 2026-08-05 — OP-1 step 3a: Oracle C1

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = f0db01f`. Full suite 4265 passed, 6 skipped, 0 failed; 4271 collected.**
**Ratchet 4225 → 4271 (+46). A VERIFICATION COMMIT — no production code changed.**

Companion documents:
`SESSION_2026-08-05_op1-step2-typed-outcome.md`,
`SESSION_2026-08-04_op1-step1-exact-sweep.md`.

---

## 1. Oracle C1 holds

For the **six estimands both paths compute** — sensitivity, specificity, positive
and negative predictive value, F1 and the Matthews correlation coefficient —
`metrics_from_counts` and `registry.compute` agree on **status, reason and
value** across six applicability regimes at the registry's canonical threshold.

**That validates a decision made deliberately in step 2.** Every reason string in
the count path is one the registry already emits, and the statuses follow REG-2's
measured boundary. Had step 2 coined a private vocabulary, C1 would now report
six failures per fixture and **none of them would be defects** — the oracle would
be comparing two dialects.

The vocabulary was chosen so it could be tested. It has been.

---

## 2. Why one threshold serves the whole comparison

Measured against the live registry: of **24 descriptors**, **nine** carry a
`ThresholdParameters` and all nine carry `(0.5, GREATER_OR_EQUAL,
fixed_default)`. The other fifteen are ranking or calibration metrics with no
threshold at all.

So one fixture family suffices — and **every fixture must contain a score of
exactly 0.5**, because the sweep's candidates are the observed score values.
Without one there is no candidate at the registry's parameters and the comparison
is impossible rather than merely empty.

A test re-derives that declaration over the live graph, so a descriptor pinned
elsewhere cannot arrive unnoticed.

---

## 3. The first run failed, and the defect was mine

```
7 failed, 44 passed

AssertionError: the Oracle C fixture must expose exactly one (0.5, >=)
candidate, found 0.
```

**Two requirements act on every C1 fixture and I satisfied only one.**

- The comparison needs a candidate at the registry's parameters → the cohort
  **must contain exactly 0.5**.
- The `no_predicted_positives` regime needs **nothing flagged** at that threshold.

Under `GREATER_OR_EQUAL` those are **contradictory**: a score of exactly 0.5
satisfies `p >= 0.5` and is therefore flagged.

I wrote every score below 0.5 to satisfy the second and removed the score the
first requires. The fixture was internally coherent and externally impossible.

### 3.1 `>=` places the boundary score on the flagged side

That single fact makes one regime **unexpressible** and its mirror image trivial:

```
no_predicted_positives   scores must all be < 0.5   AND contain 0.5   -> impossible
no_predicted_negatives   scores must all be >= 0.5  AND contain 0.5   -> trivial
```

Verified directly — `[0.9, 0.5, 0.7, 0.6]` clears nothing and flags four.

**And I had read the docstring that says so.** `ThresholdOperator` records that
*"`>=` and `>` differ exactly at `prob == threshold`"*, quoted repeatedly across
this sequence, and I still wrote a fixture violating it. **The guard caught what
the reading did not.**

### 3.2 The regime is dropped, not replaced

A cohort could be built where 0.5 is present and the regime differs — every label
negative, so `TP + FP` is nonzero but the positive predictive value is 0 rather
than undefined. That would **quietly substitute a different test and call it the
same one.**

Six regimes are claimed and six are tested.

**The refusal itself is not lost.** Step 2 already pins it at the count level with
`UNDEFINED` and reason `empty_predicted_positive_set`. What C1 cannot do is
*corroborate* it against the registry — a limit of the oracle, not a gap in the
evidence.

### 3.3 The impossibility is an assertion, not a comment

`test_the_empty_predicted_positive_regime_is_unreachable_at_the_canonical_threshold`
proves it directly, and is **conditional on the operator**: if the registry ever
adopts `GREATER`, or a descriptor moves off 0.5, the regime becomes reachable and
the test fails — which is the signal to add it back **deliberately**.

---

## 4. The surface partition, codified in four disjoint sets

| surface | members | why |
|---|---|---|
| shared, compared by C1 | 6 estimands | both paths compute them |
| operating-point only | `flagged_fraction` | no registry counterpart |
| registry-only, threshold | balanced accuracy, the two likelihood ratios | derivable, deliberately not duplicated |
| registry-only, population | `prevalence` | Decision 3, 2026-08-04 |

**The three derivable metrics stay out because the formulas are not the difficult
part.** `_requires_interior_specificity` already decides that a positive
likelihood ratio with specificity at 1.0 is `UNDEFINED` with reason
`likelihood_ratio_unbounded` — a scientific policy about an unbounded quantity.
Reimplementing the formula would create a **second authority** for that policy,
for the zero-denominator applicability, the status choice, the exact reason string
and the metadata. That is the SWEEP-1 shape.

A completeness test closes the partition: a **new** threshold-carrying descriptor
must be placed on a surface deliberately — compared by C1, or excluded with a
reason — or it fails.

### 4.1 Prevalence's exclusion was corroborated independently

Decision 3 reasoned from first principles that prevalence is a population
statistic. **The live registry declares it with no threshold at all.**

A decision made by argument, later confirmed by measurement of an artifact that
knew nothing about the argument.

---

## 5. C2 is deliberately not in this commit

`metrics_from_counts` builds plain `MetricResult` instances; `registry.compute`
**enriches and finalises** them with descriptor identity, support counts,
certification metadata and population keys. Full identity is unlikely to hold
until both paths share one finaliser.

**Making C2 pass by copying registry metadata into the count path would recreate
a second implementation of the finalisation contract.** Step 3b measures the
exact difference set, and that measurement is a **finding**, not a fix.

---

## 6. One illustration corrected against the live object

The adopted design's helper reads `sweep.n_candidates`. `ExactThresholdSweep`
implements `__len__` and has no such attribute.

**Fourth illustration-versus-reality gap in this sequence**, after
`_two_class_context()`, `EvaluationPopulation.full(n_source=...)` by keyword, and
an `EvaluationPopulation` import that was documented and absent. An adopted
design's code sketches specify **intent, not API**, and each needs checking
against the live object before use.

---

## 7. A defect of mine, labelled accurately rather than conveniently

Three post-check counts were wrong, and my first instinct was to call it the
eleventh prose-matching instance. **Measuring showed every occurrence is code** —
I had undercounted uses in a file I wrote myself.

The distinction matters: the remedy for prose-matching is tokenising, and the
remedy for this is measuring. Calling it the familiar defect would have applied
the wrong fix **and inflated a pattern's evidence with a case that does not
belong to it.**

---

## 8. No sabotage line

Nothing existing changed, and **the oracle's own first run was the
falsification**: it failed, the failure was read rather than silenced, and the
correction removed an impossible cohort rather than a disagreement.

---

## 9. Acceptance

| item | value |
|---|---|
| full suite | 4265 passed, 6 skipped, 0 failed (16m31s, and 19m10s under the armed gate) |
| collected | 4271, measured by the installer |
| ratchet | 4225 → 4271 (+46) |
| tests | 46 cases from 11 functions — 6 × 6 parametrised, plus 10 |
| `test_readme_claims` | 10 passed |
| **modified source files** | **none** — checked by the installer, not asserted |

The ratchet installer gained a precondition specific to a verification commit: it
runs `git status --short` and **refuses if any file under `src/` is modified**,
because the entry makes that claim prominently.

---

## 10. Next

**Step 3b — the C2 measurement.** Which metadata keys `registry.compute` adds
that `metrics_from_counts` does not, and whether the count path could ever supply
them without duplicating the finalisation contract. The measurement decides
whether a finaliser extraction is needed; it is the deliverable, and any fix
follows from it.

Then step 4 (the selector, Objective A, closing **D12** — the last of the twelve),
step 5 (the shadow comparison), and step 6 (the cutover, which must reckon with
**GUARD-1**).

Nineteen follow-ups carried, unchanged by this step.
