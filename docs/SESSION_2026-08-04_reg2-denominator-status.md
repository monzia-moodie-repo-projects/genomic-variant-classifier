# SESSION 2026-08-04 — REG-2: a vanishing denominator is UNDEFINED, not INSUFFICIENT_SUPPORT

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = d4b4259`. Full suite 4159 passed, 6 skipped, 0 failed; 4165 collected.**
**Ratchet 4161 → 4165 (+4). One line of source; four tests that are the first thing in the repository able to notice it.**

Companion documents:
`SESSION_2026-08-03_reg1-metadata-ownership.md`,
`SESSION_2026-08-04_op0-legacy-selector-semantics.md`.

Committed evidence:
`docs/measurements/REG2_DENOMINATOR_STATUS_AFTER_2026-08-04.txt`,
`docs/measurements/REG2_PREFLIGHT_AFTER_2026-08-04.txt`.

---

## 1. How this was found, and why it is not an OP-1 decision

While drafting the OP-1 build specification I wrote
`ppv_applicability_from_counts` returning `INSUFFICIENT_SUPPORT` for an empty
predicted-positive set — copied from an illustration. Reading the `MetricStatus`
enum showed that member is reserved for *"the machinery is ready and the science
is not"*, while `TP/(TP+FP)` with `TP+FP = 0` is **mathematically undefined**.

I proposed correcting it inside OP-1. **That was procedurally wrong**, and the
ruling of 2026-08-04 said why:

> Adopting `UNDEFINED` inside OP-1 while the registry says
> `INSUFFICIENT_SUPPORT` would be an intentional semantic movement disguised as
> an implementation discrepancy — the oracle would report a difference and nobody
> could tell which side was wrong.

OP-1's Oracle C exists to prove a new count path reproduces the registry's status
semantics. So: **measure first, then branch.** Measurement made the branch obvious.

---

## 2. The registry contradicted itself, and it was not a close call

Measured across **24 registered descriptors and 6 cohort shapes**:

| status | reason | metrics |
|---|---|---|
| `undefined` | `binary_class_support_required` | **7** |
| `undefined` | `likelihood_ratio_unbounded` | 2 |
| `undefined` | `zero_confusion_margin` | 1 |
| `insufficient_support` | `positive_class_support_required` | 1 — **correct, stays** |
| `insufficient_support` | `negative_class_support_required` | 1 — **correct, stays** |
| `insufficient_support` | `empty_predicted_positive_set` | 1 — **moved** |
| `insufficient_support` | `empty_predicted_negative_set` | 1 — **moved** |

**Ten metrics already used `UNDEFINED` for a mathematically undefined state.** Two
used `INSUFFICIENT_SUPPORT` for the same kind of state. REG-2 made two agree with
ten; twelve agree now.

`matthews_correlation_coefficient` was already on the correct side, which settled
the question: this was not a defensible convention I would be overturning on a
reading of an enum comment — it was **an internal inconsistency**.

And `registry.py:954-973` argues for the change in its own words: *"applicability
must refuse EXACTLY where the kernel would return NaN"*, listing the requirement
as `TP + FP > 0`, marked **threshold-dependent**. That is a denominator condition.

---

## 3. One line, because one factory serves both

`_requires_flagged_margin` produces the predicate for **both** predictive values
through its `flagged` parameter. So REG-2 is a single status change at one site,
and writing two edits would have half-applied.

Three factories, three behaviours, one changed:

| factory | metrics | REG-2 |
|---|---|---|
| `_requires_flagged_margin` | both predictive values | **changed** |
| `_requires_class_support` | sensitivity, specificity | untouched — correct |
| `_requires_nondegenerate_confusion` | Matthews, F1 | untouched — already right |

The reason strings are composed as `f"empty_{side}_set"` and are **unchanged**:
accurate before and after.

### 3.1 A search that could not see what it looked for

My first search for the reason strings in `registry.py` found **nothing**, and I
concluded they lived elsewhere. They do not — the reasons are **composed, not
literal**, so a grep for the output text could never match.

That is the eighth substring-search defect of this sequence and the same class as
`validate_probabilities` matching `/tie|ties/`: **searching for output text and
treating absence as evidence.** The predicates were resolved instead by asking
each descriptor where its own applicability function is defined, from the live
object.

---

## 4. The green suite was the finding, not the reassurance

After the change: **4,155 passed, 6 skipped, 0 failed.** A semantic correction to
two metrics changed no test outcome anywhere.

Then, measured directly by reverting it: **54 passed before, 54 passed after.**

**Nothing in the repository asserted either status.** Nothing would have noticed
either drifting back. That is the condition REG-2 exists to end, and it is why
the four tests are not optional — without them the commit corrects a value
nothing watches.

---

## 5. Four tests, three jobs

**T1, T2 — the moved statuses**, each asserting `UNDEFINED` **and** the unchanged
reason. A status without its reason is half a contract: `undefined` alone does
not say *which* degeneracy, and the registry deliberately keeps distinct reasons
per condition.

**T3 — the boundary.** `sensitivity` and `specificity` keep
`INSUFFICIENT_SUPPORT`. An absent reference class is a property of the **cohort**,
not of a quotient. Without this the rule could be over-applied later and nothing
would object.

**T4 — structural.** It walks the live descriptor graph and fails if any refusal
naming a denominator-zero state carries anything but `UNDEFINED`.

**T4 is the one that matters most.** T1 through T3 pin today's metrics; a metric
introduced next month with the old convention would pass all three. And in the
falsification T4 caught a case **neither T1 nor T2 constructs** — the negative
predictive value's zero denominator on a single-row cohort.

### 5.1 Dead code, caught and made load-bearing

`_COHORT_SUPPORT_REASONS` was declared and never referenced — in a file whose
sibling findings are DEAD-1 through DEAD-3. Rather than delete it, T4 now asserts
the two reason sets are **disjoint**, so the boundary is machine-checked and the
two tests cannot drift apart.

---

## 6. The falsification is the sabotage run

Reverting the change is the **only mutation that could exist** for a one-line
status correction, and its outcome was predicted before it ran:

```
T1 FAIL   T2 FAIL   T4 FAIL   T3 PASS      3 failed, 55 passed
```

**T3 passing under the reverted code is the load-bearing half.** A boundary test
that failed alongside the others would have been measuring REG-2 rather than the
line REG-2 must not cross.

---

## 7. One committed artifact contradicts itself, deliberately

`REG2_PREFLIGHT_AFTER` exits **1** and prints *"DECLARED MEMBERS NOT
REPRODUCED"*, offering two explanations — unreachable cohorts, or an earlier
wrong measurement. **Both are false.** A third case obtains: the defect was
fixed.

The instrument was written before the change and cannot distinguish its own
success from failure. Its message is honest about that limit, and **editing it to
agree with the outcome would destroy the evidence** — the same reasoning that
kept REG-1's baseline mutation report uncorrected.

---

## 8. Compatibility, recorded rather than waved past

**No enum value changed.** Both `insufficient_support` and `undefined` already
existed, and the enum's comment says those values are load-bearing because
historical run manifests contain them. Historical manifests stay readable.

What changes is that a **new** artifact reports `undefined` where a **pre-REG-2**
artifact reported `insufficient_support` **for the same cohort**. A reader
comparing the two sees a status change with no code defect behind it.

---

## 9. Acceptance

| item | value |
|---|---|
| full suite | 4159 passed, 6 skipped, 0 failed (21m08s, and 19m22s under the armed gate) |
| collected | 4165 |
| ratchet | 4161 → 4165 (+4) |
| `test_readme_claims` | 10 passed |
| movement set, after | **zero**, across 24 descriptors |

---

## 10. Follow-ups — seventeen

| id | item |
|---|---|
| **REG-2-b** | *new.* `_requires_interior_specificity` returns `INSUFFICIENT_SUPPORT` with reason `specificity_undefined` — the same mismatch, in a third place no cohort shape reached |
| **ICI-1** | *new.* `integrated_calibration_index` is declared applicable then returns non-finite, on three cohort shapes |
| **F1-1** | *new.* `f1` returns `ok` with 0.0 computed from an **undefined** positive predictive value — the D5 shape, in the registry rather than the operating-point functions |
| OPCOV-1 | the operating-point selectors have almost no coverage |
| GITIGNORE-1 | `*.bak_*` appears three times in `.gitignore` |
| STRUCT-1 | structural guards now used three times, on three defect classes |
| POP-1b-M03 | no test distinguishes the source distance from the parent distance |
| POP-1b-M07 | nothing asserts on `print_report` output |
| ZERO-1 | 24 dead-connector defaults still zero |
| INF-1 | an infinite reference label is pooled with NaN as *withheld* |
| ABS-1 | the ranking channel's refusal reported as `undefined_on_cohort` |
| DEAD-1 | ~40 lines of dead absence computation in `evaluate` |
| DEAD-3 | `_assert_absence_biconditional` computes `observed_curves` twice |
| PRE-2 | section 5's PASS line swallows the KAN banner |
| LINT-1 | no lint gate anywhere |
| F821-1 | 18 undefined names; 9 need assessment |
| CMP-1 | `ModelComparison` carries a fingerprint with no scope beside it |

---

## 11. Next

**OP-1 step 1** — `ConfusionCounts` and the exact sweep, with their own tests and
no selector. The build specification's `INSUFFICIENT_SUPPORT` is now corrected to
`UNDEFINED` as a **transcription fix**, matching a registry that is right, rather
than as a semantic movement smuggled into a rewrite.

Seven register defects remain open: D1, D2-D5, D6, D9, D10, D11, D12. One sort
with cumulative sums closes D1, D9, D10 and D11 together.
