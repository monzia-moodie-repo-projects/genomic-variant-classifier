# SESSION 2026-08-04 — OP-1 step 1: the exact threshold sweep

**Author: Monzia Moodie**
**Project: genomic-variant-classifier**
**Base: `HEAD = origin/main = b698c24`. Full suite 4197 passed, 6 skipped, 0 failed; 4203 collected.**
**Ratchet 4175 → 4203 (+28), measured by the installer. NO SELECTOR — nothing imports the sweep yet.**

Companion documents:
`SESSION_2026-08-04_thr1a-threshold-vocabulary.md`,
`SESSION_2026-08-04_thr1b-evaluation-sweep-source.md`.

---

## 1. Four register defects close by one design decision

| defect | |
|---|---|
| **D1** | the thousand-point grid misses achievable thresholds |
| **D9** | O(k·n): 2.25 × 10¹² element operations at 1.5 million variants |
| **D10** | threshold-invariant work recomputed inside the sweep |
| **D11** | two incompatible notions of which thresholds exist |

One sort with cumulative sums makes all four disappear at once, because **the
achievable thresholds *are* the unique score values**: there is no grid to miss
them, no per-candidate rescan, no loop to hoist invariant work from, and one
sweep serves both selectors.

```
n           grid O(1000n)      unique O(k·n)      sort + cumsum
1,000           1,000,000          1,000,000              9,965
100,000       100,000,000     10,000,000,000          1,660,964
1,500,000   1,500,000,000  2,250,000,000,000         30,774,796
```

Patching four separately would have left the architecture that produced them.

### 1.1 D1 demonstrated rather than asserted

On scores spaced at **0.0001** — finer than the grid's 1/999 step — the exact
sweep finds **five** distinct operating points and `np.linspace(0, 1, 1000)`
reaches **two**. Three of five are unreachable by construction, and a test pins
the comparison so the fixture cannot quietly stop demonstrating it.

---

## 2. The canonical domain, and the candidate the grid could not express

```
{(max p, GREATER)} ∪ {(s, GREATER_OR_EQUAL) : s ∈ unique(p)}
```

**k+1 candidates, not both operators at every score.** For adjacent distinct
values, `p > s` and `p ≥ s′` induce the **same partition**, so enumerating both
would duplicate candidates and make indices representation-dependent.

The `GREATER` entry is the **empty candidate**. `ThresholdParameters` constrains
a threshold to `[0, 1]`, so "flag nothing" cannot be a value above the maximum
when the maximum is 1.0; `GREATER` at the maximum expresses it instead. **The
grid sweep silently lacked this operating point altogether.**

That is why THR-1a and THR-1b came first: the operator is part of the
declaration, and `EVALUATION_SWEEP` labels every candidate as enumerated rather
than chosen.

---

## 3. Correctness is agreement with the definition

The sweep is an **optimisation**, and an optimisation is correct only if it
agrees with the thing it replaces. So the oracle is a brute-force `(p >= t)`
computation, not another clever construction.

Every candidate on eight cohorts — all-distinct, one tied pair, all tied, ties at
the top, a single row, all-positive, all-negative, and boundary scores of exactly
0.0 and 1.0 — was compared against it. **Zero mismatches.**

### 3.1 Not a third implementation

The same construction already exists **twice** in `metrics.py`, in `auprc` and
`_roc_points`, by different index arithmetic. Their raw count sequences were
compared across eleven cohorts on 2026-08-04 and found identical — **SWEEP-1**,
duplication rather than disagreement.

This module is written to be the one those two can **later** be rebuilt on, which
is why it lives in `thresholds.py` and imports no scikit-learn. Rebuilding them
here would make any numerical movement uninterpretable — the OP-0 lesson.

---

## 4. Storage: owned immutable arrays, lazy candidate views

`sweep[i]` builds a `ThresholdSweepCandidate` on demand and **nothing is stored
per candidate.** A 1.5-million-row cohort with distinct scores has 1.5 million
achievable thresholds, and materialising that many frozen dataclasses would cost
more than the sweep computing them.

Measured: bytes scale at **exactly 2.00× per doubling** — 12,525 → 25,025 →
50,025.

The arrays are **copied and marked read-only**, and a test mutates the caller's
input afterwards to prove the sweep is unchanged. **A sweep is evidence, and
evidence that can change after the fact is not evidence.**

Slicing is **refused** with a message pointing at the arrays, because it would
materialise one object per candidate — the cost the array backing exists to
avoid.

---

## 5. Counts only, and every refusal raises

A rate can be undefined while a count never is. `TP + FP = 0` makes the positive
predictive value undefined, and reporting it as `0.0` is the D2-D5 defect —
REG-2 established on 2026-08-04 that such a state is `UNDEFINED` rather than
`INSUFFICIENT_SUPPORT`, and step 2's typed outcome refuses accordingly.

`ConfusionCounts` **refuses a float stored as a count**, which is how that defect
begins.

Seven refusal paths raise rather than filtering and continuing: a non-finite
score or label, an empty cohort, scores outside `[0, 1]`, mismatched lengths, a
label that is not 0 or 1, and a population whose size disagrees with the arrays.

`p >= t` evaluates **false** for a NaN, and `evaluator.py` records that letting
one through moved a measured operating point from sensitivity 0.90 to 0.50 with
no exception and no warning.

---

## 6. A design correction the first dry run forced

The original append **documented an import of `EvaluationPopulation` that did not
exist.** It duck-typed `population` through `getattr`, and built a precondition
around the transitive scikit-learn risk of an import it never made.

The behaviour satisfied the ruling. **The type contract did not** — an
unannotated field in a type whose whole purpose is carrying identity is the
weaker half of the design.

Now imported and annotated at three sites, and the precautionary precondition
became **load-bearing**: it proved `population.py` imports only `__future__`,
`dataclasses`, `enum`, `hashlib`, `logging`, `numpy` and `typing`, so the
downward dependency cannot defeat THR-1a's structural guarantee.

The dry run found this. That is what dry runs are for, and it is the second time
this week that documentation asserted something the code did not do.

---

## 7. EXTRACT-1 paid for itself, one commit after being named

The pre-work inventory ran **before** the work — the discipline named after
THR-1a's inventory became a postmortem. It confirmed zero name collisions, no
structural assertion on `thresholds.py`, and 251 test modules in `tests/unit`, so
a separate test file is the convention and **a sweep is not vocabulary.**

And it found a real constraint:

### GUARD-1

`test_computation_path_guards.py:241-255` instruments threshold application on
the report path and asserts

```python
assert all(t == (0.5, ">=") for t in thresholds)
assert len(set(thresholds)) == 1
```

The exact sweep applies **every unique score**, and its empty candidate uses
**`GREATER`**. That guard would fire the moment the sweep is wired in.

**It does not fire now, because step 1 wires nothing.** But step 6's cutover must
either scope it to the legacy path or extend it to accept swept candidates
**deliberately**.

Found as a constraint on future work rather than as a red suite three commits
later. **That is a discipline returning something**, which is worth recording
separately from naming one.

---

## 8. No sabotage line

Nothing existing changed, and the correctness claim is **agreement with the
definition** across eight cohorts and every candidate — stronger than any
mutation devisable for a new pure function.

---

## 9. Acceptance

| item | value |
|---|---|
| full suite | 4197 passed, 6 skipped, 0 failed (16m07s, and 16m54s under the armed gate) |
| collected | 4203, measured by the installer |
| ratchet | 4175 → 4203 (+28) |
| `test_readme_claims` | 10 passed |
| tests | 19 functions, 28 cases |
| nothing else moved | nothing imports the sweep |

**19 functions, 28 cases**: the brute-force comparison is parametrised over eight
cohorts (+7) and the non-finite refusal over `nan`, `inf`, `-inf` (+2). That
arithmetic is exactly what went wrong three times today, so the ratchet
**measured its own target** and kept the prediction only as a cross-check.

---

## 10. What remains, and where

**Three register defects after step 1's four**: D2-D5, D6 and D12.

| step | closes |
|---|---|
| 2 — the typed `OperatingPointOutcome` | **D2-D5** (refuse, never fabricate), **D6** (store computed, round for display) |
| 3 — the two oracles | — |
| 4 — the selector, Objective A | **D12** (a declared, persisted tie-break) |
| 5 — the shadow comparison | — |
| 6 — the cutover | must reckon with **GUARD-1** |

Nineteen follow-ups carried, GUARD-1 newly among them.
