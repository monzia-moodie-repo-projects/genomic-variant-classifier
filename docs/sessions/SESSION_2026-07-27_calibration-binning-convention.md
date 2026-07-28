# SESSION 2026-07-27 — the calibration binning convention (commit 2b-1)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `951fd82`, ratchet 3318
**Roadmap position:** Tier 1 item 6, commit 2b-1
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. Why 2b was split

Commit 2b as ruled contained two things: the expected-calibration-error semantics,
and registering the remaining point estimates. They were separated for the reason
recorded in `registry.py`'s own module docstring and already applied to 2a and
2a-1:

> Rewriting it here would mix an architectural addition with a behaviour change.
> A regression would be impossible to localise.

The binning repair CHANGES NUMBERS in the kernel. Registering descriptors for the
Matthews correlation coefficient, F1, the maximum calibration error and prevalence
is ADDITIVE and must change none. Landing the binning first means any figure that
moves during 2b-2 is a signal rather than noise.

---

## 2. The defect

`metrics.expected_calibration_error` opened with "Equal-width binning, TOP BIN
CLOSED" — `[lo, hi)` with only the final bin closed at 1.0 — and implemented
`np.digitize(..., right=True)`, which makes EVERY bin `(lo, hi]`. Every
probability sitting exactly on an interior decade edge landed one bin LOWER than
documented.

`ClinicalEvaluator._calibration_error` had implemented the documented convention
since the 2026-07-10 top-bin repair, so the two disagreed about every interior
edge for seventeen days.

### 2.1 The measured separation

On a cohort where an edge-exact value shares a bin with non-edge values of the
OPPOSITE calibration sign:

    superseded (lo, hi] convention : 0.3242857142857143
    documented [lo, hi) convention : 0.06428571428571427
    relative difference            : 404.44%

### 2.2 Why seventeen days of tests never saw it

The expected calibration error is

    (1/N) * sum_b | sum_{i in b} (y_i - p_i) |

and is therefore INVARIANT to regrouping whenever every merged group shares the
sign of (accuracy − confidence): combining same-sign groups cannot change the
total. Ordinary fixtures land in that regime by default. This is a theorem, not
folklore, and it is now pinned by a test.

`tests/unit/test_calibration_implementations_agree.py` placed all of its mass at
0.0, at 1.0, or strictly inside a bin. It contained NO interior-edge value, so it
separated the TOP-bin definitions and was structurally incapable of separating
`[lo, hi)` from `(lo, hi]`.

### 2.3 Reachability

| probability vector | rows on an interior edge |
|---|---|
| continuous scores | 0.00% |
| mean of the thirteen base models | 0.00% |
| mean of ten folds | 55.02% |
| rounded to two decimals | 5.82% |
| rounded to one decimal | 60.40% |

---

## 3. What was built

### 3.1 `equal_width_bin_indices`

A named function rather than an inline expression, because the convention is a
scientific decision, it has been got wrong once, and a named function is
somewhere a validation and a test can attach.

`searchsorted(edges, v, side="right") - 1` places an edge-exact value in the bin
it OPENS; the clip closes the top at 1.0. It FAILS CLOSED on non-finite and
out-of-range input rather than clipping, because an unguarded clip would place
such values in the first or last bin and the calibration figure would silently
describe a different population.

Verified against the ruled worked example — `0.0→0, 0.1→1, 0.2→2, 0.9→9, 1.0→9`
— and the ruled boundary vector, which returns `[0, 1, 0, 1, 9, 9]`.

### 3.2 `CalibrationBins`

ONE table. The expected and maximum calibration errors are two summaries of it,
not two functions that each bin again. Only OCCUPIED bins are retained: an empty
bin has no accuracy and no confidence, and inventing zero for either would drag
the maximum toward that bin's midpoint and pull the weighted mean toward nothing.

`definition()` carries `binning`, `interval_convention`, `n_bins` and
`metric_definition_version` with the numbers, because a calibration figure
without its binning convention is not reproducible — the same predictions gave
0.3242857 and 0.0642857 under the two conventions.

### 3.3 One binning was not enough; one SUMMATION was needed

After the kernel and the evaluator were both binning through the shared table,
they still differed by `3.5e-18` on one fixture. The cause was real: the kernel
retained its own summation loop while the evaluator read `CalibrationBins.expected`.
Binning once but summing twice still leaves two implementations that can drift.

The kernel now reads the table. Measured across all fixtures afterwards: worst
absolute difference **0.000e+00**, bit-identical.

`ClinicalEvaluator._calibration_error` no longer contains a binning loop at all;
it delegates. `maximum_calibration_error` was added to the kernel and reads the
same table.

### 3.4 No published figure moves

Every published calibration number came from the evaluator, which implemented the
documented convention already. What changes is that there is now one binning and
one summation rather than two that happened to agree on the fixtures anyone had
thought to write.

---

## 4. Carried item (k) discharged

`test_calibration_implementations_agree.py` gained an interior-edge fixture, a
proof that the fixture separates the two conventions, and assertions that both
the kernel and the evaluator match the documented one. A test that cannot fail on
the axis its name implies is not evidence, and that module was in exactly that
condition for seventeen days. Its header now records why.

---

## 5. Verification

### 5.1 Regression

The 38 modules touching the evaluation stack produce a BYTE-IDENTICAL `FAILED`
list before and after: 40 failures, all sandbox dependency gaps (`pyarrow`,
`xgboost`), green in continuous integration. Baseline taken in a separate pristine
clone.

No test was lost: the agreement module's function names were diffed against the
baseline, and three were added with none removed. That check exists because the
previous commit destroyed eight test cases through an edit anchored to end of
file, caught only by the measured collection delta.

### 5.2 Sabotage matrix

Ten breaks applied, **ten detected, zero undetected**, clean on the first pass.

| break | detected | tests fired |
|---|---|---|
| B1 restore the superseded left-open binning | yes | 17 |
| B2 top-bin clip dropped, reopening the 2026-07-10 defect | yes | 5 |
| B3 helper accepts non-finite input | yes | 2 |
| B4 helper accepts out-of-range input | yes | 2 |
| B5 empty bins counted as perfect | yes | 17 |
| B6 the maximum read from a different table | yes | 1 |
| B7 the evaluator stops delegating and bins again | yes | 11 |
| B8 the definition reports the wrong convention | yes | 1 |
| B9 the bin count is no longer validated | yes | 1 |
| B10 the expected error sums independently again | yes | 15 |

This is the first matrix in this stack to come back clean on its first run. The
detection counts are substantial rather than marginal, which is the reason to
believe it rather than merely to hope.

---

## 6. Files

    src/genomic_variant_classifier/evaluation/metrics.py      helper, table, MCE kernel
    src/genomic_variant_classifier/evaluation/evaluator.py    delegates; duplicate binning removed
    tests/unit/test_calibration_binning_convention.py         NEW, 33 tests
    tests/unit/test_calibration_implementations_agree.py      5 -> 8, interior edges

Ratchet 3318 -> 3354 (+36), measured by `pytest --collect-only`.

---

## 7. Next

Commit 2b-2 registers the remaining point estimates — the Matthews correlation
coefficient, F1, the maximum calibration error and prevalence — with a
`ResultKind` category vocabulary and declared threshold provenance, replacing the
0.5 currently buried at `evaluator.py:481-482`. It must move no number.

---

*Written 2026-07-27.*
