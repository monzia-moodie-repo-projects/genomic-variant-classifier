# SESSION 2026-07-27 — the fail-closed prediction-input contract (commit 2a)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `974d426`, ratchet 3247
**Roadmap position:** Tier 1 item 6, commit 2a — prerequisite to registry integration
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The ruling this commit implements

Ruled 2026-07-27:

> No numerical kernel may select, filter, normalise or redefine its evaluation
> population. Population construction is an explicit upstream operation, and
> every result must describe exactly that population.

## 2. A withdrawn commit, and a correction to the record

An earlier package this session, commit 3a, combined the calibration binning
repair with a population repair built the OPPOSITE way: it left the kernels
filtering and merely recorded what they had dropped, as `n_observations_used`
and `n_excluded_non_finite`. One of its tests asserted
`certification_eligible is True` on a cohort containing non-finite model output.

That package was withdrawn before installation. Nothing from it reached the
repository. Its binning work is carried to commit 2b, where the ruled sequence
places it.

A second correction belongs on the record. It was claimed that
`metrics.evaluate` "already does what the ruling wants" because instrumentation
showed zero non-finite values reaching any kernel from it. The instrumentation
was accurate; the conclusion was not. `evaluate` reaches that state by filtering
the predictions itself and reporting the narrowing as `n_input`, `n`,
`n_dropped` and `dropped_fraction`. That is population-accounting TRANSPARENCY,
not fail-closed behaviour: it still computes over the survivors. Visible
narrowing is not validity, and `evaluate` must never be cited as evidence that
strict kernels tolerate filtering. It is marked non-certifiable in its own
docstring and pinned by a test.

## 3. What was wrong

Every kernel in `metrics.py` routed through `clean_arrays`, which dropped
non-finite rows on ONE JOINT MASK covering labels, scores and probabilities
alike. A metric therefore returned a value over a silently narrowed population
while `MetricContext.support()` named the wider one. Measured:

    rows supplied to the context           : 1000
    rows the kernel actually computed over :  980
    result.n_observations reported         : 1000
    result.status                          : ok
    result.certification_eligible          : True

## 4. The distinction that makes the repair coherent

A non-finite predicted probability is a MODEL-OUTPUT FAILURE. A withheld
reference label is an ORDINARY MISSING OBSERVATION, first-class in this project
by design and carried as NaN by `CanonicalVariantTable`. The two must not share
a mask.

    labels        selected upstream by a NAMED transitional selector,
                  pending EvaluationPopulation
    predictions   never selected; validated, and failed closed
    kernels       assert their prediction inputs and raise
    registry      owns the refusal and its diagnostics

## 5. What was built

**`registry.py`** — a finiteness gate composed into all three applicability
predicates, ahead of `is_probability`, which documents that it IGNORES
non-finite values and would otherwise wave them through. A refusal produces
`status=FAILED`, `value=NaN`, support naming the ATTEMPTED population, and
diagnostics.

The reviewing document specified `validate_probability_context(...) -> MetricResult | None`.
That is the exact shape `registry.py` rejected on the day it was written: such a
validator may return ANY result, including an OK one carrying a value, so
"refused" and "computed" stop being distinguishable at the type level. The
equivalent that fits is an applicability predicate returning FAILED, which
`MetricStatus.FAILED` already covers in its own definition — "a prerequisite
validated and found contradictory before the computation could begin". The
ruling is honoured; the rejected type is not reintroduced.

**`metrics.py`** — `_require_finite_scores` and `_require_finite_probabilities`,
metric-specific rather than universal, so a probability-only metric is never
failed by an irrelevant score array. Six kernels assert their prediction input:
`auroc`, `auprc`, `auprc_gain`, `brier_score`, `log_loss`,
`expected_calibration_error`.

`select_finite_reference_labels` is the named transitional label selector.
`clean_arrays` now delegates the label decision to it rather than owning it, so
the residual population debt is one precise deletion target for the commit that
introduces `EvaluationPopulation`.

**FINITENESS RAISES; RANGE DOES NOT.** These are different categories and the
distinction is load-bearing. A vector outside [0, 1] was never a probability
vector — `is_probability` returns False and the calibration kernels return NaN,
pinned by `test_calibration_metrics_are_nan_on_non_probability_scores` — and THE
SAME ARRAY remains a perfectly valid score for a ranking metric on the same rows,
which that test also asserts. Raising on range would conflate "not a probability"
with "no prediction" and break a landed, correct contract. The ORDER is itself
the contract and is pinned by a test, because moving the assertion ahead of the
range guard would silently convert a documented NaN into an exception.

**`capabilities.py`** — four vocabulary members and four validated accessors:
`n_nonfinite_probabilities`, `n_finite_probabilities`, and score equivalents.
Probabilities and scores are named separately because the ranking metrics consume
scores, which need not lie in the unit interval; the finiteness contract is
identical and the arrays are not.

**`canonical.py`** — the seam's contract amended in the present tense. It
previously stated that `clean_arrays` "drops non-finite rows on ONE joint mask",
which the split makes false in executable documentation. No chronology was added
to the source; that belongs here and in the changelog.

**`metrics.evaluate`** — UNCHANGED in behaviour, marked in its docstring as a
legacy survivor-filtering interface and not a certifiable path.

## 6. Verification

### 6.1 Regression

The 38 test modules touching the evaluation stack were run against a SEPARATE
pristine clone and against the working tree. Both produced 40 failures; the two
`FAILED` lists are BYTE-IDENTICAL. All 40 are sandbox dependency gaps
(`pyarrow`, `xgboost`) that are green in continuous integration. The baseline was
taken in a separate clone, never by reverting the working tree.

Exactly one landed test failed on the first strict run, and it was the intended
one — see §7.

### 6.2 The sabotage matrix

Twelve deliberate breaks, each applied to a working copy and restored from a copy
taken beforehand.

| break | detected | tests fired |
|---|---|---|
| B1 auroc stops validating its scores | yes | 2 |
| B2 brier_score stops validating its probabilities | yes | 1 |
| B3 the finiteness gate never fires | yes | 9 |
| B4 the gate is removed from the calibration predicate | yes | 1 |
| B5 the gate is removed from the ranking predicate | yes | 4 |
| B6 support reports survivors instead of the attempted population | yes | 7 |
| B7 the gate becomes universal instead of metric-specific | yes | 2 |
| B8 the finiteness assertion moves ahead of the range guard | yes | 1 |
| B9 clean_arrays reabsorbs the label decision | yes | 1 |
| B10 the seam drops its transitional declaration | yes | 1 |
| B11 a kernel applies the label selector to a prediction input | yes | 1 |
| B12 the accessor stops validating its type | yes | 4 |

Twelve applied, twelve detected, zero undetected, green after restore.

### 6.3 THE FIRST RUN LEFT FOUR BREAKS UNDETECTED

Recorded because the outcome is the evidence, not the intention.

**B4 was a real gap in the tests.** Removing the gate from one applicability
predicate still produced a FAILED result, because the strict kernel raised and
`compute` caught it. Status alone cannot distinguish "refused before dispatch"
from "blew up during dispatch", and only the first carries diagnostics or leaves
the population untouched. Closed by
`test_every_metric_refuses_at_the_gate_not_by_raising`, parametrised over every
registered metric and asserting the REASON, not merely the status.

**B10 was a real weakness in a tripwire.** It asserted the word "transitional"
appeared somewhere in `canonical.py`; the break removed the contract sentence
while the word survived in an unrelated method docstring. The tripwire now binds
the MODULE docstring, where the contract is stated, and asserts the prediction
contract positively rather than merely checking the false claim is absent.

**B7 and B8 were malformed breaks, not undetected defects.** B7 validated the
same array twice under a different name; universal validation can only occur at
registry level. B8 added a range check to a function reachable only after
`is_probability` has already returned — dead code, not a behaviour change. Both
were rebuilt as real behaviour changes and both are now detected.

## 7. A landed test that codified the defect

`tests/unit/test_evaluation_metrics.py` contained:

```python
def test_auroc_ignores_nonfinite():
    assert auroc([0, 1, 1], [0.1, 0.9, np.nan]) == pytest.approx(auroc([0, 1], [0.1, 0.9]))
```

Its name approved of the behaviour the ruling forbids. This is the shape recorded
in roadmap 6.28, where a test's own comment named a fabrication approvingly. It
was INVERTED, not renamed and not deleted: the old expectation is now the
sabotage, and `test_auroc_rejects_nonfinite_scores` requires the refusal. A
companion asserts the kernel still computes on wholly finite input, so the
refusal cannot degenerate into a blanket break.

## 8. What this commit deliberately does NOT do

- `EvaluationPopulation` is not introduced. Label selection remains on the
  transitional selector, tripwired so it cannot quietly become permanent.
- `metrics.evaluate` is not modified. Whether it is frozen permanently as
  historical compatibility or gains a strict mode is a deliberate decision for
  its own commit, not an incidental change.
- The calibration binning convention is untouched; it is commit 2b.

## 9. Files

    src/genomic_variant_classifier/evaluation/metrics.py       validators, named selector, 6 strict kernels
    src/genomic_variant_classifier/evaluation/registry.py      the fail-closed gate
    src/genomic_variant_classifier/evaluation/capabilities.py  4 members, 4 accessors
    src/genomic_variant_classifier/evaluation/canonical.py     seam contract amended
    tests/unit/test_prediction_input_contract.py               NEW, 25 tests
    tests/unit/test_evaluation_metrics.py                      inverted test, 30 -> 31

Ratchet 3247 -> 3273 (+26), measured by `pytest --collect-only`.

---

## 10. The installation itself, 2026-07-27

### 10.1 First installer block in this project, and its cause

The installer refused to launch: "cannot be loaded ... is not digitally signed."
Diagnosed rather than assumed. The execution policy is RemoteSigned at both
CurrentUser and LocalMachine scope, and the file carried a Zone.Identifier
alternate data stream of 73 bytes -- the Mark of the Web, applied because the
file was downloaded. Under RemoteSigned a locally authored script runs unsigned
while a downloaded one must be signed.

Unblock-File removed only that stream. The SHA-256 before and after was
identical, E104DA94ACA697047AC62337B260D8FCC26F7BD948E4C8F569C296621B096491,
confirming by measurement that the alternate data stream is not file content and
that the payload was unaltered. No execution policy was changed and no system or
security setting was modified.

### 10.2 The install

Baseline collection measured 3247, matching the ratchet. Nine payload files
written. Post-install collection measured 3273, an exact delta of +26. Ratchet
and badge advanced to 3273. The eight affected test modules returned 208 passed
in 18.65 seconds, the identical count observed on Linux during development, so
the change behaves the same on both platforms. Working tree: nine modified, two
untracked, eleven total, matching the shape the installer computed from the
payload plus the ratchet and badge.

### 10.3 An unplanned verification of the idempotency gate

The installer was then run a second time. It verified all nine payload hashes,
passed the HEAD gate -- correct, since the commit had not yet been made -- and
refused at the clean-tree gate with "WORKING TREE IS NOT CLEAN. Nothing was
written."

Nothing was written. That throw precedes the try block, so no backup was taken
and no rollback was required, and the output contains no write lines for that
run. The failure it prevented is specific and would have been costly: a second
pass would have measured a baseline of 3273, installed the same payload again,
and then demanded a further +26 to reach 3299, corrupting the ratchet with a
duplicate install that changed no test.

The gate was therefore observed to fire against a real condition rather than a
constructed one. It is recorded because the standing lesson asks for exactly
that evidence, and here it arrived unplanned.

---

*Written 2026-07-27.*
