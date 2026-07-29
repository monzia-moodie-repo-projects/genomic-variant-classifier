# SESSION 2026-07-29 — a scientific assertion that had never executed

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `d8d04ab`, ratchet 3711
**Roadmap position:** skip-surface audit, after the register was fully discharged
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What was wrong

`test_aleatoric_higher_near_decision_boundary` asserts a real calibration
property: **aleatoric uncertainty must peak near p = 0.5**, where binary entropy
is maximal. It is a genuine scientific claim about the Monte Carlo Dropout
decomposition.

**It had never executed.** The test skipped unless the fitted model produced
predictions in BOTH the boundary band (0.4 to 0.6) AND an extreme band (below 0.1
or above 0.9), and at five epochs it never did. One `s` in a 3,711-test run,
presumably since the day it was written.

This is the same shape as the empty-parameter-set skip closed in `d8d04ab` -- a
guard reporting success while checking nothing -- but invisible rather than
obvious.

## 2. The measurement, and a hypothesis it refuted

Seven configurations, measured on the real machine with seed 42:

    configuration                range              boundary  extreme  verdict
    current fixture,  5 epochs   [0.283, 0.731]        252       0     SKIPS
    same data,       10 epochs   [0.226, 0.771]        193       0     SKIPS
    same data,       25 epochs   [0.066, 0.840]         81       3     spans
    same data,       50 epochs   [0.025, 0.919]         25      58     SPANS
    designed margin,  5 epochs   [0.342, 0.720]        262       0     SKIPS
    designed margin, 15 epochs   [0.214, 0.821]        151       0     SKIPS
    designed margin, 30 epochs   [0.085, 0.866]         86       5     spans

**IT IS UNDERTRAINING, NOT THE DATA.** At five epochs the model has learned
almost nothing -- every prediction sits between 0.28 and 0.73, so no row is
confident enough to reach an extreme band.

**MY HYPOTHESIS WAS WRONG.** I predicted the fix would be a corpus with designed
margin structure and said so explicitly before measuring. That corpus was WORSE:
it needed thirty epochs where plain data needed twenty-five, because forcing rows
close to the separating plane adds ambiguous rows without adding confident ones.

Twenty-five epochs is the cheapest span but leaves only THREE extreme rows -- an
average over three samples, which would flicker back to skipping on any small
change. Fifty gives twenty-five and fifty-eight, healthy on both sides.

## 3. The fix

The fixture trains for fifty epochs, and **the precondition is ASSERTED rather
than skipped**. If the corpus ever stops spanning both regions that is now a
FAILURE requiring the training budget to be re-measured, not a reason to stop
testing the property.

## 4. The result

**The assertion passes.** 1 passed in 7.09 seconds, confirmed twice.

The Monte Carlo Dropout decomposition genuinely exhibits the calibration property
it claims. That was never in evidence before today -- the test asserting it had
never run.

## 5. What I could not verify here, and said so

PyTorch is absent from the environment where this was written, so neither the
prediction distribution nor the assertion could be measured there. The fixture
was therefore designed against numbers measured on the real machine, and the
change was run as a single test from a scratch copy BEFORE being packaged, so
that a failure would be a finding rather than a rollback.

The collection delta is REASONED, not measured: the test always appeared as `s`,
meaning it was collected and skipped, so the count cannot change. The installer
measures it and aborts if that reasoning is wrong.

## 6. Three naming errors of mine, recorded

Within this one file I invented a module path (`models.tabular_nn`, when the
class lives in `variant_ensemble`), a class name (`TestUncertaintyDecomposition`,
when it is `TestPredictProbaSinglePassScientificProperties`), and earlier a
helper name. Each was a plausible construction accepted instead of reading the
source -- the same fault, three times, in one afternoon.

## 7. Files

    tests/unit/test_tabular_nn_mc_dropout.py   fifty epochs; precondition asserted

Ratchet unchanged at 3711. **Skip surface 7 -> 6.**

---

*Written 2026-07-29.*
