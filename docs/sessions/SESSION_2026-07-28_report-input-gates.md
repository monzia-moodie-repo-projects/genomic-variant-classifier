# SESSION 2026-07-28 — report-path input gates (CI-t)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `0b50dcc`, ratchet 3558
**Roadmap position:** CI-t, the prerequisite to CI-q
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. How this was found

CI-q — the shared-population model comparison — could not be built on a loop
that already died unpredictably during scoring. Investigating why produced this
commit.

## 2. Why validation must precede dispatch

Five scikit-learn calls sit in the report path on the same `(y, p)` pair. Their
behaviour under bad input is inconsistent. Measured 2026-07-28:

| defect | `roc_curve` | `pr_curve` | `calibration_curve` | `roc_auc` | `avg_precision` |
|---|---|---|---|---|---|
| non-finite probabilities | **raises** | **raises** | **returns** | **raises** | **raises** |
| outside the unit interval | returns | returns | **raises** | returns | returns |
| single class | warns | warns | returns | warns | warns |

No consistent rule exists to translate, so the library cannot be allowed to
decide which defect becomes which status. It does not agree with itself.

## 3. Three defects in landed code, ascending in danger

**`roc_curve` raises and aborts the report** — after the point metrics have
already been computed successfully, discarding them.

**`calibration_curve` neither raises nor warns.** With 40 of 200 probabilities
non-finite it returns a degenerate one-point curve carrying NaN, down from ten
points. The failure surfaces only at persistence, where strict JSON refuses the
artifact and names the calibration curve rather than the corrupt model.

**THE OPERATING-POINT SWEEP SHIPPED A WRONG NUMBER.** `preds = (p >= t)`
evaluates FALSE for a NaN, so every unusable prediction silently became a
PREDICTED NEGATIVE. Measured with 100 of 200 true positives corrupted:

    clean     threshold 0.6366   sensitivity 0.90   specificity 1.00   ppv 1.0000
    corrupt   threshold 0.0000   sensitivity 0.50   specificity 0.00   ppv 0.3333

An exception is loud. A poisoned curve fails at persistence. This shipped a
plausible clinical decision threshold over a cohort nobody declared.

## 4. What was built

`evaluation/input_validation.py` with three composable validators — reference
labels, ranking scores, probabilities — separated so a failed probability check
cannot suppress valid score-based ranking.

**Ten call sites gated** before dispatch: three curves, four interval calls
through `_interval_fields`, and three operating points.

**Refusal is component-level.** A corrupt model yields a structurally complete
report: attempted population unchanged, prevalence still valid, typed results
present, each withheld component named in a warning.

**A `scores` channel**, validated WITHOUT a range restriction, because a score is
an ordering and not a magnitude on any scale. An out-of-range array is refused as
a probability and accepted as a score — the same array, two channels, two correct
answers.

## 5. The caller measurement that made this a correctness fix

Every production caller obtains its array from `predict_proba(...)[:, 1]`:

    scripts/train.py:587           ensemble.predict_proba(...)[:, 1]
    scripts/run9_ablations.py:737  ensemble.predict_proba(X_test_abl, None)[:, 1]
    compare_models                 documented `{model_name: proba_array}`
    26 test call sites             bounded to [0, 1] by construction

There are NO callers passing arbitrary scores through `y_proba`. Enforcing the
unit interval is therefore a correctness fix with zero compatibility break, and
the staged migration the design contemplated was unnecessary.

## 6. Five defects in my own work, each caught by measurement

**Gating two of three operating points** left `at_high_ppv` — a separate function
— still reporting sensitivity 0.5, specificity 0.875, positive predictive value
0.8. Found by checking all three report fields rather than the two just changed.

**Gating the curves on the RANKING channel** let an out-of-range array reach
`roc_curve` while `calibration_curve` refused it, preserving the exact incoherent
contract this commit removes.

**An ordering violation**, exposed by my own printed check: `ranking_values`
defined at character 36090, the registry call at 39555 — defined after use.

**The fallback `ranking_values = p`** left the seam open: the registry ranked an
invalid array as `auroc 1.0` while the curve computed from the same values was
withheld. One input, two layers, opposite verdicts.

**A refused `scores` array was still forwarded** to the registry, so a mis-sized
array raised `ValueError` from the context's own length check — turning a
refusal this gate exists to make graceful back into an exception three layers
down. Found by a test written for the sabotage matrix.

### 6.1 And twice my prose contradicted the number beside it

Once claiming a `calibration_curve` output was "finite" when the measurement one
line above read `NaN`. Once claiming the seam was closed when the row read
`flat=1.0 typed=ok`. Both times the measurement was right.

This is the most dangerous hazard recorded in this project, because unlike a
failing test it produces a confident, plausible, WRONG statement. It belongs
beside the four malformed probes — every one of which searched a SUPERSET of what
the question asked and returned an answer to a question nobody posed.

## 7. Verification

Regression `FAILED` list byte-identical at 40. The frozen report oracle moves
only `schema_version`, which is commit 3b-2's already-declared field.

**Sabotage: eleven mutations, eleven detected, zero undetected.**

The first run left two, and both were coverage gaps for the two pieces written
LAST — the seam fix and the `scores` channel. The tests had been written against
an earlier design and never caught up. Closing them found the fifth defect above.

## 8. Files

    src/genomic_variant_classifier/evaluation/input_validation.py  NEW
    src/genomic_variant_classifier/evaluation/evaluator.py         ten call sites gated
    tests/unit/test_report_input_gates.py                          NEW, 30 tests
    tests/unit/test_carried_item_register.py                       CI-t predicate
    docs/CARRIED_ITEMS.md                                          CI-t recorded and discharged

Ratchet 3558 -> 3589 (+31), measured.

## 9. Next

CI-q can now assume every submitted model completes into a report object, even
when one model's predictions are unusable. That is the guarantee its
shared-population design requires.

---

*Written 2026-07-28.*
