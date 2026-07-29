# SESSION 2026-07-28 — CI-t was discharged prematurely; the enumeration that proves it now

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `19c19a1`, ratchet 3589
**Roadmap position:** correction to CI-t, immediately before CI-q
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What went wrong

Commit `19c19a1` gated the report path against unvalidated model output and
declared CI-t discharged. The claim rested on a HAND COUNT of ten call sites.

**A parsed enumeration found twelve.** `_consequence_breakdown` calls
`roc_auc_score` and `average_precision_score` directly, and raises on a corrupt
model:

    evaluator.py:1280  roc_auc_score            _consequence_breakdown   UNGATED
    evaluator.py:1281  average_precision_score  _consequence_breakdown   UNGATED

That path is reached ONLY when `meta` is supplied. **Every corrupt-model test
written for CI-t passed `meta=None`**, so the fixture shape hid it — the same
failure that hid the calibration binning defect for seventeen days, and the
second time in this session my own fixtures were blind to a path I had just
claimed to cover.

It was found by accident: the next measurement supplied `meta` in order to
exercise clustered bootstrap intervals, and the call raised.

## 2. The fix is not the gate; it is the enumeration

Adding the missing gate would leave the same method in place — count by hand,
declare the class closed, wait for a fixture to stumble into the next gap.

`test_every_metric_call_sits_inside_a_gated_function` parses the module, finds
every scikit-learn metric call, and requires that its enclosing function contain
a validator whose result GOVERNS A BRANCH. A hand count is a claim; this is a
check, and a call site added tomorrow fails it immediately.

### 2.1 The guard was too weak twice before it was right

**First version: vocabulary, not structure.** It asked whether a validator NAME
appeared in the function. Disabling a gate with `if False:` leaves the name in
place, so a dead gate satisfied it — the identical weakness already found once in
the carried-item register's `"ast" in text` predicate.

**Second version: one hop only.** Requiring the validator's result to appear
directly in an `if` test flagged `evaluate` as ungated, because its chain is
three hops:

    probability_check  = validate_probabilities(...)
    probability_usable = label_check.ok and probability_check.ok
    if probability_usable: ...

A false positive is not harmless. A guard that cries wolf on correct code gets
weakened until it catches nothing. The check now propagates through assignments
to a fixed point.

**Third weakness, found by sabotage.** Reverting the enumeration's CALL SITE to a
substring test survived, because `_validation_governs_a_branch` itself was
untouched and all its own tests still passed. Closed by asserting, from the parsed
test source, that the enumeration actually calls it.

## 3. Measurements completed

### 3.1 Interval formatting cannot distinguish four states

    case                              ci_status             certified  rendered
    unavailable, no bootstrap         insufficient_support  False      unavailable (...)
    attempted, no cluster identifier  insufficient_support  False      unavailable (...)
    attempted, with clusters          insufficient_data     False      unavailable (...)
    FAILED, invalid model input       failed                False      unavailable (...)

`format_ci` renders all four identically and `certified` is `False` in all four.
Only the typed status separates them — and rows one and two are indistinguishable
even by status.

**Consequence for CI-q: ranking admissibility must come from the typed AUROC
point result, never from the formatted string or the certification Boolean.**

### 3.2 The comparison artifact has no consumers

The only reference to `output_csv` outside `compare_models` is a test passing
`os.devnull`. Nothing reads `models/model_comparison.csv`. The ordinal-access
hits found by search are all in unrelated modules on different frames.

**Consequence for CI-q:** the staged compatibility migration exists to protect
readers, and there are none. CI-q can define the artifact properly in one commit.
The eleven columns are still worth preserving — a human reading the file expects
them — but on grounds of churn, not compatibility.

## 4. Verification

Regression `FAILED` list byte-identical at 40. The frozen report oracle moves
only `schema_version`, commit 3b-2's declared field.

**Sabotage: six mutations, six detected, zero undetected.**

| break | detected |
|---|---|
| B1 the breakdown gate is disabled with a constant | yes |
| B2 the breakdown gate is deleted entirely | yes |
| B3 a NEW ungated metric call is added | yes |
| B4 the enumeration stops following assignment chains | yes |
| B5 the enumeration accepts a mere name mention | yes |
| B6 the enumeration finds nothing at all | yes |

B5 required two attempts. Weakening a guard is invisible on clean code, because
the weak and strong checks agree there; the difference appears only in
combination with another break. It is closed by a guard-the-guard asserting the
check REJECTS a dead gate and ACCEPTS a chained one, plus a parsed assertion that
the enumeration calls it.

## 5. The register

CI-t's row now records that its discharge was premature and why. Its predicate no
longer duplicates the check: it delegates to the gates suite, because two
implementations of one check is the defect the register exists to prevent.

## 6. Files

    src/genomic_variant_classifier/evaluation/evaluator.py   subgroup breakdown gated
    tests/unit/test_report_input_gates.py                    30 -> 35
    tests/unit/test_carried_item_register.py                 predicate delegated
    docs/CARRIED_ITEMS.md                                    premature discharge recorded

Ratchet 3589 -> 3594 (+5), measured.

---

*Written 2026-07-28.*
