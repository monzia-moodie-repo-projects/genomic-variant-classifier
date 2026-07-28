# SESSION 2026-07-28 — the derived single-class AUPRC rule (commit 3b-1b)

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `6029d74`, ratchet 3517
**Roadmap position:** Tier 1 item 6, commit 3b-1b — the last step before the authority switch
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What this commit closes

The shadow comparison between the legacy computation and the registry-backed
projection stood at two mismatches after 3b-1a, both AUPRC on single-class
cohorts. This commit takes it to **zero**, which is the precondition 3b-2's
authority switch has been waiting on.

    before 3b-1a   6 mismatches
    after  3b-1a   2, exactly the two AUPRC identities
    after  3b-1b   0

---

## 2. AUPRC stays canonically undefined; the legacy value is DERIVED

AUPRC is a ranking quantity built around the positive class. On an all-negative
cohort there are no positives to retrieve, so recall has no meaningful
denominator. On an all-positive cohort precision is trivially one at every
retrieval point, but no negatives exist against which ranking quality could be
assessed — the returned value is determined by class composition rather than by
discrimination. scikit-learn warns in both cases while returning its conventional
answer, and that warning is the evidence the answer is not a measurement.

So the typed surface keeps refusing:

    status  UNDEFINED
    reason  binary_class_support_required

And the legacy scalar is DERIVED from the registered prevalence:

    prevalence 0.0  ->  legacy AUPRC 0.0
    prevalence 1.0  ->  legacy AUPRC 1.0

This is not a second AUPRC computation and not a table constant. It is an
explicit schema-version-2 serialisation rule keyed on class composition, which
means the rule states WHY the value is what it is and cannot silently produce it
for a cohort where the premise does not hold.

### 2.1 It fails closed on every way the premise can be absent

    no sibling results        -> LegacyProjectionError
    no prevalence result      -> LegacyProjectionError
    prevalence not OK         -> LegacyProjectionError
    prevalence not degenerate -> LegacyProjectionError

The last is the important one. An AUPRC refused for a SINGLE-CLASS cohort while
prevalence reads 0.42 is a contradiction between two statements about the same
data. Emitting a plausible legacy value there would hide the inconsistency rather
than surface it, and a derived value must never rest on a quantity that was
itself refused.

---

## 3. A third projection source, and why it needed a name

`ProjectionDecision` now records three sources rather than two:

    typed_value   the metric's own value
    substitute    a CONSTANT authorised by exact reason
    derived       computed from a sibling result

All three can produce `0.0` on the same report:

    constant_classifier.f1        0.0   typed_value   a measurement
    degenerate_all_negative.f1    0.0   substitute    a compatibility constant
    degenerate_all_negative.auprc 0.0   derived       from prevalence 0.0

Three identical numbers, three different meanings, and no assertion comparing
values can separate them. A test pins that the three sources are distinguishable
precisely because the values are not.

The derived rule is also proved NOT to be a constant in disguise: the same rule
produces 0.0 on one degenerate cohort and 1.0 on the other, which no constant
could do.

---

## 4. A signature change that broke a spy, correctly

The resolver gained `metric_results` so the derived rule can read prevalence.
That broke the resolver-counting spy in the calibration suite, which still had
the old signature — and the break is the right behaviour. A spy that silently
swallowed extra keyword arguments would keep passing while no longer standing in
for the real function, which is the failure mode that test exists to prevent one
layer down.

---

## 5. Verification

### 5.1 Shadow equality reached

Eighty policy-covered values across ten cohorts, ZERO mismatches. This is the
executable equivalence proof 3b-2 requires: the old and new implementations agree
on the same execution, not merely on a final tree.

### 5.2 Both oracles

    LEGACY REPORT ORACLE   480 values   ZERO movements
    TYPED REGISTRY ORACLE  384 values   10 movements, ALL from 3b-1a's declared
                                        set, NOTHING new

### 5.3 Sabotage matrix

Nine mutations, **nine detected, zero undetected**, clean on the first pass.

| break | detected | tests |
|---|---|---|
| B1 the derived rule becomes a constant 0.0 | yes | 7 |
| B2 a mixed prevalence is accepted instead of refused | yes | 1 |
| B3 a non-OK prevalence is accepted | yes | 2 |
| B4 a missing prevalence is tolerated | yes | 1 |
| B5 absent siblings are tolerated | yes | 1 |
| B6 prevalence 1.0 maps to 0.0 | yes | 2 |
| B7 the derived source is relabelled as a substitution | yes | 4 |
| B8 the AUPRC rule fires for any undefined reason | yes | 2 |
| B9 the AUPRC rule is swapped onto the ZERO rule | yes | 13 |

B1 and B9 are the two that matter most. B1 replaces the derivation with the
constant it superficially resembles on one cohort — caught by seven tests,
because the other cohort derives 1.0. B9 is a rule swap of exactly the kind
maintenance produces, caught by thirteen.

---

## 6. Files

    src/genomic_variant_classifier/evaluation/legacy_projection.py   SINGLE_CLASS_AUPRC
    tests/unit/test_legacy_projection.py                             48 -> 60
    tests/unit/test_calibration_binning_convention.py                spy signature

Ratchet 3517 -> 3529 (+12), measured by `pytest --collect-only`.

---

## 7. Next

**3b-2**, the last commit of Tier 1 item 6. `project_legacy_fields` becomes
authoritative; `evaluate()` gains an optional `source_id` and constructs one of
two honest population identities; the evaluator's local computation at lines
481-482 and 511 is deleted; the narrowed abstract-syntax-tree guard and the
invocation-count guards activate; `certification_eligible = False` for
unattributed populations. The diff is almost entirely subtractive, which is the
best possible structure for retiring duplicated scientific computation, and the
acceptance criterion is 480 legacy values with zero movement.

---

*Written 2026-07-28.*
