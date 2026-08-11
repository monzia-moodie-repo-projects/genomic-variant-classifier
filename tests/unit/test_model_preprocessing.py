"""Tests for TabularModelPreprocessor -- MISSINGNESS-POLICY-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from genomic_variant_classifier.models.model_preprocessing import (
    DECLARED_MISSINGNESS, MASK_SUFFIX, MissingCapability, MissingFeaturePolicy,
    MissingStrategy, TabularModelPreprocessor, UndeclaredMissingnessError,
    policy_fingerprint,
)

FEATURES = ("cadd_phred", "gene_constraint_oe", "gene_is_constrained")


def _frame(oe, constrained, cadd=None):
    n = len(oe)
    return pd.DataFrame({
        "cadd_phred": [10.0] * n if cadd is None else cadd,
        "gene_constraint_oe": oe,
        "gene_is_constrained": constrained,
    })


def _fitted(capability=MissingCapability.REQUIRES_NUMERIC):
    train = _frame([0.2, 0.4, 0.6, np.nan], [1.0, 1.0, 0.0, np.nan])
    return TabularModelPreprocessor(FEATURES, capability=capability).fit(train), train


# ---- the schema is declared, never discovered ----------------------------
def test_output_schema_is_fixed_by_POLICY_not_by_the_data():
    """scikit-learn's add_indicator emits a mask only for features missing
    DURING FIT. A column complete in training and missing at serving would then
    produce no mask, handing the model a shape it has never seen."""
    complete = _frame([0.2, 0.4], [1.0, 0.0])
    p = TabularModelPreprocessor(FEATURES).fit(complete)
    out = p.transform(complete)
    assert list(out.columns) == [
        "cadd_phred", "gene_constraint_oe", "gene_is_constrained",
        "gene_constraint_oe" + MASK_SUFFIX, "gene_is_constrained" + MASK_SUFFIX]
    assert float(out["gene_constraint_oe" + MASK_SUFFIX].sum()) == 0.0


def test_schema_is_identical_whether_or_not_training_had_missing_values():
    a = TabularModelPreprocessor(FEATURES).fit(_frame([0.2, 0.4], [1.0, 0.0]))
    b, _ = _fitted()
    assert a.output_schema() == b.output_schema()


def test_a_column_complete_in_training_but_missing_at_serving_still_gets_a_mask():
    p = TabularModelPreprocessor(FEATURES).fit(_frame([0.2, 0.4], [1.0, 0.0]))
    out = p.transform(_frame([np.nan, 0.4], [1.0, 0.0]))
    assert float(out.loc[0, "gene_constraint_oe" + MASK_SUFFIX]) == 1.0
    assert float(out.loc[1, "gene_constraint_oe" + MASK_SUFFIX]) == 0.0
    assert not out["gene_constraint_oe"].isna().any()


# ---- the three-state encoding -------------------------------------------
def test_the_three_state_encoding_is_injective():
    """(value, mask) must distinguish all three semantic states.

        (0,0) known false   (1,0) known true   (0,1) unknown

    A median on gene_is_constrained would collapse unknown into whichever
    class training prevalence favours -- a biological assertion, which is the
    defect DUPLICATE-1 was about.
    """
    p, _ = _fitted()
    out = p.transform(_frame([0.2, 0.2, 0.2], [0.0, 1.0, np.nan]))
    pairs = list(zip(out["gene_is_constrained"],
                     out["gene_is_constrained" + MASK_SUFFIX]))
    assert pairs == [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assert len(set(pairs)) == 3, "two distinct states collapsed to one encoding"


def test_unknown_is_not_imputed_to_the_majority_class():
    """Training is 2/3 constrained; a median would make unknown = 1."""
    train = _frame([0.2, 0.3, 0.9, np.nan], [1.0, 1.0, 0.0, np.nan])
    p = TabularModelPreprocessor(FEATURES).fit(train)
    out = p.transform(_frame([0.2], [np.nan]))
    assert float(out.loc[0, "gene_is_constrained"]) == 0.0
    assert float(out.loc[0, "gene_is_constrained" + MASK_SUFFIX]) == 1.0


# ---- fitted on training only ---------------------------------------------
def test_the_median_comes_from_TRAINING_and_does_not_move_at_transform():
    p, _ = _fitted()
    assert p.medians_["gene_constraint_oe"] == 0.4
    out = p.transform(_frame([np.nan, 99.0, 99.0, 99.0], [0.0, 0.0, 0.0, 0.0]))
    assert float(out.loc[0, "gene_constraint_oe"]) == 0.4, (
        "the imputed value moved with the transform batch -- statistics leaked "
        "out of the training partition")


def test_a_feature_entirely_missing_in_training_RAISES():
    train = _frame([np.nan, np.nan], [1.0, 0.0])
    try:
        TabularModelPreprocessor(FEATURES).fit(train)
    except UndeclaredMissingnessError as exc:
        assert "entirely missing" in str(exc) and "leak" in str(exc)
        return
    raise AssertionError("a median was invented for a feature with no observations")


# ---- fail closed ---------------------------------------------------------
def test_undeclared_missingness_RAISES_at_fit():
    train = _frame([0.2, 0.4], [1.0, 0.0], cadd=[10.0, np.nan])
    try:
        TabularModelPreprocessor(FEATURES).fit(train)
    except UndeclaredMissingnessError as exc:
        assert "cadd_phred" in str(exc) and "no declared policy" in str(exc)
        return
    raise AssertionError("undeclared missingness was silently filled")


def test_undeclared_missingness_RAISES_at_transform_too():
    p, _ = _fitted()
    try:
        p.transform(_frame([0.2], [0.0], cadd=[np.nan]))
    except UndeclaredMissingnessError as exc:
        assert "cadd_phred" in str(exc)
        return
    raise AssertionError("new undeclared missingness passed through transform")


def test_FORBID_raises_rather_than_imputing():
    pol = dict(DECLARED_MISSINGNESS)
    pol["cadd_phred"] = MissingFeaturePolicy(MissingStrategy.FORBID, False, "x")
    try:
        TabularModelPreprocessor(FEATURES, policies=pol).fit(
            _frame([0.2, 0.4], [1.0, 0.0], cadd=[10.0, np.nan]))
    except UndeclaredMissingnessError as exc:
        assert "forbids missing values" in str(exc)
        return
    raise AssertionError("a FORBID feature was imputed")


def test_FORBID_raises_at_TRANSFORM_when_training_was_complete():
    """The serving-time case, which is the one that matters.

    fit() and transform() carry SEPARATE FORBID checks. A test that only fits
    leaves the transform guard unexercised -- a sabotage run disabling it went
    undetected. A feature complete in training can go missing at serving, and
    that is exactly when FORBID must fire.
    """
    pol = dict(DECLARED_MISSINGNESS)
    pol["cadd_phred"] = MissingFeaturePolicy(
        MissingStrategy.FORBID, False, "a missing CADD score is a join defect")
    p = TabularModelPreprocessor(FEATURES, policies=pol).fit(
        _frame([0.2, 0.4], [1.0, 0.0], cadd=[10.0, 12.0]))   # complete: fit passes
    try:
        p.transform(_frame([0.2], [0.0], cadd=[np.nan]))
    except UndeclaredMissingnessError as exc:
        assert "forbids missing values" in str(exc) and "cadd_phred" in str(exc)
        return
    raise AssertionError("a FORBID feature was filled at transform time")


def test_FORBID_is_not_bypassed_by_a_native_capability():
    """The NATIVE early-return must not skip the FORBID check: 'this estimator
    tolerates NaN' and 'this feature may be missing' are different claims."""
    pol = dict(DECLARED_MISSINGNESS)
    pol["cadd_phred"] = MissingFeaturePolicy(MissingStrategy.FORBID, False, "x")
    p = TabularModelPreprocessor(
        FEATURES, policies=pol, capability=MissingCapability.NATIVE).fit(
            _frame([0.2, 0.4], [1.0, 0.0], cadd=[10.0, 12.0]))
    try:
        p.transform(_frame([0.2], [0.0], cadd=[np.nan]))
    except UndeclaredMissingnessError:
        return
    raise AssertionError(
        "FORBID was bypassed because the estimator tolerates NaN")


def test_transform_before_fit_raises():
    try:
        TabularModelPreprocessor(FEATURES).transform(_frame([0.2], [0.0]))
    except RuntimeError:
        return
    raise AssertionError("transform ran on an unfitted preprocessor")


def test_a_missing_input_column_raises():
    p, _ = _fitted()
    try:
        p.transform(pd.DataFrame({"cadd_phred": [1.0]}))
    except ValueError as exc:
        assert "missing column" in str(exc)
        return
    raise AssertionError("a truncated semantic matrix was accepted")


# ---- native-missing estimators -------------------------------------------
def test_native_estimators_keep_their_NaN_and_still_get_the_masks():
    """Imputing for xgboost/lightgbm/catboost would deprive them of their own
    missing-value machinery and confound the algorithm comparison."""
    p, _ = _fitted(capability=MissingCapability.NATIVE)
    out = p.transform(_frame([np.nan], [np.nan]))
    assert pd.isna(out.loc[0, "gene_constraint_oe"])
    assert float(out.loc[0, "gene_constraint_oe" + MASK_SUFFIX]) == 1.0
    assert float(out.loc[0, "gene_is_constrained" + MASK_SUFFIX]) == 1.0


def test_both_capabilities_receive_the_SAME_INFORMATION():
    """Only the encoding of 'unavailable' differs; the masks are identical, so
    a performance difference between estimators cannot be an artefact of one
    receiving a richer missingness representation."""
    numeric, train = _fitted(MissingCapability.REQUIRES_NUMERIC)
    native = TabularModelPreprocessor(
        FEATURES, capability=MissingCapability.NATIVE).fit(train)
    probe = _frame([np.nan, 0.3], [np.nan, 1.0])
    a, b = numeric.transform(probe), native.transform(probe)
    assert list(a.columns) == list(b.columns)
    for m in ("gene_constraint_oe" + MASK_SUFFIX, "gene_is_constrained" + MASK_SUFFIX):
        assert a[m].tolist() == b[m].tolist()


# ---- identity ------------------------------------------------------------
def test_policy_fingerprint_changes_with_the_policy():
    a = policy_fingerprint(DECLARED_MISSINGNESS)
    altered = dict(DECLARED_MISSINGNESS)
    altered["gene_is_constrained"] = MissingFeaturePolicy(
        MissingStrategy.MEDIAN, True, "changed")
    assert a != policy_fingerprint(altered)
    assert a == policy_fingerprint(dict(DECLARED_MISSINGNESS))


def test_identity_records_the_schema_and_policy():
    p, _ = _fitted()
    d = p.identity_.as_dict()
    assert d["n_input_features"] == 3 and d["n_output_features"] == 5
    assert len(d["policy_fingerprint"]) == 64


def test_a_policy_for_an_undeclared_feature_is_refused():
    pol = dict(DECLARED_MISSINGNESS)
    pol["not_a_feature"] = MissingFeaturePolicy(MissingStrategy.MEDIAN, True, "x")
    try:
        TabularModelPreprocessor(FEATURES, policies=pol)
    except ValueError as exc:
        assert "absent from the contract" in str(exc)
        return
    raise AssertionError("a policy referenced a feature outside the contract")


def test_duplicate_feature_names_are_refused():
    try:
        TabularModelPreprocessor(("a", "b", "a"))
    except ValueError as exc:
        assert "duplicate" in str(exc)
        return
    raise AssertionError("a duplicated feature name was accepted")


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except Exception as exc:                       # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
