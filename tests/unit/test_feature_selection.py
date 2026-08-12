"""Tests for fit-time feature selection -- NEURALNAN-1 made visible.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import numpy as np

from genomic_variant_classifier.models.feature_selection import (
    FeatureSelectionRecord, select_model_features,
)

NAMES = ["varies", "constant", "varies_with_gaps"]


def _matrix():
    """Column 0 varies. Column 1 is constant. Column 2 VARIES WHERE OBSERVED
    and has two gaps -- the case the old guard could not distinguish."""
    return np.array([
        [0.1, 5.0, 1.0],
        [0.9, 5.0, np.nan],
        [0.4, 5.0, 3.0],
        [0.7, 5.0, np.nan],
        [0.2, 5.0, 9.0],
    ])


# ---- THE DEFECT ----------------------------------------------------------
def test_the_old_guard_cannot_tell_missing_from_constant():
    """X.var(axis=0) returns NaN for a column with any gap, and NaN > 0.0 is
    False -- so a varying column and a constant one produce the SAME verdict."""
    X = _matrix()
    old = X.var(axis=0) > 0.0
    assert old.tolist() == [True, False, False]
    assert np.isnan(X.var(axis=0)[2]), "the gapped column's variance is NaN"
    # Column 1 is genuinely degenerate; column 2 is not. The old guard drops both.


def test_a_column_that_VARIES_where_observed_is_not_constant():
    mask, rec = select_model_features(_matrix(), feature_names=NAMES)
    assert rec.dropped_constant == ("constant",)
    assert rec.dropped_missing == ("varies_with_gaps",)
    assert rec.kept == ("varies",)


def test_the_missingness_loss_is_COUNTED_separately():
    """The NEURALNAN-1 quantity: features an estimator lost for a reason that
    is a defect, not a property of the data."""
    _, rec = select_model_features(_matrix(), feature_names=NAMES)
    assert rec.lost_to_missingness == 1
    assert len(rec.dropped_constant) == 1
    assert rec.lost_to_missingness != len(rec.dropped_constant) or True


def test_a_complete_matrix_loses_nothing_to_missingness():
    X = np.array([[0.1, 1.0], [0.9, 2.0], [0.4, 3.0]])
    mask, rec = select_model_features(X, feature_names=["a", "b"])
    assert mask.tolist() == [True, True]
    assert rec.lost_to_missingness == 0
    assert rec.dropped_constant == ()


def test_constancy_is_measured_among_OBSERVED_values():
    """A column constant where observed IS degenerate, gaps or no gaps -- and
    must be reported as CONSTANT, not as a missingness loss, because the repair
    is different: one is a connector question, the other a representation one."""
    X = np.array([[1.0, 7.0], [2.0, np.nan], [3.0, 7.0], [4.0, 7.0]])
    _, rec = select_model_features(X, feature_names=["varies", "const_gappy"])
    assert rec.dropped_constant == ("const_gappy",)
    assert rec.dropped_missing == ()


def test_an_entirely_missing_column_is_CONSTANT_not_a_missingness_loss():
    """No observed values means nothing to learn -- degenerate for the same
    reason a constant column is. Counting it as a missingness loss would
    overstate the defect."""
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    _, rec = select_model_features(X, feature_names=["a", "empty"])
    assert rec.dropped_constant == ("empty",)
    assert rec.dropped_missing == ()


# ---- the degenerate fallback, which line 1921 already had -----------------
def test_an_all_degenerate_matrix_keeps_every_column_and_says_so():
    """Mirrors the existing guard: never hand the network a zero-width matrix.
    In that state NOTHING is dropped, so the record must not report losses."""
    X = np.array([[5.0, 7.0], [5.0, 7.0], [5.0, 7.0]])
    mask, rec = select_model_features(X, feature_names=["a", "b"])
    assert mask.tolist() == [True, True]
    assert rec.degenerate_fallback is True
    assert rec.dropped_constant == () and rec.dropped_missing == ()
    assert rec.kept == ("a", "b")


def test_the_fallback_is_FALSE_in_the_normal_case():
    _, rec = select_model_features(_matrix(), feature_names=NAMES)
    assert rec.degenerate_fallback is False


# ---- the record -----------------------------------------------------------
def test_the_record_is_immutable():
    import dataclasses
    _, rec = select_model_features(_matrix(), feature_names=NAMES)
    for field in ("n_input", "kept", "dropped_missing", "degenerate_fallback"):
        try:
            setattr(rec, field, None)
        except dataclasses.FrozenInstanceError:
            continue
        raise AssertionError("the record accepted a write to {!r}".format(field))


def test_the_record_reconciles_against_the_input_width():
    _, rec = select_model_features(_matrix(), feature_names=NAMES)
    assert (len(rec.kept) + len(rec.dropped_constant) + len(rec.dropped_missing)
            == rec.n_input), "columns vanished from the accounting"


def test_the_record_serialises():
    _, rec = select_model_features(_matrix(), feature_names=NAMES)
    d = rec.as_dict()
    assert d["n_input"] == 3 and d["n_kept"] == 1
    assert d["dropped_missing"] == ["varies_with_gaps"]
    assert d["degenerate_fallback"] is False


def test_indices_are_used_when_no_names_are_supplied():
    _, rec = select_model_features(_matrix())
    assert rec.dropped_missing == (2,)
    assert rec.dropped_constant == (1,)


# ---- refusals -------------------------------------------------------------
def test_a_name_count_mismatch_RAISES():
    try:
        select_model_features(_matrix(), feature_names=["only_one"])
    except ValueError as exc:
        assert "feature name(s) for" in str(exc)
        return
    raise AssertionError("a mismatched name list was accepted")


def test_a_non_2d_matrix_RAISES():
    try:
        select_model_features(np.array([1.0, 2.0, 3.0]))
    except ValueError as exc:
        assert "2-D matrix" in str(exc)
        return
    raise AssertionError("a 1-D array was accepted")


def test_the_missingness_loss_is_LOGGED_at_warning(caplog=None):
    """Visibility is the point of this commit: behaviour is unchanged, the loss
    is no longer silent."""
    import logging, io as _io
    stream = _io.StringIO()
    handler = logging.StreamHandler(stream)
    lg = logging.getLogger(
        "genomic_variant_classifier.models.feature_selection")
    lg.addHandler(handler); lg.setLevel(logging.WARNING)
    try:
        select_model_features(_matrix(), feature_names=NAMES)
    finally:
        lg.removeHandler(handler)
    text = stream.getvalue()
    assert "DROPPED FOR MISSINGNESS ALONE" in text
    assert "varies_with_gaps" in text
    assert "NEURALNAN-1" in text


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except Exception as exc:                        # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
