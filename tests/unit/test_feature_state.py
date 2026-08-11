"""Tests for the stage-5 feature-state classifier -- HARNESS-NULL-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from genomic_variant_classifier.agent_layer.harness.feature_state import (
    CONSTANCY_STATES, DEFAULT_NEAR_CONSTANT_FRAC, MISSINGNESS_STATES,
    SILENT_ZERO_STATES, FeatureState, classify_feature_state, describe,
)


def _s(values):
    return pd.Series(values, dtype="float64")


# ---- THE DEFECT -----------------------------------------------------------
def test_an_entirely_missing_column_is_NOT_reported_as_zero():
    """The measured 2026-08-09 case. gene_constraint_oe was 200 of 200 NaN and
    the harness reported 'is 100% zero ... probable silent-zero connector'."""
    state, ev = classify_feature_state(_s([np.nan] * 200))
    assert state is FeatureState.ALL_MISSING
    assert ev.n_missing == 200 and ev.n_observed == 0
    msg = describe("gene_constraint_oe", state, ev)
    assert "ENTIRELY MISSING" in msg
    assert "zero" not in msg.lower().replace("zero-filled", ""), msg


def test_all_missing_and_all_zero_are_DIFFERENT_states():
    missing, _ = classify_feature_state(_s([np.nan] * 10))
    zeros, _ = classify_feature_state(_s([0.0] * 10))
    assert missing is FeatureState.ALL_MISSING
    assert zeros is FeatureState.ALL_ZERO_OBSERVED
    assert missing is not zeros


def test_missingness_is_not_in_the_silent_zero_set():
    """Missingness is adjudicated by the declared policy, not by a
    silent-zero audit. Putting it in the silent-zero set would recreate the
    defect one layer up."""
    assert MISSINGNESS_STATES.isdisjoint(SILENT_ZERO_STATES)
    assert FeatureState.ALL_MISSING not in SILENT_ZERO_STATES
    assert FeatureState.MOSTLY_MISSING not in SILENT_ZERO_STATES


def test_constancy_is_NOT_a_silent_zero_finding():
    """SCOPE, pinned. Stage 5 has never had a constancy rule.

    Measured 2026-08-10: putting CONSTANT and NEAR_CONSTANT into the stage-5
    failure set produced a stage-5 failure that had never existed, and the
    suite gate rejected the whole edit as an unexpected regression. A repair
    that also widens the thing it repairs cannot be attributed.
    """
    assert CONSTANCY_STATES.isdisjoint(SILENT_ZERO_STATES)
    assert FeatureState.CONSTANT not in SILENT_ZERO_STATES
    assert FeatureState.NEAR_CONSTANT not in SILENT_ZERO_STATES

    # A non-zero constant column: classified, but NOT a stage-5 failure.
    state, _ = classify_feature_state(_s([1.0] * 100))
    assert state is FeatureState.CONSTANT
    assert state not in SILENT_ZERO_STATES
    assert state not in MISSINGNESS_STATES


def test_the_three_sets_are_mutually_disjoint():
    assert SILENT_ZERO_STATES.isdisjoint(CONSTANCY_STATES)
    assert SILENT_ZERO_STATES.isdisjoint(MISSINGNESS_STATES)
    assert CONSTANCY_STATES.isdisjoint(MISSINGNESS_STATES)


# ---- zero rate is computed among OBSERVED values --------------------------
def test_zero_rate_is_computed_among_observed_only():
    """90 per cent missing AND every observed value zero -- two facts.

    A STATE can carry only one headline. This column is below the 0.95
    missing-rate threshold, so the headline is the dead connector; the
    missingness survives in the EVIDENCE, which is why the evidence is returned
    beside the state rather than folded into it.

    Under the old rule the same column read as "100% zero" with the 90 missing
    values invisible, so the two facts could not be separated at all.
    """
    state, ev = classify_feature_state(_s([np.nan] * 90 + [0.0] * 10))
    assert state is FeatureState.ALL_ZERO_OBSERVED
    assert ev.missing_rate == 0.90, "missingness vanished from the evidence"
    assert ev.zero_rate_observed == 1.0
    assert ev.n_observed == 10
    # The zero rate is over the TEN observed values, not the hundred rows.
    assert ev.n_observed == ev.n - ev.n_missing


def test_the_missing_rate_threshold_is_load_bearing():
    """Just under and just over the threshold must classify differently."""
    under, _ = classify_feature_state(_s([np.nan] * 94 + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    over, _ = classify_feature_state(_s([np.nan] * 96 + [1.0, 2.0, 3.0, 4.0]))
    assert under is FeatureState.HEALTHY
    assert over is FeatureState.MOSTLY_MISSING


def test_missingness_outranks_the_zero_rule_above_the_threshold():
    """At or above the threshold the headline is missingness, because a
    silent-zero verdict on four observed values would be an assertion built on
    almost no evidence."""
    state, ev = classify_feature_state(_s([np.nan] * 96 + [0.0] * 4))
    assert state is FeatureState.MOSTLY_MISSING
    assert ev.zero_rate_observed == 1.0


def test_a_few_missing_values_do_not_manufacture_a_zero_verdict():
    """Under the old rule, 96 NaN among 100 real values read as 96% zero."""
    state, ev = classify_feature_state(_s([np.nan] * 4 + list(np.arange(1, 97) / 10.0)))
    assert state is FeatureState.HEALTHY
    assert ev.zero_rate_observed == 0.0


def test_a_genuinely_dead_connector_is_still_flagged():
    state, ev = classify_feature_state(_s([0.0] * 100))
    assert state is FeatureState.ALL_ZERO_OBSERVED
    assert state in SILENT_ZERO_STATES
    assert "silent-zero" in describe("phylop_score", state, ev)


def test_mostly_zero_observed_is_flagged():
    state, _ = classify_feature_state(_s([0.0] * 96 + [1.5, 2.5, 3.5, 4.5]))
    assert state is FeatureState.MOSTLY_ZERO_OBSERVED
    assert state in SILENT_ZERO_STATES


# ---- the binary exemption --------------------------------------------------
def test_a_well_formed_indicator_is_exempt():
    state, ev = classify_feature_state(_s([0.0] * 98 + [1.0, 1.0]))
    assert ev.binary_observed is True
    assert state is FeatureState.HEALTHY


def test_an_all_zero_column_is_NOT_treated_as_an_indicator():
    """It never takes the other value. The old rule made the same distinction
    and it must survive: {0} is a dead feature, not a legitimate mask."""
    state, ev = classify_feature_state(_s([0.0] * 50))
    assert ev.binary_observed is False
    assert state is FeatureState.ALL_ZERO_OBSERVED


def test_a_binary_column_with_missing_values_is_judged_on_missingness_first():
    state, ev = classify_feature_state(_s([np.nan] * 96 + [0.0, 0.0, 1.0, 1.0]))
    assert state is FeatureState.MOSTLY_MISSING
    assert ev.binary_observed is True


# ---- constancy -------------------------------------------------------------
def test_a_non_zero_constant_is_CONSTANT_not_zero():
    state, ev = classify_feature_state(_s([1.0] * 100))
    assert state is FeatureState.CONSTANT
    assert ev.zero_rate_observed == 0.0
    assert "ONE distinct observed value" in describe("loeuf", state, ev)


def test_near_constant_is_detected_by_modal_fraction():
    state, ev = classify_feature_state(_s([0.5] * 9995 + [0.1, 0.2, 0.3, 0.4, 0.6]))
    assert state is FeatureState.NEAR_CONSTANT
    assert ev.modal_fraction_observed >= DEFAULT_NEAR_CONSTANT_FRAC


def test_a_rare_but_real_second_value_is_not_constant():
    state, ev = classify_feature_state(_s([0.5] * 900 + [0.9] * 100))
    assert state is FeatureState.HEALTHY
    assert ev.n_distinct_observed == 2


# ---- evidence --------------------------------------------------------------
def test_the_evidence_is_immutable():
    import dataclasses
    _, ev = classify_feature_state(_s([1.0, 2.0, np.nan]))
    try:
        ev.n_missing = 999
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("the evidence record accepted a write")


def test_the_evidence_reconciles():
    _, ev = classify_feature_state(_s([np.nan, 0.0, 1.0, 2.0]))
    assert ev.n_missing + ev.n_observed == ev.n
    assert ev.missing_rate == ev.n_missing / ev.n


def test_an_empty_column_does_not_raise():
    state, ev = classify_feature_state(_s([]))
    assert state is FeatureState.ALL_MISSING and ev.n == 0


def test_near_constant_threshold_matches_feature_health():
    """Consume the shared authority rather than restating its value.

    feature_health.py already defines DEFAULT_NEAR_CONSTANT_FRAC. Two constants
    with the same meaning and independent values is the drift this module was
    written to avoid.
    """
    try:
        from genomic_variant_classifier.data.feature_health import (
            DEFAULT_NEAR_CONSTANT_FRAC as AUTHORITY)
    except ImportError:
        import pytest
        pytest.skip("feature_health not importable in this tree")
    assert DEFAULT_NEAR_CONSTANT_FRAC == AUTHORITY, (
        "two independent definitions of near-constant: {} here against {} in "
        "feature_health".format(DEFAULT_NEAR_CONSTANT_FRAC, AUTHORITY))


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
