"""Operating-point selection semantics: the legacy contract, frozen.

OP-0, 2026-08-04. This module exists because `_find_high_ppv_point` was exercised
by NOTHING -- measured across the whole test tree, the only match for either
finder was a comment. OP-1 is about to replace this subsystem, and a shadow
comparison against an unexercised selector proves very little.

The two tests here are deliberately different in kind, a distinction REG-1
established on 2026-08-03:

    behavioural  proves outputs and refusals
    structural   proves ownership, derivation and authority paths

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

# The non-monotone counterexample. Measured 2026-08-04 against the live selector:
#
#     t=0.90   ppv=1.0000   sensitivity=0.5000   FEASIBLE        <- selected
#     t=0.80   ppv=0.5000   sensitivity=0.5000   violates floor  <- break fires
#     t=0.70   ppv=0.6667   sensitivity=1.0000   FEASIBLE, UNREACHABLE
#
# Positive predictive value is NOT monotone in the threshold, which is the whole
# reason Objective A and Objective B differ.
NON_MONOTONE_Y = np.array([1.0, 0.0, 1.0])
NON_MONOTONE_P = np.array([0.9, 0.8, 0.7])
PPV_FLOOR = 0.60


def _evaluator() -> ClinicalEvaluator:
    """The suite's own construction form, used at twenty sites across tests/."""
    return ClinicalEvaluator(n_bootstrap=0, random_state=42)


def test_legacy_high_ppv_selector_uses_conservative_prefix_semantics():
    """THE LEGACY CONTRACT, FROZEN. This is Objective B, not Objective A.

    The selector walks candidates from conservative to permissive and STOPS at
    the first violation of the floor, returning the last preceding candidate. On
    the fixture above it selects t=0.90 at sensitivity 0.5, and never reaches
    t=0.70 -- which satisfies the SAME floor at DOUBLE the sensitivity.

    Every value below was MEASURED on 2026-08-04, not derived. The decision of
    that date was explicit: freeze the measured output, not the illustration.

    THE PURPOSE IS NOT TO CHECK ARITHMETIC. It is to freeze the A-versus-B
    distinction, so that when OP-1's shadow comparison differs, the difference is
    a DECLARED POLICY CHANGE rather than an unexplained numerical movement --
    not a sweep defect, a tie-breaking defect, a threshold-order defect, or a
    regression in positive predictive value.
    """
    point = _evaluator()._find_high_ppv_point(
        NON_MONOTONE_Y, NON_MONOTONE_P, min_ppv=PPV_FLOOR)

    assert point is not None, (
        "the legacy selector found no operating point on a fixture where two "
        "candidates satisfy the floor")

    assert point.threshold == pytest.approx(0.9)
    assert point.sensitivity == pytest.approx(0.5)
    assert point.specificity == pytest.approx(1.0)
    assert point.ppv == pytest.approx(1.0)
    assert point.npv == pytest.approx(0.5)
    assert point.f1 == pytest.approx(0.6667, abs=1e-4)
    assert point.n_flagged == 1
    assert (point.n_tp, point.n_fp, point.n_fn, point.n_tn) == (1, 0, 1, 1)


def test_a_feasible_higher_sensitivity_candidate_is_unreachable_by_the_legacy_rule():
    """THE ARGUMENT FOR OBJECTIVE A, stated as an assertion rather than prose.

    t=0.70 satisfies the floor (ppv 2/3 >= 0.60) at sensitivity 1.0, against the
    selected point's 0.5. The legacy rule cannot reach it because the break fires
    at t=0.80. This test asserts BOTH halves: that the better candidate is
    genuinely feasible, and that the selector does not return it.

    Without this, "Objective B is worse here" would be a claim in a docstring.
    """
    selected = _evaluator()._find_high_ppv_point(
        NON_MONOTONE_Y, NON_MONOTONE_P, min_ppv=PPV_FLOOR)

    # The candidate the legacy rule skips, computed directly from the fixture.
    predictions = (NON_MONOTONE_P >= 0.7).astype(int)
    true_positive = int(((predictions == 1) & (NON_MONOTONE_Y == 1)).sum())
    false_positive = int(((predictions == 1) & (NON_MONOTONE_Y == 0)).sum())
    false_negative = int(((predictions == 0) & (NON_MONOTONE_Y == 1)).sum())
    flagged = true_positive + false_positive
    positives = true_positive + false_negative

    skipped_ppv = true_positive / flagged
    skipped_sensitivity = true_positive / positives

    assert skipped_ppv >= PPV_FLOOR, (
        "the fixture no longer contains a feasible skipped candidate; the "
        "non-monotone pattern this module depends on has been lost")
    assert skipped_sensitivity > selected.sensitivity, (
        "the skipped candidate is no longer better than the selected one")

    assert selected.threshold != pytest.approx(0.7), (
        "the legacy selector reached the higher-sensitivity feasible candidate. "
        "If that is intended, the policy has changed from Objective B to "
        "Objective A and this module must be updated deliberately -- not by "
        "relaxing the assertion")


def test_high_ppv_selector_names_the_predicted_positive_count_n_flagged():
    """A STRUCTURAL GUARD. No behavioural test can catch this.

    Before OP-0, `_find_high_ppv_point` read `n_neg = tp + fp  # n_flagged`.
    The arithmetic was correct -- every use read it as the flagged count -- and
    the NAME said the opposite, while the sibling `_find_operating_point` uses
    `n_neg` for the genuine fp + tn. One identifier, two quantities, two adjacent
    functions.

    That is a defect one edit away from becoming numerical, and no assertion on
    outputs would ever have detected it. This is the M06 lesson from REG-1
    (2026-08-03) applied to a defect of the same kind: behavioural tests prove
    outputs; structural tests prove that the code says what it means.

    The check is on the SEMANTIC ASSIGNMENT, not on whitespace or token order,
    so ordinary edits to the function do not disturb it.
    """
    source = textwrap.dedent(
        inspect.getsource(ClinicalEvaluator._find_high_ppv_point))
    tree = ast.parse(source)

    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "n_neg"
                   for target in node.targets):
            continue
        value = node.value
        if (isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.Name)
                and isinstance(value.right, ast.Name)
                and {value.left.id, value.right.id} == {"tp", "fp"}):
            offending.append(node.lineno)

    assert not offending, (
        f"line(s) {offending} of _find_high_ppv_point store tp + fp under the "
        "name n_neg. That sum is the FLAGGED / predicted-positive count; n_neg "
        "is the reference-negative count in the sibling selector. `n_flagged` "
        "is the established public vocabulary -- it is a field on OperatingPoint")


def test_the_sibling_selector_still_uses_n_neg_for_the_genuine_negatives():
    """THE OTHER HALF, so the guard above cannot be satisfied by over-renaming.

    In `_find_operating_point`, `n_neg = fp + tn` is CORRECT: there the name and
    the quantity agree. A future reader applying the rule above too broadly would
    rename it and lose that agreement, so this asserts it stays.
    """
    source = textwrap.dedent(
        inspect.getsource(ClinicalEvaluator._find_operating_point))
    tree = ast.parse(source)

    genuine = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "n_neg"
                   for target in node.targets):
            continue
        value = node.value
        if (isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.Name)
                and isinstance(value.right, ast.Name)
                and {value.left.id, value.right.id} == {"fp", "tn"}):
            genuine.append(node.lineno)

    assert len(genuine) == 1, (
        "_find_operating_point no longer assigns fp + tn to n_neg exactly once. "
        "There the name is CORRECT and must not be renamed: the OP-0 rename "
        "applied to the high-PPV selector only")
