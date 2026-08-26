"""A feature-drift check that did not run may not be reported as finding nothing.

CONTINUAL-FEATURE-DRIFT-FAILURE-AS-NO-DRIFT-1. Created 2026-08-25.

WHAT THIS GUARDS
----------------
`DriftDetector.check` REFUSES rather than degrading. It raises KeyError when the
new data lacks features the reference expects -- "Refusing to report partial
coverage as a completed drift check" -- and ValueError when a bare array cannot
be aligned by name.

Until 2026-08-25 `ContinualLearner.run` caught those DELIBERATE REFUSALS with a
bare `except Exception`, logged a warning, set `drift_report = None`, and wrote
"No significant drift detected." into `decision_<release>.json` -- a durable
scientific record. THE ASSESSMENT LAYER'S REFUSAL WAS INVERTED INTO THE EXACT
CLAIM IT REFUSED TO MAKE.

MEASURED, it was not hypothetical: drift_detector.py records a Run-15 reference
of 78 features against a tabular contract of 95, so the KeyError is the EXPECTED
path.

THESE ARE THE MODULE'S FIRST TESTS
----------------------------------
CONTINUAL-TRAINER-UNTESTED-1, confirmed 2026-08-25 by enumeration: of three test
files that mention `continual_trainer`, one cites it in a docstring, one lists
it as a FORBIDDEN import in a layering assertion, and one states in prose that
it "has no test coverage". Three independent documents record the same gap.

THE DECISION IS EXECUTED, NOT TRANSCRIBED
-----------------------------------------
An earlier draft of this file RE-IMPLEMENTED the decision expression and pinned
the copy to the source by asserting fragments appeared in both. A sabotage
matrix showed the weakness exactly: deleting the not-checked branch from the
module left every behavioural test GREEN, because none of them ran module code.
Only the pin fired.

A guard that cannot observe the thing it guards is the shape this repository
keeps finding, and a transcription is that shape in test form. So the decision
was hoisted into `render_retraining_decision`, a pure module-level function, and
these tests CALL IT.

Importing the module costs numpy and pandas -- both already project
dependencies imported throughout this suite, so the cost is a lookup rather
than a load.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.training.continual_trainer import (
    render_retraining_decision,
)

TRAINER = (Path(__file__).resolve().parents[2] / "src"
           / "genomic_variant_classifier" / "training" / "continual_trainer.py")

#: The string that may be emitted ONLY when both assessments ran and both found
#: nothing.
HONEST_NEGATIVE = "No significant drift detected."


def decide(*, feature_drift_checked, not_checked_reason,
           feature_drift_triggered, label_drift_triggered):
    """Calls the REAL function. Not a copy of it.

    A previous draft re-implemented this expression; mutating the module could
    not then change any behavioural test. This adapter only supplies the two
    detail strings, which `run` formats from live report objects.
    """
    return render_retraining_decision(
        feature_drift_checked=feature_drift_checked,
        feature_drift_not_checked_reason=not_checked_reason,
        feature_drift_triggered=feature_drift_triggered,
        label_drift_triggered=label_drift_triggered,
        feature_drift_detail="Feature drift: 4 features with PSI>0.2",
        label_drift_detail="Label drift: flip_rate=1.100%, "
                           "weighted_impact=0.900%",
    )


# ---------------------------------------------------------------------------
# 1. THE FALSE NEGATIVE IS GONE
# ---------------------------------------------------------------------------

def test_a_refused_assessment_is_not_reported_as_finding_nothing():
    """The defect, stated as a test.

    A KeyError from `check` means the comparison was REFUSED. Before this
    repair the same inputs produced "No significant drift detected."
    """
    d = decide(feature_drift_checked=False,
               not_checked_reason="KeyError: The new data is missing 17 "
                                  "feature(s) the reference expects",
               feature_drift_triggered=False, label_drift_triggered=False)
    assert d["reason"] != HONEST_NEGATIVE
    assert "FEATURE DRIFT NOT CHECKED" in d["reason"]
    assert "missing 17 feature(s)" in d["reason"], (
        "the reason must carry WHY, not merely that something failed")


def test_the_gap_is_reported_even_when_label_drift_fires():
    """A finding must not bury the gap.

    `should_retrain` becomes True on label drift alone, and a reader could
    reasonably assume both assessments contributed. The not-checked entry is
    appended FIRST for exactly this case.
    """
    d = decide(feature_drift_checked=False, not_checked_reason="KeyError: ...",
               feature_drift_triggered=False, label_drift_triggered=True)
    assert d["should_retrain"] is True
    assert d["reason"].startswith("FEATURE DRIFT NOT CHECKED")
    assert "Label drift" in d["reason"]


def test_the_honest_negative_survives():
    """The repair removes the FALSE negative, not the true one."""
    d = decide(feature_drift_checked=True, not_checked_reason=None,
               feature_drift_triggered=False, label_drift_triggered=False)
    assert d["reason"] == HONEST_NEGATIVE
    assert d["should_retrain"] is False


def test_a_real_finding_is_unaffected():
    d = decide(feature_drift_checked=True, not_checked_reason=None,
               feature_drift_triggered=True, label_drift_triggered=False)
    assert d["should_retrain"] is True
    assert d["reason"].startswith("Feature drift:")
    assert "NOT CHECKED" not in d["reason"]


@pytest.mark.parametrize(
    "checked,ftrig,ltrig",
    [(False, False, False), (False, False, True),
     (False, True, False), (False, True, True)],
    ids=["unchecked-quiet", "unchecked-label", "unchecked-feature",
         "unchecked-both"])
def test_the_honest_negative_is_unreachable_when_unchecked(checked, ftrig, ltrig):
    """Across EVERY combination, not one representative case."""
    d = decide(feature_drift_checked=checked, not_checked_reason="X: y",
               feature_drift_triggered=ftrig, label_drift_triggered=ltrig)
    assert d["reason"] != HONEST_NEGATIVE


# ---------------------------------------------------------------------------
# 2. THE PERSISTED ARTIFACT CARRIES THE DISTINCTION
# ---------------------------------------------------------------------------

def test_the_decision_records_whether_the_check_RAN():
    """`feature_drift` alone cannot express it.

    False means "ran, found nothing" AND "did not run" without this flag --
    the exact overload `DriftReport.joint_tests_run` exists to prevent one
    layer down.
    """
    d = decide(feature_drift_checked=False, not_checked_reason="KeyError: ...",
               feature_drift_triggered=False, label_drift_triggered=False)
    assert d["feature_drift_checked"] is False
    assert d["feature_drift"] is False
    assert d["feature_drift_not_checked_reason"]


def test_the_decision_survives_json_serialisation():
    """`run` writes decision_<release>.json. The distinction must reach disk;
    it is the durable record, not the log line."""
    d = decide(feature_drift_checked=False, not_checked_reason="KeyError: ...",
               feature_drift_triggered=False, label_drift_triggered=True)
    back = json.loads(json.dumps(d, default=str))
    assert back["feature_drift_checked"] is False
    assert back["feature_drift_not_checked_reason"]
    assert back["reason"].startswith("FEATURE DRIFT NOT CHECKED")


def test_a_successful_check_leaves_the_reason_null():
    d = decide(feature_drift_checked=True, not_checked_reason=None,
               feature_drift_triggered=False, label_drift_triggered=False)
    assert d["feature_drift_checked"] is True
    assert d["feature_drift_not_checked_reason"] is None


# ---------------------------------------------------------------------------
# 3. THE MODULE ITSELF, BY PARSING
# ---------------------------------------------------------------------------

def test_run_delegates_to_the_pure_function():
    """`run` must CALL render_retraining_decision, not rebuild the dict.

    Two implementations of one decision would drift, and the tests would follow
    only one of them. Parsed, so a docstring naming the function is not counted
    as a call.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    calls = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "render_retraining_decision"]
    assert calls, (
        "ContinualLearner.run does not call render_retraining_decision. If the "
        "decision were rebuilt inline, every behavioural test in this file "
        "would pass against a module that no longer contains the logic.")


def test_no_second_implementation_of_the_reason_string_exists():
    """The honest negative may be produced in exactly ONE place."""
    text = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    literals = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value == HONEST_NEGATIVE]
    assert len(literals) == 1, (
        "{!r} is produced at {} sites. A second implementation would drift "
        "from the one these tests exercise.".format(HONEST_NEGATIVE, literals))


def test_the_pipeline_import_is_not_inside_the_try():
    """An ImportError is a deployment fault, not a drift result.

    Until 2026-08-25 the import sat inside the `try`, so a missing module was
    caught and rendered as a scientific negative.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.ImportFrom) and sub.module \
                        and "api.pipeline" in sub.module:
                    offenders.append(sub.lineno)
    assert not offenders, (
        "the pipeline import is inside a try at line(s) {}. An ImportError "
        "would be swallowed and reported as a drift result.".format(offenders))


def test_the_swallowing_handler_records_the_gap():
    """The handler must set BOTH new variables, not merely log.

    A log line is not a record: the decision is what reaches disk.
    """
    text = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "the module has no except handler at all"
    assigning = []
    for node in handlers:
        names = {t.id for st in ast.walk(node) if isinstance(st, ast.Assign)
                 for t in st.targets if isinstance(t, ast.Name)}
        if "drift_report" in names:
            assigning.append(names)
    assert assigning, "no handler assigns drift_report"
    for names in assigning:
        assert "feature_drift_checked" in names, names
        assert "feature_drift_not_checked_reason" in names, names


def test_the_failure_is_logged_at_error_level():
    """A warning nobody surfaces is not a record of a failed measurement."""
    text = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        names = {t.id for st in ast.walk(node) if isinstance(st, ast.Assign)
                 for t in st.targets if isinstance(t, ast.Name)}
        if "drift_report" not in names:
            continue
        body = "\n".join(text.split("\n")[node.lineno - 1:node.end_lineno])
        assert "logger.error" in body, (
            "the swallowed assessment is logged below error level")
        assert "logger.warning" not in body


def test_the_module_does_not_quote_a_stale_feature_contract():
    """DETECTOR-CONTRACT-COMMENT-STALE-1, guarded here.

    MEASURED 2026-08-25: EXPECTED_TABULAR_FEATURE_COUNT is 95, defined once in
    models/variant_ensemble.py:193, where TABULAR_FEATURES holds exactly 95
    entries. The figure 97 appeared in drift_detector.py and was ITSELF
    corrected once by a preflight gate -- 97 (88+3+6) became 95 (86+3+6).

    THE GUARD DISTINGUISHES A CLAIM FROM A RECORD.
    STALE-NUMBER-GUARD-CANNOT-SEE-HISTORY-1, found 2026-08-25 while writing the
    detector's repair: an earlier version of this test rejected ANY line
    containing "97" and "feature". That rule would have forbidden the very
    sentence recording that the number was corrected -- and METHODS M1
    established the opposite principle on 2026-08-24: "a document that erases
    its own former claims cannot be audited."

    So a superseded figure may APPEAR, and may not be ASSERTED. A line citing
    it must mark itself as history.
    """
    text = TRAINER.read_text(encoding="utf-8")
    history = ("until", "was itself", "said ", "corrected", "STALE",
               "formerly", "no longer", "superseded")
    offenders = []
    for i, line in enumerate(text.split("\n"), 1):
        if "97" not in line:
            continue
        if "contract" not in line.lower() and "feature" not in line.lower():
            continue
        if any(marker in line for marker in history):
            continue
        offenders.append((i, line.strip()))
    assert not offenders, (
        "these lines cite a 97-feature contract without marking it as "
        "history: {}. The contract is 95.".format(offenders))


def test_that_guard_accepts_a_HISTORICAL_mention():
    """Guards the guard, in the direction that matters.

    A stale-number check that cannot be satisfied by an honest correction
    would push the next author to DELETE the record instead of marking it.
    Proven on a constructed line rather than assumed.
    """
    history = ("until", "was itself", "said ", "corrected", "STALE",
               "formerly", "no longer", "superseded")
    honest = "# this comment said 97 until 2026-08-25, corrected to 95"
    assert any(m in honest for m in history), (
        "an honest historical correction would be rejected by this guard")
    dishonest = "# the current tabular contract is 97 features"
    assert not any(m in dishonest for m in history), (
        "a current claim would slip past this guard")
