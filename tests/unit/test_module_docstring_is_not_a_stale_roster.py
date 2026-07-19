"""The module header of variant_ensemble.py must not restate the roster.

Until 2026-07-19 that docstring read "Implements 8 base classifiers", enumerated eight by
name, and attributed cnn_1d and the feedforward network to TensorFlow/Keras -- which the
module does not import. The roster was thirteen base models plus a stacking meta-learner and
two graph branches. The header was the first text a reader saw, and it was wrong on the
count, wrong on the framework, and headed by a section describing a phase long past.

The correction was to DELETE the enumeration rather than update it. A corrected copy goes
stale again on the next model; a pointer to `_build_estimators` cannot.

This file exists so the enumeration cannot come back quietly. Same purpose as the suite-size
ratchet and the README badge test: a claim that can drift is turned into a claim that fails
loudly when it does.

EVERY CHECK HERE CARRIES A NEGATIVE CONTROL. The old docstring is embedded verbatim below and
asserted to FAIL each check. A checker that has never rejected anything has not been shown to
work -- and three checkers written earlier in the same session passed their own controls while
counting prose as code.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import genomic_variant_classifier.models.variant_ensemble as VE

# The header as it stood before 2026-07-19, verbatim. Used ONLY as a negative control.
_OLD_HEADER = """
Ensemble Model Framework for Genomic Variant Classification
============================================================
Implements 8 base classifiers + 1 stacking meta-learner.

Base classifiers:
  1. Random Forest         (sklearn)
  2. XGBoost               (xgboost)
  3. LightGBM              (lightgbm)
  4. SVM (RBF kernel)      (sklearn)
  5. Logistic Regression   (sklearn)
  6. Gradient Boosting     (sklearn)
  7. 1D-CNN                (TensorFlow/Keras)  -- sequence-based
  8. Feedforward NN        (TensorFlow/Keras)  -- tabular features

Meta-learner:
  Logistic Regression stacker trained on OOF predictions

CHANGES FROM PHASE 1:
  - Consolidated src/models/ensemble.py + src/models/variant_ensemble.py
    into this single file (Issue A).
"""

_FRAMEWORKS = ("tensorflow", "keras", "pytorch", "torch")

# "Implements 8 base classifiers", "13 base models", "eight permanent base models".
#
# The number must MODIFY the noun -- at most two words may intervene. An earlier version
# allowed 40 characters of anything, and matched the "19" of the date "2026-07-19" against
# a "base classifiers" later in the same sentence. A date is not a count claim, and a check
# that fires on prose describing the rule is the failure this file exists to prevent (it has
# now happened six times in one session).
_COUNT_CLAIM = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)\b"
    r"(?:\s+\w+){0,2}\s+base\s+(classifier|model|estimator)s?\b",
    re.IGNORECASE,
)

# A numbered enumeration line: "  1. Random Forest"
_ENUMERATION = re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE)


def _module_docstring() -> str:
    doc = ast.get_docstring(ast.parse(Path(VE.__file__).read_text(encoding="utf-8")))
    assert doc, "variant_ensemble.py has no module docstring at all"
    return doc


def _claims_a_count(text: str) -> bool:
    return _COUNT_CLAIM.search(text) is not None


def _enumerates(text: str) -> bool:
    return _ENUMERATION.search(text) is not None


def _attributes_a_framework(text: str) -> bool:
    low = text.lower()
    return any(f in low for f in _FRAMEWORKS)


def test_the_header_does_not_claim_a_model_count():
    assert not _claims_a_count(_module_docstring()), (
        "the module docstring states a base-model count. That number is a copy of a fact "
        "defined in _build_estimators and will go stale the next time a model is added -- "
        "which is exactly how it came to say 8 while the roster held 13."
    )


def test_the_count_check_rejects_the_old_header():
    """NEGATIVE CONTROL. The check must catch the defect it was written for."""
    assert _claims_a_count(_OLD_HEADER), (
        "the count check does not flag the pre-2026-07-19 header, so it would not have "
        "caught the defect it exists to prevent"
    )


def test_the_header_does_not_enumerate_models():
    assert not _enumerates(_module_docstring()), (
        "the module docstring contains a numbered enumeration. The roster belongs in "
        "_build_estimators and nowhere else; a second copy cannot be kept in step."
    )


def test_the_enumeration_check_rejects_the_old_header():
    """NEGATIVE CONTROL."""
    assert _enumerates(_OLD_HEADER)


def test_the_header_does_not_attribute_a_framework_to_any_model():
    assert not _attributes_a_framework(_module_docstring()), (
        "the module docstring names a machine-learning framework. It said TensorFlow/Keras "
        "for two models that are not built with it; framework choice is visible in the "
        "import block and in _build_estimators, which cannot disagree with themselves."
    )


def test_the_framework_check_rejects_the_old_header():
    """NEGATIVE CONTROL."""
    assert _attributes_a_framework(_OLD_HEADER)


def test_the_header_points_at_where_the_roster_actually_lives():
    """Removing the copy is only half the fix; the reader still needs the original."""
    doc = _module_docstring()
    for anchor in ("_build_estimators", "base_estimators", "SEQUENCE_MODELS"):
        assert anchor in doc, (
            f"the module docstring no longer enumerates the roster but does not say where "
            f"it lives either; {anchor!r} is missing, leaving the reader with nothing."
        )
