"""The threshold vocabulary moved down, and it moved by IDENTITY.

THR-1, 2026-08-04. A zero-movement extraction: the three classes now live in
`thresholds.py` and are re-exported from `registry.py`.

These tests exist because a re-export that produced a DISTINCT class object would
break `ThresholdParameters`'s stated invariant -- "one instance is shared by a
descriptor, its kernel adapter and its applicability predicate, asserted BY
IDENTITY at import time" -- and the failure would be silent until something
asserted identity.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest


def test_registry_reexports_the_canonical_threshold_types_by_identity():
    """`is`, not `==`. Two enum classes with identical members are not
    interchangeable: `isinstance(x, A)` is False for an instance of B."""
    from genomic_variant_classifier.evaluation import registry
    from genomic_variant_classifier.evaluation import thresholds

    assert registry.ThresholdOperator is thresholds.ThresholdOperator
    assert registry.ThresholdSource is thresholds.ThresholdSource
    assert registry.ThresholdParameters is thresholds.ThresholdParameters


def test_the_shared_instance_invariant_still_holds_through_the_re_export():
    """A ThresholdParameters built through either name must satisfy the other's
    isinstance check -- which is exactly what a distinct class object would
    break."""
    from genomic_variant_classifier.evaluation import registry
    from genomic_variant_classifier.evaluation import thresholds

    built_from_registry = registry.ThresholdParameters(
        threshold=0.5,
        operator=registry.ThresholdOperator.GREATER_OR_EQUAL,
        source=registry.ThresholdSource.FIXED_DEFAULT)

    assert isinstance(built_from_registry, thresholds.ThresholdParameters)
    assert isinstance(built_from_registry.operator,
                      thresholds.ThresholdOperator)
    assert isinstance(built_from_registry.source, thresholds.ThresholdSource)


def test_the_serialisation_is_unchanged():
    """THR-1 moves text. If a serialised mapping differs, it moved more."""
    from genomic_variant_classifier.evaluation import thresholds

    parameters = thresholds.ThresholdParameters(
        threshold=0.5,
        operator=thresholds.ThresholdOperator.GREATER_OR_EQUAL,
        source=thresholds.ThresholdSource.FIXED_DEFAULT)

    assert parameters.to_mapping() == {
        "decision_threshold": 0.5,
        "threshold_operator": ">=",
        "threshold_source": "fixed_default"}


def test_the_validation_moved_with_the_class():
    """Every refusal the class made before, it must still make."""
    from genomic_variant_classifier.evaluation import thresholds

    operator = thresholds.ThresholdOperator.GREATER_OR_EQUAL
    source = thresholds.ThresholdSource.FIXED_DEFAULT

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        thresholds.ThresholdParameters(threshold=1.5, operator=operator,
                                       source=source)
    with pytest.raises(TypeError, match="numeric"):
        thresholds.ThresholdParameters(threshold="0.5", operator=operator,
                                       source=source)
    with pytest.raises(TypeError, match="ThresholdOperator member"):
        thresholds.ThresholdParameters(threshold=0.5, operator=">=",
                                       source=source)
    with pytest.raises(TypeError, match="ThresholdSource member"):
        thresholds.ThresholdParameters(threshold=0.5, operator=operator,
                                       source="fixed_default")


def test_thresholds_imports_nothing_it_must_not():
    """THE CONSTRAINT THAT IS THE MODULE'S REASON FOR EXISTING.

    It sits beneath `registry.py` and `metrics.py`, so it may import neither.
    And it must import cleanly with scikit-learn absent, which is what lets it
    be depended upon from the package root -- the boundary
    `evaluation/__init__.py` documents at length after commit 015ff94 broke it.

    Checked STRUCTURALLY, over the source: a runtime check would pass merely
    because scikit-learn happens to be installed.
    """
    import ast
    import inspect

    from genomic_variant_classifier.evaluation import thresholds

    tree = ast.parse(inspect.getsource(thresholds))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
            if node.level:
                imported.add(f"relative:{node.module}")

    for forbidden in ("sklearn", "relative:registry", "relative:metrics"):
        assert forbidden not in imported, (
            f"thresholds.py imports {forbidden}, which inverts the layering it "
            "exists to establish")
def test_thresholds_owns_the_vocabulary_it_defines():
    """OWNERSHIP, not merely identity. `__module__` changed, deliberately.

    THR-1a moved these three classes out of `registry.py` and re-exported them
    back. The re-export preserves IDENTITY -- `registry.ThresholdOperator is
    thresholds.ThresholdOperator` -- but it does NOT preserve the qualified
    module name, which moved from

        genomic_variant_classifier.evaluation.registry

    to

        genomic_variant_classifier.evaluation.thresholds

    That is the point of the extraction, and until this assertion existed it was
    written down nowhere. `__module__` affects `repr`, generated documentation,
    pickled output bytes and type-name provenance, so an unrecorded change to it
    is an unrecorded change to an artifact's contents.

    THE PATTERN IS THIS PROJECT'S OWN. `test_metric_result_relocation.py` pins
    exactly this after exactly this kind of move: `MetricResult` was relocated
    into `capabilities.py`, re-exported from `clustering_metrics`, and the tests
    assert BOTH where it is defined and that consumers resolve the same object.
    THR-1a wrote the second half and omitted the first; the wording below is the
    precedent's, because it is the same claim about a different layer.

    A future move back, or a re-export mistaken for a definition, now fails here
    rather than silently changing what artifacts say about themselves.
    """
    from genomic_variant_classifier.evaluation import thresholds

    for cls in (thresholds.ThresholdOperator,
                thresholds.ThresholdSource,
                thresholds.ThresholdParameters):
        assert cls.__module__ == (
            "genomic_variant_classifier.evaluation.thresholds"
        ), (f"{cls.__name__} must be DEFINED in the vocabulary layer, not "
            "re-exported into it")
# --------------------------------------------------------------------------- #
# THR-1b (2026-08-04): the vocabulary is now GATED, in both directions
# --------------------------------------------------------------------------- #

# The exact member set and the exact serialised value of each. Measured, not
# recalled: `fixed_default`, `calibrated` and `user_supplied` predate THR-1b and
# their strings are LOAD-BEARING -- `test_registry_vocabulary_completion.py:832`
# asserts `mcc_print["threshold"] == (0.5, ">=", "fixed_default")`, so the
# descriptor fingerprint embeds the serialised string in a tuple compared by
# equality. A renamed value would silently orphan every historical record that
# carries the old one.
_EXPECTED_THRESHOLD_SOURCES = {
    "FIXED_DEFAULT": "fixed_default",
    "CALIBRATED": "calibrated",
    "USER_SUPPLIED": "user_supplied",
    "EVALUATION_SWEEP": "evaluation_sweep",
}

_EXPECTED_THRESHOLD_OPERATORS = {
    "GREATER_OR_EQUAL": ">=",
    "GREATER": ">",
}


def test_the_threshold_source_vocabulary_is_exactly_this():
    """BOTH DIRECTIONS, on the pattern of `test_conformal_package_exports.py`.

    Before THR-1b, FIVE tests referenced `ThresholdSource` and NOT ONE
    enumerated its members -- they used it as a value. So a member could have
    appeared, disappeared, or changed its serialised string and nothing in 4,171
    tests would have objected.

    That is the REG-2 shape one layer down: there, a semantic correction to two
    metrics changed no test outcome anywhere, and the repair was the assertions
    that would notice it next time.

    A STEALTH ADDITION and a STEALTH REMOVAL both fail here, and so does a
    RENAMED VALUE -- which is the one that would silently orphan historical
    records, since the descriptor fingerprint embeds these strings.
    """
    from genomic_variant_classifier.evaluation import thresholds

    actual = {member.name: member.value
              for member in thresholds.ThresholdSource}

    assert actual == _EXPECTED_THRESHOLD_SOURCES, (
        "the ThresholdSource vocabulary changed.\n"
        f"  expected: {_EXPECTED_THRESHOLD_SOURCES}\n"
        f"  actual  : {actual}\n"
        "If a member was ADDED deliberately, add it here in the same commit. If "
        "a VALUE changed, stop: these strings are load-bearing -- the descriptor "
        "fingerprint embeds them, and historical artifacts carry them.")


def test_the_threshold_operator_vocabulary_is_exactly_this():
    """The sibling enum, gated identically.

    `>=` and `>` differ exactly at `probability == threshold`, and OP-1's sweep
    needs BOTH: the empty candidate -- flagging nothing -- requires a threshold
    above every score, which is unrepresentable in [0, 1] when the maximum is
    1.0. `GREATER` at the maximum expresses it. A member lost here would remove
    an operating point the data can express.
    """
    from genomic_variant_classifier.evaluation import thresholds

    actual = {member.name: member.value
              for member in thresholds.ThresholdOperator}

    assert actual == _EXPECTED_THRESHOLD_OPERATORS, (
        f"the ThresholdOperator vocabulary changed: {actual}")


def test_evaluation_sweep_is_constructible_and_serialises():
    """The new member must work end to end, not merely exist.

    A member that cannot be placed in a `ThresholdParameters` would be a
    vocabulary entry with no way to use it.
    """
    from genomic_variant_classifier.evaluation import thresholds

    parameters = thresholds.ThresholdParameters(
        threshold=0.7,
        operator=thresholds.ThresholdOperator.GREATER,
        source=thresholds.ThresholdSource.EVALUATION_SWEEP)

    assert parameters.to_mapping() == {
        "decision_threshold": 0.7,
        "threshold_operator": ">",
        "threshold_source": "evaluation_sweep"}


def test_the_pre_existing_sources_kept_their_serialised_values():
    """THE COMPATIBILITY HALF, asserted separately from the set.

    The set test would fail for ANY change. This one names the three that
    predate THR-1b, so a failure says immediately whether an EXISTING value
    moved -- which reinterprets artifacts -- or a NEW one was added, which does
    not.
    """
    from genomic_variant_classifier.evaluation import thresholds

    assert thresholds.ThresholdSource.FIXED_DEFAULT.value == "fixed_default"
    assert thresholds.ThresholdSource.CALIBRATED.value == "calibrated"
    assert thresholds.ThresholdSource.USER_SUPPLIED.value == "user_supplied"
