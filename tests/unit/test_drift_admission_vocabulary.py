"""The admission layer's vocabulary exists, and this layer emits none of it.

DRIFT-1 Phase 1D, vocabulary. Created 2026-08-27.

WHAT THIS GUARDS
----------------
`drift_readiness.py` names five layers and says no layer may author a fact
owned by a downstream one:

    Repository capability code   whether capability exists.
    Discovery authority          whether an observation exists.
    Admission code               whether populations are comparable.
    Assessment code              whether drift exists.
    Policy code                  whether action is required.

The module owns only the FIRST. It already carries
`NO_NEW_OBSERVATION_POPULATION` for the second -- present in the vocabulary,
emitted by nothing, with a test proving it -- and gives the reason:

    "Its absence from the vocabulary would force a future migration; its
     emission here would be a claim no layer at this level owns."

The four ADMISSION reasons follow that precedent exactly. This file proves each
one is unemitted, so a later commit cannot quietly begin emitting an
admission-layer verdict from a capability-layer function.

WHY THESE FOUR, AND WHY NOW
---------------------------
MEASURED 2026-08-27 from the artifact itself:
`data/reference/drift/run15_reference_profile.json` is 1,089,400 bytes,
format_version 1, 78 features over 1,038,974 rows -- and its entire provenance
is one field, `source`, holding a machine-local path.

That is true regardless of how the reference is later regenerated. A rebuilt
95-feature reference would ALSO be unattributed unless it records a
representation, a source manifest and a population fingerprint. So the
vocabulary is decision-independent, and adding it now asserts no verdict.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from genomic_variant_classifier.monitoring.drift_readiness import (
    DriftReadinessReason,
    DriftReadinessStatus,
    current_feature_drift_readiness,
)

MODULE = (Path(__file__).resolve().parents[2] / "src"
          / "genomic_variant_classifier" / "monitoring" / "drift_readiness.py")

#: The four reasons only admission code may emit, BY NAME.
#
# Strings rather than attribute access, and that is a measured decision. An
# earlier version referenced the members directly at module level -- so
# renaming or deleting one raised AttributeError during COLLECTION, every test
# in this file errored at once, and `test_the_vocabulary_did_not_lose_a_member`
# could never fire to say which member had gone.
#
# A sabotage matrix found it: two mutations were detected, and neither by the
# test written to detect them. Loud is not the same as informative.
ADMISSION_REASON_NAMES = (
    "REFERENCE_REPRESENTATION_UNIDENTIFIED",
    "REPRESENTATION_MISMATCH",
    "SOURCE_RELEASE_DIVERGENT",
    "POPULATION_UNATTRIBUTED",
)

#: Every member this vocabulary must carry, across all layers.
EXPECTED_MEMBER_NAMES = (
    "CANDIDATE_DISCOVERY_NOT_IMPLEMENTED",
    "NO_NEW_OBSERVATION_POPULATION",
) + ADMISSION_REASON_NAMES


def _reason(name):
    """Resolve a member by name, FAILING with a message rather than erroring."""
    member = getattr(DriftReadinessReason, name, None)
    assert member is not None, (
        "DriftReadinessReason has no member {!r}. It carries {}. A member that "
        "was renamed rather than added retires an identity underneath the "
        "count.".format(name, sorted(m.name for m in DriftReadinessReason)))
    return member


@pytest.mark.parametrize("name", ADMISSION_REASON_NAMES,
                         ids=list(ADMISSION_REASON_NAMES))
def test_each_admission_reason_is_in_the_vocabulary(name):
    """Present so a later migration is not forced."""
    assert _reason(name).value in {m.value for m in DriftReadinessReason}


@pytest.mark.parametrize("name", ADMISSION_REASON_NAMES,
                         ids=list(ADMISSION_REASON_NAMES))
def test_this_layer_emits_no_admission_reason(name):
    """The invariant. A capability-layer function may not claim comparability.

    Mirrors `test_no_new_observation_population_is_in_the_vocabulary_but_
    unemitted`, which guards the discovery layer the same way.
    """
    assert current_feature_drift_readiness().reason is not _reason(name)


def test_no_admission_reason_appears_in_the_module_as_a_RETURNED_value():
    """Parsed, not grepped: a name in a docstring is not an emission.

    The members are DEFINED here, so a text search would match their own
    declarations. This walks every `return` and every `DriftReadiness(...)`
    call and refuses an admission reason appearing as a value in either.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = set(ADMISSION_REASON_NAMES)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            text = ast.unparse(node.value)
            for name in names:
                if "DriftReadinessReason.{}".format(name) in text:
                    offenders.append((node.lineno, name))
        elif isinstance(node, ast.Call):
            fn = node.func
            if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != "DriftReadiness":
                continue
            text = ast.unparse(node)
            for name in names:
                if "DriftReadinessReason.{}".format(name) in text:
                    offenders.append((node.lineno, name))
    assert not offenders, (
        "drift_readiness.py emits admission-layer reason(s) {}. Only admission "
        "code may state whether populations are comparable.".format(offenders))


def test_the_capability_reason_is_still_what_this_layer_emits():
    """The permissive direction: adding vocabulary must not change the verdict."""
    readiness = current_feature_drift_readiness()
    assert readiness.status is DriftReadinessStatus.UNDETERMINED
    assert readiness.reason is (
        DriftReadinessReason.CANDIDATE_DISCOVERY_NOT_IMPLEMENTED)
    assert readiness.checked is False


def test_every_reason_value_is_lower_snake_case_and_unique():
    """A vocabulary whose spellings drift is a vocabulary nobody can grep.

    `UNKNOWN` is excluded from every domain enumeration by an existing test;
    this pins the positive form so a new member cannot arrive shouting.
    """
    values = [m.value for m in DriftReadinessReason]
    assert len(set(values)) == len(values), values
    for v in values:
        assert v == v.lower(), v
        assert " " not in v and "-" not in v, v


def test_the_admission_reasons_are_distinct_from_the_other_layers():
    """Four layers, four vocabularies, no member serving two."""
    admission = {_reason(n).value for n in ADMISSION_REASON_NAMES}
    capability = {DriftReadinessReason.CANDIDATE_DISCOVERY_NOT_IMPLEMENTED.value}
    discovery = {DriftReadinessReason.NO_NEW_OBSERVATION_POPULATION.value}
    assert not admission & capability
    assert not admission & discovery
    assert len(admission) == 4


def test_the_vocabulary_did_not_lose_a_member():
    """Guards against a member being renamed rather than added.

    An earlier unit in this programme declared an ADDITION while a rename had
    retired an identity underneath the count. Naming the expected set makes
    that shape impossible to miss.
    """
    assert {m.name for m in DriftReadinessReason} == set(EXPECTED_MEMBER_NAMES)
