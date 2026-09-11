"""Operation kind: shape, canonical paths, and the terminating obligation.

Author: Monzia Moodie

The four transition cases and two path forms below were each REPRODUCED as
acceptances before the corrections that refuse them.
"""
from __future__ import annotations

import copy

import pytest

from genomic_variant_classifier.operations.operation_kind import (
    ARTIFACTS_SUBTREE, MANIFEST_PATH, ClassificationError, FileEffect,
    Obligation, OperationKind, classify_admission_candidate,
    classify_operation, normalize_effects, obligations_for,
    require_archive_only_targets, require_canonical_repository_path,
    require_complete_approved_additions, require_exact_effects,
    require_preserved_predecessor)


def _artifact(i):
    return ARTIFACTS_SUBTREE + "install-attestation-{:03d}.json".format(i)


@pytest.fixture
def approved():
    targets = {MANIFEST_PATH: {"action": "patch",
                               "content_sha256": "a" * 64,
                               "size_bytes": 85213}}
    for i in range(5):
        targets[_artifact(i)] = {"action": "create",
                                 "content_sha256": "{:064x}".format(i),
                                 "size_bytes": 100 + i}
    return targets


PREDECESSOR = frozenset({MANIFEST_PATH, "src/module.py"})


def _classify(approved, observed):
    return classify_admission_candidate(
        approved_targets=approved, observed_transition=observed,
        predecessor_paths=PREDECESSOR, approved_addition_count=5)


def test_an_archive_admission_owes_only_maintenance_evidence():
    obligations = obligations_for(OperationKind.ARCHIVE_ADMISSION)
    assert obligations == frozenset({Obligation.MAINTENANCE_EVIDENCE})
    assert Obligation.INSTALLATION_ARCHIVE not in obligations


def test_an_installation_owes_an_installation_archive_admission():
    assert obligations_for(OperationKind.INSTALLATION) == frozenset(
        {Obligation.INSTALLATION_ARCHIVE})


def test_the_approved_transition_is_identified(approved):
    kind, obligations = _classify(approved, copy.deepcopy(approved))
    assert kind is OperationKind.ARCHIVE_ADMISSION
    assert obligations == frozenset({Obligation.MAINTENANCE_EVIDENCE})


@pytest.mark.parametrize("field,value", [
    ("content_sha256", "f" * 64),
    ("size_bytes", 999999),
])
def test_an_observed_payload_that_differs_is_refused(approved, field, value):
    """REPRODUCED as an acceptance: comparing only dictionary KEYS was blind
    to every observed value."""
    observed = copy.deepcopy(approved)
    observed[_artifact(0)][field] = value
    with pytest.raises(ClassificationError):
        _classify(approved, observed)


def test_a_symbolic_link_mode_is_refused(approved):
    observed = copy.deepcopy(approved)
    observed[_artifact(0)]["mode"] = "120000"
    with pytest.raises(ClassificationError):
        _classify(approved, observed)


def test_an_unsupported_observed_manifest_action_is_refused(approved):
    observed = copy.deepcopy(approved)
    observed[MANIFEST_PATH] = {"action": "anything",
                               "content_sha256": "a" * 64, "size_bytes": 1}
    with pytest.raises(ClassificationError):
        _classify(approved, observed)


@pytest.mark.parametrize("path", [
    ARTIFACTS_SUBTREE + "..\\..\\..\\escape.json",
    ARTIFACTS_SUBTREE + "a.json:stream",
    ARTIFACTS_SUBTREE + "CON.json",
    ARTIFACTS_SUBTREE + "a.json ",
    ARTIFACTS_SUBTREE + "a.json.",
    ARTIFACTS_SUBTREE + "./x.json",
    ARTIFACTS_SUBTREE + "sub//x.json",
    "/absolute/x.json",
    "C:/drive/x.json",
    "",
])
def test_a_noncanonical_path_is_refused(path):
    """The first two were REPRODUCED as acceptances: the code split only on
    '/' while Windows also recognises '\\', and nothing examined the
    basename."""
    with pytest.raises(ClassificationError):
        require_canonical_repository_path(path)


def test_destinations_colliding_case_insensitively_are_refused():
    with pytest.raises(ClassificationError):
        normalize_effects({
            _artifact(0): {"action": "create", "content_sha256": "0" * 64,
                           "size_bytes": 1},
            _artifact(0).upper(): {"action": "create",
                                   "content_sha256": "0" * 64,
                                   "size_bytes": 1}})


@pytest.mark.parametrize("spec", [
    {"action": "delete", "content_sha256": "0" * 64, "size_bytes": 1},
    {"action": "create", "content_sha256": "0" * 64, "size_bytes": True},
    {"action": "create", "content_sha256": "0" * 64, "size_bytes": -1},
    {"action": "create", "content_sha256": "A" * 64, "size_bytes": 1},
    {"action": "create", "content_sha256": "abc", "size_bytes": 1},
    {"action": "create", "mode": "120000", "content_sha256": "0" * 64,
     "size_bytes": 1},
])
def test_an_invalid_effect_is_refused_at_construction(spec):
    with pytest.raises(ClassificationError):
        FileEffect(action=spec.get("action"), mode=spec.get("mode", "100644"),
                   size_bytes=spec.get("size_bytes"),
                   content_sha256=spec.get("content_sha256"))


def test_a_target_outside_the_archive_subtree_is_refused(approved):
    targets = dict(approved)
    targets["src/genomic_variant_classifier/state/json_state_store.py"] = {
        "action": "patch", "content_sha256": "0" * 64, "size_bytes": 1}
    with pytest.raises(ClassificationError):
        require_archive_only_targets(targets)


def test_the_manifest_does_not_escape_payload_validation():
    with pytest.raises(ClassificationError):
        require_archive_only_targets({MANIFEST_PATH: {"action": "patch"}})


def test_an_admission_must_patch_the_manifest(approved):
    targets = {k: v for k, v in approved.items() if k != MANIFEST_PATH}
    with pytest.raises(ClassificationError):
        require_archive_only_targets(targets)


def test_an_undeclared_path_and_an_absent_one_are_both_refused(approved):
    normalized = normalize_effects(approved)
    extra = dict(approved)
    extra[_artifact(99)] = {"action": "create", "content_sha256": "0" * 64,
                            "size_bytes": 1}
    with pytest.raises(ClassificationError):
        require_exact_effects(normalized, normalize_effects(extra))
    fewer = {k: v for k, v in approved.items() if k != _artifact(0)}
    with pytest.raises(ClassificationError):
        require_exact_effects(normalized, normalize_effects(fewer))


def test_a_delete_is_refused(approved):
    observed = dict(approved)
    observed[_artifact(1)] = {"action": "delete"}
    with pytest.raises(ClassificationError):
        require_preserved_predecessor(PREDECESSOR, observed)


@pytest.mark.parametrize("count", [4, 6, True])
def test_a_wrong_addition_count_is_refused(approved, count):
    with pytest.raises(ClassificationError):
        require_complete_approved_additions(count, approved)


def test_the_renamed_function_refuses_rather_than_changing_meaning():
    """`classify_operation` implied it established the operation KIND. It
    establishes the transition's SHAPE."""
    with pytest.raises(ClassificationError):
        classify_operation(approved_targets={}, observed_transition={})
