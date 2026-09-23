"""Tests for genomic_variant_classifier.containment.

Ported to pytest from the two verified reference packages (2026-09-22):
  GVC_containment_boundary_reference.zip -- all 34 tests
  GVC_quarantine_reference.zip           -- 15 of 17; the two exercising guarded_annotation are
                                             retired with it (checked_annotation supersedes it)
plus the exception-hierarchy tests for the one deliberate change in the merge.

The policy names below are FIXTURES. Production imports the project's one reviewed policy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.containment import (
    ContainmentError, Contract, QuarantineError, bind_model, checked_annotation,
    load_after_admission, require_contract_binding, require_matrix,
    require_model_compatibility, require_producer_enabled, require_scientific_contract,
)

Q = ("alphafold_plddt", "solvent_accessibility", "secondary_structure_context",
     "dist_to_active_site")
BLOCKED = ("AlphaFoldConnector", "ProteinStructurePipeline")
CURRENT = tuple(f"core_{i}" for i in range(91))
OLD = CURRENT + Q
SHA = hashlib.sha256(b"synthetic reviewed contract").hexdigest()


@pytest.fixture
def frame():
    return pd.DataFrame({"variant_id": ["a", "b"], "ref": ["A", "C"]}, index=[11, 19])


def _annotate(frame, producer, **kw):
    args = dict(producer_id="ReviewedOtherProducer", producer=producer,
                declared_outputs=("annotation",), quarantined=Q,
                identity_columns=("variant_id",), blocked_producers=BLOCKED)
    args.update(kw)
    return checked_annotation(frame, **args)


# ---------------------------------------------------------------------------
# Boundary tests (containment boundary reference, 34)
# ---------------------------------------------------------------------------
def test_constructor_refuses_before_resource_factory():
    io = Mock(side_effect=AssertionError("resource accessed"))

    class Producer:
        def __init__(self):
            require_producer_enabled("ProteinStructurePipeline", BLOCKED)
            io()
    with pytest.raises(ContainmentError):
        Producer()
    io.assert_not_called()


def test_existing_instance_entry_still_refuses():
    class Producer:
        def annotate(self, frame):
            require_producer_enabled("AlphaFoldConnector", BLOCKED)
            raise AssertionError("body reached")
    old_instance = object.__new__(Producer)
    with pytest.raises(ContainmentError):
        old_instance.annotate(object())


def test_blocked_producer_refuses_before_frame_inspection():
    producer = Mock()
    with pytest.raises(ContainmentError):
        checked_annotation(object(), producer_id=BLOCKED[0], producer=producer,
                           declared_outputs=("annotation",), quarantined=Q,
                           identity_columns=("variant_id",), blocked_producers=BLOCKED)
    producer.assert_not_called()


def test_legacy_contract_refuses_before_builder():
    builder = Mock()
    with pytest.raises(ContainmentError):
        require_scientific_contract(OLD, Q)
        builder()
    builder.assert_not_called()


def test_active_contract_is_structurally_eligible():
    require_scientific_contract(CURRENT, Q)


def test_duplicate_contract_names_refuse():
    with pytest.raises(ContainmentError):
        require_scientific_contract(("a", "a"), Q)


def test_missing_binding_refuses():
    with pytest.raises(ContainmentError):
        require_contract_binding(None, SHA)


def test_prefix_digest_refuses():
    with pytest.raises(ContainmentError):
        require_contract_binding(SHA[:16], SHA)


def test_wrong_binding_refuses():
    with pytest.raises(ContainmentError):
        require_contract_binding("f" * 64, SHA)


def test_equal_binding_passes_only_this_check():
    require_contract_binding(SHA, SHA)


def test_valid_matrix():
    require_matrix(pd.DataFrame([[1, 2]], columns=["a", "b"]), ("a", "b"), Q)


def test_same_count_wrong_order_refuses():
    with pytest.raises(ContainmentError):
        require_matrix(pd.DataFrame([[1, 2]], columns=["b", "a"]), ("a", "b"), Q)


def test_anonymous_array_refuses():
    with pytest.raises(ContainmentError):
        require_matrix(np.zeros((1, 2)), ("a", "b"), Q)


def test_matrix_with_quarantined_extra_refuses():
    with pytest.raises(ContainmentError):
        require_matrix(pd.DataFrame({"a": [1], Q[0]: [50]}), ("a",), Q)


def test_no_zero_filling_for_missing_column():
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(ContainmentError):
        require_matrix(frame, ("a", "b"), Q)
    assert list(frame) == ["a"]


def test_actual_quarantined_output_refuses_despite_innocent_declaration(frame):
    def producer(f):
        f["annotation"] = 1
        f[Q[0]] = 90
        return f
    with pytest.raises(ContainmentError, match="emitted a quarantined"):
        _annotate(frame, producer)
    assert Q[0] not in frame


def test_quarantined_input_refuses_without_dropping(frame):
    frame[Q[0]] = 50
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer)
    producer.assert_not_called()
    assert Q[0] in frame


def test_quarantined_declaration_refuses_before_call(frame):
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer, declared_outputs=(Q[0],))
    producer.assert_not_called()


def test_benign_annotation_passes_without_input_mutation(frame):
    result = _annotate(frame, lambda f: f.assign(annotation=[3, 4]))
    assert result.annotation.tolist() == [3, 4]
    assert "annotation" not in frame


def test_undeclared_column_refuses(frame):
    with pytest.raises(ContainmentError):
        _annotate(frame, lambda f: f.assign(annotation=3, unexpected=4))


def test_missing_declared_output_refuses(frame):
    with pytest.raises(ContainmentError):
        _annotate(frame, lambda f: f)


def test_changed_protected_value_refuses(frame):
    with pytest.raises(ContainmentError):
        _annotate(frame, lambda f: f.assign(ref="G", annotation=3))
    assert frame.ref.tolist() == ["A", "C"]


def test_reordered_rows_refuse(frame):
    with pytest.raises(ContainmentError):
        _annotate(frame, lambda f: f.assign(annotation=3).iloc[::-1])


def test_duplicate_output_columns_refuse(frame):
    def producer(f):
        f["annotation"] = 1
        return pd.concat([f, f[["annotation"]]], axis=1)
    with pytest.raises(ContainmentError):
        _annotate(frame, producer)


def test_nested_object_mutation_is_isolated_and_detected(frame):
    frame["metadata"] = [{"tags": ["original"]}, {"tags": []}]

    def producer(f):
        f.at[11, "metadata"]["tags"].append("changed")
        f["annotation"] = 1
        return f
    with pytest.raises(ContainmentError):
        _annotate(frame, producer)
    assert frame.at[11, "metadata"] == {"tags": ["original"]}


def test_empty_frame_does_not_allow_quarantined_producer(frame):
    frame = frame.iloc[:0]
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer, producer_id=BLOCKED[1])
    producer.assert_not_called()


def test_identity_cannot_be_declared_as_mutable_output(frame):
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer, declared_outputs=("annotation", "variant_id"))
    producer.assert_not_called()


def test_missing_identity_refuses(frame):
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer, identity_columns=("observation_id",))
    producer.assert_not_called()


def test_duplicate_identity_refuses(frame):
    frame["variant_id"] = "same"
    producer = Mock()
    with pytest.raises(ContainmentError):
        _annotate(frame, producer)
    producer.assert_not_called()


def test_admission_failure_precedes_file_open_and_deserialization():
    loader = Mock()

    def admission():
        require_scientific_contract(OLD, Q)
    with patch.object(Path, "open", side_effect=AssertionError("opened")) as op:
        with pytest.raises(ContainmentError):
            load_after_admission(Path("absent.model"), admit=admission,
                                 artifact_sha256=SHA, deserialize=loader)
        op.assert_not_called()
    loader.assert_not_called()


def test_missing_binding_blocks_artifact_open():
    def admission():
        require_contract_binding(None, SHA)
    with patch.object(Path, "open", side_effect=AssertionError("opened")) as op:
        with pytest.raises(ContainmentError):
            load_after_admission(Path("absent.model"), admit=admission,
                                 artifact_sha256=SHA, deserialize=Mock())
        op.assert_not_called()


def test_wrong_artifact_digest_never_deserializes(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"known bytes")
    loader = Mock()
    with pytest.raises(ContainmentError):
        load_after_admission(path, admit=lambda: None,
                             artifact_sha256="f" * 64, deserialize=loader)
    loader.assert_not_called()


def test_deserializer_reads_verified_snapshot_even_if_path_changes(tmp_path):
    path = tmp_path / "model.bin"
    data = b"known bytes"
    path.write_bytes(data)

    def loader(stream):
        path.write_bytes(b"changed after hash")
        return stream.read()
    result = load_after_admission(path, admit=lambda: None,
                                  artifact_sha256=hashlib.sha256(data).hexdigest(),
                                  deserialize=loader)
    assert result == data


def test_containment_exception_propagates_from_producer(frame):
    def producer(f):
        require_producer_enabled(BLOCKED[0], BLOCKED)
    with pytest.raises(ContainmentError):
        _annotate(frame, producer)


# ---------------------------------------------------------------------------
# Contract, lineage and binding tests (quarantine reference, 15 of 17)
# ---------------------------------------------------------------------------
@pytest.fixture
def contract():
    return Contract("candidate", ("length", "consequence", "structure"),
                    frozenset({"structure"}), "a" * 64)


def test_active_preserves_order_and_catalog(contract):
    assert contract.active == ("length", "consequence")
    assert "structure" in contract.catalog


def test_exact_matrix_passes(contract):
    contract.require_raw_matrix(("length", "consequence"))


def test_reordered_matrix_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_raw_matrix(("consequence", "length"))


def test_extra_cached_column_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_raw_matrix(contract.catalog)


def test_renamed_quarantined_dependency_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("innocent",), {"innocent": ("structure",)})


def test_mask_and_transitive_dependency_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("scaled_mask",), {
            "mask": ("structure",), "scaled_mask": ("mask",)})


def test_unknown_lineage_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("unregistered",), {})


def test_cycle_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("a",), {"a": ("b",), "b": ("a",)})


def test_raw_feature_redefinition_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("length",), {"length": ("consequence",)})


def test_compatible_binding(contract):
    b = bind_model(contract, contract.active, {}, "b" * 64)
    require_model_compatibility(b, contract, contract.active, {}, "b" * 64)


def test_stale_preprocessor_refused(contract):
    b = bind_model(contract, contract.active, {}, "b" * 64)
    with pytest.raises(QuarantineError):
        require_model_compatibility(b, contract, contract.active, {}, "c" * 64)


def test_changed_semantics_same_names_refused(contract):
    b = bind_model(contract, contract.active, {}, "b" * 64)
    newer = Contract("candidate", contract.catalog, contract.quarantined, "d" * 64)
    with pytest.raises(QuarantineError):
        require_model_compatibility(b, newer, newer.active, {}, "b" * 64)


def test_old_unquarantined_contract_refused(contract):
    old = Contract("historical", contract.catalog, frozenset(), "a" * 64)
    b = bind_model(old, old.active, {}, "b" * 64)
    with pytest.raises(QuarantineError):
        require_model_compatibility(b, contract, contract.active, {}, "b" * 64)


def test_digest_prefix_refused(contract):
    with pytest.raises(QuarantineError):
        Contract("candidate", contract.catalog, contract.quarantined, "a" * 16)


def test_duplicate_columns_refused(contract):
    with pytest.raises(QuarantineError):
        contract.require_lineage(("length", "length"), {})


# ---------------------------------------------------------------------------
# The one deliberate change: a single exception hierarchy (addendum section 3.1)
# ---------------------------------------------------------------------------
def test_quarantine_error_is_a_containment_error():
    assert issubclass(QuarantineError, ContainmentError)


def test_quarantine_error_is_not_a_value_error():
    # A ValueError would be caught by ordinary "source unavailable" fallbacks.
    assert not issubclass(QuarantineError, ValueError)


def test_quarantine_refusal_escapes_the_prescribed_handler_pattern(contract):
    """Replays the scenario demonstrated on 2026-09-22, where the reference packages'
    unrelated exception types let this refusal be swallowed by `except ValueError`."""
    swallowed = False
    with pytest.raises(QuarantineError):
        try:
            contract.require_raw_matrix(contract.catalog)
        except ContainmentError:
            raise
        except ValueError:
            swallowed = True
    assert not swallowed


def test_every_quarantine_refusal_is_caught_as_containment(contract):
    refusals = [
        lambda: contract.require_raw_matrix(("consequence", "length")),
        lambda: contract.require_annotation_request(("structure",)),
        lambda: contract.require_lineage(("x",), {"x": ("structure",)}),
        lambda: Contract("v", ("a",), frozenset({"b"}), "a" * 64),
    ]
    for refuse in refusals:
        with pytest.raises(ContainmentError):
            refuse()
