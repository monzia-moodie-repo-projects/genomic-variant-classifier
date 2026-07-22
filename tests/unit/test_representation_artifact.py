"""The extraction boundary persists the representation faithfully, and every
guarantee it claims can fail when violated.

WHAT IS PINNED
==============
Four guarantees (see representation_artifact.py):
  1. identity     -- the persisted matrix is exactly the exported embedding,
                     detached to CPU float64.
  2. row order    -- a SHA-256 over ordered row keys; a reorder is detectable.
  3. provenance   -- git SHA, partition, name, shape, dtype, timestamp, frozen.
  4. partition    -- the role travels with the matrix and is validated.

Plus the ladder discipline: this module persists and NOTHING more -- no probe,
no whitening, no metric. Those are later rungs.

These tests import NO torch. extract_focal_embeddings is duck-typed on
has_embeddings / focal_embeddings, so a tiny mock stands in for GNNOutput and a
numpy array stands in for the tensor. The identity-and-detach behaviour on a real
torch tensor is covered by test_gnn_typed_output.py's exact-classifier-input
hook; here we pin the boundary's own logic.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.representation_artifact import (
    RepresentationArtifact,
    extract_focal_embeddings,
    hash_row_order,
    RowOrderMismatch,
    VALID_PARTITION_ROLES,
)


# --------------------------------------------------------------------------- #
# a mock GNNOutput -- duck-typed, no torch
# --------------------------------------------------------------------------- #
class _MockGNNOutput:
    def __init__(self, focal_embeddings):
        self.focal_embeddings = focal_embeddings

    @property
    def has_embeddings(self):
        return self.focal_embeddings is not None


class _MockTensor:
    """Mimics the torch methods extract_focal_embeddings duck-types on."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.detached = False
        self.moved = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.moved = True
        return self

    def numpy(self):
        return self._arr


def _emb(n=6, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def _keys(n=6):
    return [f"variant_{i}" for i in range(n)]


# --------------------------------------------------------------------------- #
# 1. identity and detach
# --------------------------------------------------------------------------- #
def test_extract_returns_the_matrix_values():
    arr = _emb()
    art = extract_focal_embeddings(
        _MockGNNOutput(arr), _keys(),
        representation_name="gnn.focal.pre_classifier",
        partition_role="TRAIN", model_class="VariantGAT", git_sha="abc")
    np.testing.assert_array_equal(art.embeddings, arr)


def test_extract_detaches_and_moves_a_tensor_like():
    t = _MockTensor(_emb())
    extract_focal_embeddings(
        _MockGNNOutput(t), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    assert t.detached, "extraction must detach from autograd"
    assert t.moved, "extraction must move to CPU"


def test_extracted_matrix_is_float64():
    arr = _emb().astype(np.float32)
    art = extract_focal_embeddings(
        _MockGNNOutput(arr), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    assert art.embeddings.dtype == np.float64, (
        "downstream linear algebra must not silently run in float32")


def test_extracted_matrix_is_read_only():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    with pytest.raises(ValueError):
        art.embeddings[0, 0] = 999.0


# --------------------------------------------------------------------------- #
# 2. row order hashing
# --------------------------------------------------------------------------- #
def test_row_order_hash_is_order_sensitive():
    a = hash_row_order(["x", "y", "z"])
    b = hash_row_order(["x", "z", "y"])
    assert a != b, "a reorder must change the hash"


def test_row_order_hash_no_delimiter_collision():
    assert hash_row_order(["ab", "c"]) != hash_row_order(["a", "bc"])


def test_verify_row_order_accepts_the_same_order():
    keys = _keys()
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), keys,
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    art.verify_row_order(keys)  # must not raise


def test_verify_row_order_rejects_a_reorder():
    keys = _keys()
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), keys,
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    reordered = list(reversed(keys))
    with pytest.raises(RowOrderMismatch):
        art.verify_row_order(reordered)


# --------------------------------------------------------------------------- #
# 3. provenance and immutability
# --------------------------------------------------------------------------- #
def test_provenance_fields_are_populated():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb(6, 8)), _keys(),
        representation_name="gnn.focal.pre_classifier",
        partition_role="STRUCTURE", model_class="VariantGATGPS", git_sha="deadbeef")
    m = art.to_manifest()
    assert m["representation_name"] == "gnn.focal.pre_classifier"
    assert m["partition_role"] == "STRUCTURE"
    assert m["model_class"] == "VariantGATGPS"
    assert m["git_sha"] == "deadbeef"
    assert m["n_rows"] == 6 and m["dim"] == 8
    assert m["dtype"] == "float64"
    assert m["created_utc"].endswith("+00:00"), "timestamp must be UTC ISO-8601"


def test_artifact_is_frozen():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    with pytest.raises(FrozenInstanceError):
        art.partition_role = "TEST"


def test_git_sha_defaults_when_not_supplied():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT")  # no git_sha
    assert isinstance(art.git_sha, str) and art.git_sha


# --------------------------------------------------------------------------- #
# 4. partition role
# --------------------------------------------------------------------------- #
def test_partition_role_travels_with_the_matrix():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), _keys(),
        representation_name="r", partition_role="STRUCTURE_TEST",
        model_class="VariantGAT", git_sha="abc")
    assert art.partition_role == "STRUCTURE_TEST"


def test_invalid_partition_role_is_rejected():
    with pytest.raises(ValueError):
        extract_focal_embeddings(
            _MockGNNOutput(_emb()), _keys(),
            representation_name="r", partition_role="TARIN",  # typo
            model_class="VariantGAT", git_sha="abc")


def test_all_valid_roles_are_accepted():
    for role in VALID_PARTITION_ROLES:
        art = extract_focal_embeddings(
            _MockGNNOutput(_emb()), _keys(),
            representation_name="r", partition_role=role,
            model_class="VariantGAT", git_sha="abc")
        assert art.partition_role == role


# --------------------------------------------------------------------------- #
# 5. caller-error failure modes
# --------------------------------------------------------------------------- #
def test_extract_refuses_output_without_embeddings():
    # Match on the guard's SPECIFIC message. Without this, a None embedding still
    # raises ValueError -- but from the shape check downstream (np.asarray(None)
    # becomes a 1-D nan array, which fails the 2-D assertion), so a test that only
    # asserted "some ValueError" would pass for the wrong reason if the guard were
    # removed. Pinning the message makes removing the guard turn this red.
    with pytest.raises(ValueError, match="no focal_embeddings"):
        extract_focal_embeddings(
            _MockGNNOutput(None), _keys(),
            representation_name="r", partition_role="TRAIN",
            model_class="VariantGAT", git_sha="abc")


def test_extract_rejects_key_count_mismatch():
    with pytest.raises(ValueError):
        extract_focal_embeddings(
            _MockGNNOutput(_emb(6, 8)), _keys(5),  # 5 keys, 6 rows
            representation_name="r", partition_role="TRAIN",
            model_class="VariantGAT", git_sha="abc")


def test_construction_rejects_inconsistent_hash():
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb()), _keys(),
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    import dataclasses
    with pytest.raises(ValueError):
        dataclasses.replace(art, row_order_sha256="0" * 64)


# --------------------------------------------------------------------------- #
# 6. persistence round-trip
# --------------------------------------------------------------------------- #
def test_save_writes_matrix_and_manifest(tmp_path):
    art = extract_focal_embeddings(
        _MockGNNOutput(_emb(6, 8)), _keys(),
        representation_name="gnn.focal.pre_classifier",
        partition_role="TRAIN", model_class="VariantGAT", git_sha="abc")
    paths = art.save(tmp_path)
    assert paths["matrix"].exists() and paths["manifest"].exists()
    man = json.loads(paths["manifest"].read_text())
    assert man["row_order_sha256"] == art.row_order_sha256


def test_saved_matrix_round_trips_values_and_order(tmp_path):
    arr = _emb(6, 8)
    keys = _keys()
    art = extract_focal_embeddings(
        _MockGNNOutput(arr), keys,
        representation_name="r", partition_role="TRAIN",
        model_class="VariantGAT", git_sha="abc")
    paths = art.save(tmp_path)
    import pandas as pd
    df = pd.read_parquet(paths["matrix"])
    assert list(df["row_key"]) == keys, "persisted row order must match"
    recon = df[[f"e{j:04d}" for j in range(8)]].to_numpy()
    np.testing.assert_allclose(recon, arr, rtol=0, atol=0)


def test_no_probe_or_metric_surface_is_exposed():
    """Ladder discipline: this module persists and stops. It must not expose a
    probe, whitening, or metric -- those are later rungs. If a future edit adds
    one here, this turns red and forces the author to put it in the right commit."""
    import genomic_variant_classifier.evaluation.representation_artifact as m
    forbidden = [n for n in dir(m)
                 if any(w in n.lower() for w in ("whiten", "probe", "fit_",
                                                 "metric", "auroc", "silhouette"))]
    assert not forbidden, f"extraction boundary leaked an analysis surface: {forbidden}"
