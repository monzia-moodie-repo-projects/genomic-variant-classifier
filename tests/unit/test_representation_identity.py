"""A representation is what two frames must share to be comparable.

DRIFT-1 Phase 1B. Created 2026-08-27.

WHAT THIS GUARDS
----------------
`DriftDetector.check` raises when the new data lacks features the reference
expects, and `_aligned_lsif_matrices` refuses same-width-but-reordered frames.
Both are correct, and both fire at comparison time on data already loaded.

`RepresentationIdentity` states the contract BEFORE either side exists, so a
mismatch is refused with a named cause rather than discovered as a KeyError.

THE PERMISSIVE DIRECTION MATTERS AS MUCH AS THE REFUSALS.
"A guard that only knows how to refuse eventually becomes unusable" -- a
representation rebuilt from the same declaration must compare EQUAL, or every
legitimate rebuild becomes a false mismatch and the type gets worked around.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib

import pytest

from genomic_variant_classifier.monitoring.drift.representation import (
    RepresentationIdentity,
    RepresentationMismatch,
    RepresentationPlane,
    assert_same_representation,
)

_POLICY = "a" * 64
_MANIFEST = "b" * 64
_NAMES = ("gnomad_af", "spliceai_max", "cadd_phred", "phylop_score")


def ident(**over):
    kw = dict(plane=RepresentationPlane.SEMANTIC_TABULAR,
              feature_names=_NAMES,
              preprocessing_policy_sha256=_POLICY,
              source_manifest_sha256=_MANIFEST)
    kw.update(over)
    return RepresentationIdentity(**kw)


# ---------------------------------------------------------------------------
# 1. The permissive direction
# ---------------------------------------------------------------------------

def test_a_representation_rebuilt_from_the_same_declaration_is_equal():
    """The case that keeps the type usable.

    If a rebuild did not compare equal, every legitimate regeneration would
    look like drift in the contract itself.
    """
    assert ident() == ident()
    assert_same_representation(ident(), ident())


def test_equality_is_on_the_whole_object_not_the_width():
    """`shape[1] == 95` admits a frame whose columns were substituted."""
    other = ident(feature_names=("w", "x", "y", "z"))
    assert other.n_features == ident().n_features
    assert other != ident()


def test_the_contract_digest_is_derived_and_stable():
    """Derived, so it cannot disagree with the names it digests."""
    expected = hashlib.sha256("\n".join(_NAMES).encode("utf-8")).hexdigest()
    assert ident().feature_contract_digest == expected
    assert ident().feature_contract_digest == ident().feature_contract_digest


def test_the_contract_digest_changes_with_ORDER():
    """Order is part of the contract, so it must be part of the digest."""
    reordered = ident(feature_names=tuple(reversed(_NAMES)))
    assert set(reordered.feature_names) == set(_NAMES)
    assert reordered.feature_contract_digest != ident().feature_contract_digest


def test_there_is_no_stored_contract_digest_field():
    """Guards the design decision itself.

    Storing the digest beside the names would be two fields for one fact, and
    a caller who edited one and not the other would produce an identity that
    is internally false with nothing to notice.
    """
    assert "feature_contract_digest" not in ident().__dict__
    assert isinstance(
        type(ident()).feature_contract_digest, property)


# ---------------------------------------------------------------------------
# 2. Construction refuses what cannot be an identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over,fragment",
    [({"feature_names": ()}, "must enumerate its features"),
     ({"feature_names": ("a", "b", "a")}, "must be unique"),
     ({"feature_names": list(_NAMES)}, "must be a TUPLE"),
     ({"feature_names": ("a", "")}, "non-empty string"),
     ({"feature_names": ("a", "b\nc")}, "digest separator"),
     ({"preprocessing_policy_sha256": "a" * 16}, "64 lowercase hexadecimal"),
     ({"preprocessing_policy_sha256": "A" * 64}, "64 lowercase hexadecimal"),
     ({"source_manifest_sha256": None}, "64 lowercase hexadecimal"),
     ({"plane": "semantic_tabular"}, "must name its plane")],
    ids=["empty", "duplicate", "list-not-tuple", "empty-name", "newline-name",
         "short-policy-digest", "uppercase-digest", "null-manifest",
         "plane-as-string"])
def test_construction_refuses(over, fragment):
    with pytest.raises(ValueError) as exc:
        ident(**over)
    assert fragment in str(exc.value)


def test_a_newline_in_a_feature_name_is_refused_because_it_is_the_separator():
    """Not fussiness: two different tuples could otherwise digest identically.

    ("a\\nb",) and ("a", "b") both join to "a\\nb".
    """
    a = hashlib.sha256("\n".join(("a", "b")).encode("utf-8")).hexdigest()
    b = hashlib.sha256("\n".join(("a\nb",)).encode("utf-8")).hexdigest()
    assert a == b, "the collision this rule prevents must be real"
    with pytest.raises(ValueError):
        ident(feature_names=("a\nb",))


# ---------------------------------------------------------------------------
# 3. Mismatch names WHICH field, and why it matters
# ---------------------------------------------------------------------------

def test_a_reordered_representation_is_refused_and_says_so():
    """The case a width check cannot see."""
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(ident(),
                                   ident(feature_names=tuple(reversed(_NAMES))))
    assert "DIFFERENT ORDER" in str(exc.value)


def test_a_substituted_feature_names_what_is_missing_and_what_is_extra():
    swapped = ("gnomad_af", "spliceai_max", "cadd_phred", "revel_score")
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(ident(), ident(feature_names=swapped))
    assert "phylop_score" in str(exc.value)
    assert "revel_score" in str(exc.value)


def test_a_changed_preprocessing_policy_is_refused():
    """Same values imputed differently are not the same observation."""
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(
            ident(), ident(preprocessing_policy_sha256="c" * 64))
    assert "preprocessing policy differs" in str(exc.value)


def test_a_changed_source_manifest_is_refused_as_MEASUREMENT_drift():
    """The distinction that keeps the science honest.

    Same variants, new dbNSFP release, CADD moves. The population did not
    drift; the measurement process did. Calling that population drift would be
    a scientific error.
    """
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(
            ident(), ident(source_manifest_sha256="d" * 64))
    assert "MEASUREMENT-PROCESS drift" in str(exc.value)


def test_a_different_plane_is_refused():
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(
            ident(), ident(plane=RepresentationPlane.MODEL_INPUT))
    assert "different" in str(exc.value)


def test_every_plane_is_distinct_and_named():
    values = [p.value for p in RepresentationPlane]
    assert len(set(values)) == len(values) == 3
    assert "semantic_tabular" in values and "model_input" in values


def test_describe_carries_every_identifying_field():
    text = ident().describe()
    for fragment in ("semantic_tabular", "4 features",
                     ident().feature_contract_digest[:12], _POLICY[:12],
                     _MANIFEST[:12]):
        assert fragment in text
