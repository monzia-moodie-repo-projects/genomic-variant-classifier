"""Representation and source state are independent coordinate axes.

DRIFT-1 Phase 1B.1. Created 2026-08-27.

THE DEFECT THIS GUARDS
----------------------
MEASURED against the previous implementation on 2026-08-27:

    reference   ClinVar 2026-07, dbNSFP 4.7a
    candidate   ClinVar 2026-08, dbNSFP 4.7a
    same plane, same feature names, same policy
        -> assert_same_representation REFUSED

That is exactly the temporal comparison DRIFT-1 exists to make. The prose was
right -- an annotation release moving IS measurement-process drift -- but the
FIELD PLACEMENT was wrong. It collapsed a source-state difference into a
representation incompatibility, so the two could not be told apart at all.

WHY THE TYPE, NOT THE CHECK
---------------------------
The repair is not "ignore the source field". `RepresentationIdentity` no longer
HAS one, so a future refactor cannot reintroduce the coupling by editing a
comparison. The type is incapable of conflating them.

THE FOUR QUADRANTS
------------------
    representation    source state    expected
    same              same            comparable on both axes
    same              different       source movement only
    different         same            representation mismatch only
    different         different       both preserved, independently

Pinning all four is what stops a future refactor re-coupling them. Proving only
the failing case would leave three ways back.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.monitoring.drift import (
    RepresentationDeltaKind,
    RepresentationIdentity,
    RepresentationMismatch,
    RepresentationPlane,
    SourceArtifactIdentity,
    SourceDependency,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRole,
    TransformationComponent,
    TransformationComponentKind,
    TransformationError,
    TransformationIdentity,
    assert_same_representation,
    differing_components,
    differing_releases,
    representation_differences,
)

_NAMES = ("gnomad_af", "spliceai_max", "cadd_phred", "phylop_score")


def transformation(missingness="a" * 64, engineering="b" * 64):
    return TransformationIdentity.of((
        TransformationComponent(TransformationComponentKind.MISSINGNESS,
                                1, missingness),
        TransformationComponent(TransformationComponentKind.FEATURE_ENGINEERING,
                                3, engineering)))


def representation(names=_NAMES, transform=None,
                   plane=RepresentationPlane.SEMANTIC_TABULAR):
    return RepresentationIdentity(plane=plane, feature_names=names,
                                  transformation=transform or transformation())


def sources(clinvar="2026-07", sha="c"):
    return SourceManifest(evidence=SourceEvidenceManifest.of((
        SourceDependency(
            identity=SourceArtifactIdentity(
                source="ClinVar", release_id=clinvar, genome_build="GRCh38",
                artifact_sha256=sha * 64),
            roles=frozenset({SourceRole.OBSERVATION, SourceRole.LABEL})),
        SourceDependency(
            identity=SourceArtifactIdentity(
                source="dbNSFP", release_id="4.7a", genome_build="GRCh38",
                artifact_sha256="d" * 64),
            roles=frozenset({SourceRole.ANNOTATION})))))


# ---------------------------------------------------------------------------
# 1. The type cannot express the coupling
# ---------------------------------------------------------------------------

def test_a_representation_has_no_source_field():
    """Structural, not behavioural.

    Proving that two differing source digests are IGNORED would leave a field
    a future refactor could start honouring again. There is no field.
    """
    fields = set(vars(representation()))
    assert "source_manifest_sha256" not in fields
    assert not any("source" in f for f in fields), fields
    assert fields == {"plane", "feature_names", "transformation"}


def test_a_representation_cannot_be_built_without_transformation_semantics():
    """Same columns computed differently are not the same space."""
    with pytest.raises(ValueError) as exc:
        RepresentationIdentity(plane=RepresentationPlane.SEMANTIC_TABULAR,
                               feature_names=_NAMES, transformation="a" * 64)
    assert "what semantics produced its values" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. The four quadrants
# ---------------------------------------------------------------------------

def test_quadrant_same_representation_same_sources():
    assert_same_representation(representation(), representation())
    assert differing_releases(sources(), sources()) == ()


def test_quadrant_same_representation_DIFFERENT_sources():
    """The comparison the old model could not express.

    This is the temporal case: ClinVar moves, dbNSFP is held, and the feature
    space is untouched. Both facts must coexist.
    """
    assert_same_representation(representation(), representation())
    assert differing_releases(sources("2026-07", "c"),
                              sources("2026-08", "e")) == ("ClinVar",)


def test_quadrant_DIFFERENT_representation_same_sources():
    """The orthogonal basis vector."""
    with pytest.raises(RepresentationMismatch):
        assert_same_representation(
            representation(), representation(transform=transformation(
                engineering="f" * 64)))
    assert differing_releases(sources(), sources()) == ()


def test_quadrant_both_different_preserves_both():
    """Neither difference masks the other."""
    diffs = representation_differences(
        representation(), representation(transform=transformation(
            engineering="f" * 64)))
    assert [d.kind for d in diffs] == [RepresentationDeltaKind.TRANSFORMATION]
    assert differing_releases(sources("2026-07", "c"),
                             sources("2026-08", "e")) == ("ClinVar",)


# ---------------------------------------------------------------------------
# 3. Differences are complete and typed, not first-failure strings
# ---------------------------------------------------------------------------

def test_every_difference_is_reported_not_only_the_first():
    """Admission needs the whole delta.

    A comparison where the plane matches, the feature SET changed AND the
    transformation moved is two facts; an exception reports one.
    """
    a = representation()
    b = representation(names=("gnomad_af", "revel_score"),
                       transform=transformation(engineering="f" * 64),
                       plane=RepresentationPlane.MODEL_INPUT)
    kinds = {d.kind for d in representation_differences(a, b)}
    assert kinds == {RepresentationDeltaKind.PLANE,
                     RepresentationDeltaKind.FEATURE_SET,
                     RepresentationDeltaKind.TRANSFORMATION}


def test_a_reorder_is_distinguished_from_a_substitution():
    """A width check sees neither; a set check sees only the second."""
    (delta,) = representation_differences(
        representation(), representation(names=tuple(reversed(_NAMES))))
    assert delta.kind is RepresentationDeltaKind.FEATURE_ORDER
    assert "DIFFERENT ORDER" in delta.detail


def test_the_strict_adapter_raises_on_the_same_differences():
    """One authority computes; the adapter only decides whether to raise."""
    a, b = representation(), representation(names=("x", "y"))
    diffs = representation_differences(a, b)
    assert diffs
    with pytest.raises(RepresentationMismatch) as exc:
        assert_same_representation(a, b)
    for d in diffs:
        assert d.kind.value in str(exc.value)


def test_no_difference_means_no_exception():
    """The permissive direction: a rebuild must not look like drift."""
    assert representation_differences(representation(), representation()) == ()
    assert_same_representation(representation(), representation())


# ---------------------------------------------------------------------------
# 4. Transformation is compositional, and names which component moved
# ---------------------------------------------------------------------------

def test_a_transformation_names_which_component_moved():
    """"the transformation differs" is not a statement about a pipeline."""
    moved = differing_components(transformation(),
                                 transformation(engineering="f" * 64))
    assert moved == (TransformationComponentKind.FEATURE_ENGINEERING,)


def test_missingness_alone_is_not_the_whole_transformation():
    """Same features, same missingness policy, DIFFERENT join policy.

    The previous design carried one `preprocessing_policy_sha256` holding the
    missingness fingerprint, which cannot distinguish this case at all.
    """
    a = transformation()
    b = TransformationIdentity.of(a.components + (
        TransformationComponent(TransformationComponentKind.JOIN_POLICY,
                                1, "e" * 64),))
    assert a != b
    assert differing_components(a, b) == (
        TransformationComponentKind.JOIN_POLICY,)


def test_one_component_kind_twice_is_refused():
    with pytest.raises(TransformationError) as exc:
        TransformationIdentity.of((
            TransformationComponent(TransformationComponentKind.MISSINGNESS,
                                    1, "a" * 64),
            TransformationComponent(TransformationComponentKind.MISSINGNESS,
                                    2, "b" * 64)))
    assert "more than once" in str(exc.value)


def test_an_empty_transformation_is_refused():
    with pytest.raises(TransformationError) as exc:
        TransformationIdentity.of(())
    assert "at least one component" in str(exc.value)


def test_component_order_does_not_move_identity():
    m = TransformationComponent(TransformationComponentKind.MISSINGNESS,
                                1, "a" * 64)
    f = TransformationComponent(TransformationComponentKind.FEATURE_ENGINEERING,
                                3, "b" * 64)
    assert TransformationIdentity.of((m, f)) == TransformationIdentity.of((f, m))


def test_a_transformation_built_out_of_canonical_order_is_refused():
    m = TransformationComponent(TransformationComponentKind.MISSINGNESS,
                                1, "a" * 64)
    f = TransformationComponent(TransformationComponentKind.FEATURE_ENGINEERING,
                                3, "b" * 64)
    with pytest.raises(TransformationError) as exc:
        TransformationIdentity(components=(m, f))
    assert "canonical order" in str(exc.value)


def test_a_component_schema_version_is_part_of_its_identity():
    """A declaration whose SHAPE changed is a different declaration."""
    a = TransformationIdentity.of((TransformationComponent(
        TransformationComponentKind.MISSINGNESS, 1, "a" * 64),))
    b = TransformationIdentity.of((TransformationComponent(
        TransformationComponentKind.MISSINGNESS, 2, "a" * 64),))
    assert a.digest != b.digest
