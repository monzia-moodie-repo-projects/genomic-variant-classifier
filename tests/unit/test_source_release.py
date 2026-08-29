"""The source identity kernel, and the four corrections it now carries.

DRIFT-1 Phase 1B.3. Created 2026-08-28.

IDENTITY KERNEL
---------------
    SourceArtifactIdentity

    equal iff   artifact KEY (authority + artifact kind), release identifier,
                coordinate context, artifact bytes
    ignores     retrieval timestamp, retrieval location, transport,
                row-count observation

WHAT THIS FILE GUARDS THAT ITS PREDECESSOR COULD NOT
----------------------------------------------------
Four defects, each MEASURED before it was repaired:

  1. ONE ARTIFACT PER SOURCE was false. Across 3,420 artifact files, TEN
     authorities hold more than one kind, and one module consumes THREE
     distinct ClinVar artifacts -- `monitoring/registry.py` names
     `index.parquet`, `parquet` and `variant_summary.txt`.

  2. MANDATORY GRCh37/GRCh38 was false for SIX of sixteen authorities. A
     UniProt accession has no genomic position.

  3. FREE-FORM SOURCE NAMES let `ClinVar`, `clinvar` and `NCBI-ClinVar` be
     three identities. No registry existed to prevent it.

  4. A ROLE CHANGE moved the manifest digest and produced NO delta. Measured
     against the installed code: digests differ, `source_deltas` returned `()`.

INVARIANCE AND SENSITIVITY BOTH
-------------------------------
A test proving only "X changes the identity" is satisfied by an identity that
changes for everything; one proving only "Y does not" is satisfied by one that
never changes. The pair pins the kernel.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.monitoring.drift import (
    ArtifactKind,
    CoordinateContext,
    CoordinateContextKind,
    CoordinateError,
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDeltaKind,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
    SourceTransition,
    differing_releases,
    source_transitions,
)

GRCH38 = CoordinateContext.assembly("GRCh38")


def identity(**over):
    kw = dict(key=SourceArtifactKey("clinvar",
                                    ArtifactKind.PRIMARY_RELEASE),
              release_id="2026-08", coordinate_context=GRCH38,
              artifact_sha256="a" * 64)
    kw.update(over)
    return SourceArtifactIdentity(**kw)


def provenance(**over):
    kw = dict(retrieved_at="2026-08-01T00:00:00Z", observed_row_count=4_400_000)
    kw.update(over)
    return SourceRetrievalProvenance(**kw)


def dependency(ident=None, *roles):
    return SourceDependency(identity=ident or identity(),
                            roles=frozenset(roles or (SourceRole.OBSERVATION,)))


def evidence(*deps):
    return SourceEvidenceManifest.of(deps or (dependency(),))


# ---------------------------------------------------------------------------
# 1. CORRECTION ONE -- several artifacts from one authority
# ---------------------------------------------------------------------------

def test_one_authority_may_contribute_several_artifact_kinds():
    """The case `monitoring/registry.py` actually exercises.

    MEASURED: it names ClinVar's index.parquet, parquet AND variant_summary.
    The previous invariant made that unrepresentable, and the workaround would
    have been fake authorities such as `ClinVarVCF` -- turning an ARTIFACT
    distinction into a SOURCE distinction and losing that both came from one
    release.
    """
    m = SourceEvidenceManifest.of((
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.PRIMARY_RELEASE))),
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.DERIVED_INDEX),
            artifact_sha256="b" * 64)),
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.VARIANT_SUMMARY),
            artifact_sha256="c" * 64)),
    ))
    assert len(m.dependencies) == 3
    assert m.sources == ("clinvar",)
    # THE CANONICAL NAME, lower_snake_case. `artifacts_of` no longer
    # resolves a spelling through a vocabulary: the standard, section 3, says
    # a source has exactly ONE canonical name, and `SourceRegistry` is what
    # resolves an alias to it.
    assert len(m.artifacts_of("clinvar")) == 3


def test_two_artifacts_of_the_SAME_kind_are_still_refused():
    """The invariant did not disappear; it moved to the right key."""
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of((
            dependency(identity()),
            dependency(identity(artifact_sha256="b" * 64))))
    assert "appear more than once" in str(exc.value)


def test_artifacts_of_returns_nothing_for_an_absent_authority():
    """It no longer REFUSES an unknown name, and that is deliberate.

    The previous version raised `SourceVocabularyError` for any spelling absent
    from an invented eighteen-member enum -- including all eight aliases this
    project declares and sixteen of its thirty-two sources. Asking a manifest
    about a source it does not contain is a QUESTION, not an error; whether the
    name is a DECLARED source is a separate question for `SourceRegistry`.
    """
    assert evidence().artifacts_of("ClinVarPlus") == ()
    assert len(evidence().artifacts_of("clinvar")) == 1


def test_artifacts_of_matches_the_name_EXACTLY_not_as_a_substring():
    """MEASURED GAP: the fixture above cannot see a substring match.

    `"clinvarplus" in "clinvar"` is False and `"clinvar" in "clinvar"` is
    True, so a loose comparison satisfies both assertions. Sabotaging
    `artifacts_of` to match loosely changed NO test.

    Real source names nest: the manifest declares `dbsnp` with the alias
    `dbsnp156`, and `1kgp` with `1000genomes`. A source whose name CONTAINS
    another must not be returned for it, or an alias pending migration would
    silently answer for its canonical form.
    """
    m = SourceEvidenceManifest.of((
        dependency(identity(key=SourceArtifactKey(
            "dbsnp156", ArtifactKind.PRIMARY_RELEASE))),))
    assert m.artifacts_of("dbsnp156") != ()
    assert m.artifacts_of("dbsnp") == (), (
        "a source whose name is a PREFIX of another was matched; "
        "`dbsnp` is a declared source and `dbsnp156` its alias, and they are "
        "not interchangeable until the auditor folds one into the other")


# ---------------------------------------------------------------------------
# 2. CORRECTION TWO -- coordinate context, not mandatory assembly
# ---------------------------------------------------------------------------

def test_build_independent_evidence_may_accompany_any_assembly():
    """Six of sixteen authorities carry no genomic coordinates."""
    m = SourceEvidenceManifest.of((
        dependency(identity()),
        dependency(identity(
            key=SourceArtifactKey("uniprot",
                                  ArtifactKind.SEQUENCE_FASTA),
            release_id="2026_03", artifact_sha256="b" * 64,
            coordinate_context=CoordinateContext.build_independent()))))
    assert sorted(m.assemblies) == ["GRCh38"]


def test_two_genomic_assemblies_are_still_refused():
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of((
            dependency(identity()),
            dependency(identity(
                key=SourceArtifactKey("gnomad",
                                      ArtifactKind.CONSTRAINT_TABLE),
                release_id="v4.1", artifact_sha256="b" * 64,
                coordinate_context=CoordinateContext.assembly("GRCh37")))))
    assert "silently wrong" in str(exc.value)


def test_build_independent_evidence_alone_has_no_assembly():
    m = SourceEvidenceManifest.of((dependency(identity(
        key=SourceArtifactKey("reactome", ArtifactKind.NETWORK_EDGES),
        release_id="v88", coordinate_context=CoordinateContext.build_independent())),))
    assert m.assemblies == frozenset()


@pytest.mark.parametrize(
    "kwargs,fragment",
    [({"kind": CoordinateContextKind.GENOMIC_ASSEMBLY, "identifier": None},
      "must be one of"),
     ({"kind": CoordinateContextKind.GENOMIC_ASSEMBLY, "identifier": "hg38"},
      "NOT interchangeable"),
     ({"kind": CoordinateContextKind.BUILD_INDEPENDENT, "identifier": "GRCh38"},
      "must not INVENT an assembly"),
     ({"kind": "genomic_assembly", "identifier": "GRCh38"},
      "must state whether")],
    ids=["assembly-without-id", "unknown-assembly", "independent-with-id",
         "kind-as-string"])
def test_a_coordinate_context_refuses(kwargs, fragment):
    with pytest.raises(CoordinateError) as exc:
        CoordinateContext(**kwargs)
    assert fragment in str(exc.value)


# ---------------------------------------------------------------------------
# 3. CORRECTION THREE -- the ingestion boundary
# ---------------------------------------------------------------------------

def test_a_raw_string_is_now_the_CORRECT_input(tmp_path):
    """INVERTED. This test previously asserted the opposite.

    `test_the_key_refuses_a_raw_string_bypassing_the_boundary` required a
    `SourceName` member and refused a string, so that "an unregistered name
    cannot mint an identity". The enum it validated against named 18 of 32
    declared sources and refused all 8 declared aliases, so the guard rejected
    real sources and admitted nothing that was not invented.

    Identity now takes a validated STRING. Whether the name is DECLARED is an
    admission question, answered by `SourceRegistry.canonical_for` against
    `configs/data_manifest.yaml`, where the eight real aliases resolve.
    """
    key = SourceArtifactKey("tcga", ArtifactKind.PRIMARY_RELEASE)
    assert key.source == "tcga"
    assert key.canonical_key == ("tcga", "primary_release")
    assert key.as_record()["source"] == "tcga"


@pytest.mark.parametrize(
    "bad", ["", "clin var", "ClinVar/extra", "-leading", None],
    ids=["empty", "spaced", "slashed", "leading-dash", "none"])
def test_the_key_still_refuses_a_name_that_cannot_BE_a_source(bad):
    """A validated string is not an unvalidated one.

    Dropping the enum removed the REGISTRY question, not the SYNTAX one: a
    name with a space or a separator could not be a directory under
    `data/external/`, whatever the manifest says.

    A LEADING DIGIT IS ALLOWED, and my first version of this test wrongly
    expected `9lives` to be refused. The manifest declares `1kgp`, so a pattern
    rejecting leading digits would reject a real source -- the exact failure
    mode this whole unit repairs.
    """
    with pytest.raises(SourceError):
        SourceArtifactKey(bad, ArtifactKind.PRIMARY_RELEASE)


def test_sixteen_sources_the_retired_enum_could_not_NAME(tmp_path):
    """The defect this unit repairs, stated as a test.

    MEASURED 2026-08-29: the enum declared 18 members against the manifest's
    32. Four of the sixteen it could not name are `irreplaceable` and
    constrained -- `tcga` and `topmed` are `controlled`, `rnaseq` and
    `validation_cohort` are `review`. Every one is now expressible.
    """
    for name in ("tcga", "topmed", "rnaseq", "validation_cohort", "1kgp",
                 "clingen", "dbsnp", "ensembl", "finngen", "lovd", "mim2gene",
                 "primateai3d", "revel"):
        key = SourceArtifactKey(name, ArtifactKind.PRIMARY_RELEASE)
        assert key.source == name


# ---------------------------------------------------------------------------
# 4. INVARIANCE -- nuisance perturbations must NOT move identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [{"retrieved_at": "2026-08-02T00:00:00Z"},
     {"observed_row_count": 4_400_001},
     {"origin_locator": "https://elsewhere.example/clinvar.parquet"},
     {"transport": "rclone"}],
    ids=["retrieved_at", "row_count", "origin", "transport"])
def test_acquisition_facts_do_not_move_evidence_identity(over):
    a = SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(identity(), provenance()),))
    b = SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(identity(),
                                                       provenance(**over)),))
    assert a.evidence_digest == b.evidence_digest
    assert differing_releases(a, b) == ()


def test_a_redownload_preserves_identity_but_not_the_record():
    monday = SourceAcquisition(identity(), provenance())
    tuesday = SourceAcquisition(
        identity(), provenance(retrieved_at="2026-08-02T00:00:00Z"))
    assert monday.identity == tuesday.identity
    assert monday.provenance != tuesday.provenance
    assert monday != tuesday


def test_the_digest_cannot_REACH_retrieval_time():
    """Structural: retrieval time is not on the object the digest serialises."""
    record = identity().as_record()
    assert set(record) == {"key", "release_id", "coordinate_context",
                           "artifact_sha256"}
    assert "retrieved_at" not in str(record)


def test_source_transitions_cannot_be_given_acquisitions():
    ev = evidence()
    full = SourceManifest(evidence=ev,
                          acquisitions=(SourceAcquisition(identity(),
                                                          provenance()),))
    assert source_transitions(ev, ev) == ()
    with pytest.raises(AttributeError):
        source_transitions(full, full)


def test_role_order_and_manifest_order_do_not_move_identity():
    x = dependency(identity(), SourceRole.OBSERVATION, SourceRole.LABEL)
    y = dependency(identity(), SourceRole.LABEL, SourceRole.OBSERVATION)
    assert x == y
    other = dependency(identity(key=SourceArtifactKey(
        "dbnsfp", ArtifactKind.PRIMARY_RELEASE),
        release_id="4.7a", artifact_sha256="b" * 64))
    assert (SourceEvidenceManifest.of((x, other))
            == SourceEvidenceManifest.of((other, x)))


# ---------------------------------------------------------------------------
# 5. SENSITIVITY -- defining perturbations MUST move identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [{"key": SourceArtifactKey("gnomad", ArtifactKind.PRIMARY_RELEASE)},
     {"key": SourceArtifactKey("clinvar", ArtifactKind.VCF)},
     {"release_id": "2026-09"},
     {"coordinate_context": CoordinateContext.assembly("GRCh37")},
     {"artifact_sha256": "c" * 64}],
    ids=["authority", "artifact_kind", "release_id", "coordinates", "bytes"])
def test_defining_facts_move_evidence_identity(over):
    """Each in isolation: every other field held constant."""
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(identity(**over)),))
    assert a.digest != b.digest


def test_adding_a_role_moves_evidence_identity():
    a = SourceEvidenceManifest.of((dependency(identity(), SourceRole.OBSERVATION),))
    b = SourceEvidenceManifest.of((dependency(
        identity(), SourceRole.OBSERVATION, SourceRole.LABEL),))
    assert a.digest != b.digest


def test_the_digest_is_domain_separated_and_versioned():
    from genomic_variant_classifier.monitoring.drift._digest import (
        canonical_json, domain_digest)
    payload = {"schema_version": 3, "dependencies": []}
    assert domain_digest("a-v1", payload) != domain_digest("b-v1", payload)
    assert domain_digest("a-v1", payload) != domain_digest("a-v2", payload)
    with pytest.raises(ValueError):
        domain_digest("no-version-suffix", payload)
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


# ---------------------------------------------------------------------------
# 6. CORRECTION FOUR -- every movement has an attributable delta
# ---------------------------------------------------------------------------

def test_a_role_change_now_yields_a_transition():
    """MEASURED against the previous code: digests differed and the delta was
    empty. An identity movement with nothing to attribute it."""
    a = SourceEvidenceManifest.of((dependency(identity(), SourceRole.OBSERVATION),))
    b = SourceEvidenceManifest.of((dependency(
        identity(), SourceRole.OBSERVATION, SourceRole.LABEL),))
    assert a.digest != b.digest
    (t,) = source_transitions(a, b)
    assert t.changes == frozenset({SourceDeltaKind.ROLE_CHANGED})
    assert "roles" in t.describe()


def test_every_change_is_reported_not_a_precedence_winner():
    """MEASURED: three facts moved and ONE was reported, because the assembly
    branch ran first. Branch ORDER had become scientific interpretation."""
    a = SourceEvidenceManifest.of((dependency(identity(), SourceRole.OBSERVATION),))
    b = SourceEvidenceManifest.of((dependency(
        identity(release_id="2026-09", artifact_sha256="f" * 64,
                 coordinate_context=CoordinateContext.assembly("GRCh37")),
        SourceRole.LABEL),))
    (t,) = source_transitions(a, b)
    assert t.changes == frozenset({SourceDeltaKind.RELEASE_MOVED,
                                   SourceDeltaKind.COORDINATE_CONTEXT_CHANGED,
                                   SourceDeltaKind.ROLE_CHANGED})


def test_same_release_different_bytes_is_named_as_such():
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(identity(artifact_sha256="c" * 64)),))
    (t,) = source_transitions(a, b)
    assert t.changes == frozenset(
        {SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE})
    assert "UNCHANGED but bytes differ" in t.describe()


def test_a_release_move_does_not_also_claim_byte_corruption():
    """The two byte-related kinds are mutually exclusive by construction: a
    new release SHOULD have new bytes, and reporting corruption for it would
    make the serious kind meaningless."""
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(
        identity(release_id="2026-09", artifact_sha256="c" * 64)),))
    (t,) = source_transitions(a, b)
    assert t.changes == frozenset({SourceDeltaKind.RELEASE_MOVED})


def test_added_and_removed_artifacts_are_named():
    one = SourceEvidenceManifest.of((dependency(identity()),))
    two = SourceEvidenceManifest.of((
        dependency(identity()),
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.VCF), artifact_sha256="b" * 64))))
    added = source_transitions(one, two)
    assert [t.changes for t in added] == [
        frozenset({SourceDeltaKind.ARTIFACT_ADDED})]
    assert added[0].key.artifact_kind is ArtifactKind.VCF
    removed = source_transitions(two, one)
    assert [t.changes for t in removed] == [
        frozenset({SourceDeltaKind.ARTIFACT_REMOVED})]


def test_differing_releases_reports_each_authority_once():
    """One authority may contribute several artifacts; naming it twice would
    suggest two authorities moved."""
    a = SourceEvidenceManifest.of((
        dependency(identity()),
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.VCF), artifact_sha256="b" * 64))))
    b = SourceEvidenceManifest.of((
        dependency(identity(release_id="2026-09", artifact_sha256="c" * 64)),
        dependency(identity(key=SourceArtifactKey(
            "clinvar", ArtifactKind.VCF), release_id="2026-09",
            artifact_sha256="d" * 64))))
    assert len(source_transitions(a, b)) == 2
    assert differing_releases(SourceManifest(evidence=a),
                              SourceManifest(evidence=b)) == ("clinvar",)


@pytest.mark.parametrize(
    "changes,ref,cand",
    [(frozenset({SourceDeltaKind.ARTIFACT_ADDED}), True, True),
     (frozenset({SourceDeltaKind.ARTIFACT_REMOVED}), True, True),
     (frozenset({SourceDeltaKind.RELEASE_MOVED}), False, True),
     (frozenset(), True, True)],
    ids=["added-with-ref", "removed-with-cand", "change-missing-ref", "empty"])
def test_a_transition_refuses_a_state_that_cannot_exist(changes, ref, cand):
    dep = dependency()
    with pytest.raises(ValueError):
        SourceTransition(key=dep.key,
                         reference=dep if ref else None,
                         candidate=dep if cand else None,
                         changes=changes)


def test_an_added_identity_field_cannot_pass_unclassified():
    """Guards a branch UNREACHABLE with today's fields.

    `SourceDependency` is exactly an identity and a role set, and the identity
    is exactly key, release, coordinate context and bytes. All five are
    compared, so two dependencies cannot differ without a named change --
    which makes the branch dead by construction.

    This repository has ruled on unreachable defences: `suite_transition.py`
    DELETED three, and `publish()`'s re-parse survived only once a reachable
    case was found. The case here IS the future the branch guards: a field
    added without a corresponding `SourceDeltaKind`, invisible to every
    comparison. It is constructed directly.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class WithNewField(SourceArtifactIdentity):
        curation_tier: str = "gold"

    base = dict(key=SourceArtifactKey("clinvar",
                                      ArtifactKind.PRIMARY_RELEASE),
                release_id="2026-08", coordinate_context=GRCH38,
                artifact_sha256="a" * 64)
    a = WithNewField(**base)
    b = WithNewField(curation_tier="silver", **base)
    assert a != b, "the fixture must actually differ"
    assert a.canonical_key == b.canonical_key, (
        "the new field must be invisible to every existing comparison, which "
        "is exactly the condition the guard exists for")

    ref = SourceEvidenceManifest.of((dependency(a, SourceRole.OBSERVATION),))
    cand = SourceEvidenceManifest.of((dependency(b, SourceRole.OBSERVATION),))
    with pytest.raises(RuntimeError) as exc:
        source_transitions(ref, cand)
    assert "UNCLASSIFIED" in str(exc.value)


def test_identical_manifests_yield_no_transitions():
    assert source_transitions(evidence(), evidence()) == ()


# ---------------------------------------------------------------------------
# 7. What a manifest may not be
# ---------------------------------------------------------------------------

def test_an_empty_manifest_is_refused():
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of(())
    assert "at least one dependency" in str(exc.value)


def test_a_dependency_must_declare_a_role():
    with pytest.raises(SourceError) as exc:
        SourceDependency(identity=identity(), roles=frozenset())
    assert "at least one role" in str(exc.value)


def test_a_manifest_built_out_of_canonical_order_is_refused():
    x = dependency(identity())
    y = dependency(identity(key=SourceArtifactKey(
        "dbnsfp", ArtifactKind.PRIMARY_RELEASE),
        release_id="4.7a", artifact_sha256="b" * 64))
    lo, hi = sorted((x, y), key=lambda d: d.canonical_key)
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest(dependencies=(hi, lo))
    assert "canonical order" in str(exc.value)


def test_an_acquisition_of_an_undeclared_artifact_is_refused():
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(
                           identity(key=SourceArtifactKey(
                               "cosmic", ArtifactKind.PRIMARY_RELEASE)),
                           provenance()),))
    assert "does not declare" in str(exc.value)


@pytest.mark.parametrize(
    "over,fragment",
    [({"artifact_sha256": "a" * 16}, "A PATH IS NOT IDENTITY"),
     ({"artifact_sha256": "A" * 64}, "64 lowercase hexadecimal"),
     ({"release_id": "2026 08"}, "no whitespace"),
     ({"coordinate_context": "GRCh38"}, "must state whether")],
    ids=["short-digest", "uppercase", "spaced-release", "context-as-string"])
def test_an_identity_refuses(over, fragment):
    with pytest.raises(SourceError) as exc:
        identity(**over)
    assert fragment in str(exc.value)


@pytest.mark.parametrize(
    "over,fragment",
    [({"retrieved_at": "2026-08-01"}, "YYYY-MM-DDTHH:MM:SSZ"),
     ({"observed_row_count": -1}, "non-negative integer"),
     ({"observed_row_count": True}, "non-negative integer"),
     ({"origin_locator": ""}, "non-empty string or None")],
    ids=["unformatted", "negative", "bool", "empty-origin"])
def test_provenance_refuses(over, fragment):
    with pytest.raises(SourceError) as exc:
        provenance(**over)
    assert fragment in str(exc.value)
