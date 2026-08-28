"""Source identity is stable under irrelevant events and moves under relevant ones.

DRIFT-1 Phase 1B.1. Created 2026-08-27.

IDENTITY KERNEL
---------------
    SourceArtifactIdentity

    equal iff   source, release identifier, coordinate context, artifact bytes
    ignores     retrieval timestamp, retrieval location, row-count observation

THE DEFECT THIS GUARDS
----------------------
MEASURED against the previous implementation: two byte-identical downloads of
one release, differing only in `retrieved_at`, produced DIFFERENT manifest
digests -- and `differing_releases` compared the whole record, so the same
re-download was reported as a release change at the interpretation layer too.

Repairing the digest alone would have left the second half standing. Both are
tested here, and `source_deltas` now takes an EVIDENCE manifest so retrieval
time is structurally unreachable rather than merely unread.

WHY INVARIANCE AND SENSITIVITY BOTH
-----------------------------------
A test that only proves "X changes the identity" is satisfied by an identity
that changes for everything. A test that only proves "Y does not" is satisfied
by one that never changes. The pair is what pins the kernel.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.monitoring.drift import (
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceDelta,
    SourceDeltaKind,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
    differing_releases,
    source_deltas,
)


def identity(**over):
    kw = dict(source="ClinVar", release_id="2026-08", genome_build="GRCh38",
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
    return SourceEvidenceManifest.of(deps or (
        dependency(identity(), SourceRole.OBSERVATION, SourceRole.LABEL),
        dependency(identity(source="dbNSFP", release_id="4.7a",
                            artifact_sha256="b" * 64), SourceRole.ANNOTATION)))


def manifest(ev=None, *acqs):
    return SourceManifest(evidence=ev or evidence(), acquisitions=tuple(acqs))


# ---------------------------------------------------------------------------
# 1. INVARIANCE -- nuisance perturbations must NOT move identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [{"retrieved_at": "2026-08-02T00:00:00Z"},
     {"observed_row_count": 4_400_001},
     {"origin_locator": "https://elsewhere.example/clinvar.parquet"},
     {"transport": "rclone"}],
    ids=["retrieved_at", "row_count", "origin", "transport"])
def test_acquisition_facts_do_not_move_evidence_identity(over):
    """The kernel: how we obtained bytes says nothing about which bytes."""
    a = manifest(evidence(), SourceAcquisition(identity(), provenance()))
    b = manifest(evidence(), SourceAcquisition(identity(), provenance(**over)))
    assert a.evidence_digest == b.evidence_digest
    assert differing_releases(a, b) == ()


def test_a_redownload_preserves_evidence_identity_but_not_the_record():
    """All three predicates, and each says something different.

    Making `__eq__` ignore provenance would reverse the original conflation
    rather than repair it: the acquisition event is a real fact.
    """
    monday = SourceAcquisition(identity(), provenance())
    tuesday = SourceAcquisition(
        identity(), provenance(retrieved_at="2026-08-02T00:00:00Z"))
    assert monday.identity == tuesday.identity
    assert monday.provenance != tuesday.provenance
    assert monday != tuesday


def test_dependency_role_order_does_not_move_identity():
    """Roles are a SET; declaring them in another order is the same claim."""
    a = dependency(identity(), SourceRole.OBSERVATION, SourceRole.LABEL)
    b = dependency(identity(), SourceRole.LABEL, SourceRole.OBSERVATION)
    assert a == b
    assert SourceEvidenceManifest.of((a,)).digest == \
        SourceEvidenceManifest.of((b,)).digest


def test_manifest_member_order_does_not_move_identity():
    x = dependency(identity(), SourceRole.OBSERVATION)
    y = dependency(identity(source="dbNSFP", release_id="4.7a",
                            artifact_sha256="b" * 64), SourceRole.ANNOTATION)
    assert SourceEvidenceManifest.of((x, y)) == SourceEvidenceManifest.of((y, x))


# ---------------------------------------------------------------------------
# 2. SENSITIVITY -- defining perturbations MUST move identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [{"source": "ClinVarPlus"},
     {"release_id": "2026-09"},
     {"genome_build": "GRCh37"},
     {"artifact_sha256": "c" * 64}],
    ids=["source", "release_id", "genome_build", "artifact_bytes"])
def test_defining_facts_move_evidence_identity(over):
    """Each in isolation: every other field held constant.

    An earlier test in this package changed two fields together and proved
    only "something matters", not which.
    """
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(identity(**over)),))
    assert a.digest != b.digest


def test_adding_a_role_moves_evidence_identity():
    """A source used as both observation and label is a different analysis."""
    a = SourceEvidenceManifest.of((dependency(identity(), SourceRole.OBSERVATION),))
    b = SourceEvidenceManifest.of((dependency(
        identity(), SourceRole.OBSERVATION, SourceRole.LABEL),))
    assert a.digest != b.digest


def test_the_digest_is_domain_separated():
    """A 64-character digest must not be interchangeable across kinds.

    Two digests of different KINDS are both 64 lowercase hex characters, and
    version 3 of the attestation schema types every digest field as exactly
    that. Domain separation is about TYPE CONFUSION, not collisions.
    """
    from genomic_variant_classifier.monitoring.drift._digest import (
        canonical_json, domain_digest)
    payload = {"schema_version": 2, "dependencies": []}
    assert domain_digest("a-v1", payload) != domain_digest("b-v1", payload)
    assert domain_digest("a-v1", payload) != domain_digest("a-v2", payload)
    with pytest.raises(ValueError):
        domain_digest("no-version-suffix", payload)
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


# ---------------------------------------------------------------------------
# 3. What a source record may not be
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over,fragment",
    [({"artifact_sha256": "a" * 16}, "A PATH IS NOT IDENTITY"),
     ({"artifact_sha256": "A" * 64}, "64 lowercase hexadecimal"),
     ({"genome_build": "hg38"}, "NOT interchangeable"),
     ({"release_id": "2026 08"}, "no whitespace"),
     ({"source": "9lives"}, "expected a name")],
    ids=["short-digest", "uppercase", "unknown-build", "spaced", "numeric"])
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
    ids=["unformatted-time", "negative", "bool", "empty-origin"])
def test_provenance_refuses(over, fragment):
    with pytest.raises(SourceError) as exc:
        provenance(**over)
    assert fragment in str(exc.value)


def test_a_dependency_must_declare_a_role():
    with pytest.raises(SourceError) as exc:
        SourceDependency(identity=identity(), roles=frozenset())
    assert "at least one role" in str(exc.value)


def test_one_source_twice_is_refused_and_says_to_use_roles():
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of((
            dependency(identity(), SourceRole.OBSERVATION),
            dependency(identity(artifact_sha256="c" * 64), SourceRole.LABEL)))
    assert "several roles" in str(exc.value)


def test_mixed_genome_builds_are_refused():
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of((
            dependency(identity(), SourceRole.OBSERVATION),
            dependency(identity(source="gnomAD", release_id="v4.1",
                                genome_build="GRCh37",
                                artifact_sha256="b" * 64),
                       SourceRole.ANNOTATION)))
    assert "silently wrong" in str(exc.value)


def test_an_empty_manifest_is_refused():
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest.of(())
    assert "at least one dependency" in str(exc.value)


def test_a_manifest_built_out_of_canonical_order_is_refused():
    """Order is enforced in ONE place, so it can be shown to matter."""
    x = dependency(identity(), SourceRole.OBSERVATION)
    y = dependency(identity(source="dbNSFP", release_id="4.7a",
                            artifact_sha256="b" * 64), SourceRole.ANNOTATION)
    lo, hi = sorted((x, y), key=lambda d: d.canonical_key)
    with pytest.raises(SourceError) as exc:
        SourceEvidenceManifest(dependencies=(hi, lo))
    assert "canonical order" in str(exc.value)


def test_an_acquisition_of_an_undeclared_source_is_refused():
    """Recording how something was obtained that was never used is nothing."""
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(
                           identity(source="COSMIC", release_id="v99"),
                           provenance()),))
    assert "does not declare" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. Deltas name the KIND of movement
# ---------------------------------------------------------------------------

def test_a_release_move_is_named_as_such():
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(
        identity(release_id="2026-09", artifact_sha256="c" * 64)),))
    (delta,) = source_deltas(a, b)
    assert delta.kind is SourceDeltaKind.RELEASE_MOVED
    assert "2026-08 -> 2026-09" in delta.describe()


def test_same_release_different_bytes_is_NOT_ordinary_movement():
    """The case that most needs its own name.

    Same label, different bytes means an upstream replacement, a corrupted
    download, or non-reproducible packaging. Calling it RELEASE_MOVED would
    hide a serious problem inside a routine one.
    """
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(
        identity(artifact_sha256="c" * 64)),))
    (delta,) = source_deltas(a, b)
    assert delta.kind is SourceDeltaKind.ARTIFACT_CHANGED_UNDER_SAME_RELEASE
    assert "UNCHANGED but bytes differ" in delta.describe()


@pytest.mark.parametrize(
    "over,kind",
    [({"genome_build": "GRCh37"}, SourceDeltaKind.GENOME_BUILD_CHANGED),
     ({"release_id": "2026-09", "artifact_sha256": "c" * 64},
      SourceDeltaKind.RELEASE_MOVED)],
    ids=["build", "release"])
def test_delta_kinds_are_distinguished(over, kind):
    a = SourceEvidenceManifest.of((dependency(identity()),))
    b = SourceEvidenceManifest.of((dependency(identity(**over)),))
    (delta,) = source_deltas(a, b)
    assert delta.kind is kind


def test_added_and_removed_sources_are_named():
    one = SourceEvidenceManifest.of((dependency(identity()),))
    two = evidence()
    added = source_deltas(one, two)
    assert [d.kind for d in added] == [SourceDeltaKind.SOURCE_ADDED]
    assert added[0].source == "dbNSFP"
    removed = source_deltas(two, one)
    assert [d.kind for d in removed] == [SourceDeltaKind.SOURCE_REMOVED]


def test_a_delta_that_describes_a_change_requires_both_identities():
    """A half-populated delta describes a state that cannot exist."""
    with pytest.raises(ValueError):
        SourceDelta("X", SourceDeltaKind.RELEASE_MOVED, identity(), None)
    with pytest.raises(ValueError):
        SourceDelta("X", SourceDeltaKind.SOURCE_ADDED, identity(), identity())


def test_the_digest_cannot_REACH_retrieval_time():
    """Structural, not remembered.

    `SourceEvidenceManifest.digest` serialises dependencies, which serialise
    identities -- and `SourceArtifactIdentity` has no `retrieved_at`. The
    digest cannot include retrieval time because retrieval time is not on the
    object it serialises.

    A sabotage that adds a constant to the record shifts every digest equally
    and proves nothing; this asserts the reachable field set directly.
    """
    ident = identity()
    assert "retrieved_at" not in ident.as_record()
    assert "observed_row_count" not in ident.as_record()
    assert set(ident.as_record()) == {
        "source", "release_id", "genome_build", "artifact_sha256"}
    dep_record = dependency(ident, SourceRole.OBSERVATION).as_record()
    assert set(dep_record) == {"identity", "roles"}


def test_source_deltas_cannot_be_given_acquisitions_at_all():
    """The strongest form of the repair.

    `source_deltas` takes EVIDENCE manifests. A caller holding a
    `SourceManifest` -- which does carry acquisitions -- must reach through
    `.evidence` to call it, so retrieval time is not merely unread but
    unreachable from the argument.
    """
    ev = evidence()
    full = manifest(ev, SourceAcquisition(identity(), provenance()))
    assert source_deltas(ev, ev) == ()
    with pytest.raises(AttributeError):
        source_deltas(full, full)


def test_an_added_identity_field_cannot_pass_unclassified(monkeypatch):
    """Guards a branch that is UNREACHABLE with today's four fields.

    It fires only when two identities differ while source, release_id,
    genome_build and artifact_sha256 all match -- and `source` is the dict key,
    so two identities always share it. With four fields there is no fifth way
    to differ.

    This repository has ruled on unreachable defences: `suite_transition.py`
    DELETED three, and `publish()`'s re-parse survived only once a reachable
    case was found. The case here is the future the branch guards -- a field
    added without a matching delta kind -- so it is constructed directly.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class WithNewField(SourceArtifactIdentity):
        curation_tier: str = "gold"

    a = WithNewField(source="ClinVar", release_id="2026-08",
                     genome_build="GRCh38", artifact_sha256="a" * 64)
    b = WithNewField(source="ClinVar", release_id="2026-08",
                     genome_build="GRCh38", artifact_sha256="a" * 64,
                     curation_tier="silver")
    assert a != b, "the fixture must actually differ"
    assert a.canonical_key == b.canonical_key, (
        "the new field must be invisible to the existing comparisons, which "
        "is exactly the condition the guard exists for")

    ref = SourceEvidenceManifest.of((dependency(a, SourceRole.OBSERVATION),))
    cand = SourceEvidenceManifest.of((dependency(b, SourceRole.OBSERVATION),))
    with pytest.raises(RuntimeError) as exc:
        source_deltas(ref, cand)
    assert "UNCLASSIFIED" in str(exc.value)


def test_identical_manifests_yield_no_deltas():
    assert source_deltas(evidence(), evidence()) == ()
    assert differing_releases(manifest(), manifest()) == ()
