"""A retrieval record must describe the bytes the evidence actually names.

Phase 1C Unit 3A+. Created 2026-09-02.

WHAT WAS WRONG
--------------
`SOURCE-ACQUISITION-KEY-ONLY-MATCH-1`, measured 2026-09-02 BY CONSTRUCTION.

`SourceManifest.__post_init__` matched an acquisition to the evidence on
`a.identity.key`. A `SourceArtifactKey` is `(source, artifact_kind, product)`;
`release_id`, `coordinate_context` and `artifact_sha256` are NOT in it.

So this was accepted:

    evidence     clinvar/primary_release  release 2026-08  digest aaaa...
    acquisition  clinvar/primary_release  release 2026-07  digest bbbb...

A July retrieval record satisfied August evidence. And this was accepted too:

    evidence     clinvar/primary_release  GRCh38
    acquisition  clinvar/primary_release  GRCh37

`CoordinateContext` exists precisely because "GRCh37 and GRCh38 coordinates are
NOT interchangeable, and comparing across them would pair unrelated loci" --
its own error message says so -- and the acquisition match could not see it.

A `SourceManifest` answers "how was this evidence obtained". It could answer
with a record describing different bytes, a different release and a different
genome build, and report nothing.

WHY THE ORIGINAL WAS WRITTEN THAT WAY
-------------------------------------
It is not careless. `SourceEvidenceManifest` enforces one dependency per key,
so `evidence.keys` is a natural uniqueness set and matching against it reads as
correct. The defect is that UNIQUENESS WITHIN EVIDENCE and CORRESPONDENCE
BETWEEN AN ACQUISITION AND ITS EVIDENCE are different questions, and a set of
keys answers only the first.

TWO MISTAKES, TWO MESSAGES
--------------------------
"this artifact was never used" and "this artifact was used, but you have
described a different materialization of it" are different errors. The first
message is preserved VERBATIM, because
`test_an_acquisition_of_an_undeclared_artifact_is_refused` asserts its wording
and a caller may be matching on it.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.provenance import (
    ArtifactKind,
    CoordinateContext,
    SourceAcquisition,
    SourceArtifactIdentity,
    SourceArtifactKey,
    SourceDependency,
    SourceError,
    SourceEvidenceManifest,
    SourceManifest,
    SourceRetrievalProvenance,
    SourceRole,
)

GRCH38 = CoordinateContext.assembly("GRCh38")
GRCH37 = CoordinateContext.assembly("GRCh37")
CLINVAR = SourceArtifactKey("clinvar", ArtifactKind.PRIMARY_RELEASE)


def identity(**over):
    kw = dict(key=CLINVAR, release_id="2026-08", coordinate_context=GRCH38,
              artifact_sha256="a" * 64)
    kw.update(over)
    return SourceArtifactIdentity(**kw)


def provenance(**over):
    kw = dict(retrieved_at="2026-08-01T00:00:00Z",
              observed_row_count=4_400_000,
              origin_locator="fixture://clinvar", transport="fixture")
    kw.update(over)
    return SourceRetrievalProvenance(**kw)


def evidence(*idents):
    return SourceEvidenceManifest.of(tuple(
        SourceDependency(identity=i, roles=frozenset({SourceRole.OBSERVATION}))
        for i in (idents or (identity(),))))


# ---------------------------------------------------------------------------
# 1. what must still be ACCEPTED
# ---------------------------------------------------------------------------

def test_the_matching_acquisition_is_accepted():
    m = SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(identity(),
                                                       provenance()),))
    assert len(m.acquisitions) == 1


@pytest.mark.parametrize(
    "over",
    [{"retrieved_at": "2026-08-02T00:00:00Z"},
     {"observed_row_count": 4_400_001},
     {"origin_locator": "https://elsewhere.example/clinvar.parquet"},
     {"transport": "rclone"}],
    ids=["retrieved_at", "row_count", "origin", "transport"])
def test_a_RETRIEVAL_fact_still_does_not_affect_the_match(over):
    """The sensitivity half. Tightening the match must not start refusing a
    re-download, which changes WHEN and HOW but not WHICH BYTES."""
    m = SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(identity(),
                                                       provenance(**over)),))
    assert m.evidence_digest == evidence().digest


def test_several_acquisitions_of_ONE_identity_are_accepted():
    """A re-download is a second record of the same bytes, not a conflict."""
    m = SourceManifest(evidence=evidence(), acquisitions=(
        SourceAcquisition(identity(), provenance()),
        SourceAcquisition(identity(),
                          provenance(retrieved_at="2026-08-02T00:00:00Z")),))
    assert len(m.acquisitions) == 2


def test_an_empty_acquisition_tuple_is_accepted():
    """Evidence without retrieval records is incomplete, not invalid."""
    assert SourceManifest(evidence=evidence()).acquisitions == ()


# ---------------------------------------------------------------------------
# 2. what must now be REFUSED
# ---------------------------------------------------------------------------

def test_a_DIFFERENT_RELEASE_is_refused():
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(
                identity(release_id="2026-07", artifact_sha256="b" * 64),
                provenance()),))
    msg = str(exc.value)
    assert "DIFFERENT MATERIALIZATION" in msg
    assert "release_id '2026-08' declared vs '2026-07' acquired" in msg
    assert "declared vs" in msg and "acquired" in msg


def test_a_DIFFERENT_ASSEMBLY_is_refused():
    """The serious case. GRCh37 and GRCh38 are not interchangeable."""
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(identity(coordinate_context=GRCH37),
                              provenance()),))
    msg = str(exc.value)
    assert "DIFFERENT MATERIALIZATION" in msg
    assert "coordinates GRCh38 declared vs GRCh37 acquired" in msg


def test_a_DIFFERENT_DIGEST_ALONE_is_refused():
    """Same authority, same release, same build -- different bytes.

    This is the case a human is most likely to wave through, and the one where
    a silent acceptance does the most damage: the manifest would claim these
    bytes were obtained when they never were.
    """
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(identity(artifact_sha256="c" * 64),
                              provenance()),))
    msg = str(exc.value)
    assert "digest aaaaaaaaaaaaaaaa... declared vs cccccccccccccccc... acquired" in msg


def test_the_message_names_ONLY_the_fields_that_differ():
    """Reporting every field would bury the one that matters."""
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(identity(coordinate_context=GRCH37),
                              provenance()),))
    msg = str(exc.value)
    assert "coordinates" in msg
    assert "release_id" not in msg, "release_id is identical and must not appear"
    assert "digest" not in msg, "the digest is identical and must not appear"


def test_a_FRESH_but_EQUAL_coordinate_is_not_reported_as_differing():
    """Equality, not object identity, throughout.

    MEASURED BY SABOTAGE 2026-09-02: changing the helper's coordinate
    comparison from `!=` to `is not` changed NO test, because every test here
    reused the module-level GRCH38 singleton, so `is` and `==` coincided.

    A caller constructing `CoordinateContext.assembly("GRCh38")` freshly gets
    an EQUAL but DISTINCT object. Under `is not`, the message would claim the
    coordinates differ when they do not -- sending a reader to look at the one
    field that is actually correct.
    """
    fresh = CoordinateContext.assembly("GRCh38")
    assert fresh == GRCH38, "the premise: equal"
    assert fresh is not GRCH38, "the premise: not the same object"

    # A release_id BUILT AT RUNTIME rather than written as a literal. Python
    # interns short string literals, so `"2026-08" is "2026-08"` holds and an
    # identity comparison would pass unnoticed -- exactly how sabotage case 10
    # went undetected until this line existed.
    same_release = "".join(("2026", "-", "08"))
    assert same_release == "2026-08"
    assert same_release is not identity().release_id

    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(
                identity(coordinate_context=fresh, release_id=same_release,
                         artifact_sha256="c" * 64), provenance()),))
    msg = str(exc.value)
    assert "digest" in msg, msg
    assert "coordinates" not in msg, (
        "the coordinate contexts are EQUAL and must not be reported as "
        "differing: {}".format(msg))
    assert "release_id" not in msg, (
        "the release identifiers are EQUAL and must not be reported as "
        "differing: {}".format(msg))


def test_a_FRESH_but_EQUAL_digest_is_not_reported_as_differing():
    """The third field, and the one that hid longest.

    MEASURED BY SABOTAGE 2026-09-02: changing the digest comparison to
    `is not` changed NO test. `"a" * 64` written as a literal expression is
    CONSTANT-FOLDED by CPython, so every call to `identity()` returns the SAME
    string object and `is` holds by accident.

    Building the digest at runtime defeats the folding and exercises the
    comparison the code actually needs.
    """
    folded = "a" * 64
    built = "".join("a" for _ in range(64))
    assert built == folded, "the premise: equal"
    assert built is not folded, "the premise: distinct objects"

    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(
                identity(coordinate_context=GRCH37, artifact_sha256=built),
                provenance()),))
    msg = str(exc.value)
    assert "coordinates" in msg, msg
    assert "digest" not in msg, (
        "the digests are EQUAL and must not be reported as differing: "
        "{}".format(msg))


def test_a_FRESH_but_EQUAL_identity_is_ACCEPTED():
    """The acceptance half of the same point.

    A manifest reconstructed from a record -- rather than holding the original
    objects -- must still match. Anything keyed on object identity would refuse
    a perfectly correct acquisition.
    """
    rebuilt = SourceArtifactIdentity(
        key=SourceArtifactKey("clinvar", ArtifactKind.PRIMARY_RELEASE),
        release_id="2026-08",
        coordinate_context=CoordinateContext.assembly("GRCh38"),
        artifact_sha256="a" * 64)
    assert rebuilt == identity()
    assert rebuilt is not identity()
    m = SourceManifest(evidence=evidence(),
                       acquisitions=(SourceAcquisition(rebuilt, provenance()),))
    assert len(m.acquisitions) == 1


def test_an_UNDECLARED_artifact_keeps_its_ORIGINAL_message():
    """Two mistakes, two messages -- and this wording is depended upon.

    `test_an_acquisition_of_an_undeclared_artifact_is_refused` in
    `test_source_release.py` asserts this fragment. Rewriting it into
    something tidier would break a test for no scientific gain.
    """
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=evidence(), acquisitions=(
            SourceAcquisition(
                identity(key=SourceArtifactKey("cosmic",
                                               ArtifactKind.PRIMARY_RELEASE)),
                provenance()),))
    msg = str(exc.value)
    assert "does not declare" in msg
    assert "DIFFERENT MATERIALIZATION" not in msg, (
        "an artifact that was never used is not a materialization mismatch")


def test_the_two_refusals_are_DISTINGUISHABLE():
    """A caller must be able to tell which mistake they made."""
    def refuse(ident):
        with pytest.raises(SourceError) as exc:
            SourceManifest(evidence=evidence(), acquisitions=(
                SourceAcquisition(ident, provenance()),))
        return str(exc.value)

    never_used = refuse(identity(key=SourceArtifactKey(
        "cosmic", ArtifactKind.PRIMARY_RELEASE)))
    wrong_bytes = refuse(identity(artifact_sha256="c" * 64))
    assert never_used != wrong_bytes
    assert ("does not declare" in never_used) is True
    assert ("does not declare" in wrong_bytes) is False


# ---------------------------------------------------------------------------
# 3. the match uses IDENTITY, and the key alone is not enough
# ---------------------------------------------------------------------------

def test_the_key_ALONE_no_longer_admits_an_acquisition():
    """The defect, stated as an invariant.

    MEASURED 2026-09-02: two identities sharing one key differ in release and
    digest. Before the repair, `SourceManifest` accepted either against the
    other, because it compared keys.
    """
    august = identity()
    july = identity(release_id="2026-07", artifact_sha256="b" * 64)
    assert august.key == july.key, "the premise: one key"
    assert august != july, "the premise: two identities"
    with pytest.raises(SourceError):
        SourceManifest(evidence=evidence(august),
                       acquisitions=(SourceAcquisition(july, provenance()),))


def test_evidence_declaring_TWO_products_matches_each_exactly():
    """Widening the match must not confuse two artifacts of one authority."""
    tx = SourceArtifactIdentity(
        key=SourceArtifactKey("gencode", ArtifactKind.SEQUENCE_FASTA,
                              "transcripts"),
        release_id="release_50", coordinate_context=GRCH38,
        artifact_sha256="d" * 64)
    pc = SourceArtifactIdentity(
        key=SourceArtifactKey("gencode", ArtifactKind.SEQUENCE_FASTA,
                              "pc_transcripts"),
        release_id="release_50", coordinate_context=GRCH38,
        artifact_sha256="e" * 64)
    ev = evidence(tx, pc)
    m = SourceManifest(evidence=ev, acquisitions=(
        SourceAcquisition(tx, provenance()),
        SourceAcquisition(pc, provenance()),))
    assert len(m.acquisitions) == 2
    crossed = SourceArtifactIdentity(
        key=tx.key, release_id="release_50", coordinate_context=GRCH38,
        artifact_sha256="e" * 64)          # transcripts key, pc_transcripts bytes
    with pytest.raises(SourceError) as exc:
        SourceManifest(evidence=ev,
                       acquisitions=(SourceAcquisition(crossed, provenance()),))
    assert "DIFFERENT MATERIALIZATION" in str(exc.value)
