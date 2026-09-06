"""Evidence that was written must come back, and a bad record must NOT.

DRIFT-1 Phase 1 (P1-i). Created 2026-09-06.

WHY
---
MEASURED 2026-09-06 at 85d0247, across all 1,072 tracked Python files by parse
tree: the seven source-kernel types have ZERO construction sites under `src/`,
and `provenance/serialization.py` -- which defines the digest primitives --
names no kernel type at all. Five types carried `as_record`; NOTHING anywhere
reconstructed an object from one.

So the chain the adopted plan requires

    artifact -> identity -> evidence -> persistence -> RELOAD -> consumer

had no reload link. This unit adds it, following the convention already proven
in `repository_records/archive_manifest.py` and
`transactions/repository_transaction.py` rather than inventing a second one.

WHAT MUST BE TRUE, AND WHY EACH MATTERS
---------------------------------------
Reload must not be a back door. Every `from_record` returns `cls(...)`, so
`__post_init__` runs and a hand-edited file cannot construct what fresh
construction refuses. Section 5 below is that claim, tested nine ways: a
truncated digest, an invented assembly on build-independent evidence, a
mixed-assembly manifest, an acquisition describing a different
materialization. `SourceArtifactKey.of` was hardened on 2026-09-01 precisely
because a permissive factory admits what a strict one refuses; a permissive
PARSER would reopen that door.

An undeclared key is refused, not ignored. ATTESTATION-SCHEMA-DRIFT-1 happened
because nine documents were hand-built as dictionaries and each installer
added what it had learned.

REFUSALS ASSERT THE MESSAGE
---------------------------
`test_archive_manifest.py` records why: two of its own adversarial cases were
built by REUSING an entry, so an earlier check fired first -- they refused and
proved nothing about the invariants they named. Every negative case here is
constructed so ONLY the invariant under test can fire, and `refuses()` asserts
the message text.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.provenance.artifact import ArtifactKind
from genomic_variant_classifier.provenance.coordinate import (
    CoordinateContext,
    CoordinateError,
)
from genomic_variant_classifier.provenance.source import (
    SOURCE_EVIDENCE_SCHEMA,
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


def a_key(product=None):
    return SourceArtifactKey.of("clinvar", ArtifactKind.VCF, product)


def an_identity(**over):
    kw = {"key": a_key(), "release_id": "2026-08",
          "coordinate_context": CoordinateContext.assembly("GRCh38"),
          "artifact_sha256": "a" * 64}
    kw.update(over)
    return SourceArtifactIdentity(**kw)


def a_dependency(**over):
    kw = {"identity": an_identity(),
          "roles": frozenset({SourceRole.OBSERVATION})}
    kw.update(over)
    return SourceDependency(**kw)


def a_provenance(**over):
    kw = {"retrieved_at": "2026-09-06T07:31:48Z", "observed_row_count": 10,
          "origin_locator": None, "transport": None}
    kw.update(over)
    return SourceRetrievalProvenance(**kw)


def an_acquisition(**over):
    kw = {"identity": an_identity(), "provenance": a_provenance()}
    kw.update(over)
    return SourceAcquisition(**kw)


def a_manifest(**over):
    kw = {"evidence": SourceEvidenceManifest.of([a_dependency()]),
          "acquisitions": (an_acquisition(),)}
    kw.update(over)
    return SourceManifest(**kw)


def without(record, key):
    trimmed = dict(record)
    del trimmed[key]
    return trimmed


def refuses(fn, fragment):
    """Assert the refusal fires on the invariant it CLAIMS to test."""
    with pytest.raises((SourceError, CoordinateError)) as exc:
        fn()
    assert fragment in str(exc.value), (
        "refused, but on the WRONG check.\n  expected the message to contain: "
        "{!r}\n  actual: {}".format(fragment, exc.value))


# ---------------------------------------------------------------------------
# 1. The round trip, every type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build,cls", [
    (lambda: a_key("pc_transcripts"), SourceArtifactKey),
    (lambda: CoordinateContext.assembly("GRCh37"), CoordinateContext),
    (lambda: CoordinateContext.build_independent(), CoordinateContext),
    (lambda: an_identity(), SourceArtifactIdentity),
    (lambda: a_provenance(origin_locator="x", transport="https"),
     SourceRetrievalProvenance),
    (lambda: a_dependency(roles=frozenset({SourceRole.OBSERVATION,
                                           SourceRole.LABEL})),
     SourceDependency),
    (lambda: an_acquisition(), SourceAcquisition),
    (lambda: SourceEvidenceManifest.of([a_dependency()]),
     SourceEvidenceManifest),
    (lambda: a_manifest(), SourceManifest),
])
def test_every_type_round_trips_through_its_record(build, cls):
    obj = build()
    assert cls.from_record(obj.as_record()) == obj


def test_the_manifest_round_trips_through_BYTES():
    """Object equality and BYTE equality are different claims.

    An object round trip can pass while a field is reconstructed from a
    default that `render` then re-emits.
    """
    m = a_manifest()
    rendered = m.render()
    assert rendered == m.render(), "two renders of one manifest differ"
    assert SourceManifest.parse(rendered) == m
    assert SourceManifest.parse(rendered).render() == rendered


def test_the_rendered_bytes_are_ascii_and_carry_no_platform_newline():
    rendered = a_manifest().render()
    assert not any(b > 0x7F for b in rendered), "ensure_ascii should hold"
    assert b"\r\n" not in rendered


def test_the_evidence_digest_survives_the_round_trip():
    m = a_manifest()
    assert SourceManifest.parse(m.render()).evidence_digest == m.evidence_digest


def test_a_reordered_dependency_list_reloads_canonically():
    """`of()` sorts; `__post_init__` proves the result is ordered. Order is
    enforced in ONE place and reload does not become a second."""
    other = a_dependency(identity=an_identity(
        key=SourceArtifactKey.of("gnomad", ArtifactKind.VCF)))
    canonical = SourceEvidenceManifest.of([a_dependency(), other])
    shuffled = {"dependencies": list(reversed(
        canonical.as_record()["dependencies"]))}
    assert SourceEvidenceManifest.from_record(shuffled).dependencies == \
        canonical.dependencies


# ---------------------------------------------------------------------------
# 2. An undeclared key is refused at EVERY level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,build", [
    (SourceArtifactKey, a_key),
    (SourceArtifactIdentity, an_identity),
    (SourceRetrievalProvenance, a_provenance),
    (SourceDependency, a_dependency),
    (SourceAcquisition, an_acquisition),
    (SourceEvidenceManifest,
     lambda: SourceEvidenceManifest.of([a_dependency()])),
    (SourceManifest, a_manifest),
    (CoordinateContext, lambda: CoordinateContext.assembly("GRCh38")),
])
def test_an_undeclared_key_is_refused(cls, build):
    """The drift mechanism itself: each writer added what it had learned."""
    refuses(lambda: cls.from_record(dict(build().as_record(), extra_field=1)),
            "undeclared key")


@pytest.mark.parametrize("cls,build,key", [
    (SourceArtifactKey, a_key, "product"),
    (SourceArtifactIdentity, an_identity, "release_id"),
    (SourceRetrievalProvenance, a_provenance, "transport"),
    (SourceDependency, a_dependency, "roles"),
    (SourceAcquisition, an_acquisition, "provenance"),
    (SourceEvidenceManifest,
     lambda: SourceEvidenceManifest.of([a_dependency()]), "dependencies"),
    (SourceManifest, a_manifest, "acquisitions"),
    (CoordinateContext, lambda: CoordinateContext.assembly("GRCh38"),
     "identifier"),
])
def test_a_missing_key_is_refused_rather_than_defaulted(cls, build, key):
    """A default would decide a scientific question inside a parser."""
    refuses(lambda: cls.from_record(without(build().as_record(), key)),
            "missing")


def test_a_non_object_record_is_refused():
    refuses(lambda: SourceArtifactKey.from_record(["clinvar", "vcf", ""]),
            "must be an object")


# ---------------------------------------------------------------------------
# 3. Vocabulary
# ---------------------------------------------------------------------------

def test_an_unknown_artifact_kind_is_refused():
    refuses(lambda: SourceArtifactKey.from_record(
        dict(a_key().as_record(), artifact_kind="not_a_kind")),
        "unknown artifact kind")


def test_an_unknown_role_is_refused():
    refuses(lambda: SourceDependency.from_record(
        dict(a_dependency().as_record(), roles=["not_a_role"])),
        "unrecognised role vocabulary")


def test_an_unknown_coordinate_kind_is_refused():
    refuses(lambda: CoordinateContext.from_record(
        {"kind": "not_a_kind", "identifier": None}),
        "unrecognised coordinate kind")


def test_duplicate_roles_are_refused_because_a_set_cannot_express_them():
    refuses(lambda: SourceDependency.from_record(
        dict(a_dependency().as_record(), roles=["observation", "observation"])),
        "contain a duplicate")


# ---------------------------------------------------------------------------
# 4. Schema identity and shape
# ---------------------------------------------------------------------------

def test_the_version_comes_from_the_SCHEMA_OBJECT_not_a_literal():
    """EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1: the domain was bumped to v4 and
    an embedded literal was left at 3. Two writable declarations for one
    semantic version permit divergence BY CONSTRUCTION."""
    assert a_manifest().as_record()["schema_version"] == \
        SOURCE_EVIDENCE_SCHEMA.version


def test_a_foreign_schema_name_is_refused():
    refuses(lambda: SourceManifest.from_record(
        dict(a_manifest().as_record(), schema="gvc.something-else")),
        "expected")


def test_an_unjudged_schema_version_is_refused():
    """A v4 payload must be REFUSED, not read as v5. v4 historically described
    records carrying `schema_version: 3`, frozen at
    tests/fixtures/source_evidence_epoch_v4/epoch.json."""
    refuses(lambda: SourceManifest.from_record(
        dict(a_manifest().as_record(),
             schema_version=SOURCE_EVIDENCE_SCHEMA.version - 1)),
        "judges version")


def test_malformed_json_is_refused_as_a_domain_error():
    refuses(lambda: SourceManifest.parse(b"{not json"), "not valid JSON")


def test_a_non_object_payload_is_refused():
    refuses(lambda: SourceManifest.parse(b"[]"), "must be an object")


def test_parse_refuses_a_str_rather_than_encoding_it_silently():
    refuses(lambda: SourceManifest.parse("{}"), "parse takes bytes")


@pytest.mark.parametrize("build,fragment", [
    (lambda: SourceEvidenceManifest.from_record({"dependencies": {}}),
     "carries a LIST"),
    (lambda: SourceManifest.from_record(
        dict(a_manifest().as_record(), acquisitions={})), "carries a LIST"),
    (lambda: SourceDependency.from_record(
        dict(a_dependency().as_record(), roles="observation")), "sorted LIST"),
])
def test_a_collection_field_must_be_a_list(build, fragment):
    refuses(build, fragment)


# ---------------------------------------------------------------------------
# 5. RELOAD IS NOT A BACK DOOR -- the load-bearing group
# ---------------------------------------------------------------------------

def test_a_truncated_digest_is_refused_by_the_CONSTRUCTOR():
    """A PATH IS NOT IDENTITY, and a prefix is not a digest."""
    refuses(lambda: SourceArtifactIdentity.from_record(
        dict(an_identity().as_record(), artifact_sha256="a" * 16)),
        "expected 64 lowercase hexadecimal")


def test_build_independent_evidence_may_not_carry_an_assembly():
    """A UniProt accession has no genomic position; recording one would be a
    claim about coordinates that do not exist."""
    refuses(lambda: CoordinateContext.from_record(
        {"kind": "build_independent", "identifier": "GRCh38"}),
        "must not INVENT an assembly")


def test_a_genomic_context_without_an_assembly_is_refused():
    refuses(lambda: CoordinateContext.from_record(
        {"kind": "genomic_assembly", "identifier": None}), "must be one of")


def test_a_release_identifier_with_whitespace_is_refused():
    refuses(lambda: SourceArtifactIdentity.from_record(
        dict(an_identity().as_record(), release_id="2026 08")),
        "no whitespace or separators")


def test_a_dependency_with_no_role_is_refused():
    refuses(lambda: SourceDependency.from_record(
        dict(a_dependency().as_record(), roles=[])), "at least one role")


def test_an_empty_evidence_manifest_is_refused():
    """An empty manifest would digest to a constant and make every
    representation compare equal on its sources."""
    refuses(lambda: SourceEvidenceManifest.from_record({"dependencies": []}),
            "at least one dependency")


def test_a_mixed_assembly_manifest_is_refused_on_RELOAD():
    """Coordinates from different assemblies are not comparable, and a join
    across them would be silently wrong."""
    grch37 = a_dependency(identity=an_identity(
        key=SourceArtifactKey.of("gnomad", ArtifactKind.VCF),
        coordinate_context=CoordinateContext.assembly("GRCh37")))
    refuses(lambda: SourceEvidenceManifest.from_record(
        {"dependencies": [a_dependency().as_record(), grch37.as_record()]}),
        "mixes genome assemblies")


def test_a_duplicate_artifact_key_is_refused_on_RELOAD():
    record = a_dependency().as_record()
    refuses(lambda: SourceEvidenceManifest.from_record(
        {"dependencies": [record, dict(record)]}), "more than once")


def test_an_acquisition_of_a_DIFFERENT_MATERIALIZATION_is_refused():
    """SOURCE-ACQUISITION-KEY-ONLY-MATCH-1: a JULY retrieval record satisfied
    AUGUST evidence because only the KEY was compared. Reload must not reopen
    that door."""
    other = an_acquisition(identity=an_identity(release_id="2026-09"))
    refuses(lambda: SourceManifest.from_record(
        dict(a_manifest().as_record(), acquisitions=[other.as_record()])),
        "DIFFERENT MATERIALIZATION")


def test_a_boolean_row_count_is_refused():
    """`True == 1` in Python, so a boolean would silently become a count."""
    refuses(lambda: SourceRetrievalProvenance.from_record(
        dict(a_provenance().as_record(), observed_row_count=True)),
        "non-negative integer")


def test_a_malformed_retrieval_timestamp_is_refused():
    refuses(lambda: SourceRetrievalProvenance.from_record(
        dict(a_provenance().as_record(), retrieved_at="2026-09-06")),
        "YYYY-MM-DDTHH:MM:SSZ")


# ---------------------------------------------------------------------------
# 6. The empty product means ABSENCE, and only absence
# ---------------------------------------------------------------------------

def test_an_empty_product_reloads_as_None_and_not_as_a_product():
    """`canonical_key` has fixed arity and renders absence as "". The
    constructor refuses an empty product, so the two cannot collide."""
    assert SourceArtifactKey.from_record(a_key().as_record()).product is None


def test_a_real_product_survives_the_round_trip():
    reloaded = SourceArtifactKey.from_record(a_key("pc_transcripts").as_record())
    assert reloaded.product == "pc_transcripts"


def test_an_empty_product_is_still_refused_at_construction():
    refuses(lambda: SourceArtifactKey.of("clinvar", ArtifactKind.VCF, ""),
            "an empty product is forbidden")
