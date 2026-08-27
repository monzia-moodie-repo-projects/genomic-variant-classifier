"""A release manifest identifies the evidence, or it identifies nothing.

DRIFT-1 Phase 1B. Created 2026-08-27.

WHAT THIS GUARDS
----------------
MEASURED 2026-08-27: source-release identity was the ONE fact of eight with no
owner in this repository. A Layer-B scan reported one and was wrong -- it
matched `anchor_manifest_sha256` in `moe_identity.py`, a field about
mechanistic anchor sets.

The manifest exists so a distribution change can be attributed. Same ClinVar
variants, new dbNSFP release, CADD moves: the population did not drift, the
measurement process did, and only a complete manifest can tell those apart.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib

import pytest

from genomic_variant_classifier.monitoring.drift.source_release import (
    GENOME_BUILDS,
    SourceManifest,
    SourceManifestError,
    SourceRelease,
    differing_releases,
)


def rel(**over):
    kw = dict(source="ClinVar", release_id="2026-08", genome_build="GRCh38",
              artifact_sha256="a" * 64, row_count=4_400_000,
              retrieved_at="2026-08-01T00:00:00Z")
    kw.update(over)
    return SourceRelease(**kw)


def manifest(*releases):
    return SourceManifest.of(releases or (
        rel(),
        rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64),
        rel(source="dbNSFP", release_id="4.7a", artifact_sha256="c" * 64),
    ))


# ---------------------------------------------------------------------------
# 1. A release identifies bytes, not a path
# ---------------------------------------------------------------------------

def test_a_release_records_the_digest_of_what_was_actually_read():
    assert rel().artifact_sha256 == "a" * 64


@pytest.mark.parametrize(
    "over,fragment",
    [({"artifact_sha256": "a" * 16}, "A path is not identity"),
     ({"artifact_sha256": "A" * 64}, "64 lowercase hexadecimal"),
     ({"genome_build": "hg38"}, "NOT interchangeable"),
     ({"release_id": "2026 08"}, "no whitespace"),
     ({"release_id": ""}, "expected an identifier"),
     ({"source": "9lives"}, "expected a name"),
     ({"row_count": -1}, "non-negative integer"),
     ({"row_count": True}, "non-negative integer"),
     ({"retrieved_at": "2026-08-01"}, "YYYY-MM-DDTHH:MM:SSZ")],
    ids=["short-digest", "uppercase-digest", "unknown-build", "spaced-release",
         "empty-release", "numeric-source", "negative-rows", "bool-rows",
         "unformatted-time"])
def test_a_release_refuses(over, fragment):
    with pytest.raises(ValueError) as exc:
        rel(**over)
    assert fragment in str(exc.value)


def test_both_admitted_genome_builds_are_accepted():
    """The permissive direction: a guard that refuses everything is unusable."""
    for build in GENOME_BUILDS:
        assert rel(genome_build=build).genome_build == build


# ---------------------------------------------------------------------------
# 2. The manifest is a SET, and its digest is derived
# ---------------------------------------------------------------------------

def test_member_order_does_not_change_identity():
    """A manifest has no meaningful order; the set of releases is the fact."""
    a = rel()
    b = rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64)
    assert SourceManifest.of((a, b)).digest == SourceManifest.of((b, a)).digest


def test_every_field_of_a_release_reaches_the_digest():
    """Structural, so no single field can silently drop out.

    An earlier test changed `release_id` and `artifact_sha256` together, so
    removing `release_id` from the record left the digest still changing. It
    proved "something matters", not "the release identifier is part of the
    identity".
    """
    r = rel()
    record = r.as_record()
    for field in (r.source, r.release_id, r.genome_build, r.artifact_sha256,
                  str(r.row_count), r.retrieved_at):
        assert field in record, "{!r} is absent from the canonical record".format(field)
    assert record.count("\x1f") == 5, "expected six fields joined by five separators"


def test_only_the_release_identifier_changing_changes_the_digest():
    """Isolated: every other field held constant."""
    a = SourceManifest.of((rel(release_id="2026-08"),))
    b = SourceManifest.of((rel(release_id="2026-09"),))
    assert a.digest != b.digest


def test_a_manifest_constructed_out_of_order_is_refused():
    """Order is enforced in ONE place, so it can be shown to matter."""
    x = rel()
    y = rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64)
    lo, hi = sorted((x, y))
    with pytest.raises(SourceManifestError) as exc:
        SourceManifest(releases=(hi, lo))
    assert "canonical order" in str(exc.value)


def test_of_accepts_any_order_and_yields_EQUAL_manifests():
    """The permissive direction: `of()` is what callers use."""
    x = rel()
    y = rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64)
    assert SourceManifest.of((x, y)) == SourceManifest.of((y, x))


def test_the_digest_is_derived_not_stored():
    assert "digest" not in manifest().__dict__
    assert isinstance(type(manifest()).digest, property)


def test_any_one_release_moving_changes_the_digest():
    """The whole purpose: a dbNSFP bump must not be invisible."""
    before = manifest()
    after = SourceManifest.of((
        rel(),
        rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64),
        rel(source="dbNSFP", release_id="4.8a", artifact_sha256="d" * 64),
    ))
    assert before.digest != after.digest


def test_the_same_release_rebuilt_digests_identically():
    """Permissive direction. A rebuild must not look like a release change."""
    assert manifest().digest == manifest().digest
    assert len(manifest().digest) == 64


def test_a_field_separator_cannot_be_smuggled_into_a_field():
    """Two manifests must not collide through concatenation.

    Every field is validated against a pattern that excludes the separators,
    so the joined record has exactly one reading.
    """
    for bad in ("Clin\x1fVar", "Clin\x1eVar"):
        with pytest.raises(ValueError):
            rel(source=bad)


# ---------------------------------------------------------------------------
# 3. What a manifest may not be
# ---------------------------------------------------------------------------

def test_an_empty_manifest_is_refused():
    """An empty manifest digests to a constant, making every representation
    compare equal on its sources."""
    with pytest.raises(SourceManifestError) as exc:
        SourceManifest.of(())
    assert "at least one release" in str(exc.value)


def test_one_source_twice_is_refused():
    with pytest.raises(SourceManifestError) as exc:
        SourceManifest.of((rel(), rel(release_id="2026-09",
                                      artifact_sha256="e" * 64)))
    assert "ClinVar" in str(exc.value)


def test_mixed_genome_builds_are_refused():
    with pytest.raises(SourceManifestError) as exc:
        SourceManifest.of((rel(), rel(source="gnomAD", genome_build="GRCh37",
                                      artifact_sha256="b" * 64)))
    assert "silently wrong" in str(exc.value)


def test_a_list_is_refused_as_the_releases_field():
    with pytest.raises(SourceManifestError) as exc:
        SourceManifest(releases=[rel()])
    assert "must be a TUPLE" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. A refusal must name WHICH release moved
# ---------------------------------------------------------------------------

def test_differing_releases_names_the_source_that_moved():
    """"dbNSFP moved" is a scientific statement; "the manifests differ" is not."""
    after = SourceManifest.of((
        rel(),
        rel(source="gnomAD", release_id="v4.1", artifact_sha256="b" * 64),
        rel(source="dbNSFP", release_id="4.8a", artifact_sha256="d" * 64),
    ))
    assert differing_releases(manifest(), after) == ("dbNSFP",)


def test_differing_releases_reports_an_added_or_removed_source():
    fewer = SourceManifest.of((
        rel(), rel(source="gnomAD", release_id="v4.1",
                   artifact_sha256="b" * 64)))
    assert differing_releases(manifest(), fewer) == ("dbNSFP",)


def test_identical_manifests_differ_in_nothing():
    assert differing_releases(manifest(), manifest()) == ()


def test_release_of_names_the_manifest_when_a_source_is_absent():
    with pytest.raises(SourceManifestError) as exc:
        manifest().release_of("SpliceAI")
    assert "ClinVar" in str(exc.value)


def test_describe_carries_every_release():
    text = manifest().describe()
    for fragment in ("ClinVar@2026-08", "gnomAD@v4.1", "dbNSFP@4.7a",
                     "GRCh38", manifest().digest[:12]):
        assert fragment in text
