"""The archive index is a contract, and its refusals must ISOLATE.

ADR-0004 section G. Created 2026-08-23.

WHY
---
ATTESTATION-SCHEMA-DRIFT-1 happened because nine documents were hand-built as
dictionaries under one unchanging version. A preservation manifest is itself a
durable record, so introducing `gvc.installation-attestation-archive` without a
typed owner and adversarial controls would reproduce that defect one level up --
inside the unit preserving the evidence of having fixed it.

WHAT THIS PROVES
----------------
Not that a well-formed manifest is accepted; that is four tests. That every
malformed one is REFUSED, and refused ON THE INVARIANT IT VIOLATES.

That second clause is not pedantry. Measured 2026-08-23, while building this
file: two adversarial cases were constructed by REUSING an entry, so an earlier
check fired first. They refused, and proved nothing about the invariants they
named -- a check passing for the wrong reason, inside the matrix built to catch
checks passing for the wrong reasons.

Every negative case below therefore asserts the MESSAGE TEXT, not merely that
something raised. `refuses()` exists for exactly that.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.repository_records.archive_manifest import (
    SCHEMA,
    SCHEMA_VERSION,
    ArchiveEntry,
    ArchiveManifest,
    ArchiveManifestError,
)
from genomic_variant_classifier.repository_records.classification import (
    DisclosureClass,
    PreservationDisposition,
    ProvenanceRelation,
    RetentionClass,
)
from genomic_variant_classifier.repository_records.identity import (
    ArtifactInstance,
    RecordId,
    RecordIdentity,
    allocate_record_id,
)
from genomic_variant_classifier.repository_records.roles import (
    ArtifactRole,
    RecordsOntologyError,
)

ATT = ArtifactRole.INSTALLATION_ATTESTATION
ROOT = "records/attestations/installations/artifacts"

#: Three of the sixteen real attestations, measured 2026-08-23 at 8ff0ea3.
#: Real digests and sizes, so the fixtures cannot drift into fantasy.
MEASURED = (
    ("install-attestation-D0a-ADR-0001-20260821T233547Z.json", 1013,
     "96a1522f7769c5a5522b7d4dc7215be4271962ad6ff408107abae1c1e070662d",
     1, "b115bab"),
    ("install-attestation-ADR-0003-2026-08-22-20260822T044349Z.json", 1558,
     "1a8a4024a2f6b1d80b6001c2d7ccb2ad295f4d1d8a2dcf91d4680ceb12f33cb5",
     1, "e1a5297"),
    ("install-attestation-ADR-0004-2026-08-22-20260822T220407Z.json", 2088,
     "aa92006e7b685b7214921c76140279a51ff60410a30f7f41f7b5e319cc6f9749",
     2, "f567381"),
)


def an_entry(basename, *, aliases=None, digest="a" * 64, size=100,
             record_id=None, **over):
    kw = {
        "cited_by": ("aaaaaaa",),
        "disclosure": DisclosureClass.PUBLIC_VERBATIM,
        "preservation": PreservationDisposition.ADMITTED_VERBATIM,
        "provenance": (ProvenanceRelation.EMITTED_BY_INSTALLER,),
        "retention": RetentionClass.PERMANENT_EVIDENCE,
        "artifact_schema_version": 1,
    }
    kw.update(over)
    return ArchiveEntry(
        identity=RecordIdentity(
            record_id=RecordId(record_id or allocate_record_id()),
            instance=ArtifactInstance(content_sha256=digest,
                                      canonical_path=ROOT + "/" + basename,
                                      size_bytes=size),
            role=ATT,
            legacy_aliases=tuple(aliases if aliases is not None
                                 else (basename,))),
        **kw)


def measured_entries():
    return tuple(
        an_entry(name, digest=digest, size=size, cited_by=(commit,),
                 artifact_schema_version=version,
                 provenance=(ProvenanceRelation.EMITTED_BY_INSTALLER,
                             ProvenanceRelation.IMPORTED_FROM_STAGING))
        for name, size, digest, version, commit in MEASURED)


def a_manifest(entries=None, **over):
    entries = measured_entries() if entries is None else entries
    kw = {"artifact_class": "installation_attestation",
          "genesis_cardinality": len(entries),
          "genesis_aliases": tuple(
              a for e in entries for a in e.identity.legacy_aliases),
          "entries": entries}
    kw.update(over)
    return ArchiveManifest(**kw)


def refuses(fn, fragment):
    """Assert the refusal fires on the invariant it CLAIMS to test.

    Asserting only that something raised would let an earlier check absorb the
    case, which is precisely what happened twice while this file was written.
    """
    with pytest.raises((ArchiveManifestError, RecordsOntologyError)) as exc:
        fn()
    assert fragment in str(exc.value), (
        "refused, but on the WRONG check.\n  expected the message to contain: "
        "{!r}\n  actual: {}".format(fragment, exc.value))


# ---------------------------------------------------------------------------
# 1. Well-formed behaviour
# ---------------------------------------------------------------------------

def test_a_manifest_over_the_measured_corpus_constructs():
    m = a_manifest()
    assert len(m.entries) == len(MEASURED)
    assert m.genesis_cardinality == len(MEASURED)


def test_rendering_is_deterministic_and_round_trips():
    m = a_manifest()
    first = m.render()
    assert first == m.render(), "two renders of one manifest differ"
    assert ArchiveManifest.parse(first).render() == first


def test_the_rendered_manifest_is_diffable_not_one_line():
    """Determinism does not require unreadability.

    Sorted keys and fixed indentation are fully deterministic; a compact
    separator form is no more so, and turns a durable record that reviewers
    must audit in a pull request into a single unreadable line.
    """
    rendered = a_manifest().render()
    assert rendered.count(b"\n") > 20, rendered.count(b"\n")
    assert b'"schema": "gvc.installation-attestation-archive"' in rendered


def test_the_manifest_is_AUTHORED_and_the_artifacts_are_PRESERVED():
    """The asymmetry ADR-0004 section C exists for, visible in one file.

    Measured 2026-08-23: SIXTEEN of sixteen attestations end WITHOUT a
    newline, because json.dumps does not append one. This manifest DOES,
    because it is authored. Applying one policy to both would refuse every
    file the archive exists to hold.
    """
    rendered = a_manifest().render()
    assert rendered.endswith(b"\n")
    assert not any(b > 0x7F for b in rendered), "ensure_ascii should hold"
    assert b"\r\n" not in rendered


def test_a_historical_citation_resolves_through_the_index():
    """Retaining a basename is NOT resolution. This is."""
    m = a_manifest()
    hit = m.resolve(MEASURED[0][0])
    assert hit is not None
    assert hit.identity.instance.size_bytes == MEASURED[0][1]
    assert hit.identity.instance.canonical_path == ROOT + "/" + MEASURED[0][0]
    assert m.resolve("install-attestation-never-existed.json") is None


def test_identical_content_in_two_records_is_LEGITIMATE():
    """The same bytes may occur in two evidentiary contexts; the same IDENTITY
    may not. content_sha256 is deliberately not unique."""
    twins = (an_entry("first.json", digest="e" * 64),
             an_entry("second.json", digest="e" * 64))
    m = a_manifest(entries=twins)
    assert len({e.identity.instance.content_sha256 for e in m.entries}) == 1
    assert len({e.identity.record_id.value for e in m.entries}) == 2


def test_the_archive_may_GROW_but_never_SHRINK():
    entries = measured_entries()
    grown = a_manifest(entries=entries + (an_entry("later.json"),),
                       genesis_cardinality=len(entries),
                       genesis_aliases=tuple(
                           a for e in entries
                           for a in e.identity.legacy_aliases))
    assert len(grown.entries) == len(entries) + 1
    refuses(lambda: a_manifest(
        entries=entries[:2], genesis_cardinality=len(entries),
        genesis_aliases=tuple(a for e in entries
                              for a in e.identity.legacy_aliases)),
        "may never shrink")


# ---------------------------------------------------------------------------
# 2. Identity invariants, each ISOLATED
# ---------------------------------------------------------------------------

def test_two_entries_may_not_share_a_record_identifier():
    """Distinct paths and aliases, so ONLY the identifier check can fire."""
    shared = allocate_record_id()
    refuses(lambda: a_manifest(entries=(
        an_entry("p.json", record_id=shared),
        an_entry("q.json", record_id=shared, digest="b" * 64))),
        "duplicate record identifier")


def test_two_entries_may_not_share_a_canonical_path():
    """Distinct identifiers and aliases, so ONLY the path check can fire."""
    refuses(lambda: a_manifest(entries=(
        an_entry("same.json", aliases=("alias-a.json",)),
        an_entry("same.json", aliases=("alias-b.json",), digest="b" * 64))),
        "same canonical path")


def test_one_alias_may_not_resolve_to_two_records():
    """Distinct identifiers AND distinct paths, so ONLY the alias check fires.

    An ambiguous alias would make a historical citation resolve to whichever
    record happened to be scanned first.
    """
    # genesis_aliases is given EXPLICITLY as a single alias. The default
    # helper derives it from every entry, which would list "shared.json"
    # twice and trip the duplicate-genesis check first -- measured
    # 2026-08-23, and caught only because refuses() asserts the message.
    refuses(lambda: a_manifest(
        entries=(an_entry("one.json", aliases=("shared.json",)),
                 an_entry("two.json", aliases=("shared.json",),
                          digest="c" * 64)),
        genesis_cardinality=1, genesis_aliases=("shared.json",)),
        "resolves to more than one record")


def test_a_genesis_alias_must_still_resolve():
    """Caught even when the COUNT fits -- a set equality that passes on the
    wrong members is the failure mode a cardinality check cannot see."""
    entries = (an_entry("first.json"), an_entry("second.json", digest="e" * 64))
    refuses(lambda: a_manifest(
        entries=entries, genesis_cardinality=2,
        genesis_aliases=("first.json", "ghost.json")),
        "no longer resolvable")


def test_an_EMPTY_archive_cannot_satisfy_itself():
    """entries == aliases == {} makes every set equality pass. A genesis
    cardinality below one is what would allow it."""
    refuses(lambda: ArchiveManifest(
        artifact_class="x", genesis_cardinality=0, genesis_aliases=(),
        entries=()), "would let an EMPTY archive")


def test_the_genesis_alias_count_must_match_the_cardinality():
    entries = measured_entries()
    refuses(lambda: a_manifest(entries=entries, genesis_cardinality=3,
                               genesis_aliases=(MEASURED[0][0],)),
            "genesis alias(es) for a cardinality")


def test_duplicate_genesis_aliases_are_refused():
    entries = measured_entries()
    refuses(lambda: a_manifest(entries=entries, genesis_cardinality=3,
                               genesis_aliases=(MEASURED[0][0],
                                                MEASURED[0][0],
                                                MEASURED[1][0])),
            "duplicate genesis alias")


def test_an_empty_artifact_class_is_refused():
    refuses(lambda: a_manifest(artifact_class="   "), "artifact_class")


# ---------------------------------------------------------------------------
# 3. Entry-level invariants
# ---------------------------------------------------------------------------

def test_an_entry_needs_provenance():
    refuses(lambda: an_entry("x.json", provenance=()), "no provenance")


def test_duplicate_provenance_is_refused():
    refuses(lambda: an_entry(
        "x.json", provenance=(ProvenanceRelation.EMITTED_BY_INSTALLER,
                              ProvenanceRelation.EMITTED_BY_INSTALLER)),
        "duplicate provenance")


def test_duplicate_citing_commits_are_refused():
    refuses(lambda: an_entry("x.json", cited_by=("aaaaaaa", "aaaaaaa")),
            "duplicate citing commit")


def test_a_defect_disposition_requires_a_note():
    for disposition in (PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE,
                        PreservationDisposition.QUARANTINED,
                        PreservationDisposition.REJECTED):
        refuses(lambda d=disposition: an_entry("x.json", preservation=d),
                "requires a defect_note")


def test_a_clean_disposition_may_not_carry_a_defect_note():
    refuses(lambda: an_entry("x.json", defect_note="unexplained"),
            "may not carry a defect_note")


def test_restricted_bytes_may_not_be_admitted_verbatim():
    """A redacted copy presented as the original is a forgery, however well
    intentioned. Route it to a restricted channel instead."""
    refuses(lambda: an_entry(
        "x.json", disclosure=DisclosureClass.RESTRICTED_VERBATIM),
        "may not be ADMITTED_VERBATIM")


def test_an_artifact_outside_the_canonical_root_is_refused():
    """Placement follows ROLE. Filing a record elsewhere is how six evidence
    locations happened."""
    refuses(lambda: ArchiveEntry(
        identity=RecordIdentity(
            record_id=RecordId(allocate_record_id()),
            instance=ArtifactInstance(content_sha256="a" * 64,
                                      canonical_path="docs/x.json",
                                      size_bytes=10),
            role=ATT),
        cited_by=("aaaaaaa",), disclosure=DisclosureClass.PUBLIC_VERBATIM,
        preservation=PreservationDisposition.ADMITTED_VERBATIM,
        provenance=(ProvenanceRelation.EMITTED_BY_INSTALLER,),
        retention=RetentionClass.PERMANENT_EVIDENCE,
        artifact_schema_version=1),
        "canonical root")


def test_a_nonpositive_artifact_schema_version_is_refused():
    refuses(lambda: an_entry("x.json", artifact_schema_version=0),
            "must be positive")


# ---------------------------------------------------------------------------
# 4. Parsing refuses drift
# ---------------------------------------------------------------------------

def test_an_undeclared_entry_key_is_refused():
    """The drift mechanism itself: each installer added what it had learned."""
    record = dict(measured_entries()[0].as_record(), warning_kinds={})
    refuses(lambda: ArchiveEntry.from_record(record), "undeclared key")


def test_a_missing_entry_key_is_refused():
    record = dict(measured_entries()[0].as_record())
    del record["cited_by"]
    refuses(lambda: ArchiveEntry.from_record(record), "missing")


def test_an_unknown_vocabulary_term_is_refused():
    record = dict(measured_entries()[0].as_record(), role="not_a_role")
    refuses(lambda: ArchiveEntry.from_record(record), "unrecognised vocabulary")


def test_an_undeclared_manifest_key_is_refused():
    payload = json.loads(a_manifest().render().decode("utf-8"))
    payload["extra"] = 1
    refuses(lambda: ArchiveManifest.parse(
        json.dumps(payload).encode("utf-8")), "undeclared key")


def test_a_wrong_schema_name_is_refused():
    payload = json.loads(a_manifest().render().decode("utf-8"))
    payload["schema"] = "gvc.something-else"
    refuses(lambda: ArchiveManifest.parse(
        json.dumps(payload).encode("utf-8")), "expected")


def test_a_future_schema_version_is_refused():
    payload = json.loads(a_manifest().render().decode("utf-8"))
    payload["schema_version"] = SCHEMA_VERSION + 1
    refuses(lambda: ArchiveManifest.parse(
        json.dumps(payload).encode("utf-8")), "judges version")


def test_malformed_json_is_refused():
    refuses(lambda: ArchiveManifest.parse(b"{not json"), "not valid JSON")


def test_a_non_object_manifest_is_refused():
    refuses(lambda: ArchiveManifest.parse(b"[]"), "must be an object")


def test_entries_must_be_a_list():
    payload = json.loads(a_manifest().render().decode("utf-8"))
    payload["entries"] = {}
    refuses(lambda: ArchiveManifest.parse(
        json.dumps(payload).encode("utf-8")), "must be a list")


def test_the_schema_constants_are_what_the_render_declares():
    payload = json.loads(a_manifest().render().decode("utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["schema_version"] == SCHEMA_VERSION
