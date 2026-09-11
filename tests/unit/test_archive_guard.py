"""The archive guard: preservation, approved addition, type-sensitive comparison.

Author: Monzia Moodie

Every case here was REPRODUCED against the implementation before it was
written. The two marked BYPASS are the ones ordinary `==` would miss; the rest
are ordinary change detection and prove nothing about type-sensitivity.
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.repository_records.archive_guard import (
    changed_fields, projection_bytes, require_approved_addition,
    require_archive_preservation)
from genomic_variant_classifier.repository_records.archive_manifest import (
    ArchiveManifest, ArchiveManifestError)


def _entry(record_id, name, payload, version=3, cited="aaaaaaa"):
    import hashlib
    return {"record_id": record_id,
            "canonical_path":
                "records/attestations/installations/artifacts/" + name,
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload), "legacy_aliases": [name],
            "cited_by": [cited], "role": "installation_attestation",
            "disclosure": "public_verbatim",
            "preservation": "admitted_verbatim",
            "retention": "permanent_evidence",
            "provenance": ["emitted_by_installer", "imported_from_staging"],
            "artifact_schema_version": version}


def _render(document):
    return ArchiveManifest.parse(
        (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
         + "\n").encode("utf-8"))


@pytest.fixture
def archive():
    genesis = [_entry("REC-" + "{:032x}".format(i),
                      "install-attestation-GENESIS-{}.json".format(i),
                      json.dumps({"g": i}, sort_keys=True).encode())
               for i in range(2)]
    added = [_entry("REC-" + "{:032x}".format(100 + i),
                    "install-attestation-ADDED-{}.json".format(i),
                    json.dumps({"a": i}, sort_keys=True).encode())
             for i in range(3)]
    base = {"schema": "gvc.installation-attestation-archive",
            "schema_version": 1,
            "artifact_class": "installation_attestation",
            "genesis_cardinality": 2,
            "genesis_aliases": sorted(e["legacy_aliases"][0] for e in genesis),
            "entries": sorted(genesis, key=lambda r: r["canonical_path"])}
    before = _render(base)
    after_doc = json.loads(before.render().decode("utf-8"))
    after_doc["entries"] = sorted(after_doc["entries"] + added,
                                  key=lambda r: r["canonical_path"])
    after = _render(after_doc)
    rendered = {e["record_id"]: e
                for e in json.loads(after.render().decode("utf-8"))["entries"]}
    approved = {e["record_id"]: rendered[e["record_id"]] for e in added}
    return before, after, approved, after_doc


def test_projection_distinguishes_a_boolean_from_an_integer():
    """BYPASS: `{"v": True} == {"v": 1}` is True in Python."""
    assert {"v": True} == {"v": 1}
    assert projection_bytes({"v": True}) != projection_bytes({"v": 1})


def test_projection_distinguishes_an_integer_from_a_float():
    """BYPASS: `{"v": 1} == {"v": 1.0}` is True in Python."""
    assert {"v": 1} == {"v": 1.0}
    assert projection_bytes({"v": 1}) != projection_bytes({"v": 1.0})


def test_projection_ignores_key_ordering():
    assert projection_bytes({"a": 1, "b": 2}) == projection_bytes({"b": 2, "a": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_projection_refuses_values_with_no_json_form(value):
    with pytest.raises(ValueError):
        projection_bytes({"v": value})


def test_changed_fields_reports_a_present_or_absent_key():
    assert changed_fields({"a": 1}, {}) == ["a"]
    assert changed_fields({}, {"a": 1}) == ["a"]


def test_the_approved_transition_is_accepted(archive):
    before, after, approved, _ = archive
    report = require_approved_addition(before, after,
                                       approved_entries=approved)
    assert report["carried_forward"] == 2
    assert report["admitted_count"] == 3
    assert report["approved_content_verified"] == 3


def test_preservation_alone_permits_additions_it_does_not_approve(archive):
    before, after, _approved, _ = archive
    assert require_archive_preservation(before, after)["admitted_count"] == 3


def test_a_preserved_entry_changed_to_a_float_is_refused(archive):
    """BYPASS: the two entry dictionaries compare EQUAL under `==`."""
    before, _after, _approved, after_doc = archive
    victim = before.entries[0].identity.legacy_aliases[0]
    document = json.loads(json.dumps(after_doc))
    for entry in document["entries"]:
        if entry["legacy_aliases"][0] == victim:
            entry["size_bytes"] = float(entry["size_bytes"])
    with pytest.raises(ArchiveManifestError):
        require_archive_preservation(before, _render(document))


def test_a_preserved_NON_GENESIS_entry_removed_is_refused():
    """The gap the guard exists to close.

    An earlier version of this test removed a GENESIS record and changed the
    header to match, so header checking alone could make it pass. Here the
    predecessor already contains a non-genesis record, only that record is
    removed, and the header is untouched -- so the refusal can come only from
    the entry-preservation check.
    """
    genesis = [_entry("REC-" + "{:032x}".format(i),
                      "install-attestation-GENESIS-{}.json".format(i),
                      json.dumps({"g": i}, sort_keys=True).encode())
               for i in range(2)]
    carried = _entry("REC-" + "{:032x}".format(50),
                     "install-attestation-CARRIED.json",
                     json.dumps({"c": 1}, sort_keys=True).encode())
    document = {"schema": "gvc.installation-attestation-archive",
                "schema_version": 1,
                "artifact_class": "installation_attestation",
                "genesis_cardinality": 2,
                "genesis_aliases": sorted(e["legacy_aliases"][0]
                                          for e in genesis),
                "entries": sorted(genesis + [carried],
                                  key=lambda r: r["canonical_path"])}
    before = _render(document)
    after_doc = json.loads(json.dumps(document))
    after_doc["entries"] = [e for e in after_doc["entries"]
                            if e["record_id"] != carried["record_id"]]
    assert after_doc["genesis_cardinality"] == document["genesis_cardinality"]
    assert after_doc["genesis_aliases"] == document["genesis_aliases"]
    with pytest.raises(ArchiveManifestError, match="disappeared"):
        require_archive_preservation(before, _render(after_doc))


def test_an_added_record_with_unapproved_content_is_refused(archive):
    """An approved identifier does not bind what is stored under it."""
    before, after, approved, _ = archive
    forged = json.loads(json.dumps(approved))
    forged[sorted(forged)[0]]["content_sha256"] = "0" * 64
    with pytest.raises(ArchiveManifestError):
        require_approved_addition(before, after, approved_entries=forged)


def test_an_undeclared_addition_is_refused(archive):
    before, after, approved, _ = archive
    fewer = {k: v for k, v in list(approved.items())[1:]}
    with pytest.raises(ArchiveManifestError):
        require_approved_addition(before, after, approved_entries=fewer)


def test_a_declared_addition_that_is_absent_is_refused(archive):
    before, after, approved, _ = archive
    extra = dict(approved)
    extra["REC-" + "b" * 32] = list(approved.values())[0]
    with pytest.raises(ArchiveManifestError):
        require_approved_addition(before, after, approved_entries=extra)


def test_a_header_change_is_refused(archive):
    before, _after, _approved, after_doc = archive
    document = json.loads(json.dumps(after_doc))
    document["artifact_class"] = "something_else"
    with pytest.raises(ArchiveManifestError):
        require_archive_preservation(before, _render(document))
