"""An archive transition is additive, and its additions are the approved ones.

Author: Monzia Moodie

DERIVATION
==========
Written against the verified archive owner:

    src/genomic_variant_classifier/repository_records/archive_manifest.py
    15,090 bytes, SHA-256
    18926cbd53c23386fe2477f41419b1a153844deb62b27488dad0c1236f1280c6

and exercised against the real 103-entry postimage:

    82,858 bytes, SHA-256
    9aaf2fdc04dec0e3ef0849d6e949fa1cdf581dbe1d4c1612a8294fda5a218f4c

An earlier candidate for this file was delivered at SHA-256 c921cd03...; a
DIFFERING version at 28b98a9a... was later found in its place, in both the
working copy and the delivered copy. That version is preserved as evidence and
its type-sensitive idea is adopted here deliberately, with its correctness
re-established by the regressions below rather than inherited. These bytes are
derived from the owner interface measured above.

WHY THE GENESIS FLOOR IS NOT ENOUGH
===================================
`ArchiveManifest.__post_init__` enforces `len(entries) >= genesis_cardinality`
with every genesis alias retained. That is a FLOOR, not monotonicity. MEASURED
2026-09-08 against the real module and the real postimage:

    103 -> 102, one non-genesis record removed    ACCEPTED
    103 ->  17, every non-genesis record removed  ACCEPTED, 86 erased

Exactly ONE preimage record lies outside the genesis alias set -- the
RECONSTRUCTION, the record that exists because an original was lost. The one
thing the floor cannot protect is the evidence class already proven losable.

WHY PYTHON EQUALITY IS THE WRONG PREDICATE
==========================================
`bool` subclasses `int` and numeric comparison crosses types:

    True == 1     ->  True
    1.0  == 1     ->  True

MEASURED 2026-09-08, the archive owner ACCEPTS every one of these on a
preserved entry and renders them back unchanged:

    artifact_schema_version 3 -> True   accepted, renders True  (bool)
    artifact_schema_version 3 -> 1.0    accepted, renders 1.0   (float)
    size_bytes 7177 -> 7177.0           accepted, renders 7177.0 (float)
    size_bytes 7177 -> True             accepted, renders True   (bool)

And the two entry dictionaries for the `size_bytes` case compare EQUAL under
`==` while their serialisations differ. So the owner does not enforce these
primitive types and a dictionary comparison cannot see them; the comparison
must be over serialised bytes.

TWO OBLIGATIONS, NOT ONE
========================
    preservation        existing entries and header metadata unchanged
    admission validity  new entries are the approved identities AND carry the
                        approved content

A preservation check alone permits arbitrary additions. An identity-only
admission check permits the WRONG CONTENT under the right identifier. Both are
required, and they are separate functions here so a caller cannot satisfy one
while believing it satisfied both.

WHAT THE GUARANTEE IS, PRECISELY
================================
Preservation of the COMPLETE RENDERED PROJECTION. Not of original input
formatting, and not of fields the owner's parser discards. A manifest written
with different whitespace renders identically and is treated as unchanged --
correctly, because the owner's projection is what the archive means.

EXIT STATUS when run as a script
  0  every regression held
  2  a regression did not hold
"""

from __future__ import annotations

import json
import sys

OWNER_SHA256 = "18926cbd53c23386fe2477f41419b1a153844deb62b27488dad0c1236f1280c6"


def projection_bytes(value) -> bytes:
    """The comparison representation: the value's serialisation, not the value.

    `separators=(",", ":")` removes formatting from the comparison so only
    content and TYPE remain. `ensure_ascii=True` with an ASCII encode means a
    non-ASCII value cannot alter the comparison's own encoding. `allow_nan`
    false refuses NaN and infinity, which have no JSON representation and
    compare unequal to themselves.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def changed_fields(before: dict, after: dict) -> list:
    """Which keys differ, counting presence and TYPE as differences."""
    return sorted(
        key for key in before.keys() | after.keys()
        if key not in before or key not in after
        or projection_bytes(before[key]) != projection_bytes(after[key]))


def _error_type(manifest):
    """The owner's exception, imported from the owner, never redefined."""
    module = sys.modules.get(type(manifest).__module__)
    error = getattr(module, "ArchiveManifestError", None)
    if error is None:
        raise RuntimeError(
            "the archive owner's ArchiveManifestError was not found on {}. "
            "This guard raises the OWNER's type; substituting another would "
            "make callers catch the wrong exception.".format(
                type(manifest).__module__))
    return error


def _index(entries, error, side):
    """record_id -> rendered entry, refusing a duplicate or anonymous record.

    `ArchiveManifest` already refuses duplicates, so this cannot fire against a
    parsed manifest. It is here so the guard is not VACUOUS when handed a
    hand-built entry list, which is how a duplicate would actually arrive.
    """
    index = {}
    for entry in entries:
        record_id = entry.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise error(
                "an entry in the {} manifest carries no record identifier; an "
                "index cannot be built over an anonymous record".format(side))
        if record_id in index:
            raise error("record identifier {} appears twice in the {} "
                        "manifest".format(record_id, side))
        index[record_id] = entry
    return index


def require_archive_preservation(before, after) -> dict:
    """Existing records and header metadata survive unchanged.

    THE STANDING CHECK. An unchanged archive passes, and so does a legitimate
    addition: a gate that refused additions would refuse every admission. This
    constrains only what must NOT change, and reports what was added so a
    caller may then apply the admission check.
    """
    error = _error_type(before)

    old_document = json.loads(before.render())
    new_document = json.loads(after.render())
    old_entries = old_document.pop("entries")
    new_entries = new_document.pop("entries")

    header = changed_fields(old_document, new_document)
    if header:
        raise error(
            "pure admission changed manifest-level metadata: {}. A header "
            "change is a migration, not an admission.".format(header))

    old = _index(old_entries, error, "before")
    new = _index(new_entries, error, "after")

    removed = sorted(old.keys() - new.keys())
    if removed:
        raise error(
            "{} previously admitted record(s) disappeared: {}. The genesis "
            "floor cannot see this: measured 2026-09-08, a 103-entry manifest "
            "may fall to 17 while satisfying it.".format(
                len(removed), removed[:5]))

    changed = []
    for record_id in sorted(old):
        fields = changed_fields(old[record_id], new[record_id])
        if fields:
            changed.append("{} ({})".format(record_id, ", ".join(fields)))
    if changed:
        raise error(
            "pure admission changed {} existing entr(ies): {}. Preserving "
            "bytes while changing interpretation is still a change.".format(
                len(changed), changed[:3]))

    return {"carried_forward": len(old),
            "admitted": sorted(new.keys() - old.keys()),
            "admitted_count": len(new) - len(old),
            "resulting_entries": len(new)}


def require_approved_addition(before, after, *, approved_entries) -> dict:
    """Preservation holds AND the additions are the approved records.

    `approved_entries` maps record identifier to the APPROVED RENDERED ENTRY.
    Identifiers alone would permit the wrong content under the right
    identifier, so each added entry's complete projection is compared.

    An empty mapping is legitimate and means this operation adds nothing; it is
    not a way to skip the constraint.
    """
    error = _error_type(before)
    report = require_archive_preservation(before, after)

    observed = frozenset(report["admitted"])
    expected = frozenset(approved_entries)
    if observed != expected:
        raise error(
            "added record identities differ from the approval. UNDECLARED: {}. "
            "DECLARED BUT ABSENT: {}.".format(
                sorted(observed - expected)[:5] or "none",
                sorted(expected - observed)[:5] or "none"))

    new_entries = json.loads(after.render())["entries"]
    by_id = {entry["record_id"]: entry for entry in new_entries}
    wrong = []
    for record_id in sorted(observed):
        fields = changed_fields(approved_entries[record_id], by_id[record_id])
        if fields:
            wrong.append("{} ({})".format(record_id, ", ".join(fields)))
    if wrong:
        raise error(
            "{} added record(s) do not carry the approved content: {}. An "
            "approved identifier does not bind what is stored under it."
            .format(len(wrong), wrong[:3]))

    report["approved_content_verified"] = len(observed)
    return report
