"""The record ontology is a contract, not a set of helpful enumerations.

ADR-0004, 2026-08-22.

WHY
---
Twenty-six machine-evidence documents were committed across SIX directories
while eleven install attestations lived outside version control entirely. The
cause was that machine records never acquired an architectural layer, so each
subsystem filed them under whichever documentation noun was convenient.

Naming a better noun would have preserved the category error. What prevents a
seventh location is that PLACEMENT FOLLOWS ROLE, executably:

    destination(artifact) = f(ArtifactRole)

WHAT THIS PROVES
----------------
Not that the enumerations exist -- that is trivial. That the ontology REFUSES
every malformed classification, placement and identity. Twenty-six of the
thirty-four cases below are negative controls, and they exercise the shipping
code, so a control cannot pass against a broken implementation.

Author: Monzia Moodie
"""
from __future__ import annotations

import re

import pytest

from genomic_variant_classifier.repository_records.classification import (
    DisclosureClass,
    PreservationDisposition,
    ProvenanceRelation,
    RecordDisposition,
    RetentionClass,
)
from genomic_variant_classifier.repository_records.identity import (
    ArtifactInstance,
    RecordId,
    RecordIdentity,
    allocate_record_id,
)
from genomic_variant_classifier.repository_records.roles import (
    CANONICAL_RECORD_ROOTS,
    RECORDS_ROOT,
    ArtifactRole,
    RecordsOntologyError,
    canonical_root,
    is_within_canonical_root,
    role_for_path,
)
from genomic_variant_classifier.repository_records.validation import (
    AuthoringPolicyError,
    PreservationPolicyError,
    digest,
    validate_authored_text,
    validate_verbatim_artifact,
)

ATT = ArtifactRole.INSTALLATION_ATTESTATION
GOOD_PATH = "records/attestations/installations/artifacts/x.json"
GOOD_SHA = "a" * 64


def an_instance(**over) -> ArtifactInstance:
    kw = {"content_sha256": GOOD_SHA, "canonical_path": GOOD_PATH,
          "size_bytes": 10}
    kw.update(over)
    return ArtifactInstance(**kw)


def a_disposition(**over) -> RecordDisposition:
    kw = {"role": ATT, "disclosure": DisclosureClass.PUBLIC_VERBATIM,
          "preservation": PreservationDisposition.ADMITTED_VERBATIM,
          "provenance": (ProvenanceRelation.EMITTED_BY_INSTALLER,),
          "retention": RetentionClass.PERMANENT_EVIDENCE}
    kw.update(over)
    return RecordDisposition(**kw)


# ---------------------------------------------------------------------------
# 1. The registry IS the authority
# ---------------------------------------------------------------------------

def test_every_role_has_exactly_one_canonical_root():
    """A role without a root cannot place anything; a default would place it
    wrongly and silently."""
    assert set(CANONICAL_RECORD_ROOTS) == set(ArtifactRole)


def test_no_two_roles_share_a_root():
    """Shared roots make placement ambiguous and role inference impossible."""
    roots = list(CANONICAL_RECORD_ROOTS.values())
    assert len(set(roots)) == len(roots), roots


def test_every_root_is_beneath_the_plane():
    for role, root in CANONICAL_RECORD_ROOTS.items():
        assert root.parts[0] == RECORDS_ROOT.parts[0], (role, root)
        assert root != RECORDS_ROOT, (
            "{} claims the plane root itself; a role must own a subtree, not "
            "the whole plane".format(role))


def test_no_root_is_nested_inside_another():
    """Nested roots would make role_for_path ambiguous by construction."""
    roots = sorted(str(r) for r in CANONICAL_RECORD_ROOTS.values())
    for a in roots:
        for b in roots:
            if a != b:
                assert not b.startswith(a + "/"), (a, b)


def test_roles_are_not_documentation_paths():
    """`docs/` is what humans author. `records/` is what happened."""
    for root in CANONICAL_RECORD_ROOTS.values():
        assert not str(root).startswith("docs/"), root


# ---------------------------------------------------------------------------
# 2. Containment, never a string prefix
# ---------------------------------------------------------------------------

def test_containment_accepts_a_real_descendant():
    assert is_within_canonical_root(GOOD_PATH, ATT)


def test_containment_rejects_a_sibling_that_shares_a_text_prefix():
    """`records/audits2/x` starts with `records/audits` as TEXT while being no
    descendant of it. ADR-0002 records the same distinction for runtime paths."""
    assert not is_within_canonical_root(
        "records/audits2/x.json", ArtifactRole.AUDIT_RESULT)


def test_containment_rejects_the_root_itself():
    assert not is_within_canonical_root(
        str(canonical_root(ATT)), ATT)


def test_containment_rejects_another_roles_subtree():
    assert not is_within_canonical_root("records/measurements/m.json", ATT)


def test_containment_normalises_windows_separators():
    assert is_within_canonical_root(
        GOOD_PATH.replace("/", "\\"), ATT)


def test_role_inference_exists_only_for_auditing():
    assert role_for_path("records/measurements/m.json") is \
        ArtifactRole.EXECUTION_MEASUREMENT


def test_role_inference_refuses_a_path_outside_every_root():
    with pytest.raises(RecordsOntologyError):
        role_for_path("docs/measurements/m.json")


# ---------------------------------------------------------------------------
# 3. Identity
# ---------------------------------------------------------------------------

def test_an_allocated_identifier_is_well_formed_and_unique():
    ids = {allocate_record_id() for _ in range(64)}
    assert len(ids) == 64, "allocation collided"
    for value in ids:
        assert re.match(r"^REC-[0-9a-f]{32}$", value), value
        RecordId(value)


def test_a_sequential_identifier_is_refused():
    """`REC-0001` implies a global ordering nothing enforces and invites
    renumbering; a renumbered durable identity is not durable."""
    for bad in ("REC-0001", "REC-1", "ATT-INSTALL-0001", "rec-" + "a" * 32,
                "REC-" + "A" * 32, "REC-" + "a" * 31, ""):
        with pytest.raises(RecordsOntologyError):
            RecordId(bad)


def test_an_instance_requires_a_real_digest():
    for bad in ("", "abc", "A" * 64, "g" * 64, "a" * 63):
        with pytest.raises(RecordsOntologyError):
            an_instance(content_sha256=bad)


def test_an_instance_refuses_a_zero_length_artifact():
    with pytest.raises(RecordsOntologyError):
        an_instance(size_bytes=0)


def test_an_instance_refuses_a_negative_size():
    with pytest.raises(RecordsOntologyError):
        an_instance(size_bytes=-1)


def test_an_instance_refuses_a_windows_path():
    with pytest.raises(RecordsOntologyError):
        an_instance(canonical_path=GOOD_PATH.replace("/", "\\"))


def test_an_instance_refuses_an_absolute_path():
    with pytest.raises(RecordsOntologyError):
        an_instance(canonical_path="/" + GOOD_PATH)


def test_an_identity_must_lie_beneath_its_roles_root():
    """Filing a record elsewhere is how six evidence locations happened."""
    with pytest.raises(RecordsOntologyError):
        RecordIdentity(record_id=RecordId(allocate_record_id()),
                       instance=an_instance(
                           canonical_path="docs/audits/x.json"),
                       role=ATT)


def test_an_identity_accepts_a_correct_placement():
    ident = RecordIdentity(record_id=RecordId(allocate_record_id()),
                           instance=an_instance(), role=ATT,
                           legacy_aliases=("install-attestation-old.json",))
    assert ident.basename == "x.json"


def test_an_alias_may_not_be_a_path():
    """An alias is the BASENAME a historical citation used. A path would be a
    second locator competing with canonical_path."""
    for bad in ("docs/x.json", "a\\b.json"):
        with pytest.raises(RecordsOntologyError):
            RecordIdentity(record_id=RecordId(allocate_record_id()),
                           instance=an_instance(), role=ATT,
                           legacy_aliases=(bad,))


def test_duplicate_aliases_are_refused():
    with pytest.raises(RecordsOntologyError):
        RecordIdentity(record_id=RecordId(allocate_record_id()),
                       instance=an_instance(), role=ATT,
                       legacy_aliases=("a.json", "a.json"))


def test_an_empty_alias_is_refused():
    with pytest.raises(RecordsOntologyError):
        RecordIdentity(record_id=RecordId(allocate_record_id()),
                       instance=an_instance(), role=ATT,
                       legacy_aliases=("   ",))


# ---------------------------------------------------------------------------
# 4. Four orthogonal axes
# ---------------------------------------------------------------------------

def test_a_well_formed_disposition_is_accepted():
    d = a_disposition()
    assert d.is_publicly_preservable


def test_provenance_may_not_be_empty():
    """'Where did this come from' must not be inferable only from a directory."""
    with pytest.raises(RecordsOntologyError):
        a_disposition(provenance=())


def test_duplicate_provenance_is_refused():
    with pytest.raises(RecordsOntologyError):
        a_disposition(provenance=(ProvenanceRelation.EMITTED_BY_INSTALLER,
                                  ProvenanceRelation.EMITTED_BY_INSTALLER))


def test_a_defect_disposition_requires_a_note():
    for disp in (PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE,
                 PreservationDisposition.QUARANTINED,
                 PreservationDisposition.REJECTED):
        with pytest.raises(RecordsOntologyError):
            a_disposition(preservation=disp, defect_note="  ")


def test_a_clean_disposition_may_not_carry_a_defect_note():
    with pytest.raises(RecordsOntologyError):
        a_disposition(defect_note="something")


def test_restricted_bytes_may_not_be_admitted_verbatim_to_a_public_repository():
    """A redacted copy presented as the original is a forgery, however well
    intentioned. Route it to a restricted channel instead."""
    with pytest.raises(RecordsOntologyError):
        a_disposition(disclosure=DisclosureClass.RESTRICTED_VERBATIM)


def test_a_defect_note_does_not_prevent_preservation():
    """Preservation validity and interchange validity are orthogonal: a
    malformed historical artifact is still historical evidence."""
    d = a_disposition(
        preservation=PreservationDisposition.ADMITTED_WITH_DEFECT_NOTE,
        defect_note="truncated JavaScript Object Notation, preserved as found")
    assert d.is_publicly_preservable


def test_a_quarantined_artifact_is_not_publicly_preservable():
    d = a_disposition(preservation=PreservationDisposition.QUARANTINED,
                      defect_note="provenance disputed")
    assert not d.is_publicly_preservable


# ---------------------------------------------------------------------------
# 5. The three validation policies are not aliases
# ---------------------------------------------------------------------------

def test_the_authoring_policy_demands_house_style():
    validate_authored_text("ok.md", b"# fine\n")
    for bad in (b"\xef\xbb\xbfx\n", b"a\r\nb\n", "e\u2014m\n".encode("utf-8"),
                b"no newline", b"   \n"):
        with pytest.raises(AuthoringPolicyError):
            validate_authored_text("bad", bad)


def test_the_preservation_policy_accepts_what_authoring_refuses():
    """THE DISTINCTION. Eleven of eleven attestations end without a newline;
    an importer reusing the authoring predicate would refuse every file it
    exists to preserve."""
    historical = b'{"schema": "gvc.install-attestation"}'
    assert not historical.endswith(b"\n")
    with pytest.raises(AuthoringPolicyError):
        validate_authored_text("historical", historical)
    validate_verbatim_artifact("historical", historical, historical,
                               digest(historical))


def test_the_preservation_policy_accepts_non_ascii_and_a_byte_order_mark():
    for historical in ("# Secrets \u2014 never bake in".encode("utf-8"),
                       b"\xef\xbb\xbf{}"):
        validate_verbatim_artifact("h", historical, historical)


def test_the_preservation_policy_refuses_any_mutation():
    source = b'{"a": 1}'
    for mutated in (source + b"\n", source.replace(b" ", b""), b"", source[:-1]):
        with pytest.raises(PreservationPolicyError):
            validate_verbatim_artifact("h", source, mutated)


def test_the_preservation_policy_refuses_a_digest_that_does_not_match():
    source = b'{"a": 1}'
    with pytest.raises(PreservationPolicyError):
        validate_verbatim_artifact("h", source, source, "b" * 64)


def test_the_preservation_policy_refuses_an_empty_source():
    with pytest.raises(PreservationPolicyError):
        validate_verbatim_artifact("h", b"", b"")
