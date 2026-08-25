"""A reconstruction may record what it does not know, and may not invent it.

PROOF-AFTER-IRREVERSIBILITY-1. Created 2026-08-25.

WHAT THIS GUARDS
----------------
On 2026-08-25 the DRIFT-1 installer applied eight targets, proved its suite
transition by node identity, passed a 5,479-case gate, committed `abcb22e`, and
then refused:

    ATTESTATION INVALID AFTER A SUCCESSFUL COMMIT:
        a deliberate retirement requires a justification

`gvc.install-attestation` v2 requires `started_at`. MEASURED 2026-08-25, that
value is UNRECOVERABLE within a 1,434-second interval: the installer samples its
clock after the heavy package imports and no witness closes the upper end of the
window. A v2 document carrying an invented `started_at` would PASS `validate()`,
because that validator checks presence, not semantic validity.

So this schema exists to make the invented point estimate UNCONSTRUCTIBLE rather
than merely discouraged.

WHY EVERY NEGATIVE CASE ASSERTS MESSAGE TEXT
--------------------------------------------
`tests/unit/test_archive_manifest.py` records why, from experience: two
adversarial cases there were built by reusing an entry, so an earlier check
fired first -- "they refused, and proved nothing about the invariants they
named". `refuses()` below follows that file's convention exactly.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.transactions.install_attestation_reconstruction import (
    SCHEMA,
    SCHEMA_VERSION,
    EvidencedField,
    EvidenceKind,
    EvidenceRef,
    GitIdentity,
    KnowledgeState,
    PublicationFailure,
    ReconstructionDocument,
    ReconstructionSchemaError,
    ReconstructionStatus,
)

#: The real identities, measured 2026-08-25 by `git rev-parse`. Real values so
#: the fixtures cannot drift into fantasy -- the convention MEASURED at
#: tests/unit/test_archive_manifest.py lines 62-74.
COMMIT = "abcb22eaf4e833f417130ad871c41adffffc4660"
TREE = "061315ad453f6a3068dbcd60a872a9397ba74b93"
PARENT = "2e7e435a0dbf1bf1e8e25ec516fe7f62f194261d"
APPLY_LOG = "gvc_install_drift_readiness_v4_apply_2026-08-24.txt"
APPLY_LOG_SHA = "026fa356ade04655c84b0de425cfe100863c20730c791adf9a418c530dd725d5"
CITED = ("install-attestation-DRIFT-1-READINESS-P0-2026-08-24-"
         "20260825T002451Z.json")


def a_witness(**over):
    kw = {"kind": EvidenceKind.FILESYSTEM_METADATA, "locator": APPLY_LOG,
          "claim": "creation time bounds the value from below"}
    kw.update(over)
    return EvidenceRef(**kw)


def finished_at_field():
    """DERIVED_EXACT: squeezed between two independent witnesses.

    The committer date of `abcb22e` is 2026-08-25T00:24:52Z and the apply log's
    last write is the same second. The installer samples `finished_at` after
    `git commit` returns and before the failure line prints, so both bounds
    coincide and the value is determined -- derived, not observed.
    """
    return EvidencedField(
        name="finished_at", state=KnowledgeState.DERIVED_EXACT,
        value="2026-08-25T00:24:52Z", resolution="1s",
        derivation="interval_squeeze",
        witnesses=(
            EvidenceRef(kind=EvidenceKind.GIT_METADATA, locator=COMMIT,
                        claim="committer date bounds the value from below"),
            a_witness(claim="last-write time bounds the value from above",
                      sha256=APPLY_LOG_SHA)))


def started_at_field():
    """BOUNDED: a 1,434-second window with no witness closing it."""
    return EvidencedField(
        name="started_at", state=KnowledgeState.BOUNDED,
        lower_bound="2026-08-25T00:00:58Z", upper_bound="2026-08-25T00:24:52Z",
        witnesses=(a_witness(sha256=APPLY_LOG_SHA),))


def a_document(**over):
    kw = {
        "subject_unit": "DRIFT-1-READINESS-P0-2026-08-24",
        "intended_legacy_alias": CITED,
        "repository": GitIdentity(commit_oid=COMMIT, tree_oid=TREE,
                                  parent_oid=PARENT),
        "failure": PublicationFailure(
            finding="PROOF-AFTER-IRREVERSIBILITY-1",
            publication_error="a deliberate retirement requires a justification"),
        "fields": (finished_at_field(), started_at_field()),
        "suite_transition": {"kind": "deliberate_retirement",
                             "before_digest": "3a7e1ef0" + "0" * 56,
                             "after_digest": "a99c059a" + "0" * 56},
        "acceptance": {"passed": 5464, "skipped": 15, "returncode": 0},
        "targets": ({"path": "README.md", "action": "patch"},),
        "reconstructed_at": "2026-08-25T12:00:00Z",
    }
    kw.update(over)
    return ReconstructionDocument(**kw)


def refuses(fn, fragment):
    """Assert the refusal fires on the invariant it CLAIMS to test."""
    with pytest.raises(ReconstructionSchemaError) as exc:
        fn()
    assert fragment in str(exc.value), (
        "refused, but on the WRONG check.\n  expected the message to contain: "
        "{!r}\n  actual: {}".format(fragment, exc.value))


# ---------------------------------------------------------------------------
# 1. THE INVENTED POINT ESTIMATE IS UNCONSTRUCTIBLE
# ---------------------------------------------------------------------------

def test_a_bounded_field_may_not_claim_an_exact_value():
    """The whole reason this schema exists.

    `started_at` is known only to lie in a 1,434-second window. A v2
    attestation would accept any second inside it; this type accepts none.
    """
    refuses(lambda: EvidencedField(
        name="started_at", state=KnowledgeState.BOUNDED,
        value="2026-08-25T00:12:00Z", lower_bound="2026-08-25T00:00:58Z",
        upper_bound="2026-08-25T00:24:52Z", witnesses=(a_witness(),)),
        "invented point estimate")


def test_an_exactly_known_field_may_not_carry_bounds():
    refuses(lambda: EvidencedField(
        name="finished_at", state=KnowledgeState.OBSERVED, value="v",
        lower_bound="a", upper_bound="b", witnesses=(a_witness(),)),
        "may not carry bounds")


def test_an_unrecoverable_field_may_not_carry_data():
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.UNRECOVERABLE, value="guessed"),
        "may not carry invented data")


def test_one_bound_is_not_an_interval():
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.BOUNDED,
        lower_bound="2026-08-25T00:00:58Z", witnesses=(a_witness(),)),
        "requires BOTH bounds")


def test_bounds_may_not_be_inverted():
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.BOUNDED,
        lower_bound="2026-08-25T00:24:52Z", upper_bound="2026-08-25T00:00:58Z",
        witnesses=(a_witness(),)), "exceeds upper bound")


# ---------------------------------------------------------------------------
# 2. DERIVED IS NOT OBSERVED
# ---------------------------------------------------------------------------

def test_a_derived_value_must_state_its_resolution_and_derivation():
    """"Uniquely determined at one-second resolution by an interval squeeze" is
    a different claim from "observed", and the difference must be recorded."""
    refuses(lambda: EvidencedField(
        name="finished_at", state=KnowledgeState.DERIVED_EXACT,
        value="2026-08-25T00:24:52Z", witnesses=(a_witness(),)),
        "requires a resolution and a derivation")


def test_only_a_derived_value_may_carry_a_derivation():
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.OBSERVED, value="v",
        resolution="1s", derivation="squeeze", witnesses=(a_witness(),)),
        "only DERIVED_EXACT")


def test_the_measured_finished_at_is_derived_not_observed():
    field = finished_at_field()
    assert field.state is KnowledgeState.DERIVED_EXACT
    assert field.state is not KnowledgeState.OBSERVED
    assert field.resolution == "1s"
    assert field.derivation == "interval_squeeze"
    assert len(field.witnesses) == 2, (
        "a squeeze requires TWO witnesses; one bound is not a squeeze")


# ---------------------------------------------------------------------------
# 3. A CLAIM WITHOUT A WITNESS IS NOT EVIDENCE
# ---------------------------------------------------------------------------

def test_a_known_field_requires_a_witness():
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.OBSERVED, value="v"),
        "requires at least one witness")


def test_an_unrecoverable_field_may_not_cite_witnesses():
    """If a witness constrains it, it is bounded, not unrecoverable."""
    refuses(lambda: EvidencedField(
        name="x", state=KnowledgeState.UNRECOVERABLE,
        witnesses=(a_witness(),)), "it is not unrecoverable")


def test_a_witness_must_say_what_it_establishes():
    refuses(lambda: EvidenceRef(
        kind=EvidenceKind.GIT_OBJECT, locator="abcb22e", claim="   "),
        "must state what it establishes")


def test_a_witness_digest_must_be_a_sha256():
    refuses(lambda: a_witness(sha256="abc"), "not a SHA-256 digest")


# ---------------------------------------------------------------------------
# 4. IDENTITY IS NOT A DISPLAY STRING
# ---------------------------------------------------------------------------

def test_an_abbreviated_commit_identifier_is_refused():
    """v2 accepted seven-character forms; this class is new and does not
    inherit that weakness."""
    refuses(lambda: GitIdentity(commit_oid="abcb22e", tree_oid=TREE,
                                parent_oid=PARENT), "full 40-character")


def test_a_commit_may_not_be_its_own_parent():
    refuses(lambda: GitIdentity(commit_oid=COMMIT, tree_oid=TREE,
                                parent_oid=COMMIT), "its own parent")


def test_the_cited_alias_must_be_a_basename():
    """A path would compete with the canonical location, which is the defect
    `legacy_aliases` exists to avoid."""
    refuses(lambda: a_document(
        intended_legacy_alias="records/attestations/x.json"),
        "is a path, not a basename")


# ---------------------------------------------------------------------------
# 5. THE FAILURE IS PART OF THE RECORD
# ---------------------------------------------------------------------------

def test_a_reconstruction_of_a_validly_emitted_artifact_is_refused():
    """If the original was emitted validly, preserve it. Do not reconstruct."""
    refuses(lambda: PublicationFailure(
        finding="F", publication_error="e",
        original_artifact_validly_emitted=True), "nothing to reconstruct")


def test_the_failure_names_both_the_finding_and_the_error():
    doc = a_document()
    record = doc.payload["failure"]
    assert record["finding"] == "PROOF-AFTER-IRREVERSIBILITY-1"
    assert "justification" in record["publication_error"]
    assert record["original_artifact_validly_emitted"] is False


# ---------------------------------------------------------------------------
# 6. STATUS IS DERIVED FROM THE FIELDS
# ---------------------------------------------------------------------------

def test_a_bounded_field_makes_the_reconstruction_partial():
    assert a_document().status is ReconstructionStatus.PARTIAL


def test_only_exact_fields_make_it_complete():
    doc = a_document(fields=(finished_at_field(),))
    assert doc.status is ReconstructionStatus.COMPLETE


def test_the_status_cannot_be_declared_independently_of_the_fields():
    """There is no field to assert it in: `status` is a property.

    A caller-supplied status would eventually disagree with the evidence, which
    is the failure mode this whole class exists to prevent.
    """
    with pytest.raises(TypeError):
        a_document(reconstruction_status="complete")


def test_duplicate_field_names_are_refused():
    refuses(lambda: a_document(
        fields=(finished_at_field(), finished_at_field())), "duplicate field")


def test_a_reconstruction_with_no_fields_establishes_nothing():
    refuses(lambda: a_document(fields=()), "establishes nothing")


# ---------------------------------------------------------------------------
# 7. SERIALIZATION
# ---------------------------------------------------------------------------

def test_the_render_is_authored_and_ends_with_a_newline():
    """The asymmetry ADR-0004 section C exists for.

    MEASURED 2026-08-23: seventeen of seventeen preserved attestations end
    WITHOUT a newline because `json.dumps` does not append one. This document is
    AUTHORED, so it must end WITH one.
    """
    raw = a_document().render()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert not any(b > 0x7F for b in raw), "ensure_ascii should hold"


def test_rendering_is_deterministic_and_round_trips():
    doc = a_document()
    first = doc.render()
    assert first == doc.render(), "two renders of one document differ"
    assert ReconstructionDocument.parse(first) == doc.payload


def test_the_render_is_diffable_not_one_line():
    raw = a_document().render()
    assert raw.count(b"\n") > 20, raw.count(b"\n")


def test_a_wrong_schema_name_is_refused():
    payload = dict(a_document().payload, schema="gvc.something-else")
    refuses(lambda: ReconstructionDocument.parse(
        json.dumps(payload).encode("utf-8")), "expected")


def test_a_future_schema_version_is_refused():
    payload = dict(a_document().payload, schema_version=SCHEMA_VERSION + 1)
    refuses(lambda: ReconstructionDocument.parse(
        json.dumps(payload).encode("utf-8")), "judges version")


def test_malformed_json_is_refused():
    refuses(lambda: ReconstructionDocument.parse(b"{not json"),
            "not valid JSON")


def test_an_unrecognised_knowledge_state_is_refused():
    """The vocabulary is closed. A state nobody declared cannot be read back."""
    payload = a_document().payload
    payload["recovered_fields"][0]["state"] = "probably_fine"
    refuses(lambda: ReconstructionDocument.parse(
        json.dumps(payload).encode("utf-8")), "unrecognised knowledge state")


def test_the_schema_constants_are_what_the_render_declares():
    payload = json.loads(a_document().render().decode("utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_class"] == "installation_attestation_reconstruction"


def test_the_document_does_not_claim_to_be_the_missing_original():
    """It resolves the citation; it does not impersonate the artifact.

    `schema` differs from `gvc.install-attestation`, and the alias is recorded
    as INTENDED rather than as this document's own name.
    """
    payload = a_document().payload
    assert payload["schema"] != "gvc.install-attestation"
    assert payload["intended_legacy_alias"] == CITED
    assert payload["failure"]["original_artifact_validly_emitted"] is False
