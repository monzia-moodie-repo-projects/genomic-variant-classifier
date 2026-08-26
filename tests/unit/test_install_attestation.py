"""An attestation in which the evidence contradicts itself is not evidence.

ATTESTATION-SCHEMA-DRIFT-1, 2026-08-22.
ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1, 2026-08-26.

WHY
---
Nine install attestations exist, all declaring `"schema_version": 1`, in THREE
shapes -- because every installer hand-built its own dictionary and each learned
something the previous had not recorded. A version that does not change when the
shape changes cannot be used to interpret the document.

The suite-transition unit at `a60f18f` produced identity digests that belong in
an attestation and deliberately did not add them: widening a corrupt format to
carry better evidence corrupts the evidence. This is the prerequisite that was
owed first.

WHAT IS PROVEN HERE
-------------------
Not that a well-formed document is accepted -- that is one test. That every
MALFORMED document is REFUSED. Twenty-one of the twenty-seven tests below are
negative controls, and they exercise the shipping `validate`, so a control
cannot pass against a broken validator.

The cross-field checks are the point. A field list alone prevents only shape
drift. These prevent an attestation from contradicting itself: a counter that
disagrees with the identity delta, a gate summary that accounts for a different
number of cases than the counter records, a neutral transition with different
identity digests, a PUBLISHED status over a failing gate.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy

import pytest

from genomic_variant_classifier.transactions.install_attestation import (
    SCHEMA,
    SCHEMA_VERSION,
    AttestationDocument,
    AttestationSchemaError,
    validate,
)

ADDED = ["tests/unit/test_x.py::test_one", "tests/unit/test_x.py::test_two"]

#: Version 3 records BOTH forms and requires the abbreviation to be a
#: PREFIX of the full identifier. These are consistent by construction, so
#: a test that breaks the relationship has to do so DELIBERATELY.
PRE_OID = "a60f18f" + "1" * 33
POST_OID = "beefcaf" + "2" * 33


def base() -> dict:
    """A minimal, internally consistent, version-2 attestation."""
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "unit": "EXAMPLE-2026-08-22",
        "plan_digest": "0" * 64,
        "status": "PUBLISHED",
        "started_at": "2026-08-22T07:00:00Z",
        "finished_at": "2026-08-22T07:20:00Z",
        "python": "3.12.10",
        "platform": "Windows-11",
        "repository": {
            "pre_head": "a60f18f", "post_head": "beefcaf",
            "pre_head_oid": PRE_OID, "post_head_oid": POST_OID,
        },
        "counter": {"scope": "tests", "before": 100, "after": 102},
        "acceptance": {
            "scope": "tests", "returncode": 0, "passed": 95, "skipped": 7,
            "failed": 0, "errors": 0, "xfailed": 0, "xpassed": 0,
            "warnings": 3, "summary": "95 passed, 7 skipped",
            "seconds": 12.5, "measured_by": {"python": "3.12.10"},
        },
        "targets": [{"path": "src/x.py", "action": "create",
                     "post_sha256": "1" * 64, "post_size": 10}],
        "suite_transition": {
            "kind": "addition",
            "expected_added_nodeids": list(ADDED),
            "expected_removed_nodeids": [],
            "observed_added_nodeids": list(ADDED),
            "observed_removed_nodeids": [],
            "before_count": 100, "after_count": 102,
            "before_digest": "a" * 64, "after_digest": "b" * 64,
        },
    }


# ---------------------------------------------------------------------------
# 1. The well-formed cases
# ---------------------------------------------------------------------------

def test_a_consistent_addition_is_accepted():
    validate(base())


def test_a_consistent_neutral_is_accepted():
    d = base()
    d["counter"] = {"scope": "tests", "before": 100, "after": 100}
    d["suite_transition"].update(
        kind="neutral", expected_added_nodeids=[], observed_added_nodeids=[],
        before_count=100, after_count=100,
        before_digest="c" * 64, after_digest="c" * 64)
    d["acceptance"].update(passed=93, skipped=7, summary="93 passed, 7 skipped")
    validate(d)


def test_a_consistent_retirement_is_accepted():
    d = base()
    d["counter"] = {"scope": "tests", "before": 100, "after": 98}
    d["suite_transition"].update(
        kind="deliberate_retirement",
        expected_added_nodeids=[], observed_added_nodeids=[],
        expected_removed_nodeids=list(ADDED),
        observed_removed_nodeids=list(ADDED),
        before_count=100, after_count=98,
        justification="relocated to the domain boundary")
    d["acceptance"].update(passed=91, skipped=7, summary="91 passed, 7 skipped")
    validate(d)


def test_a_pending_publication_is_accepted_with_its_error():
    d = base()
    d["status"] = "INSTALL_APPLIED_PUBLICATION_PENDING"
    d["publication_error"] = "git add exited 1"
    validate(d)


def test_construction_validates_and_round_trips():
    doc = AttestationDocument(payload=base())
    assert AttestationDocument.from_json(doc.to_json()).payload == base()


# ---------------------------------------------------------------------------
# 2. Shape refusals -- the drift itself
# ---------------------------------------------------------------------------

def test_version_one_documents_are_not_judged_by_this_schema():
    """Nine of them exist. They are history, not candidates for migration."""
    with pytest.raises(AttestationSchemaError) as exc:
        validate({"schema": SCHEMA, "schema_version": 1, "unit": "D0a"})
    assert "historical evidence" in str(exc.value)


def test_an_unknown_schema_name_is_refused():
    d = base(); d["schema"] = "gvc.something-else"
    with pytest.raises(AttestationSchemaError):
        validate(d)


#: Explicit, and explicitly used as the parametrize identifiers. pytest would
#: derive identifiers from these strings anyway, but "would anyway" is an
#: assumption about a tool's behaviour, and an installer must DECLARE the node
#: identities this file contributes. Literal identifiers make them certain.
REQUIRED_KEYS = (
    "acceptance", "counter", "finished_at", "plan_digest", "platform",
    "python", "repository", "started_at", "status", "suite_transition",
    "targets", "unit",
)


@pytest.mark.parametrize("key", REQUIRED_KEYS, ids=REQUIRED_KEYS)
def test_every_required_top_level_key_is_required(key):
    d = base(); del d[key]
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_undeclared_top_level_field_is_refused():
    """The drift mechanism: each installer added what it had just learned."""
    d = base(); d["warning_kinds"] = {"UserWarning": 1}
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "warning_kinds" in str(exc.value)


def test_an_undeclared_acceptance_field_is_refused():
    d = base(); d["acceptance"]["deselected"] = 0
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_missing_acceptance_field_is_refused():
    d = base(); del d["acceptance"]["measured_by"]
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_undeclared_target_field_is_refused():
    d = base(); d["targets"][0]["reason"] = "why"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_install_that_wrote_nothing_is_refused():
    d = base(); d["targets"] = []
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_non_object_is_refused():
    for bad in ([], "text", 7, None):
        with pytest.raises(AttestationSchemaError):
            validate(bad)


def test_malformed_json_is_refused():
    with pytest.raises(AttestationSchemaError):
        AttestationDocument.from_json("{not json")


# ---------------------------------------------------------------------------
# 3. Cross-field refusals -- an attestation may not contradict itself
# ---------------------------------------------------------------------------

def test_the_counter_delta_must_equal_the_identity_delta():
    d = base(); d["counter"]["after"] = 103
    d["suite_transition"]["after_count"] = 103
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "delta" in str(exc.value)


def test_the_transition_counts_must_equal_the_counter_counts():
    d = base(); d["suite_transition"]["before_count"] = 99
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_the_gate_must_account_for_exactly_the_collected_count():
    """Two measurements of one suite. Disagreement is not evidence."""
    d = base(); d["acceptance"]["skipped"] = 6
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "two measurements" in str(exc.value)


def test_expected_and_observed_identities_must_agree():
    d = base()
    d["suite_transition"]["expected_added_nodeids"] = [ADDED[0]]
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_identity_may_not_be_both_added_and_removed():
    d = base()
    d["counter"] = {"scope": "tests", "before": 100, "after": 100}
    d["suite_transition"].update(
        expected_removed_nodeids=list(ADDED),
        observed_removed_nodeids=list(ADDED), after_count=100)
    d["acceptance"].update(passed=93, summary="93 passed, 7 skipped")
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_neutral_transition_may_not_change_the_identity_digest():
    """The exact defect: equal counts, different suite."""
    d = base()
    d["counter"] = {"scope": "tests", "before": 100, "after": 100}
    d["suite_transition"].update(
        kind="neutral", expected_added_nodeids=[], observed_added_nodeids=[],
        after_count=100, before_digest="c" * 64, after_digest="d" * 64)
    d["acceptance"].update(passed=93, summary="93 passed, 7 skipped")
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "digest" in str(exc.value)


def test_a_neutral_transition_may_not_record_added_identities():
    d = base(); d["suite_transition"]["kind"] = "neutral"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_addition_may_not_record_a_removal():
    d = base()
    d["suite_transition"].update(
        expected_removed_nodeids=["tests/unit/test_y.py::test_z"],
        observed_removed_nodeids=["tests/unit/test_y.py::test_z"],
        after_count=101)
    d["counter"]["after"] = 101
    d["acceptance"]["passed"] = 94
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_retirement_requires_a_justification():
    d = base()
    d["counter"] = {"scope": "tests", "before": 100, "after": 98}
    d["suite_transition"].update(
        kind="deliberate_retirement", expected_added_nodeids=[],
        observed_added_nodeids=[], expected_removed_nodeids=list(ADDED),
        observed_removed_nodeids=list(ADDED), after_count=98)
    d["acceptance"].update(passed=91, summary="91 passed, 7 skipped")
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_unknown_transition_kind_is_refused():
    d = base(); d["suite_transition"]["kind"] = "tidy_up"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_an_unknown_status_is_refused():
    d = base(); d["status"] = "DONE"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_publication_error_without_the_pending_status_is_refused():
    d = base(); d["publication_error"] = "something"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_the_pending_status_without_an_error_is_refused():
    d = base(); d["status"] = "INSTALL_APPLIED_PUBLICATION_PENDING"
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_published_over_a_failing_gate_is_refused():
    """A green record over a red gate is the defect this project keeps finding."""
    d = base(); d["acceptance"]["returncode"] = 1
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_validation_does_not_mutate_the_document():
    d = base(); before = copy.deepcopy(d)
    validate(d)
    assert d == before


# ---------------------------------------------------------------------------
# 4. Version-3 typing -- a field whose SHAPE is part of its meaning
# ---------------------------------------------------------------------------
#
# ATTESTATION-V2-STRUCTURAL-TYPING-INCOMPLETE-1. Version 2 enforced cross-field
# consistency and almost nothing about primitive types. An audit on 2026-08-26
# applied the typing `install_attestation_reconstruction.py` had used since
# 2026-08-25 to the preserved corpus:
#
#     version-2 documents preserved   8
#     their ONLY typing failure       repository.pre_head, post_head
#     every other typed field         already conformed
#
# So the corpus was already fully typed but for one field pair -- and MEASURED
# across the delivered installers, 102 `rev-parse --short` call sites and ZERO
# full ones. No producer had ever captured a full object identifier.
#
# Version 3 RECORDS BOTH, and binds them: the abbreviation must be a PREFIX of
# the full identifier. Two independently recorded fields would double the
# surface for a wrong value while proving nothing; the prefix relationship is
# what makes the pair evidence.


def test_a_version_two_document_is_no_longer_judged():
    """Eight of them are preserved. They join version 1 as history.

    The sibling of `test_version_one_documents_are_not_judged_by_this_schema`,
    and it exists because version 2 stopped being the judged version on
    2026-08-26 -- a fact with no test until this one.
    """
    d = base(); d["schema_version"] = 2
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "historical evidence" in str(exc.value)


def test_the_abbreviation_must_be_a_prefix_of_the_full_identifier():
    """The rule that makes recording both worth doing."""
    d = base(); d["repository"]["pre_head"] = "deadbee"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "not a prefix" in str(exc.value)


def test_the_post_abbreviation_must_also_be_a_prefix():
    d = base(); d["repository"]["post_head"] = "deadbee"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "not a prefix" in str(exc.value)


def test_an_abbreviation_where_the_full_identifier_belongs_is_refused():
    """The exact shape version 2 accepted for two years of attestations."""
    d = base(); d["repository"]["pre_head_oid"] = "a60f18f"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "40-character" in str(exc.value)


def test_a_pending_install_may_record_neither_post_identifier():
    """It never committed, and recording null for both is the truth."""
    d = base()
    d["status"] = "INSTALL_APPLIED_PUBLICATION_PENDING"
    d["publication_error"] = "git add exited 128"
    d["repository"]["post_head"] = None
    d["repository"]["post_head_oid"] = None
    validate(d)


def test_one_null_post_identifier_beside_one_value_is_refused():
    """A state that cannot exist: half-committed."""
    d = base()
    d["status"] = "INSTALL_APPLIED_PUBLICATION_PENDING"
    d["publication_error"] = "git add exited 128"
    d["repository"]["post_head"] = None
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "cannot exist" in str(exc.value)


def test_a_published_document_may_not_have_a_null_post_identifier():
    d = base()
    d["repository"]["post_head"] = None
    d["repository"]["post_head_oid"] = None
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "has a commit" in str(exc.value)


def test_an_unformatted_timestamp_is_refused():
    d = base(); d["started_at"] = "2026-08-26 07:00:00"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "YYYY-MM-DD" in str(exc.value)


def test_a_digest_that_is_not_sixty_four_characters_is_refused():
    """FABRICATED-DIGEST-3 was a real 16-character prefix with 48 invented
    characters. A prefix is not a digest, and the schema now says so."""
    d = base(); d["plan_digest"] = "0" * 16
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "64-character" in str(exc.value)


def test_an_uppercase_digest_is_refused():
    """One digest, one spelling. Two would not compare equal."""
    d = base(); d["plan_digest"] = "A" * 64
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_suite_identity_digest_must_be_sixty_four_characters():
    d = base(); d["suite_transition"]["before_digest"] = "a" * 40
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_count_that_is_a_string_is_refused():
    d = base(); d["counter"]["before"] = "100"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "expected an integer" in str(exc.value)


def test_a_boolean_is_not_a_count():
    """`True` is an int in Python, and it is not a number of tests."""
    d = base(); d["acceptance"]["failed"] = True
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "expected an integer" in str(exc.value)


def test_seconds_that_are_a_string_are_refused():
    d = base(); d["acceptance"]["seconds"] = "12.5"
    with pytest.raises(AttestationSchemaError) as exc:
        validate(d)
    assert "expected a number" in str(exc.value)


def test_a_target_digest_of_the_wrong_length_is_refused():
    d = base(); d["targets"][0]["post_sha256"] = "1" * 40
    with pytest.raises(AttestationSchemaError):
        validate(d)


def test_a_target_size_that_is_a_string_is_refused():
    d = base(); d["targets"][0]["post_size"] = "10"
    with pytest.raises(AttestationSchemaError):
        validate(d)
