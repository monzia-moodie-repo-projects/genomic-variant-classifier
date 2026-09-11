"""Evidence authorization: fail-closed on every branch that was fail-open.

Author: Monzia Moodie

Each case below was REPRODUCED as an acceptance before the correction. The
expected names and counts come from the caller's verified census and bound
manifest, never from constants here.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from genomic_variant_classifier.operations.evidence_validator import (
    EvidenceError, load_bound_json, require_citation_evidence,
    require_current_citation_shape)

PREDECESSOR = "f" * 40
CENSUS = "c" * 64
NAMES = ["install-attestation-{}.json".format(i) for i in range(3)]
ACCEPTED = 2


def _report():
    return {"schema": "gvc.citation-derivation", "schema_version": 1,
            "checks_status": "passed", "failed_checks": [],
            "repository_is_shallow": False,
            "measured_at_head": PREDECESSOR, "census_sha256": CENSUS,
            "accepted_records_differing": [],
            "accepted_records_skipped_no_alias": [],
            "candidates_without_citation": [],
            "accepted_records_in_manifest": ACCEPTED,
            "accepted_records_examined": ACCEPTED,
            "accepted_records_reproduced": ACCEPTED,
            "derived": {n: {"cited_by": ["abc1234"],
                            "cited_by_oids": ["abc1234" + "0" * 33]}
                        for n in NAMES}}


def _shape(report):
    require_current_citation_shape(report, expected_basenames=NAMES,
                                   expected_accepted_count=ACCEPTED)


def test_a_well_formed_report_is_accepted():
    _shape(_report())
    summary = require_citation_evidence(_report(), predecessor=PREDECESSOR,
                                        census_sha256=CENSUS)
    assert summary["candidates"] == 3


@pytest.mark.parametrize("mutate,label", [
    (lambda d: (d.pop("checks_status"), d.pop("failed_checks")), "no status"),
    (lambda d: d.update(failed_checks=False), "failed_checks false"),
    (lambda d: d.update(failed_checks=None), "failed_checks null"),
    (lambda d: d.update(failed_checks={}), "failed_checks object"),
    (lambda d: d.update(failed_checks=["x"]), "failed_checks non-empty"),
    (lambda d: d.update(derived={}), "derived empty"),
    (lambda d: d.update(derived=None), "derived null"),
    (lambda d: d.update(overall_status="failed"), "legacy status present"),
    (lambda d: d.update(schema="unrelated"), "unrelated schema"),
    (lambda d: d.update(schema_version=True), "boolean schema version"),
    (lambda d: d.update(schema_version=1.0), "float schema version"),
    (lambda d: d.update(checks_status="failed"), "checks failed"),
    (lambda d: d.update(checks_status="probably fine"), "unsupported status"),
    (lambda d: d.update(accepted_records_examined=1), "wrong examined count"),
    (lambda d: d.update(accepted_records_differing=[{"x": 1}]), "a discrepancy"),
    (lambda d: d.update(candidates_without_citation=["a"]), "an uncited row"),
    (lambda d: d.update(accepted_records_skipped_no_alias=["a"]), "a skip"),
])
def test_a_malformed_current_report_is_refused(mutate, label):
    report = _report()
    mutate(report)
    with pytest.raises(EvidenceError):
        _shape(report)


def test_derived_membership_must_equal_the_census():
    report = _report()
    report["derived"].pop(NAMES[0])
    with pytest.raises(EvidenceError):
        _shape(report)
    report = _report()
    report["derived"]["extra.json"] = {"cited_by": ["abc1234"],
                                       "cited_by_oids": ["abc1234" + "0" * 33]}
    with pytest.raises(EvidenceError):
        _shape(report)


@pytest.mark.parametrize("row", [
    {"cited_by": [], "cited_by_oids": []},
    {"cited_by": ["abc1234"], "cited_by_oids": ["z" * 40]},
    {"cited_by": ["0000000"], "cited_by_oids": ["abc1234" + "0" * 33]},
    {"cited_by": ["abc1234"], "cited_by_oids": []},
    {"cited_by": ["abc1234", "abc1234"],
     "cited_by_oids": ["abc1234" + "0" * 33, "abc1234" + "0" * 33]},
])
def test_a_malformed_citation_row_is_refused(row):
    report = _report()
    report["derived"][NAMES[0]] = row
    with pytest.raises(EvidenceError):
        _shape(report)


@pytest.mark.parametrize("value", ["false", 0, None, 1])
def test_a_shallow_status_that_is_not_a_boolean_is_refused(value):
    """`"false"` is TRUTHY in Python and would pass a naive check."""
    report = _report()
    report["repository_is_shallow"] = value
    with pytest.raises(EvidenceError):
        require_citation_evidence(report, predecessor=PREDECESSOR,
                                  census_sha256=CENSUS)


def test_a_shallow_repository_is_refused():
    report = _report()
    report["repository_is_shallow"] = True
    with pytest.raises(EvidenceError):
        require_citation_evidence(report, predecessor=PREDECESSOR,
                                  census_sha256=CENSUS)


@pytest.mark.parametrize("field,value", [
    ("measured_at_head", "0" * 40),
    ("measured_at_head", "fdb5473"),
    ("census_sha256", "0" * 64),
])
def test_a_binding_mismatch_is_refused(field, value):
    report = _report()
    report[field] = value
    with pytest.raises(EvidenceError):
        require_citation_evidence(report, predecessor=PREDECESSOR,
                                  census_sha256=CENSUS)


def test_the_digest_is_checked_before_parsing():
    with pytest.raises(EvidenceError):
        load_bound_json(b'{"a": 1}', "0" * 64)


def test_an_abbreviated_expected_digest_is_refused():
    with pytest.raises(EvidenceError):
        load_bound_json(b"{}", "abc123")


def test_a_duplicate_json_key_is_refused():
    raw = b'{"a": 1, "a": 2}'
    with pytest.raises(EvidenceError):
        load_bound_json(raw, hashlib.sha256(raw).hexdigest())


def test_a_nonstandard_json_constant_is_refused():
    raw = b'{"a": NaN}'
    with pytest.raises(EvidenceError):
        load_bound_json(raw, hashlib.sha256(raw).hexdigest())


def test_a_well_formed_document_parses_at_its_digest():
    raw = (json.dumps(_report(), sort_keys=True)).encode("utf-8")
    assert load_bound_json(raw, hashlib.sha256(raw).hexdigest())["schema"] \
        == "gvc.citation-derivation"
