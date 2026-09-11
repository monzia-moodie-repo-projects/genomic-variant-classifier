"""Refuse an admission whose evidence contradicts itself.

Author: Monzia Moodie

WHY THIS IS EXECUTABLE AND NOT A CONVENTION
===========================================
The delivered citation report at

    f4b7383cd85c178032da996ac08ffda6fc43e49e610bf4e03891cde8ccfaebfc

contains `"repository_is_shallow": true` while its own log printed "the
repository is NOT shallow  false". The admission plan bound that digest, so
the inconsistency was part of the plan's evidence dependency.

CAUSE, ESTABLISHED 2026-09-08 by direct measurement of the delivered file at
that exact digest:

    line 228:  "repository_is_shallow": shallow == "false",

`git rev-parse --is-shallow-repository` returns the string "false" for a deep
repository, so a DEEP repository serialised `true`. The field's NAME said "is
shallow" while its VALUE meant "is not shallow". A review reported reading
`shallow == "true"` in that file; the grep above, taken at the matching
digest, shows otherwise, and that line would have produced no contradiction at
all. The cause is now read, not inferred.

A cryptographic digest establishes WHICH bytes were used. It does not make
contradictory bytes trustworthy. So the contradiction is made into a refusal.

WHAT THIS DOES NOT DO
=====================
It does not establish historical completeness, verify Git messages, or replace
the citation query. It prevents an internally inconsistent report from
authorising admission.

A JSON output file may exist after an unsuccessful probe, so file existence is
never a success signal.

EXIT STATUS when run as a script
  0  every self-test held
  2  a self-test did not hold
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_OID = re.compile(r"\A[0-9a-f]{40}\Z")
#: An abbreviated commit identifier. Git's default is seven characters
#: and it never abbreviates below four; an empty string is not one.
_SHORT_OID = re.compile(r"\A[0-9a-f]{4,40}\Z")

#: The status field this validator understands. `overall_status` was the
#: earlier name and conflated the check outcome with the process exit; both
#: are accepted for reading, and neither is treated as a process result.
_STATUS_KEYS = ("checks_status", "overall_status")


class EvidenceError(ValueError):
    """The evidence cannot authorise anything. Never repaired here."""


def no_duplicate_keys(pairs):
    """A duplicate JSON key silently discards one of two conflicting values."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def reject_constant(value):
    """NaN and Infinity are not JSON, and compare unequal to themselves."""
    raise EvidenceError("nonstandard JSON constant: {}".format(value))


def load_bound_json(raw: bytes, expected_sha256: str):
    """Parse ONLY after the bytes match the approved digest.

    Strict parsing: duplicate keys and nonstandard constants are refused
    rather than silently resolved.
    """
    if not _SHA256.fullmatch(expected_sha256):
        raise EvidenceError(
            "the expected digest must be 64 lowercase hexadecimal digits")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise EvidenceError(
            "evidence digest mismatch: {} is not the approved {}".format(
                observed, expected_sha256))
    return json.loads(raw.decode("utf-8"),
                      object_pairs_hook=no_duplicate_keys,
                      parse_constant=reject_constant)


CITATION_SCHEMA = "gvc.citation-derivation"
CITATION_SCHEMA_VERSION = 1


def require_current_citation_shape(report, *, expected_basenames,
                                   expected_accepted_count) -> None:
    """The CURRENT report shape, with missing fields refused, not defaulted.

    MEASURED 2026-09-08 against the previous predicate, every one ACCEPTED:

        checks_status and failed_checks removed   -> status None, accepted
        failed_checks = false                     -> falsy, accepted
        derived = {}                              -> candidates 0, accepted
        overall_status "failed" beside
            checks_status "passed"                -> first key wins, accepted
        unrelated schema, Boolean schema_version  -> never examined, accepted

    Each was a fail-OPEN branch: absence or a wrongly typed value became an
    acceptable default. A gate whose checks can be removed by deleting fields
    is not a gate.

    `expected_basenames` comes from the VERIFIED CENSUS and
    `expected_accepted_count` from the BOUND PREDECESSOR MANIFEST. Neither is
    hard-coded here: a second 88-and-18 authority would drift from the first.
    """
    if type(report) is not dict:
        raise EvidenceError("the citation report must be an object")
    if report.get("schema") != CITATION_SCHEMA:
        raise EvidenceError(
            "unsupported citation schema {!r}".format(report.get("schema")))
    version = report.get("schema_version")
    if type(version) is not int or version != CITATION_SCHEMA_VERSION:
        raise EvidenceError(
            "citation schema_version must be the integer {}, not {!r} of type "
            "{}".format(CITATION_SCHEMA_VERSION, version,
                        type(version).__name__))

    # A legacy reader may inspect old reports. It must not silently upgrade
    # one into evidence sufficient for CURRENT admission.
    if "overall_status" in report:
        raise EvidenceError(
            "the legacy status field overall_status is not authorised here; a "
            "report carrying both representations can disagree with itself")
    if report.get("checks_status") != "passed":
        raise EvidenceError(
            "a successful check status is required, not {!r}".format(
                report.get("checks_status")))
    failures = report.get("failed_checks")
    if type(failures) is not list or failures:
        raise EvidenceError(
            "failed_checks must be an EMPTY LIST, not {!r} of type {}".format(
                failures, type(failures).__name__))

    derived = report.get("derived")
    if type(derived) is not dict:
        raise EvidenceError(
            "derived citations must be an object, not {}".format(
                type(derived).__name__))
    expected = set(expected_basenames)
    if set(derived) != expected:
        raise EvidenceError(
            "derived citation membership differs from the census. MISSING: "
            "{}. UNEXPECTED: {}.".format(
                sorted(expected - set(derived))[:5] or "none",
                sorted(set(derived) - expected)[:5] or "none"))

    for field in ("accepted_records_in_manifest", "accepted_records_examined",
                  "accepted_records_reproduced"):
        value = report.get(field)
        if type(value) is not int or value != expected_accepted_count:
            raise EvidenceError(
                "{} must be the integer {}, not {!r}".format(
                    field, expected_accepted_count, value))
    for field in ("accepted_records_differing",
                  "accepted_records_skipped_no_alias",
                  "candidates_without_citation"):
        value = report.get(field)
        if type(value) is not list or value:
            raise EvidenceError(
                "{} must be an EMPTY LIST, not {!r} of type {}".format(
                    field, value, type(value).__name__))

    # PER-ROW structure. A membership check says which names are present; it
    # says nothing about whether their citations are usable.
    for basename in sorted(derived):
        row = derived[basename]
        if type(row) is not dict:
            raise EvidenceError("{}: citation row must be an object".format(
                basename))
        shorts = row.get("cited_by")
        oids = row.get("cited_by_oids")
        if type(shorts) is not list or not shorts:
            raise EvidenceError(
                "{}: cited_by must be a nonempty list".format(basename))
        if type(oids) is not list or len(oids) != len(shorts):
            raise EvidenceError(
                "{}: cited_by_oids must be a list of the same length as "
                "cited_by".format(basename))
        if len(set(shorts)) != len(shorts):
            raise EvidenceError(
                "{}: duplicate citing commit".format(basename))
        for short, oid in zip(sorted(shorts), sorted(oids)):
            if type(oid) is not str or not _OID.fullmatch(oid):
                raise EvidenceError(
                    "{}: a citing commit identifier must be 40 lowercase "
                    "hexadecimal digits".format(basename))
            # MEASURED 2026-09-08: `oid.startswith("")` is True, so an EMPTY
            # short citation passed. The abbreviation's own syntax must be
            # checked BEFORE it is used as a prefix.
            if type(short) is not str or not _SHORT_OID.fullmatch(short):
                raise EvidenceError(
                    "{}: {!r} is not a supported abbreviated commit "
                    "identifier".format(basename, short))
            if not oid.startswith(short):
                raise EvidenceError(
                    "{}: {!r} is not a prefix of {!r}".format(
                        basename, short, oid))


def require_citation_evidence(report, *, predecessor: str,
                              census_sha256: str) -> dict:
    """Every field checked by TYPE and value, with no truthiness anywhere.

    `report["repository_is_shallow"]` must be a Boolean and must be False.
    A string "false" is truthy in Python and would pass a naive check, which
    is the same class of defect as the one that produced this validator.
    """
    if not isinstance(report, dict):
        raise EvidenceError("the citation report is not an object")

    shallow = report.get("repository_is_shallow")
    if type(shallow) is not bool:
        raise EvidenceError(
            "shallow status must be a Boolean, not {!r} of type {}".format(
                shallow, type(shallow).__name__))
    if shallow:
        raise EvidenceError(
            "the citation evidence declares a SHALLOW repository. A shallow "
            "clone returns few or no citations while exiting zero.")

    head = report.get("measured_at_head")
    if not isinstance(head, str) or not _OID.fullmatch(head):
        raise EvidenceError(
            "measured_at_head must be 40 lowercase hexadecimal digits")
    if head != predecessor:
        raise EvidenceError(
            "citation predecessor mismatch: measured at {} and approved for "
            "{}".format(head, predecessor))

    bound = report.get("census_sha256")
    if bound != census_sha256:
        raise EvidenceError(
            "citation census binding mismatch: {} is not the approved "
            "{}".format(bound, census_sha256))

    differing = report.get("accepted_records_differing")
    if differing != []:
        raise EvidenceError(
            "accepted citation discrepancies remain: {}".format(
                differing if differing is not None else "field absent"))

    uncited = report.get("candidates_without_citation")
    if uncited != []:
        raise EvidenceError(
            "this batch requires every candidate to be cited; {}".format(
                "field absent" if uncited is None
                else "{} without citation".format(len(uncited))))

    # THE SELF-CONSISTENCY THE ORIGINAL REPORT LACKED. A status saying the
    # checks passed, beside a list saying which failed, is exactly the shape
    # this validator exists to refuse.
    status = None
    for key in _STATUS_KEYS:
        if key in report:
            status = report[key]
            break
    failed = report.get("failed_checks")
    if status is not None:
        if status not in ("passed", "failed"):
            raise EvidenceError(
                "unsupported checks status {!r}".format(status))
        if status == "failed":
            raise EvidenceError(
                "the citation report records its own checks as FAILED: "
                "{}".format(failed))
        if failed:
            raise EvidenceError(
                "the report claims its checks passed while listing failures: "
                "{}".format(failed))

    return {"predecessor": head, "census_sha256": bound,
            "checks_status": status,
            "candidates": len(report.get("derived") or {})}
