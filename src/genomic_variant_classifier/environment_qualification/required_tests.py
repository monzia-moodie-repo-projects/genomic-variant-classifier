"""The required-test OUTCOME gate over a dedicated JUnit report (owner ruling 2026-10-05b).

A green overall suite can coexist with skipped R tests: the 13-stage validations on the owner's machine ran with Rscript absent from
PATH, so the R-backed tests SKIPPED there. Environmental qualification therefore admits a dedicated run only when EVERY case of a
frozen, reviewed set executed and passed -- skipped, missing, duplicated, substituted, failed or errored cases, a malformed report
or an unsuccessful process all refuse. It is a completeness and outcome gate; it does not authenticate arbitrary XML or prove that
the tests are scientifically adequate.

THE REQUIRED SET COMES FROM THE REVIEWED PLAN, never from the report being admitted (the principle operations/admission_verifier.py
applies to suite COLLECTION identity; this gate judges OUTCOMES).

MEASURED pytest/JUnit mapping (this project's report, 7,464 cases, 2026-10-06): root <testsuites> holding one <testsuite>; each
<testcase> carries classname / name / time; classname = the dotted module path ("tests.unit.test_inference_backend_trace"), plus
".TestClass" for class-nested tests; name = the function name including parametrised ids ("test_x[...]"); a skip is a <skipped>
child -- an expected failure is also <skipped> -- so run the qualification with `-o xfail_strict=true` (pyproject.toml does not set it).

The code below is the owner's reference (ruling generation 19ed556a, lines 575-649), unchanged apart from the project conventions.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["QualificationError", "Case", "admit_junit"]


class QualificationError(ValueError):
    pass


def check(ok, code):
    if not ok:
        raise QualificationError(code)


@dataclass(frozen=True, order=True)
class Case:
    classname: str
    name: str

    def __post_init__(self):
        check(
            type(self.classname) is str and bool(self.classname),
            "case_classname",
        )
        check(type(self.name) is str and bool(self.name), "case_name")


def admit_junit(xml_bytes, *, expected_cases, process_exit_code):
    """Admit a dedicated qualification run.

    expected_cases comes from the reviewed plan, never from this report.
    Run pytest with -o xfail_strict=true.
    """
    check(
        type(process_exit_code) is int and process_exit_code == 0,
        "test_process_failed",
    )
    check(type(xml_bytes) is bytes, "report_bytes")
    check(0 < len(xml_bytes) <= 8 * 1024 * 1024, "report_size")
    check(
        type(expected_cases) is frozenset and bool(expected_cases),
        "expected_cases_required",
    )
    check(
        all(type(case) is Case for case in expected_cases),
        "expected_case_type",
    )

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise QualificationError("report_xml") from exc

    check(root.tag in {"testsuite", "testsuites"}, "report_root")
    check(not list(root.iter("error")), "test_error")
    check(not list(root.iter("failure")), "test_failure")
    check(not list(root.iter("skipped")), "test_skipped")

    actual = set()
    for element in root.iter("testcase"):
        case = Case(element.get("classname"), element.get("name"))
        check(case not in actual, "duplicate_case")
        actual.add(case)

    check(actual == expected_cases, "case_set_mismatch")

    return {
        "schema": "gvc.required-test-evidence/1",
        "report_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "required_cases_passed": len(actual),
        "case_ids": [
            {"classname": case.classname, "name": case.name}
            for case in sorted(actual)
        ],
    }
