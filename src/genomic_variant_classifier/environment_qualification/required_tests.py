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
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["QualificationError", "Case", "admit_junit", "admit_qualification_rows", "admit_loaded_namespaces"]


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


def admit_qualification_rows(expected, observed) -> dict:
    """EXACT membership of a replay's qualification rows (owner ruling 2026-10-07b, reference L308-349): every expected (package, version,
    role) observed EXACTLY ONCE with status "OK". The earlier replay check compared only the row COUNT and the statuses, so a duplicated row
    could replace a missing one without changing the count.

    expected: iterable of (package, version, role); observed: iterable of (package, version, role, status, details)."""
    expected = tuple(tuple(row) for row in expected)
    observed = tuple(tuple(row) for row in observed)
    if not expected:
        raise QualificationError("qualification.expected_empty")
    if any(len(row) != 3 or any(type(v) is not str or not v for v in row) for row in expected):
        raise QualificationError("qualification.expected_shape")
    names = [row[0] for row in expected]
    if len(names) != len(set(names)):
        raise QualificationError("qualification.expected_duplicate")
    if any(len(row) != 5 or any(type(v) is not str for v in row) for row in observed):
        raise QualificationError("qualification.observed_shape")
    if Counter(row[:3] for row in observed) != Counter(expected):
        raise QualificationError("qualification.identity_mismatch")
    if any(row[3] != "OK" for row in observed):
        raise QualificationError("qualification.result_failed")
    return {"qualified": len(expected), "identities": sorted(expected)}


def admit_loaded_namespaces(identity, *, expected_r_version: str, expected_platform: str, expected: dict, approved_roots: dict,
                            r_home: str, base_packages: frozenset, case_insensitive_paths: bool) -> dict:
    """What the fixture process ACTUALLY loaded, against INDEPENDENT expectations (owner ruling 2026-10-07b, section 3): the runtime
    version and platform; every expected namespace loaded; every loaded non-base namespace expected, at its exact version, resolved EXACTLY
    at <approved root>/<package>; every base namespace inside R_HOME/library. Approved roots are supplied PER EXECUTION (content identity
    vs execution location, section 7). Case-insensitive path comparison only when the caller declares the platform's paths so (Windows).

    identity: the parsed in-process record {"r_version", "platform", "loaded_namespaces": [{"package", "version", "path"}]}.
    expected: package -> version; approved_roots: package -> the directory its namespace must resolve under."""
    check(isinstance(identity, dict), "identity.shape")
    check(identity.get("r_version") == expected_r_version, "identity.r_version")
    check(identity.get("platform") == expected_platform, "identity.platform")
    rows = identity.get("loaded_namespaces")
    check(isinstance(rows, list) and len(rows) > 0, "identity.namespaces_shape")
    check(isinstance(expected, dict) and len(expected) > 0 and set(approved_roots) == set(expected), "identity.expectation_shape")
    norm = (lambda x: x.replace("\\", "/").rstrip("/").casefold()) if case_insensitive_paths else (lambda x: x.replace("\\", "/").rstrip("/"))
    loaded = {}
    for row in rows:
        check(isinstance(row, dict) and all(type(row.get(k)) is str and row.get(k) for k in ("package", "version", "path")), "identity.row_shape")
        check(row["package"] not in loaded, "identity.duplicate_namespace:" + row["package"])
        loaded[row["package"]] = row
    missing = sorted(set(expected) - set(loaded))
    check(not missing, "identity.required_not_loaded:" + ",".join(missing))
    base_root = norm(r_home) + "/library"
    for name, row in sorted(loaded.items()):
        if name in base_packages:
            check(norm(row["path"]) == base_root + "/" + norm(name), "identity.base_location:" + name)
            continue
        check(name in expected, "identity.unexpected_namespace:" + name)
        check(row["version"] == expected[name], "identity.version:" + name)
        check(norm(row["path"]) == norm(approved_roots[name]) + "/" + norm(name), "identity.location:" + name)
    return {"loaded": len(loaded), "expected": len(expected), "base": sorted(set(loaded) & set(base_packages))}
