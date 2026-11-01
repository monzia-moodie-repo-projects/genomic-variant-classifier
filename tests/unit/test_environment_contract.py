"""Tests for deps/environment_contract.py: each rule refuses exactly what it exists to refuse. Author: Monzia Moodie"""
from __future__ import annotations

import hashlib
import json
import logging

import pytest
from packaging.markers import default_environment

from genomic_variant_classifier.deps.environment_contract import (
    ContractRefusal, IssueCode, closure_issues, complete_marker_environment, inventory_issues, read_verified_json,
    target_issues, verify,
)

logger = logging.getLogger(__name__)

LINUX = {"implementation_name": "cpython", "implementation_version": "3.12.3", "os_name": "posix", "platform_machine": "x86_64",
         "platform_release": "6.8.0", "platform_system": "Linux", "platform_version": "#1 SMP", "platform_python_implementation": "CPython",
         "python_full_version": "3.12.3", "python_version": "3.12", "sys_platform": "linux"}


def row(name, version, requires=(), extras=(), path=None, requires_python=None):
    return {"name": name, "version": version, "requires_dist": list(requires), "provides_extra": list(extras),
            "metadata_path": path or "/site/{}-{}.dist-info".format(name, version), "requires_python": requires_python}


def observation(rows, *, venv=True, cfg="include-system-site-packages = false\n", pip_exit=0, env=None):
    return {"schema": "gvc.environment-observation/1", "distributions": rows, "marker_environment": dict(env or LINUX),
            "interpreter": {"is_virtual_environment": venv, "pyvenv_cfg": cfg, "prefix": "/v"},
            "pip_check": {"exit_code": pip_exit, "stdout": "No broken requirements found.\n" if pip_exit == 0 else "x 1 requires y"}}


def codes(issues):
    return [(i.code, i.distribution) for i in issues]


def test_the_fixture_marker_environment_is_complete():
    # Guards the fixtures themselves: a test of target rules on an incomplete environment would test the refusal instead.
    assert set(LINUX) == set(default_environment())


# --- inventory -------------------------------------------------------------------------------------------------------------
def test_exact_inventory_is_satisfied():
    assert inventory_issues([row("a", "1.0"), row("B_c", "2.0")], {"a": "1.0", "b-c": "2.0"}) == []


def test_same_version_duplicate_metadata_is_refused():
    # The owner's .venv312 (2026-10-31): the project's dist-info AND src/...egg-info, both 0.1.0, and pip check passed.
    rows = [row("genomic-variant-classifier", "0.1.0", path="/site/x.dist-info"), row("genomic_variant_classifier", "0.1.0", path="/src/x.egg-info")]
    issues = inventory_issues(rows, {"genomic-variant-classifier": "0.1.0"})
    assert codes(issues) == [(IssueCode.DUPLICATE_METADATA, "genomic-variant-classifier")]
    assert "/src/x.egg-info" in issues[0].detail


@pytest.mark.parametrize("rows, expected, code", [
    ([row("a", "1.0")], {"a": "1.0", "b": "1.0"}, IssueCode.MISSING),
    ([row("a", "1.1")], {"a": "1.0"}, IssueCode.VERSION_MISMATCH),
    ([row("a", "1.0"), row("faker", "40.0")], {"a": "1.0"}, IssueCode.UNEXPECTED),
])
def test_each_inventory_difference_is_named(rows, expected, code):
    assert [i.code for i in inventory_issues(rows, expected)] == [code]


def test_version_equality_is_semantic_but_local_labels_are_not_dropped():
    assert inventory_issues([row("a", "1.0.0")], {"a": "1.0"}) == []                       # PEP 440: 1.0.0 == 1.0
    assert [i.code for i in inventory_issues([row("torch", "2.11.0+cpu")], {"torch": "2.11.0"})] == [IssueCode.VERSION_MISMATCH]


def test_forbidden_distribution_is_named_even_when_expected():
    assert IssueCode.FORBIDDEN in [i.code for i in inventory_issues([row("nannyml", "0.13.1")], {"nannyml": "0.13.1"}, forbidden=["NannyML"])]


@pytest.mark.parametrize("bad", [[], [{"version": "1.0"}], [row("a", "not a version")]])
def test_malformed_inventory_is_a_refusal_not_a_verdict(bad):
    with pytest.raises(ContractRefusal):
        inventory_issues(bad, {"a": "1.0"})


# --- closure ---------------------------------------------------------------------------------------------------------------
def test_closure_follows_selected_extras_only():
    rows = [row("app", "1.0", requires=["lib[fast]>=1"]), row("lib", "1.2", requires=["speed; extra == 'fast'", "slow; extra == 'slow'"], extras=["fast", "slow"])]
    issues, evaluated = closure_issues(rows, ["app"], LINUX)
    assert codes(issues) == [(IssueCode.DEPENDENCY_MISSING, "lib")]                      # speed is needed; slow is not
    assert ("speed; extra == 'fast'", "fast") in evaluated and ("slow; extra == 'slow'", "slow") not in evaluated


def test_closure_names_version_extra_and_python_problems():
    rows = [row("app", "1.0", requires=["lib[ghost]>=2"]), row("lib", "1.0", requires_python=">=3.13")]
    found = {i.code for i in closure_issues(rows, ["app"], LINUX)[0]}
    assert found == {IssueCode.DEPENDENCY_VERSION, IssueCode.UNDECLARED_EXTRA, IssueCode.REQUIRES_PYTHON}


def test_closure_refuses_a_direct_url_dependency_as_an_issue():
    issues, _ = closure_issues([row("app", "1.0", requires=["lib @ https://example.invalid/lib.whl"])], [], LINUX)
    assert [i.code for i in issues] == [IssueCode.DIRECT_URL_DEPENDENCY]


# --- target ----------------------------------------------------------------------------------------------------------------
def test_an_operating_system_update_is_provenance_not_a_refusal():
    host = dict(LINUX, platform_version="#2 SMP", platform_release="6.8.1")
    issues, changes = target_issues(LINUX, host, {("x; sys_platform == 'linux'", "")}, python_policy="==3.12.*")
    assert issues == [] and set(changes) == {"platform_version", "platform_release"}


def test_a_marker_that_reads_the_os_build_changes_outcome_and_is_refused():
    marker = "x; platform_release == '6.8.0'"
    issues, _ = target_issues(LINUX, dict(LINUX, platform_release="6.8.1"), {(marker, "")}, python_policy="==3.12.*")
    assert [i.code for i in issues] == [IssueCode.MARKER_OUTCOME_CHANGED]


def test_target_family_and_python_policy():
    win = dict(LINUX, os_name="nt", platform_system="Windows", sys_platform="win32", platform_machine="AMD64")
    issues, _ = target_issues(LINUX, win, set(), python_policy="==3.12.*")
    assert {i.distribution for i in issues if i.code is IssueCode.TARGET_FAMILY_CHANGED} == {"os_name", "platform_system", "sys_platform", "platform_machine"}
    issues, _ = target_issues(LINUX, LINUX, set(), python_policy="==3.11.*")
    assert [i.code for i in issues] == [IssueCode.PYTHON_OUTSIDE_POLICY]


@pytest.mark.parametrize("env", [dict(LINUX, extra=""), {k: v for k, v in LINUX.items() if k != "sys_platform"},
                                 dict(LINUX, python_version="3.11"), dict(LINUX, os_name="")])
def test_an_incomplete_or_inconsistent_target_is_refused(env):
    with pytest.raises(ContractRefusal):
        complete_marker_environment(env)


# --- evidence --------------------------------------------------------------------------------------------------------------
def test_evidence_is_read_only_at_the_vouched_digest(tmp_path):
    p = tmp_path / "o.json"
    p.write_bytes(b'{"a": 1}')
    assert read_verified_json(p, hashlib.sha256(b'{"a": 1}').hexdigest()) == {"a": 1}
    for raw, digest in [(b'{"a": 1}', "0" * 64), (b'{"a": 1, "a": 2}', None), (b'{"a": NaN}', None)]:
        p.write_bytes(raw)
        with pytest.raises(ContractRefusal):
            read_verified_json(p, digest or hashlib.sha256(raw).hexdigest())
    with pytest.raises(ContractRefusal):
        read_verified_json(p, "ABC")


# --- the whole contract ----------------------------------------------------------------------------------------------------
def test_verify_satisfied_and_each_environment_level_refusal():
    rows = [row("app", "1.0", requires=["lib>=1"]), row("lib", "1.0"), row("pip", "26.1.2")]
    ok = verify(observation(rows), {"app": "1.0", "lib": "1.0"}, auxiliary={"pip": "26.1.2"}, roots=["app"], locked_target=LINUX, python_policy="==3.12.*")
    assert ok.satisfied and json.loads(json.dumps(ok.as_dict()))["satisfied"] is True
    for kwargs, code in [({"venv": False}, IssueCode.NOT_A_VIRTUAL_ENVIRONMENT),
                         ({"cfg": "include-system-site-packages = true\n"}, IssueCode.SYSTEM_SITE_PACKAGES),
                         ({"pip_exit": 1}, IssueCode.PIP_CHECK_FAILED)]:
        v = verify(observation(rows, **kwargs), {"app": "1.0", "lib": "1.0"}, auxiliary={"pip": "26.1.2"}, roots=["app"], locked_target=LINUX, python_policy="==3.12.*")
        assert [i.code for i in v.issues] == [code]


def test_an_auxiliary_tool_cannot_also_be_locked():
    with pytest.raises(ContractRefusal):
        verify(observation([row("pip", "26.1.2")]), {"pip": "26.1.2"}, auxiliary={"PIP": "26.1.2"}, roots=[], locked_target=LINUX, python_policy="==3.12.*")


def test_an_unknown_observation_schema_is_refused():
    o = observation([row("a", "1.0")])
    o["schema"] = "gvc.environment-observation-reference/1"
    with pytest.raises(ContractRefusal):
        verify(o, {"a": "1.0"}, auxiliary={}, roots=[], locked_target=LINUX, python_policy="==3.12.*")
