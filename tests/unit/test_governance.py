"""Tests for the measured dependency classifications.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import pytest

from genomic_variant_classifier.deps.governance import (
    CENSUS_2026_08_13, CLASSIFICATIONS, ClassifiedDependency, EvidenceStrength,
    ScopeEvidence, classification_for, required_by, resolved_classifications,
    unresolved_classifications,
)
from genomic_variant_classifier.deps.model import (
    Capability, DeploymentProfile, DistributionName,
)


# ---- every claim carries its evidence -----------------------------------
@pytest.mark.parametrize("c", CLASSIFICATIONS, ids=lambda c: c.name)
def test_every_classification_cites_at_least_two_observations(c):
    """A classification without a date and an instrument is an opinion.

    A stale limitation phrased in the present tense became a false premise in
    this repository once already; the fix is that every claim carries the
    measurement that produced it.
    """
    assert len(c.evidence) >= 2, c.name
    for e in c.evidence:
        assert e.measured_on, c.name
        assert e.method, c.name
        assert len(e.detail) > 30, (c.name, e.detail)


@pytest.mark.parametrize("c", CLASSIFICATIONS, ids=lambda c: c.name)
def test_every_classification_states_a_rationale(c):
    assert len(c.intent.rationale) > 40, c.name


@pytest.mark.parametrize("c", CLASSIFICATIONS, ids=lambda c: c.name)
def test_every_classification_has_a_capability_and_a_profile(c):
    """Scope is two axes. A one-dimensional label forced the earlier mistake in
    both directions -- understating a production capability, or bloating an
    image that provably does not use the package."""
    assert c.intent.capabilities, c.name
    assert c.intent.profiles, c.name


# ---- the specific measured findings -------------------------------------
def test_seaborn_and_jinja2_are_REPORTING_not_development():
    """Both are unguarded imports in src/, with sns.set_style() executing at
    module scope -- so importing report_generator requires them."""
    for name in ("seaborn", "jinja2"):
        c = classification_for(name)
        assert c is not None, name
        assert c.intent.capabilities == frozenset({Capability.REPORTING}), name
        assert c.resolved, name


def test_reporting_packages_are_ABSENT_from_the_API_profile():
    """Established by `python -X importtime -c "import api.main"`, which showed
    no seaborn, jinja2, reports or report_generator in the import graph. That
    is why adding them to the API lock would be dependency accretion."""
    for name in ("seaborn", "jinja2"):
        c = classification_for(name)
        assert not c.intent.required_by(DeploymentProfile.API), name
        assert c.intent.required_by(DeploymentProfile.TRAINING), name


def test_pyfaidx_is_REFERENCE_SEQUENCE_not_generic_operational():
    """"Operational" would become a bag holding unrelated dependencies. The
    capability name makes "does the API need pyfaidx?" answerable."""
    c = classification_for("pyfaidx")
    assert c.intent.capabilities == frozenset({Capability.REFERENCE_SEQUENCE})
    assert c.intent.required_by(DeploymentProfile.DATA_PREP)
    assert not c.intent.required_by(DeploymentProfile.API)


def test_pre_commit_is_DEV_TOOLING_and_zero_imports_is_CORRECT():
    """Zero imports is the right census result for a console script. The
    evidence must say so explicitly, or a later reader will call it unused."""
    c = classification_for("pre-commit")
    assert c.intent.capabilities == frozenset({Capability.DEV_TOOLING})
    assert c.resolved
    strengths = {e.strength for e in c.evidence}
    assert EvidenceStrength.NON_IMPORT_CHANNEL in strengths
    assert any("UNUSED" in e.detail for e in c.evidence)


def test_httpx_and_anyio_remain_UNRESOLVED():
    """They are reached only through fastapi.testclient. Whether their versions
    belong in the tested compatibility surface has NOT been measured, and
    recording them as "transitive and misplaced" would assert more than the
    evidence carries."""
    for name in ("httpx", "anyio"):
        c = classification_for(name)
        assert c is not None, name
        assert not c.resolved, name
        assert EvidenceStrength.UNRESOLVED in {e.strength for e in c.evidence}
        assert "UNRESOLVED" in c.intent.rationale


def test_the_anyio_record_preserves_the_COMMENT_false_positive():
    """A text scan counted a mention inside a comment; the AST census did not.

    That distinction is the census's whole justification, so the record keeps
    it. An earlier version of this test mixed strings and ScopeEvidence objects
    in one comprehension and called .detail on a string -- a confused assertion
    rather than a finding.
    """
    c = classification_for("anyio")
    haystack = c.intent.rationale + " " + " ".join(e.detail for e in c.evidence)
    assert "COMMENT" in haystack, haystack
    assert "check_lock_satisfies" in haystack


# ---- resolved versus unresolved is a real partition ---------------------
def test_resolved_and_unresolved_partition_the_classifications():
    r, u = resolved_classifications(), unresolved_classifications()
    assert len(r) + len(u) == len(CLASSIFICATIONS)
    assert not (set(x.name for x in r) & set(x.name for x in u))
    assert len(u) == 2, [x.name for x in u]


def test_no_classification_is_silently_both():
    for c in CLASSIFICATIONS:
        assert isinstance(c.resolved, bool), c.name


# ---- lookup and profile queries -----------------------------------------
def test_lookup_is_by_CANONICAL_name():
    """pre-commit, pre_commit and PRE.COMMIT are one distribution."""
    for spelling in ("pre-commit", "pre_commit", "PRE.COMMIT"):
        assert classification_for(spelling) is not None, spelling


def test_lookup_returns_None_for_an_unclassified_package():
    assert classification_for("numpy") is None


def test_the_API_profile_requires_NONE_of_the_six():
    """Measured: the API imports none of them. If that changes, this test is
    the thing that should fail."""
    assert required_by(DeploymentProfile.API) == ()


def test_the_developer_profile_requires_all_six():
    assert len(required_by(DeploymentProfile.DEVELOPER)) == len(CLASSIFICATIONS)


def test_the_census_provenance_is_recorded():
    """A re-run should be COMPARED against this, not silently replace it."""
    assert CENSUS_2026_08_13["files_walked"] == 941
    assert CENSUS_2026_08_13["files_parsed"] == 941
    assert CENSUS_2026_08_13["parse_failures"] == 0
    assert CENSUS_2026_08_13["roots"] == ("src", "scripts", "tests")


# ---- immutability -------------------------------------------------------
def test_a_classification_is_immutable():
    import dataclasses
    try:
        CLASSIFICATIONS[0].resolved = False
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("a classification was mutable")


def test_evidence_is_immutable():
    import dataclasses
    try:
        CLASSIFICATIONS[0].evidence[0].detail = "rewritten"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("evidence was mutable")


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except TypeError:
            print("  SKIP  {} (parametrized)".format(name))
        except Exception as exc:                        # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} failed".format(len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
