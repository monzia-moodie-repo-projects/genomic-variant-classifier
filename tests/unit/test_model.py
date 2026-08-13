"""Tests for the shared dependency vocabulary.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import pytest

from genomic_variant_classifier.deps.model import (
    ArtifactAuthority, ArtifactFreshness, ArtifactPurpose, Capability,
    DependencyIntent, DeploymentProfile, DistributionImportMapping,
    DistributionName, ImportName, MappingSource,
)


# ---- distribution identity ---------------------------------------------
@pytest.mark.parametrize("raw,canonical", [
    ("foo-bar", "foo-bar"),
    ("foo_bar", "foo-bar"),
    ("foo.bar", "foo-bar"),
    ("FOO--BAR", "foo-bar"),
    ("pyBigWig", "pybigwig"),
    ("Jinja2", "jinja2"),
    ("zope.interface", "zope-interface"),
    ("backports_abc", "backports-abc"),
])
def test_distribution_names_canonicalise(raw, canonical):
    """MEASURED with packaging 26.0. Six of ten sampled names differ from a
    naive .lower(), which is what both analyzers used independently."""
    assert DistributionName(raw).canonical == canonical


def test_every_spelling_yields_ONE_identity():
    """Two records for the same distribution must not file under two keys."""
    names = {DistributionName(s) for s in
             ("foo-bar", "foo_bar", "foo.bar", "FOO_BAR", "Foo.Bar")}
    assert len(names) == 1


def test_a_distribution_name_is_immutable():
    import dataclasses
    d = DistributionName("pyBigWig")
    try:
        d.canonical = "something-else"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("a distribution name was mutable")


# ---- import identity is a DIFFERENT vocabulary --------------------------
def test_a_module_path_is_NOT_canonicalised_like_a_distribution():
    """zope.interface is a real dotted import path. Canonicalising it to
    zope-interface would produce something not importable at all."""
    assert DistributionName("zope.interface").canonical == "zope-interface"
    assert ImportName("zope.interface").module == "zope.interface"
    assert ImportName("zope.interface").top_level == "zope"


@pytest.mark.parametrize("module,top", [
    ("matplotlib.pyplot", "matplotlib"),
    ("seaborn",           "seaborn"),
    ("zope.interface",    "zope"),
    ("ruamel.yaml",       "ruamel"),
    ("pytest_cov",        "pytest_cov"),
    ("backports_abc",     "backports_abc"),
])
def test_top_level_is_what_a_dependency_provides(module, top):
    """THE MODULE'S CENTRAL CLAIM, and it needs the cases that separate the two
    vocabularies.

    An earlier version used only matplotlib.pyplot and seaborn -- both of which
    give the same answer whether you split on "." or canonicalise and split on
    "-". So a mutation applying DISTRIBUTION canonicalisation to a MODULE name
    went undetected.

    zope.interface distinguishes them: top level "zope", but canonicalised as a
    distribution it becomes "zope-interface" and splitting on "-" gives "zope"
    too. pytest_cov and backports_abc are the discriminating cases --
    canonicalisation turns the underscore into a hyphen, so splitting yields
    "pytest" rather than "pytest_cov".
    """
    assert ImportName(module).top_level == top


def test_the_two_types_do_not_compare_equal():
    """Distinct types, deliberately -- so one cannot be passed where the other
    is expected and silently normalise differently."""
    assert DistributionName("pytest-cov") != ImportName("pytest_cov")


# ---- mappings carry their evidence --------------------------------------
def test_a_mapping_records_WHERE_it_came_from():
    m = DistributionImportMapping(
        distribution=DistributionName("pytest-cov"),
        modules=(ImportName("pytest_cov"),),
        source=MappingSource.PACKAGE_METADATA)
    assert m.provides("pytest_cov")
    assert m.source is MappingSource.PACKAGE_METADATA


def test_a_mapping_is_not_derivable_by_string_surgery():
    """beautifulsoup4 imports as bs4. No hyphen-to-underscore rule produces
    that, which is why the mapping is evidence rather than a transformation."""
    m = DistributionImportMapping(
        distribution=DistributionName("beautifulsoup4"),
        modules=(ImportName("bs4"),),
        source=MappingSource.PACKAGE_METADATA)
    assert m.provides("bs4")
    assert not m.provides("beautifulsoup4")


def test_ASSUMED_IDENTICAL_is_its_own_weakest_source():
    """Recorded as a distinct source because it means nobody checked."""
    assert MappingSource.ASSUMED_IDENTICAL != MappingSource.PACKAGE_METADATA
    m = DistributionImportMapping(
        distribution=DistributionName("seaborn"),
        modules=(ImportName("seaborn"),),
        source=MappingSource.ASSUMED_IDENTICAL)
    assert m.source is MappingSource.ASSUMED_IDENTICAL


def test_an_import_name_REFUSES_a_non_string():
    """A dataclass annotation is documentation, not enforcement.

    ImportName(ImportName("x")) constructed happily and failed later inside
    .top_level with an AttributeError about `split` -- three tests deep and far
    from the cause. That is the same defect class as a parser accepting input
    it cannot represent.
    """
    with pytest.raises(TypeError) as exc:
        ImportName(ImportName("seaborn"))
    assert "different vocabulary" in str(exc.value).lower()


def test_a_mapping_REFUSES_bare_strings_for_modules():
    """Passing a raw string where an ImportName is required is exactly how the
    two vocabularies get mixed."""
    with pytest.raises(TypeError) as exc:
        DistributionImportMapping(
            distribution=DistributionName("seaborn"),
            modules=("seaborn",),
            source=MappingSource.PACKAGE_METADATA)
    assert "separate vocabularies" in str(exc.value)


def test_provides_accepts_either_a_string_or_an_ImportName():
    m = DistributionImportMapping(
        distribution=DistributionName("matplotlib"),
        modules=(ImportName("matplotlib"),),
        source=MappingSource.PACKAGE_METADATA)
    assert m.provides("matplotlib.pyplot")
    assert m.provides(ImportName("matplotlib.pyplot"))


def test_provides_matches_on_TOP_LEVEL():
    m = DistributionImportMapping(
        distribution=DistributionName("matplotlib"),
        modules=(ImportName("matplotlib"),),
        source=MappingSource.PACKAGE_METADATA)
    assert m.provides("matplotlib.pyplot")


# ---- capability x profile, the two-axis model ---------------------------
def test_a_dependency_is_scoped_on_BOTH_axes():
    """Neither "seaborn = development" nor "seaborn = runtime" is true. It is a
    REPORTING dependency, present in TRAINING and DEVELOPER, absent from API --
    established by importtime measurement of api.main, not by its filename."""
    intent = DependencyIntent(
        distribution=DistributionName("seaborn"),
        capabilities=frozenset({Capability.REPORTING}),
        profiles=frozenset({DeploymentProfile.TRAINING,
                            DeploymentProfile.DEVELOPER}),
        specifier=">=0.13,<1",
        rationale="Unguarded import by report_generator; module-level style setup.")
    assert intent.required_by(DeploymentProfile.TRAINING)
    assert not intent.required_by(DeploymentProfile.API)


def test_the_same_capability_can_span_several_profiles():
    intent = DependencyIntent(
        distribution=DistributionName("pyfaidx"),
        capabilities=frozenset({Capability.REFERENCE_SEQUENCE}),
        profiles=frozenset({DeploymentProfile.DATA_PREP,
                            DeploymentProfile.TRAINING,
                            DeploymentProfile.DEVELOPER}),
        specifier=">=0.8,<1",
        rationale="23 script imports, 16 unguarded; cohort and window building.")
    for p in (DeploymentProfile.DATA_PREP, DeploymentProfile.TRAINING):
        assert intent.required_by(p)
    assert not intent.required_by(DeploymentProfile.API)


def test_an_intent_is_immutable():
    import dataclasses
    intent = DependencyIntent(
        distribution=DistributionName("jinja2"),
        capabilities=frozenset({Capability.REPORTING}),
        profiles=frozenset({DeploymentProfile.TRAINING}),
        specifier=">=3.1,<4", rationale="report templating")
    try:
        intent.specifier = ">=999"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("a dependency intent was mutable")


# ---- artifact axes are INDEPENDENT --------------------------------------
def test_authority_and_purpose_are_separate_questions():
    """requirements.lock is GENERATED and a REFERENCE_RESOLUTION. Conflating
    "nothing installs it" with "it has no role" nearly deleted a hash-pinned
    artifact that requirements.txt references three times."""
    assert ArtifactAuthority.GENERATED != ArtifactPurpose.REFERENCE_RESOLUTION
    assert ArtifactPurpose.REFERENCE_RESOLUTION != ArtifactPurpose.INSTALLATION


def test_SOURCE_STALE_is_distinct_from_ARTIFACT_STALE():
    """requirements-dev.lock does not disagree with its source; its SOURCE
    disagrees with reality. Calling it merely "stale" invites regenerating it,
    which would drop five declared packages."""
    assert ArtifactFreshness.SOURCE_STALE != ArtifactFreshness.ARTIFACT_STALE
    assert ArtifactFreshness.SOURCE_STALE != ArtifactFreshness.CURRENT


def test_every_vocabulary_is_a_str_enum_so_it_serialises():
    for member in (Capability.REPORTING, DeploymentProfile.API,
                   ArtifactAuthority.AUTHORED, ArtifactPurpose.INSTALLATION,
                   ArtifactFreshness.SOURCE_STALE, MappingSource.PACKAGE_METADATA):
        assert isinstance(member, str)
        assert member.value == str(member.value)


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except Exception as exc:                        # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed".format(len(tests) - len(failures), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
