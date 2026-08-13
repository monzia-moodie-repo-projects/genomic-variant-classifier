"""Canonical concepts for dependency governance. One vocabulary, one owner.

WHY THIS MODULE EXISTS
======================
Two analyzers were written independently and each invented its own identity
rule:

    requirements_parse.py     req.name.lower()
    dependency_census.py      p.lower().replace("-", "_")

That is the embryonic form of parallel vocabulary drift -- the same failure the
rest of this repository keeps finding, one level up from the data. Separate
analyzers are fine; separate AUTHORITIES are not.

DISTRIBUTION IDENTITY IS NOT IMPORT IDENTITY
============================================
Measured 2026-08-13 with packaging 26.0. `canonicalize_name` collapses runs of
`-`, `_` and `.` into a single `-`, which is correct for DISTRIBUTIONS:

    foo_bar         -> foo-bar
    foo.bar         -> foo-bar
    FOO--BAR        -> foo-bar
    zope.interface  -> zope-interface
    ruamel.yaml     -> ruamel-yaml

Six of ten sampled names differ from a naive `.lower()`.

Applying that same rule to a MODULE name would be wrong: `zope.interface` is a
real dotted import path, and `zope-interface` is not importable at all. So the
two are distinct types here, and neither is derivable from the other by string
surgery -- `pyBigWig` the distribution imports as `pyBigWig`, `pytest-cov`
imports as `pytest_cov`, and `beautifulsoup4` imports as `bs4`.

A mapping between them is EVIDENCE, and it must record where it came from.

Author: Monzia Moodie
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from packaging.utils import canonicalize_name


@dataclass(frozen=True)
class DistributionName:
    """A package as PyPI names it. Canonicalised on construction.

    Constructing from any spelling yields one identity, so two records for the
    same distribution cannot be filed under different keys.
    """
    canonical: str

    def __init__(self, raw: str) -> None:
        object.__setattr__(self, "canonical", canonicalize_name(str(raw)))

    def __str__(self) -> str:
        return self.canonical

    def __repr__(self) -> str:
        return "DistributionName({!r})".format(self.canonical)


@dataclass(frozen=True)
class ImportName:
    """A module as Python imports it. NOT canonicalised the same way.

    Dots are meaningful here -- `zope.interface` is a package path -- so the
    only normalisation is the top-level segment, because that is what a
    dependency actually provides.
    """
    module: str

    def __post_init__(self) -> None:
        # A dataclass annotation is documentation, not enforcement. Without
        # this, ImportName(ImportName("x")) constructs happily and fails later
        # inside .top_level with an AttributeError about `split` -- which is
        # exactly what happened, three tests deep and far from the cause.
        if not isinstance(self.module, str):
            raise TypeError(
                "ImportName takes a module path string, got {}: {!r}. A "
                "distribution name is a DIFFERENT vocabulary and must not be "
                "passed here.".format(type(self.module).__name__, self.module))

    @property
    def top_level(self) -> str:
        return self.module.split(".")[0]

    def __str__(self) -> str:
        return self.module

    def __repr__(self) -> str:
        return "ImportName({!r})".format(self.module)


class MappingSource(str, Enum):
    """Where a distribution-to-module mapping came from.

    A mapping asserted without provenance is a guess wearing a type.
    """
    PACKAGE_METADATA = "package_metadata"
    EXPLICIT_OVERRIDE = "explicit_override"
    ASSUMED_IDENTICAL = "assumed_identical"


@dataclass(frozen=True)
class DistributionImportMapping:
    """Which modules a distribution provides, and on what evidence.

    ASSUMED_IDENTICAL is recorded as its own source precisely because it is the
    weakest claim: it means nobody checked, and `beautifulsoup4` importing as
    `bs4` is the standing counterexample.
    """
    distribution: DistributionName
    modules: tuple
    source: MappingSource

    def __post_init__(self) -> None:
        bad = [m for m in self.modules if not isinstance(m, ImportName)]
        if bad:
            raise TypeError(
                "modules must be ImportName instances, got {!r}. Distribution "
                "and import identities are separate vocabularies and mixing "
                "them is what this type exists to prevent.".format(bad[:3]))

    def provides(self, module) -> bool:
        """Whether this distribution supplies `module`, matched on top level.

        `self.modules` already holds ImportName objects, so they are used
        directly. Re-wrapping them produced ImportName(ImportName(...)) and an
        AttributeError far from its cause.
        """
        query = module if isinstance(module, ImportName) else ImportName(module)
        return any(m.top_level == query.top_level for m in self.modules)


class Capability(str, Enum):
    """What a dependency lets this project DO.

    Scope is not a single axis. "runtime versus development" cannot express
    that seaborn is required by the reporting capability and irrelevant to the
    API deployment, which is exactly what the import census measured.
    """
    CORE_MODELING = "core_modeling"
    REPORTING = "reporting"
    REFERENCE_SEQUENCE = "reference_sequence"
    API = "api"
    TESTING = "testing"
    DEV_TOOLING = "dev_tooling"


class DeploymentProfile(str, Enum):
    """A runnable environment that PROMISES a set of capabilities."""
    API = "api"
    TRAINING = "training"
    DATA_PREP = "data_prep"
    CI_TEST = "ci_test"
    DEVELOPER = "developer"


class ArtifactAuthority(str, Enum):
    """Who may edit an artifact. Exactly one authored source per domain."""
    AUTHORED = "authored"
    GENERATED = "generated"


class ArtifactPurpose(str, Enum):
    """What an artifact is FOR -- independent of who authors it.

    requirements.lock is GENERATED and a REFERENCE_RESOLUTION: it is never
    installed from, and that non-installation is part of its contract. Calling
    it "vestigial" because nothing installs it conflated these two axes and
    nearly deleted a hash-pinned artifact.
    """
    RUNTIME_DECLARATION = "runtime_declaration"
    TEST_DECLARATION = "test_declaration"
    INSTALLATION = "installation"
    DEPLOYMENT_LOCK = "deployment_lock"
    REFERENCE_RESOLUTION = "reference_resolution"


class ArtifactFreshness(str, Enum):
    """Whether an artifact still describes reality.

    SOURCE_STALE is the distinction that prevents the wrong remediation:
    requirements-dev.lock does not disagree with its source; its SOURCE
    disagrees with reality. Calling it merely "stale" invites regenerating it,
    which would drop five declared packages.
    """
    CURRENT = "current"
    SOURCE_STALE = "source_stale"
    ARTIFACT_STALE = "artifact_stale"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class DependencyIntent:
    """Why a package exists, expressed on both axes.

    Neither "seaborn = development" nor "seaborn = runtime" is true. It is a
    dependency of the REPORTING capability, present in the TRAINING and
    DEVELOPER profiles and absent from API -- which the import-time measurement
    of api.main established directly.
    """
    distribution: DistributionName
    capabilities: frozenset
    profiles: frozenset
    specifier: str
    rationale: str

    def required_by(self, profile: "DeploymentProfile") -> bool:
        return profile in self.profiles
