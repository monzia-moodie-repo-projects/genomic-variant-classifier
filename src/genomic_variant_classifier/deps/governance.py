"""The measured scope of each dependency, and the evidence for it.

DEPENDENCY-ONTOLOGY-1
=====================
Scope was encoded by FILENAME. `requirements-dev.txt` says "development" in its
own comment and its only measured consumer is continuous integration, so its
contents record history rather than intent.

An import census run on 2026-08-13 against `src`, `scripts` and `tests` -- 941
files walked, 941 parsed, zero failures -- measured what actually imports what.
NOT ONE of six packages was correctly scoped by the file it sits in.

    seaborn, jinja2   src/.../reports/report_generator.py, UNGUARDED, with
                      sns.set_style() executing at module scope
    pyfaidx           25 imports: 23 in scripts/, 2 in tests/, 7 try-guarded
    anyio, httpx      zero imports; reached only through fastapi.testclient
    pre_commit        zero imports, and correctly so -- a console script

WHY TWO AXES AND NOT ONE
Neither "seaborn = development" nor "seaborn = runtime" is true. It is a
dependency of the REPORTING capability, required by the training and developer
profiles and absent from the API -- which `python -X importtime -c "import
api.main"` established directly, showing no seaborn, jinja2, reports or
report_generator in the import graph.

A one-dimensional label cannot express that, and trying to forced the earlier
mistake in both directions: leaving them in a development file understates a
production capability, while adding them to the API lock would bloat an image
that provably does not use them.

WHAT THIS MODULE IS NOT
It is not an installer, a resolver, or a policy engine. It is the RECORD of
what was measured, with each claim attached to the observation that supports
it, so a later reconciliation has something to reconcile AGAINST.

Author: Monzia Moodie
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from genomic_variant_classifier.deps.model import (
    ArtifactAuthority, ArtifactFreshness, ArtifactPurpose, Capability, DependencyIntent, DeploymentProfile, DistributionName,
)


class EvidenceStrength(str, Enum):
    """How well a classification is supported.

    UNRESOLVED is a first-class state. `httpx` and `anyio` are reached only
    through `fastapi.testclient`, and whether their versions belong in the
    tested compatibility surface -- or should float transitively -- has NOT
    been measured. Recording that as "transitive and misplaced" would assert
    more than the evidence carries.
    """
    DIRECT_IMPORT = "direct_import"
    INDIRECT_CAPABILITY = "indirect_capability"
    NON_IMPORT_CHANNEL = "non_import_channel"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ScopeEvidence:
    """One observation supporting a classification.

    `measured_on` and `method` are fields because a classification without a
    date and an instrument is an opinion. A stale limitation phrased in the
    present tense became a false premise once already in this repository.
    """
    strength: EvidenceStrength
    measured_on: str
    method: str
    detail: str


@dataclass(frozen=True)
class ClassifiedDependency:
    """An intent plus the evidence that produced it."""
    intent: DependencyIntent
    evidence: tuple
    resolved: bool

    @property
    def name(self) -> str:
        return str(self.intent.distribution)


def _intent(distribution, capabilities, profiles, specifier, rationale):
    return DependencyIntent(
        distribution=DistributionName(distribution),
        capabilities=frozenset(capabilities),
        profiles=frozenset(profiles),
        specifier=specifier,
        rationale=rationale)


#: The import census, verbatim. Kept as a constant so a re-run can be compared
#: against it rather than replacing it silently.
CENSUS_2026_08_13 = {
    "files_walked": 941,
    "files_parsed": 941,
    "parse_failures": 0,
    "roots": ("src", "scripts", "tests"),
}


CLASSIFICATIONS = (
    ClassifiedDependency(
        intent=_intent(
            "seaborn",
            {Capability.REPORTING},
            {DeploymentProfile.TRAINING, DeploymentProfile.DEVELOPER},
            ">=0.13,<1",
            "Unguarded import by report_generator, with sns.set_style() at "
            "module scope -- importing the module requires the package."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "import census",
                "src/genomic_variant_classifier/reports/report_generator.py:45, "
                "unguarded, in the same block as matplotlib, numpy, pandas, "
                "scipy and sklearn -- all production dependencies."),
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "importtime",
                "python -X importtime -c 'import api.main' shows no seaborn, "
                "reports or report_generator: the API does NOT reach the "
                "reporting capability, so this is absent from that profile."),
        ),
        resolved=True),

    ClassifiedDependency(
        intent=_intent(
            "jinja2",
            {Capability.REPORTING},
            {DeploymentProfile.TRAINING, DeploymentProfile.DEVELOPER},
            ">=3.1,<4",
            "Unguarded import by report_generator for report templating."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "import census",
                "src/genomic_variant_classifier/reports/report_generator.py:46, "
                "`from jinja2 import Environment`, unguarded."),
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "lockfile scan",
                "absent from requirements-api.lock, while matplotlib -- two "
                "lines above it in the same import block -- is present at "
                "line 24. Someone traced that dependency and missed these."),
        ),
        resolved=True),

    ClassifiedDependency(
        intent=_intent(
            "pyfaidx",
            {Capability.REFERENCE_SEQUENCE},
            {DeploymentProfile.DATA_PREP, DeploymentProfile.TRAINING,
             DeploymentProfile.DEVELOPER},
            ">=0.8,<1",
            "Reference-window extraction and cohort building. Named "
            "REFERENCE_SEQUENCE rather than 'operational', which would become "
            "a bag holding unrelated dependencies."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "import census",
                "25 imports: 23 in scripts/, 2 in tests/, of which 7 are "
                "try-guarded and 18 are hard. Declared development-only."),
            ScopeEvidence(
                EvidenceStrength.DIRECT_IMPORT, "2026-08-13", "pip show",
                "installed at 0.9.0.4, satisfying >=0.8,<1. Added 2026-05-31 "
                "by the ref/alt delta window-extraction commit."),
        ),
        resolved=True),

    ClassifiedDependency(
        intent=_intent(
            "pre-commit",
            {Capability.DEV_TOOLING},
            {DeploymentProfile.DEVELOPER},
            ">=4,<5",
            "A console script, never imported. Zero imports is the CORRECT "
            "census result for this package, not evidence of disuse."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.NON_IMPORT_CHANNEL, "2026-08-13",
                "import census",
                "zero imports across 941 files. AST sees one of at least four "
                "consumer channels; a command-line tool is invoked, not "
                "imported, which is why 'NO IMPORT ANYWHERE' must never be "
                "read as 'UNUSED'."),
            ScopeEvidence(
                EvidenceStrength.NON_IMPORT_CHANNEL, "2026-08-13",
                "declaration scan",
                "declared in requirements-dev.in and ABSENT from "
                "requirements-dev.txt -- the reverse of the other five. It "
                "should not be forced into the test runtime merely to make "
                "two files equal."),
        ),
        resolved=True),

    ClassifiedDependency(
        intent=_intent(
            "httpx",
            {Capability.TESTING},
            {DeploymentProfile.CI_TEST, DeploymentProfile.DEVELOPER},
            ">=0.27,<1",
            "UNRESOLVED. Reached only through fastapi.testclient.TestClient. "
            "Whether its version is part of the tested compatibility surface "
            "or an implementation detail that should float transitively has "
            "NOT been measured."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.UNRESOLVED, "2026-08-13", "import census",
                "zero direct imports across 941 files."),
            ScopeEvidence(
                EvidenceStrength.INDIRECT_CAPABILITY, "2026-08-13",
                "text scan",
                "the only consumers are fastapi.testclient.TestClient uses in "
                "tests/unit/test_api.py and tests/unit/test_runtime_"
                "attribution.py. TestClient requires an HTTP client "
                "implementation."),
        ),
        resolved=False),

    ClassifiedDependency(
        intent=_intent(
            "anyio",
            {Capability.TESTING},
            {DeploymentProfile.CI_TEST, DeploymentProfile.DEVELOPER},
            ">=4.0",
            "UNRESOLVED, on the same evidence as httpx. Note that the ONE "
            "measured mention outside a test was a COMMENT in "
            "scripts/check_lock_satisfies.py:32 -- which a text search "
            "counted and the AST census correctly did not."),
        evidence=(
            ScopeEvidence(
                EvidenceStrength.UNRESOLVED, "2026-08-13", "import census",
                "zero direct imports across 941 files."),
            ScopeEvidence(
                EvidenceStrength.INDIRECT_CAPABILITY, "2026-08-13",
                "lockfile scan",
                "present in requirements-dev.lock at 4.13.0 as a transitive "
                "dependency, and declared directly in requirements-dev.txt."),
        ),
        resolved=False),
)


#: Distributions whose classification is settled by measurement.
def resolved_classifications() -> tuple:
    return tuple(c for c in CLASSIFICATIONS if c.resolved)


#: Distributions still requiring measurement before any declaration changes.
def unresolved_classifications() -> tuple:
    return tuple(c for c in CLASSIFICATIONS if not c.resolved)


def classification_for(distribution):
    """Look up by canonical distribution name, or None."""
    key = DistributionName(distribution)
    for c in CLASSIFICATIONS:
        if c.intent.distribution == key:
            return c
    return None


def required_by(profile) -> tuple:
    """Which classified distributions a deployment profile needs."""
    return tuple(c for c in CLASSIFICATIONS if c.intent.required_by(profile))


# ============================================================================================================================
# DEPENDENCY-PROFILES-1 (owner rulings 2026-10-31): every dependency FILE and every INSTALL PATH, declared.
#
# A file's name says nothing reliable about what it is for -- requirements-dev.txt said "development" and was the CI's test
# declaration -- so each file is declared on the model's own axes, and a test refuses an undeclared file at the repository root.
# Each install path carries an explicit migration status: a path that is not lock-qualified must SAY so, so its results cannot be
# described as reproducible from a lock. Measured 2026-10-31 at 9feaf1a1 (the install-line census of every workflow, the Dockerfile
# and scripts/vm_bootstrap_run.sh).
# ============================================================================================================================

@dataclass(frozen=True)
class DependencyArtifact:
    path: str
    authority: ArtifactAuthority
    purpose: ArtifactPurpose
    profiles: frozenset
    freshness: ArtifactFreshness
    generated_from: tuple
    note: str


def _artifact(path, authority, purpose, profiles, freshness, generated_from, note):
    return DependencyArtifact(path, authority, purpose, frozenset(profiles), freshness, tuple(generated_from), note)


_TT = (DeploymentProfile.TRAINING, DeploymentProfile.CI_TEST)
_TT_SOURCES = ("requirements.txt", "requirements-dev.txt", "constraints-observed-venv312.txt")

ARTIFACTS = (
    _artifact("requirements.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.RUNTIME_DECLARATION, _TT, ArtifactFreshness.CURRENT, (),
              "57 exact pins, 7 ranges, 1 unbounded (measured 2026-10-31); NOT a lock: it pins no transitive dependency."),
    _artifact("requirements-dev.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.TEST_DECLARATION, (DeploymentProfile.CI_TEST, DeploymentProfile.DEVELOPER),
              ArtifactFreshness.CURRENT, (), "7 ranges; its only measured consumer is continuous integration."),
    _artifact("constraints-observed-venv312.txt", ArtifactAuthority.GENERATED, ArtifactPurpose.MEASURED_CONSTRAINTS, _TT, ArtifactFreshness.CURRENT, (),
              "249 versions measured in the owner's .venv312 (observation a4fc5829..., 2026-10-31); the project itself excluded (duplicate metadata)."),
    _artifact("requirements-train-test-linux.lock", ArtifactAuthority.GENERATED, ArtifactPurpose.INSTALLATION, _TT, ArtifactFreshness.CURRENT, _TT_SOURCES,
              "Linux, Python 3.11 and 3.12: two independent resolutions, byte-identical bodies (2026-11-01)."),
    _artifact("requirements-train-test-windows.lock", ArtifactAuthority.GENERATED, ArtifactPurpose.INSTALLATION, _TT, ArtifactFreshness.CURRENT, _TT_SOURCES,
              "Windows, Python 3.12, generated on the owner's machine by block L1 (2026-11-01, 6b071da5...)."),
    _artifact("requirements-generator-toolchain.in", ArtifactAuthority.AUTHORED, ArtifactPurpose.GENERATOR_TOOLCHAIN, (), ArtifactFreshness.CURRENT, (),
              "The owner's measured tool versions; colorama because click and build need it on Windows."),
    _artifact("requirements-generator-toolchain.txt", ArtifactAuthority.GENERATED, ArtifactPurpose.GENERATOR_TOOLCHAIN, (), ArtifactFreshness.CURRENT,
              ("requirements-generator-toolchain.in",), "Installed with --require-hashes --no-deps into a separate generator environment."),
    _artifact("build-constraints-train-test.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.BUILD_CONSTRAINTS, _TT, ArtifactFreshness.CURRENT, (),
              "deflate-dict 1.2.2, dict-hash 1.3.7 and pyspark 3.5.8 exist only as source archives; measured 2026-11-01, an isolated build "
              "used setuptools 84.2.1 (unpinned); with this file the build backend is the lock's setuptools 70.2.0 and wheel 0.47.0."),
    _artifact("requirements.in", ArtifactAuthority.AUTHORED, ArtifactPurpose.RUNTIME_DECLARATION, (), ArtifactFreshness.SOURCE_STALE, (),
              "Source of the reference resolution requirements.lock; diverged from requirements.txt (DEVDEPS-DIVERGED-1)."),
    _artifact("requirements-dev.in", ArtifactAuthority.AUTHORED, ArtifactPurpose.TEST_DECLARATION, (), ArtifactFreshness.SOURCE_STALE, (),
              "Source of requirements-dev.lock; last changed 2026-05-07 (b1f14b2)."),
    _artifact("requirements.lock", ArtifactAuthority.GENERATED, ArtifactPurpose.REFERENCE_RESOLUTION, (), ArtifactFreshness.SOURCE_STALE, ("requirements.in",),
              "Never installed from; generated 2026-05-07."),
    _artifact("requirements-dev.lock", ArtifactAuthority.GENERATED, ArtifactPurpose.REFERENCE_RESOLUTION, (), ArtifactFreshness.SOURCE_STALE,
              ("requirements-dev.in",), "Never installed from; against today's declarations: 10 missing, 21 conflicting on Linux (2026-10-31)."),
    _artifact("requirements-api.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.RUNTIME_DECLARATION, (DeploymentProfile.API,), ArtifactFreshness.CURRENT, (),
              "The API image's declaration."),
    _artifact("requirements-api.lock", ArtifactAuthority.GENERATED, ArtifactPurpose.DEPLOYMENT_LOCK, (DeploymentProfile.API,), ArtifactFreshness.CURRENT,
              ("requirements-api.txt",), "57 exact pins, NO hashes; checked by scripts/check_lock_satisfies.py."),
    _artifact("requirements-drift.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.INSTALLATION, (DeploymentProfile.DRIFT_MONITOR,), ArtifactFreshness.UNVERIFIED, (),
              "The isolated drift environment; exact pins without hashes; not yet migrated to a hashed lock."),
    _artifact("requirements-source-monitor.txt", ArtifactAuthority.AUTHORED, ArtifactPurpose.INSTALLATION, (DeploymentProfile.SOURCE_MONITOR,),
              ArtifactFreshness.CURRENT, (), "PyYAML only, hash-locked; installed --require-hashes --no-deps --only-binary=:all:."),
)


@dataclass(frozen=True)
class InstallConsumer:
    """A file that installs Python packages, and whether its result is reproducible from a hashed lock."""
    path: str
    lock_qualified: bool
    status: str


INSTALL_CONSUMERS = (
    InstallConsumer(".github/workflows/ci.yml", False,
                    "MIXED: job 'test' installs requirements-train-test-linux.lock exactly (DEPENDENCY-PROFILES-1); job 'drift' installs "
                    "requirements-drift.txt and unpinned pytest/packaging; the lockfile job resolves requirements-api.lock with --dry-run only."),
    InstallConsumer(".github/workflows/teardown_abort_diagnostic.yml", False, "NOT MIGRATED: requirements-api.lock + requirements.txt + requirements-dev.txt."),
    InstallConsumer(".github/workflows/drift_monitor.yml", False, "NOT MIGRATED: requirements-drift.txt, unpinned pandas/pyarrow, requirements-api.txt."),
    InstallConsumer(".github/workflows/data_freshness.yml", False, "NOT MIGRATED: four exact pins and an editable install with dependencies."),
    InstallConsumer(".github/workflows/source_monitor.yml", True, "Hashed, --no-deps, binary-only: requirements-source-monitor.txt."),
    InstallConsumer(".github/workflows/source_monitor_verify.yml", True, "Hashed, --no-deps, binary-only: requirements-source-monitor.txt."),
    InstallConsumer("Dockerfile", False, "NOT MIGRATED: requirements-api.lock (fallback requirements-api.txt); trainer stage requirements.txt."),
    InstallConsumer("scripts/vm_bootstrap_run.sh", False, "NOT MIGRATED: --no-deps pins read from the reference lock, unpinned installs, re-pins."),
    # Historical run launchers (2026-04 to 2026-07), measured 2026-11-01: none installs from a lock. Their runs' results are not
    # reproducible from a lock, and a re-run through them would not be either.
    InstallConsumer("scripts/gcp_run6_startup.sh", False, "NOT MIGRATED (Run 6, 2026-04-20): requirements.txt + torch/torchvision from the cu118 index."),
    InstallConsumer("scripts/gcp_run7_startup.sh", False, "NOT MIGRATED (Run 7, 2026-04-08): unpinned torch-geometric, torch-scatter, torch-sparse."),
    InstallConsumer("scripts/gcp_run8_startup.sh", False, "NOT MIGRATED (Run 8, 2026-04-08): unpinned transformers."),
    InstallConsumer("scripts/launch_run9_vm.sh", False, "NOT MIGRATED (Run 9, 2026-05-11): editable install with dependencies."),
    InstallConsumer("scripts/launch_run10_vm.sh", False, "NOT MIGRATED (Run 10, 2026-05-13): editable install with dependencies."),
    InstallConsumer("scripts/launch_run10a_vm.sh", False, "NOT MIGRATED (Run 10a, 2026-05-23): unpinned pykan + editable install."),
    InstallConsumer("scripts/launch_run16_vm.sh", False, "NOT MIGRATED (Run 16, 2026-07-13): requirements.txt + editable --no-deps."),
    InstallConsumer("scripts/launch_run17_baseline.sh", False, "NOT MIGRATED (Run 17, 2026-07-06): 'pybigtools>=0.3.0', failure ignored (|| true)."),
    InstallConsumer("scripts/launch_run17_r12only.sh", False, "NOT MIGRATED (Run 17, 2026-07-06): 'pybigtools>=0.3.0', failure ignored (|| true)."),
    InstallConsumer("scripts/launch_run17_r13only.sh", False, "NOT MIGRATED (Run 17, 2026-07-06): 'pybigtools>=0.3.0', failure ignored (|| true)."),
)


def artifact_for(path: str):
    """The declaration of a repository-root dependency file, or None."""
    return next((a for a in ARTIFACTS if a.path == path), None)
