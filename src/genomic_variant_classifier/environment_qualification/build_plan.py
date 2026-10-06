"""Build planning for the qualified environment (owner rulings 2026-10-08 to 2026-10-10).

The owner's reference code, integrated by measured line ranges from the ruling generations:
  41474c47 L151-259  validate_graph, rebuild_closure, installation_order  (rebuild selection vs installation order -- different questions)
  41474c47 L473-627  BuildExpectation, BuildReceipt, checked_digest, checked_dependencies, admit_build  (a receipt never validates itself)
  a4a5a4f9 L98-220   DependencyExpectation, DependencyObservation, admit_dependencies  (the INSTALLED dependency actually used by a build)
  ce2726fc L107-206  select_routes  (B = reverseClosure_LinkingTo(B_policy U B_no-admitted-binary); one route per package)
Mechanical transformations, each asserted when generated: ValueError -> AdmissionError (a ValueError subclass; same messages); the dependency
block's private AdmissionError / require / HEX256 / checked_digest are dropped or renamed -- AdmissionError comes from r_runtime (its owner),
its single-argument checked_digest becomes _dependency_digest (reason "digest.invalid", unchanged) so it cannot shadow admit_build's two-argument
checked_digest, and its Artifact becomes DependencyArtifact (install_plan.Artifact owns that name); select_routes' local require is dropped in
favour of r_runtime.require. Verified on the REAL data (2026-10-10): from the 159 admitted artifacts, select_routes gives 71 upstream binaries,
12 local builds, 3 runtime packages, 1 bootstrap.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Mapping, Set
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, require

logger = logging.getLogger(__name__)

__all__ = ["validate_graph", "rebuild_closure", "installation_order", "BuildExpectation", "BuildReceipt", "checked_digest",
           "checked_dependencies", "admit_build", "DependencyArtifact", "DependencyExpectation", "DependencyObservation",
           "admit_dependencies", "select_routes"]

_HEX256 = re.compile(r"[0-9a-f]{64}")


def _dependency_digest(value: str) -> str:
    require(type(value) is str and _HEX256.fullmatch(value) is not None, "digest.invalid")
    return value




def validate_graph(
    graph: Mapping[str, Set[str]],
    *,
    planned: Set[str],
    runtime_packages: Set[str],
    label: str,
) -> None:
    if set(graph) != set(planned):
        missing = set(planned) - set(graph)
        extra = set(graph) - set(planned)
        raise AdmissionError(
            f"{label}.coverage:"
            f"missing={sorted(missing)},extra={sorted(extra)}"
        )

    allowed = set(planned) | set(runtime_packages)

    for consumer, providers in graph.items():
        if isinstance(providers, (str, bytes)):
            raise AdmissionError(f"{label}.invalid_edges:{consumer}")

        unknown = set(providers) - allowed
        if unknown:
            raise AdmissionError(
                f"{label}.unknown_dependencies:"
                f"{consumer}:{sorted(unknown)}"
            )

        if consumer in providers:
            raise AdmissionError(f"{label}.self_dependency:{consumer}")


def rebuild_closure(
    linking_to: Mapping[str, Set[str]],
    *,
    seeds: Set[str],
    planned: Set[str],
    runtime_packages: Set[str],
) -> dict[str, tuple[str, ...]]:
    """
    Return each selected package's explanation path.

    Path order:
        original seed -> dependent -> dependent ...

    This is a conservative rebuild policy, not an ABI proof.
    """
    validate_graph(
        linking_to,
        planned=planned,
        runtime_packages=runtime_packages,
        label="linking_to",
    )

    if not set(seeds) <= set(planned):
        raise AdmissionError("rebuild.unknown_seed")

    reverse = {package: set() for package in planned}

    for consumer, providers in linking_to.items():
        for provider in providers:
            if provider in reverse:
                reverse[provider].add(consumer)

    reasons = {package: (package,) for package in sorted(seeds)}
    queue = deque(sorted(seeds))

    while queue:
        provider = queue.popleft()

        for consumer in sorted(reverse[provider]):
            if consumer not in reasons:
                reasons[consumer] = reasons[provider] + (consumer,)
                queue.append(consumer)

    return reasons


def installation_order(
    dependencies: Mapping[str, Set[str]],
    *,
    planned: Set[str],
    runtime_packages: Set[str],
) -> tuple[str, ...]:
    """
    dependencies combines Depends, Imports and LinkingTo.
    Runtime packages are already supplied by the qualified runtime.
    """
    validate_graph(
        dependencies,
        planned=planned,
        runtime_packages=runtime_packages,
        label="installation",
    )

    graph = {
        package: tuple(sorted(set(dependencies[package]) & set(planned)))
        for package in sorted(planned)
    }

    try:
        return tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise AdmissionError("installation.dependency_cycle") from error




@dataclass(frozen=True)
class BuildExpectation:
    package: str
    version: str
    source_sha256: str
    runtime_record_sha256: str
    build_environment_sha256: str
    command_record_sha256: str

    # Observed installed-identity records selected for this build.
    dependency_identities: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BuildReceipt:
    package: str
    version: str
    source_sha256: str
    runtime_record_sha256: str
    build_environment_sha256: str
    command_record_sha256: str
    dependency_identities: tuple[tuple[str, str], ...]

    exit_code: int
    binary_sha256: str
    inspection_record_sha256: str


def checked_digest(value: object, reason: str) -> str:
    if type(value) is not str or _HEX256.fullmatch(value) is None:
        raise AdmissionError(reason)
    return value


def checked_dependencies(
    rows: tuple[tuple[str, str], ...],
    *,
    reason: str,
) -> dict[str, str]:
    if type(rows) is not tuple:
        raise AdmissionError(f"{reason}.shape")

    result = {}
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            raise AdmissionError(f"{reason}.row")

        name, digest = row
        if type(name) is not str or not name:
            raise AdmissionError(f"{reason}.name")
        if name in result:
            raise AdmissionError(f"{reason}.duplicate:{name}")

        result[name] = checked_digest(
            digest, f"{reason}.digest:{name}"
        )

    return result


def admit_build(
    expected: BuildExpectation,
    observed: BuildReceipt,
    *,
    inspected_package: str,
    inspected_version: str,
    inspected_binary_sha256: str,
    admitted_inspection_record_sha256: str,
) -> str:
    """
    Return the admitted binary digest.

    The inspection arguments must come from the independently admitted
    artifact inspection, not be copied from the build receipt.
    """
    for label, record in (
        ("expected", expected),
        ("observed", observed),
    ):
        for field in ("package", "version"):
            value = getattr(record, field)
            if type(value) is not str or not value:
                raise AdmissionError(f"build.{label}.{field}")

        for field in (
            "source_sha256",
            "runtime_record_sha256",
            "build_environment_sha256",
            "command_record_sha256",
        ):
            checked_digest(
                getattr(record, field),
                f"build.{label}.{field}",
            )

    if type(observed.exit_code) is not int or observed.exit_code != 0:
        raise AdmissionError("build.unsuccessful")

    for field in (
        "package",
        "version",
        "source_sha256",
        "runtime_record_sha256",
        "build_environment_sha256",
        "command_record_sha256",
    ):
        if getattr(expected, field) != getattr(observed, field):
            raise AdmissionError(f"build.mismatch:{field}")

    wanted = checked_dependencies(
        expected.dependency_identities, reason="build.expected_dependencies"
    )
    actual = checked_dependencies(
        observed.dependency_identities, reason="build.observed_dependencies"
    )
    if actual != wanted:
        raise AdmissionError("build.dependency_mismatch")

    checked_digest(observed.binary_sha256, "build.binary_digest")
    checked_digest(
        observed.inspection_record_sha256,
        "build.inspection_digest",
    )
    checked_digest(
        inspected_binary_sha256,
        "build.independent_binary_digest",
    )
    checked_digest(
        admitted_inspection_record_sha256,
        "build.independent_inspection_digest",
    )

    if (
        inspected_package != expected.package
        or inspected_version != expected.version
    ):
        raise AdmissionError("build.output_identity_mismatch")

    if inspected_binary_sha256 != observed.binary_sha256:
        raise AdmissionError("build.output_bytes_mismatch")

    if (
        admitted_inspection_record_sha256
        != observed.inspection_record_sha256
    ):
        raise AdmissionError("build.inspection_record_mismatch")

    return observed.binary_sha256


@dataclass(frozen=True)
class DependencyArtifact:
    package: str
    version: str
    sha256: str


@dataclass(frozen=True)
class DependencyExpectation:
    artifact: DependencyArtifact
    # None for an admitted upstream binary.
    producer_receipt_sha256: str | None
    installation_record_sha256: str
    installed_tree_sha256: str
    # Canonicalized by the Windows-aware path inspector.
    installed_location: str


@dataclass(frozen=True)
class DependencyObservation:
    package: str
    version: str
    archive_sha256: str
    producer_receipt_sha256: str | None
    installation_record_sha256: str
    installed_tree_sha256: str
    installed_location: str


def admit_dependencies(
    expected: tuple[DependencyExpectation, ...],
    observed: tuple[DependencyObservation, ...],
) -> None:
    """Compare independently obtained dependency expectations and observations."""
    planned: dict[str, DependencyExpectation] = {}
    measured: dict[str, DependencyObservation] = {}

    for item in expected:
        name = item.artifact.package
        require(type(name) is str and bool(name), "dependency.name")
        require(name not in planned, "dependency.duplicate_expected")
        require(
            type(item.artifact.version) is str and bool(item.artifact.version),
            "dependency.version",
        )
        _dependency_digest(item.artifact.sha256)
        _dependency_digest(item.installation_record_sha256)
        _dependency_digest(item.installed_tree_sha256)
        if item.producer_receipt_sha256 is not None:
            _dependency_digest(item.producer_receipt_sha256)
        require(
            type(item.installed_location) is str
            and bool(item.installed_location),
            "dependency.location",
        )
        planned[name] = item

    for item in observed:
        require(
            type(item.package) is str and bool(item.package),
            "dependency.name",
        )
        require(item.package not in measured, "dependency.duplicate_observed")
        _dependency_digest(item.archive_sha256)
        _dependency_digest(item.installation_record_sha256)
        _dependency_digest(item.installed_tree_sha256)
        if item.producer_receipt_sha256 is not None:
            _dependency_digest(item.producer_receipt_sha256)
        measured[item.package] = item

    require(set(planned) == set(measured), "dependency.set_mismatch")

    for name, want in planned.items():
        got = measured[name]
        checks = (
            (got.version, want.artifact.version, "version"),
            (got.archive_sha256, want.artifact.sha256, "archive"),
            (
                got.producer_receipt_sha256,
                want.producer_receipt_sha256,
                "producer_receipt",
            ),
            (
                got.installation_record_sha256,
                want.installation_record_sha256,
                "installation_record",
            ),
            (
                got.installed_tree_sha256,
                want.installed_tree_sha256,
                "installed_tree",
            ),
            (
                got.installed_location,
                want.installed_location,
                "location",
            ),
        )
        for actual, required, field in checks:
            require(actual == required, f"dependency.{field}_mismatch")


def select_routes(
    *,
    packages: frozenset[str],
    runtime: frozenset[str],
    bootstrap: frozenset[str],
    source_available: frozenset[str],
    binary_available: frozenset[str],
    linking_to: Mapping[str, frozenset[str]],
    policy_roots: frozenset[str],
) -> dict[str, str]:
    """Select routes from admitted exact-version artifacts.

    `available` means admitted for the target runtime/platform,
    not merely mentioned in an index.

    `linking_to` includes every locked package, including packages
    with no LinkingTo dependencies.
    """

    require(
        not runtime & bootstrap,
        "plan.origin_overlap",
    )
    require(
        (runtime | bootstrap) <= packages,
        "plan.unknown_origin",
    )
    require(
        set(linking_to) == packages,
        "graph.node_set_mismatch",
    )
    require(
        all(deps <= packages for deps in linking_to.values()),
        "graph.unknown_provider",
    )

    managed = packages - runtime - bootstrap

    require(
        (source_available | binary_available) <= managed,
        "plan.unknown_artifact",
    )
    require(
        policy_roots <= managed,
        "plan.invalid_root",
    )

    # Packages lacking an admitted binary become source-build roots.
    build = set(policy_roots | (managed - binary_available))

    while True:
        affected = {
            package
            for package, providers in linking_to.items()
            if providers & build
        }

        # Never silently change a fixed runtime/bootstrap origin.
        require(
            not affected & (runtime | bootstrap),
            "plan.fixed_origin_affected",
        )

        expanded = build | affected
        if expanded == build:
            break
        build = expanded

    require(
        build <= source_available,
        "plan.required_source_unavailable",
    )

    routes = {
        package: (
            "runtime" if package in runtime
            else "bootstrap" if package in bootstrap
            else "local_build" if package in build
            else "upstream_binary"
        )
        for package in sorted(packages)
    }

    require(
        {
            package
            for package, route in routes.items()
            if route == "upstream_binary"
        } <= binary_available,
        "plan.required_binary_unavailable",
    )

    return routes
