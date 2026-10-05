"""The sealed installation plan's admission boundary (owner ruling 2026-10-06).

A package manager executes an APPROVED selection; it must not revise that selection during qualification (measured: renv 1.2.3
retrieve.R L1163-1166 overwrites the requested version with the retrieved one, which is how S4Arrays 1.12.1 replaced the locked
1.12.0). This layer admits a plan selecting exactly one inspected artifact per non-bundled locked package, every runtime-bundled
package from the approved set, bound to the exact lockfile bytes, the admitted runtime record and the platform. It performs NO
downloads, extraction or installation; Artifact instances must come from a separate inspector that checked archive bytes and the
internal DESCRIPTION. A plan digest identifies the plan; it does not authenticate its author.

The code is the owner's reference (ruling generation 027757ee, lines 563-747), transformed mechanically (ValueError -> AdmissionError,
same messages), with ONE structural change: its private strict JSON reader is replaced by r_runtime.strict_json, the package's single
owner of strict parsing (duplicate keys and non-finite constants refused there); non-UTF-8 lockfile bytes -> "lock.encoding". REFINED 2026-10-06 (measured on real CRAN archives): Artifact records the binary's built R series
and whether it has compiled code; admit_plan REQUIRES expected_r_series and admits a Windows binary only if it was built for that series and
its platform is the expected one -- or empty for a package without compiled code.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal, Mapping

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, strict_json

logger = logging.getLogger(__name__)

__all__ = ["HEX256", "Artifact", "Bundled", "InstallPlan", "admit_plan", "plan_digest"]

HEX256 = re.compile(r"[0-9a-f]{64}")

# Renamed from the reference: provenance/artifact.py canonically owns the shorter name for DATA artifact formats (VCF, parquet,
# FASTA), and tests/unit/test_provenance_ownership.py requires one definition per governed name (measured 2026-10-06).
PackageArtifactKind = Literal["source", "windows_binary"]


@dataclass(frozen=True)
class Artifact:
    package: str
    version: str
    kind: PackageArtifactKind
    sha256: str
    size: int
    description_sha256: str

    # Platform identity is required for a selected Windows binary. MEASURED 2026-10-06: CRAN's Built field for a package WITHOUT
    # compiled code has an EMPTY platform ("R 4.7.0; ; ...; windows"), so "" is legitimate there, and only there.
    platform: str | None = None
    # The R major.minor series a binary was BUILT for (from its Built field) and whether it has compiled code. The reference plan never
    # checked the built R version, so a binary built for R 4.7 would have been admitted for R 4.6 (forbidden by the 2026-10-04b ruling).
    built_r_series: str | None = None
    needs_compilation: bool | None = None


@dataclass(frozen=True)
class Bundled:
    package: str
    version: str
    priority: Literal["recommended"]


@dataclass(frozen=True)
class InstallPlan:
    schema_version: int
    lock_sha256: str
    runtime_record_sha256: str
    platform: str
    artifacts: tuple[Artifact, ...]
    bundled: tuple[Bundled, ...]


def require_digest(value: object, reason: str) -> None:
    if type(value) is not str or HEX256.fullmatch(value) is None:
        raise AdmissionError(reason)


def require_text(value: object, reason: str) -> None:
    if type(value) is not str or not value:
        raise AdmissionError(reason)


def admit_plan(
    lock_bytes: bytes,
    plan: InstallPlan,
    *,
    expected_runtime_record_sha256: str,
    expected_platform: str,
    expected_r_series: str,
    approved_bundled: Mapping[str, str],
) -> None:
    if type(plan.schema_version) is not int or plan.schema_version != 1:
        raise AdmissionError("plan.schema_version")

    require_digest(plan.lock_sha256, "plan.lock_digest")
    require_digest(plan.runtime_record_sha256, "plan.runtime_digest")

    if plan.lock_sha256 != sha256(lock_bytes).hexdigest():
        raise AdmissionError("plan.lock_mismatch")

    if plan.runtime_record_sha256 != expected_runtime_record_sha256:
        raise AdmissionError("plan.runtime_mismatch")

    if plan.platform != expected_platform:
        raise AdmissionError("plan.platform_mismatch")
    if type(expected_r_series) is not str or re.fullmatch(r"[0-9]+\.[0-9]+", expected_r_series) is None:
        raise AdmissionError("plan.r_series")

    try:
        lock = strict_json(lock_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        raise AdmissionError("lock.encoding")
    if not isinstance(lock, dict) or not isinstance(
        lock.get("Packages"), dict
    ):
        raise AdmissionError("lock.shape")

    expected = {}
    for name, record in lock["Packages"].items():
        if not isinstance(record, dict):
            raise AdmissionError(f"lock.record_shape:{name}")
        if record.get("Package") != name:
            raise AdmissionError(f"lock.package_name:{name}")
        require_text(record.get("Version"), f"lock.version:{name}")
        expected[name] = record["Version"]

    selected = {}

    def add(package, version):
        require_text(package, "plan.package")
        require_text(version, f"plan.version:{package}")

        if package in selected:
            raise AdmissionError(f"plan.duplicate_package:{package}")
        if package not in expected:
            raise AdmissionError(f"plan.unexpected_package:{package}")
        if version != expected[package]:
            raise AdmissionError(f"plan.version_mismatch:{package}")

        selected[package] = version

    for artifact in plan.artifacts:
        add(artifact.package, artifact.version)

        if artifact.package in approved_bundled:
            raise AdmissionError(
                f"plan.bundled_package_has_artifact:{artifact.package}"
            )

        if artifact.kind not in {"source", "windows_binary"}:
            raise AdmissionError("plan.artifact_kind")

        require_digest(artifact.sha256, "plan.artifact_digest")
        require_digest(
            artifact.description_sha256,
            "plan.description_digest",
        )

        if type(artifact.size) is not int or artifact.size <= 0:
            raise AdmissionError("plan.artifact_size")

        if artifact.kind == "windows_binary":
            platform_ok = artifact.platform == expected_platform or (artifact.platform == "" and artifact.needs_compilation is False)
            if not platform_ok:
                raise AdmissionError("plan.binary_platform")
            if artifact.built_r_series != expected_r_series:
                raise AdmissionError("plan.binary_r_series")
        elif artifact.platform is not None or artifact.built_r_series is not None:
            raise AdmissionError("plan.source_has_binary_platform")

    observed_bundled = {}
    for package in plan.bundled:
        add(package.package, package.version)

        if package.priority != "recommended":
            raise AdmissionError("plan.bundled_priority")

        observed_bundled[package.package] = package.version

    if observed_bundled != dict(approved_bundled):
        raise AdmissionError("plan.bundled_selection_mismatch")

    missing = set(expected) - set(selected)
    if missing:
        raise AdmissionError(
            "plan.missing_packages:" + ",".join(sorted(missing))
        )


def plan_digest(plan: InstallPlan) -> str:
    """Schema-specific deterministic identity; not a signature."""
    record = asdict(plan)
    # asdict() keeps the plan's TUPLES as tuples (measured 2026-10-06: the reference's in-place .sort() raised AttributeError, so
    # plan_digest could not digest any plan); sorted copies work for any sequence.
    record["artifacts"] = sorted(record["artifacts"], key=lambda item: item["package"])
    record["bundled"] = sorted(record["bundled"], key=lambda item: item["package"])

    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    return sha256(b"gvc-install-plan-v1\n" + encoded).hexdigest()
