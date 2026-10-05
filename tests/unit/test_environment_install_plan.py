"""The sealed installation plan and the source-only repair admission (owner ruling 2026-10-06).

The first part is the owner's reference tests (ruling generation 027757ee, lines 761-845), unchanged apart from the import below;
the rest adds a refusal for every remaining branch, the plan digest's determinism, and source-repair cases on the REAL lockfile. The fixture's
kwargs gained ONE entry, "expected_r_series": "4.6", because admit_plan now requires it (the built-R-version refinement, 2026-10-06).

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.environment_qualification.install_plan import Artifact, Bundled, InstallPlan, admit_plan, plan_digest
from dataclasses import replace
from hashlib import sha256
import json
import pytest


def fixture():
    raw = json.dumps({
        "Packages": {
            "S4Arrays": {
                "Package": "S4Arrays",
                "Version": "1.12.0",
            },
            "Matrix": {
                "Package": "Matrix",
                "Version": "1.7-5",
            },
        }
    }).encode()

    artifact = Artifact(
        package="S4Arrays",
        version="1.12.0",
        kind="source",
        sha256="a" * 64,
        size=100,
        description_sha256="b" * 64,
    )
    plan = InstallPlan(
        schema_version=1,
        lock_sha256=sha256(raw).hexdigest(),
        runtime_record_sha256="c" * 64,
        platform="x86_64-w64-mingw32",
        artifacts=(artifact,),
        bundled=(Bundled("Matrix", "1.7-5", "recommended"),),
    )
    kwargs = {
        "expected_runtime_record_sha256": "c" * 64,
        "expected_platform": "x86_64-w64-mingw32",
        "expected_r_series": "4.6",
        "approved_bundled": {"Matrix": "1.7-5"},
    }
    return raw, plan, kwargs


def test_valid_selection():
    raw, plan, kwargs = fixture()
    admit_plan(raw, plan, **kwargs)


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda p: replace(p, schema_version=True),
            "plan.schema_version",
        ),
        (
            lambda p: replace(
                p,
                artifacts=(replace(p.artifacts[0], version="1.12.1"),),
            ),
            "plan.version_mismatch:S4Arrays",
        ),
        (
            lambda p: replace(p, artifacts=p.artifacts * 2),
            "plan.duplicate_package:S4Arrays",
        ),
        (
            lambda p: replace(p, artifacts=()),
            "plan.missing_packages:S4Arrays",
        ),
        (
            lambda p: replace(
                p,
                artifacts=(replace(p.artifacts[0], size=True),),
            ),
            "plan.artifact_size",
        ),
    ],
)
def test_refusals(mutation, reason):
    raw, plan, kwargs = fixture()
    with pytest.raises(ValueError) as error:
        admit_plan(raw, mutation(plan), **kwargs)
    assert str(error.value) == reason


# ------------------------------------------------------------------ every remaining branch of admit_plan

def _refused(raw, plan, reason, **override):
    _, _, kwargs = fixture()
    kwargs.update(override)
    with pytest.raises(ValueError) as error:
        admit_plan(raw, plan, **kwargs)
    assert str(error.value) == reason


def test_more_refusals():
    raw, plan, kwargs = fixture()
    _refused(raw + b" ", plan, "plan.lock_mismatch")
    _refused(raw, plan, "plan.runtime_mismatch", expected_runtime_record_sha256="d" * 64)
    _refused(raw, plan, "plan.platform_mismatch", expected_platform="aarch64-apple-darwin")
    _refused(raw, replace(plan, artifacts=plan.artifacts + (replace(plan.artifacts[0], package="Extra"),)), "plan.unexpected_package:Extra")
    _refused(raw, replace(plan, artifacts=plan.artifacts + (replace(plan.artifacts[0], package="Matrix", version="1.7-5"),),
                          bundled=()), "plan.bundled_package_has_artifact:Matrix")
    _refused(raw, replace(plan, artifacts=(replace(plan.artifacts[0], kind="windows_binary", platform="x86_64-pc-linux-gnu"),)), "plan.binary_platform")
    _refused(raw, replace(plan, artifacts=(replace(plan.artifacts[0], platform="x86_64-w64-mingw32"),)), "plan.source_has_binary_platform")
    _refused(raw, replace(plan, artifacts=(replace(plan.artifacts[0], sha256="A" * 64),)), "plan.artifact_digest")
    _refused(raw, replace(plan, bundled=(Bundled("Matrix", "1.7-5", "base"),)), "plan.bundled_priority")
    _refused(raw, plan, "plan.bundled_selection_mismatch", approved_bundled={"Matrix": "1.7-4"})


def test_a_windows_binary_on_the_right_platform_is_admitted():
    raw, plan, kwargs = fixture()
    admit_plan(raw, replace(plan, artifacts=(replace(plan.artifacts[0], kind="windows_binary", platform="x86_64-w64-mingw32",
                                                     built_r_series="4.6", needs_compilation=True),)), **kwargs)


def _binary(plan, **fields):
    return replace(plan, artifacts=(replace(plan.artifacts[0], kind="windows_binary", **fields),))


def test_a_pure_r_binary_with_an_empty_platform_is_admitted():
    raw, plan, kwargs = fixture()        # measured: CRAN writes "Built: R 4.x; ; ...; windows" for packages without compiled code
    admit_plan(raw, _binary(plan, platform="", built_r_series="4.6", needs_compilation=False), **kwargs)


@pytest.mark.parametrize("fields, reason", [
    (dict(platform="", built_r_series="4.6", needs_compilation=True), "plan.binary_platform"),      # compiled code needs a platform
    (dict(platform="", built_r_series="4.7", needs_compilation=False), "plan.binary_r_series"),     # the DANDELION 4.7.0 binary's case
    (dict(platform="x86_64-w64-mingw32", built_r_series=None, needs_compilation=True), "plan.binary_r_series"),
])
def test_binary_identity_refusals(fields, reason):
    raw, plan, kwargs = fixture()
    with pytest.raises(ValueError) as error:
        admit_plan(raw, _binary(plan, **fields), **kwargs)
    assert str(error.value) == reason


def test_a_source_artifact_may_not_carry_a_built_r_series():
    raw, plan, kwargs = fixture()
    with pytest.raises(ValueError) as error:
        admit_plan(raw, replace(plan, artifacts=(replace(plan.artifacts[0], built_r_series="4.6"),)), **kwargs)
    assert str(error.value) == "plan.source_has_binary_platform"


@pytest.mark.parametrize("series", ["4", "4.6.1", "", 4.6])
def test_a_malformed_expected_r_series_is_refused(series):
    raw, plan, kwargs = fixture()
    with pytest.raises(ValueError) as error:
        admit_plan(raw, plan, **dict(kwargs, expected_r_series=series))
    assert str(error.value) == "plan.r_series"


def test_non_utf8_lockfile_bytes_are_refused():
    raw, plan, kwargs = fixture()
    bad = raw + b"\xff"
    with pytest.raises(ValueError) as error:
        admit_plan(bad, replace(plan, lock_sha256=sha256(bad).hexdigest()), **kwargs)
    assert str(error.value) == "lock.encoding"


def test_plan_digest_is_deterministic_and_order_independent():
    _, plan, _ = fixture()
    second = Artifact("Other", "1.0", "source", "e" * 64, 5, "f" * 64)
    a = replace(plan, artifacts=(plan.artifacts[0], second))
    b = replace(plan, artifacts=(second, plan.artifacts[0]))
    assert plan_digest(a) == plan_digest(b) and len(plan_digest(a)) == 64
    assert plan_digest(a) != plan_digest(replace(a, platform="other"))


# ------------------------------------------------------------------ source-only repair on the REAL lockfile

import copy  # noqa: E402
from pathlib import Path  # noqa: E402

from genomic_variant_classifier.environment_qualification.r_runtime import strict_json  # noqa: E402
from genomic_variant_classifier.environment_qualification.source_repair import admit_source_repair  # noqa: E402

LOCK = strict_json((Path(__file__).resolve().parents[2] / "renv.lock").read_text(encoding="utf-8"))
TARGETS = {n for n, r in LOCK["Packages"].items() if "r-universe" in str(r.get("Repository", ""))}


def _repair(mutate, targets=TARGETS):
    new = copy.deepcopy(LOCK)
    mutate(new)
    return admit_source_repair(LOCK, new, targets)


def test_the_real_lockfile_has_23_r_universe_targets():
    assert len(TARGETS) == 23 and "S4Arrays" in TARGETS


def test_a_provenance_field_change_on_a_target_is_admitted():
    assert _repair(lambda d: d["Packages"]["S4Arrays"].__setitem__("Repository", "BioCsoft")) == {"S4Arrays": ["Repository"]}


@pytest.mark.parametrize("mutate, targets, reason", [
    (lambda d: d["Packages"]["S4Arrays"].__setitem__("Version", "1.12.1"), TARGETS, "lock.forbidden_fields:S4Arrays:Version"),
    (lambda d: d["R"].__setitem__("Version", "4.6.1"), TARGETS, "lock.metadata_changed:R"),
    (lambda d: d["Packages"].pop("S4Arrays"), TARGETS, "lock.package_set_changed"),
    (lambda d: d["Packages"]["Matrix"].__setitem__("Repository", "elsewhere"), TARGETS, "lock.unapproved_package:Matrix"),
    (lambda d: d["Packages"]["S4Arrays"].__setitem__("Title", "x"), TARGETS, "lock.forbidden_fields:S4Arrays:Title"),
    (lambda d: None, TARGETS | {"NotAPackage"}, "lock.unknown_repair_target"),
    (lambda d: d.__setitem__("Extra", {}), TARGETS, "lock.top_level_changed"),
])
def test_source_repair_refusals(mutate, targets, reason):
    with pytest.raises(ValueError) as error:
        _repair(mutate, targets)
    assert str(error.value) == reason
