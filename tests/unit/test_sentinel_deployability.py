"""A sentinel must exist in every environment where the root must be found.

RUNTIME-SENTINEL-TEST-ARTEFACT-1
================================
PROJECT_SENTINELS once included "tests/EXPECTED_SUITE_SIZE". That is a
TEST-SUITE ARTEFACT being used to identify a DEPLOYMENT root, and every correct
deployment excludes tests -- so the conjunction could not hold in a container by
construction.

MEASURED 2026-08-17 against this repository's own Dockerfile and .dockerignore:

    Dockerfile:185    COPY . .          (trainer stage, WORKDIR /app)
    .dockerignore     tests/            excluded

    pyproject.toml                  in trainer image : True
    src/genomic_variant_classifier  in trainer image : True
    tests/EXPECTED_SUITE_SIZE       in trainer image : FALSE

So `resolve_project_root()` would have RAISED on import of any module reaching
agent_layer.config inside the trainer image -- which is where cloud training
runs. The defect was latent from 69a9597 only because nothing imported
runtime_paths yet; PROJECT-ROOT-HARDCODED-1 is its first consumer, and the
container was a second environment I had not measured.

The API image is unaffected for a different reason: Dockerfile lines 109-126
copy only api/, models/, utils/, monitoring/model_registry.py and the package
__init__.py, so agent_layer/ is not present at all.

WHAT THIS FILE ASSERTS
Not that the current sentinels are correct by inspection, but that EVERY
sentinel survives the container build -- checked against the real
.dockerignore, so adding a sentinel that a deployment excludes fails here
rather than in a container.

Author: Monzia Moodie
"""
from __future__ import annotations

import fnmatch
import io
from pathlib import Path

import pytest

from genomic_variant_classifier.paths.runtime_paths import (
    PROJECT_NAME, PROJECT_SENTINELS, looks_like_project_root,
)

_REPO = Path(__file__).resolve().parents[2]
_DOCKERIGNORE = _REPO / ".dockerignore"


def _ignore_patterns() -> list:
    if not _DOCKERIGNORE.is_file():
        return []
    return [l.strip().rstrip("/")
            for l in io.open(_DOCKERIGNORE, encoding="utf-8").read().splitlines()
            if l.strip() and not l.strip().startswith("#")]


def _survives_docker_copy(rel: str, patterns: list) -> bool:
    """Whether `COPY . .` would place `rel` in the image."""
    top = rel.split("/")[0]
    return not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(top, p)
                   for p in patterns)


@pytest.mark.skipif(not _DOCKERIGNORE.is_file(),
                    reason=".dockerignore is absent; nothing to check against")
@pytest.mark.parametrize("sentinel", PROJECT_SENTINELS)
def test_every_sentinel_survives_the_container_build(sentinel):
    """THE DEFECT, as a property.

    A sentinel excluded by .dockerignore cannot be found in the trainer image,
    so discovery raises there. "tests/EXPECTED_SUITE_SIZE" failed exactly this.
    """
    patterns = _ignore_patterns()
    assert _survives_docker_copy(sentinel, patterns), (
        "{!r} is excluded by .dockerignore, so `COPY . .` will not place it in "
        "the trainer image and resolve_project_root() would RAISE there. A "
        "sentinel must exist in EVERY environment where the root must be "
        "found.".format(sentinel))


@pytest.mark.parametrize("sentinel", PROJECT_SENTINELS)
def test_no_sentinel_is_a_test_or_documentation_artefact(sentinel):
    """The category error, stated directly.

    Test artefacts, documentation and build outputs are absent from correct
    deployments. Requiring one to identify a deployment root is a contradiction
    regardless of what any particular .dockerignore happens to say today.
    """
    first = sentinel.split("/")[0]
    forbidden = {"tests", "test", "docs", "doc", "build", "dist", "notebooks",
                 "htmlcov", ".github", "logs", "outputs"}
    assert first not in forbidden, (
        "{!r} lives under {!r}, which correct deployments exclude".format(
            sentinel, first))


def test_the_sentinels_still_identify_this_repository():
    """Removing a sentinel must not cost identification."""
    assert looks_like_project_root(_REPO), _REPO


@pytest.mark.parametrize("other", [
    "..", "../..",
])
def test_the_sentinels_still_reject_directories_above_this_one(other):
    """And must not cost DISCRIMINATION. Measured 2026-08-17: with two
    sentinels plus the declared name, C:/Users/monzi, C:/Projects, C:/Windows
    and C:/ all return False."""
    candidate = (_REPO / other).resolve()
    if candidate == _REPO:
        pytest.skip("resolved to the repository itself")
    assert not looks_like_project_root(candidate), candidate


def test_a_directory_with_the_sentinels_but_a_DIFFERENT_name_is_rejected(tmp_path):
    """The declared name carries the discrimination the third sentinel used to
    be credited with."""
    for s in PROJECT_SENTINELS:
        p = tmp_path / s
        if "." in Path(s).name:
            p.parent.mkdir(parents=True, exist_ok=True)
            io.open(p, "w", encoding="utf-8", newline="\n").write(
                '[project]\nname = "some-other-project"\nversion = "0.1.0"\n')
        else:
            p.mkdir(parents=True, exist_ok=True)
    assert not looks_like_project_root(tmp_path)


def test_at_least_two_sentinels_remain():
    """A single sentinel would not be a conjunction. pyproject.toml alone is
    common; src/<package> alone is common; together with the declared name they
    are not."""
    assert len(PROJECT_SENTINELS) >= 2, PROJECT_SENTINELS
    assert PROJECT_NAME == "genomic-variant-classifier"


def test_the_removed_sentinel_is_NOT_silently_reintroduced():
    """A regression guard naming the specific artefact, so a future edit that
    re-adds it fails with the reason rather than passing quietly."""
    assert "tests/EXPECTED_SUITE_SIZE" not in PROJECT_SENTINELS, (
        "RUNTIME-SENTINEL-TEST-ARTEFACT-1: this sentinel is excluded by "
        ".dockerignore and cannot be found in the trainer image")
