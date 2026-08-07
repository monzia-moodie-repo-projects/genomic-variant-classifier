"""tests/unit/test_docker_image_covers_the_api.py

Author: Monzia Moodie
Written 2026-08-07. DOCKERCOPY-1.

Every module the web service imports must be inside the container image.

WHY THIS EXISTS. `4d334f9` added `api/attribution.py`, which imports
`genomic_variant_classifier.monitoring.model_registry` at module level. The
Dockerfile's `api` stage copies `api/`, `models/`, `utils/` and the package
`__init__.py` -- and NOT `monitoring/`. Inside the image that import raised
ModuleNotFoundError, gunicorn never bound, the container exited, and the
smoke test failed with `curl` returning nothing.

NOTHING IN THE REPOSITORY COULD HAVE CAUGHT IT. The import-resolution gate
runs against the full source tree, where the import resolves perfectly. The
test suite imports from the same tree. Only the IMAGE has the narrower file
surface, and the only thing that exercised the image was a smoke test that
greps for the literal `"status"` -- which would have passed had the service
started and lied.

The author enumerated the consumers of `api.main` in PYTHON and shipped. The
container's file surface is a consumer expressed in a Dockerfile, and no
Python search finds it.

WHAT THIS CHECKS, AND WHAT IT DOES NOT. It walks the static import graph from
`api/main.py` and asserts every reachable first-party module lives under a path
the `api` stage copies. It says nothing about third-party dependencies -- those
are `requirements-api.txt`'s business and `pip check`'s gate -- and nothing
about imports performed dynamically by string. Static first-party imports are
the class that broke, and the class this closes.
"""
from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path

import pytest

PACKAGE = "genomic_variant_classifier"
ENTRY_POINT = f"{PACKAGE}.api.main"

#: The build stage the web service ships from. `trainer` copies the whole tree
#: (`COPY . .`) and is deliberately out of scope.
API_STAGE = "api"


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "Dockerfile").is_file() and (
                candidate / "src" / PACKAGE).is_dir():
            return candidate
    raise AssertionError(
        "could not locate the repository root: expected an ancestor of "
        f"{Path(__file__).resolve()} containing both Dockerfile and src/")


def _module_file(dotted: str, src: Path) -> Path | None:
    """The file backing a dotted module name, or None if it is not ours.

    Handles regular packages and IMPLICIT NAMESPACE PACKAGES alike:
    `monitoring/` has no `__init__.py`, which is exactly why copying
    `model_registry.py` alone is sufficient and why a check that insisted on
    `__init__.py` would mis-describe the requirement.
    """
    base = src / Path(*dotted.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    module = base.with_suffix(".py")
    return module if module.is_file() else None


def import_closure(entry: str, src: Path) -> dict[str, Path]:
    """Every first-party module statically reachable from `entry`.

    Relative imports are resolved to absolute names first, and each
    `from X import Y` also queues `X.Y` because Y may itself be a submodule
    rather than an attribute -- the distinction that produced eleven phantom
    failures in an earlier check, and which is settled here by simply asking
    whether a file exists at that path.
    """
    found: dict[str, Path] = {}
    seen: set[str] = set()
    queue = deque([entry])
    while queue:
        dotted = queue.popleft()
        if dotted in seen:
            continue
        seen.add(dotted)
        path = _module_file(dotted, src)
        if path is None:
            continue
        found[dotted] = path
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        package = (dotted if path.name == "__init__.py"
                   else dotted.rpartition(".")[0])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    bits = package.split(".")
                    if node.level > 1:
                        bits = bits[: -(node.level - 1)]
                    target = ".".join(bits + ([node.module] if node.module
                                              else []))
                else:
                    target = node.module or ""
                if not target.startswith(PACKAGE):
                    continue
                queue.append(target)
                for alias in node.names:
                    if alias.name != "*":
                        queue.append(f"{target}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(PACKAGE):
                        queue.append(alias.name)
    return found


def copied_paths(dockerfile: str, stage: str) -> list[str]:
    """Source paths a named build stage copies from the build context.

    `COPY --from=<stage>` is EXCLUDED: it copies from another image layer, not
    from the repository, so it cannot satisfy a source-file requirement.
    """
    # JOIN LINE CONTINUATIONS FIRST. A `COPY a \\` spanning two lines would
    # otherwise be read as `COPY a` with `\\` as its destination -- which
    # happens to give the right answer for one source and the wrong one for
    # two. Correct by construction beats correct by luck.
    joined, buffer = [], ""
    for raw in dockerfile.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)

    inside, copied = False, []
    for raw in joined:
        line = raw.strip()
        match = re.match(r"^FROM\s+.*\bAS\s+(\S+)\s*$", line, re.IGNORECASE)
        if match:
            inside = match.group(1).lower() == stage.lower()
            continue
        if not inside or not line.upper().startswith("COPY"):
            continue
        if "--from=" in line:
            continue
        parts = line.split()[1:]
        parts = [p for p in parts if not p.startswith("--")]
        if len(parts) >= 2:
            copied.extend(parts[:-1])
    return copied


def _is_covered(relative: str, copied: list[str]) -> bool:
    """Is `relative` (a path under the build context) brought in by a COPY?"""
    for source in copied:
        source = source.rstrip("/")
        if source in (".", "./"):
            return True
        if relative == source or relative.startswith(source + "/"):
            return True
    return False


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_the_api_image_contains_every_module_the_service_imports():
    root = _repository_root()
    src = root / "src"
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    closure = import_closure(ENTRY_POINT, src)
    assert ENTRY_POINT in closure, (
        f"{ENTRY_POINT} itself was not resolved, so the walk never started")

    copied = copied_paths(dockerfile, API_STAGE)
    assert copied, (
        f"no COPY instructions were found in the '{API_STAGE}' stage, so "
        "either the stage name changed or the parser is broken")

    missing = sorted(
        (dotted, path.relative_to(root).as_posix())
        for dotted, path in closure.items()
        if not _is_covered(path.relative_to(root).as_posix(), copied))

    if missing:
        rendered = "\n".join(f"    {dotted}\n        {where}"
                             for dotted, where in missing)
        raise AssertionError(
            f"{len(missing)} module(s) the web service imports are NOT in the "
            f"'{API_STAGE}' image. Inside the container each raises "
            "ModuleNotFoundError at startup, gunicorn never binds, and the "
            "only symptom is a container that exits:\n" + rendered
            + "\n\n  the stage copies:\n"
            + "\n".join(f"    {c}" for c in copied))


# --------------------------------------------------------------------------- #
# The gate's own correctness. An assertion never observed to fail is not
# evidence, so every helper is exercised against inputs whose answer is known.
# --------------------------------------------------------------------------- #

def test_the_closure_reaches_the_module_that_broke_the_image():
    """`monitoring.model_registry` is reachable from `api.main` and lives in a
    sub-package the image did not copy until DOCKERCOPY-1."""
    root = _repository_root()
    closure = import_closure(ENTRY_POINT, root / "src")
    assert f"{PACKAGE}.monitoring.model_registry" in closure
    assert f"{PACKAGE}.api.attribution" in closure


def test_a_namespace_package_module_is_resolved():
    """`monitoring/` has no `__init__.py`. A resolver that required one would
    silently drop the module and the gate would pass while the image broke."""
    root = _repository_root()
    src = root / "src"
    assert not (src / PACKAGE / "monitoring" / "__init__.py").exists()
    assert _module_file(
        f"{PACKAGE}.monitoring.model_registry", src) is not None


def test_a_module_outside_the_package_is_not_followed():
    root = _repository_root()
    closure = import_closure(ENTRY_POINT, root / "src")
    assert all(d.startswith(PACKAGE) for d in closure)
    assert not any("pydantic" in d or "fastapi" in d for d in closure)


@pytest.mark.parametrize("relative,copied,expected", [
    ("src/genomic_variant_classifier/api/main.py",
     ["src/genomic_variant_classifier/api/"], True),
    ("src/genomic_variant_classifier/monitoring/model_registry.py",
     ["src/genomic_variant_classifier/api/"], False),
    ("src/genomic_variant_classifier/monitoring/model_registry.py",
     ["src/genomic_variant_classifier/monitoring/model_registry.py"], True),
    ("src/genomic_variant_classifier/__init__.py",
     ["src/genomic_variant_classifier/__init__.py"], True),
    ("anything/at/all.py", ["."], True),
    ("src/genomic_variant_classifier/apiary/x.py",
     ["src/genomic_variant_classifier/api"], False),
])
def test_coverage_matching_is_by_path_boundary_not_prefix(relative, copied,
                                                          expected):
    """The last case is the one that matters: a plain prefix test would count
    `api` as covering `apiary`."""
    assert _is_covered(relative, copied) is expected


def test_a_continued_copy_is_read_as_one_instruction():
    """A `COPY` split over two lines with a backslash must yield both sources,
    not one source and a stray backslash."""
    dockerfile = (
        "FROM base AS api\n"
        "COPY src/pkg/one.py \\\n"
        "     src/pkg/one.py\n"
        "COPY src/pkg/a.py src/pkg/b.py \\\n"
        "     dest/\n")
    assert copied_paths(dockerfile, "api") == [
        "src/pkg/one.py", "src/pkg/a.py", "src/pkg/b.py"]


def test_the_stage_parser_ignores_copies_from_other_layers():
    """`COPY --from=builder /opt/venv /opt/venv` brings in an image layer, not
    a repository file, so it can never satisfy a source requirement."""
    dockerfile = (
        "FROM base AS builder\n"
        "COPY requirements.txt ./\n"
        "FROM base AS api\n"
        "COPY --from=builder /opt/venv /opt/venv\n"
        "COPY src/pkg/api/ src/pkg/api/\n"
        "FROM base AS trainer\n"
        "COPY . .\n")
    assert copied_paths(dockerfile, "api") == ["src/pkg/api/"]
    assert copied_paths(dockerfile, "trainer") == ["."]
    assert copied_paths(dockerfile, "builder") == ["requirements.txt"]


def test_an_unknown_stage_yields_nothing_rather_than_everything():
    """A renamed stage must make the gate FAIL LOUD, not pass vacuously."""
    dockerfile = "FROM base AS api\nCOPY src/ src/\n"
    assert copied_paths(dockerfile, "does-not-exist") == []
