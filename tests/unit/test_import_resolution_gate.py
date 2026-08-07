"""tests/unit/test_import_resolution_gate.py

Author: Monzia Moodie
Written 2026-08-07. REGISTRY-1a.

Every intra-package `from X import Y` in `src/` and `scripts/` must resolve.

WHY THIS TEST EXISTS. On 2026-08-07 `ModelRegistry` was found to be imported at
`continual_trainer.py:127`, `:266` and in `drift_monitor.yml`, and defined
nowhere. `git log --all -S "class ModelRegistry"` is empty: never written. It
survived because both Python imports are FUNCTION-LOCAL -- module collection
never executes them -- and `continual_trainer.py` has no test coverage. The
adaptive-retraining and model-promotion chain could not run, and nothing said
so. This test is the thing that catches it.

WHY IT DOES NOT REIMPLEMENT IMPORT RESOLUTION. A first attempt checked
`hasattr(module, name)` and reported ELEVEN working submodule imports as
broken, because `hasattr(package, "submodule")` is False until that submodule
has been imported -- `hasattr(email, "message")` is False while
`from email import message` succeeds. Any hand-written approximation of
`from X import Y` will drift from the real thing. So this test executes the
actual statement and lets Python answer.

WHY A CHILD INTERPRETER. Importing this package in-process mutates third-party
module globals: `variant_ensemble` applies an in-process repair to
`imodelsx.kan.kan_sklearn`, and the graph branch initialises PyTorch. A
structural integrity check must not reshape the interpreter the rest of the
suite runs in. Discovery happens here; resolution happens in a subprocess and
comes back as JSON.

WHAT THIS GATE DOES *NOT* COVER, STATED SO IT CANNOT BE MISREAD. It checks
IMPORTED NAMES. It does not check ATTRIBUTE REFERENCES. On the same day
`ModelRegistry` was found, `continual_trainer.py:299` was found to call
`current_pipe._prepare(...)` on an `InferencePipeline` that has no `_prepare`
method -- an `AttributeError` this gate cannot see, because no import mentions
it. A separate check would be needed for that class, and none exists yet.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "genomic_variant_classifier"

#: Generous: a cold import of the package pulls in scikit-learn, PyTorch and
#: the graph branch. A timeout is reported as a failure with its own message
#: rather than as a silent skip -- "the check did not finish" and "the check
#: passed" are different statements.
_RESOLVER_TIMEOUT_SECONDS = 600


def _repository_root() -> Path:
    """Walk up from this file until both `src` and `scripts` are present."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / PACKAGE).is_dir() and (
                candidate / "scripts").is_dir():
            return candidate
    raise AssertionError(
        "could not locate the repository root from "
        f"{Path(__file__).resolve()}; expected an ancestor containing both "
        "src/ and scripts/")


def module_name_for(path: Path, src_root: Path) -> str:
    """Dotted module name for a file inside the package source tree."""
    parts = list(path.relative_to(src_root).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def resolve_relative(level: int, module: str | None,
                     containing_package: str) -> str:
    """Turn `from ..a.b import c` into an absolute module name.

    `level` is the number of leading dots. Level 1 means "this package";
    each further dot climbs one package. Mirrors CPython's own rule, and is
    unit-tested below against hand-computed cases rather than trusted.
    """
    if level == 0:
        return module or ""
    bits = containing_package.split(".") if containing_package else []
    climb = level - 1
    if climb:
        if climb > len(bits):
            raise ValueError(
                f"relative import climbs {climb} level(s) above "
                f"{containing_package!r}")
        bits = bits[:-climb]
    base = ".".join(bits)
    if not module:
        return base
    return f"{base}.{module}" if base else module


def discover_import_pairs(root: Path) -> list[tuple[str, str, str, int]]:
    """Every intra-package `from X import Y`, as (module, name, file, line).

    Relative imports are resolved to absolute names, so the returned pairs are
    exactly what a child interpreter can execute directly.
    """
    src_root = root / "src"
    pairs: list[tuple[str, str, str, int]] = []
    for scan_root in (src_root, root / "scripts"):
        for path in sorted(scan_root.rglob("*.py")):
            if ".bak" in path.name:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8",
                                                errors="replace"))
            except SyntaxError:
                # A file that does not parse is a different defect, and the
                # suite has other gates for it. Reporting it here would mean
                # one failure standing for two causes.
                continue
            if path.is_relative_to(src_root):
                own = module_name_for(path, src_root)
                containing = (own if path.name == "__init__.py"
                              else own.rpartition(".")[0])
            else:
                containing = ""
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                try:
                    absolute = resolve_relative(node.level, node.module,
                                                containing)
                except ValueError:
                    absolute = ""
                if not absolute.startswith(PACKAGE):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    pairs.append((absolute, alias.name,
                                  str(path.relative_to(root).as_posix()),
                                  node.lineno))
    return pairs


_CHILD_RESOLVER = r'''
import json, sys
payload = json.loads(sys.stdin.read())
# REPLACE sys.path with the PARENT's, rather than prepending to it.
#
# `python -c` implicitly prepends "" -- the working directory. Prepending the
# parent's entries on top of that leaves "" in place, and a directory in the
# repository root whose name collides with an installed distribution then
# shadows it: Python 3 treats a directory without __init__.py as an implicit
# NAMESPACE PACKAGE, which imports successfully and has none of the real
# module's attributes. Measured 2026-08-07: this produced sixteen
# `AttributeError: module 'catalogue' has no attribute 'create'` failures
# against a tree whose suite imports that package perfectly well.
#
# Replacing makes the child's import environment exactly the parent's, which
# is the only environment whose verdict means anything here.
sys.path[:] = list(payload["path"])
failures = []
for module, name in payload["pairs"]:
    try:
        exec("from %s import %s" % (module, name), {})
    except Exception as exc:
        failures.append([module, name, "%s: %s" % (type(exc).__name__, exc)])
sys.stdout.write(json.dumps(failures))
'''


def resolve_in_child(pairs: list[tuple[str, str]], *, cwd: Path,
                     path: list[str] | None = None,
                     timeout: int = _RESOLVER_TIMEOUT_SECONDS) -> list[list]:
    """Execute each `from module import name` in a fresh interpreter.

    Returns the failures. Real import semantics, no approximation, and no
    contamination of the interpreter running the suite.

    `path` defaults to the parent's `sys.path`, which is correct when the
    parent is pytest. A CALLER WHOSE OWN DIRECTORY IS NOT A CLEAN IMPORT ROOT
    MUST PASS ITS OWN. Measured 2026-08-07: an installer run from a downloads
    folder put that folder at `sys.path[0]`, and a `catalogue.py` sitting there
    shadowed the installed `catalogue` distribution, producing sixteen
    `AttributeError: module 'catalogue' has no attribute 'create'` failures
    from inside thinc. The tree was fine; the path was not.
    """
    unique = sorted({(module, name) for module, name in pairs})
    # RESOLVE EVERY ENTRY TO AN ABSOLUTE PATH FIRST. An empty string -- and any
    # relative entry -- means THE PARENT'S working directory; sent verbatim it
    # would mean THE CHILD'S, which is a different directory whenever `cwd`
    # differs. Measured 2026-08-07: that let a shadowing module in the child's
    # working directory win despite the caller passing a clean path.
    base = os.getcwd()
    chosen = list(sys.path if path is None else path)
    absolute = [base if not entry else os.path.abspath(entry)
                for entry in chosen]
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_RESOLVER],
        input=json.dumps({"pairs": unique, "path": absolute}),
        capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
    if completed.returncode != 0:
        raise AssertionError(
            "the import resolver subprocess failed with code "
            f"{completed.returncode}. stderr tail:\n"
            f"{completed.stderr[-2000:]}")
    text = completed.stdout.strip()
    if not text:
        raise AssertionError(
            "the import resolver subprocess produced no output; it neither "
            "succeeded nor reported a failure. stderr tail:\n"
            f"{completed.stderr[-2000:]}")
    return json.loads(text)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_every_intra_package_imported_name_resolves():
    """The gate itself. A name that cannot be imported is a dead call site."""
    root = _repository_root()
    pairs = discover_import_pairs(root)
    assert pairs, (
        "no intra-package `from X import Y` statement was discovered at "
        "all, which means the discovery is broken rather than the code "
        "being clean")

    located: dict[tuple[str, str], list[str]] = {}
    for module, name, where, line in pairs:
        located.setdefault((module, name), []).append(f"{where}:{line}")
    failures = resolve_in_child([(m, n) for m, n, _, _ in pairs], cwd=root)

    if failures:
        rendered = "\n".join(
            f"    from {module} import {name}  ->  {why}\n"
            + "\n".join(f"        {site}"
                        for site in located.get((module, name), ["?"]))
            for module, name, why in failures)
        raise AssertionError(
            f"{len(failures)} imported name(s) do not resolve. Each is a call "
            "site that raises the moment it executes, and a function-local "
            "import of one is invisible to collection:\n" + rendered)


# --------------------------------------------------------------------------- #
# The gate's own correctness. An assertion never observed to fail is not
# evidence, so the resolver is exercised against a deliberate break here rather
# than only against a tree that happens to be clean.
# --------------------------------------------------------------------------- #

def test_the_resolver_reports_a_name_that_does_not_exist():
    root = _repository_root()
    failures = resolve_in_child(
        [(f"{PACKAGE}.monitoring.registry", "NoSuchNameExistsHere")], cwd=root)
    assert len(failures) == 1
    module, name, why = failures[0]
    assert name == "NoSuchNameExistsHere"
    assert "ImportError" in why


def test_the_resolver_accepts_a_submodule_import():
    """The false-positive class that a `hasattr` check gets wrong.

    `from package import submodule` is valid while `hasattr(package,
    "submodule")` is False until the submodule has been imported. An earlier
    approximation reported eleven working imports as broken on exactly this.
    """
    root = _repository_root()
    failures = resolve_in_child(
        [(f"{PACKAGE}.monitoring", "registry"),
         (f"{PACKAGE}.monitoring", "model_registry")], cwd=root)
    assert failures == []


def test_the_resolver_accepts_a_genuine_attribute_import():
    root = _repository_root()
    failures = resolve_in_child(
        [(f"{PACKAGE}.monitoring.model_registry", "ModelRegistry")], cwd=root)
    assert failures == []


def test_an_explicit_path_is_honoured_and_a_shadow_on_it_is_visible(tmp_path):
    """The `path` argument is real, and shadowing is a genuine hazard.

    `statistics` is chosen because the child interpreter imports only `json`
    and `sys` before replacing its path, so nothing has cached it. A module of
    that name at the front of the path therefore wins, exactly as
    `catalogue.py` in a downloads folder did on 2026-08-07.
    """
    (tmp_path / "statistics.py").write_text("# a shadow\n", encoding="utf-8")
    root = _repository_root()

    clean = resolve_in_child([("statistics", "mean")], cwd=root)
    assert clean == [], f"the stdlib import failed on a clean path: {clean}"

    shadowed = resolve_in_child([("statistics", "mean")], cwd=root,
                                path=[str(tmp_path)] + list(sys.path))
    assert len(shadowed) == 1, (
        "a module at the front of the path did not shadow the standard "
        f"library, so the `path` argument is not being honoured: {shadowed}")
    assert shadowed[0][1] == "mean"


def test_the_resolver_ignores_a_shadow_in_the_working_directory(tmp_path):
    """The defect that made this gate report sixteen phantom failures.

    A directory named after an installed distribution, carrying no
    `__init__.py`, is an implicit namespace package. If the child interpreter
    has the working directory on its path, that directory shadows the real
    module -- imports "succeed" and every attribute is missing. The child now
    takes the parent's path verbatim, so the working directory contributes
    nothing unless the parent already had it.
    """
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue" / "notes.md").write_text("data, not a package",
                                                     encoding="utf-8")
    root = _repository_root()
    failures = resolve_in_child(
        [(f"{PACKAGE}.monitoring", "model_registry")], cwd=tmp_path)
    assert failures == [], (
        "a directory in the working directory shadowed an installed package: "
        f"{failures}")
    assert root.is_dir()


@pytest.mark.parametrize("level,module,containing,expected", [
    (0, "genomic_variant_classifier.evaluation.population", "",
     "genomic_variant_classifier.evaluation.population"),
    (1, "population", "genomic_variant_classifier.evaluation",
     "genomic_variant_classifier.evaluation.population"),
    (1, None, "genomic_variant_classifier.evaluation",
     "genomic_variant_classifier.evaluation"),
    (2, "models", "genomic_variant_classifier.evaluation",
     "genomic_variant_classifier.models"),
    (2, "capabilities.core", "genomic_variant_classifier.a.b",
     "genomic_variant_classifier.a.capabilities.core"),
    (3, "data", "genomic_variant_classifier.a.b",
     "genomic_variant_classifier.data"),
])
def test_relative_imports_resolve_the_way_python_resolves_them(
        level, module, containing, expected):
    assert resolve_relative(level, module, containing) == expected


def test_a_relative_import_climbing_past_the_root_is_reported():
    with pytest.raises(ValueError, match="climbs"):
        resolve_relative(5, "x", "genomic_variant_classifier")


@pytest.mark.parametrize("relative,expected", [
    ("genomic_variant_classifier/__init__.py", "genomic_variant_classifier"),
    ("genomic_variant_classifier/evaluation/__init__.py",
     "genomic_variant_classifier.evaluation"),
    ("genomic_variant_classifier/evaluation/thresholds.py",
     "genomic_variant_classifier.evaluation.thresholds"),
])
def test_module_names_are_derived_from_the_path(relative, expected, tmp_path):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    assert module_name_for(path, tmp_path) == expected


def test_discovery_covers_both_source_and_scripts():
    root = _repository_root()
    where = {entry[2] for entry in discover_import_pairs(root)}
    assert any(w.startswith("src/") for w in where), (
        "no imports discovered under src/, so the discovery is broken")
    assert any(w.startswith("scripts/") for w in where), (
        "no imports discovered under scripts/; the entry points import the "
        "package and this gate is meant to cover them")


def test_discovery_finds_function_local_imports():
    """The defect class this gate exists for. `ModelRegistry` was imported
    inside a method body, so collection never touched it. `ast.walk` reaches
    nested statements; a top-level-only scan would not."""
    root = _repository_root()
    tree = ast.parse(
        "def f():\n"
        "    from genomic_variant_classifier.monitoring import registry\n"
        "    return registry\n")
    nested = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert len(nested) == 1, "ast.walk must reach imports inside a function"
    assert root.is_dir()
