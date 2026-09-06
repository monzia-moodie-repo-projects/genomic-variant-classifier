"""The contract layer must stay loadable when everything else is broken.

Created 2026-09-05. ADR-0005.

WHY
---
`repository_measurement` exists so that diagnostic and repair tools can state
what they measured. Those tools are needed exactly when the application cannot
initialise. A contract layer that imports the model stack, the transaction
machinery or the provenance package is absent at the moment it matters most.

WHY AST AND NOT GREP
--------------------
The adopted plan is explicit: do not use a brittle grep, use AST. This project
has the evidence. On 2026-08-30 a scan for `import audit_data_tree` reported
ZERO while the gate was demonstrably wired, because the wiring loads by path;
the same scan counted 31 "invocations" of `preflight_data_guard`, every one a
line of Markdown prose. A source check that reads text is wrong in both
directions -- it passes on dead code and fails on a clean refactor.

Parsing the import statements is not a heuristic. It is the actual set of
module-level and function-level imports the interpreter will execute.

WHAT THIS DOES NOT CHECK
------------------------
Dynamic imports -- `importlib.import_module(name)` with a computed name, or
`__import__` -- are not recoverable by parsing, and this file does not pretend
otherwise. A test asserting it caught them would be worse than one that says
it does not. The package is small, standard-library-only by construction, and
its dynamic-import surface is asserted to be empty by
`test_the_package_performs_no_dynamic_import`.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = (Path("src") / "genomic_variant_classifier"
           / "repository_measurement")

#: Siblings this layer must never depend on. Importing any of them would make
#: the diagnostic layer fail whenever the subject of diagnosis fails.
FORBIDDEN_PREFIXES = (
    "genomic_variant_classifier.provenance",
    "genomic_variant_classifier.repository_records",
    "genomic_variant_classifier.transactions",
    "genomic_variant_classifier.state",
    "genomic_variant_classifier.evaluation",
    "genomic_variant_classifier.models",
)

#: Everything the package is permitted to import from outside itself. The
#: adopted plan names the standard-library set; `re` is included because a
#: schema parser needs it and it carries no dependency.
PERMITTED_ROOTS = frozenset({
    "__future__", "dataclasses", "enum", "hashlib", "json", "typing",
    "collections", "re",
})

#: Names that would make imports unrecoverable by parsing. Asserted absent so
#: the AST result is COMPLETE for this package rather than merely correct as
#: far as it goes.
DYNAMIC_IMPORT_NAMES = ("import_module", "__import__", "spec_from_file_location")


@pytest.fixture(scope="module")
def sources() -> dict[str, ast.Module]:
    if not PACKAGE.is_dir():
        pytest.fail(
            f"{PACKAGE} does not exist. ADR-0005 sites the measurement "
            f"contract there; if it moved, amend ADR-0005 and this locator "
            f"together."
        )
    found = {}
    for p in sorted(PACKAGE.glob("*.py")):
        found[p.name] = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    assert found, (
        f"no Python sources under {PACKAGE}. A guard with nothing to guard is "
        f"not a guard."
    )
    return found


def _absolute_imports(tree: ast.Module):
    """(module, name) for every absolute import, at any nesting depth."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                out.append((node.module or "", alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, alias.name))
    return out


def test_the_package_has_the_modules_adr_0005_declares(sources):
    """ADR-0005 enumerates the file set. A module appearing or vanishing
    silently would make the record describe something that no longer exists."""
    assert set(sources) == {
        "__init__.py", "corpus.py", "evidence.py", "claims.py", "report.py",
        "serialization.py",
    }, sorted(sources)


def test_the_package_imports_no_sibling_machinery(sources):
    """The dependency direction is one-way: consumers may depend on this
    layer; it depends on nothing in the package."""
    violations = []
    for name, tree in sorted(sources.items()):
        for module, _sym in _absolute_imports(tree):
            if any(module.startswith(p) for p in FORBIDDEN_PREFIXES):
                violations.append(f"    {name}: {module}")
    assert not violations, (
        "the measurement contract imports sibling machinery:\n"
        + "\n".join(violations)
        + "\n\nThis layer must load when the application cannot. A diagnostic "
          "that fails whenever its subject fails is absent when it matters."
    )


def test_the_package_imports_only_the_standard_library(sources):
    """Standard-library only, so the layer survives a broken environment."""
    foreign = []
    for name, tree in sorted(sources.items()):
        for module, _sym in _absolute_imports(tree):
            root = module.split(".")[0]
            if root and root not in PERMITTED_ROOTS:
                foreign.append(f"    {name}: {module}")
    assert not foreign, (
        "the measurement contract imports outside the standard library:\n"
        + "\n".join(foreign)
        + f"\n\nPermitted roots: {sorted(PERMITTED_ROOTS)}\n"
          f"Do not widen this to make an import pass -- the constraint is the "
          f"point."
    )


def test_the_package_performs_no_dynamic_import(sources):
    """Parsing recovers static imports only, so the dynamic surface must be
    empty for the AST result to be COMPLETE rather than merely correct as far
    as it goes."""
    dynamic = []
    for name, tree in sorted(sources.items()):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = getattr(node.func, "id", "") or getattr(
                    node.func, "attr", "")
                if target in DYNAMIC_IMPORT_NAMES:
                    dynamic.append(f"    {name}:{node.lineno}: {target}")
    assert not dynamic, (
        "the measurement contract imports dynamically:\n" + "\n".join(dynamic)
        + "\n\nA dynamic import is not recoverable by parsing, so the two "
          "tests above would no longer be complete."
    )


def test_the_package_neither_mutates_the_repository_nor_shells_out(sources):
    """A measurement model describes evidence; it does not acquire or publish
    it. Acquisition belongs to the instrument, publication to transactions."""
    banned = ("subprocess", "shutil", "os")
    found = []
    for name, tree in sorted(sources.items()):
        for module, _sym in _absolute_imports(tree):
            if module.split(".")[0] in banned:
                found.append(f"    {name}: {module}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = getattr(node.func, "attr", "")
                if target in ("write_text", "write_bytes", "unlink", "mkdir",
                              "rmtree", "run", "system"):
                    found.append(f"    {name}:{node.lineno}: {target}()")
    assert not found, (
        "the measurement contract acquires or mutates state:\n"
        + "\n".join(found)
        + "\n\nPrefer pure functions. This layer expresses scope and meaning; "
          "Git, the filesystem and the transaction machinery belong elsewhere."
    )


# ---------------------------------------------------------------------------
# Negative controls -- proof that these checks can REJECT
# ---------------------------------------------------------------------------

def test_the_forbidden_check_rejects_a_sibling_import():
    tree = ast.parse(
        "from genomic_variant_classifier.transactions.install_plan "
        "import InstallPlan\n")
    hits = [m for m, _s in _absolute_imports(tree)
            if any(m.startswith(p) for p in FORBIDDEN_PREFIXES)]
    assert hits == ["genomic_variant_classifier.transactions.install_plan"], (
        "the sibling-import check failed to reject a forbidden import."
    )


def test_the_stdlib_check_rejects_a_third_party_import():
    tree = ast.parse("import pandas as pd\nfrom numpy import array\n")
    foreign = [m for m, _s in _absolute_imports(tree)
               if m.split(".")[0] not in PERMITTED_ROOTS]
    assert sorted(set(foreign)) == ["numpy", "pandas"], (
        "the standard-library check failed to reject third-party imports."
    )


def test_the_import_walker_sees_a_FUNCTION_LEVEL_import():
    """A module-level-only walker would miss an import inside a function, and
    a deferred import is still an import."""
    tree = ast.parse("def f():\n    import subprocess\n    return subprocess\n")
    assert ("subprocess", "subprocess") in _absolute_imports(tree), (
        "the walker only inspects module scope; a deferred import would "
        "bypass every check in this file."
    )


def test_the_relative_import_is_not_treated_as_absolute():
    """`from .corpus import X` is internal and must not be judged foreign."""
    tree = ast.parse("from .corpus import CorpusSpec\n")
    assert _absolute_imports(tree) == [], (
        "a relative import was treated as an absolute one; every internal "
        "import would then be reported as a violation."
    )
