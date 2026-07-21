"""The conformal package exports every module it contains.

WHY THIS FILE EXISTS
====================
`src/genomic_variant_classifier/conformal/__init__.py` imported five of the
seven modules that existed on disk. Verified by outcome on 2026-07-21 against
commit c663c89:

    hasattr(conformal, 'calibrate') = False
    hasattr(conformal, 'ordinal')   = False

The consequence is narrow, real, and confusing when it bites. This works, because
Python imports the submodule directly:

    from genomic_variant_classifier.conformal import ordinal          # OK

while this raises AttributeError, because nothing bound the name on the package:

    from genomic_variant_classifier import conformal
    conformal.ordinal.OrdinalConformalClassifier                      # AttributeError

The two forms look equivalent to a reader. They are not.

The list went stale TWICE -- once when calibrate.py landed, once when ordinal.py
landed -- because nothing connected "a file exists in this directory" to "its
name appears on that import line". Adding the two missing names would fix today
and guarantee a third occurrence.

These tests supply the missing connection. They walk the package directory and
assert that every module file is reachable as an attribute and declared in
__all__. Adding a submodule without exporting it now turns the suite RED, and
the failure message names the file and the fix. This is the same principle as
the suite-size ratchet: a list that cannot go stale, because going stale is
automatically detected rather than noticed by memory.

WHY NOT JUST IMPORT DYNAMICALLY
-------------------------------
A pkgutil loop in __init__.py would never go stale. It would also make the
public surface implicit, defeat static analysis and editor completion, and
silently import anything dropped into the directory -- a scratch file, a
half-finished module, an editor backup. Explicit imports are the right design;
they simply need a guard, which is what this file is.

Placement: tests/unit/test_conformal_package_exports.py
Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import genomic_variant_classifier.conformal as conformal


def _module_files() -> list[str]:
    """Every module in the package directory, from DISK, not from a hard-coded
    list. Reading the directory is the whole point: a constant here would go
    stale in exactly the way this file exists to prevent."""
    pkg_dir = Path(conformal.__file__).parent
    return sorted(
        p.stem for p in pkg_dir.glob("*.py")
        if p.stem != "__init__" and not p.stem.startswith("_")
    )


def test_the_package_directory_is_readable_and_non_empty():
    """If this fails, every other test here would pass vacuously by iterating an
    empty list -- which is the failure mode of any data-driven test suite."""
    mods = _module_files()
    assert mods, "found no modules in the conformal package directory"
    assert len(mods) >= 7, f"expected at least 7 modules, found {len(mods)}: {mods}"


@pytest.mark.parametrize("name", _module_files())
def test_every_module_is_reachable_as_a_package_attribute(name):
    assert hasattr(conformal, name), (
        f"conformal.{name} is not reachable.\n"
        f"src/genomic_variant_classifier/conformal/{name}.py exists on disk but "
        f"{name!r} is missing from the import in __init__.py.\n"
        f"Fix: add {name!r} to both the `from . import (...)` block and __all__."
    )


@pytest.mark.parametrize("name", _module_files())
def test_every_module_is_declared_in_dunder_all(name):
    assert name in conformal.__all__, (
        f"{name!r} is missing from conformal.__all__.\n"
        "__all__ declares the public surface; a module reachable but undeclared "
        "is reachable by accident rather than by intent."
    )


def test_dunder_all_contains_nothing_that_does_not_exist():
    """The complement. A name in __all__ with no module behind it breaks
    `from ... import *` at import time, for every consumer at once."""
    on_disk = set(_module_files())
    declared = set(conformal.__all__)
    phantom = declared - on_disk
    assert not phantom, (
        f"__all__ declares {sorted(phantom)}, which do not exist on disk")


def test_dunder_all_and_the_directory_agree_exactly():
    """Set equality in both directions, stated once, so a future reader can see
    the invariant without reconstructing it from the two tests above."""
    assert set(conformal.__all__) == set(_module_files())


def test_attribute_access_actually_resolves_to_the_module():
    """hasattr can be satisfied by anything -- a stray string constant would
    pass. This asserts the attribute IS the module it claims to be."""
    for name in _module_files():
        attr = getattr(conformal, name)
        expected = importlib.import_module(
            f"genomic_variant_classifier.conformal.{name}")
        assert attr is expected, (
            f"conformal.{name} is bound to {attr!r}, not to the module")


def test_the_two_previously_missing_modules_are_present():
    """A named regression test for the specific defect. The parametrised tests
    above would catch it, but they are generic; this one fails with the history
    attached, so a future reader learns what happened rather than only that
    something is wrong."""
    assert hasattr(conformal, "calibrate"), (
        "calibrate was unreachable from 2026-05 until 2026-07-21")
    assert hasattr(conformal, "ordinal"), (
        "ordinal was unreachable on the day it was added, 2026-07-21, commit c663c89")


def test_docstring_documents_the_modules():
    """A package docstring listing five of seven modules is a stale snapshot in
    the place a reader looks first."""
    doc = conformal.__doc__ or ""
    for name in _module_files():
        assert name in doc, (
            f"{name!r} is exported but absent from the package docstring")
