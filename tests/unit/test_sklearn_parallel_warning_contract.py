"""Pin the premise behind the scikit-learn `parallel.delayed` warning filter.

WHY THIS TEST EXISTS
====================
pyproject.toml silences one specific scikit-learn UserWarning:

    `sklearn.utils.parallel.delayed` should be used with
    `sklearn.utils.parallel.Parallel` to make it possible to propagate the
    scikit-learn configuration of the current thread to the joblib workers.

That warning is a thread-config-propagation note emitted by scikit-learn's OWN
estimators (e.g. RandomForest with n_jobs != 1). It is not a correctness signal
and it is not this project misusing `delayed`. Silencing it is only safe as long
as that remains true. This test pins the premise so that the day it stops being
true -- the day a bare joblib.delayed is fed to a scikit-learn Parallel in project
code, which WOULD make the warning our own signal -- the suite goes red and the
filter must be reconsidered.

The warning itself is execution-order dependent and cannot be reliably reproduced
from a single test, so this test does NOT try to trigger it. It pins the STRUCTURAL
premise instead: every `delayed(...)` call in the project source is paired with a
joblib `Parallel` in the same module, never handed to a scikit-learn Parallel.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "genomic_variant_classifier"


def _python_files():
    return sorted(_SRC.rglob("*.py"))


def test_filter_is_present_and_message_pinned():
    """The filter must exist and be pinned to the exact message, never broadened
    to a bare category."""
    pyproject = _SRC.parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "sklearn.utils.parallel.delayed" in text
    assert ".*:UserWarning" in text
    # never silence the whole category
    assert '"ignore::UserWarning"' not in text


def test_project_delayed_usage_is_paired_with_joblib_parallel():
    """Every module that imports or calls `delayed` must also use joblib `Parallel`
    in the same module. A bare joblib.delayed handed to a scikit-learn Parallel is
    exactly what would make the silenced warning our own signal; if this ever
    appears, this test fails and the filter must be revisited."""
    offenders = []
    for path in _python_files():
        src = path.read_text(encoding="utf-8")
        if "delayed(" not in src:
            continue
        # a module using delayed( must also reference Parallel( in the same file
        if "Parallel(" not in src:
            offenders.append(str(path.relative_to(_SRC)))
    assert not offenders, (
        "these modules call delayed(...) without a joblib Parallel(...) in the "
        f"same module -- the parallel.delayed warning filter may be masking a real "
        f"misuse: {offenders}")


def test_delayed_imports_are_from_joblib_not_sklearn():
    """The `delayed` used in project code must be joblib's, paired with joblib's
    Parallel. If a module imported `delayed` from sklearn.utils.parallel, the
    warning would be a genuine signal about that module and must not be silenced."""
    sklearn_delayed_importers = []
    for path in _python_files():
        src = path.read_text(encoding="utf-8")
        if "delayed" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "sklearn" in node.module and "parallel" in node.module:
                    for alias in node.names:
                        if alias.name == "delayed":
                            sklearn_delayed_importers.append(
                                str(path.relative_to(_SRC)))
    assert not sklearn_delayed_importers, (
        "these modules import `delayed` from sklearn.utils.parallel; the silenced "
        "warning would be a real signal for them and the filter must not hide it: "
        f"{sklearn_delayed_importers}")
