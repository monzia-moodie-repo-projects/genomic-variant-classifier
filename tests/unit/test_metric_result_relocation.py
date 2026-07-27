"""Acceptance criteria for the MetricResult vocabulary relocation (2026-07-27).

WHAT MOVED, AND WHY
===================
`MetricResult` was defined in `clustering_metrics.py` -- a 1,326-line panel
module -- and imported by two others, `representation_geometry.py` and
`norm_angle_probe.py`. It was therefore already a SHARED result contract living
inside a single panel, and its `__post_init__` depends on `MetricStatus`, which
lives in `capabilities.py`. The dependency ran UPWARD, from the vocabulary layer
into a panel.

It now lives in `capabilities.py` beside `MetricStatus`, and `clustering_metrics`
re-exports it, so every historical import path keeps working and resolves to the
same object.

THE PRECEDENT
-------------
`BootstrapUnit` received this exact relocation for this exact reason, and the
identity guarantee is already pinned for the status enum by
`test_there_is_exactly_one_metric_status_class`, whose docstring reads: *two
enums sharing a name is the divergence problem removed in b8275a0, where the
legacy evaluator was DELETED rather than wrapped because two evaluation contracts
in one codebase invite drift.*

This file states the same guarantee for the result type, one level up.

WHAT THIS COMMIT DELIBERATELY DOES NOT DO
-----------------------------------------
No registry exists yet. No metric behaviour changes. `metrics.evaluate()` is
untouched. The relocation lands alone so that a later registry regression cannot
be confused with a vocabulary regression -- the sequencing lesson from the
`BootstrapUnit` move: stabilise the vocabulary layer before adding consumers.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import (
    capabilities,
    clustering_metrics,
    norm_angle_probe,
    representation_geometry,
)
from genomic_variant_classifier.evaluation.capabilities import MetricResult, MetricStatus

_SRC = Path(capabilities.__file__).parent


# --------------------------------------------------------------------------- #
# 1. Defined once, and every path resolves to THE SAME OBJECT
# --------------------------------------------------------------------------- #
def test_capabilities_owns_metric_result():
    assert MetricResult.__module__ == (
        "genomic_variant_classifier.evaluation.capabilities"
    ), "MetricResult must be DEFINED in the vocabulary layer, not re-exported into it"


def test_clustering_metrics_re_exports_the_same_object():
    """The identity check the relocation exists to guarantee. Two classes sharing
    a name -- even with identical fields -- is the divergence problem removed in
    b8275a0."""
    assert clustering_metrics.MetricResult is MetricResult


@pytest.mark.parametrize("module", [representation_geometry, norm_angle_probe])
def test_every_existing_consumer_resolves_the_same_object(module):
    """Both imported it from clustering_metrics before the move. Neither may end
    up holding a duplicate type with equivalent fields."""
    assert module.MetricResult is MetricResult


def test_there_is_exactly_one_metric_result_class_in_the_package():
    """A grep-level guarantee, so a second definition cannot be added quietly in
    a module nobody thought to import here."""
    defs = []
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "MetricResult":
                defs.append(f"{path.name}:{node.lineno}")
    assert len(defs) == 1, (
        f"expected exactly ONE MetricResult definition in the package; found "
        f"{len(defs)}: {defs}. A second class with the same name and equivalent "
        "fields is the divergence this relocation exists to prevent.")
    assert defs[0].startswith("capabilities.py:"), (
        f"the single definition must live in capabilities.py; found {defs[0]}")


# --------------------------------------------------------------------------- #
# 2. Behaviour is UNCHANGED -- every invariant still fires
# --------------------------------------------------------------------------- #
def test_an_ok_result_with_a_finite_value_constructs():
    r = MetricResult(value=0.9, status=MetricStatus.OK)
    assert r.value == 0.9 and r.status is MetricStatus.OK and r.reason is None


@pytest.mark.parametrize("kwargs,fragment", [
    (dict(value=0.9, status=MetricStatus.OK, reason="why"), "must not carry a reason"),
    (dict(value=float("nan"), status=MetricStatus.OK), "must carry a finite value"),
    (dict(value=float("nan"), status=MetricStatus.UNDEFINED), "requires a nonempty reason"),
    (dict(value=0.5, status=MetricStatus.UNDEFINED, reason="r"), "must carry NaN"),
])
def test_the_invariants_still_fire(kwargs, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        MetricResult(**kwargs)


def test_a_non_metric_status_is_still_a_TypeError():
    with pytest.raises(TypeError, match="must be a MetricStatus"):
        MetricResult(value=0.9, status="ok")


def test_np_isfinite_semantics_are_preserved_not_swapped_for_math():
    """np.isfinite was kept deliberately, and this test must be able to DETECT a
    swap to math.isfinite.

    Measured 2026-07-27: the two agree on every SCALAR input -- python float,
    python int, numpy float64/float32/int64, NaN, infinity, bool, 0-d array --
    and both reject None and str. They differ on exactly one shape: a ONE-ELEMENT
    ARRAY, which numpy silently accepts as finite (`not np.isfinite([0.5])` is
    False) while math raises TypeError.

    So a one-element array is the ONLY discriminating input. An earlier version of
    this test used np.float64 and therefore passed under BOTH implementations --
    a guard that could not fail, proven by sabotage on 2026-07-27.

    Note what is being pinned: that a one-element array is ACCEPTED. That is
    arguably wrong -- `value` is typed `float` -- but it is the CURRENT behaviour,
    and this commit's acceptance criterion is that behaviour does not change.
    Tightening it is a deliberate follow-up, not a side effect of a relocation.
    """
    # scalars: identical either way, so these prove nothing about the swap
    assert MetricResult(value=np.float64(0.5), status=MetricStatus.OK).value == 0.5
    with pytest.raises(ValueError):
        MetricResult(value=np.float64("nan"), status=MetricStatus.OK)

    # THE DISCRIMINATING CASE. Under np.isfinite this constructs; under
    # math.isfinite it raises TypeError.
    r = MetricResult(value=np.array([0.5]), status=MetricStatus.OK)
    assert r.value == np.array([0.5]), (
        "a one-element array must still be accepted, as np.isfinite does; if this "
        "raised, np.isfinite has been swapped for math.isfinite and the behaviour "
        "of the result vocabulary has changed")


# --------------------------------------------------------------------------- #
# 3. The layering the relocation exists to restore
# --------------------------------------------------------------------------- #
def test_capabilities_does_not_import_any_evaluation_panel():
    """The vocabulary layer sits at the BOTTOM. If it imported a panel, the
    dependency it was moved to fix would simply have reversed."""
    tree = ast.parse((_SRC / "capabilities.py").read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level > 0 or node.module.startswith("genomic_variant_classifier"):
                bad.append(node.module or f"(relative level {node.level})")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("genomic_variant_classifier"):
                    bad.append(a.name)
    assert not bad, f"capabilities.py must not import project modules; found {bad}"


def test_capabilities_still_imports_without_scikit_learn():
    """capabilities.py gained a numpy import in this commit. The contract that
    governs the evaluation layer forbids scikit-learn, NOT numpy -- but the
    guarantee is worth stating directly here rather than only through
    evaluator.py, because this module is what the registry will import next."""
    code = textwrap.dedent("""
        import builtins, importlib
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        m = importlib.import_module("genomic_variant_classifier.evaluation.capabilities")
        r = m.MetricResult(value=0.5, status=m.MetricStatus.OK)
        assert r.value == 0.5
        print("CAPABILITIES_IMPORT_OK")
    """)
    env = {"PYTHONPATH": ":".join(p for p in sys.path if p)}
    import os
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, **env})
    assert "CAPABILITIES_IMPORT_OK" in r.stdout, (
        f"capabilities must import without scikit-learn.\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def test_the_relocation_did_not_remove_anything_else_from_clustering_metrics():
    """The first attempt at this move extracted from MetricResult to the NEXT
    dataclass and silently swallowed `aggregate`, a top-level function that sat
    between them. tests/unit/test_clustering_metrics.py caught it with an
    ImportError. This pins the neighbours so a future extraction cannot repeat
    it."""
    for name in ("aggregate", "EstimatedMetric", "ClusteringPopulationAccounting"):
        assert hasattr(clustering_metrics, name), (
            f"clustering_metrics.{name} disappeared; the relocation must move "
            "MetricResult and nothing else")


# --------------------------------------------------------------------------- #
# 4. No registry yet -- this commit is the vocabulary move ALONE
# --------------------------------------------------------------------------- #
def test_no_metric_registry_exists_yet():
    """Commit 1 is the relocation on its own, so a later registry regression can
    never be mistaken for a vocabulary regression. Delete this test in the commit
    that introduces the registry."""
    assert not (_SRC / "registry.py").exists(), (
        "evaluation/registry.py exists; the vocabulary relocation was supposed to "
        "land alone. If the registry is being added, remove this test in that commit.")
