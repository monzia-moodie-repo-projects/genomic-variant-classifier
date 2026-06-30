"""test_orchestrator_lazy_registry.py -- Monzia Moodie

Phase 1 regression locks for the lazy agent registry. These guarantee the Data Freshness CI failure
cannot silently return: the eager registry pulled sklearn (via ModelInsightsAgent ->
model_insights_detector) and torch (via the EWC chain) at Orchestrator construction, which crashes in a
minimal environment that has neither. The lazy registry defers all agent imports to first use.

The "imports without sklearn" checks run in a SUBPROCESS. Blocking sklearn by mutating this
interpreter's sys.modules / builtins.__import__ would pollute module identity for every later test
(re-imported sklearn classes become new objects -> joblib.dump of sklearn-bearing models fails with
"not the same object"). A child process keeps the parent's module graph pristine.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run_in_subprocess(body: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(body)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_orchestrator_constructs_without_sklearn():
    """The whole point of Phase 1: construct the Orchestrator with sklearn unavailable."""
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(), dry_run=True)
        assert len(o._agent_registry) >= 1
        print("CONSTRUCT_NO_SKLEARN_OK", len(o._agent_registry))
        """
    )
    assert "CONSTRUCT_NO_SKLEARN_OK" in r.stdout, (
        "Orchestrator must construct without sklearn (lazy registry).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_orchestrator_constructs_without_sklearn_or_torch():
    """Stronger: neither sklearn NOR torch available at construction (full minimal-CI simulation)."""
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        _BLOCKED = ("sklearn", "torch", "xgboost", "lightgbm", "catboost", "shap", "transformers")
        def blk(n, *a, **k):
            if any(n == b or n.startswith(b + ".") for b in _BLOCKED):
                raise ModuleNotFoundError(f"No module named '{n}' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(), dry_run=True)
        print("CONSTRUCT_NO_HEAVY_OK", len(o._agent_registry))
        """
    )
    assert "CONSTRUCT_NO_HEAVY_OK" in r.stdout, (
        "Orchestrator must construct with all heavy ML deps unavailable (lazy registry).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_ci_data_freshness_pipeline_runs_without_sklearn():
    """The exact CI guarantee: the database_monitor pipeline runs to completion with sklearn blocked.

    This is the red->green proof for the Data Freshness workflow: it constructs the Orchestrator and
    runs the freshness pipeline (whose agents do not need sklearn) in a sklearn-free interpreter.
    """
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(), dry_run=True)
        results = o.run_pipeline("database_monitor")
        assert "DatabaseFreshnessMonitorAgent" in results, results
        action = results["DatabaseFreshnessMonitorAgent"].get("action")
        assert action != "error", f"freshness agent errored under sklearn-block: {results}"
        print("CI_FRESHNESS_OK", action)
        """
    )
    assert "CI_FRESHNESS_OK" in r.stdout, (
        "database_monitor pipeline must run clean without sklearn (the CI guarantee).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_registry_values_are_lazy():
    """Registry values must be _Lazy (not eagerly-imported classes)."""
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy

    o = Orchestrator(SharedState(), dry_run=True)
    assert o._agent_registry, "registry is empty"
    assert all(isinstance(v, _Lazy) for v in o._agent_registry.values()), (
        "every registry value must be a _Lazy descriptor"
    )
