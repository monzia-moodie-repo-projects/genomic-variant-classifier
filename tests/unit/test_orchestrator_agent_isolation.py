"""test_orchestrator_agent_isolation.py -- Monzia Moodie

Phase 1 regression lock for graceful failure handling (criterion #4).

With the lazy registry, an agent's import happens on first use inside run_pipeline -- at the
hasattr/from_default_baseline/constructor step, not at Orchestrator construction. The per-agent guard
was therefore widened to wrap construction as well as agent.run(), so a single agent whose import fails
(a missing optional dependency, or a broken module) is isolated: it is recorded as
{"action": "error", ...} and the pipeline continues with the remaining agents.

These tests prove that isolation directly, by injecting a _Lazy that raises at resolution time into a
real Orchestrator's registry and confirming the pipeline survives.
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy


class _ExplodingLazy(_Lazy):
    """A _Lazy whose resolution always raises -- simulates an agent with a missing/broken import."""

    def __init__(self, message: str = "simulated missing dependency"):
        super().__init__("nonexistent.module.that.will.never.exist:Agent")
        object.__setattr__(self, "_message", message)

    def resolve(self):  # type: ignore[override]
        raise ModuleNotFoundError(object.__getattribute__(self, "_message"))


def _pick_real_agents(orch: Orchestrator, n: int):
    """Return up to n registry keys whose agents construct + run cleanly (no heavy deps needed)."""
    healthy = []
    for name, lazy in orch._agent_registry.items():
        try:
            cls = lazy.resolve()
        except Exception:
            continue
        healthy.append(name)
        if len(healthy) >= n:
            break
    return healthy


def test_broken_agent_is_isolated_pipeline_survives(monkeypatch):
    """A broken agent in the MIDDLE of a pipeline must not crash agents before or after it."""
    orch = Orchestrator(SharedState(), dry_run=True)

    healthy = _pick_real_agents(orch, 2)
    if len(healthy) < 2:
        pytest.skip("need >=2 import-clean agents to test mid-pipeline isolation")

    # Inject a broken agent into the registry, between two healthy ones.
    broken_name = "__InjectedBrokenAgent__"
    orch._agent_registry[broken_name] = _ExplodingLazy("simulated missing dependency for isolation test")

    # Build a pipeline: healthy[0] -> broken -> healthy[1], by calling the loop via a temp pipeline.
    # run_pipeline reads PIPELINE_DEFINITIONS, so we drive the agent list directly through a monkeypatch.
    from genomic_variant_classifier.agent_layer import orchestrator as orch_mod
    pipeline = [healthy[0], broken_name, healthy[1]]
    monkeypatch.setitem(orch_mod.PIPELINE_DEFINITIONS, "__isolation_test__", pipeline)

    results = orch.run_pipeline("__isolation_test__")

    assert results.get(healthy[0], {}).get("action") != "error", (
        f"healthy agent before the broken one must run: {results.get(healthy[0])}"
    )
    assert results.get(broken_name, {}).get("action") == "error", (
        f"broken agent must be recorded as error, not crash the pipeline: {results.get(broken_name)}"
    )
    assert results.get(healthy[1], {}).get("action") != "error", (
        f"healthy agent AFTER the broken one must still run (pipeline survived): {results.get(healthy[1])}"
    )


def test_broken_agent_error_is_recorded_not_raised(monkeypatch):
    """The pipeline must return normally (no exception) even when an agent's construction fails."""
    orch = Orchestrator(SharedState(), dry_run=True)
    broken_name = "__InjectedBrokenAgent2__"
    orch._agent_registry[broken_name] = _ExplodingLazy()

    from genomic_variant_classifier.agent_layer import orchestrator as orch_mod
    monkeypatch.setitem(orch_mod.PIPELINE_DEFINITIONS, "__broken_only__", [broken_name])

    # Must NOT raise:
    results = orch.run_pipeline("__broken_only__")
    assert results[broken_name]["action"] == "error"
    assert "error" in results[broken_name]
