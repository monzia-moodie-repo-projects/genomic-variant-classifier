"""test_lazy_agent.py -- Monzia Moodie

Unit tests for the _Lazy agent descriptor (Phase 1 lazy registry).

_Lazy is the value type in Orchestrator._agent_registry. Each value imports and constructs its agent
only on first use, so the Orchestrator imports zero agent modules at construction. These tests lock the
contract every consumer depends on:

  * construction (creating a _Lazy) imports NOTHING
  * calling it with state constructs the agent: _Lazy("mod:Cls")(state) -> instance
  * attribute access delegates to the resolved class, so hasattr(lazy, "from_default_baseline")
    correctly reflects the real class (the orchestrator's drift-agent routing depends on this)
  * resolution is cached (imported once)
  * a bad spec is rejected early
"""
from __future__ import annotations

import sys
import types

import pytest

from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy


def _make_fake_module(mod_name: str, with_fdb: bool):
    """Create + register a throwaway module exposing an Agent class, tracking instantiation."""
    mod = types.ModuleType(mod_name)
    created = {"count": 0}

    class Agent:
        def __init__(self, state):
            self.state = state
            self.via = "ctor"
            created["count"] += 1

        def run(self, dry_run=False):
            return {"action": "ok"}

    if with_fdb:
        def from_default_baseline(cls, state):
            obj = cls(state)
            obj.via = "from_default_baseline"
            return obj
        Agent.from_default_baseline = classmethod(from_default_baseline)

    mod.Agent = Agent
    sys.modules[mod_name] = mod
    return mod, created


def test_construction_imports_nothing():
    """Creating a _Lazy must not import its target module."""
    mod_name = "_lazytest_noimport"
    sys.modules.pop(mod_name, None)
    lz = _Lazy(f"{mod_name}:Agent")
    # The module was never created/registered, yet _Lazy construction succeeds:
    assert mod_name not in sys.modules, "_Lazy construction must not import the target module"
    assert repr(lz).endswith("unresolved)")


def test_call_with_state_constructs_instance():
    mod_name = "_lazytest_call"
    _make_fake_module(mod_name, with_fdb=False)
    lz = _Lazy(f"{mod_name}:Agent")
    inst = lz("STATE")
    assert type(inst).__name__ == "Agent"
    assert inst.state == "STATE"
    assert inst.via == "ctor"


def test_attribute_delegation_reflects_from_default_baseline():
    """hasattr(lazy, 'from_default_baseline') must reflect the resolved class (drift routing depends on it)."""
    drift_name = "_lazytest_drift"
    plain_name = "_lazytest_plain"
    _make_fake_module(drift_name, with_fdb=True)
    _make_fake_module(plain_name, with_fdb=False)
    drift_lz = _Lazy(f"{drift_name}:Agent")
    plain_lz = _Lazy(f"{plain_name}:Agent")
    assert hasattr(drift_lz, "from_default_baseline") is True
    assert hasattr(plain_lz, "from_default_baseline") is False
    # And the delegated classmethod actually works:
    inst = drift_lz.from_default_baseline("STATE")
    assert inst.via == "from_default_baseline"


def test_resolution_is_cached():
    mod_name = "_lazytest_cache"
    mod, created = _make_fake_module(mod_name, with_fdb=False)
    lz = _Lazy(f"{mod_name}:Agent")
    c1 = lz.resolve()
    c2 = lz.resolve()
    assert c1 is c2, "resolve() must cache the class"
    assert repr(lz).endswith("resolved)")


def test_bad_spec_rejected_early():
    with pytest.raises(ValueError):
        _Lazy("no_colon_here")


def test_orchestrator_construction_imports_no_agent_modules():
    """End-to-end: constructing the real Orchestrator must not import any agent submodule."""
    # Drop any already-imported agent submodules so we can detect fresh imports.
    agents_pkg = "genomic_variant_classifier.agent_layer.agents"
    preexisting = {m for m in sys.modules if m.startswith(agents_pkg + ".")}

    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
    from genomic_variant_classifier.agent_layer.shared_state import SharedState

    before = {m for m in sys.modules if m.startswith(agents_pkg + ".")}
    _ = Orchestrator(SharedState(), dry_run=True)
    after = {m for m in sys.modules if m.startswith(agents_pkg + ".")}

    newly = after - before
    # Construction itself must not pull in agent submodules that weren't already loaded.
    # (We allow modules that some earlier test already imported; the point is *construction*
    # doesn't add new ones.)
    assert not newly, f"Orchestrator construction imported agent modules: {sorted(newly)}"
