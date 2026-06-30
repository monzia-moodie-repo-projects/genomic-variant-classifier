"""_lazy_agent.py -- Monzia Moodie

Lazy agent value-descriptor for the Orchestrator's agent registry.

The registry is a plain dict literal ``{ "AgentName": _Lazy("module:Class"), ... }``. Keeping it a
literal means the agent-liveness checker (scripts/check_agents_active.py), which AST-parses the
orchestrator source for the registry's string keys, still finds every registered agent without
importing anything.

Each value is a ``_Lazy`` that imports and constructs its agent only when called:

    registry["DataReadinessAgent"](shared_state)        -> a DataReadinessAgent instance

and transparently delegates attribute access to the resolved class, so the orchestrator's drift-agent
routing keeps working:

    if hasattr(agent_cls, "from_default_baseline"):     -> reflects the real class
        agent = agent_cls.from_default_baseline(state)
    else:
        agent = agent_cls(state)

No agent module is imported at Orchestrator construction. The first attribute access or call resolves
(imports) the class and caches it; resolution happens in run_pipeline (when the agent is about to run),
never at construction. Each agent's import -- and its transitive heavy dependencies (torch, sklearn,
the detector modules) -- is isolated to its own first use, so a missing/broken optional dependency in
one agent cannot break Orchestrator construction or any other agent.
"""
from __future__ import annotations

import importlib
from typing import Any


class _Lazy:
    """Callable, attribute-delegating descriptor that imports + caches an agent class on first use.

    Parameters
    ----------
    spec : str
        ``"module.dotted.path:ClassName"`` locating the agent class.
    """

    __slots__ = ("_spec", "_cls")

    def __init__(self, spec: str) -> None:
        if ":" not in spec:
            raise ValueError(f"_Lazy spec must be 'module:Class', got {spec!r}")
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_cls", None)

    def resolve(self) -> type:
        """Import (once) and return the agent class, caching it."""
        cls = object.__getattribute__(self, "_cls")
        if cls is None:
            spec = object.__getattribute__(self, "_spec")
            modpath, clsname = spec.split(":", 1)
            cls = getattr(importlib.import_module(modpath), clsname)
            object.__setattr__(self, "_cls", cls)
        return cls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Construct the agent: ``registry[name](state, **kw)`` -> instance."""
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access (e.g. ``from_default_baseline``) to the resolved class.

        __getattr__ is only invoked for attributes NOT found via __slots__ (i.e. not _spec/_cls/
        the methods), so ``hasattr(lazy, "from_default_baseline")`` resolves the class and reflects
        whether the real class defines it. This runs in run_pipeline (agent-execution time), never at
        Orchestrator construction.
        """
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        cls = object.__getattribute__(self, "_cls")
        spec = object.__getattribute__(self, "_spec")
        return f"_Lazy({spec!r}, {'resolved' if cls is not None else 'unresolved'})"
