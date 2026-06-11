#!/usr/bin/env python3
"""simulate_ci_no_optional_deps.py -- reproduce the CI runner locally by making
`pandera` and `river` unimportable in-process, then verify (1) the agent layer
imports and (2) the affected tests PASS or SKIP -- never ERROR. Exit 0 = green.

Your .venv312 keeps pandera/river installed; this only blocks them for this run,
so you can confirm the CI fix without building a fresh venv. Author: Monzia Moodie.
"""
from __future__ import annotations
import builtins
import sys

BLOCK = {"pandera", "river"}
_real_import = builtins.__import__


def _blocking_import(name, *args, **kwargs):
    if name.split(".")[0] in BLOCK:
        raise ModuleNotFoundError(
            f"No module named '{name.split('.')[0]}' (simulated CI: optional dep absent)"
        )
    return _real_import(name, *args, **kwargs)


import pytest  # noqa: E402  -- import before installing the block

builtins.__import__ = _blocking_import
for _m in list(sys.modules):
    if _m.split(".")[0] in BLOCK:
        del sys.modules[_m]

# 1) orchestrator (-> every wrapper/detector) must import with both libs absent
try:
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator  # noqa: F401
    print("[1/2] orchestrator imports without pandera/river: OK")
except Exception as exc:  # noqa: BLE001
    print(f"[1/2] FAIL: orchestrator import broke without optional deps: {exc!r}")
    sys.exit(1)

# 2) affected tests must pass/skip (never error) in this lib-less process
rc = pytest.main(
    [
        "tests/unit/test_drift_monitor_agents.py",
        "tests/unit/test_schema_drift_monitor_agent.py",
        "-q",
        "-rs",
    ]
)
if rc == 0:
    print("[2/2] affected tests pass/skip cleanly under simulated CI: OK")
    sys.exit(0)
print(f"[2/2] FAIL: pytest returned {rc} under simulated CI (expected 0)")
sys.exit(int(rc))
