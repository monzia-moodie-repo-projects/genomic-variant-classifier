# INCIDENT 2026-06-11 -- CI red: agent-layer optional deps (pandera, river) absent in CI

**Status:** RESOLVED -- verified green at CI run #304 (commit 92ff4a2); both pytest 3.11 and 3.12 legs passing.

## Summary
The agent-layer repair (commits 8619afc, 21e835d) added drift detectors and tests that import
`pandera` (schema_drift_agent) and `river` (annotation_policy_agent) at module level. Neither
library is declared in any requirements file, so the CI runner -- which installs
requirements-api.lock + requirements.txt + requirements-dev.txt + `pip install -e .` -- does not
have them. The local dev venv (.venv312) does, so the full suite passed locally (873 passed / 6
skipped) while CI failed. CI was red from #302 through #303.

## Timeline
- #301 (5eeb670) and earlier: green.
- #302 (21e835d): first run carrying the new agent tests; FAILED at collection:
  `tests/unit/test_schema_drift_monitor_agent.py:6: import pandera.pandas as pa`
  -> `ModuleNotFoundError: No module named 'pandera'`. pytest `-x` stopped there.
- #303 (0bbeb6d, docs commit): same failure (code unchanged).
- 2026-06-11: detected from the Actions tab; local runs had masked it.
- #304 (92ff4a2): green -- fix verified.

## Root cause
1. schema_drift_agent.py imported `pandera.pandas` at module top.
2. annotation_policy_agent.py imported `river.drift` at module top.
3. test_schema_drift_monitor_agent.py imported `pandera.pandas` at module top.
The orchestrator imports every wrapper, and each wrapper imports its detector, so these
module-level imports made the whole agent layer un-importable without pandera/river.
`pytest -x` masked the river failure behind the earlier pandera collection error (river would
have surfaced at runtime of the parametrized annotation_policy awaiting_baseline case).
scipy (label_shift) was NOT a problem -- it is declared (requirements.txt:108).

## Fix (commit 92ff4a2)
- schema_drift_agent.py: pandera import moved into detect() (lazy). The field/annotation type is
  a string under `from __future__ import annotations`, so the module imports without pandera.
- annotation_policy_agent.py: `from river import drift` guarded with try/except ModuleNotFoundError
  -> river_drift = None. Module imports without river; active detection (not yet wired) raises loudly.
- test_schema_drift_monitor_agent.py: hard import replaced with pytest.importorskip("pandera.pandas"),
  matching the repo convention (test_ablate_gnn.py uses importorskip for torch_geometric).
- No requirements changed. The schema/annotation ok-path detection tests SKIP in CI (libs absent)
  and RUN locally (libs present); the awaiting_baseline/registration paths run in both.

## Verification
- Local CI reproduction (scripts/simulate_ci_no_optional_deps.py): blocks pandera+river in-process;
  orchestrator imports, affected tests pass/skip (8 passed, 1 skipped), exit 0.
- Local full suite (libs present): 873 passed / 6 skipped, unchanged.
- CI #304 (92ff4a2): Success -- lockfile-check green, pytest 3.11 (4m 0s), pytest 3.12 (4m 23s),
  docker smoke green, coverage-report artifact produced.

## Lesson / prevention
- "Full test suite green" in the standing run-gates must mean CI green, not just local-venv green.
  A dependency present locally but absent in CI is invisible to a local run.
- A new module-level third-party import in src/ that is not in requirements is a CI regression even
  when local tests pass.
- Prevention now in-repo: scripts/simulate_ci_no_optional_deps.py reproduces the lib-absent
  environment in-process. Run it (or a clean venv installing only the CI requirement set) before
  declaring a suite sealed whenever agent-layer / optional-dep code changed.
- Optional follow-up: declare pandera + river in requirements-dev.txt so CI RUNS (not skips) the
  schema/annotation ok-path tests -- pending confirmation that requirements-dev.txt is not pip-compiled.
