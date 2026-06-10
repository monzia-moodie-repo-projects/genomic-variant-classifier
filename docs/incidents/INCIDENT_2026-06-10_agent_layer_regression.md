# INCIDENT 2026-06-10 -- Agent-layer regression: 8 of 12 agents orphaned; VersionMonitor classless

**Status:** RESOLVED (agent re-wiring) -- 2026-06-10. Residuals tracked in the Resolution section.
**Severity:** Medium -- no production-inference impact; monitoring/agentic layer was degraded and the README/roadmap overstated operational scope.
**Discovered:** 2026-06-10, during a README accuracy pass.
**Author:** Monzia Moodie
**Tooling (committed):** `scripts/audit_agent_roster.py`, `scripts/audit_agent_operational.py`, `scripts/patch_register_drift_agents.py`.

## Summary

A measurement-based audit found that, of the classes the README presented as a "thirteen-agent monitoring layer," only **four were operational**. Eight drift/specialised agents were implemented and importable but **orphaned** (no `BaseAgent`, no `run()`, unregistered, referenced by nothing). `agents/version_monitor_agent.py` exists and compiles but **defines no class**. Root cause: the April->May decomposition of the monolithic `DriftMonitorAgent` into eight detectors plus the C1 migration (`c0a4dba`) orchestrator rewrite carried the four framework agents into the new dict registry but not the eight detectors or `VersionMonitorAgent`.

## Pre-resolution ground truth

- OPERATIONAL (4): `DataFreshnessAgent`, `TrainingLifecycleAgent`, `InterpretabilityAgent`, `LiteratureScoutAgent`.
- ORPHANED (8): `SchemaDriftAgent`, `ConceptDriftAgent`, `LabelShiftAgent`, `CalibrationDriftAgent`, `InfrastructureDriftAgent`, `FairnessSubgroupAgent`, `AdversarialSubmissionAgent`, `AnnotationPolicyAgent` -- importable `@dataclass` detectors with `detect()`/`persist()`.
- `version_monitor_agent.py`: classless remnant; no `VersionMonitorAgent` anywhere.

## Resolution (2026-06-10)

The eight detectors were re-wired onto the `BaseAgent` contract via a shared adapter, leaving the detector logic untouched:

- `agents/drift_monitor_base.py` -- `DriftMonitorBase(BaseAgent)`: common `run()` / `SharedState` / `dry_run` scaffolding; reports `status="awaiting_baseline"` (loud, never a false green) until a detector + inputs are supplied.
- Eight thin subclasses `agents/*_monitor_agent.py` -- each constructs/loads its detector, calls the existing `detect()`/`persist()`, and surfaces that detector's `...Result` fields via `_summarize()`. The original detector dataclasses are unchanged and are now **COMPOSED** (referenced by their wrappers), not orphaned.
- Registered in `Orchestrator._register_agents()` via `scripts/patch_register_drift_agents.py` (count-guarded, idempotent, backup-first, `py_compile`-gated).
- Tests: `tests/unit/test_drift_monitor_agents.py` (8 parametrised `awaiting_baseline` smoke tests) + `tests/unit/test_schema_drift_monitor_agent.py` (ok-path). **10 passed.**
- **Acceptance gate met:** `python scripts/audit_agent_operational.py` -> `operational=12 composed=8 orphaned=0 not-registered=0 total=20`.

### Audit-tooling defects found and fixed during resolution
- `audit_agent_roster.py` v2 -- was keying classes by name in a dict, silently overwriting duplicate class names; now keyed by (file, class) and reports duplicates. Confirms 0 duplicates.
- `audit_agent_operational.py` v3 -- was checking `BaseAgent`/`run()` on direct bases only, false-negativing agents that inherit via `DriftMonitorBase`; now resolves inheritance and `run()` transitively. (Caught when the base refactor briefly dropped the count to 4 despite passing tests.)

## Residuals (tracked, not blocking)

1. **`awaiting_baseline`** -- the eight wrappers are wired/registered but report `awaiting_baseline` until their reference inputs exist (schema baseline, BBSE confusion matrix, NannyML CBPE outputs, golden set, ClinVar submission feeds). Activation is a data/config task, not agent-layer code. The May-7 note recorded `data/reference/` as absent.
2. **`version_monitor_agent.py`** -- classless remnant. Its concern (watch *upstream* releases) is distinct from `InfrastructureDriftAgent` (which diffs *installed* versions), so worth restoring rather than deleting. Repair-or-rebuild (from `401735e`) pending review of the file body.
3. **`alibi_detect`** -- uninstalled; not imported by any current detector (`InfrastructureDriftAgent` lists it only as a monitored-package string), so it blocks nothing.

## Documentation actions
- README -> "twelve agents: four active monitors + eight drift agents (wired and registered; activating as reference baselines populate)."
- ROADMAP -> add drift-agent baseline-input wiring + `version_monitor` repair as explicit backlog items referencing this incident.
