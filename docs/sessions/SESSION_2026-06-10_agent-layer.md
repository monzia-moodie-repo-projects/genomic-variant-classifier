# SESSION 2026-06-10 -- Agent-layer audit and repair (4 -> 13 operational)

## Focus
What began as a README accuracy pass surfaced that the "agent layer" was far less wired
than documented. Measured, root-caused, and repaired it; reconciled the README.

## Discovered (by measurement, not memory)
- Only **4 of the claimed agents were operational** (BaseAgent + run() + registered + tested):
  DataFreshnessAgent, TrainingLifecycleAgent, InterpretabilityAgent, LiteratureScoutAgent.
- **8 drift agents were orphaned**: importable @dataclass detectors with detect()/persist()
  but no BaseAgent, no run(), unregistered, referenced by nothing.
- **version_monitor_agent.py was classless**: a complete module-level upstream-release monitor
  (pykan/ClinVar/AlphaMissense/torch-geometric) with a module run(), but no VersionMonitorAgent class.
- README was internally inconsistent: badge/intro/ASCII said "7", table/tree said "13".

## Root cause
The April->May decomposition of the monolithic DriftMonitorAgent into 8 specialised detectors,
plus the C1 migration (c0a4dba) orchestrator rewrite to a dict registry, carried the 4 framework
agents into the new registry but not the 8 detectors, and dropped VersionMonitorAgent's class.

## Repair
- `agents/drift_monitor_base.py` -- DriftMonitorBase(BaseAgent): shared run()/SharedState/dry_run
  scaffolding; reports status='awaiting_baseline' (loud, never a false green) until inputs exist.
- 8 thin wrappers `agents/*_monitor_agent.py` -- call the existing detect()/persist() unchanged;
  detectors are now COMPOSED, not orphaned.
- `agents/version_monitor_agent.py` -- added VersionMonitorAgent(BaseAgent) over the existing
  module run(); genuinely active (network release checks), not awaiting_baseline.
- Registered all 9 via scripts/patch_register_drift_agents.py.
- README reconciled 7 -> 13; stale "Py 3.14.3" -> "Python 3.12.10".

## Audit tooling (committed) -- including two of my own bugs, found and fixed
- `scripts/audit_agent_roster.py` v2 -- AST enumeration; v1 silently overwrote duplicate class
  names in a dict; v2 keys by (file, class) and reports duplicates (confirmed 0).
- `scripts/audit_agent_operational.py` v3 -- per-agent scorecard; v2 checked BaseAgent/run() on
  direct bases only and false-negatived agents inheriting via DriftMonitorBase (briefly read
  operational=4 despite green tests); v3 resolves inheritance and run() transitively.

## Verification
- pytest: 11 new agent smoke tests pass; full suite 872 passed / 6 skipped (no regressions;
  the torch_scatter/torch_sparse 0xc0000139 traces are the known importorskip skip-path, not failures).
- Gate: `python scripts/audit_agent_operational.py` -> operational=13 composed=8 orphaned=0 total=21.

## Commits (pushed to origin/main)
- 8619afc -- drift re-wiring (DriftMonitorBase + 8 wrappers + audit tooling + incident doc).
- 21e835d -- VersionMonitorAgent + README reconciliation.

## Residuals (tracked)
1. The 8 drift wrappers report awaiting_baseline until reference inputs (schema baseline, BBSE
   confusion matrix, NannyML CBPE outputs, golden set, ClinVar submission feeds) are populated --
   a data/config task, not agent-layer code. (data/reference/ recorded absent on May 7.)
2. README registry table still names the 8 detector classes rather than the *MonitorAgent
   wrappers (concern-accurate; optional precision pass).
3. Intro line still reads "thirteen specialised agents -- plus a committed drift-detection suite"
   (mildly redundant; cosmetic).
4. alibi_detect uninstalled; not imported by any current detector.

## Lessons
- Counting classes by regex or name-keyed dict fails silently; AST keyed by (file, name) with
  duplicate detection is the robust tool. Verify operational status by cross-referencing imports,
  not by naming convention.
- View-first held: several confident hypotheses this session (duplicate LiteratureScoutAgent,
  migration-broken imports, classless-by-syntax-error) were DISPROVEN by direct checks. Guessing
  would have shipped wrong fixes.
- "Operational" must distinguish wired (registered + run()) from active (has its inputs);
  awaiting_baseline makes that distinction explicit rather than papering it over.
