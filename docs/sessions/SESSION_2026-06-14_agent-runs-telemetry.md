# Session 2026-06-14 -- agent_runs telemetry (AgentOps error-rate + perf-drift)

## Context
Continuation after AgentOpsMonitorAgent shipped green (13cd9f6 + 2db8215), which documented that error-rate and
perf-drift were not computable (no run telemetry). This closes that gap at the source.

## Built
- Orchestrator: _record_run_telemetry appends per-agent {ts,status,duration_ms,error} to 'agent_runs' (real runs
  only, capped 50, defensive). Grounded the run loop first: it already wraps each agent.run() in try/except, so the
  insertion is non-invasive and cannot change agent behavior.
- agent_ops_detector: scan_run_telemetry + telemetry_flags; analyze includes telemetry.
- agent_ops_monitor_agent: Run-telemetry table in the report + agents_with_errors/perf_drift_agents in state.

## Verification
orchestrator telemetry 4 + detector +3 + monitor +1 = 8 new tests; the run_pipeline tests that exercise the
modified loop (science_claw + drift, incl. a real dry_run=False run) stay green; collection 884. No test asserts
an exact set of state sections -> the new 'agent_runs' section is safe. User ran the FULL suite before committing.

## Next / last
GpuOrchestratorAgent/FinOps (#3) is the only remaining proposed agent -- highest risk (paid infra); recommend a
design review before building.
