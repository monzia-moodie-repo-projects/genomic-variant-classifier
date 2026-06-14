# Session 2026-06-14 -- DataReadinessAgent

## Context
Continuation after ModelInsightsAgent shipped green (f10902e + 8b40076). CI confirmed green before this work.

## Built: DataReadinessAgent (proposed agent #2)
Verify-only pre-run readiness gate. Grounded first: confirmed scripts/preflight_gate.py already validates the
launch COMMAND, so this agent covers the complementary DATA/ENVIRONMENT dimension and reuses existing code
(registry.critical_assets, data.feature_health.col_health) rather than reinventing checks.

- evaluation/data_readiness_detector.py -- pure: check_assets, feature_health_summary, readiness_verdict, analyze.
- agent_layer/agents/data_readiness_agent.py -- BaseAgent; cls(shared_state); root/splits_path injectable;
  defensive splits load; NO_GO -> HITL override gate; never mutates / never runs data-prep.
- orchestrator wiring: 'data_readiness' pipeline + 'full'; auto in 'all'.

## Design decision (recorded)
Verify-only (option A), not active-invocation (option B). A is as low-risk as the other monitors and honors
"no silent mutation"; B (shelling out to smoke/preflight_gate) is an optional follow-up if wanted.

## Verification
detector 6 + adapter 4 + wiring 3 = 13 new tests pass; full pipeline surface 50 green (reclass leak-check holds
with 'all' at 19 agents); collection 865. User ran the FULL suite before committing.

## Next
AgentOpsMonitor (lower risk) recommended before GpuOrchestratorAgent (highest risk, build last).
