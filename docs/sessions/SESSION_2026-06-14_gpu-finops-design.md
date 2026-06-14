# Session 2026-06-14 -- GpuOrchestrator/FinOps design review (proposed agent #3)

## Context
The low-risk agent backlog is complete (ModelInsights, DataReadiness, AgentOps + agent_runs telemetry, all green).
The only remaining proposed agent is GpuOrchestratorAgent/FinOps (#3) -- the single HIGH-risk one (paid infra).
Per the standing plan it is NOT one-shotted; this session is the design-review pass.

## Grounding (actual repo state, not assumed)
- launch_run16.py: pick_offer (pure, tested in test_preflight_run16_inputs.py) selects cheapest single-GPU 4090 by
  (dph_total, -reliability2, -cpu_ram); SEARCH_QUERY criteria; real `vastai create` (spends); `--dry` preview.
- Cost model: hours * hourly_rate (run14_observability.py). Auto-destroy on failure (launch_run*_vm.sh);
  confirm-on-terminate (Vastai_Destroy_Confirmed.ps1). preflight_gate.py --emit = the HITL recommended-command pattern.
- TrainingLifecycleAgent does NOT provision (remote launch retired, INCIDENT_2026-04-29). RunPod: net-new, docs-only.

## Output
docs/design/GPU_FINOPS_DESIGN.md -- current state, gap analysis, the pivotal decision (autonomous vs recommend-only),
recommendation (recommend-only first), an exact first-slice spec (extract pick_offer to a package lib;
finops_detector + finops_advisor_agent; wiring; tests; explicit non-goals), a phased plan, and the cost-safety
guardrails. No code written.

## Open decision
Confirm recommend-only advisor as the first slice (zero spend) or request autonomous provisioning (gated). Build
follows confirmation.
