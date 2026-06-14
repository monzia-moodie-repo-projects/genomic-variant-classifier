# Design review: GpuOrchestratorAgent / FinOps (proposed agent #3)

Status: DESIGN REVIEW -- no money-adjacent code written yet. This is the last proposed agent and the only
HIGH-risk one (it touches PAID infrastructure). This document grounds what already exists, states the one decision
that shapes the whole design, recommends a path, and specifies an exact first slice that has ZERO spend capability.

## 1. Grounded current state (what already exists -- nothing to reinvent)

- `scripts/launch_run16.py`
  - `pick_offer(offers: list[dict]) -> dict | None` -- PURE, already unit-tested (tests/unit/test_preflight_run16_inputs.py).
    Selects the cheapest SINGLE-GPU offer by `(dph_total, -reliability2, -cpu_ram)`. Offer fields read:
    `dph_total`/`dph` (price), `reliability2`/`reliability`, `cpu_ram`, `num_gpus`.
  - `SEARCH_QUERY` -- the standing criteria: reliability > 0.99, dlperf >= 95, pcie_bw >= 12, gpu_name = RTX_4090,
    num_gpus = 1, cuda_max_good >= 12.0, disk_space >= 200, cpu_ram >= 64, rentable = true.
  - Real provisioning: `vastai create instance <id> --image ...` (SPENDS MONEY), poll, then run/stage over ssh/scp.
  - `--dry` preview mode (prints the vastai/ssh commands instead of running them).
- `scripts/launch_run*_vm.sh` -- cost-safety auto-destroy on preflight failure (`echo y | vastai destroy instance`).
- `scripts/Vastai_Destroy_Confirmed.ps1` -- confirm-on-terminate.
- `scripts/preflight_gate.py --emit` and `scripts/preflight_run16.py` -- the HITL-gated "recommended command"
  precedent: validate the launch COMMAND, the human runs it. Monzia is already comfortable with this pattern.
- `scripts/run14_observability.py` -- cost model: `cost = hours * hourly_rate` (`$/hr` = `dph_total`).
- `TrainingLifecycleAgent` -- remote launch is RETIRED (INCIDENT_2026-04-29). NO agent currently provisions or spends.

Conclusion: the optimal-selection logic (`pick_offer`), the criteria, the cost model, and the terminate-safety all
already exist as a MANUAL, human-run workflow (`launch_run16.py`, previewable with `--dry`).

## 2. Gap analysis

- RunPod: NET-NEW. Zero code footprint (only mentioned in docs). A provider-agnostic offer schema would be needed.
- No agent-layer FinOps surface: no `finops` state section, no cost advisory visible in the agent layer / AgentOps pane.
- No budget ledger: cumulative spend across runs is not tracked anywhere.
- No idle-instance leak detector (a box left running = silent cost) -- would require live `vastai show instances`.

## 3. The pivotal decision

**Does the agent ever PROVISION autonomously, or only RECOMMEND?**

- Option A -- Autonomous provisioning: the agent calls `vastai create/destroy` itself -> spends real money without a
  human in the loop. Mirrors `launch_run16.py` but agent-driven. HIGH risk; a defect, a stale offer snapshot, or a
  bad poll could leave a paid box running or spin up the wrong instance. Would require hard guardrails: budget caps,
  HITL-approve before each spend, confirm-on-terminate, idle kill-switch -- and even then the blast radius is real money.
- Option B -- Recommend-only / emit-only: the agent reads a `vastai search offers --raw` SNAPSHOT, reuses `pick_offer`,
  estimates cost, checks a budget cap, and EMITS the exact launch command for the human to run (exactly like
  `preflight_gate.py --emit`). ZERO spend, ZERO live account calls. The human stays in the loop precisely as today.

**Recommendation: Option B first.** It captures essentially all the value (optimal selection + cost estimate + one
recommended command, surfaced in the agent layer) with none of the spend risk, and it composes with the existing
manual workflow rather than replacing it. Option A stays a deliberate NON-GOAL until a separate, explicit sign-off
with the full guardrail set.

## 4. Recommended first slice -- recommend-only FinOps advisor (Option B)

Exact, mirrors the established detector + adapter + wiring + tests pattern; ZERO spend capability.

1. Extract the tested selection logic into the package (the ONLY refactor; behavior-preserving):
   - New `src/genomic_variant_classifier/infra/offer_selection.py`: move `pick_offer` + the criteria constants +
     a small `estimate_cost(hours, dph)` helper. `scripts/launch_run16.py` imports them (ONE home for the logic);
     re-point the existing `pick_offer` test to the new module (keep it green).
2. `src/genomic_variant_classifier/evaluation/finops_detector.py` (pure, unit-tested):
   - Input: an offers snapshot (list[dict] parsed from `vastai search offers --raw`), an estimated run duration
     (hours), and a budget cap (USD).
   - `analyze` -> chosen offer (via `pick_offer`), `est_cost = est_hours * dph_total`, verdict
     WITHIN_BUDGET / OVER_BUDGET / NO_SUITABLE_OFFER, and an EMITTED recommended command string (never executed).
3. `src/genomic_variant_classifier/agent_layer/agents/finops_advisor_agent.py` (BaseAgent):
   - Reads an offers snapshot FILE (path injectable). No snapshot -> "skipped" (NO live vastai call ever).
   - Records a `finops` state section (chosen offer id, est cost, verdict, recommended command) + a report.
   - OVER_BUDGET / NO_SUITABLE_OFFER -> opens a HITL review item. NEVER calls vastai, provisions, or spends.
4. Orchestrator wiring: `finops` pipeline + `full`; auto in `all`. AgentOps surfaces the new `finops` section for
   free (heartbeat); add the verdict to `scan_flags` so a non-WITHIN_BUDGET verdict shows in the ops pane.
5. Tests: detector (selection reuse, cost estimate, the three verdicts, emit-only -- asserts NO subprocess/vastai),
   adapter (snapshot present -> recommendation + state; absent -> skipped; over-budget -> HITL), wiring.

NON-GOALS for this slice (explicit, not silent): no vastai calls, no provisioning, no autonomous spend; RunPod
provider-agnostic schema deferred; budget ledger / cumulative-spend tracking deferred; idle-instance leak detector
deferred (needs live account reads).

## 5. Phased plan

- B1: recommend-only FinOps advisor (section 4). Zero spend. <- recommended next build, pending the section 3 decision.
- B2: provider-agnostic offer schema (normalize Vast.ai + RunPod offer dicts behind one shape) so `pick_offer` works
  across providers. Still recommend-only.
- B3: budget ledger -- record per-run actual spend (hours * rate) to a `finops_ledger` state section; cumulative cap.
- A (GATED, separate sign-off): autonomous provisioning behind budget caps + HITL-approve + confirm-on-terminate +
  idle kill-switch. NOT scheduled.

## 6. Cost-safety guardrails (from the roadmap, made concrete)

HITL-approve before any spend; budget caps; confirm-on-terminate (`Vastai_Destroy_Confirmed.ps1` exists); never
auto-spend; idle kill-switch. For Option B these are satisfied by construction -- there is no spend code path.

## 7. Open decision for Monzia

Confirm Option B (recommend-only advisor, section 4) as the first slice, or request Option A (autonomous, gated).
No money-adjacent code will be written until this is confirmed.
