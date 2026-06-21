# SESSION 2026-06-19 (cont.) — Agent-layer liveness, ProvisioningAgent foundation, Run-17 readiness diagnosis

**Author:** Monzia Moodie
**Status:** files placed on-box; unit suite 34 passed; agent layer **22/22 ACTIVE** (ProvisioningAgent live after `--resolve 0` + `--pipeline provision`). **Commit pending.**
**Continues:** `SESSION_2026-06-19.md` (Run-17 launch-kit) and `RNASEQ_ABLATION_FINDINGS_2026-06-19.md`.
**Scope:** agent-layer dormancy root-cause + liveness-checker upgrade; the separate ProvisioningAgent foundation (schema/docs/agent/wiring); diagnosis of the Run-17 data-readiness NO_GO. **No training run** — no per-model algorithm comparison or new metrics-glossary entry this session.

---

## 1. Why the 21-agent layer looked dormant (root cause)

The layer had only ever been **dry-run**: every section in the real SharedState (`src/genomic_variant_classifier/agent_layer/agent_state.json`) carried `"dry_run": true`, and the orchestrator writes `agent_runs` telemetry **only on non-dry-run executions** (`if not self._dry_run: self._record_run_telemetry(...)`). No telemetry → the liveness checker (correctly, per the "active = telemetry" rule) saw no real activity. A real `run_agents.py --pipeline all` (22:30 UTC) stamped `agent_runs` for all 21 → all `ACTIVE`. Secondary cause: the checker's section fallback covered only 3/21 agents, mislabeling the dry-run section-writers `NEVER_RUN` (fixed, §2). Dry-run-only is a defensible build posture, but it left the layer never truly operated and the drift agents without baselines.

## 2. check_agents_active.py — upgraded

Authoritative signal is `agent_runs`; a section timestamp alone never reads `ACTIVE`. Complete agent→section map (18 + ProvisioningAgent; ModelInsights/DataReadiness/AgentOps are telemetry-only by design). New honest statuses: **`ERRORED`** (crashed last run, no longer silent), **`DRY_RUN_ONLY`**, **`SECTION_ONLY`** — all hard-fail; `STALE` hard-fails under `--strict`. On the real pre-run snapshot it now reports 12 `DRY_RUN_ONLY` + 4 `SECTION_ONLY` + 5 `NEVER_RUN` (the truth), not "20 NEVER_RUN".

## 3. ProvisioningAgent foundation (separate from recommend-only FinOps)

- `provisioning/offer_schema.py` — canonical offer + Vast/RunPod normalizers + budget-capped cross-provider `pick_offer` (9 tests).
- `provisioning/provisioning_docs.py` — `provisioning` SharedState section + capped history recorder + markdown run-doc writer covering every audit field the documentation rule requires (8 tests).
- `agents/provisioning_agent.py` — `ProvisioningAgent(BaseAgent)`: budget-capped selection, records section + writes doc. **Select-only; NO live spend.** Live Vast/RunPod create/destroy behind HITL-approve + confirm-on-terminate = next increment.
- `scripts/apply_provisioning_wiring.py` — idempotent, self-verifying orchestrator patcher; its verifier caught its own `AnnAssign` bug before writing. Applied on-box: **22 agents registered + scheduled**, `provision` pipeline added, `orchestrator.py.prewiring.bak` created.

## 4. Review-item gate (surprise; resolution)

`run_agents.py --pipeline provision` and `--pipeline all` both **aborted before running** — `run_pipeline` gates on unresolved review items ("Proceed anyway? [y/N]"). The item: `AdaptationAgent: 20 new version candidate(s) … (plan-only; not yet evaluated)`, added during the 22:30 real run; AdaptationAgent **never suppresses** this alert (re-adds on every real run with new candidates). Selecting N aborted both runs → `ProvisioningAgent` never ran → it is the single `NEVER_RUN` in the current report. Fixed on-box: `run_agents.py --resolve 0`, then `--pipeline provision` (excludes Adaptation) → ProvisioningAgent `ACTIVE`; liveness now **22/22, exit 0**.

## 5. Run-17 "BLOCKER" was a FALSE ALARM — data-readiness NO_GO from wrong cwd (RESOLVED)

The `--pipeline all` run reported NO_GO on 11 ACTIVE `critical_assets()`, but `audit_run17_assets.py` (run from the repo root) found **all 11 PRESENT**. Root cause: `DataReadinessAgent.__init__` defaulted `root="."`, so it resolved repo-relative asset paths against **cwd** — and the orchestrator was launched from `src/.../agent_layer/` (wrong guidance), so every asset read as missing. `run_agents.py` is designed to run **from the project root** (it `sys.path.insert`s `agent_layer/`; comment: "when run from the project root").

**Resolution:** (a) operational — launch as `python src\genomic_variant_classifier\agent_layer\run_agents.py --pipeline <p>` from the repo root; (b) hardening — `scripts/apply_data_readiness_root_fix.py` anchors the default root to the canonical `PROJECT_ROOT` (the same anchor InterpretabilityAgent uses), making the gate cwd-independent. The Run-17 data is present; DataReadiness returns GO from the correct cwd. The `audit_run17_assets.py` tool (registry-driven; 6 tests) stays as the standing GO/PATH-FIX/MISSING diagnostic.

## 6. Bug / risk inventory (from the real SharedState + run logs)

- **DataReadinessAgent cwd-relative root** → false NO_GO when launched outside the repo root (FIXED: anchored to `PROJECT_ROOT`).
- LiteratureScout extractor degenerate (stopwords as candidates: `the`/`and`/`that`/`where`/`probability`/`learning`/`variant`).
- `version_monitor.last_run = "t"` (literal string).
- Two version stores disagree: SharedState `deps_outdated_count = 3` vs `data/agent_state.json` = 99.
- Missing deps: `feedparser` (bioRxiv dead), `ewc_utils` (TrainingLifecycle drift silently → "no drift").
- ClinGen 404; LOVD 403.
- ClinVar `clinvar_papu.vcf.gz` looks malformed (FTP parse bug; matches other store's `md5("")` header hash).
- Drift suite (9 agents) `awaiting_baseline` — wired but inert until baselines exist.
- schema_drift hash `6b428d5d…` vs preflight_run17 `efca0d85a28d` — reconcile (likely different hash definitions).
- `database_freshness`: 5 changed (clinvar, alphamissense, gnomad, gnomad_constraint, esm2), 2 unreachable (alphafold, lovd).

## 7. Next steps (priority)

1. Apply `apply_data_readiness_root_fix.py`; confirm `--pipeline data_readiness` returns **GO** from any cwd; run the full unit suite.
2. Commit + push this session's work (agent liveness, provisioning, data-readiness fix) + docs.
3. Live Vast/RunPod provisioning backends (HITL-approve + confirm-on-terminate + budget cap).
4. Drift baselines; LiteratureScout extractor fix; feedparser/ewc_utils install-or-guard; ClinGen URL; reconcile version stores + `last_run="t"`.
5. (hardening) make `run_agents.py` chdir to `PROJECT_ROOT` at startup so every agent's relative outputs (FinOps/Provisioning reports) land at the repo root regardless of launch cwd.

## 8. Files added/changed (commit pending)

`scripts/check_agents_active.py` (upgraded), `scripts/apply_provisioning_wiring.py`, `scripts/audit_run17_assets.py`, `…/provisioning/{offer_schema,provisioning_docs}.py`, `…/agents/provisioning_agent.py`, `…/agent_layer/orchestrator.py` (+3 wiring lines; `.prewiring.bak`), `scripts/audit_run17_assets.py`, `scripts/apply_data_readiness_root_fix.py` (anchors `…/agents/data_readiness_agent.py` to `PROJECT_ROOT`; `.pre_rootfix.bak`), tests under `tests/unit/`. Unit suite **34 passed**.
