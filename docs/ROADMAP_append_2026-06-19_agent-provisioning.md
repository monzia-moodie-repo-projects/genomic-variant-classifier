# ROADMAP append — 2026-06-19 (agent layer liveness + provisioning)

Fold into `ROADMAP.md` / `ROADMAP.docx`. Continues `ROADMAP_append_2026-06-11*.md`.

## Agent layer — status change

- **De-dormanted.** The 21-agent layer had only ever been dry-run; a real `run_agents.py --pipeline all` now stamps `agent_runs` telemetry → 21/21 `ACTIVE`. ProvisioningAgent pending its first real run (gated last night by an unresolved review item).
- **Liveness gate is now a hard preflight prerequisite.** `scripts/check_agents_active.py --strict` must exit 0 before any GPU run; it parses the orchestrator (registry + `PIPELINE_DEFINITIONS`) and the real SharedState, and hard-fails on `NEVER_RUN` / `UNSCHEDULED` / `MISSING_IMPL` / `ERRORED` / `DRY_RUN_ONLY` / `SECTION_ONLY` (and `STALE` under `--strict`).
- **Run-NN preflight prerequisite (new, standing):** (a) `python src\genomic_variant_classifier\agent_layer\run_agents.py --pipeline all` (real, non-dry-run, **from the repo root**) — note this re-adds the AdaptationAgent version-candidate review item every run; (b) `check_agents_active.py --strict` green; (c) triage/resolve review items (`--reviews` / `--resolve INDEX`) so `run_pipeline`'s gate does not abort the launch sequence. Bake into `new_run_preflight.py`.

## ProvisioningAgent (NEW — separate from recommend-only FinOps)

- **Built + wired this session (select-only).** Registered in the orchestrator and scheduled in a new `provision` pipeline (22 agents total). Enforces the budget cap now; **no live spend.**
- Components: `agent_layer/provisioning/offer_schema.py` (provider-agnostic Vast+RunPod), `agent_layer/provisioning/provisioning_docs.py` (`provisioning` section + run-doc writer documenting every provisioning detail), `agents/provisioning_agent.py`, `scripts/apply_provisioning_wiring.py` (idempotent, self-verifying, rollback).
- **Next increment (gated):** live Vast/RunPod create/destroy behind HITL-approve (on provision) + confirm-on-terminate (on teardown) + budget cap. Mirrors the option A-GATED design in `docs/design/GPU_FINOPS_DESIGN.md`. FinOpsAdvisorAgent stays strictly recommend-only.

## Run 17 — now GATED on data readiness

- **BLOCKER:** `DataReadinessAgent` NO_GO — 11 ACTIVE `critical_assets()` missing/empty locally. Partly a path problem (registry `finngen` path is a documented FILENAME TYPO 'finnge'; `spliceai`/`alphamissense` have `data/processed/` dups; `dbsnp`/`string` have cache artifacts). `scripts/audit_run17_assets.py` separates `FOUND_AT_ALT` (registry path fix) from genuine `MISSING` (re-acquire/re-prep). **Run 17 must not launch until DataReadiness returns GO** (override is recorded but is not remediation).

## Bug backlog (opened this session)

DataReadinessAgent cwd-relative root → false NO_GO (FIXED, anchored to PROJECT_ROOT); LiteratureScout candidate extractor emits stopwords; `version_monitor.last_run="t"`; two version stores disagree (3 vs 99); missing deps `feedparser` + `ewc_utils` (silent drift downgrade); ClinGen 404; ClinVar `clinvar_papu.vcf.gz` FTP-parse anomaly; drift suite (9) `awaiting_baseline`; schema-hash definition mismatch (`6b428d5d…` vs `efca0d85a28d`).
