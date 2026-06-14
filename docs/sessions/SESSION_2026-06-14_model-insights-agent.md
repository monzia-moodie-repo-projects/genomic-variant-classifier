# Session 2026-06-14 -- ModelInsightsAgent

## Context
Continuation after the bbb9d5c CI regression was repaired (7fc16a1 green: the auto-derived 'all' pipeline was
exempted from the reclassification leak-check). CI confirmed green before any new feature work began.

## Built: ModelInsightsAgent (proposed agent #1)
Read-only per-model comparison + integrity monitor. Grounded against the real artifact schema before coding:
RunArtifactWriter writes oof_predictions.parquet (variant_id, gene_symbol, fold, label, <model>_prob...,
ensemble_prob); evaluator.py computes AUROC/AUPRC/MCC/Brier; docs/METRICS.md is the glossary.

- evaluation/model_insights_detector.py -- pure sklearn analysis (per_model_metrics, integrity_flags,
  gene_disjoint_check, rank_by_balanced [MCC not AUROC], discover_latest_run, analyze).
- agent_layer/agents/model_insights_agent.py -- BaseAgent adapter; cls(shared_state); outputs_root/root
  injectable; 'skipped' contract when no run artifacts; emits FEATURE_INSTABILITY only on serious flags.
- orchestrator wiring: 'model_insights' pipeline + 'full'; auto in 'all'.

## Guardrail
Scientific integrity > metrics: diagnostics + flags only; a near-perfect AUROC is a LEAKAGE_SUSPICION (cross-ref
the n_pathogenic_in_gene gene-prevalence-memorization lesson), never a trophy; ranks by MCC.

## Verification
detector 7 + adapter 4 + wiring 3 = 14 new tests pass; full pipeline-touching surface 39 green (reclass
leak-check holds with 'all' at 18 agents); full collection 852. User ran the FULL suite before committing.

## Next
DataReadinessAgent (pre-run gate), then GpuOrchestratorAgent (+RunPod, cost-safety), then AgentOpsMonitor.
