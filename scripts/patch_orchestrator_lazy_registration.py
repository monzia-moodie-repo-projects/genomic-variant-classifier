#!/usr/bin/env python3
"""
patch_orchestrator_lazy_registration.py -- convert Orchestrator._register_agents from EAGER
(imports all 22 agent modules at construction) to LAZY (each agent module imports only when its
agent is first instantiated in run_pipeline).

WHY: constructing the Orchestrator for ANY pipeline eagerly imports every agent module. One of them
(ModelInsightsAgent -> evaluation.evaluator -> sklearn.calibration) needs sklearn, which the
deliberately-minimal Data Freshness Monitor CI does not install -> ModuleNotFoundError, red CI. The
freshness dry-run only needs DatabaseFreshnessMonitorAgent. Lazy registration imports each agent
module on first instantiation, so the orchestrator constructs with zero agent-module imports.

INVARIANTS PRESERVED (verified in sandbox + by post-checks):
  - All 22 string KEYS stay in a dict literal assigned to self._agent_registry (check_agents_active.py
    parses these keys via AST -> still counts 22/22).
  - PIPELINE_DEFINITIONS untouched -> scheduling byte-identical.
  - Every agent stays registered, scheduled, and behaviorally identical; only import TIMING changes.

TWO EDITS (both required, applied together, written once):
  1. Replace the eager-import block + class-valued registry dict with a lazy-factory dict (values are
     zero-arg factories returning the CLASS).
  2. Update the sole runtime read (run_pipeline: `agent_cls = self._agent_registry.get(agent_name)`)
     to CALL the factory so it resolves to the class before instantiation.

SEMANTIC NOTE (documented in CHANGELOG): a broken/missing agent import now raises at instantiation
(when its pipeline runs) rather than at orchestrator construction. More correct (pay for what you
use); check_agents_active.py still flags dormancy via telemetry.

Anchored, idempotent (sentinel), .bak backup, ast.parse syntax-guard with rollback, abort-on-mismatch.
"""
from __future__ import annotations
import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/orchestrator.py")
SENTINEL = "# LAZY agent registration (import-on-first-instantiation)"

ANCHOR_START = "        from genomic_variant_classifier.agent_layer.agents.data_freshness_agent import DataFreshnessAgent"
ANCHOR_LAST_ENTRY = '            "ProvisioningAgent": ProvisioningAgent,'
ANCHOR_END = "        }"

CALLSITE_OLD = "            agent_cls = self._agent_registry.get(agent_name)"
CALLSITE_NEW = (
    "            _agent_factory = self._agent_registry.get(agent_name)\n"
    "            agent_cls = _agent_factory() if _agent_factory is not None else None"
)
CALLSITE_SENTINEL = "_agent_factory = self._agent_registry.get(agent_name)"

AGENTS = [
    ("DataFreshnessAgent", "data_freshness_agent", "DataFreshnessAgent"),
    ("TrainingLifecycleAgent", "training_lifecycle_agent", "TrainingLifecycleAgent"),
    ("InterpretabilityAgent", "interpretability_agent", "InterpretabilityAgent"),
    ("LiteratureScoutAgent", "literature_scout_agent", "LiteratureScoutAgent"),
    ("VersionMonitorAgent", "version_monitor_agent", "VersionMonitorAgent"),
    ("ConceptDriftMonitorAgent", "concept_drift_monitor_agent", "ConceptDriftMonitorAgent"),
    ("LabelShiftMonitorAgent", "label_shift_monitor_agent", "LabelShiftMonitorAgent"),
    ("CalibrationDriftMonitorAgent", "calibration_drift_monitor_agent", "CalibrationDriftMonitorAgent"),
    ("InfrastructureDriftMonitorAgent", "infrastructure_drift_monitor_agent", "InfrastructureDriftMonitorAgent"),
    ("FairnessSubgroupMonitorAgent", "fairness_subgroup_monitor_agent", "FairnessSubgroupMonitorAgent"),
    ("AdversarialSubmissionMonitorAgent", "adversarial_submission_monitor_agent", "AdversarialSubmissionMonitorAgent"),
    ("AnnotationPolicyMonitorAgent", "annotation_policy_monitor_agent", "AnnotationPolicyMonitorAgent"),
    ("SchemaDriftMonitorAgent", "schema_drift_monitor_agent", "SchemaDriftMonitorAgent"),
    ("FeatureCoverageSentinelMonitorAgent", "feature_coverage_sentinel_monitor_agent", "FeatureCoverageSentinelMonitorAgent"),
    ("ReclassificationSentinelMonitorAgent", "reclassification_sentinel_monitor_agent", "ReclassificationSentinelMonitorAgent"),
    ("DatabaseFreshnessMonitorAgent", "database_freshness_monitor_agent", "DatabaseFreshnessMonitorAgent"),
    ("ModelInsightsAgent", "model_insights_agent", "ModelInsightsAgent"),
    ("DataReadinessAgent", "data_readiness_agent", "DataReadinessAgent"),
    ("AgentOpsMonitorAgent", "agent_ops_monitor_agent", "AgentOpsMonitorAgent"),
    ("FinOpsAdvisorAgent", "finops_advisor_agent", "FinOpsAdvisorAgent"),
    ("AdaptationAgent", "adaptation_agent", "AdaptationAgent"),
    ("ProvisioningAgent", "provisioning_agent", "ProvisioningAgent"),
]
_PKG = "genomic_variant_classifier.agent_layer.agents"


def build_lazy_block() -> str:
    L = []
    L.append("        " + SENTINEL)
    L.append("        # Values are zero-arg factories: each imports its agent module on first call")
    L.append("        # (in run_pipeline), so constructing the Orchestrator imports NO agent module.")
    L.append("        # Keys are unchanged (check_agents_active.py parses them via AST; all 22 stay).")
    L.append("        def _lazy(modname: str, clsname: str):")
    L.append("            def _factory():")
    L.append("                import importlib")
    L.append("                mod = importlib.import_module(modname)")
    L.append("                return getattr(mod, clsname)")
    L.append("            return _factory")
    L.append("")
    L.append("        self._agent_registry = {")
    for name, mod, cls in AGENTS:
        L.append(f'            "{name}": _lazy("{_PKG}.{mod}", "{cls}"),')
    L.append("        }")
    return "\n".join(L)


def main() -> int:
    if not TARGET.exists():
        print(f"[FAIL] {TARGET} not found (run from repo root)")
        return 2
    text = TARGET.read_text(encoding="utf-8")

    already_reg = SENTINEL in text
    already_call = CALLSITE_SENTINEL in text
    if already_reg and already_call:
        print("[idempotent] orchestrator already lazy (both edits present); no change.")
        return 0
    if already_reg != already_call:
        print(f"[FAIL] partial prior patch detected (registry={already_reg}, callsite={already_call}). "
              f"Manual review needed; aborting.")
        return 10

    # --- locate + validate the registry block ---
    if text.count(ANCHOR_START) != 1:
        print(f"[FAIL] ANCHOR_START not found exactly once ({text.count(ANCHOR_START)}).")
        return 3
    if text.count(ANCHOR_LAST_ENTRY) != 1:
        print(f"[FAIL] ProvisioningAgent entry not found exactly once ({text.count(ANCHOR_LAST_ENTRY)}).")
        return 4
    if text.count(CALLSITE_OLD) != 1:
        print(f"[FAIL] call site not found exactly once ({text.count(CALLSITE_OLD)}).")
        return 5

    start_idx = text.index(ANCHOR_START)
    last_entry_idx = text.index(ANCHOR_LAST_ENTRY)
    brace_idx = text.index("\n" + ANCHOR_END, last_entry_idx + len(ANCHOR_LAST_ENTRY))
    end_idx = brace_idx + 1 + len(ANCHOR_END)

    old_block = text[start_idx:end_idx]
    if not old_block.startswith(ANCHOR_START) or not old_block.rstrip().endswith("}"):
        print("[FAIL] computed block boundaries look wrong; aborting.")
        return 6
    miss = [n for n, _, _ in AGENTS if f'"{n}"' not in old_block]
    if miss:
        print(f"[FAIL] block missing expected keys: {miss[:3]}...; aborting.")
        return 7

    # --- build new_text with BOTH edits, then write ONCE ---
    new_text = text[:start_idx] + build_lazy_block() + text[end_idx:]
    if new_text.count(CALLSITE_OLD) != 1:
        print(f"[FAIL] after registry edit, call site count != 1 ({new_text.count(CALLSITE_OLD)}); aborting.")
        return 8
    new_text = new_text.replace(CALLSITE_OLD, CALLSITE_NEW)

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(new_text, encoding="utf-8")

    # --- syntax guard ---
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        shutil.copy2(bak, TARGET)
        print(f"[FAIL] post-patch syntax error ({e}); restored from .bak.")
        return 9

    after = TARGET.read_text(encoding="utf-8")
    sentinel_ok = SENTINEL in after
    keys_ok = all(f'"{n}"' in after for n, _, _ in AGENTS)
    no_eager = "agents.data_freshness_agent import DataFreshnessAgent" not in after
    callsite_ok = CALLSITE_SENTINEL in after
    ok = sentinel_ok and keys_ok and no_eager and callsite_ok
    print(f"[{'ok' if ok else 'FAIL'}] sentinel={sentinel_ok} all22keys={keys_ok} "
          f"eager_removed={no_eager} callsite_updated={callsite_ok}")
    print(f"[ok] backup at {bak} (remove before committing)")
    return 0 if ok else 11


if __name__ == "__main__":
    sys.exit(main())
