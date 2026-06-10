#!/usr/bin/env python3
"""patch_register_drift_agents.py -- register the drift-monitor wrappers in the Orchestrator.

Count-guarded, idempotent (skips already-registered), backup-first, py_compile-gated.
Anchored on the LiteratureScoutAgent import + registry lines. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

ORCH = Path("src/genomic_variant_classifier/agent_layer/orchestrator.py")
IMPORT_ANCHOR = ("        from genomic_variant_classifier.agent_layer.agents."
                 "literature_scout_agent import LiteratureScoutAgent\n")
REGISTRY_ANCHOR = '            "LiteratureScoutAgent": LiteratureScoutAgent,\n'

AGENTS = [
    ("schema_drift_monitor_agent",          "SchemaDriftMonitorAgent"),
    ("concept_drift_monitor_agent",         "ConceptDriftMonitorAgent"),
    ("label_shift_monitor_agent",           "LabelShiftMonitorAgent"),
    ("calibration_drift_monitor_agent",     "CalibrationDriftMonitorAgent"),
    ("infrastructure_drift_monitor_agent",  "InfrastructureDriftMonitorAgent"),
    ("fairness_subgroup_monitor_agent",     "FairnessSubgroupMonitorAgent"),
    ("adversarial_submission_monitor_agent","AdversarialSubmissionMonitorAgent"),
    ("annotation_policy_monitor_agent",     "AnnotationPolicyMonitorAgent"),
]

def fail(msg: str) -> None:
    print(f"ABORT: {msg}"); sys.exit(1)

def main() -> int:
    if not ORCH.exists():
        fail(f"not found: {ORCH.resolve()}")
    txt = ORCH.read_text(encoding="utf-8")
    if txt.count(IMPORT_ANCHOR) != 1:
        fail(f"import anchor count != 1 (got {txt.count(IMPORT_ANCHOR)})")
    if txt.count(REGISTRY_ANCHOR) != 1:
        fail(f"registry anchor count != 1 (got {txt.count(REGISTRY_ANCHOR)})")

    new_imports, new_entries, skipped, added = "", "", [], []
    for module, cls in AGENTS:
        if cls in txt:
            skipped.append(cls); continue
        new_imports += (f"        from genomic_variant_classifier.agent_layer.agents."
                        f"{module} import {cls}\n")
        new_entries += f'            "{cls}": {cls},\n'
        added.append(cls)
    if not added:
        print(f"no-op: all {len(AGENTS)} agents already registered"); return 0

    patched = txt.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + new_imports, 1)
    patched = patched.replace(REGISTRY_ANCHOR, REGISTRY_ANCHOR + new_entries, 1)
    bak = ORCH.with_suffix(".py.bak")
    shutil.copy2(ORCH, bak)
    ORCH.write_text(patched, encoding="utf-8")
    try:
        py_compile.compile(str(ORCH), doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(bak, ORCH); fail(f"py_compile failed, reverted: {e}")
    print(f"registered ({len(added)}): {', '.join(added)}")
    if skipped:
        print(f"already present (skipped): {', '.join(skipped)}")
    print(f"backup: {bak}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
