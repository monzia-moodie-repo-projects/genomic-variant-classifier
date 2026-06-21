#!/usr/bin/env python3
"""apply_provisioning_wiring.py -- Author: Monzia Moodie

Idempotently wire ProvisioningAgent into the orchestrator so the agent-liveness
checker counts it (registered + scheduled, not UNSCHEDULED). Inserts three lines
via unique anchors -- the import, the _agent_registry entry, and a new `provision`
pipeline (the auto-computed PIPELINE_DEFINITIONS["all"] union picks it up for free):

  1. from genomic_variant_classifier.agent_layer.agents.provisioning_agent import ProvisioningAgent
  2. "ProvisioningAgent": ProvisioningAgent,            (inside self._agent_registry)
  3. "provision": ["ProvisioningAgent"],                (inside PIPELINE_DEFINITIONS)

Safety: backs up orchestrator.py -> orchestrator.py.prewiring.bak, ast-parses the
result, verifies ProvisioningAgent is registered AND scheduled in 'provision', and
ROLLS BACK on any failure. Re-running is a no-op once wired.

Usage:
  python scripts/apply_provisioning_wiring.py --repo-root .
  python scripts/apply_provisioning_wiring.py --repo-root . --check   # report only, no write
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

IMPORT_ANCHOR = ("        from genomic_variant_classifier.agent_layer.agents."
                 "finops_advisor_agent import FinOpsAdvisorAgent\n")
IMPORT_INSERT = ("        from genomic_variant_classifier.agent_layer.agents."
                 "provisioning_agent import ProvisioningAgent\n")

REGISTRY_ANCHOR = '            "AdaptationAgent": AdaptationAgent,\n'
REGISTRY_INSERT = '            "ProvisioningAgent": ProvisioningAgent,\n'

PIPELINE_ANCHOR = '    "finops": ["FinOpsAdvisorAgent"],\n'
PIPELINE_INSERT = '    "provision": ["ProvisioningAgent"],\n'


def _verify(source: str) -> tuple[bool, str]:
    """Confirm ProvisioningAgent is registered AND scheduled in 'provision'."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error after patch: {exc}"
    registered, pipelines = set(), {}
    for node in ast.walk(tree):
        # _agent_registry = { "Name": Name, ... }  (Assign with Attribute target)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == "_agent_registry"
                        and isinstance(node.value, ast.Dict)):
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            registered.add(k.value)
        # PIPELINE_DEFINITIONS = { ... }  -- may be annotated (AnnAssign) or plain (Assign)
        tnames = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tnames = [node.target.id]
        elif isinstance(node, ast.Assign):
            tnames = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PIPELINE_DEFINITIONS" in tnames and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.List):
                    pipelines[k.value] = [e.value for e in v.elts
                                          if isinstance(e, ast.Constant)]
    if "ProvisioningAgent" not in registered:
        return False, "ProvisioningAgent not in _agent_registry after patch"
    if "ProvisioningAgent" not in pipelines.get("provision", []):
        return False, "ProvisioningAgent not in 'provision' pipeline after patch"
    return True, f"registered ({len(registered)} agents) + scheduled in 'provision'"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true", help="report only; do not write")
    args = ap.parse_args(argv)

    orch = (Path(args.repo_root) / "src" / "genomic_variant_classifier"
            / "agent_layer" / "orchestrator.py")
    if not orch.exists():
        print(f"ERROR: orchestrator not found: {orch}")
        return 2
    src = orch.read_text(encoding="utf-8")

    if "ProvisioningAgent" in src:
        ok, msg = _verify(src)
        print(f"Already wired -- {'OK' if ok else 'PROBLEM'}: {msg}")
        return 0 if ok else 1

    if args.check:
        print("Not wired yet. Anchors present:",
              {"import": IMPORT_ANCHOR.strip()[:40] in src or IMPORT_ANCHOR in src,
               "registry": REGISTRY_ANCHOR in src, "pipeline": PIPELINE_ANCHOR in src})
        return 0

    # Each anchor must be present exactly once (loud failure otherwise).
    for label, anchor in (("import", IMPORT_ANCHOR), ("registry", REGISTRY_ANCHOR),
                          ("pipeline", PIPELINE_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            print(f"ERROR: {label} anchor found {n} times (expected 1); aborting, no changes made.")
            return 1

    patched = (src
               .replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_INSERT, 1)
               .replace(REGISTRY_ANCHOR, REGISTRY_ANCHOR + REGISTRY_INSERT, 1)
               .replace(PIPELINE_ANCHOR, PIPELINE_ANCHOR + PIPELINE_INSERT, 1))

    ok, msg = _verify(patched)
    if not ok:
        print(f"ERROR: verification failed BEFORE writing ({msg}); no changes made.")
        return 1

    backup = orch.with_suffix(".py.prewiring.bak")
    if not backup.exists():
        backup.write_bytes(orch.read_bytes())
    orch.write_bytes(patched.encode("utf-8"))

    # Re-read from disk and verify the written file (defence in depth).
    ok2, msg2 = _verify(orch.read_text(encoding="utf-8"))
    if not ok2:
        orch.write_bytes(backup.read_bytes())   # rollback
        print(f"ERROR: post-write verification failed ({msg2}); ROLLED BACK from {backup.name}.")
        return 1

    print(f"OK: ProvisioningAgent wired -- {msg2}. Backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
