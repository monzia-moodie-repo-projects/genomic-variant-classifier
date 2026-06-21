#!/usr/bin/env python3
"""apply_data_readiness_root_fix.py -- Author: Monzia Moodie

Fix the DataReadinessAgent false-NO_GO root cause: it defaulted root="." so it
resolved registry.critical_assets() (repo-relative paths) against the CURRENT
WORKING DIRECTORY. When the orchestrator is launched from src/.../agent_layer
(as run_agents.py's bare `from orchestrator import` historically encouraged), cwd
was the agent_layer dir and every asset read as missing -> spurious NO_GO, even
though the data is present at the repo root.

This anchors the default root to the canonical PROJECT_ROOT
(agent_layer/config.py) -- the SAME anchor InterpretabilityAgent already uses via
CHECKPOINT_DIR = PROJECT_ROOT / "models" -- so the gate is correct from any cwd.
root stays injectable (tests pass an explicit root); only the default changes.

Idempotent (skips if already patched), backs up to .pre_rootfix.bak, ast-verifies
the result, and rolls back on any failure.

Usage:  python scripts/apply_data_readiness_root_fix.py --repo-root C:\\Projects\\genomic-variant-classifier
        python scripts/apply_data_readiness_root_fix.py --repo-root . --check
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REL = Path("src/genomic_variant_classifier/agent_layer/agents/data_readiness_agent.py")

IMPORT_ANCHOR = "from genomic_variant_classifier.monitoring import registry as R\n"
IMPORT_INSERT = "from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT\n"

SIG_OLD = '    def __init__(self, shared_state, root: str = ".", splits_path: str | None = None) -> None:\n'
SIG_NEW = '    def __init__(self, shared_state, root: str | None = None, splits_path: str | None = None) -> None:\n'

BODY_OLD = "        self._root = root\n"
BODY_NEW = "        self._root = root if root is not None else str(PROJECT_ROOT)\n"

_MARKER = "else str(PROJECT_ROOT)"


def _verify(source: str) -> tuple[bool, str]:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error after patch: {exc}"
    if IMPORT_INSERT.strip() not in source:
        return False, "PROJECT_ROOT import missing"
    if _MARKER not in source:
        return False, "root default not anchored to PROJECT_ROOT"
    if 'root: str = "."' in source:
        return False, 'old root="." default still present'
    return True, "root anchored to PROJECT_ROOT; injectable root preserved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    path = Path(args.repo_root) / REL
    if not path.exists():
        print(f"ERROR: not found: {path}")
        return 2
    src = path.read_text(encoding="utf-8")

    if _MARKER in src:
        ok, msg = _verify(src)
        print(f"Already patched -- {'OK' if ok else 'PROBLEM'}: {msg}")
        return 0 if ok else 1

    if args.check:
        print("Not patched. Anchors present:",
              {"import": IMPORT_ANCHOR in src, "signature": SIG_OLD in src, "body": BODY_OLD in src})
        return 0

    for label, anchor in (("import", IMPORT_ANCHOR), ("signature", SIG_OLD), ("body", BODY_OLD)):
        n = src.count(anchor)
        if n != 1:
            print(f"ERROR: {label} anchor found {n}x (expected 1); aborting, no changes.")
            return 1

    patched = (src
               .replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_INSERT, 1)
               .replace(SIG_OLD, SIG_NEW, 1)
               .replace(BODY_OLD, BODY_NEW, 1))

    ok, msg = _verify(patched)
    if not ok:
        print(f"ERROR: verification failed before writing ({msg}); no changes.")
        return 1

    backup = path.with_suffix(".py.pre_rootfix.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_bytes(patched.encode("utf-8"))

    ok2, msg2 = _verify(path.read_text(encoding="utf-8"))
    if not ok2:
        path.write_bytes(backup.read_bytes())
        print(f"ERROR: post-write verification failed ({msg2}); ROLLED BACK from {backup.name}.")
        return 1

    print(f"OK: DataReadinessAgent root anchored -- {msg2}. Backup: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
