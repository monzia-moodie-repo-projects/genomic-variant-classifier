#!/usr/bin/env python3
"""apply_agent_root_anchor_fix.py -- Author: Monzia Moodie

Generalise apply_data_readiness_root_fix.py to the FIVE agents that still carry
the defect it repaired.

SCOPE WAS MEASURED, AND MY FIRST COUNT WAS WRONG. I searched the five agents I
already had in hand and found four defective. A search across ALL of src and
scripts found a fifth -- provisioning_agent.py:45, which writes via
PD.write_provisioning_doc(self._root, event) at lines 104 and 123. Shipping the
four-agent version would have looked complete and left one agent defective,
which is worse than not shipping.

THE DEFECT, AS THE EARLIER SCRIPT DIAGNOSED IT
    "it defaulted root="." so it resolved registry.critical_assets()
     (repo-relative paths) against the CURRENT WORKING DIRECTORY. When the
     orchestrator is launched from src/.../agent_layer ... cwd was the
     agent_layer dir and every asset read as missing -> spurious NO_GO, even
     though the data is present at the repo root."

That is not hypothetical here. Three generated reports sit inside the source
tree today, at

    src/genomic_variant_classifier/agent_layer/reports/agent_ops/OPS_2026-06-20.md
    src/genomic_variant_classifier/agent_layer/reports/data_freshness/FRESHNESS_2026-06-20.md
    src/genomic_variant_classifier/agent_layer/reports/data_readiness/READINESS_2026-06-20.md

written by exactly this mechanism: the orchestrator was launched from the
package directory and every root="." agent resolved against it.

MEASURED 2026-08-14 -- the five still defective, and the one already repaired:

    agent_ops_monitor_agent.py:28          root: str = "."     _root == "."
    database_freshness_monitor_agent.py:26 root: str = "."     _root == "."
    finops_advisor_agent.py:27             root: str = "."     _root == "."
    model_insights_agent.py:39             root: str = "."     _root == "."
    provisioning_agent.py:45               root: str = "."     _root == "."
    data_readiness_agent.py:38             anchored            _root == PROJECT_ROOT

The right-hand column is CONSTRUCTED state, not source text: each agent was
instantiated with a stub shared_state and its _root read directly.

DELIBERATELY OUT OF SCOPE -- four function defaults that are NOT defects:

    database_freshness_detector.py:96  check_local(..., root: str = ".")
    database_freshness_detector.py:115 scan(..., root: str = ".")
    data_readiness_detector.py:40      check_assets(..., root: str = ".")
    data_readiness_detector.py:94      analyze(..., root: str = ".")

These are pure functions taking root as a parameter, which is the correct
design, and every measured caller passes it explicitly -- the monitor agent at
line 34, check_local's own caller at 124, and four test call sites. Anchoring
them to PROJECT_ROOT would import agent-layer configuration into the evaluation
layer: a dependency inversion, and a worse defect than the one it fixes.

WHY PROJECT_ROOT AND NOT SOMETHING ELSE
The earlier script names the precedent: PROJECT_ROOT is "the SAME anchor
InterpretabilityAgent already uses via CHECKPOINT_DIR = PROJECT_ROOT /
'models'". Introducing a third convention would be the parallel-vocabulary
failure this repository keeps eliminating.

    RECORDED, NOT REPAIRED HERE: config.py:17 reads
        PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT", r"C:\\Projects\\..."))
    and GVC_PROJECT_ROOT is set NOWHERE -- not in continuous integration, not
    in the Dockerfile, not in any script. So the fallback is a Windows literal
    that cannot exist on the Linux runner. That is PROJECT-ROOT-HARDCODED-1, a
    separate and larger finding. This unit does not widen it: it makes five
    agents consistent with the one that was already repaired, and every
    existing test injects root explicitly so none depends on the default.

ROOT STAYS INJECTABLE. Only the default changes. All eight measured test call
sites pass root=str(tmp_path), which is why the default is currently
unexercised -- and why this ships with a test that exercises it.

Idempotent, ast-verifies before AND after writing, backs up to
.pre_rootanchor.bak, and rolls back on any failure. Anchors are per-agent and
byte-exact, because the five signatures differ: finops spans three lines,
provisioning is keyword-only with nine parameters, and
model_insights carries outputs_root on an adjacent line that also contains the
substring "root".

Usage:  python scripts/apply_agent_root_anchor_fix.py --repo-root . --check
        python scripts/apply_agent_root_anchor_fix.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

BASE = Path("src/genomic_variant_classifier/agent_layer/agents")

#: The import line each agent already has, after which PROJECT_ROOT is added.
#: Measured per file from the parse tree -- they are NOT uniform.
IMPORT_INSERT = "from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT\n"

_MARKER = "else str(PROJECT_ROOT)"

AGENTS = {
    "agent_ops_monitor_agent.py": {
        "import_anchor":
            "from genomic_variant_classifier.evaluation import agent_ops_detector as D\n",
        "sig_old":
            '    def __init__(self, shared_state, stale_after_hours: float = D.DEFAULT_STALE_HOURS, root: str = ".") -> None:\n',
        "sig_new":
            "    def __init__(self, shared_state, stale_after_hours: float = D.DEFAULT_STALE_HOURS,\n"
            "                 root: str | None = None) -> None:\n",
        "body_old": "        self._root = root\n",
        "body_new": "        self._root = root if root is not None else str(PROJECT_ROOT)\n",
    },
    "database_freshness_monitor_agent.py": {
        "import_anchor":
            "from genomic_variant_classifier.monitoring import registry as R\n",
        "sig_old":
            '    def __init__(self, shared_state, probe=None, root: str = ".") -> None:\n',
        "sig_new":
            "    def __init__(self, shared_state, probe=None, root: str | None = None) -> None:\n",
        "body_old": "        self._root = root\n",
        "body_new": "        self._root = root if root is not None else str(PROJECT_ROOT)\n",
    },
    "finops_advisor_agent.py": {
        "import_anchor":
            "from genomic_variant_classifier.evaluation import finops_detector as D\n",
        "sig_old": '                 root: str = ".") -> None:\n',
        "sig_new": "                 root: str | None = None) -> None:\n",
        "body_old": "        self._root = root\n",
        "body_new": "        self._root = root if root is not None else str(PROJECT_ROOT)\n",
    },
    "provisioning_agent.py": {
        # Keyword-only after shared_state, nine parameters. Anchors are the
        # single `root` line and the assignment, both byte-exact -- line 45 is
        # indented eight spaces and ends with a comma, unlike the other four.
        "import_anchor":
            "from genomic_variant_classifier.agent_layer.provisioning import provisioning_docs as PD\n",
        "sig_old": '        root: str = ".",\n',
        "sig_new": "        root: str | None = None,\n",
        "body_old": "        self._root = root\n",
        "body_new": "        self._root = root if root is not None else str(PROJECT_ROOT)\n",
    },
    "model_insights_agent.py": {
        "import_anchor":
            "from genomic_variant_classifier.evaluation import model_insights_detector as D\n",
        "sig_old":
            '    def __init__(self, shared_state, outputs_root: str = "outputs", root: str = ".") -> None:\n',
        "sig_new":
            '    def __init__(self, shared_state, outputs_root: str = "outputs",\n'
            "                 root: str | None = None) -> None:\n",
        # NOTE: line 41 is `self._outputs_root = outputs_root`, which also
        # contains "root". The anchor below is byte-exact and does not match it.
        "body_old": "        self._root = root\n",
        "body_new": "        self._root = root if root is not None else str(PROJECT_ROOT)\n",
    },
}


def _verify(source: str, name: str) -> tuple:
    """The four properties the earlier script checks, plus one for this unit."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)
    if IMPORT_INSERT.strip() not in source:
        return False, "PROJECT_ROOT import missing"
    if _MARKER not in source:
        return False, "root default not anchored to PROJECT_ROOT"
    if 'root: str = "."' in source:
        return False, 'old root="." default still present'
    # This unit's addition: the injectable parameter must survive, or the fix
    # would have removed the hermetic-test seam the docstrings promise.
    if "root" not in source:
        return False, "root parameter vanished"
    return True, "root anchored to PROJECT_ROOT; injectable root preserved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root)
    plans = []
    already = 0

    for name, spec in sorted(AGENTS.items()):
        path = root / BASE / name
        if not path.exists():
            print("ERROR: not found: {}".format(path))
            return 2
        src = path.read_text(encoding="utf-8")

        if _MARKER in src:
            ok, msg = _verify(src, name)
            print("  {:<38} already patched -- {}: {}".format(
                name, "OK" if ok else "PROBLEM", msg))
            if not ok:
                return 1
            already += 1
            continue

        counts = {k: src.count(spec[k]) for k in ("import_anchor", "sig_old", "body_old")}
        bad = {k: v for k, v in counts.items() if v != 1}
        if bad:
            print("  {:<38} ERROR: anchor count(s) {} (expected 1 each); "
                  "aborting, NO changes to ANY file.".format(name, bad))
            return 1
        print("  {:<38} anchors OK  {}".format(name, counts))
        plans.append((path, src, spec, name))

    if args.check:
        print("\n  --check: {} pending, {} already patched. Nothing written.".format(
            len(plans), already))
        return 0

    if not plans:
        # Derived from AGENTS, not a literal. The earlier text said "All four"
        # after the fifth agent was added -- a hard-coded count that had gone
        # stale, which is the defect class this repository treats as real.
        print("\n  All {} agent(s) already anchored. Nothing to do.".format(len(AGENTS)))
        return 0

    # Patch and verify EVERY file in memory before writing ANY of them.
    patched = []
    for path, src, spec, name in plans:
        new = (src
               .replace(spec["import_anchor"],
                        spec["import_anchor"] + IMPORT_INSERT, 1)
               .replace(spec["sig_old"], spec["sig_new"], 1)
               .replace(spec["body_old"], spec["body_new"], 1))
        ok, msg = _verify(new, name)
        if not ok:
            print("  ERROR: {} failed verification BEFORE writing ({}); "
                  "no changes to any file.".format(name, msg))
            return 1
        patched.append((path, new, name))

    written = []
    for path, new, name in patched:
        backup = path.with_suffix(".py.pre_rootanchor.bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_bytes(new.encode("utf-8"))
        written.append((path, backup, name))
        print("  wrote {}".format(name))

    for path, backup, name in written:
        ok, msg = _verify(path.read_text(encoding="utf-8"), name)
        if not ok:
            for p2, b2, _ in written:
                p2.write_bytes(b2.read_bytes())
            print("  ERROR: {} failed POST-WRITE verification ({}); "
                  "ROLLED BACK all {} file(s).".format(name, msg, len(written)))
            return 1

    print("\n  {} agent(s) anchored to PROJECT_ROOT; {} already were.".format(
        len(written), already))
    return 0


if __name__ == "__main__":
    sys.exit(main())
