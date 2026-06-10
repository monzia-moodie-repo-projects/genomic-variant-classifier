#!/usr/bin/env python3
"""patch_add_version_monitor_agent.py -- wrap the module-level version monitor in a BaseAgent.

Adds the BaseAgent import + a VersionMonitorAgent class (delegating to the existing
module-level run()) and corrects the stale argparse name. Count-guarded, idempotent,
backup-first, py_compile-gated. Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/agents/version_monitor_agent.py")
IMPORT_ANCHOR = "from __future__ import annotations\n"
CLASS_ANCHOR  = '\nif __name__ == "__main__":\n'
IMPORT_LINE = ("from __future__ import annotations\n\n"
               "from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent\n")
CLASS_BLOCK = '''

_run_watch_targets = run  # module-level watch-target orchestrator (aliased before the method below)


class VersionMonitorAgent(BaseAgent):
    """Upstream-release monitor: pykan / ClinVar / AlphaMissense / torch-geometric.

    Distinct from InfrastructureDriftAgent (which diffs *installed* package versions):
    this watches for *new upstream releases*. BaseAgent adapter over the module-level
    watch-target functions; surfaces a summary into the 'version_monitor' section.
    """

    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)
        updates = _run_watch_targets(dry_run=dry_run)
        alerts = updates.get("literature_scout.alerts", [])
        result = {
            "status": "ok",
            "n_alerts": len(alerts),
            "alerts": alerts,
            "pykan_installed": updates.get("literature_scout.pykan_installed"),
            "pykan_latest": updates.get("literature_scout.pykan_latest"),
            "pykan_alert": updates.get("literature_scout.pykan_alert", False),
            "last_run": updates.get("literature_scout.last_run"),
            "checked_at": self._now_iso(),
            "dry_run": dry_run,
        }
        self._update_section("version_monitor", result)
        self._log_finish(result)
        return result

'''

def fail(m): print(f"ABORT: {m}"); sys.exit(1)

def main() -> int:
    if not TARGET.exists(): fail(f"not found: {TARGET.resolve()}")
    txt = TARGET.read_text(encoding="utf-8")
    if "class VersionMonitorAgent(" in txt:
        print("no-op: VersionMonitorAgent already present"); return 0
    if txt.count(IMPORT_ANCHOR) != 1: fail(f"import anchor count != 1 ({txt.count(IMPORT_ANCHOR)})")
    if txt.count(CLASS_ANCHOR) != 1:  fail(f"class anchor (__main__) count != 1 ({txt.count(CLASS_ANCHOR)})")
    patched = txt.replace(IMPORT_ANCHOR, IMPORT_LINE, 1)
    patched = patched.replace(CLASS_ANCHOR, CLASS_BLOCK + CLASS_ANCHOR, 1)
    patched = patched.replace('description="LiteratureScoutAgent"', 'description="VersionMonitorAgent"', 1)
    bak = TARGET.with_suffix(".py.bak"); shutil.copy2(TARGET, bak)
    TARGET.write_text(patched, encoding="utf-8")
    try: py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(bak, TARGET); fail(f"py_compile failed, reverted: {e}")
    print(f"VersionMonitorAgent added; backup: {bak}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
