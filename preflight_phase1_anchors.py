"""preflight_phase1_anchors.py -- verify the Phase 1 edit-script's anchors match the real orchestrator
EXACTLY, before running any expensive gate. Exits nonzero (and prints why) on any mismatch."""
from __future__ import annotations
import sys
from pathlib import Path

ORCH = "src/genomic_variant_classifier/agent_layer/orchestrator.py"


def main() -> int:
    p = Path(ORCH)
    if not p.is_file():
        print(f"ANCHOR-PREFLIGHT FAIL: orchestrator not found at {ORCH}")
        return 1
    s = p.read_text(encoding="utf-8")

    problems = []

    # Already applied?
    if "# >>> PHASE1_LAZY_REGISTRY <<<" in s or "# >>> PHASE1_GUARDED_CONSTRUCTION <<<" in s:
        print("ANCHOR-PREFLIGHT: Phase 1 sentinels already present -- edit-script would no-op (idempotent).")
        # Not a failure; the installer treats this as 'already applied'.
        return 0

    # Region 1 anchors: _register_agents def + the dict literal opener (exactly once).
    if s.count("def _register_agents(self)") != 1:
        problems.append(f"'def _register_agents(self)' count = {s.count('def _register_agents(self)')} (expect 1)")
    if s.count("self._agent_registry = {") != 1:
        problems.append(f"'self._agent_registry = {{' count = {s.count('self._agent_registry = {')} (expect 1)")

    # Region 1 needs eager 'from <agents>.<mod> import <Class>' lines in the method.
    base = "from genomic_variant_classifier.agent_layer.agents."
    if base not in s:
        problems.append(f"no eager agent imports ('{base}...') found")

    # Region 2 anchors: the eager per-agent guard pattern (hasattr ... _t0/_err/try/result=agent.run).
    needles = [
        'if hasattr(agent_cls, "from_default_baseline"):',
        "agent = agent_cls.from_default_baseline(self._state)",
        "agent = agent_cls(self._state)",
        "_t0 = time.monotonic()",
        "_err = None",
        "result = agent.run(dry_run=self._dry_run)",
    ]
    for n in needles:
        c = s.count(n)
        if c != 1:
            problems.append(f"guard anchor {n!r} count = {c} (expect 1)")

    if problems:
        print("ANCHOR-PREFLIGHT FAIL:")
        for pr in problems:
            print("  - " + pr)
        return 1
    print("ANCHOR-PREFLIGHT OK: all Region 1 + Region 2 anchors match exactly once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
