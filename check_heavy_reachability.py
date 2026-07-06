#!/usr/bin/env python3
"""check_heavy_reachability.py -- Monzia Moodie

Definitively answer, for EACH registered agent: can it reach heavy compute (torch / sklearn / shap /
transformers / gudhi / .fit/.predict) through its LOCAL import graph, and via what path?

Why: the runtime profile uses observed durations, but dry-run SKIPS most agents' heavy work (observed=0ms),
and a one-level static scan can miss compute buried deeper (e.g. agent -> util -> ewc_utils -> torch).
This walks the local import graph transitively (bounded depth) so no heavy agent is mislabeled "trivial"
and then given a tight timeout that would abort legitimate training/explanation work.

Pure stdlib, import-free w.r.t. the project. Run from repo root:  python check_heavy_reachability.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src"
ORCH = SRC / "genomic_variant_classifier" / "agent_layer" / "orchestrator.py"
MAX_DEPTH = 4  # agent -> helper -> helper -> helper -> helper
_BARE_WARN: list = []  # (module, raw_import, resolved_sibling) -- runtime-suspect bare imports

_REG_RE = re.compile(r'"(\w+)"\s*:\s*_Lazy\(\s*["\']([^"\']+)["\']')
_HEAVY_RE = re.compile(r'(import\s+torch|import\s+shap|from\s+sklearn|import\s+sklearn|\.fit\(|\.predict\(|import\s+transformers|import\s+gudhi|import\s+torch_geometric)')
_LOCALIMP_RE = re.compile(r'^\s*from\s+(\.*)([\w.]*)\s+import\s+(.+)$', re.MULTILINE)


def registry_map(t: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _REG_RE.finditer(t)}


def module_to_path(module: str) -> Path:
    return SRC / (module.replace(".", "/") + ".py")


def _resolve(cur_mod: str, dots: str, tail: str) -> str | None:
    if dots:
        pkg = cur_mod.split(".")[:-1]
        up = len(dots) - 1
        base = pkg[: len(pkg) - up] if up > 0 else pkg
        return ".".join(base + (tail.split(".") if tail else []))
    if tail.startswith("genomic_variant_classifier"):
        return tail
    # bare import (e.g. `from ewc_utils import X`): treat as a sibling of cur_mod. At runtime this is
    # src-layout-SUSPECT (a bare top-level import of a package-internal module usually fails), but for
    # reachability it reveals the sibling's compute. Caller filters by .exists().
    if tail:
        return ".".join(cur_mod.split(".")[:-1] + tail.split("."))
    return None


# A bare from-import whose target resolves to an existing sibling module file: runtime-suspect.
def _bare_sibling_imports(cur_mod: str, src: str) -> list[tuple[str, str]]:
    out = []
    for dots, tail, _clause in _LOCALIMP_RE.findall(src):
        if dots or not tail or tail.startswith("genomic_variant_classifier"):
            continue
        sib = ".".join(cur_mod.split(".")[:-1] + tail.split("."))
        if module_to_path(sib).exists():
            out.append((tail, sib))
    return out


def _names(clause: str) -> list[str]:
    clause = clause.split("#", 1)[0].strip().strip("()")
    out = []
    for part in clause.split(","):
        nm = part.strip().split(" as ")[0].strip()
        if nm and nm != "*":
            out.append(nm)
    return out


def _local_imports(cur_mod: str, src: str) -> list[str]:
    mods: set[str] = set()
    for dots, tail, clause in _LOCALIMP_RE.findall(src):
        base = _resolve(cur_mod, dots, tail)
        if base is None:
            continue
        if module_to_path(base).exists():
            mods.add(base)
        for nm in _names(clause):
            sub = f"{base}.{nm}" if base else nm
            if module_to_path(sub).exists():
                mods.add(sub)
    return sorted(mods)


def heavy_reach(start_mod: str) -> tuple[set[str], list[str]]:
    """BFS the local import graph from start_mod. Return (heavy_markers, shortest_path_to_first_heavy)."""
    seen: set[str] = set()
    # queue of (module, path_list)
    queue: list[tuple[str, list[str]]] = [(start_mod, [start_mod.rsplit(".", 1)[-1]])]
    markers: set[str] = set()
    first_path: list[str] = []
    while queue:
        mod, path = queue.pop(0)
        if mod in seen or len(path) > MAX_DEPTH + 1:
            continue
        seen.add(mod)
        p = module_to_path(mod)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for raw, sib in _bare_sibling_imports(mod, text):
            _BARE_WARN.append((mod.rsplit(".",1)[-1], raw, sib))
        hits = set(_HEAVY_RE.findall(text))
        if hits:
            markers |= hits
            if not first_path:
                first_path = path
        for imp in _local_imports(mod, text):
            if imp not in seen:
                queue.append((imp, path + [imp.rsplit(".", 1)[-1]]))
    return markers, first_path


def main() -> int:
    if not ORCH.exists():
        print(f"ERROR: orchestrator not found at {ORCH}", file=sys.stderr)
        return 2
    reg = registry_map(ORCH.read_text(encoding="utf-8"))
    print("=" * 100)
    print(f"HEAVY-COMPUTE REACHABILITY  (transitive local-import scan, max depth {MAX_DEPTH})")
    print("=" * 100)
    heavy, light = [], []
    for name, spec in sorted(reg.items()):
        mod = spec.rsplit(":", 1)[0]
        markers, path = heavy_reach(mod)
        if markers:
            heavy.append((name, sorted(markers), path))
        else:
            light.append(name)
    print(f"\nHEAVY-REACHABLE ({len(heavy)}) -- real runtime likely minutes; do NOT give a tight ceiling:")
    for name, markers, path in heavy:
        print(f"  {name}")
        print(f"      markers: {', '.join(markers)}")
        print(f"      via:     {' -> '.join(path)}")
    print(f"\nLIGHTWEIGHT ({len(light)}) -- no heavy compute reachable; safe for a bounded ceiling:")
    for name in light:
        print(f"  {name}")
    if _BARE_WARN:
        print("\n" + "!" * 100)
        print("SUSPECT BARE IMPORTS (a sibling module imported as a bare top-level name -- likely")
        print("ModuleNotFoundError at runtime in this src-layout; should be relative or full-package):")
        for mod, raw, sib in sorted(set(_BARE_WARN)):
            print(f"  in {mod}:  `from {raw} import ...`  -> should target  {sib}")
        print("!" * 100)
    print("\n" + "=" * 100)
    print(f"{len(reg)} agents. HEAVY={len(heavy)} LIGHT={len(light)}.  suspect_bare_imports={len(set(_BARE_WARN))}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
