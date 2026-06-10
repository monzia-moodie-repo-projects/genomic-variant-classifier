#!/usr/bin/env python3
"""scripts/audit_agent_operational.py -- per-agent OPERATIONAL scorecard (READ-ONLY).

v3: resolves inheritance and run() TRANSITIVELY through intermediate bases (e.g.
DriftMonitorBase), so agents that inherit BaseAgent/run() via a shared base are
correctly counted operational. Author: Monzia Moodie.
"""
from __future__ import annotations
import ast, re, sys
from pathlib import Path

PKG    = Path("src/genomic_variant_classifier")
ALAYER = PKG / "agent_layer"
ORCH   = ALAYER / "orchestrator.py"
SCAN_ROOTS = [PKG, Path("scripts"), Path("tests")]

def _short(node) -> str:
    try: return ast.unparse(node).split("[")[0].split(".")[-1].strip()
    except Exception: return "<?>"

def class_graph(root: Path) -> dict:
    g = {}
    for f in sorted(root.rglob("*.py")):
        try: tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e: print(f"  !! parse {f}: {e}"); continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                g[n.name] = {
                    "file": f,
                    "bases": [_short(b) for b in n.bases],
                    "methods": {m.name for m in n.body
                                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))},
                }
    return g

def inherits(name, target, g, seen=None):
    seen = seen or set()
    if name in seen: return False
    seen.add(name)
    node = g.get(name)
    if not node: return False
    if target in node["bases"]: return True
    return any(inherits(b, target, g, seen) for b in node["bases"] if b in g)

def has_method(name, meth, g, seen=None):
    seen = seen or set()
    if name in seen: return False
    seen.add(name)
    node = g.get(name)
    if not node: return False
    if meth in node["methods"]: return True
    return any(has_method(b, meth, g, seen) for b in node["bases"] if b in g)

def orchestrator_registered() -> set:
    if not ORCH.exists(): return set()
    return set(re.findall(r"\b(\w+Agent)\b", ORCH.read_text(encoding="utf-8")))

def references(name: str, def_file: Path) -> list:
    hits = []
    for root in SCAN_ROOTS:
        if not root.exists(): continue
        for f in root.rglob("*.py"):
            if f.resolve() == def_file.resolve(): continue
            try: t = f.read_text(encoding="utf-8")
            except Exception: continue
            if re.search(rf"\b{name}\b", t): hits.append(f.as_posix())
    return hits

def has_test(name: str) -> bool:
    td = Path("tests")
    if not td.exists(): return False
    for f in td.rglob("*.py"):
        try:
            if re.search(rf"\b{name}\b", f.read_text(encoding="utf-8")): return True
        except Exception: pass
    return False

def main() -> int:
    if not ALAYER.exists(): print(f"not found: {ALAYER.resolve()}"); return 1
    g = class_graph(ALAYER)
    agents = sorted(n for n in g if n.endswith("Agent") and not n.startswith("_") and n != "BaseAgent")
    reg = orchestrator_registered()
    print(f"agents found: {len(agents)} | orchestrator references: {sorted(reg)}\n")
    op = comp = orphan = other = 0
    for name in agents:
        node = g[name]
        inh = inherits(name, "BaseAgent", g)
        run = has_method(name, "run", g)
        registered = name in reg
        refs = references(name, node["file"])
        test = has_test(name)
        if inh and run and registered:   verdict, tag = "OPERATIONAL (BaseAgent+run+registered)", "OK"; op+=1
        elif inh and run:                verdict, tag = "IMPLEMENTED, NOT REGISTERED", "??"; other+=1
        elif refs:                       verdict, tag = "COMPOSED/LIBRARY (no run(), called directly)", "~~"; comp+=1
        else:                            verdict, tag = "ORPHANED (no run(), no references)", "XX"; orphan+=1
        print(f"[{tag}] {name}")
        print(f"      inherits BaseAgent: {inh} (transitive) | run(): {run} (incl. inherited) | registered: {registered} | test: {test}")
        print(f"      referenced by: {refs or '(NONE)'}")
        print(f"      -> {verdict}\n")
    print(f"SUMMARY: operational={op} composed={comp} orphaned={orphan} not-registered={other} total={len(agents)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
