#!/usr/bin/env python3
"""scripts/audit_agent_roster.py -- definitive agent enumeration via AST (READ-ONLY).

v2: keys by (file, class) instead of class-name, so duplicate class names defined in
different files are NO LONGER silently overwritten -- they are reported explicitly.
Resolves multi-line signatures, intermediate bases, transitive BaseAgent inheritance.
Author: Monzia Moodie.
"""
from __future__ import annotations
import ast, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("src/genomic_variant_classifier/agent_layer")

def collect(root: Path):
    defs = []  # list of (name, bases:list[str], file, lineno)
    for f in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            print(f"  !! PARSE ERROR {f}: {e}"); continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    try: bases.append(ast.unparse(b))
                    except Exception: bases.append("<?>")
                defs.append((node.name, bases, f.relative_to(root).as_posix(), node.lineno))
    return defs

def _short(bases):
    out = []
    for b in bases:
        b = b.split("[")[0].strip()
        out.append(b.split(".")[-1])
    return out

def is_agent(name, by_name, cache=None):
    if cache is None: cache = {}
    if name in cache: return cache[name]
    if name not in by_name:
        cache[name] = (name == "BaseAgent"); return cache[name]
    cache[name] = False
    res = False
    for (_n, bases, _f, _l) in by_name[name]:
        for bn in _short(bases):
            if bn == "BaseAgent" or (bn in by_name and is_agent(bn, by_name, cache)):
                res = True; break
        if res: break
    cache[name] = res
    return res

def main() -> int:
    if not ROOT.exists():
        print(f"not found: {ROOT.resolve()}"); return 1
    defs = collect(ROOT)
    by_name = defaultdict(list)
    for d in defs: by_name[d[0]].append(d)
    print(f"class definitions parsed: {len(defs)} | distinct names: {len(by_name)}\n")
    for name in sorted(by_name):
        for (_n, bases, f, ln) in by_name[name]:
            print(f"  {name:34s} <- {(', '.join(bases) or '(none)'):26s} [{f}:{ln}]")
    # DUPLICATES -- the thing the v1 dict hid
    dups = {n: v for n, v in by_name.items() if len(v) > 1}
    print(f"\n=== DUPLICATE class names (defined in >1 file): {len(dups)} ===")
    if dups:
        for n, v in sorted(dups.items()):
            print(f"  !! {n} defined in: " + ", ".join(f"{f}:{ln}" for (_n, _b, f, ln) in v))
    else:
        print("  (none)")
    agent_names = {n for n in by_name if is_agent(n, by_name) and n != "BaseAgent" and not n.startswith("_")}
    used_as_base = set()
    for n in agent_names:
        for (_n, bases, _f, _l) in by_name[n]:
            for bn in _short(bases):
                if bn in agent_names: used_as_base.add(bn)
    leaves = sorted(agent_names - used_as_base)
    print(f"\n=== BaseAgent-descendant agent NAMES: {len(agent_names)} "
          f"(intermediates: {sorted(used_as_base) or 0}) ===")
    print(f"=== concrete (leaf) agent names: {len(leaves)} ===")
    for a in leaves: print(f"  {a}")
    vm = [n for n in by_name if "VersionMonitor" in n]
    print(f"\nVersionMonitor class present? {'YES: ' + ', '.join(vm) if vm else 'NO'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
