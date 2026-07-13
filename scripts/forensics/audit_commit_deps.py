#!/usr/bin/env python
"""audit_commit_deps.py (2026-07-11) -- READ-ONLY: resolve the last commit-grouping question. (1) does
the re-baseline machinery (split_protocol_v2, real_data_prep.run_v2, delta_window_builder,
seq_window_manifest) import allele_classify? -> determines if allele_classify is a commit-3 dependency
or separate. (2) do split_protocol_v2 + the conformal package import cleanly (self-contained)? Commits
nothing. ASCII-safe.
"""
from __future__ import annotations
import ast, importlib, io, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)

def imports_of(path):
    """Return the set of module names imported by a file (top-level + from)."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return set(), f"(parse failed: {e})"
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names: mods.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module: mods.add(node.module)
    return mods, None

def main() -> int:
    print("="*78); print("COMMIT DEPENDENCY AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

    print("\n### 1. Does the re-baseline machinery import allele_classify? ###")
    machinery = [
        "src/genomic_variant_classifier/data/split_protocol_v2.py",
        "src/genomic_variant_classifier/data/real_data_prep.py",
        "src/genomic_variant_classifier/data/delta_window_builder.py",
        "src/genomic_variant_classifier/data/seq_window_manifest.py",
        "scripts/build_seq_windows.py",
    ]
    any_allele = False
    for m in machinery:
        if not Path(m).exists():
            print(a(f"  {m}: MISSING")); continue
        mods, err = imports_of(m)
        if err:
            print(a(f"  {m}: {err}")); continue
        allele_refs = [x for x in mods if "allele_classify" in x]
        if allele_refs:
            any_allele = True
            print(a(f"  {m}: IMPORTS allele_classify {allele_refs}"))
        else:
            print(a(f"  {m}: no allele_classify import"))
    print(a(f"  -> allele_classify is {'a COMMIT-3 DEPENDENCY' if any_allele else 'SEPARATE (commit 4)'}"))

    print("\n### 2. Does allele_classify import the recovery scripts (reverse dep)? ###")
    ac = "src/genomic_variant_classifier/data/allele_classify.py"
    if Path(ac).exists():
        mods, err = imports_of(ac)
        internal = sorted(x for x in mods if "genomic_variant_classifier" in x)
        print(a(f"  allele_classify internal imports: {internal or '(none)'}"))
    else:
        print("  (allele_classify.py missing)")

    print("\n### 3. Do split_protocol_v2 + conformal package import cleanly? ###")
    sys.path.insert(0, "src")
    for mod in [
        "genomic_variant_classifier.data.split_protocol_v2",
        "genomic_variant_classifier.conformal",
        "genomic_variant_classifier.conformal.calibrate",
        "genomic_variant_classifier.conformal.coverage",
        "genomic_variant_classifier.conformal.grouped",
        "genomic_variant_classifier.conformal.mondrian",
        "genomic_variant_classifier.conformal.scores",
        "genomic_variant_classifier.conformal.split",
    ]:
        try:
            importlib.import_module(mod)
            print(a(f"  ok   {mod}"))
        except Exception as e:
            print(a(f"  FAIL {mod}: {type(e).__name__}: {str(e)[:120]}"))

    print("\n### 4. conformal package internal imports (self-contained?) ###")
    cdir = Path("src/genomic_variant_classifier/conformal")
    if cdir.exists():
        allmods = set()
        for p in cdir.glob("*.py"):
            mods, _ = imports_of(str(p))
            allmods |= {x for x in mods if "genomic_variant_classifier" in x}
        for m in sorted(allmods):
            print(a(f"  {m}"))

    line("=")
    print("COMMIT DEPENDENCY AUDIT COMPLETE. Read-only.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
