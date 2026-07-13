#!/usr/bin/env python
"""audit_cohort_commit_scope.py (2026-07-11) -- READ-ONLY: build the COMPLETE, correctly-paired
add-list for the cohort-pipeline commit. For each of the 7 green cohort test files, parse its imports
to find the SOURCE module(s) it exercises; confirm each source is present + its tracked state; list
any cohort SOURCE among untracked files not yet in the candidate set (e.g. diff_cohorts.py, diffcore.py,
a dedup module). Also confirm which candidate scripts are scratch-diagnostics (diagnose_/probe_/recover_)
to EXCLUDE. Output: a curated source+test add-list where every test has its source and vice-versa.
Commits/edits/deletes NOTHING. ASCII-safe.
"""
from __future__ import annotations
import ast, io, subprocess, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)
def tracked(path):
    return subprocess.run(["git","ls-files","--error-unmatch",path],
                          capture_output=True, text=True).returncode == 0

COHORT_TESTS = [
    "tests/test_build_cohort_from_source.py",
    "tests/test_ingest_clinvar_snapshot.py",
    "tests/test_cohort_v3_rebuild.py",
    "tests/test_rebuild_cohort_v3_by_sid.py",
    "tests/test_rebuild_cohort_v3_final.py",
    "tests/test_dedup_collapse.py",
    "tests/test_diff_cohorts.py",
]
CANDIDATE_SCRIPTS = [
    "scripts/build_cohort_from_source.py","scripts/ingest_clinvar_snapshot.py",
    "scripts/inventory_clinvar_snapshots.py","scripts/rebuild_cohort_v3.py",
    "scripts/rebuild_cohort_v3_by_sid.py","scripts/rebuild_cohort_v3_final.py",
    "scripts/rederive_pathogenicity.py","scripts/finalize_fresh_parquet.py",
]

def imported_names(path):
    """Return raw module strings imported by a test (to trace which source it exercises)."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return [], f"(parse failed: {e})"
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module)
        elif isinstance(n, ast.Import):
            for nm in n.names: mods.append(nm.name)
    return sorted(set(mods)), None

def resolve_source(mod):
    """Map a dotted/plain module to an on-disk source path if it's a project source or scripts/ module."""
    hits = []
    rel = mod.replace(".", "/")
    for base in ["src","."]:
        for cand in [Path(base)/f"{rel}.py", Path(base)/rel/"__init__.py"]:
            if cand.exists(): hits.append(str(cand))
    # scripts are often imported by basename via sys.path insertion of scripts/
    for cand in [Path("scripts")/f"{mod.split('.')[-1]}.py"]:
        if cand.exists(): hits.append(str(cand))
    return hits

def main() -> int:
    print("="*78); print("COHORT COMMIT SCOPE AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

    print("\n### 1. each green cohort test -> source module(s) it imports/exercises ###")
    needed_sources = set()
    for t in COHORT_TESTS:
        if not Path(t).exists():
            print(a(f"\n  {t}: MISSING")); continue
        mods, err = imported_names(t)
        if err: print(a(f"\n  {t}: {err}")); continue
        proj = [m for m in mods if ("genomic_variant_classifier" in m) or
                any(Path("scripts",f"{m.split('.')[-1]}.py").exists() for _ in [0])]
        print(a(f"\n  {t}"))
        print(a(f"    imports: {mods}"))
        for m in mods:
            srcs = resolve_source(m)
            if srcs:
                for s in srcs:
                    # only care about our cohort sources / project sources, not stdlib/3rd-party
                    if s.startswith("scripts/") or s.startswith("src/"):
                        needed_sources.add(s)
                        print(a(f"      source: {s}  [{'tracked' if tracked(s) else 'UNTRACKED'}]"))

    print("\n### 2. candidate scripts NOT referenced by any test (ship anyway? or CLI-only) ###")
    for s in CANDIDATE_SCRIPTS:
        if s not in needed_sources:
            st = "tracked" if tracked(s) else "UNTRACKED"
            print(a(f"  {s}  [{st}]  -- not imported by a test (likely a CLI entrypoint; ship with pipeline)"))

    print("\n### 3. sources a test needs that are NOT in the candidate 8 (orphan-source check) ###")
    extra = sorted(s for s in needed_sources if s.startswith("scripts/") and s not in CANDIDATE_SCRIPTS)
    if extra:
        for s in extra:
            print(a(f"  EXTRA source needed: {s}  [{'tracked' if tracked(s) else 'UNTRACKED'}]"))
    else:
        print("  (none -- every test's source is already in the candidate set)")

    print("\n### 4. scratch to EXCLUDE (diagnostics that produced the pipeline, not the pipeline) ###")
    rc = subprocess.run(["git","status","--short","scripts"], capture_output=True, text=True)
    scratch = [l[3:] for l in rc.stdout.splitlines() if l.startswith("?? ")
               and any(k in l for k in ["diagnose_","probe_","recover_","scan_","characterize_",
                                        "investigate_","read_indel","harden_","reconcile_","locate_"])]
    print(a(f"  {len(scratch)} diagnostic/scratch scripts -> KEEP UNTRACKED (not product):"))
    for s in scratch[:12]:
        print(a(f"    {s}"))
    if len(scratch) > 12: print(a(f"    ... +{len(scratch)-12} more"))

    line("=")
    print("Use section 1+3 to build COMMIT 6 add-list: every source + its test, nothing orphaned.")
    print("READ-ONLY. Nothing committed, edited, or deleted.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
