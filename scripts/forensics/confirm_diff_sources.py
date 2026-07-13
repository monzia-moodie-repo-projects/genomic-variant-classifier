#!/usr/bin/env python
"""confirm_diff_sources.py (2026-07-11) -- READ-ONLY final insurance before COMMIT 6: confirm the two
newly-identified cohort sources (diff_cohorts.py, diffcore.py) are UNTRACKED (need git add) and compile
clean, and re-confirm the original 8 + 7 tests are all still untracked+syntax-ok. Commits nothing.
"""
from __future__ import annotations
import subprocess, sys, io
from pathlib import Path
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def tracked(p): return subprocess.run(["git","ls-files","--error-unmatch",p],capture_output=True,text=True).returncode==0

SOURCES = ["scripts/build_cohort_from_source.py","scripts/ingest_clinvar_snapshot.py",
 "scripts/inventory_clinvar_snapshots.py","scripts/rebuild_cohort_v3.py",
 "scripts/rebuild_cohort_v3_by_sid.py","scripts/rebuild_cohort_v3_final.py",
 "scripts/rederive_pathogenicity.py","scripts/finalize_fresh_parquet.py",
 "scripts/diff_cohorts.py","scripts/diffcore.py"]
TESTS = ["tests/test_build_cohort_from_source.py","tests/test_ingest_clinvar_snapshot.py",
 "tests/test_cohort_v3_rebuild.py","tests/test_rebuild_cohort_v3_by_sid.py",
 "tests/test_rebuild_cohort_v3_final.py","tests/test_dedup_collapse.py","tests/test_diff_cohorts.py"]

print("="*78); print("COMMIT 6 FINAL CONFIRM (READ-ONLY) -- 2026-07-11"); print("="*78)
print("\n### 10 SOURCES -- untracked + compile-clean? ###")
allok=True
for s in SOURCES:
    p=Path(s)
    if not p.exists(): print(a(f"  MISSING  {s}")); allok=False; continue
    rc=subprocess.run([sys.executable,"-m","py_compile",s],capture_output=True,text=True)
    st="tracked" if tracked(s) else "UNTRACKED"
    tag="ADDED" if s.endswith(("diff_cohorts.py","diffcore.py")) else ""
    print(a(f"  [{st:9}] compile={'ok' if rc.returncode==0 else 'FAIL'}  {s}  {tag}"))
    if rc.returncode!=0: allok=False
print("\n### 7 TESTS -- untracked? ###")
for t in TESTS:
    p=Path(t)
    print(a(f"  [{'tracked' if tracked(t) else 'UNTRACKED'}]  {t}{'' if p.exists() else '  MISSING'}"))
print("\n"+"="*78)
print(a(f"ALL SOURCES COMPILE-CLEAN: {allok}. If UNTRACKED + compile ok -> add-list is safe."))
print("READ-ONLY. Nothing committed.")
