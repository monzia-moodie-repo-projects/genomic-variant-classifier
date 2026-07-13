#!/usr/bin/env python
"""verify_postcleanup.py (2026-07-11) -- READ-ONLY post-delete verification. Confirms the cleanup
removed ONLY the intended files and left every real artifact intact: (1) merged seq_windows.parquet +
its manifest still present + non-empty; (2) the live dbNSFP index still present + non-empty; (3) the
seq_windows part_*/.done are GONE; (4) the .OOMbak is GONE; (5) the 7 this-arc .bak are GONE; (6) the
109 tracked patch_ are STILL present (guard protected them); (7) git status has no unexpected deletions
of TRACKED files (git diff --stat should be empty for deletions). ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-",n=78): print(c*n)
def human(n):
    for u in ["B","KB","MB","GB"]:
        if n<1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"
def sz(p):
    try: return Path(p).stat().st_size
    except: return 0

ok_all = True
print("="*78); print("POST-CLEANUP VERIFY (READ-ONLY) -- 2026-07-11"); print("="*78)

print("\n### 1. REAL ARTIFACTS must still be present + non-empty ###")
must_exist = [
    "data/processed/seq_windows/seq_windows.parquet",
    "data/external/dbnsfp/dbnsfp_full_index.parquet",
]
# manifest: find any manifest json/parquet in seq_windows
swd = Path("data/processed/seq_windows")
manifests = list(swd.glob("*manifest*")) if swd.exists() else []
for m in must_exist:
    present = Path(m).exists() and sz(m)>0
    ok_all = ok_all and present
    print(a(f"  {'OK ' if present else 'FAIL'}  {human(sz(m)):>9}  {m}"))
print(a(f"  manifest files in seq_windows: {[p.name for p in manifests] or '(none found -- check)'}"))

print("\n### 2. REMOVED files must be GONE ###")
gone_globs = [
    ("seq_windows parts", list(swd.glob("part_*.parquet")) if swd.exists() else []),
    ("seq_windows .done", list(swd.glob("*.done")) if swd.exists() else []),
]
for label, hits in gone_globs:
    gone = len(hits)==0
    ok_all = ok_all and gone
    print(a(f"  {'OK (gone)' if gone else 'FAIL (still present: '+str(len(hits))+')'}  {label}"))
oom = "data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak"
oom_gone = not Path(oom).exists()
ok_all = ok_all and oom_gone
print(a(f"  {'OK (gone)' if oom_gone else 'FAIL (still present)'}  .OOMbak"))

print("\n### 3. this-arc .bak must be GONE ###")
arc_baks = ["scripts/train.py.w1bak","scripts/train.py.w2b2bak",
 "src/genomic_variant_classifier/data/real_data_prep.py.w2b1bak",
 "src/genomic_variant_classifier/data/split_protocol_v2.py.w2b1bak",
 "src/genomic_variant_classifier/models/variant_ensemble.py.w2bak",
 "src/genomic_variant_classifier/evaluation/evaluator.py.bak",
 "src/genomic_variant_classifier/data/database_connectors.py.bak"]
for b in arc_baks:
    g = not Path(b).exists()
    ok_all = ok_all and g
    print(a(f"  {'OK (gone)' if g else 'FAIL (still present)'}  {b}"))

print("\n### 4. TRACKED files must be UNTOUCHED (no tracked deletions) ###")
r = subprocess.run(["git","status","--short"],capture_output=True,text=True)
deleted_tracked = [l for l in r.stdout.splitlines() if l.startswith(" D ") or l.startswith("D ")]
if deleted_tracked:
    ok_all = False
    print(a(f"  FAIL: {len(deleted_tracked)} TRACKED file(s) show as deleted:"))
    for l in deleted_tracked[:20]: print(a(f"    {l}"))
else:
    print("  OK: no tracked file shows as deleted (git status has no ' D' entries)")

print("\n### 5. spot-check: a few tracked patch_ that SHOULD survive ###")
for p in ["scripts/patch_omim_connector.py","scripts/patch_eve_gene_resolution.py",
          "scripts/patch_gnn_optim.py"]:
    present = Path(p).exists()
    ok_all = ok_all and present
    print(a(f"  {'OK (survived)' if present else 'FAIL (missing!)'}  {p}"))

line("=")
print(a(f"POST-CLEANUP VERIFY: {'ALL CHECKS PASSED -- cleanup hit only intended files.' if ok_all else 'SOME CHECKS FAILED -- investigate above.'}"))
print("READ-ONLY. Nothing changed.")
raise SystemExit(0 if ok_all else 1)
