#!/usr/bin/env python
"""audit_history_blobs.py (2026-07-11) -- READ-ONLY: scan the ENTIRE git history for large blobs
(>50MB) that live in past commits (not just HEAD). These are what bloat .git AND block the GitHub push
(>100MB per-file). Lists each with size, the path(s) it was committed under, and whether it still
exists on disk (regenerable-cache check). Rewrites NOTHING. This is the pre-rewrite audit. ASCII-safe.
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

print("="*78); print("HISTORY BLOB AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

# 1. All blob objects in history, sorted by size. rev-list --objects across --all, then cat-file
#    batch-check for sizes.
print("\n### scanning ALL objects across full history (this may take a moment) ###")
rl = subprocess.run(["git","rev-list","--objects","--all"],capture_output=True,text=True)
objs = {}
for lnum,linetext in enumerate(rl.stdout.splitlines()):
    parts = linetext.split(" ",1)
    sha = parts[0]
    path = parts[1] if len(parts)>1 else ""
    objs[sha]=path

# batch-check sizes
cf = subprocess.run(["git","cat-file","--batch-check=%(objecttype) %(objectname) %(objectsize)"],
                    input="\n".join(objs.keys()), capture_output=True, text=True)
blobs = []
for linetext in cf.stdout.splitlines():
    p = linetext.split()
    if len(p)==3 and p[0]=="blob":
        sha,size = p[1],int(p[2])
        if size > 50*1024*1024:
            blobs.append((size,sha,objs.get(sha,"")))
blobs.sort(reverse=True)

print(a(f"\n### LARGE BLOBS IN HISTORY (>50MB) -- {len(blobs)} found ###"))
over100 = []
for size,sha,path in blobs:
    on_disk = Path(path).exists() if path else False
    flag = " *** >100MB: BLOCKS PUSH ***" if size>100*1024*1024 else ""
    if size>100*1024*1024: over100.append((size,sha,path))
    print(a(f"  {human(size):>9}  {path or '(no path)'}"))
    print(a(f"             sha={sha[:12]} on-disk-now={on_disk}{flag}"))

print(a(f"\n### SUMMARY ###"))
print(a(f"  {len(blobs)} blobs >50MB in history; {len(over100)} are >100MB (each BLOCKS the push)."))
tot = sum(s for s,_,_ in blobs)
tot100 = sum(s for s,_,_ in over100)
print(a(f"  total >50MB blob bytes in history: {human(tot)}"))
print(a(f"  total >100MB (must-purge-to-push): {human(tot100)}"))

print(a("\n### paths to purge (the >100MB set) -- these go into the filter-repo path list ###"))
paths_seen = []
for size,sha,path in over100:
    if path and path not in paths_seen:
        paths_seen.append(path)
        regenerable = any(k in path.lower() for k in ["cache","_cache",".joblib",".tar.gz","outputs/","run1"])
        print(a(f"  {human(size):>9}  {path}   [{'likely regenerable/artifact' if regenerable else 'REVIEW -- may be source'}]"))

line("=")
print("READ-ONLY. Nothing rewritten. Review the >100MB paths -> these must be purged from history to push.")
print("NEXT (after review): backup .git, then git filter-repo --path <each> --invert-paths to excise them.")
raise SystemExit(0)
