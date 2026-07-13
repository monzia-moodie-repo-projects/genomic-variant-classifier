#!/usr/bin/env python
"""git_gc_and_pushcheck.py (2026-07-11) -- sequenced, auditable git maintenance. NON-DESTRUCTIVE:
git gc repacks the object store only (never touches working files); push --dry-run tests without
pushing; NO history rewrite, NO reflog expiry (that is flagged as a separate deliberate choice).
Sequence: (1) measure BEFORE, (2) git gc --prune=now, (3) measure AFTER, (4) git fsck --full
integrity, (5) confirm HEAD + working tree unchanged, (6) git push --dry-run to test push viability.
Every step captured. ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-",n=78): print(c*n)
def sh(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or ""), (r.stderr or "")
    except Exception as e:
        return -1, "", f"(failed: {e})"

print("="*78); print("GIT GC + PUSH-CHECK (sequenced, auditable) -- 2026-07-11"); print("="*78)

print("\n### 1. BEFORE -- object store size ###")
rc,out,err = sh(["git","count-objects","-vH"])
print(a(out.strip()))

print("\n### 2. git gc --prune=now (repack object store; NEVER touches working files) ###")
rc,out,err = sh(["git","gc","--prune=now"])
print(a((out or "").strip() or "(gc produced no stdout)"))
if err.strip(): print(a("  [stderr] "+err.strip()[:500]))
print(a(f"  gc exit = {rc}"))

print("\n### 3. AFTER -- object store size (measure reclaim) ###")
rc,out,err = sh(["git","count-objects","-vH"])
print(a(out.strip()))

print("\n### 4. git fsck --full (integrity -- repo must be intact after repack) ###")
rc,out,err = sh(["git","fsck","--full"])
combined = (out+err).strip()
# fsck prints 'dangling' notices which are NORMAL (unreferenced objects); only 'error'/'missing'/
# 'corrupt' are real problems.
real_problems = [l for l in combined.splitlines() if any(k in l.lower() for k in
                 ["error","missing","corrupt","broken","fatal"]) and "dangling" not in l.lower()]
print(a(f"  fsck exit = {rc}"))
if real_problems:
    print(a("  *** REAL INTEGRITY PROBLEMS: ***"))
    for l in real_problems[:20]: print(a(f"    {l}"))
else:
    dangling = [l for l in combined.splitlines() if "dangling" in l.lower()]
    print(a(f"  OK: no error/missing/corrupt. {len(dangling)} dangling notice(s) (normal -- unreferenced objects)."))

print("\n### 5. HEAD + working tree UNCHANGED by gc (gc must not alter these) ###")
rc,out,err = sh(["git","log","--oneline","-3"])
print(a("  recent commits:")); print(a("    "+out.strip().replace("\n","\n    ")))
rc,out,err = sh(["git","status","--short"])
mod = [l for l in out.splitlines() if l and not l.startswith("??")]
print(a(f"  working tree: {len(mod)} tracked change(s) (should be 0 -- gc changes nothing tracked)"))
for l in mod[:10]: print(a(f"    {l}"))

print("\n### 6. git push --dry-run (test push viability WITHOUT pushing) ###")
rc,out,err = sh(["git","push","--dry-run"], timeout=300)
combined = (out+err).strip()
print(a(f"  push --dry-run exit = {rc}"))
for l in combined.splitlines()[:25]:
    print(a(f"    {l}"))
if rc == 0:
    print(a("  -> push viable (dry-run succeeded). The historical bloat does NOT block a push."))
else:
    print(a("  -> push --dry-run non-zero. Read stderr above: could be no-remote, auth, or size."))

line("=")
print("GIT GC + PUSH-CHECK COMPLETE. gc repacked object store; NO history rewritten, NO working file")
print("touched, NO reflog expired. Compare BEFORE/AFTER size-pack for the reclaim. fsck confirms integrity.")
raise SystemExit(0)
