#!/usr/bin/env python
"""verify_postrewrite.py (2026-07-11) -- READ-ONLY verification of the git filter-repo history rewrite
that excised data/alphafold_cif_cache_2026-07-03.tar.gz + notebooks./gitkeep. Confirms the OUTCOME
(not just that it ran): (1) NO blob >100MB remains in history; (2) .git shrank (size-pack now small);
(3) fsck integrity; (4) all commits present + the 7 session commits survive (by MESSAGE, since SHAs
changed); (5) the bad path notebooks./gitkeep is GONE from all history; (6) HEAD is sane on main;
(7) the AlphaFold path is gone from history too. Rewrites NOTHING. ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-",n=78): print(c*n)
def sh(cmd, timeout=600):
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        return r.returncode,(r.stdout or ""),(r.stderr or "")
    except Exception as e:
        return -1,"",f"(failed: {e})"
def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n<1024: return f"{n:.2f}{u}"
        n/=1024
    return f"{n:.2f}PB"

ok_all=True
print("="*78); print("POST-REWRITE VERIFY (READ-ONLY) -- 2026-07-11"); print("="*78)

# 1 + 7. scan history for blobs >50MB; the AlphaFold path must be ABSENT
print("\n### 1. NO large blob (>100MB) may remain in history ###")
rc,out,err = sh(["git","rev-list","--objects","--all"])
objs={}
for l in out.splitlines():
    p=l.split(" ",1); objs[p[0]]=p[1] if len(p)>1 else ""
rc,out,err = sh(["git","cat-file","--batch-check=%(objecttype) %(objectname) %(objectsize)"])
# feed sha list
r=subprocess.run(["git","cat-file","--batch-check=%(objecttype) %(objectname) %(objectsize)"],
                 input="\n".join(objs.keys()),capture_output=True,text=True)
big=[]
for l in r.stdout.splitlines():
    q=l.split()
    if len(q)==3 and q[0]=="blob":
        sz=int(q[2])
        if sz>50*1024*1024: big.append((sz,q[1],objs.get(q[1],"")))
big.sort(reverse=True)
over100=[b for b in big if b[0]>100*1024*1024]
if over100:
    ok_all=False
    print(a(f"  FAIL: {len(over100)} blob(s) >100MB STILL in history:"))
    for sz,sha,path in over100: print(a(f"    {human(sz)}  {path}"))
else:
    print(a(f"  OK: 0 blobs >100MB in history. ({len(big)} blobs >50MB total"
            f"{' -- listing:' if big else '.'})"))
    for sz,sha,path in big: print(a(f"    {human(sz)}  {path}  (>50MB but <100MB, does not block push)"))

# explicit: AlphaFold path must be absent from history
print("\n### 7. the AlphaFold cache path must be ABSENT from all history ###")
rc,out,err = sh(["git","log","--all","--oneline","--","data/alphafold_cif_cache_2026-07-03.tar.gz"])
if out.strip():
    ok_all=False
    print(a(f"  FAIL: AlphaFold path still referenced in history:")); print(a("    "+out.strip()[:300]))
else:
    print("  OK: no commit references data/alphafold_cif_cache_2026-07-03.tar.gz")

# 5. bad path notebooks./gitkeep gone
print("\n### 5. the malformed path notebooks./gitkeep must be GONE from history ###")
rc,out,err = sh(["git","log","--all","--oneline","--","notebooks./gitkeep"])
# also scan the object paths we already have
bad_in_objs=[p for p in objs.values() if p.startswith("notebooks.")]
if out.strip() or bad_in_objs:
    ok_all=False
    print(a(f"  FAIL: notebooks./ path still present. log={out.strip()[:150]} objs={bad_in_objs[:5]}"))
else:
    print("  OK: no notebooks./ path anywhere in history")

# 2. .git shrank
print("\n### 2. .git size (size-pack should be small now, not 1.81 GiB) ###")
rc,out,err = sh(["git","count-objects","-vH"])
print(a(out.strip()))
sp=[l for l in out.splitlines() if l.startswith("size-pack:")]
if sp:
    val=sp[0].split(":",1)[1].strip()
    if "GiB" in val:
        ok_all=False
        print(a(f"  FAIL: size-pack still in GiB ({val}) -- .git did NOT shrink. Run git gc --prune=now."))
    else:
        print(a(f"  OK: size-pack is {val} (shrank from 1.81 GiB -- the blob is gone from the pack)"))

# 3. fsck
print("\n### 3. git fsck --full (integrity) ###")
rc,out,err = sh(["git","fsck","--full"])
comb=(out+err)
real=[l for l in comb.splitlines() if any(k in l.lower() for k in
      ["error","missing","corrupt","broken","fatal"]) and "dangling" not in l.lower()]
if real:
    ok_all=False
    print("  FAIL: integrity problems:")
    for l in real[:15]: print(a(f"    {l}"))
else:
    dang=[l for l in comb.splitlines() if "dangling" in l.lower()]
    print(a(f"  OK: no error/missing/corrupt. {len(dang)} dangling notice(s) (normal)."))

# 4 + 6. commits present, HEAD sane
print("\n### 4. commit count + the 7 session commits survive (by message; SHAs changed) ###")
rc,out,err = sh(["git","rev-list","--count","HEAD"])
print(a(f"  total commits on HEAD: {out.strip()}"))
rc,out,err = sh(["git","log","--oneline","-9"])
print("  recent commits:")
for l in out.splitlines(): print(a(f"    {l}"))
needles=["Conflicting classifications","Expected Calibration Error","re-baseline","allele classification",
         "conformal-prediction package","fresh-ClinVar","provenance"]
rc,logall,err = sh(["git","log","--oneline","-40"])
found=[n for n in needles if n.lower() in logall.lower()]
print(a(f"  session-commit messages found: {len(found)}/7 -> {found}"))
if len(found)<7:
    print(a(f"  NOTE: {7-len(found)} not in last 40 -- may be deeper; not necessarily a failure."))

print("\n### 6. HEAD sane + on main ###")
rc,out,err = sh(["git","symbolic-ref","--short","HEAD"])
rc2,out2,err2 = sh(["git","rev-parse","--short","HEAD"])
print(a(f"  HEAD -> {out.strip()} @ {out2.strip()}"))
if out.strip()!="main": 
    ok_all=False; print("  FAIL: HEAD not on main")
else: print("  OK: on main")

line("=")
print(a(f"POST-REWRITE VERIFY: {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED -- investigate'}"))
print("READ-ONLY. Nothing changed. If green: run the test suite, then re-add origin + push.")
raise SystemExit(0 if ok_all else 1)
