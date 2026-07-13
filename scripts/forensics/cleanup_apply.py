#!/usr/bin/env python
"""cleanup_apply.py (2026-07-11) -- GUARDED cleanup executor. TWO-PHASE: default = DRY-RUN (lists what
it WOULD delete, each safety condition RE-VERIFIED live); with '--apply' it deletes. Every item is
re-checked at delete time (not trusting any prior snapshot); anything tracked in git is SKIPPED; every
removal is logged to outputs/cleanup_deleted.txt with size. Categories A-F from the cleanup proposal.
ASCII-safe. Refuses to delete a tracked path under any circumstance.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)
APPLY = "--apply" in sys.argv

def tracked(p):
    return subprocess.run(["git","ls-files","--error-unmatch",p],capture_output=True,text=True).returncode==0
def committed_head(p):
    return subprocess.run(["git","cat-file","-e",f"HEAD:{p}"],capture_output=True,text=True).returncode==0
def ignored(p):
    return subprocess.run(["git","check-ignore",p],capture_output=True,text=True).returncode==0
def human(n):
    for u in ["B","KB","MB","GB"]:
        if n<1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"
def sz(p):
    try: return Path(p).stat().st_size
    except: return 0

plan = []   # (category, path, reason_ok:bool, why)

# (A) this-arc .bak -- ok iff live committed@HEAD
A = {
 "scripts/train.py.w1bak":"scripts/train.py",
 "scripts/train.py.w2b2bak":"scripts/train.py",
 "src/genomic_variant_classifier/data/real_data_prep.py.w2b1bak":"src/genomic_variant_classifier/data/real_data_prep.py",
 "src/genomic_variant_classifier/data/split_protocol_v2.py.w2b1bak":"src/genomic_variant_classifier/data/split_protocol_v2.py",
 "src/genomic_variant_classifier/models/variant_ensemble.py.w2bak":"src/genomic_variant_classifier/models/variant_ensemble.py",
 "src/genomic_variant_classifier/evaluation/evaluator.py.bak":"src/genomic_variant_classifier/evaluation/evaluator.py",
 "src/genomic_variant_classifier/data/database_connectors.py.bak":"src/genomic_variant_classifier/data/database_connectors.py",
}
for bak,live in A.items():
    if Path(bak).exists():
        ok = committed_head(live) and not tracked(bak)
        plan.append(("A",bak,ok,f"live committed@HEAD={committed_head(live)}, bak tracked={tracked(bak)}"))

# (B) .gitignore.prebakfix -- ok iff .gitignore committed
if Path(".gitignore.prebakfix").exists():
    ok = committed_head(".gitignore") and not tracked(".gitignore.prebakfix")
    plan.append(("B",".gitignore.prebakfix",ok,f".gitignore@HEAD={committed_head('.gitignore')}"))

# (C) seq_windows parts + .done -- ok iff merged exists + non-empty
swd = Path("data/processed/seq_windows")
merged = swd/"seq_windows.parquet"
merged_ok = merged.exists() and sz(str(merged))>0
if swd.exists():
    for p in sorted(swd.glob("part_*.parquet")) + sorted(swd.glob("*.done")):
        ok = merged_ok and not tracked(str(p))
        plan.append(("C",str(p).replace("\\","/"),ok,f"merged_ok={merged_ok}"))

# (D) dead .OOMbak -- ok iff live index exists + non-empty
oom="data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak"
live="data/external/dbnsfp/dbnsfp_full_index.parquet"
if Path(oom).exists():
    ok = Path(live).exists() and sz(live)>0 and not tracked(oom)
    plan.append(("D",oom,ok,f"live index present={Path(live).exists()} size={human(sz(live))}"))

# (E) root install_*.py -- ok iff ignored + not tracked
for p in sorted(Path(".").glob("install_*.py")):
    ok = ignored(str(p)) and not tracked(str(p))
    plan.append(("E",str(p).replace("\\","/"),ok,f"ignored={ignored(str(p))}"))

# (F) scripts/dump_*.py + scripts/patch_*.py -- ok iff untracked
for pat in ["dump_*.py","patch_*.py"]:
    for p in sorted(Path("scripts").glob(pat)):
        ok = not tracked(str(p))
        plan.append(("F",str(p).replace("\\","/"),ok,f"untracked={not tracked(str(p))}"))

print("="*78)
print(a(f"GUARDED CLEANUP -- {'APPLY (deleting)' if APPLY else 'DRY-RUN (nothing deleted; pass --apply to delete)'} -- 2026-07-11"))
print("="*78)

by_cat = {}
skipped = []
for cat,p,ok,why in plan:
    by_cat.setdefault(cat,{"ok":[],"skip":[]})
    (by_cat[cat]["ok"] if ok else by_cat[cat]["skip"]).append((p,why))
    if not ok: skipped.append((cat,p,why))

total = 0; ndel = 0
deleted_log = []
for cat in "ABCDEF":
    if cat not in by_cat: continue
    oks = by_cat[cat]["ok"]; sk = by_cat[cat]["skip"]
    catsz = sum(sz(p) for p,_ in oks)
    print(a(f"\n### ({cat})  {len(oks)} eligible ({human(catsz)}), {len(sk)} SKIPPED ###"))
    for p,why in oks[:6]:
        print(a(f"  {'DELETE' if APPLY else 'would-delete'}  {human(sz(p)):>9}  {p}"))
    if len(oks)>6: print(a(f"  ... +{len(oks)-6} more"))
    for p,why in sk:
        print(a(f"  SKIP (guard failed): {p}  [{why}]"))
    total += catsz
    if APPLY:
        for p,_ in oks:
            try:
                s = sz(p); Path(p).unlink()
                deleted_log.append(f"{human(s):>9}  {p}"); ndel += 1
            except Exception as e:
                print(a(f"  ERROR deleting {p}: {e}"))

line("=")
if APPLY:
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/cleanup_deleted.txt").write_text(
        "CLEANUP DELETED -- 2026-07-11\n"+"\n".join(deleted_log)+f"\n\nTOTAL: {ndel} files, ~{human(total)}\n",
        encoding="utf-8")
    print(a(f"DELETED {ndel} files, reclaimed ~{human(total)}. Log -> outputs/cleanup_deleted.txt"))
else:
    print(a(f"DRY-RUN: would delete {sum(len(by_cat[c]['ok']) for c in by_cat)} files, ~{human(total)}."))
    print("Re-run with '--apply' to delete. Every condition was RE-VERIFIED live above.")
if skipped:
    print(a(f"NOTE: {len(skipped)} item(s) SKIPPED because a guard failed (tracked or condition not met)."))
raise SystemExit(0)
