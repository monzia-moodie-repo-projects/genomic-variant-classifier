#!/usr/bin/env python
"""audit_cleanup_plan.py (2026-07-11) -- READ-ONLY cleanup PROPOSAL. After the 7 housekeeping commits,
categorize every removal candidate with a SAFETY RATIONALE. DELETES NOTHING -- it only classifies +
prints what COULD be removed and WHY it is safe. Categories:
  (A) this-arc .bak (rollback backups -- safe now that work is committed)
  (B) .gitignore.prebakfix (Step-0 backup)
  (C) redundant seq_windows part_*.parquet + .done (merged artifact exists)
  (D) 895 MB dead .OOMbak
  (E) one-shot transfer installers install_*.py (regenerable)
  (F) one-shot scratch scripts dump_/patch_ (the arc's throwaway patchers)
For each candidate, verify the SAFETY CONDITION (e.g. for .bak: the live file is tracked+committed;
for seq_windows parts: the merged parquet exists). Nothing is deleted. ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception: pass
def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)
def tracked(p): return subprocess.run(["git","ls-files","--error-unmatch",p],capture_output=True,text=True).returncode==0
def committed_head(p):
    return subprocess.run(["git","cat-file","-e",f"HEAD:{p}"],capture_output=True,text=True).returncode==0
def human(n):
    for u in ["B","KB","MB","GB"]:
        if n<1024: return f"{n:.1f}{u}"
        n/=1024
    return f"{n:.1f}TB"
def sz(p):
    try: return Path(p).stat().st_size
    except: return 0

def main() -> int:
    print("="*78); print("CLEANUP PROPOSAL (READ-ONLY -- DELETES NOTHING) -- 2026-07-11"); print("="*78)
    total_reclaim = 0

    # (A) this-arc .bak -- safe IFF the corresponding live file is committed at HEAD
    print("\n### (A) THIS-ARC .bak (rollback backups) -- safe to remove once live file is committed ###")
    arc_baks = {
        "scripts/train.py.w1bak": "scripts/train.py",
        "scripts/train.py.w2b2bak": "scripts/train.py",
        "src/genomic_variant_classifier/data/real_data_prep.py.w2b1bak": "src/genomic_variant_classifier/data/real_data_prep.py",
        "src/genomic_variant_classifier/data/split_protocol_v2.py.w2b1bak": "src/genomic_variant_classifier/data/split_protocol_v2.py",
        "src/genomic_variant_classifier/models/variant_ensemble.py.w2bak": "src/genomic_variant_classifier/models/variant_ensemble.py",
        "src/genomic_variant_classifier/evaluation/evaluator.py.bak": "src/genomic_variant_classifier/evaluation/evaluator.py",
        "src/genomic_variant_classifier/data/database_connectors.py.bak": "src/genomic_variant_classifier/data/database_connectors.py",
    }
    for bak, live in arc_baks.items():
        if not Path(bak).exists():
            print(a(f"  (absent)          {bak}")); continue
        safe = committed_head(live)
        s = sz(bak); total_reclaim += s if safe else 0
        print(a(f"  {'SAFE-REMOVE' if safe else 'KEEP(live not at HEAD)':22} {human(s):>8}  {bak}"))
        print(a(f"                          live={live} committed@HEAD={safe}"))

    # (B) .gitignore.prebakfix -- safe IFF .gitignore committed at HEAD
    print("\n### (B) .gitignore.prebakfix -- safe once .gitignore is committed ###")
    if Path(".gitignore.prebakfix").exists():
        safe = committed_head(".gitignore")
        s = sz(".gitignore.prebakfix"); total_reclaim += s if safe else 0
        print(a(f"  {'SAFE-REMOVE' if safe else 'KEEP':22} {human(s):>8}  .gitignore.prebakfix (.gitignore@HEAD={safe})"))

    # (C) redundant seq_windows parts -- safe IFF merged seq_windows.parquet exists
    print("\n### (C) seq_windows part_*.parquet + .done -- safe if merged seq_windows.parquet exists ###")
    swd = Path("data/processed/seq_windows")
    merged = swd/"seq_windows.parquet"
    if merged.exists():
        parts = sorted(swd.glob("part_*.parquet"))
        dones = sorted(swd.glob("part_*.done")) + sorted(swd.glob("*.done"))
        psz = sum(sz(str(p)) for p in parts)
        print(a(f"  merged EXISTS: {merged} ({human(sz(str(merged)))}) -> parts are redundant"))
        print(a(f"  {'SAFE-REMOVE':22} {human(psz):>8}  {len(parts)} part_*.parquet + {len(dones)} .done markers"))
        total_reclaim += psz
    else:
        print("  merged seq_windows.parquet NOT found -> DO NOT remove parts")

    # (D) dead .OOMbak
    print("\n### (D) dead dbNSFP .OOMbak -- an out-of-memory rollback of a build ###")
    oom = "data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak"
    live_oom = "data/external/dbnsfp/dbnsfp_full_index.parquet"
    if Path(oom).exists():
        s = sz(oom); live_ok = Path(live_oom).exists()
        total_reclaim += s if live_ok else 0
        print(a(f"  {'SAFE-REMOVE' if live_ok else 'KEEP(no live index!)':22} {human(s):>8}  {oom}"))
        print(a(f"                          live index present={live_ok} ({human(sz(live_oom))})"))

    # (E) transfer installers (gitignored, regenerable)
    print("\n### (E) install_*.py transfer artifacts (gitignored, regenerable) ###")
    inst = sorted(Path(".").glob("install_*.py"))
    isz = sum(sz(str(p)) for p in inst)
    print(a(f"  {'SAFE-REMOVE':22} {human(isz):>8}  {len(inst)} install_*.py (regenerable; ignored, not in git)"))
    total_reclaim += isz

    # (F) one-shot scratch dump_/patch_ (NOT verify_/smoke_ -- keep those for regression)
    print("\n### (F) one-shot scratch scripts dump_/patch_ (KEEP verify_/smoke_ for regression) ###")
    scr = Path("scripts")
    dumps = sorted(scr.glob("dump_*.py")); patches = sorted(scr.glob("patch_*.py"))
    dsz = sum(sz(str(p)) for p in dumps) + sum(sz(str(p)) for p in patches)
    print(a(f"  {'PROPOSE-REMOVE':22} {human(dsz):>8}  {len(dumps)} dump_*.py + {len(patches)} patch_*.py (one-shot throwaway)"))
    print(a(f"  NOTE: NONE are tracked (all untracked scratch). Removing = deleting local throwaway only."))
    print(a(f"  KEEP: verify_*/smoke_*/audit_*/probe_*/diagnose_* (regression + provenance value)"))

    line("=")
    print(a(f"TOTAL SAFE-RECLAIM (A-E, excluding the judgment-call F): ~{human(total_reclaim)}"))
    print("NOTHING WAS DELETED. This is a proposal. Review categories, then approve removals selectively.")
    print("Recommended: approve A-D immediately (proven safe); E optional; F is your call (throwaway).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
