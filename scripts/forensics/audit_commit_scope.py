#!/usr/bin/env python
"""audit_commit_scope.py (2026-07-11) -- READ-ONLY: resolve the exact commit scope. (1) list the
conformal/ package contents, (2) find the test files for the new machinery + their git-tracked state,
(3) check-ignore status of the backup files (.wNbak, .bak2-6) to confirm the .gitignore gap, (4) show
which of the new-machinery files have matching tests. Commits nothing, deletes nothing. ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)
def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (("\n[rc="+str(r.returncode)+"] "+r.stderr) if r.returncode!=0 and r.stderr else "")
    except Exception as e:
        return f"(failed: {e})"

def tracked(path):
    r = subprocess.run(["git","ls-files","--error-unmatch",path], capture_output=True, text=True)
    return r.returncode == 0

def main() -> int:
    print("="*78); print("COMMIT SCOPE AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

    print("\n### 1. conformal/ PACKAGE CONTENTS ###")
    cdir = Path("src/genomic_variant_classifier/conformal")
    if cdir.exists():
        for p in sorted(cdir.rglob("*")):
            if p.is_file():
                print(a(f"  {'[tracked]' if tracked(str(p)) else '[UNTRACKED]'}  {p}  ({p.stat().st_size} B)"))
    else:
        print("  (conformal/ dir not found)")

    print("\n### 2. NEW-MACHINERY FILES + git-tracked state ###")
    machinery = [
        "src/genomic_variant_classifier/data/split_protocol_v2.py",
        "src/genomic_variant_classifier/data/delta_window_builder.py",
        "src/genomic_variant_classifier/data/seq_window_manifest.py",
        "src/genomic_variant_classifier/data/seq_window_join.py",
        "src/genomic_variant_classifier/data/allele_classify.py",
        "scripts/build_seq_windows.py",
        "scripts/benchmark_seq_windows.py",
        "scripts/hygiene_pass.py",
        "scripts/run_conformal_calibration.py",
    ]
    for m in machinery:
        p = Path(m)
        state = "MISSING"
        if p.exists():
            state = "[tracked]" if tracked(m) else "[UNTRACKED -> needs git add]"
        print(a(f"  {state:32} {m}"))

    print("\n### 3. TEST FILES for the new machinery + tracked state ###")
    tdir = Path("tests")
    if tdir.exists():
        pats = ["*split_protocol*", "*delta_window*", "*seq_window*", "*allele*", "*conformal*", "*ece*"]
        seen = set()
        for pat in pats:
            for p in sorted(tdir.rglob(pat)):
                if p.is_file() and p.suffix == ".py" and p not in seen:
                    seen.add(p)
                    print(a(f"  {'[tracked]' if tracked(str(p)) else '[UNTRACKED]'}  {p}"))
        if not seen:
            print("  (no matching test files found)")

    print("\n### 4. BACKUP-PATTERN .gitignore GAP (check-ignore) ###")
    for bak in ["scripts/train.py.w2b2bak", "scripts/train.py.w1bak",
                "src/genomic_variant_classifier/data/split_protocol_v2.py.w2b1bak",
                "scripts/build_cohort_from_source.py.bak2"]:
        if Path(bak).exists():
            r = sh(["git","check-ignore","-v", bak]).strip()
            print(a(f"  {bak}: {'IGNORED ['+r+']' if r else 'NOT ignored -> gap'}"))

    print("\n### 5. ECE-fix helper scripts (commit 2 scope decision) ###")
    for f in ["scripts/fix_ece_binning.py", "scripts/verify_ece_fix.py"]:
        p = Path(f)
        if p.exists():
            print(a(f"  {'[tracked]' if tracked(f) else '[UNTRACKED]'}  {f}  ({p.stat().st_size} B)"))

    line("=")
    print("COMMIT SCOPE AUDIT COMPLETE. Read-only. Curate the git add list from this.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
