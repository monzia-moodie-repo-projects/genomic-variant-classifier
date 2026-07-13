#!/usr/bin/env python
"""audit_git_detail.py (2026-07-11) -- READ-ONLY git detail audit before the housekeeping commit:
(1) the exact names of modified (tracked) files, (2) the .gitignore contents, (3) whether the
install_*/scratch/outputs paths are ignored (git check-ignore), (4) the largest files ACTUALLY
TRACKED in git (git ls-files) to diagnose the 1.8 GB .git bloat. DELETES NOTHING, commits nothing.
ASCII-safe.
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (r.stdout or "") + (("\n[stderr] "+r.stderr) if r.returncode!=0 and r.stderr else "")
    except Exception as e:
        return f"(failed: {e})"

def human(n):
    for u in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def main() -> int:
    print("="*78); print("GIT DETAIL AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

    print("\n### 1. MODIFIED (tracked) files -- the commit scope ###")
    print(a(sh(["git","status","--short","--untracked-files=no"]).strip() or "(none)"))

    print("\n### 2. staged vs unstaged summary ###")
    print(a(sh(["git","diff","--stat"]).strip()[:1500] or "(no unstaged diff)"))

    print("\n### 3. .gitignore contents ###")
    gi = Path(".gitignore")
    if gi.exists():
        print(a(gi.read_text(encoding="utf-8", errors="replace")[:3000]))
    else:
        print("  (.gitignore MISSING)")

    print("\n### 4. are install_*/scratch/outputs IGNORED? (git check-ignore) ###")
    for probe in ["install_audit_housekeeping.py", "install_verify_w2b2.py",
                  "outputs/housekeeping_audit.txt", "scripts/dump_trainpy.py"]:
        r = sh(["git","check-ignore","-v", probe]).strip()
        print(a(f"  {probe}: {'IGNORED by ['+r+']' if r else 'NOT ignored'}"))

    print("\n### 5. LARGEST FILES TRACKED IN GIT (diagnoses .git bloat) ###")
    # git ls-files gives tracked paths; stat each for size
    tracked = sh(["git","ls-files"]).splitlines()
    print(a(f"  {len(tracked)} tracked files total"))
    sized = []
    for t in tracked:
        p = Path(t)
        try:
            if p.is_file():
                sized.append((p.stat().st_size, t))
        except Exception:
            pass
    sized.sort(reverse=True)
    print("  top 25 tracked by CURRENT on-disk size:")
    for s,t in sized[:25]:
        print(a(f"    {human(s):>9}  {t}"))
    big_tracked = [x for x in sized if x[0] > 50*1024*1024]
    print(a(f"  -> {len(big_tracked)} TRACKED file(s) over 50 MB (these bloat .git)"))

    print("\n### 6. git history pack size + object count ###")
    print(a(sh(["git","count-objects","-vH"]).strip()))

    line("=")
    print("GIT DETAIL AUDIT COMPLETE. Read-only. Nothing changed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
