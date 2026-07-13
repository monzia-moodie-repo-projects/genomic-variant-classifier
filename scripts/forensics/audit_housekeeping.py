#!/usr/bin/env python
"""audit_housekeeping.py (2026-07-11) -- READ-ONLY inventory of the repo for the housekeeping pass.
Reports: .bak backup files, install_*.py transfer artifacts, scratch scripts (dump/patch/verify/smoke)
in scripts/, git status + recent log, working-tree + .git sizes, large files (>50 MB, e.g. AlphaFold
blob), and seq_windows part files. DELETES NOTHING. ASCII-safe.
"""
from __future__ import annotations
import io
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def a(s): return s.encode("ascii", "replace").decode("ascii")
def line(c="-", n=78): print(c * n)


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"(command failed: {e})"


def sz(p: Path) -> int:
    try:
        return p.stat().st_size
    except Exception:
        return 0


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    print("=" * 78)
    print("HOUSEKEEPING AUDIT (READ-ONLY -- deletes nothing) -- 2026-07-11")
    print("=" * 78)

    root = Path(".")

    # 1. .bak backup files
    print("\n### 1. BACKUP FILES (.bak / .w*bak) -- rollback safety, remove ONLY post-commit ###")
    baks = []
    for pat in ["*.bak", "*bak"]:
        for p in root.rglob(pat):
            if ".git" in p.parts or "node_modules" in p.parts:
                continue
            if p.suffix == ".py" or p.name.endswith("bak"):
                baks.append(p)
    baks = sorted(set(baks), key=lambda x: str(x))
    for p in baks:
        print(a(f"  {human(sz(p)):>9}  {p}"))
    print(a(f"  -> {len(baks)} backup file(s)"))

    # 2. install_*.py transfer artifacts
    print("\n### 2. install_*.py TRANSFER ARTIFACTS (gitignored, regenerable) ###")
    installs = sorted(root.glob("install_*.py"))
    total_i = sum(sz(p) for p in installs)
    print(a(f"  {len(installs)} install_*.py in repo root, total {human(total_i)}"))
    if installs:
        print(a(f"    e.g. {installs[0].name} ... {installs[-1].name}"))

    # 3. scratch scripts in scripts/
    print("\n### 3. SCRATCH SCRIPTS in scripts/ (dump/patch/verify/smoke this arc) ###")
    scr = Path("scripts")
    if scr.exists():
        cats = {"dump_": [], "patch_": [], "verify_": [], "smoke_": []}
        for p in sorted(scr.glob("*.py")):
            for k in cats:
                if p.name.startswith(k):
                    cats[k].append(p.name)
        for k, v in cats.items():
            print(a(f"  {k+'*':10} ({len(v)}): {', '.join(v) if v else '(none)'}"))
    else:
        print("  (no scripts/ dir)")

    # 4. git status + recent log
    print("\n### 4. GIT STATUS (short) ###")
    st = sh(["git", "status", "--short"])
    st_lines = st.splitlines()
    print(a(f"  {len(st_lines)} changed/untracked entries"))
    # summarize by status code
    from collections import Counter
    codes = Counter(l[:2] for l in st_lines if l)
    for code, cnt in codes.most_common():
        print(a(f"    '{code}': {cnt}"))
    print("\n### recent commits (last 5) ###")
    print(a(sh(["git", "log", "--oneline", "-5"]).strip()))
    print("\n### current branch ###")
    print(a(sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()))

    # 5. sizes
    print("\n### 5. SIZES ###")
    git_dir = Path(".git")
    if git_dir.exists():
        gsz = sum(sz(p) for p in git_dir.rglob("*") if p.is_file())
        print(a(f"  .git dir: {human(gsz)}"))

    # 6. large files (>50MB) outside .git/.venv
    print("\n### 6. LARGE FILES (>50 MB, e.g. AlphaFold blob) ###")
    big = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts or ".venv" in p.parts or ".venv312" in p.parts:
            continue
        s = sz(p)
        if s > 50 * 1024 * 1024:
            big.append((s, p))
    for s, p in sorted(big, reverse=True)[:25]:
        print(a(f"  {human(s):>9}  {p}"))
    print(a(f"  -> {len(big)} file(s) over 50 MB"))

    # 7. seq_windows part files
    print("\n### 7. seq_windows INTERMEDIATE PARTS (removable post-merge) ###")
    swdir = Path("data/processed/seq_windows")
    if swdir.exists():
        parts = sorted(swdir.glob("part_*.parquet"))
        dones = sorted(swdir.glob("*.done"))
        merged = swdir / "seq_windows.parquet"
        print(a(f"  part_*.parquet: {len(parts)}  ({human(sum(sz(p) for p in parts))})"))
        print(a(f"  *.done markers: {len(dones)}"))
        print(a(f"  merged seq_windows.parquet: {'EXISTS ' + human(sz(merged)) if merged.exists() else 'MISSING'}"))
    else:
        print("  (no seq_windows dir)")

    line("=")
    print("AUDIT COMPLETE. Nothing was deleted. Review, then decide commit + cleanup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
