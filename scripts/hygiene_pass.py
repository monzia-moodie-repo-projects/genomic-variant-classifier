#!/usr/bin/env python
"""hygiene_pass.py (2026-07-11) -- conservative project-tree cleanup after the sequence-window work.

Removes ONLY the genuinely-safe test artifact (the dry-run directory). Everything with rollback or
resume value (the ECE-fix backup, the precompute part files) is PRESERVED and instead added to
.gitignore so it stays out of version control until the retrain/commit makes it safe to delete.
Reports sizes before acting; never performs a blind recursive delete. ASCII-safe.
"""
from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


GITIGNORE_ENTRIES = [
    "*.bak",
    "install_*.py",
    "data/processed/seq_windows_dryrun/",
    "data/processed/seq_windows/part_*.parquet",
    "data/processed/seq_windows/part_*.done",
    "data/processed/seq_windows/*.tmp.parquet",
]


def main() -> int:
    print("=" * 78)
    print("HYGIENE PASS (conservative -- deletes only the dry-run test artifact)")
    print("=" * 78)

    # 1. REPORT everything first.
    dryrun = Path("data/processed/seq_windows_dryrun")
    bak = Path("src/genomic_variant_classifier/evaluation/evaluator.py.bak")
    parts = sorted(Path("data/processed/seq_windows").glob("part_*.parquet"))
    dones = sorted(Path("data/processed/seq_windows").glob("part_*.done"))
    merged = Path("data/processed/seq_windows/seq_windows.parquet")
    installs = sorted(Path(".").glob("install_*.py"))

    print("inventory:")
    print(_ascii_safe(f"  dry-run dir (REMOVE): {dryrun}  {dir_size(dryrun)/1e6:.1f} MB, "
                      f"exists={dryrun.exists()}"))
    print(_ascii_safe(f"  merged artifact (KEEP): {merged}  {dir_size(merged)/1e6:.1f} MB, "
                      f"exists={merged.exists()}"))
    print(_ascii_safe(f"  part files (KEEP, resume safety): {len(parts)} parquet + {len(dones)} done"))
    print(_ascii_safe(f"  ECE backup (KEEP until commit): {bak}  "
                      f"{dir_size(bak)/1e3:.1f} KB, exists={bak.exists()}"))
    print(_ascii_safe(f"  install_*.py in root (KEEP, gitignore): {len(installs)} files"))
    line()

    # 2. Remove ONLY the dry-run directory.
    if dryrun.exists():
        shutil.rmtree(dryrun)
        print(_ascii_safe(f"REMOVED: {dryrun} (dry-run test artifact, no pipeline role)"))
    else:
        print("dry-run dir already absent -- nothing to remove.")
    line()

    # 3. Idempotently ensure .gitignore entries.
    gi = Path(".gitignore")
    existing = gi.read_text().splitlines() if gi.exists() else []
    existing_set = set(x.strip() for x in existing)
    added = []
    to_write = list(existing)
    if existing and existing[-1].strip() != "":
        to_write.append("")  # spacer
    header = "# seq-window build scaffolding + backups (added by hygiene_pass 2026-07-11)"
    if header not in existing_set:
        to_write.append(header)
    for e in GITIGNORE_ENTRIES:
        if e not in existing_set:
            to_write.append(e)
            added.append(e)
    if added:
        gi.write_text("\n".join(to_write) + "\n")
        print("added to .gitignore:")
        for e in added:
            print(_ascii_safe(f"  {e}"))
    else:
        print(".gitignore already has all entries -- no change.")
    line()

    # 4. Report retained items with rationale.
    print("RETAINED (intentionally, not deleted):")
    print("  - evaluator.py.bak: rollback for the ECE fix; remove AFTER git commit.")
    print("  - seq_windows/part_*.parquet + .done: resume safety; remove AFTER the retrain")
    print("    successfully consumes seq_windows.parquet.")
    print("  - install_*.py: transient transfer installers; gitignored, remove at leisure.")
    line("=")
    print("HYGIENE PASS COMPLETE. Only the dry-run test artifact was deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
