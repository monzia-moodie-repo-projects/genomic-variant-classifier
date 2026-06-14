#!/usr/bin/env python3
"""preflight_check_data_paths.py -- Monzia Moodie

Fail-fast guard for the run gates. Verifies that the repo's data directories are REAL, traversable, writable
directories -- not dangling junctions/symlinks (the failure mode that turned data/ -> G:\\My Drive\\...\\data
into 20 fail-loud test errors on 2026-06-14), not stray files, not missing -- and (optionally) that named
critical data assets are present (else the connectors silent-stub to defaults -> degenerate features).

Distinguishes a DANGLING reparse point (reparse point present, target gone) from a plain missing path, and
prints the exact remediation, so a launch preflight or the operator gets an actionable stop rather than a
generic "file not found".

Exit codes:  0 = all OK   2 = a directory-health problem   3 = a missing critical asset (only with --asset)

Usage (run from the repo root):
    python scripts/preflight_check_data_paths.py
    python scripts/preflight_check_data_paths.py --dir data --dir outputs \
        --asset data/external/spliceai/spliceai_index.parquet
    python scripts/preflight_check_data_paths.py --no-write-check     # read-only environments
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REMEDY_DANGLING = (
    "remove the link without touching its target -- cmd /c rmdir \"{p}\" (NEVER rmdir /s / Remove-Item "
    "-Recurse on a junction) -- then restore it: git checkout -- {p}/ , or re-point it to a MOUNTED target, "
    "or robocopy the real contents into a plain local {p}\\."
)


def check_path_health(path: str, *, must_be_writable: bool = True) -> tuple[str, str]:
    """Classify a path. Returns (status, message). status in:
    ok | missing | dangling | not_a_dir | untraversable | not_writable.
    """
    p = Path(path)
    # lexists() sees the link itself; exists() resolves the target. A reparse point whose target is gone
    # is lexists() True / exists() False -- this is the dangling-junction/broken-symlink signature.
    if not os.path.lexists(path):
        return ("missing", f"{path} does not exist (expected a directory).")
    if os.path.lexists(path) and not os.path.exists(path):
        return ("dangling", f"{path} is a DANGLING symlink/junction (reparse point present, target missing); "
                            + _REMEDY_DANGLING.format(p=path))
    if not p.is_dir():
        return ("not_a_dir", f"{path} exists but is NOT a directory (stray file shadowing it). Remove/rename it "
                             f"and restore {path}/.")
    try:
        next(iter(os.scandir(path)), None)  # force a directory read (catches some reparse failures)
    except OSError as e:
        return ("untraversable", f"{path} is a directory but cannot be listed ({e}).")
    if must_be_writable:
        probe = p / f".preflight_write_test_{os.getpid()}"
        try:
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError as e:
            return ("not_writable", f"{path} is a directory but is NOT writable ({e}).")
    return ("ok", f"{path} OK (traversable, writable directory).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight: data-path health + critical-asset presence.")
    ap.add_argument("--dir", action="append", dest="dirs", default=None,
                    help="Directory to health-check (repeatable). Default: data, outputs.")
    ap.add_argument("--asset", action="append", dest="assets", default=[],
                    help="Critical file that must exist (repeatable). Missing -> exit 3.")
    ap.add_argument("--no-write-check", action="store_true", help="Skip the writability probe.")
    ns = ap.parse_args(argv)
    dirs = ns.dirs if ns.dirs else ["data", "outputs"]

    path_problem = False
    print("== directory health ==")
    for d in dirs:
        status, msg = check_path_health(d, must_be_writable=not ns.no_write_check)
        print(f"  [{'OK ' if status == 'ok' else 'FAIL'}] {status:13s} {msg}")
        path_problem |= (status != "ok")

    missing_assets = []
    if ns.assets:
        print("== critical assets ==")
        for a in ns.assets:
            present = Path(a).exists()
            print(f"  [{'OK ' if present else 'FAIL'}] {'present' if present else 'MISSING':13s} {a}")
            if not present:
                missing_assets.append(a)

    if path_problem:
        print("\nPREFLIGHT FAILED: a data directory is dangling/missing/not-a-dir/not-writable (exit 2).")
        return 2
    if missing_assets:
        print(f"\nPREFLIGHT FAILED: {len(missing_assets)} critical asset(s) missing -> connectors would "
              f"silent-stub to defaults (exit 3). Re-hydrate before running.")
        return 3
    print("\nPREFLIGHT OK: data paths healthy" + (" and all critical assets present." if ns.assets else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
