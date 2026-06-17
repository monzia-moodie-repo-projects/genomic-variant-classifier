#!/usr/bin/env python3
"""
scripts/maintenance/sync_data_to_gdrive.py  --  Monzia Moodie

Mirror the LOCAL canonical data/ to the Google Drive backup via rclone, using a
manifest-derived include filter. DRY-RUN by DEFAULT -- it prints the rclone
command and runs `rclone ... --dry-run` unless you pass --execute.

Policy enforced before any call:
  * controlled/licensed sources are NEVER pushed (hard ABORT if one is sync=true)
  * only sources with sync=true (irreplaceable / regenerable_expensive) are sent
  * one-directional: local -> G drive (G drive is the backup mirror, not the live
    store). It uses `rclone copy` (additive) by default, not `sync` (which would
    delete remote extras); pass --mirror for a true delete-extras sync.

Usage:
  python scripts/maintenance/sync_data_to_gdrive.py              # dry-run preview
  python scripts/maintenance/sync_data_to_gdrive.py --execute    # actually copy
  python scripts/maintenance/sync_data_to_gdrive.py --execute --mirror
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--manifest", default="configs/data_manifest.yaml", type=Path)
    ap.add_argument("--filter", default="configs/rclone_data_filter.txt", type=Path)
    ap.add_argument("--execute", action="store_true", help="actually run (default: dry-run)")
    ap.add_argument("--mirror", action="store_true", help="use 'rclone sync' (delete remote extras)")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ABORT] manifest not found: {args.manifest}")
        return 2
    man = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    sources = man.get("sources", {})
    remote = man.get("gdrive", {}).get("remote", "genvarcla:")
    base = man.get("gdrive", {}).get("base", "")

    # compliance gate
    bad = [s for s, m in sources.items() if m.get("sync") and m.get("tier") == "controlled"]
    if bad:
        print(f"[ABORT] controlled sources marked sync=true: {bad}. Refusing to push.")
        return 2
    if not args.filter.exists():
        print(f"[ABORT] filter not found: {args.filter}. Run setup_data_tree.py first.")
        return 2
    if shutil.which("rclone") is None:
        print("[ABORT] rclone not on PATH. Install rclone and configure the "
              f"'{remote.rstrip(':')}' remote (cloud.google -> rclone config).")
        return 2

    dest = f"{remote}{base}"
    verb = "sync" if args.mirror else "copy"
    cmd = ["rclone", verb, str(args.data_dir) + "/", dest,
           "--filter-from", str(args.filter), "--progress", "--track-renames"]
    if not args.execute:
        cmd.append("--dry-run")

    included = [f"{m.get('location','external')}/{s}" for s, m in sorted(sources.items())
                if m.get("sync") and m.get("tier") != "controlled"]
    print(f" rclone {verb} (mirror={args.mirror}) {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f" included sources ({len(included)}): {', '.join(included)}")
    print(" command: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    print(f" [done] rclone exit {rc}" + ("" if args.execute else "  (dry-run; re-run with --execute)"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
