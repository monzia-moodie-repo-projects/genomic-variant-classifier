#!/usr/bin/env python3
"""
scripts/maintenance/audit_data_tree.py  --  Monzia Moodie

READ-ONLY inventory + hygiene audit of the data/ tree against
configs/data_manifest.yaml. Never creates, moves, renames, or deletes anything.

Reports, per the DATA_LAYOUT_STANDARD:
  * data/ link status   -- real dir (goal), working junction (dangling-risk),
                           DANGLING junction/symlink, shadow non-dir file, missing
  * per source          -- present? size, files, newest mtime, class, tier, sync
  * orphans             -- on-disk dirs not in the manifest (+ alias -> rename hint)
  * missing             -- manifest sources absent on disk
  * naming hygiene      -- uppercase/space/hyphen/version-suffix/alias dirs
  * compliance          -- any tier=controlled source flagged sync=true (BLOCK)
  * rollup              -- bytes by class: must-back-up vs regenerable

Exit code: 0 clean, 1 warnings, 2 compliance violation (controlled marked sync).

Usage:
  python scripts/maintenance/audit_data_tree.py
  python scripts/maintenance/audit_data_tree.py --data-dir data --manifest configs/data_manifest.yaml --json report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

_BACKUP_CLASSES = {"irreplaceable", "regenerable_expensive"}
_KNOWN_SUBTREES = {"external", "raw", "processed", "reference", "interim", "splits"}


def _link_status(p: Path) -> tuple[str, str]:
    """(status, detail) for the data/ path itself -- catches the incident class."""
    isjunction = getattr(os.path, "isjunction", lambda _p: False)
    if not os.path.lexists(p):
        return "MISSING", "data/ does not exist (run setup_data_tree.py)"
    if os.path.lexists(p) and not p.exists():
        return "DANGLING", ("data/ is a DANGLING junction/symlink (target gone -- e.g. G: not "
                            "mounted). Restore the target or re-point. This is the 2026-06-14 class.")
    if p.exists() and not p.is_dir():
        return "SHADOW", "data/ exists but is NOT a directory (stray file shadowing it)."
    if p.is_dir() and (isjunction(p) or os.path.islink(p)):
        return "JUNCTION_OK", ("data/ is a junction/symlink that currently resolves. It WORKS now but "
                               "is the dangling-risk class; the standard recommends a real local dir.")
    return "REAL_DIR", "data/ is a real local directory (goal state)."


def _dir_stats(p: Path) -> tuple[int, int, float]:
    total = files = 0
    newest = 0.0
    for root, _dirs, fnames in os.walk(p):
        for fn in fnames:
            fp = Path(root) / fn
            try:
                st = fp.stat()
            except OSError:
                continue
            total += st.st_size
            files += 1
            newest = max(newest, st.st_mtime)
    return total, files, newest


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def _canonical_hygiene(name: str, strict: bool = False) -> list[str]:
    issues = []
    if name != name.lower():
        issues.append("uppercase")
    if " " in name:
        issues.append("space")
    if "-" in name:
        issues.append("hyphen")
    if strict and (name.endswith(("_v4", "_v5", "_fresh", "156"))
                   or "_2024" in name or "_2025" in name):
        issues.append("version-suffix?")
    return issues


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--manifest", default="configs/data_manifest.yaml", type=Path)
    ap.add_argument("--json", default=None, type=Path, help="also write a JSON report")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ABORT] manifest not found: {args.manifest}")
        return 2
    man = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    sources = man.get("sources", {})

    # alias -> canonical, and the set of all canonical names
    alias_to_canon = {}
    for canon, meta in sources.items():
        for a in (meta.get("aliases") or []):
            alias_to_canon[str(a).lower()] = canon

    report = {"data_dir": str(args.data_dir), "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("=" * 78)
    print(f" DATA TREE AUDIT (read-only): {args.data_dir}")
    print("=" * 78)

    status, detail = _link_status(args.data_dir)
    report["link_status"] = status
    flag = {"REAL_DIR": "[ok] ", "JUNCTION_OK": "[warn]", "DANGLING": "[FAIL]",
            "SHADOW": "[FAIL]", "MISSING": "[FAIL]"}[status]
    print(f"\n data/ status: {flag} {status} -- {detail}")
    if status in ("DANGLING", "SHADOW", "MISSING"):
        print("\n VERDICT: BLOCKED -- data/ is not usable; fix the above before anything else.")
        if args.json:
            args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    warnings = 0
    compliance_violations = 0

    # present dirs on disk (one level under external/processed); raw handled separately
    present = {}
    for sub in ("external", "processed"):
        base = args.data_dir / sub
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    present[(sub, child.name)] = child

    # per-source inventory
    print("\n [sources in manifest]")
    rollup = {}
    seen_locations = set()
    for canon, meta in sorted(sources.items()):
        loc = meta.get("location", "external")
        cls = meta.get("class", "?")
        tier = meta.get("tier", "?")
        sync = bool(meta.get("sync", False))
        d = args.data_dir / loc / canon
        on_disk = d.is_dir()
        seen_locations.add((loc, canon))
        size = files = 0
        newest = 0.0
        if on_disk:
            size, files, newest = _dir_stats(d)
            rollup[cls] = rollup.get(cls, 0) + size
        mark = "ok  " if on_disk else "MISS"
        # compliance: controlled marked for sync
        comp = ""
        if sync and tier == "controlled":
            comp = "  <-- COMPLIANCE VIOLATION (controlled + sync=true)"
            compliance_violations += 1
        elif sync and tier == "review" and on_disk:
            comp = "  <-- review tier before syncing (present + sync=true)"
            warnings += 1
        age = time.strftime("%Y-%m-%d", time.localtime(newest)) if newest else "-"
        print(f"   {mark}  {canon:<24} {loc:<9} {tier:<10} {cls:<22} "
              f"sync={str(sync):<5} {_human(size):>9} {files:>5}f {age}{comp}")
        report.setdefault("sources", {})[canon] = {
            "location": loc, "tier": tier, "class": cls, "sync": sync,
            "on_disk": on_disk, "bytes": size, "files": files,
        }

    # orphans + naming hygiene (external = canonical sources; processed = regenerable outputs)
    print("\n [orphans / naming hygiene]")
    orphans, infos = [], []
    for (sub, name), d in present.items():
        if (sub, name) in seen_locations:
            issues = _canonical_hygiene(name, strict=False)
            if issues:
                print(f"   [warn] {sub}/{name}: canonical but non-ideal name ({', '.join(issues)})")
                warnings += 1
            continue
        canon = alias_to_canon.get(name.lower())
        if canon:
            print(f"   [warn] {sub}/{name}: ALIAS of '{canon}' -- migrate into {sub}/{canon}/")
            warnings += 1
            orphans.append(f"{sub}/{name}")
        elif sub == "processed":
            print(f"   [info] processed/{name}: regenerable output (not manifest-tracked; prune if stale)")
            infos.append(f"processed/{name}")
        else:
            issues = _canonical_hygiene(name, strict=True)
            extra = f" ({', '.join(issues)})" if issues else ""
            print(f"   [warn] {sub}/{name}: ORPHAN in external/ (not in manifest){extra}")
            warnings += 1
            orphans.append(f"{sub}/{name}")
    # raw/: only 'cache' is expected infrastructure
    raw = args.data_dir / "raw"
    if raw.is_dir():
        for child in sorted(raw.iterdir()):
            if child.is_dir() and child.name != "cache":
                canon = alias_to_canon.get(child.name.lower())
                if canon:
                    print(f"   [warn] raw/{child.name}: ALIAS of '{canon}' -- migrate into raw/{canon}/")
                    warnings += 1
                    orphans.append(f"raw/{child.name}")
                else:
                    print(f"   [info] raw/{child.name}: raw download area (re-downloadable; not manifest-tracked)")
                    infos.append(f"raw/{child.name}")
    if not orphans and not infos:
        print("   [ok] no orphan/alias dirs")
    report["orphans"] = orphans
    report["processed_outputs"] = infos

    # security-aware rollup: where each present source's bytes should go
    print("\n [backup rollup -- security-aware]")
    buckets = {"cloud_backup": 0, "offline_only": 0, "regenerable": 0}
    for canon, meta in sources.items():
        d = args.data_dir / meta.get("location", "external") / canon
        if not d.is_dir():
            continue
        size = _dir_stats(d)[0]
        if meta.get("tier") == "controlled":
            buckets["offline_only"] += size
        elif meta.get("class") in _BACKUP_CLASSES:
            buckets["cloud_backup"] += size
        else:
            buckets["regenerable"] += size
    print(f"   {'cloud-backup (rclone -> G)':<30} {_human(buckets['cloud_backup']):>10}   syncable")
    print(f"   {'offline-only (controlled)':<30} {_human(buckets['offline_only']):>10}   encrypted/offline -- NOT cloud")
    print(f"   {'regenerable (rebuild)':<30} {_human(buckets['regenerable']):>10}   do not back up")
    report["rollup"] = buckets

    print("\n" + "=" * 78)
    if compliance_violations:
        print(f" VERDICT: COMPLIANCE VIOLATION -- {compliance_violations} controlled source(s) "
              "marked sync=true. Set sync:false before any G-drive push.")
        rc = 2
    elif warnings:
        print(f" VERDICT: {warnings} warning(s) -- aliases/orphans/naming/review above. data/ usable.")
        rc = 1
    else:
        print(" VERDICT: CLEAN -- layout matches the manifest; no aliases, orphans, or violations.")
        rc = 0
    report["warnings"] = warnings
    report["compliance_violations"] = compliance_violations
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f" JSON report -> {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
