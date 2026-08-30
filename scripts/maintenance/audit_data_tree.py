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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceState:
    """One declared source, as found on disk. No formatting."""

    name: str
    location: str
    tier: str
    cls: str
    sync: bool
    on_disk: bool
    size: int
    files: int
    newest: float


@dataclass(frozen=True)
class AuditReport:
    """What the audit MEASURED. Rendered by main() and by audit_rows().

    Split out 2026-08-30. The computation lived inside a 159-line `main()`
    interleaved with 25 `print()` calls, so nothing could consume the findings
    without re-deriving them -- and `preflight_run17` had no way to call this
    audit at all. `preflight_data_guard.py` already shows the shape: a policy
    that COMPUTES, a `storage_rows()` that renders severity rows, and a
    `main()` that prints. One computation, two renderings.
    """

    data_dir: str
    link_status: str
    link_detail: str
    sources: tuple = ()
    #: Every hygiene finding in ENCOUNTER ORDER, as (kind, name, extra).
    #: The renderer walks this rather than four separate lists, so the report
    #: prints in the same order as before the 2026-08-30 split. Grouping by
    #: category would have been a behaviour change, and this refactor is not
    #: the place for one.
    entries: tuple = ()
    orphans: tuple = ()
    aliases: tuple = ()
    infos: tuple = ()
    hygiene: tuple = ()
    review_present: tuple = ()
    violations: tuple = ()
    rollup: dict = None
    generated: str = ""

    @property
    def blocked(self) -> bool:
        """data/ itself is unusable; nothing below it is meaningful."""
        return self.link_status in ("DANGLING", "SHADOW", "MISSING")

    @property
    def warnings(self) -> int:
        return (len(self.orphans) + len(self.aliases) + len(self.hygiene)
                + len(self.review_present))

    @property
    def return_code(self) -> int:
        if self.blocked or self.violations:
            return 2
        return 1 if self.warnings else 0


def audit_tree(data_dir: Path, manifest: Path) -> AuditReport:
    """MEASURE the tree against the manifest. Prints nothing, writes nothing."""
    man = yaml.safe_load(Path(manifest).read_text(encoding="utf-8")) or {}
    sources = man.get("sources", {}) or {}
    data_dir = Path(data_dir)

    alias_to_canon = {}
    for canon, meta in sources.items():
        for a in ((meta or {}).get("aliases") or []):
            alias_to_canon[str(a).lower()] = canon

    status, detail = _link_status(data_dir)
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    if status in ("DANGLING", "SHADOW", "MISSING"):
        return AuditReport(data_dir=str(data_dir), link_status=status,
                           link_detail=detail, rollup={}, generated=generated)

    present = {}
    for sub in ("external", "processed"):
        base = data_dir / sub
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir():
                    present[(sub, child.name)] = child

    states, violations, review_present, seen = [], [], [], set()
    rollup = {"cloud_backup": 0, "offline_only": 0, "regenerable": 0}
    for canon, meta in sorted(sources.items()):
        meta = meta or {}
        loc = meta.get("location", "external")
        cls = meta.get("class", "?")
        tier = meta.get("tier", "?")
        sync = bool(meta.get("sync", False))
        d = data_dir / loc / canon
        on_disk = d.is_dir()
        seen.add((loc, canon))
        size = files = 0
        newest = 0.0
        if on_disk:
            size, files, newest = _dir_stats(d)
            if tier == "controlled":
                rollup["offline_only"] += size
            elif cls in _BACKUP_CLASSES:
                rollup["cloud_backup"] += size
            else:
                rollup["regenerable"] += size
        if sync and tier == "controlled":
            violations.append(canon)
        elif sync and tier == "review" and on_disk:
            review_present.append(canon)
        states.append(SourceState(canon, loc, tier, cls, sync, on_disk,
                                  size, files, newest))

    entries = []
    for (sub, name), _d in present.items():
        if (sub, name) in seen:
            issues = _canonical_hygiene(name, strict=False)
            if issues:
                entries.append(("hygiene", f"{sub}/{name}", tuple(issues)))
            continue
        canon = alias_to_canon.get(name.lower())
        if canon:
            entries.append(("alias", f"{sub}/{name}", canon))
        elif sub == "processed":
            entries.append(("info", f"processed/{name}", "processed"))
        else:
            entries.append(("orphan", f"{sub}/{name}",
                            tuple(_canonical_hygiene(name, strict=True))))
    raw = data_dir / "raw"
    if raw.is_dir():
        for child in sorted(raw.iterdir()):
            if child.is_dir() and child.name != "cache":
                canon = alias_to_canon.get(child.name.lower())
                if canon:
                    entries.append(("alias", f"raw/{child.name}", canon))
                else:
                    entries.append(("info", f"raw/{child.name}", "raw"))
    orphans = [(n, x) for k, n, x in entries if k == "orphan"]
    aliases = [(n, x) for k, n, x in entries if k == "alias"]
    hygiene = [(n, x) for k, n, x in entries if k == "hygiene"]
    infos = [n for k, n, _x in entries if k == "info"]

    return AuditReport(
        data_dir=str(data_dir), link_status=status, link_detail=detail,
        sources=tuple(states), entries=tuple(entries),
        orphans=tuple(orphans), aliases=tuple(aliases),
        infos=tuple(infos), hygiene=tuple(hygiene),
        review_present=tuple(review_present), violations=tuple(violations),
        rollup=rollup, generated=generated)


def audit_rows(data_dir: str | Path = "data",
               manifest: str | Path = "configs/data_manifest.yaml"
               ) -> list[tuple[str, str]]:
    """(severity, message) rows in preflight_run17's gate convention.

    It RENDERS the severities the audit already computes into `return_code`:
    a blocked tree and a controlled-tier sync violation are FAIL, orphans and
    aliases and naming and review-tier are WARN. The gate invents nothing.

    Never raises on a readable manifest: a caller decides what a WARN means.
    """
    try:
        report = audit_tree(Path(data_dir), Path(manifest))
    except OSError as exc:
        return [("FAIL", f"data-tree: cannot read {manifest} ({exc})")]
    except Exception as exc:                      # malformed YAML, bad shape
        return [("FAIL", f"data-tree: audit failed ({type(exc).__name__}: {exc})")]

    if report.blocked:
        return [("FAIL", f"data-tree: {report.link_status} -- {report.link_detail}")]

    rows: list[tuple[str, str]] = []
    for canon in report.violations:
        rows.append(("FAIL", f"data-tree: {canon} is tier=controlled AND sync=true; "
                             "set sync:false before any cloud push"))
    for kind, name, extra in report.entries:
        if kind == "orphan":
            tail = f" ({', '.join(extra)})" if extra else ""
            rows.append(("WARN", f"data-tree: {name} is an ORPHAN, not in the "
                                 f"manifest{tail}"))
        elif kind == "alias":
            rows.append(("WARN", f"data-tree: {name} is an ALIAS of {extra}; "
                                 "consolidate_aliases.py folds it"))
        elif kind == "hygiene":
            rows.append(("WARN", f"data-tree: {name} is canonical but non-ideal "
                                 f"({', '.join(extra)})"))
    for canon in report.review_present:
        rows.append(("WARN", f"data-tree: {canon} is tier=review, present, and "
                             "sync=true; confirm the tier before syncing"))
    if not rows:
        rows.append(("OK", f"data-tree: {len(report.sources)} declared source(s), "
                           "no orphans, aliases or violations"))
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--manifest", default="configs/data_manifest.yaml", type=Path)
    ap.add_argument("--json", default=None, type=Path, help="also write a JSON report")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ABORT] manifest not found: {args.manifest}")
        return 2

    rep = audit_tree(args.data_dir, args.manifest)
    report = {"data_dir": rep.data_dir, "generated": rep.generated,
              "link_status": rep.link_status}

    print("=" * 78)
    print(f" DATA TREE AUDIT (read-only): {args.data_dir}")
    print("=" * 78)
    flag = {"REAL_DIR": "[ok] ", "JUNCTION_OK": "[warn]", "DANGLING": "[FAIL]",
            "SHADOW": "[FAIL]", "MISSING": "[FAIL]"}[rep.link_status]
    print(f"\n data/ status: {flag} {rep.link_status} -- {rep.link_detail}")
    if rep.blocked:
        print("\n VERDICT: BLOCKED -- data/ is not usable; fix the above before anything else.")
        if args.json:
            args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    print("\n [sources in manifest]")
    for st in rep.sources:
        mark = "ok  " if st.on_disk else "MISS"
        comp = ""
        if st.name in rep.violations:
            comp = "  <-- COMPLIANCE VIOLATION (controlled + sync=true)"
        elif st.name in rep.review_present:
            comp = "  <-- review tier before syncing (present + sync=true)"
        age = time.strftime("%Y-%m-%d", time.localtime(st.newest)) if st.newest else "-"
        print(f"   {mark}  {st.name:<24} {st.location:<9} {st.tier:<10} {st.cls:<22} "
              f"sync={str(st.sync):<5} {_human(st.size):>9} {st.files:>5}f {age}{comp}")
        report.setdefault("sources", {})[st.name] = {
            "location": st.location, "tier": st.tier, "class": st.cls,
            "sync": st.sync, "on_disk": st.on_disk, "bytes": st.size,
            "files": st.files,
        }

    print("\n [orphans / naming hygiene]")
    for kind, name, extra in rep.entries:
        if kind == "hygiene":
            print(f"   [warn] {name}: canonical but non-ideal name ({', '.join(extra)})")
        elif kind == "alias":
            sub = name.split("/", 1)[0]
            print(f"   [warn] {name}: ALIAS of '{extra}' -- migrate into {sub}/{extra}/")
        elif kind == "orphan":
            tail = f" ({', '.join(extra)})" if extra else ""
            print(f"   [warn] {name}: ORPHAN in external/ (not in manifest){tail}")
        elif extra == "processed":
            print(f"   [info] {name}: regenerable output (not manifest-tracked; prune if stale)")
        else:
            print(f"   [info] {name}: raw download area (re-downloadable; not manifest-tracked)")
    if not (rep.orphans or rep.aliases or rep.infos):
        print("   [ok] no orphan/alias dirs")
    report["orphans"] = [n for n, _ in rep.orphans] + [n for n, _ in rep.aliases]
    report["processed_outputs"] = list(rep.infos)

    print("\n [backup rollup -- security-aware]")
    b = rep.rollup
    print(f"   {'cloud-backup (rclone -> G)':<30} {_human(b['cloud_backup']):>10}   syncable")
    print(f"   {'offline-only (controlled)':<30} {_human(b['offline_only']):>10}   encrypted/offline -- NOT cloud")
    print(f"   {'regenerable (rebuild)':<30} {_human(b['regenerable']):>10}   do not back up")
    report["rollup"] = b

    print("\n" + "=" * 78)
    rc = rep.return_code
    if rep.violations:
        print(f" VERDICT: COMPLIANCE VIOLATION -- {len(rep.violations)} controlled source(s) "
              "marked sync=true. Set sync:false before any G-drive push.")
    elif rep.warnings:
        print(f" VERDICT: {rep.warnings} warning(s) -- aliases/orphans/naming/review above. data/ usable.")
    else:
        print(" VERDICT: CLEAN -- layout matches the manifest; no aliases, orphans, or violations.")
    report["warnings"] = rep.warnings
    report["compliance_violations"] = len(rep.violations)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f" JSON report -> {args.json}")
    return rc



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
