#!/usr/bin/env python3
"""
scripts/maintenance/consolidate_aliases.py  --  Monzia Moodie

Fold alias directories into their canonical names, per configs/data_manifest.yaml.
DRY-RUN by default; pass --execute to act. Safety rules:

  * EMPTY alias (0 files)            -> remove the empty directory.
  * populated alias, canonical EMPTY -> move (rename) alias -> canonical.
  * populated alias, canonical ALSO  -> MERGE: copy each alias file into the
    populated                           canonical dir; ABORT that alias on any
                                        name collision whose CONTENT differs
                                        (never overwrite). Only after EVERY
                                        alias file is verified present in
                                        canonical (SHA-256 match) is the alias
                                        dir removed.

COMPARISON IS BY DIGEST, NOT BY SIZE. Until 2026-08-29 both the collision
check and the post-merge verification compared `st_size`, then removed the
alias directory. Two files of equal size and DIFFERENT content passed every
check and the alias file was silently lost -- the script does not overwrite,
so it discarded the source instead.

That is not hypothetical here. The artifact lineage census of 2026-08-28 found
THREE equal-size groups with different digests under `data/external/`,
including two EVE score files at exactly 612,501 bytes:

    variant_files/TPIS_HUMAN.csv   465d9fd2eee342c8...
    variant_files/TSHB_HUMAN.csv   2ef2b73abcadc062...

Merges are rare and run interactively, so hashing both sides costs little.

An alias directory is NEVER removed until its contents are confirmed in the
canonical directory. Collisions/mismatches abort that alias with no changes.

Usage:
  python scripts/maintenance/consolidate_aliases.py            # dry-run plan
  python scripts/maintenance/consolidate_aliases.py --execute  # perform it
  python scripts/maintenance/consolidate_aliases.py --only 1000genomes,clinvar_fresh
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml


def _files(d: Path) -> list[Path]:
    return [p for p in d.rglob("*") if p.is_file()] if d.is_dir() else []


def _digest(p: Path) -> str:
    """SHA-256 of a file, streamed so a large artifact is not held in memory.

    Equal size is NOT equal content. This function exists because comparing
    `st_size` let two different files satisfy a check that then DELETED one of
    them.
    """
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_content(a: Path, b: Path) -> bool:
    """Cheap size check FIRST, then the digest that decides.

    Different sizes cannot be equal content, so the digest is computed only
    for the candidates that survive -- correctness from the digest, speed from
    the size.
    """
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except OSError:
        return False
    return _digest(a) == _digest(b)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--manifest", default="configs/data_manifest.yaml", type=Path)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated alias names to act on")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ABORT] manifest not found: {args.manifest}")
        return 2
    man = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    sources = man.get("sources", {})
    only = set(s.strip() for s in args.only.split(",")) if args.only else None

    # alias -> (canonical, location)
    plan = []  # (action, alias_dir, canon_dir, detail)
    for canon, meta in sources.items():
        loc = meta.get("location", "external")
        for alias in (meta.get("aliases") or []):
            if only and alias not in only:
                continue
            alias_dir = args.data_dir / loc / alias
            if not alias_dir.is_dir():
                continue
            canon_dir = args.data_dir / loc / canon
            af = _files(alias_dir)
            if not af:
                plan.append(("remove_empty", alias_dir, canon_dir, "0 files"))
                continue
            canon_files = _files(canon_dir)
            if not canon_files:
                plan.append(("move", alias_dir, canon_dir, f"{len(af)} files -> empty/absent canonical"))
                continue
            # merge: check collisions
            collisions = []
            for f in af:
                rel = f.relative_to(alias_dir)
                tgt = canon_dir / rel
                if tgt.exists() and not _same_content(tgt, f):
                    collisions.append(str(rel))
            if collisions:
                plan.append(("ABORT_collision", alias_dir, canon_dir,
                             f"{len(collisions)} differing name-collision(s): {collisions[:3]}"))
            else:
                plan.append(("merge", alias_dir, canon_dir, f"{len(af)} files into populated canonical"))

    if not plan:
        print("  [ok] no alias directories present on disk -- nothing to consolidate.")
        return 0

    print(f"  {'EXECUTE' if args.execute else 'DRY-RUN'} -- {len(plan)} alias action(s):")
    rc = 0
    for action, alias_dir, canon_dir, detail in plan:
        tag = {"remove_empty": "[rm-empty]", "move": "[move]", "merge": "[merge]",
               "ABORT_collision": "[ABORT]"}[action]
        print(f"   {tag} {alias_dir}  ({detail})")
        if action == "ABORT_collision":
            rc = 1
            print(f"            -> skipped: resolve collisions by hand; nothing changed for this alias.")
            continue
        if not args.execute:
            continue
        if action == "remove_empty":
            # guard: re-confirm empty right before deleting
            if _files(alias_dir):
                print("            -> NOT empty at execute time; skipped."); rc = 1; continue
            shutil.rmtree(alias_dir)
            print(f"            -> removed {alias_dir}")
        elif action == "move":
            canon_dir.parent.mkdir(parents=True, exist_ok=True)
            if canon_dir.is_dir() and not _files(canon_dir):
                shutil.rmtree(canon_dir)  # remove empty placeholder so rename lands
            shutil.move(str(alias_dir), str(canon_dir))
            print(f"            -> moved -> {canon_dir}")
        elif action == "merge":
            for f in _files(alias_dir):
                rel = f.relative_to(alias_dir)
                tgt = canon_dir / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                if not tgt.exists():
                    shutil.copy2(f, tgt)
            # verify every alias file present in canonical with same size
            bad = [str(f.relative_to(alias_dir)) for f in _files(alias_dir)
                   if not (canon_dir / f.relative_to(alias_dir)).exists()
                   or not _same_content(canon_dir / f.relative_to(alias_dir), f)]
            if bad:
                print(f"            -> VERIFY FAILED for {bad[:3]}; alias KEPT (not removed)."); rc = 1; continue
            shutil.rmtree(alias_dir)
            print(f"            -> merged + verified; removed {alias_dir}")
    print("  [done] " + ("changes applied" if args.execute else "dry-run (nothing changed)"))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
