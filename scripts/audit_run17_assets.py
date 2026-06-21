#!/usr/bin/env python3
"""audit_run17_assets.py -- Author: Monzia Moodie

Diagnose the DataReadinessAgent NO_GO by auditing every ACTIVE critical asset
(monitoring.registry.critical_assets()) and, for each, also checking the KNOWN
alternate/duplicate locations documented in the registry. This separates three
very different remediation paths:

  PRESENT       primary registry path exists + non-empty            -> nothing to do
  FOUND_AT_ALT  primary missing BUT a known alternate has the data  -> cheap REGISTRY PATH FIX
  MISSING       neither primary nor any alternate present           -> genuine re-acquisition

This matters because the registry itself flags e.g. finngen's path as a FILENAME
TYPO ('finnge') and notes data/processed duplicates for spliceai/alphamissense and
a cached dbsnp/string artifact -- so "11 missing" is partly a path problem, not 11
absent files. We do NOT want to re-download 30 GB of FinnGen if the data is sitting
under a slightly different name.

The asset list is pulled from the registry (single source of truth) so it never
drifts; the alternates map is curated from the registry's own comments.

Usage:
  python scripts/audit_run17_assets.py --repo-root C:\\Projects\\genomic-variant-classifier
Exit code: 0 iff every ACTIVE asset is PRESENT; 1 otherwise (FOUND_AT_ALT or MISSING).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Known alternate/duplicate locations, keyed by registry source name. Curated from
# the registry comments (DUP copy / also / cached / index present / FILENAME TYPO).
ALTERNATES: dict[str, list[str]] = {
    "clinvar": ["data/raw/clinvar/clinvar_GRCh38.vcf.gz"],            # raw VCF -> would need re-prep
    "spliceai": ["data/processed/spliceai_index.parquet"],           # DUP copy
    "alphamissense": ["data/processed/alphamissense_index.parquet"], # also here
    "dbsnp": ["data/raw/cache/dbsnp_af_lookup.parquet"],             # cache lookup
    "string": ["data/raw/cache/string_graph_700.pkl"],              # cached graph (preflight uses this)
    "finngen": ["data/external/finngen/finngen_R12_annotated_variants_v1.gz"],  # corrected (non-typo) name
}


def stat_path(p: Path) -> tuple[bool, int]:
    """(present, size). A file is present iff it exists and is non-empty; a
    directory is present iff it contains >=1 file (size = file count)."""
    try:
        if p.is_file():
            sz = p.stat().st_size
            return sz > 0, sz
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            return n > 0, n
    except OSError:
        pass
    return False, 0


def audit(root: str | Path, items: list[dict]) -> list[dict]:
    """items: [{source, primary, alternates:[...]}] -> rows with status."""
    root = Path(root)
    rows: list[dict] = []
    for it in items:
        present, size = stat_path(root / it["primary"])
        alts_found = []
        for a in it.get("alternates", []):
            ap_present, ap_size = stat_path(root / a)
            if ap_present:
                alts_found.append({"path": a, "size": ap_size})
        status = "PRESENT" if present else ("FOUND_AT_ALT" if alts_found else "MISSING")
        rows.append({"source": it["source"], "primary": it["primary"],
                     "present": present, "size": size, "status": status,
                     "alternates_found": alts_found})
    return rows


def _items_from_registry() -> list[dict]:
    from genomic_variant_classifier.monitoring.registry import REGISTRY, Verdict
    items = []
    for s in REGISTRY:
        if s.verdict is Verdict.ACTIVE and s.local_path:
            items.append({"source": s.name, "primary": s.local_path,
                          "alternates": ALTERNATES.get(s.name, [])})
    return items


def _human(n: int, is_size: bool) -> str:
    if not is_size:
        return f"{n} files"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}"
        n /= 1024
    return str(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)

    try:
        items = _items_from_registry()
    except Exception as exc:
        print(f"ERROR: could not import registry ({exc}). Run from the repo with the venv active.")
        return 2

    rows = audit(args.repo_root, items)
    print(f"Run-17 critical-asset audit  --  {len(rows)} ACTIVE assets\n")
    for r in sorted(rows, key=lambda x: (x["status"] != "MISSING", x["status"], x["source"])):
        tag = {"PRESENT": "ok   ", "FOUND_AT_ALT": "FIX  ", "MISSING": "MISS "}[r["status"]]
        line = f"  {tag} {r['source']:18} {r['status']:13} {r['primary']}"
        print(line)
        for a in r["alternates_found"]:
            print(f"        -> data present at ALT: {a['path']}  ({a['size']:,} bytes)")
    miss = [r["source"] for r in rows if r["status"] == "MISSING"]
    fix = [r["source"] for r in rows if r["status"] == "FOUND_AT_ALT"]
    ok = [r["source"] for r in rows if r["status"] == "PRESENT"]
    print(f"\n  PRESENT={len(ok)}  FOUND_AT_ALT={len(fix)}  MISSING={len(miss)}")
    if fix:
        print(f"  PATH-FIX (data exists at an alternate): {', '.join(fix)}")
    if miss:
        print(f"  GENUINELY MISSING (re-acquire / re-prep): {', '.join(miss)}")
    return 0 if (not miss and not fix) else 1


if __name__ == "__main__":
    sys.exit(main())
