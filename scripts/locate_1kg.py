#!/usr/bin/env python3
"""locate_1kg.py -- find + verify a 1000 Genomes AF parquet for --kg-path.

ThousandGenomesConnector.fill_missing_af reads columns=["variant_id","allele_freq"]
and merges on variant_id ("chrom:pos:ref:alt", NO 'chr' prefix). This locator finds
candidate parquets, confirms that exact schema, and checks the variant_id key format
against the cohort -- a format mismatch would silently fill ZERO null AFs.

NOTE: this connector fills only the combined allele_freq. It does NOT populate
af_1kg_afr/eur/eas/sas/amr (those remain stubs; no per-population source is wired).

STRICTLY READ-ONLY (footer schema + first row-group sample).

Usage:  python scripts/locate_1kg.py [--cohort data/processed/clinvar_grch38_clean_seq.parquet]
Author: Monzia Moodie."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

GLOBS = [
    "data/external/1000g/*.parquet",
    "data/**/kg_phase3*.parquet",
    "data/**/*1000g*.parquet",
    "data/**/*1kg*.parquet",
    "data/**/kg*af*.parquet",
    "data/**/phase3*af*.parquet",
]
NEED = ["variant_id", "allele_freq"]
KEY_RE = re.compile(r"^[0-9XYMT]+:\d+:[ACGTN]+:[ACGTN]+$", re.IGNORECASE)


def _mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / 1048576, 2)
    except OSError:
        return -1.0


def _schema(p: Path):
    import pyarrow.parquet as pq
    return list(pq.read_schema(p).names)


def _sample_col(p: Path, col: str, n: int = 50):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    if col not in pf.schema_arrow.names:
        return []
    batch = next(pf.iter_batches(batch_size=n, columns=[col]))
    return [str(v) for v in batch.column(0).to_pylist() if v is not None][:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path,
                    default=Path("data/processed/clinvar_grch38_clean_seq.parquet"))
    args = ap.parse_args()

    print("=" * 74)
    print(" 1000 Genomes (--kg-path) locator (read-only)")
    print("=" * 74)

    hits = []
    for pat in GLOBS:
        for m in glob.glob(pat, recursive=True):
            mp = Path(m)
            if mp.is_file() and mp not in hits and not mp.name.endswith((".bak", ".OOMbak")):
                hits.append(mp)
    hits = sorted(set(hits), key=lambda x: str(x).lower())

    if not hits:
        print(f"\n NOT FOUND. searched: {GLOBS}")
        print(" -> 1KGP not staged. Build it from Phase 3 VCFs with")
        print("    scripts/build_1kg_parquet.py, or defer --kg-path to a later run.")
        return 1

    # cohort key-format reference
    coh_keys = _sample_col(args.cohort, "variant_id") if args.cohort.exists() else []
    coh_fmt_ok = all(KEY_RE.match(k) for k in coh_keys[:20]) if coh_keys else None

    usable = []
    for p in hits:
        cols = _schema(p)
        missing = [c for c in NEED if c not in cols]
        print(f"\n  {p}  ({_mb(p)} MB)")
        print(f"    columns: {cols[:8]}{' ...' if len(cols) > 8 else ''}")
        if missing:
            print(f"    UNUSABLE: missing required {missing} (connector reads {NEED})")
            continue
        keys = _sample_col(p, "variant_id")
        fmt_ok = all(KEY_RE.match(k) for k in keys[:20]) if keys else False
        print(f"    schema: variant_id + allele_freq PRESENT")
        print(f"    variant_id sample: {keys[:3]}  format-ok={fmt_ok}")
        if coh_keys:
            both_ok = fmt_ok and coh_fmt_ok
            print(f"    cohort variant_id sample: {coh_keys[:3]}  cohort-format-ok={coh_fmt_ok}")
            print(f"    KEY-FORMAT MATCH: {'YES' if both_ok else 'NO -- merge would fill 0 AFs'}")
            if both_ok:
                usable.append(p)
        elif fmt_ok:
            usable.append(p)

    print("\n" + "=" * 74)
    if usable:
        best = sorted(usable, key=lambda x: -_mb(x))[0]
        print(f" USABLE 1KGP parquet: {best}  ({_mb(best)} MB)")
        print(" Next: add a --kg-path flag to train.py (it is NOT exposed yet), then")
        print(f"   re-smoke with --kg-path {best} and confirm the null-AF count drops.")
        return 0
    print(" No usable 1KGP parquet (schema or key-format fail). See lines above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
