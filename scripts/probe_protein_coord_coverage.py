#!/usr/bin/env python3
"""probe_protein_coord_coverage.py (v2) -- verify OR rebuild the AlphaMissense
protein-coord index that gates ESM-2 in Run 16.

SAFETY (why v2): the connector's _build_index filters to the cohort you pass and
writes the result to the canonical cache path. So building it from a SAMPLE writes a
sample-sized cache over the full one. v1, run after deleting the cache, did exactly
that and corrupted the full index. v2's default mode is READ-ONLY: it never builds
from a sample, and it size-checks the cache (a full index is ~18 MB; a sample one is
<1 MB) so a corrupt cache FAILS even though the same sample would spuriously match.

  verify (read-only, safe -- run before every Run 16):
    python scripts/probe_protein_coord_coverage.py
  rebuild the FULL index (expensive; reads the 613 MB TSV over the full cohort):
    python scripts/probe_protein_coord_coverage.py --rebuild-full

Exit 0 PASS / 2 FAIL (low coverage or sample-sized cache) / 3 ENV or missing cache in
verify mode. Author: Monzia Moodie."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MIN_FULL_CACHE_MB = 5.0  # full index ~18 MB; anything smaller is sample-built/suspect


def _coverage(out):
    is_mm = out["is_missense"].astype(bool)
    n_mm = int(is_mm.sum())
    n_cov = int((is_mm & out["protein_pos"].notna()).sum())
    return n_mm, n_cov, n_cov / max(n_mm, 1)


def _add_missense(df):
    df["is_missense"] = (
        df["consequence"].fillna("").str.contains("missense", case=False).astype(int)
    )
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify/rebuild AlphaMissense protein-coord index.")
    ap.add_argument("--clinvar", default="data/processed/clinvar_grch38_clean_seq.parquet")
    ap.add_argument("--alphamissense", default="data/external/alphamissense/AlphaMissense_hg38.tsv.gz")
    ap.add_argument("--sample", type=int, default=50000, help="verify-mode sample size (0=full)")
    ap.add_argument("--min", type=float, default=0.90)
    ap.add_argument("--rebuild-full", action="store_true",
                    help="Delete any cache and rebuild the FULL index from the full cohort + TSV.")
    args = ap.parse_args(argv)

    try:
        import pandas as pd
        from genomic_variant_classifier.data.protein_coords import ProteinCoordConnector
    except Exception as e:  # noqa: BLE001
        print(f"ENV: cannot import deps ({e})")
        return 3

    cohort = Path(args.clinvar)
    if not cohort.exists():
        print(f"ENV: cohort not found: {cohort}")
        return 3

    want = ["chrom", "pos", "ref", "alt", "consequence"]
    pc = ProteinCoordConnector(alphamissense_file=args.alphamissense)
    cache = pc.cache_path
    exists = cache.exists()
    size_mb = (cache.stat().st_size / 1e6) if exists else 0.0
    print(f"alphamissense_file : {args.alphamissense}")
    print(f"cache_path         : {cache}  exists={exists}"
          + (f"  size={size_mb:.2f}MB" if exists else ""))

    if args.rebuild_full:
        if Path(str(args.alphamissense)).suffix == ".parquet":
            print("REFUSE: --rebuild-full needs the TSV (AlphaMissense_hg38.tsv.gz), not a parquet.")
            return 3
        if exists:
            cache.unlink()
            print(f"removed existing cache to force a clean full rebuild: {cache}")
        df = _add_missense(pd.read_parquet(cohort, columns=want))
        print(f"building FULL index over {len(df):,} cohort rows "
              f"(reads the 613 MB TSV; expect several minutes)...")
        out = pc.annotate_dataframe(df)
        if "protein_pos" not in out.columns:
            print("FAIL: connector returned no protein_pos (stub path / build failed).")
            return 2
        n_mm, n_cov, cov = _coverage(out)
        new_mb = (cache.stat().st_size / 1e6) if cache.exists() else 0.0
        print(f"rebuilt cache size : {new_mb:.2f} MB")
        print(f"full-cohort cover  : {cov:.4f} ({n_cov:,}/{n_mm:,} missense, need >= {args.min})")
        ok = cov >= args.min and new_mb >= _MIN_FULL_CACHE_MB
        print("VERDICT:", "PASS -- full index rebuilt; safe for Run 16." if ok
              else "FAIL -- low coverage or cache too small; inspect before Run 16.")
        return 0 if ok else 2

    # ---- verify mode: READ-ONLY (never build-from-sample) ----
    if not exists:
        print("ABORT (verify mode): protein-coord index cache is MISSING.")
        print("  Building from a sample would corrupt the full index. Rebuild first:")
        print("    python scripts/probe_protein_coord_coverage.py --rebuild-full")
        return 3
    if size_mb < _MIN_FULL_CACHE_MB:
        print(f"FAIL: cache is {size_mb:.2f} MB; a full index is ~18 MB. This is almost certainly")
        print("  a SAMPLE-built (corrupted) index. Rebuild the full one:")
        print("    python scripts/probe_protein_coord_coverage.py --rebuild-full")
        return 2

    df = pd.read_parquet(cohort, columns=want)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=0).reset_index(drop=True)
    df = _add_missense(df)
    out = pc.annotate_dataframe(df)  # cache present -> load path -> no write
    if "protein_pos" not in out.columns:
        print("FAIL: connector returned no protein_pos column (stub path).")
        return 2
    n_mm, n_cov, cov = _coverage(out)
    print(f"sampled rows       : {len(df):,}")
    print(f"coverage           : {cov:.4f} ({n_cov:,}/{n_mm:,} missense, need >= {args.min})")
    if cov >= args.min:
        print("VERDICT: PASS -- protein-coord index is fresh; ESM-2 will populate in Run 16.")
        return 0
    print("VERDICT: FAIL -- index STALE/mismatched. Rebuild: --rebuild-full")
    return 2


if __name__ == "__main__":
    sys.exit(main())
