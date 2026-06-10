#!/usr/bin/env python3
"""probe_protein_coord_cache.py -- READ-ONLY diagnosis of the ESM-2 coverage cap.

Author: Monzia Moodie

WHY THIS EXISTS
---------------
ESM-2 scores exactly ``len(candidates)`` where
``candidates = missense rows with non-null protein_pos/wt_aa/mut_aa`` (see
src/genomic_variant_classifier/data/esm2.py::ESM2Connector.annotate_dataframe).
That count was a *fixed* 3,451 in BOTH the 3k-row smoke and the 1.49M-row full
Run 15. There is no cap in esm2.py. ``protein_pos`` is populated only by
ProteinCoordConnector (src/.../data/protein_coords.py), whose
``_load_or_build_index`` LOADS ``alphamissense_protein_index.parquet`` if it
exists and NEVER rebuilds -- and the cache filename carries no cohort
fingerprint. So a small-cohort (smoke) build poisons every later full run.

This probe confirms or refutes that, WITHOUT mutating anything:
  (1) coord-index cache: exists? rows? unique loci? protein_pos non-null?
  (2) compare to the full missense count (the true ceiling)
  (3) AlphaMissense source: path, size, has PROTEIN_VARIANT?, chrom format

It is strictly read-only: no build, no write, no delete. Safe to run anytime.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
from pathlib import Path

# Default full-missense count measured by diagnose_esm2_coverage.py on
# data/processed/clinvar_grch38_clean_seq.parquet (override with --missense-total
# or recompute exactly with --clinvar).
MISSENSE_TOTAL_DEFAULT = 2_488_889

CACHE_NAME = "alphamissense_protein_index.parquet"
INDEX_COLS = ["_c", "_p", "_r", "_a", "protein_pos", "wt_aa", "mut_aa"]

# Where to look, relative to repo root. Recursive globs below also catch nesting.
DEFAULT_ROOTS = [
    "data/external/alphamissense",
    "data/external",
    "data/raw",
    "data/processed",
    "data",
    ".",
]
AM_SOURCE_PATTERNS = [
    "AlphaMissense_hg38.tsv.gz",
    "AlphaMissense*hg38*.tsv.gz",
    "AlphaMissense*.tsv.gz",
    "alphamissense*.tsv.gz",
    "alphamissense*index*.parquet",
    "alphamissense*.parquet",
]


def _num_rows_parquet(path: str) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def find(roots, patterns):
    hits: list[str] = []
    for r in roots:
        for pat in patterns:
            hits += glob.glob(os.path.join(r, "**", pat), recursive=True)
            hits += glob.glob(os.path.join(r, pat))
    seen, out = set(), []
    for h in hits:
        ap = os.path.abspath(h)
        if ap not in seen and os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
    return out


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def probe_cache(roots, missense_total: int) -> None:
    print("=" * 72)
    print("[1] PROTEIN-COORD INDEX CACHE  (alphamissense_protein_index.parquet)")
    print("=" * 72)
    caches = find(roots, [CACHE_NAME])
    if not caches:
        print("  NOT FOUND under any search root.")
        print("  -> Interpretation: no cache => ProteinCoordConnector rebuilds from")
        print("     the AlphaMissense TSV every run. If the full run still scored only")
        print("     3,451, the bottleneck is the AM source/join, not a stale cache.")
        return
    for c in caches:
        size = os.path.getsize(c)
        try:
            nrows = _num_rows_parquet(c)
        except Exception as exc:  # noqa: BLE001
            print(f"  {c}\n    (could not read parquet metadata: {exc})")
            continue
        print(f"  PATH : {c}")
        print(f"  SIZE : {human(size)}")
        print(f"  ROWS : {nrows:,}")
        # Detail read only when small enough to be cheap.
        if nrows <= 5_000_000:
            import pandas as pd
            cols_present = []
            try:
                import pyarrow.parquet as pq
                cols_present = pq.ParquetFile(c).schema.names
            except Exception:
                pass
            usecols = [x for x in ["_c", "_p", "_r", "_a", "protein_pos"] if x in cols_present] or None
            df = pd.read_parquet(c, columns=usecols)
            if {"_c", "_p", "_r", "_a"}.issubset(df.columns):
                n_uniq = df[["_c", "_p", "_r", "_a"]].drop_duplicates().shape[0]
                print(f"  UNIQUE LOCI (_c,_p,_r,_a): {n_uniq:,}")
            if "protein_pos" in df.columns:
                n_pp = int(df["protein_pos"].notna().sum())
                print(f"  protein_pos NON-NULL    : {n_pp:,}")
            print(f"  SCHEMA: {cols_present}")
            print("  HEAD:")
            try:
                print(df.head(5).to_string(index=False))
            except Exception:
                pass
        pct = 100.0 * nrows / max(missense_total, 1)
        print(f"  COVERAGE vs full missense ({missense_total:,}): {pct:.3f}%")
        print("  VERDICT:")
        if nrows < 50_000:
            print("    *** STALE/SMALL CACHE *** rows << full missense. This is almost")
            print("    certainly the 3,451 cap: a small-cohort build is being reloaded.")
            print("    Fix = cohort-key the cache (or delete it) and rebuild on the full")
            print("    cohort, then verify coverage climbs to the millions.")
        elif pct < 50.0:
            print("    Cache is non-trivial but still covers <50% of missense. Either a")
            print("    partial build or a genuine AM-join coverage problem -- run the")
            print("    coverage report (protein_coords.py __main__) against the full TSV.")
        else:
            print("    Cache coverage looks healthy. If ESM-2 still scored 3,451, the cap")
            print("    is downstream (is_missense gate / candidates filter) -- re-check.")


def probe_am_source(roots) -> None:
    print()
    print("=" * 72)
    print("[2] ALPHAMISSENSE SOURCE  (what ProteinCoordConnector would rebuild from)")
    print("=" * 72)
    srcs = find(roots, AM_SOURCE_PATTERNS)
    # Don't report the coord-index cache itself as a "source".
    srcs = [s for s in srcs if os.path.basename(s) != CACHE_NAME]
    if not srcs:
        print("  NO AlphaMissense source (TSV.gz or parquet) found under search roots.")
        print("  -> If absent on this box, a rebuild cannot run here; coords must be")
        print("     rebuilt where the 613 MB AlphaMissense_hg38.tsv.gz lives (Vast.ai).")
        return
    for s in srcs:
        size = os.path.getsize(s)
        print(f"\n  PATH : {s}")
        print(f"  SIZE : {human(size)}")
        low = s.lower()
        if low.endswith(".tsv.gz") or low.endswith(".tsv"):
            opener = gzip.open if low.endswith(".gz") else open
            header_cells = None
            data_row = None
            try:
                with opener(s, "rt") as f:  # type: ignore[operator]
                    for i, line in enumerate(f):
                        cells = line.rstrip("\n").split("\t")
                        norm = [c.lstrip("#").strip().upper() for c in cells]
                        if header_cells is None and {"CHROM", "POS"}.issubset(set(norm)):
                            header_cells = cells
                            norm_header = norm
                            continue
                        if header_cells is not None:
                            data_row = cells
                            break
                        if i > 400:  # header should be within the comment preamble
                            break
            except Exception as exc:  # noqa: BLE001
                print(f"    (could not read header: {exc})")
                continue
            if header_cells is None:
                print("    No CHROM/POS header row found in first 400 lines"
                      " -- not the raw AlphaMissense TSV (maybe a precomputed index).")
                continue
            has_pv = "PROTEIN_VARIANT" in norm_header
            print(f"    HEADER ({len(header_cells)} cols): {header_cells}")
            print(f"    HAS PROTEIN_VARIANT column: {has_pv}"
                  f"  <- protein_coords REQUIRES this; without it _build_index raises")
            if data_row is not None:
                chrom0 = data_row[0]
                fmt = "chrN (e.g. 'chr1')" if str(chrom0).lower().startswith("chr") else "bare ('1')"
                print(f"    SAMPLE DATA ROW: {data_row}")
                print(f"    CHROM FORMAT: {fmt}  (protein_coords._norm_chrom strips 'chr' -> OK either way)")
        elif low.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(s)
                print(f"    ROWS  : {pf.metadata.num_rows:,}")
                print(f"    SCHEMA: {pf.schema.names}")
                if "protein_variant" not in [c.lower() for c in pf.schema.names]:
                    print("    NOTE: no 'protein_variant' column -> this is the SCORE index")
                    print("    (chrom/pos/ref/alt/alphamissense_score), NOT a coords source.")
                    print("    If ac.alphamissense_path points HERE, _build_index cannot")
                    print("    parse it as a TSV and would raise (unless the cache exists).")
            except Exception as exc:  # noqa: BLE001
                print(f"    (could not read parquet: {exc})")


def maybe_recompute_missense(clinvar: str | None, default: int) -> int:
    if not clinvar:
        return default
    if not os.path.isfile(clinvar):
        print(f"  (--clinvar not found: {clinvar}; using default {default:,})")
        return default
    import pandas as pd
    s = pd.read_parquet(clinvar, columns=["consequence"])["consequence"].fillna("")
    n = int(s.str.contains("missense", case=False).sum())
    print(f"  (recomputed missense total from {clinvar}: {n:,})")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=None,
                    help="search roots (default: a sensible repo-relative set)")
    ap.add_argument("--missense-total", type=int, default=MISSENSE_TOTAL_DEFAULT)
    ap.add_argument("--clinvar", default=None,
                    help="optional: recompute the missense total from this cohort parquet")
    args = ap.parse_args()
    roots = args.roots if args.roots else DEFAULT_ROOTS
    roots = [r for r in roots if os.path.isdir(r)] or ["."]
    print("READ-ONLY probe. No files are built, written, or deleted.")
    print(f"Search roots: {roots}\n")
    total = maybe_recompute_missense(args.clinvar, args.missense_total)
    probe_cache(roots, total)
    probe_am_source(roots)
    print("\nDone. Paste this whole block back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
