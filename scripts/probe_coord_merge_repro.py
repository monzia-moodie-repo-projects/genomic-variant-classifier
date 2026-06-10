#!/usr/bin/env python3
"""probe_coord_merge_repro.py -- reproduce Run 15 step-10b LOCALLY to localize the cap.

Author: Monzia Moodie

Run 15 loaded the 2.41M-row coord cache yet step 10b populated only 3,461 protein_pos.
This probe reproduces that merge on THIS box against the SAME local cache and splits the
cause into three distinguishable outcomes:

  (a) normalization-only key OVERLAP is large, but the connector's .merge() lands few
      -> MERGE-DTYPE bug (nullable Int64 vs int64 key mismatch). Fix in code.
  (b) normalization-only key OVERLAP itself is ~3,461
      -> the cache keys don't match the cleaned-cohort keys (cache built on a different
         cohort/representation, or AlphaMissense allele rep differs). Fix = rebuild index
         keyed on the SAME cohort training uses.
  (c) connector merge lands many here, but Run 15 logged 3,461
      -> the Vast box ran an OLD/smaller cache than this local one. Fix = ship the right
         cache + add a pre-train coverage assertion.

READ-ONLY: never builds, writes, or deletes. (The connector only READS when the cache
exists; this probe points cache_dir at the existing cache and does not pass a TSV path
that could trigger a rebuild.)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

CACHE_DEFAULT = "data/external/alphamissense/alphamissense_protein_index.parquet"
COHORT_DEFAULT = "data/processed/clinvar_grch38_clean.parquet"


def _norm_chrom(s: pd.Series) -> pd.Series:
    # EXACT mirror of protein_coords._norm_chrom
    return s.astype(str).str.replace(r"(?i)^chr", "", regex=True).str.upper()


def _norm_keys(df: pd.DataFrame):
    # EXACT mirror of protein_coords._norm_keys
    c = _norm_chrom(df["chrom"])
    p = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    r = df["ref"].astype(str).str.upper()
    a = df["alt"].astype(str).str.upper()
    return c, p, r, a


def overlap_via_strings(cohort: pd.DataFrame, idx: pd.DataFrame) -> dict:
    """Normalization-only key overlap, computed on STRING-cast keys so dtype quirks
    cannot cause a false miss. This is the ground-truth 'how many keys CAN match'."""
    c, p, r, a = _norm_keys(cohort)
    cohort_keys = pd.Series(
        c.astype(str) + "|" + p.astype("Int64").astype(str) + "|" + r + "|" + a
    )
    ik = (
        idx["_c"].astype(str) + "|" + idx["_p"].astype("Int64").astype(str)
        + "|" + idx["_r"].astype(str) + "|" + idx["_a"].astype(str)
    )
    cset = set(cohort_keys.tolist())
    iset = set(ik.tolist())
    inter = cset & iset
    # missense-restricted overlap
    mm = cohort["consequence"].fillna("").str.contains("missense", case=False)
    cohort_mm_keys = set(cohort_keys[mm.values].tolist())
    return {
        "cohort_rows": len(cohort),
        "cohort_unique_keys": len(cset),
        "idx_unique_keys": len(iset),
        "overlap_all": len(inter),
        "missense_rows": int(mm.sum()),
        "overlap_missense": len(cohort_mm_keys & iset),
    }


def merge_via_connector(cohort: pd.DataFrame, cache_path: str) -> int | None:
    """Run the REAL ProteinCoordConnector.annotate_dataframe against the existing
    cache (cache_dir = the cache's folder; NO TSV path, so no rebuild). Returns the
    protein_pos non-null count, or None if the package can't be imported."""
    try:
        from genomic_variant_classifier.data.protein_coords import ProteinCoordConnector
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not import ProteinCoordConnector: {exc})")
        return None
    cache_dir = os.path.dirname(cache_path)
    # alphamissense_file=None + existing cache => annotate_dataframe loads cache, no rebuild.
    pc = ProteinCoordConnector(alphamissense_file=None, cache_dir=cache_dir)
    out = pc.annotate_dataframe(cohort.copy())
    if "protein_pos" not in out.columns:
        print("  connector returned NO protein_pos column (degradation path hit).")
        return 0
    return int(out["protein_pos"].notna().sum())


def merge_via_pandas(cohort: pd.DataFrame, idx: pd.DataFrame) -> int:
    """Replicate the connector's exact merge to expose a dtype-driven miss."""
    c, p, r, a = _norm_keys(cohort)
    left = pd.DataFrame({"_c": c.values, "_p": p.values, "_r": r.values, "_a": a.values})
    merged = left.merge(idx, on=["_c", "_p", "_r", "_a"], how="left")
    return int(merged["protein_pos"].notna().sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--cohort", default=COHORT_DEFAULT)
    ap.add_argument("--sample", type=int, default=0,
                    help="optional: sample N cohort rows for a fast first look (0=all)")
    args = ap.parse_args()

    if not os.path.isfile(args.cache):
        print(f"cache not found: {args.cache}")
        return 2
    if not os.path.isfile(args.cohort):
        print(f"cohort not found: {args.cohort}")
        return 2

    print("READ-ONLY. No build/write/delete.\n")
    idx = pd.read_parquet(args.cache)
    print(f"cache: {args.cache}")
    print(f"  rows={len(idx):,}  dtypes={{'_c': {idx['_c'].dtype}, '_p': {idx['_p'].dtype}, "
          f"'_r': {idx['_r'].dtype}, '_a': {idx['_a'].dtype}}}")

    cohort = pd.read_parquet(args.cohort, columns=["chrom", "pos", "ref", "alt", "consequence"])
    if args.sample and args.sample < len(cohort):
        cohort = cohort.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"  (sampled {len(cohort):,} cohort rows)")
    c, p, r, a = _norm_keys(cohort)
    print(f"cohort key dtypes: _c={c.dtype} _p={p.dtype} _r={r.dtype} _a={a.dtype}\n")

    print("=" * 64)
    print("[A] normalization-only key OVERLAP (string-cast; dtype-proof):")
    ov = overlap_via_strings(cohort, idx)
    for k, v in ov.items():
        print(f"    {k:>20}: {v:,}")

    print("\n[B] connector .merge() (real ProteinCoordConnector against this cache):")
    n_conn = merge_via_connector(cohort, args.cache)
    if n_conn is not None:
        print(f"    protein_pos non-null via connector: {n_conn:,}")

    print("\n[C] replicated pandas .merge() (exposes Int64-vs-int64 dtype miss):")
    n_pd = merge_via_pandas(cohort, idx)
    print(f"    protein_pos non-null via replicated merge: {n_pd:,}")

    print("\nVERDICT:")
    big = max(ov["overlap_all"], 1)
    if ov["overlap_all"] < 50_000:
        print("    (b) KEY-OVERLAP itself is small -> cache keys don't match the cleaned")
        print("        cohort. Rebuild the index keyed on the SAME cohort training uses;")
        print("        the 6/7 cache was built on clinvar_grch38.parquet, not the cleaned file.")
    elif n_pd is not None and n_pd < big * 0.5:
        print("    (a) Overlap is large but .merge() drops most -> MERGE-DTYPE bug")
        print("        (nullable Int64 vs int64). Cast keys to a common dtype before merge.")
    else:
        print("    (c) Local merge lands many but Run 15 logged 3,461 -> the Vast box ran an")
        print("        OLDER/smaller cache than this local one. Ship the right cache AND add a")
        print("        pre-train coverage assertion so a low-coverage merge fails loud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
