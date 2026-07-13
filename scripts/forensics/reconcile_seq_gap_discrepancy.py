"""
reconcile_seq_gap_discrepancy.py  (2026-07-09)  -- READ-ONLY.
==========================================================================
diagnose_seq_coverage_gap.py and rekey_seq_windows_v2.py --verify DISAGREE:
  diagnose  : padded-deletion unmapped = 0, COVERAGE_GAP = 21,091, KEY_MISMATCH = 0
  rekey     : padded-deletion unmapped = 5, COVERAGE_GAP = 0,      KEY_MISMATCH = 5

Both ran on the same cohort-v2 and the same v1 seq parquet. They cannot both be right.
This tool runs BOTH classification paths on ONE pair of frames and prints the exact 5
rows plus every fact needed to see which tool is correct and why.

The prime suspects, checked explicitly:
  * chrom dtype differs between cohort and seq (int vs str) -> key strings differ
  * pos dtype (int vs float '100' vs '100.0') -> key strings differ
  * duplicate keys collapsed by attach_delta_windows' drop_duplicates
  * the 5 are padded deletions in the cohort whose (chrom,ref,alt) IS in seq (so diagnose
    counts them mapped via a colliding key) but whose exact rekeyed key is NOT (so the join
    misses) -- i.e. a genuine key-construction difference between the two tools.

USAGE
    python scripts/reconcile_seq_gap_discrepancy.py \\
        --cohort data/processed/clinvar_grch38_clean_v2_verified.parquet \\
        --seq    data/processed/clinvar_grch38_clean_seq.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _pdel(df):
    r = df["ref"].astype(str); a = df["alt"].astype(str)
    return pd.Series([len(x) < len(y) and y.startswith(x) for x, y in zip(a, r)], index=df.index)


def _key(df):
    return (df["chrom"].astype(str) + ":" + df["pos"].astype(str)
            + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--seq", default="data/processed/clinvar_grch38_clean_seq.parquet")
    a = ap.parse_args(argv)

    cohort = pd.read_parquet(a.cohort, columns=["variant_id", "chrom", "pos", "ref", "alt"])
    seq = pd.read_parquet(a.seq, columns=["variant_id", "chrom", "pos", "ref", "alt"])

    print("=" * 74)
    print("RECONCILE SEQ-GAP DISCREPANCY")
    print("=" * 74)
    print(f"cohort dtypes: chrom={cohort['chrom'].dtype}  pos={cohort['pos'].dtype}")
    print(f"seq    dtypes: chrom={seq['chrom'].dtype}  pos={seq['pos'].dtype}")
    print(f"cohort rows {len(cohort):,}  seq rows {len(seq):,}")
    print()

    # rekey the seq exactly as both tools do
    seq_rk = seq.copy()
    dm = _pdel(seq_rk)
    seq_rk.loc[dm, "pos"] = seq_rk.loc[dm, "pos"] - 1
    seq_keys = set(_key(seq_rk))

    cohort = cohort.copy()
    cohort["_key"] = _key(cohort)
    cohort["_pdel"] = _pdel(cohort)
    cohort["_mapped"] = cohort["_key"].isin(seq_keys)

    unmapped = cohort[~cohort["_mapped"]]
    unmapped_pdel = unmapped[unmapped["_pdel"]]
    print(f"cohort padded deletions           : {int(cohort['_pdel'].sum()):,}")
    print(f"cohort padded deletions UNMAPPED   : {len(unmapped_pdel)}")
    print(f"total cohort rows unmapped         : {len(unmapped):,}")
    print()

    # duplicate-key check in the rekeyed seq (attach_delta_windows drops dup keys)
    dup = seq_rk.assign(_k=_key(seq_rk))["_k"].duplicated().sum()
    print(f"duplicate keys in rekeyed seq (dropped by the join): {int(dup):,}")
    print()

    if len(unmapped_pdel):
        print("The unmapped padded deletions (what rekey-verify flags):")
        for _, row in unmapped_pdel.head(20).iterrows():
            k = row["_key"]
            cra = f"{row['chrom']}|{row['ref']}|{row['alt']}"
            seq_cra = set(seq_rk["chrom"].astype(str) + "|" + seq_rk["ref"].astype(str)
                          + "|" + seq_rk["alt"].astype(str))
            in_cra = cra in seq_cra
            # is this row present in seq under its ORIGINAL (un-rekeyed) key?
            orig_key = f"{row['chrom']}:{int(row['pos'])+1}:{row['ref']}:{row['alt']}"
            in_seq_orig = orig_key in set(_key(seq))
            print(f"  {row['variant_id']}")
            print(f"    rekeyed key {k!r} in seq_keys: {k in seq_keys}")
            print(f"    (chrom,ref,alt) in seq: {in_cra}   -> {'KEY_MISMATCH' if in_cra else 'COVERAGE_GAP'}")
            print(f"    present in seq at pos+1 (original)?: {in_seq_orig}")
    else:
        print("No unmapped padded deletions found by THIS run.")
        print("=> diagnose_seq_coverage_gap.py is correct; the rekey-verify's 5 came from a")
        print("   different code path. Investigate the rekey-verify classification.")

    # Also directly count how the rekey-verify's classifier would score it
    print()
    print("Cross-check: does the cohort have padded deletions whose (chrom,ref,alt) appears")
    print("in seq but whose exact rekeyed key does not (the KEY_MISMATCH definition)?")
    seq_cra = set(seq_rk["chrom"].astype(str) + "|" + seq_rk["ref"].astype(str)
                  + "|" + seq_rk["alt"].astype(str))
    um_cra = (unmapped_pdel["chrom"].astype(str) + "|" + unmapped_pdel["ref"].astype(str)
              + "|" + unmapped_pdel["alt"].astype(str))
    km = int(um_cra.isin(seq_cra).sum()) if len(unmapped_pdel) else 0
    cg = int((~um_cra.isin(seq_cra)).sum()) if len(unmapped_pdel) else 0
    print(f"  KEY_MISMATCH = {km}   COVERAGE_GAP = {cg}")
    print()
    print("VERDICT:")
    if len(unmapped_pdel) == 0:
        print("  Zero unmapped padded deletions. The rekey is CORRECT. The rekey-verify's")
        print("  report of 5 must be a bug in ITS verify path (e.g. it built the meta frame")
        print("  or the window temp parquet differently). Fix the verify, not the rekey.")
    else:
        print(f"  {len(unmapped_pdel)} padded deletions genuinely do not map. Their category")
        print("  above tells whether it's a coverage gap (fine) or a real key defect (fix).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
