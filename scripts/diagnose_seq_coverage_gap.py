"""
diagnose_seq_coverage_gap.py  (2026-07-09)  -- READ-ONLY. Writes one TSV of findings.
==========================================================================
The rekey verify reported 21,091 total unmapped and 5 padded-deletion unmapped when the
rekeyed windows are attached to cohort-v2. Neither is necessarily a rekey failure -- the
seq parquet has 4,399,089 rows vs the cohort's 4,420,180 (a 21,091 gap), so some cohort
rows simply have no window. This tool itemizes exactly which rows are unmapped and WHY,
so each has an explicit disposition instead of a tolerated count.

Two categories, kept separate:
  A. COVERAGE GAP -- the cohort row's key is absent from the seq parquet entirely
     (the row was never given a window; e.g. added to the cohort after the seq build,
     or on a contig the window builder skipped). Expected to be ~21,091.
  B. KEY MISMATCH -- a padded deletion whose cohort-v2 key does not equal its rekeyed
     seq key despite both existing. This would be a real rekey defect and must be zero
     (or fully explained).

USAGE
    python scripts/diagnose_seq_coverage_gap.py \\
        --cohort data/processed/clinvar_grch38_clean_v2_verified.parquet \\
        --seq    data/processed/clinvar_grch38_clean_seq.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _startswith_elementwise(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return pd.Series([rr.startswith(aa) for rr, aa in zip(r, a)], index=ref.index, dtype=bool)


def is_padded_deletion(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return (a.str.len() < r.str.len()) & _startswith_elementwise(r, a)


def _key(df: pd.DataFrame) -> pd.Series:
    return (df["chrom"].astype(str) + ":" + df["pos"].astype(str)
            + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--seq", default="data/processed/clinvar_grch38_clean_seq.parquet")
    ap.add_argument("--out", default="outputs/seq_coverage_gap.tsv")
    a = ap.parse_args(argv)

    cohort = pd.read_parquet(a.cohort, columns=["variant_id", "chrom", "pos", "ref", "alt"])
    seq = pd.read_parquet(a.seq, columns=["chrom", "pos", "ref", "alt"])

    # Apply the SAME rekey to the seq parquet's keys so we compare cohort-v2 to rekeyed-seq
    seq = seq.copy()
    dmask = is_padded_deletion(seq["ref"], seq["alt"])
    seq.loc[dmask, "pos"] = seq.loc[dmask, "pos"] - 1
    seq_keys = set(_key(seq))

    cohort = cohort.copy()
    cohort["_key"] = _key(cohort)
    cohort["_is_pdel"] = is_padded_deletion(cohort["ref"], cohort["alt"])
    cohort["_mapped"] = cohort["_key"].isin(seq_keys)

    unmapped = cohort[~cohort["_mapped"]]
    unmapped_pdel = unmapped[unmapped["_is_pdel"]]

    print("=" * 74)
    print("SEQ COVERAGE-GAP DIAGNOSTIC")
    print("=" * 74)
    print(f"cohort rows           : {len(cohort):,}")
    print(f"seq parquet rows      : {len(seq):,}")
    print(f"row-count gap         : {len(cohort) - len(seq):,}")
    print(f"cohort rows unmapped  : {len(unmapped):,}")
    print(f"  of which padded del : {len(unmapped_pdel):,}")
    print()

    # Category A vs B: is the unmapped row's (chrom,ref,alt) present in seq at ANY pos?
    seq_cra = set(seq["chrom"].astype(str) + "|" + seq["ref"].astype(str) + "|" + seq["alt"].astype(str))
    unmapped = unmapped.assign(
        _cra=unmapped["chrom"].astype(str) + "|" + unmapped["ref"].astype(str) + "|" + unmapped["alt"].astype(str))
    unmapped = unmapped.assign(
        category=["KEY_MISMATCH" if c in seq_cra else "COVERAGE_GAP" for c in unmapped["_cra"]])

    n_gap = int((unmapped["category"] == "COVERAGE_GAP").sum())
    n_mis = int((unmapped["category"] == "KEY_MISMATCH").sum())
    print(f"category A COVERAGE_GAP (row absent from seq entirely): {n_gap:,}")
    print(f"category B KEY_MISMATCH (present but key differs)     : {n_mis:,}")
    print()
    print("The 5 padded-deletion unmapped, itemized:")
    cols = ["variant_id", "chrom", "pos", "ref", "alt", "category"]
    print(unmapped_pdel.merge(unmapped[["_key", "category"]], on="_key", how="left")
          [["variant_id", "chrom", "pos", "ref", "alt", "category"]].to_string()
          if len(unmapped_pdel) else "  (none)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    unmapped[["variant_id", "chrom", "pos", "ref", "alt", "_is_pdel", "category"]].to_csv(
        a.out, sep="\t", index=False)
    print(f"\nAll {len(unmapped):,} unmapped rows written to {a.out}")
    print()
    if n_mis == 0:
        print("VERDICT: every unmapped row is a COVERAGE GAP (row absent from the seq")
        print("parquet), NOT a rekey key mismatch. The rekey is correct. The gap is a")
        print("pre-existing coverage question: these cohort rows never had a window.")
    else:
        print(f"VERDICT: {n_mis} KEY_MISMATCH rows exist -- these are present in the seq")
        print("parquet under a different key than cohort-v2 expects. Investigate before")
        print("trusting the rekey for those rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
