#!/usr/bin/env python3
"""verify_eve_esm2_coverage.py

Post-prep coverage gate for the newly-wired EVE / ESM-2 features. Fails LOUD if a
source was configured (path passed) yet covers a surprisingly small fraction of
*eligible* rows -- catching the silent-stub-by-another-route failure (e.g. a stale
or path-mismatched EVE dir, or a UniProt index that resolves almost nothing).

KEY DESIGN: the denominator is ELIGIBLE rows, not all rows. EVE and ESM-2 are
defined only for missense single-residue substitutions; on a whole-genome cohort
the large non-missense fraction (synonymous/intronic/indel/regulatory) CORRECTLY
gets the default value. Measuring coverage over all rows would false-FAIL. We use
the HGVSp-derived protein coordinates (protein_pos/wt_aa/mut_aa populated) as the
eligibility mask -- the exact rows where a non-default score is even possible.

This mirrors the existing min_protein_coord_coverage gate philosophy (loud FAIL,
never silent) but applies it to the *output* score columns after annotation.

Reads an on-disk split/features parquet (the prep output). READ-ONLY. No training.

Usage:
    python scripts/verify_eve_esm2_coverage.py \\
        --features outputs/run17_baseline/full/splits/train.parquet \\
        --eve-min 0.30 --esm2-min 0.30

Exit 0 = PASS (or source not configured -> SKIP). Non-zero = FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Default values the connectors emit when a row is not covered.
EVE_DEFAULT = 0.5
ESM2_DEFAULT = 0.0


def _eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where a non-default EVE/ESM-2 score is even possible: missense subs
    with populated protein coordinates (HGVSp parser output)."""
    if "protein_pos" in df.columns:
        m = df["protein_pos"].notna()
        # wt_aa/mut_aa present and distinct = a real substitution
        if "wt_aa" in df.columns and "mut_aa" in df.columns:
            m = m & df["wt_aa"].notna() & df["mut_aa"].notna()
        return m
    # Fall back to consequence if coords absent (older schema)
    if "consequence" in df.columns:
        return df["consequence"].astype(str).str.contains("missense", case=False, na=False)
    return pd.Series([False] * len(df), index=df.index)


def _coverage(df: pd.DataFrame, col: str, default: float, eligible: pd.Series) -> tuple[int, int, float]:
    """(covered, eligible_n, fraction) where covered = eligible rows whose score
    differs from the default (i.e. a real lookup hit)."""
    if col not in df.columns:
        return (0, int(eligible.sum()), 0.0)
    elig_n = int(eligible.sum())
    if elig_n == 0:
        return (0, 0, 0.0)
    sub = df.loc[eligible, col]
    covered = int((sub.notna() & (sub != default)).sum())
    return (covered, elig_n, covered / elig_n if elig_n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="prep-output parquet with feature columns")
    ap.add_argument("--eve-min", type=float, default=0.30,
                    help="min covered-fraction of eligible rows for eve_score (default 0.30)")
    ap.add_argument("--esm2-min", type=float, default=0.30,
                    help="min covered-fraction for esm2_delta_norm (default 0.30)")
    ap.add_argument("--eve-configured", action="store_true",
                    help="assert EVE WAS configured (so absence of coverage is a FAIL, not a SKIP)")
    ap.add_argument("--esm2-configured", action="store_true",
                    help="assert ESM-2 WAS configured")
    ns = ap.parse_args()

    path = Path(ns.features)
    if not path.exists():
        print(f"FAIL: features parquet not found: {path}")
        return 2

    df = pd.read_parquet(path)
    elig = _eligible_mask(df)
    n = len(df)
    print(f"rows={n}  eligible(missense w/ coords)={int(elig.sum())} "
          f"({100*elig.mean():.1f}% of cohort -- low is EXPECTED on whole-genome)")

    rc = 0
    for col, default, thresh, configured in [
        ("eve_score", EVE_DEFAULT, ns.eve_min, ns.eve_configured),
        ("esm2_delta_norm", ESM2_DEFAULT, ns.esm2_min, ns.esm2_configured),
    ]:
        covered, elig_n, frac = _coverage(df, col, default, elig)
        present = col in df.columns
        if not configured:
            print(f"  SKIP {col:<16} (not asserted-configured; covered={covered}/{elig_n} "
                  f"frac={frac:.3f})")
            continue
        if not present:
            print(f"  FAIL {col:<16} configured but COLUMN ABSENT")
            rc = 4
            continue
        status = "PASS" if frac >= thresh else "FAIL"
        if status == "FAIL":
            rc = 4
        print(f"  {status} {col:<16} covered={covered}/{elig_n} frac={frac:.3f} "
              f"(min {thresh:.2f})")
        if status == "FAIL":
            print(f"       -> {col} covers only {frac:.1%} of eligible missense rows. "
                  f"Likely a stale/path-mismatched source or unresolved gene symbols. "
                  f"Investigate BEFORE trusting the run (this is the silent-stub trap).")

    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
