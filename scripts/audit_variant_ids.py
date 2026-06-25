#!/usr/bin/env python3
r"""audit_variant_ids.py -- Author: Monzia Moodie

Read-only data-quality audit of variant_id chromosome tokens in a parquet.

Whole-genome scope: a valid variant_id is "<chrom>:<pos>:<ref>:<alt>" where <chrom> is one of
1..22, X, Y, M/MT (optionally 'chr'-prefixed). This surfaces the 3,778 'gnomad'-prefixed keys
observed in gnomad_v4_exomes.parquet as MALFORMED, and reports which contigs are present so a
missing autosome/sex chromosome (e.g. chrY) is visible too. Exits nonzero if any key is malformed
OR any expected contig is absent -- so it can gate a rebuild and never passes a defect silently.

Usage:
  python scripts/audit_variant_ids.py data/processed/gnomad_v4_exomes.parquet
  python scripts/audit_variant_ids.py <parquet> --column variant_id --expect-chrY
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

_AUTOSOMES = {str(i) for i in range(1, 23)}
_VALID = _AUTOSOMES | {"X", "Y", "M", "MT"}


def normalize_chrom(token: str) -> str:
    t = token.strip()
    if t[:3].lower() == "chr":
        t = t[3:]
    return t.upper() if t.upper() in {"X", "Y", "M", "MT"} else t


def audit(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        raise KeyError(f"column {column!r} not in parquet; have {list(df.columns)[:12]}...")
    s = df[column].astype("string")
    # chrom token = substring before first ':'  (no split of whole frame; vectorized)
    chrom = s.str.split(":", n=1).str[0].map(lambda x: normalize_chrom(x) if isinstance(x, str) else x)
    valid_mask = chrom.isin(_VALID)
    counts = chrom.value_counts(dropna=False).to_dict()
    malformed = chrom[~valid_mask]
    malformed_tokens = malformed.value_counts(dropna=False).to_dict()
    examples = s[~valid_mask].head(10).tolist()
    return {
        "n_total": int(len(s)),
        "n_valid": int(valid_mask.sum()),
        "n_malformed": int((~valid_mask).sum()),
        "counts_by_chrom": {str(k): int(v) for k, v in counts.items()},
        "malformed_tokens": {str(k): int(v) for k, v in malformed_tokens.items()},
        "malformed_examples": examples,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("--column", default="variant_id")
    ap.add_argument("--expect-chrY", action="store_true",
                    help="also require chrY present (whole-genome completeness)")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet, columns=[args.column])
    rep = audit(df, args.column)

    print(f"variant_id audit: {args.parquet}  (column={args.column})")
    print(f"  total={rep['n_total']}  valid={rep['n_valid']}  MALFORMED={rep['n_malformed']}")
    present = {c for c in rep["counts_by_chrom"] if c in _VALID}
    expected = _AUTOSOMES | {"X"} | ({"Y"} if args.expect_chrY else set())
    missing = sorted(expected - present, key=lambda c: (len(c), c))
    print(f"  contigs present: {len(present & (_AUTOSOMES|{'X','Y','M','MT'}))}  missing-expected: {missing or 'none'}")
    if rep["malformed_tokens"]:
        print("  malformed chrom tokens:", rep["malformed_tokens"])
        print("  examples:", rep["malformed_examples"][:5])

    problems = rep["n_malformed"] > 0 or bool(missing)
    if problems:
        print("NO_GO -- variant_id defects present (malformed keys and/or missing expected contig)")
        return 2
    print("GO -- all variant_id chrom tokens valid; expected contigs present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
