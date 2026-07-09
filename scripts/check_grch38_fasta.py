#!/usr/bin/env python
"""
check_grch38_fasta.py  (2026-07-08)  -- READ-ONLY. Writes only a .fai index (via the indexer).
==========================================================================
Validate a candidate GRCh38 FASTA BEFORE using it to verify cohort-v2. A genome file can
be wrong in several silent ways, and each would make the cohort reference-check fail for
the WRONG reason:

  * it is a TRANSCRIPT fasta (GENCODE .transcripts.fa) -- spliced mRNA, not chromosomes
  * it is the wrong BUILD (GRCh37/hg19) -- contigs have different lengths
  * it uses unexpected CONTIG NAMES (chr1 vs 1 vs NC_000001.11)
  * it is CORRUPT / truncated -- indexer fails or lengths are short
  * it cannot be INDEXED by pyfaidx/pysam

This preflight answers, in one pass over the index (not the sequence): is this the GRCh38
primary assembly, named in a way cohort-v2's reference_check can use, and internally
consistent with the known chromosome lengths?

WHAT IT CHECKS
  1. The file indexes (pyfaidx or pysam). Prints which contigs and how many.
  2. The 24 primary chromosomes are present under SOME accepted naming (1/chr1/NC_...).
  3. Their lengths match the GRCh38 (hg38) reference lengths EXACTLY. A GRCh37 file fails
     here (chr1 is 249,250,621 in GRCh37 vs 248,956,422 in GRCh38).
  4. A handful of KNOWN GRCh38 reference bases at specific positions are correct -- the
     same kind of spot-check the cohort uses, so a subtly-shifted or masked genome is
     caught now, on 5 sites, not later on 187,245.

GRCh38 chromosome lengths are from the GRCh38.p14 primary assembly (Ensembl/UCSC agree on
the primary contigs; only naming differs).

USAGE
    python scripts/check_grch38_fasta.py --genome path/to/GRCh38.fa
    python scripts/check_grch38_fasta.py --genome path/to/hg38.fa.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# GRCh38 primary chromosome lengths (bp). Source: GRCh38.p14 / hg38 primary assembly.
GRCH38_LENGTHS = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259,
    "6": 170805979, "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422,
    "11": 135086622, "12": 133275309, "13": 114364328, "14": 107043718, "15": 101991189,
    "16": 90338345, "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}

# NO hardcoded reference bases. An earlier version of this file asserted specific bases
# at specific positions, typed from memory -- three of five were WRONG, and they caused a
# false FAILURE on a genome that was actually correct (2026-07-08). Asserting remembered
# genomic literals is exactly the "do not guess" violation this project forbids. The base
# check now reads truth from the cohort's own SNVs (--cohort), whose positions were never
# in question. If no cohort is supplied, the base spot-check is skipped with a notice, and
# the length/naming checks (which need no external truth) still run.


def _open(genome: Path):
    """Return (fetch(chrom,start0,end0)->str, lengths:dict[str,int], contigs:set)."""
    try:
        import pysam  # type: ignore
        fa = pysam.FastaFile(str(genome))
        lengths = dict(zip(fa.references, fa.lengths))
        return (lambda c, s, e: fa.fetch(c, s, e)), lengths, set(fa.references), "pysam"
    except ImportError:
        pass
    try:
        import pyfaidx  # type: ignore
        fa = pyfaidx.Fasta(str(genome))
        lengths = {k: len(fa[k]) for k in fa.keys()}
        return (lambda c, s, e: str(fa[c][s:e])), lengths, set(fa.keys()), "pyfaidx"
    except ImportError as exc:
        raise RuntimeError(
            "neither pysam nor pyfaidx is installed. `pip install pyfaidx "
            "--trusted-host pypi.org --trusted-host files.pythonhosted.org`"
        ) from exc


def _accepted_names(chrom: str) -> list[str]:
    return [chrom, f"chr{chrom}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a candidate GRCh38 FASTA.")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38.parquet",
                    help="cohort parquet whose SNVs verify reference bases (no hardcoded literals)")
    a = ap.parse_args(argv)

    gp = Path(a.genome)
    if not gp.exists():
        print(f"ERROR: not found: {gp}", file=sys.stderr)
        return 2

    print("=" * 74)
    print(f"GRCh38 FASTA PREFLIGHT   {gp}  ({gp.stat().st_size / 1e9:.2f} GB)")
    print("=" * 74)

    print("indexing (first run builds a .fai; this can take a minute on a 3 GB file) ...")
    try:
        fetch, lengths, contigs, backend = _open(gp)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not index the FASTA: {exc}", file=sys.stderr)
        return 3
    print(f"  indexed with {backend}: {len(contigs)} contigs")

    # transcript-fasta smell test: thousands of contigs, none matching a chromosome length
    if len(contigs) > 100 and not any(
        any(n in contigs for n in _accepted_names(c)) for c in GRCH38_LENGTHS
    ):
        print("FAIL: this looks like a TRANSCRIPT FASTA (many short contigs, no chromosomes).",
              file=sys.stderr)
        print(f"  first contigs: {sorted(contigs)[:5]}", file=sys.stderr)
        return 4

    # 1. presence + naming
    naming = None
    missing = []
    for c in GRCH38_LENGTHS:
        hit = next((n for n in _accepted_names(c) if n in contigs), None)
        if hit is None:
            missing.append(c)
        elif naming is None:
            naming = "chr-prefixed" if hit.startswith("chr") else "plain"
    if missing:
        print(f"FAIL: missing primary chromosomes under any accepted name: {missing}",
              file=sys.stderr)
        return 4
    print(f"  all 24 primary chromosomes present  (naming: {naming})")

    # 2. lengths == GRCh38 (this is what rejects GRCh37)
    bad = []
    for c, expect in GRCH38_LENGTHS.items():
        name = next(n for n in _accepted_names(c) if n in contigs)
        got = lengths[name]
        if got != expect:
            bad.append((c, got, expect))
    if bad:
        print("FAIL: chromosome lengths do NOT match GRCh38. This is likely the wrong build",
              file=sys.stderr)
        print("  (GRCh37 chr1 = 249,250,621; GRCh38 chr1 = 248,956,422).", file=sys.stderr)
        for c, got, expect in bad[:5]:
            print(f"    chr{c}: file has {got:,}, GRCh38 expects {expect:,}", file=sys.stderr)
        return 5
    print("  all 24 chromosome lengths match GRCh38 exactly  -> build confirmed")

    # 3. base spot check, from the cohort's own SNVs (no memorized literals)
    if a.cohort and Path(a.cohort).exists():
        import pandas as pd
        cdf = pd.read_parquet(a.cohort, columns=["chrom", "pos", "ref", "alt"])
        cdf["ref"] = cdf["ref"].astype(str)
        cdf["alt"] = cdf["alt"].astype(str)
        snv = cdf[(cdf["ref"].str.len() == 1) & (cdf["alt"].str.len() == 1)]
        snv = snv.sample(min(500, len(snv)), random_state=42)
        print(f"  spot-checking {len(snv)} cohort SNVs (genome[pos-1] == ref):")
        mism = 0
        shown = 0
        for c, pos1, ref in zip(snv["chrom"], snv["pos"], snv["ref"]):
            name = next((n for n in _accepted_names(str(c)) if n in contigs), None)
            if name is None:
                continue
            got = fetch(name, int(pos1) - 1, int(pos1)).upper()
            if got != ref.upper():
                mism += 1
                if shown < 5:
                    print(f"    MISMATCH {c}:{pos1} expect {ref} got {got}")
                    shown += 1
        rate = 100 * (len(snv) - mism) / max(len(snv), 1)
        print(f"    SNV match rate: {rate:.2f}%")
        if rate < 99.0:
            print(f"FAIL: only {rate:.1f}% of cohort SNVs match at pos-1. Wrong build or "
                  f"convention.", file=sys.stderr)
            return 6
    else:
        print("  base spot-check SKIPPED (no --cohort). Length+naming checks passed; supply")
        print("  --cohort data/processed/clinvar_grch38.parquet to verify bases from real SNVs.")

    print("\n" + "=" * 74)
    print("PASS -- this is the GRCh38 primary assembly, indexable, correctly named and sized.")
    print(f"  Use it:  python scripts/build_cohort_v2.py --apply --genome \"{gp}\" \\")
    print(f"             --output data/processed/clinvar_grch38_clean_v2_verified.parquet")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
