#!/usr/bin/env python
"""
diagnose_coordinate_convention.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
WHY THIS EXISTS

    check_grch38_fasta.py FAILED its known-base spot check (3 of 5 mismatched), and
    build_cohort_v2.py --genome then raised REFERENCE-CONSISTENCY GUARD FAILED on the
    padded deletions. The guard correctly refused to write. But the CAUSE is ambiguous:

      (a) the hardcoded KNOWN_BASES in the preflight are wrong -- typed from memory,
          which is exactly what this project forbids; OR
      (b) the coordinate convention (0- vs 1-based, chr-prefix) is off by one; OR
      (c) the genome is the wrong build; OR
      (d) the padded-deletion correction itself is wrong.

    Reasoning from seven examples cannot separate these. This tool does, by asserting
    NO hardcoded base. It uses SNVs as the ground-truth control.

THE CONTROL

    SNVs (len(ref) == len(alt) == 1) were NEVER shifted by build_cohort_v2 -- their pos
    was correct in v1 and is unchanged in v2. So for a correct genome read with the
    correct convention:

        genome[chrom][pos - 1] == ref        (VCF pos is 1-based; slice is 0-based)

    must hold for essentially ALL SNVs (barring a tiny rate of genuine build
    differences). This is a fact about the data, not a literal I typed.

WHAT IT DOES

    1. Samples SNVs from the cohort and tests SEVERAL conventions at once:
         pos-1  (standard 1-based VCF -> 0-based slice)
         pos    (if the cohort were already 0-based)
         pos-2  (double off-by-one)
       under both plain and chr-prefixed contig names. Reports the match rate of each.
       Exactly one convention should match ~100% of SNVs. That identifies the truth.

    2. Using the winning convention, checks:
         - padded_insertion at pos (never shifted): ref[0] should match
         - padded_deletion at the CORRECTED pos (v2): full ref should match
         - padded_deletion at the ORIGINAL pos (v1, = corrected+1): should NOT match
       This tells us whether the -1 correction moved deletions to the RIGHT place.

    3. Prints a definitive verdict on which of (a)-(d) is true.

USAGE
    python scripts/diagnose_coordinate_convention.py --genome data/external/grch38/GRCh38.fa
    python scripts/diagnose_coordinate_convention.py --genome ... --cohort data/processed/clinvar_grch38.parquet -n 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _open(genome: Path):
    try:
        import pysam  # type: ignore
        fa = pysam.FastaFile(str(genome))
        return (lambda c, s, e: fa.fetch(c, s, e)), set(fa.references), "pysam"
    except ImportError:
        import pyfaidx  # type: ignore
        fa = pyfaidx.Fasta(str(genome))
        return (lambda c, s, e: str(fa[c][s:e])), set(fa.keys()), "pyfaidx"


def _norm(c: object) -> str:
    s = str(c)
    return s[3:] if s.lower().startswith("chr") else s


def variant_class(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(alt) < len(ref) and ref.startswith(alt):
        return "padded_deletion"
    if len(ref) < len(alt) and alt.startswith(ref):
        return "padded_insertion"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38.parquet")
    ap.add_argument("-n", type=int, default=2000, help="SNVs to sample")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    gp, cp = Path(a.genome), Path(a.cohort)
    for p in (gp, cp):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    fetch, contigs, backend = _open(gp)
    print("=" * 78)
    print(f"COORDINATE-CONVENTION DIAGNOSTIC  ({backend})")
    print(f"  genome {gp}\n  cohort {cp}")
    print(f"  genome contigs: {len(contigs)}  sample: {sorted(contigs)[:6]}")
    print("=" * 78)

    df = pd.read_parquet(cp, columns=["chrom", "pos", "ref", "alt"])
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)
    df["cls"] = [variant_class(r, x) for r, x in zip(df["ref"], df["alt"])]

    def contig_of(c: str):
        c = _norm(c)
        for cand in (c, f"chr{c}"):
            if cand in contigs:
                return cand
        return None

    # ---- 1. find the convention that makes SNVs match ----------------------
    snv = df[df.cls == "SNV"].sample(min(a.n, (df.cls == "SNV").sum()), random_state=a.seed)
    print(f"\n--- SNV control (n={len(snv)}): which slice convention matches ref[0]? ---")
    conventions = {"pos-1 (1based VCF)": -1, "pos (0based)": 0, "pos-2": -2, "pos+1": +1}
    match = {k: 0 for k in conventions}
    total = 0
    missing_contig = 0
    for c, p, r in zip(snv["chrom"], snv["pos"], snv["ref"]):
        cc = contig_of(str(c))
        if cc is None:
            missing_contig += 1
            continue
        total += 1
        for name, off in conventions.items():
            # off is the shift from 1-based pos to 0-based index: index = pos + off
            idx = int(p) + off
            base = fetch(cc, idx, idx + 1).upper()
            if base == r.upper():
                match[name] += 1
    if missing_contig:
        print(f"  WARNING: {missing_contig} SNVs had a contig not in the genome")
    if total == 0:
        print("  FATAL: no SNV contigs matched the genome at all -- wrong contig naming or build.",
              file=sys.stderr)
        return 4
    best = max(match, key=match.get)
    for name in conventions:
        rate = 100 * match[name] / total
        flag = "  <-- WINNER" if name == best else ""
        print(f"    {name:20s} {match[name]:5d}/{total} = {rate:6.2f}%{flag}")
    best_rate = 100 * match[best] / total

    # interpret the winning offset (index = pos + off, 0-based)
    best_off = conventions[best]
    print(f"\n  winning convention: genome index = pos + ({best_off})  (0-based)")
    if best_rate < 95:
        print("  *** NO convention matches SNVs well. The genome is the WRONG BUILD, or")
        print("      contig naming/masking is broken. Do NOT trust any deletion result. ***")
        print("=" * 78)
        return 5
    if best_off != -1:
        print(f"  *** SNVs match at pos{best_off:+d}, NOT the standard pos-1. Either the cohort")
        print(f"      pos is not standard 1-based, or my reference_check slice is off. This is")
        print(f"      a CONVENTION bug in build_cohort_v2.reference_check, not a data bug. ***")

    # ---- 2. apply the winning convention to deletions ----------------------
    def read_at(chrom, pos, length, off):
        cc = contig_of(str(chrom))
        if cc is None:
            return None
        return fetch(cc, int(pos) + off, int(pos) + off + length).upper()

    print("\n--- padded deletions: does the CORRECTED pos (v2) put ref at the winning offset? ---")
    pdel = df[df.cls == "padded_deletion"].sample(
        min(500, (df.cls == "padded_deletion").sum()), random_state=a.seed)
    # in the SOURCE cohort, pos is still the UNCORRECTED (v1) Start. v2 = pos - 1.
    corr_match = orig_match = 0
    n = 0
    examples = []
    for c, p, r in zip(pdel["chrom"], pdel["pos"], pdel["ref"]):
        if contig_of(str(c)) is None:
            continue
        n += 1
        at_corrected = read_at(c, int(p) - 1, len(r), best_off)   # v2 position
        at_original = read_at(c, int(p), len(r), best_off)         # v1 position
        if at_corrected == r.upper():
            corr_match += 1
        if at_original == r.upper():
            orig_match += 1
        if len(examples) < 5:
            examples.append(f"    {c}:{p} {r[:20]}{'..' if len(r)>20 else ''}: "
                            f"v2(pos-1)={'HIT' if at_corrected==r.upper() else 'miss'} "
                            f"v1(pos)={'HIT' if at_original==r.upper() else 'miss'}")
    print(f"  corrected pos (v2, pos-1): {corr_match}/{n} = {100*corr_match/max(n,1):.2f}% match ref")
    print(f"  original  pos (v1, pos)  : {orig_match}/{n} = {100*orig_match/max(n,1):.2f}% match ref")
    for e in examples:
        print(e)

    # ---- 3. verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    if best_off == -1 and best_rate >= 99 and corr_match / max(n, 1) >= 0.95:
        print("VERDICT: convention is standard (pos-1), and the v2 correction is CORRECT.")
        print("  Padded deletions match the genome at the corrected pos. The earlier guard")
        print("  failure must have been the preflight's hardcoded KNOWN_BASES being wrong,")
        print("  not the cohort. Re-run the reference check; it should pass.")
    elif best_off == -1 and corr_match / max(n, 1) < 0.5 and orig_match / max(n, 1) >= 0.95:
        print("VERDICT: THE CORRECTION IS BACKWARDS. v1 (original pos) matches, v2 (pos-1)")
        print("  does not. Padded deletions were ALREADY correct; subtracting 1 broke them.")
        print("  The incident's direction was wrong. Revisit the Start/PositionVCF analysis.")
    elif best_off != -1 and best_rate >= 99:
        print(f"VERDICT: CONVENTION BUG. SNVs match at pos{best_off:+d}, not pos-1. My")
        print("  reference_check() slices at pos-1, so it read the wrong base for EVERY row.")
        print("  The genome is fine and the cohort may be fine; the CHECK is off by one.")
        print("  Fix reference_check to use the winning offset, then re-verify.")
    else:
        print("VERDICT: UNRESOLVED. SNV match rate or deletion pattern is inconsistent.")
        print(f"  best={best} @ {best_rate:.1f}%, v2 del match {100*corr_match/max(n,1):.1f}%.")
        print("  Inspect the examples; do not proceed to write cohort-v2.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
