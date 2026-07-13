#!/usr/bin/env python
"""diagnose_ref_mismatch.py (2026-07-10) -- close the last uncertainty before the full precompute.

The Phase A benchmark showed 3.8% ref_mismatch fallbacks. SNV center-match among BUILT windows was
100%, but that excludes any SNV that itself ref_mismatched. This breaks the ref_mismatch fallbacks
down by variant class (single-nucleotide vs indel/multi-nucleotide) so we know whether the
mismatches are expected indel-representation differences (benign) or include SNVs (a real
coordinate concern). Read-only, indexed access, ASCII-safe.

For a sample, it classifies every ref_mismatch and prints counts + concrete examples, and for a few
indel mismatches shows what the genome actually holds at the locus vs the cohort ref (to confirm the
left-anchor / trimming explanation).
"""
from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path("src").resolve()))


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def main() -> int:
    print("=" * 78)
    print("REF_MISMATCH DIAGNOSTIC (are any mismatches SNVs, or all indel-representation?)")
    print("=" * 78)
    fa_path = None
    for c in ["data/external/grch38/GRCh38.fa", "data/external/grch38/GRCh38.fasta"]:
        if Path(c).exists():
            fa_path = Path(c); break
    if fa_path is None:
        print("ABORT: reference not found."); return 2
    try:
        import pyfaidx
        from genomic_variant_classifier.data.delta_window_builder import build_window
    except Exception as e:
        print(_ascii_safe(f"ABORT: {e}")); return 3
    fa = pyfaidx.Fasta(str(fa_path), rebuild=False)

    def fetch(contig, start0, length):
        try:
            if start0 < 0:
                return None
            return str(fa[contig][start0:start0 + length])
        except Exception:
            return None

    def ref_base_at(contig, pos1, length):
        return fetch(str(contig), int(pos1) - 1, length)

    import pandas as pd
    cohort = Path("data/processed/clinvar_grch38_pathfix.parquet")
    if not cohort.exists():
        cohort = Path("data/processed/clinvar_grch38.parquet")
    df = pd.read_parquet(cohort, columns=["chrom", "pos", "ref", "alt"])
    sample = df.sample(n=min(30000, len(df)), random_state=42).reset_index(drop=True)

    snv_mm = 0
    indel_mm = 0
    snv_examples = []
    indel_examples = []
    for _, row in sample.iterrows():
        ref = str(row["ref"]).strip().upper()
        alt = str(row["alt"]).strip().upper()
        r = build_window(fetch, row["chrom"], row["pos"], ref, alt, 101)
        if r.ok or "ref_mismatch" not in r.reason:
            continue
        is_snv = len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT"
        if is_snv:
            snv_mm += 1
            if len(snv_examples) < 8:
                got = ref_base_at(row["chrom"], row["pos"], 1)
                snv_examples.append((row["chrom"], int(row["pos"]), ref, alt, got))
        else:
            indel_mm += 1
            if len(indel_examples) < 8:
                got = ref_base_at(row["chrom"], row["pos"], max(len(ref), 3))
                indel_examples.append((row["chrom"], int(row["pos"]), ref, alt, got))

    n = len(sample)
    print(f"sample size: {n:,}")
    print(f"ref_mismatch that are SNV  : {snv_mm}")
    print(f"ref_mismatch that are indel: {indel_mm}")
    line()
    if snv_examples:
        print("SNV ref_mismatch examples (chrom, pos, cohort_ref, alt, genome_base_at_pos):")
        for c, p, rf, al, got in snv_examples:
            print(_ascii_safe(f"    {c}:{p}  ref={rf} alt={al}  genome={got}"))
        print("  ^ IF these show genome_base != cohort_ref for SNVs, that's a real coordinate")
        print("    concern for a subset -- investigate before the full run.")
    else:
        print("NO SNV ref_mismatches found in the sample. All mismatches are indel/MNV.")
    line()
    if indel_examples:
        print("Indel ref_mismatch examples (chrom, pos, cohort_ref, alt, genome_slice):")
        for c, p, rf, al, got in indel_examples:
            print(_ascii_safe(f"    {c}:{p}  ref={rf} alt={al}  genome_slice={got}"))
        print("  ^ These are expected: indel ref/alt representation (anchor base, trimming) differs")
        print("    from the raw genome slice. Correctly fall back to poly WITH reason (not silent).")
    line("=")
    if snv_mm == 0:
        print("VERDICT: CLEAN -- every ref_mismatch is a non-SNV (indel/MNV representation). The")
        print("builder is correct for SNVs (the bulk). Proceed to the full precompute; indels that")
        print("cannot be cleanly windowed fall back to poly transparently and are a small minority.")
        return 0
    print(f"VERDICT: INVESTIGATE -- {snv_mm} SNV ref_mismatch(es) in sample. Examine the examples")
    print("above; a coordinate or contig edge case may affect a SNV subset.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
