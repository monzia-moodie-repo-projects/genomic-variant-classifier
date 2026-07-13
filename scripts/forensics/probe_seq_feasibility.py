#!/usr/bin/env python
"""probe_seq_feasibility.py (2026-07-10)

Feasibility gate for building fasta_seq_ref/fasta_seq_alt delta windows (to fix the degenerate
cnn_1d, one-dimensional convolutional neural network, which currently trains on poly placeholders
at AUROC 0.5419). Verifies -- without guessing -- that the GRCh38 reference genome, an index, a
reader, and the coordinate/contig conventions all line up, using a reference-base-match test as
the decisive gate. READ-ONLY (indexed random access only; never loads the whole genome). ASCII-safe.

Checks:
  1. Locate + size the reference FASTA (data/external/grch38/GRCh38.fa and common alternatives).
  2. Detect a .fai index; report whether one exists.
  3. Detect an available FASTA reader (pyfaidx, then pysam, then Biopython).
  4. Report the reference's contig names (first few) -> 'chr1' vs '1' convention.
  5. Report the cohort 'chrom' convention (sample values).
  6. DECISIVE: for a sample of single-nucleotide cohort variants, fetch the reference base at
     (chrom, pos) under BOTH 1-based and 0-based interpretations and BOTH contig conventions,
     and report which combination makes the fetched base match the cohort 'ref' allele. The
     matching combination IS the correct convention for the builder.
  7. Scope: fraction of cohort rows that are clean single-nucleotide ACGT variants (SNVs) vs
     indels (which need different window logic).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def find_fasta():
    cands = [
        "data/external/grch38/GRCh38.fa",
        "data/external/grch38/GRCh38.fasta",
        "data/external/GRCh38.fa",
        "data/reference/GRCh38.fa",
    ]
    import glob
    cands += glob.glob("data/**/*.fa", recursive=True)
    cands += glob.glob("data/**/*.fasta", recursive=True)
    for c in cands:
        p = Path(c)
        if p.exists():
            return p
    return None


def get_reader():
    """Return (name, open_callable) for the first available FASTA reader."""
    try:
        import pyfaidx  # noqa
        return "pyfaidx", lambda path: __import__("pyfaidx").Fasta(str(path), rebuild=False)
    except Exception:
        pass
    try:
        import pysam  # noqa
        return "pysam", lambda path: __import__("pysam").FastaFile(str(path))
    except Exception:
        pass
    try:
        from Bio import SeqIO  # noqa
        return "biopython", None  # biopython has no cheap random access without an index build
    except Exception:
        pass
    return None, None


def fetch_base(reader_name, fa, contig, start0, length=1):
    """Fetch `length` bases starting at 0-based `start0` from `contig`. Returns str or None."""
    try:
        if reader_name == "pyfaidx":
            return str(fa[contig][start0:start0 + length]).upper()
        if reader_name == "pysam":
            return fa.fetch(contig, start0, start0 + length).upper()
    except Exception:
        return None
    return None


def main() -> int:
    print("=" * 78)
    print("SEQUENCE-WINDOW FEASIBILITY PROBE (can we build real ref/alt delta windows?)")
    print("=" * 78)

    # 1. locate FASTA
    fa_path = find_fasta()
    if fa_path is None:
        print("ABORT: no reference FASTA found under data/. Cannot build sequence windows.")
        return 2
    size_gb = fa_path.stat().st_size / 1e9
    print(_ascii_safe(f"1. reference FASTA: {fa_path}  ({size_gb:.2f} GB)"))

    # 2. index
    fai = Path(str(fa_path) + ".fai")
    print(_ascii_safe(f"2. .fai index: {'PRESENT' if fai.exists() else 'ABSENT (reader may build it)'}"))

    # 3. reader
    reader_name, opener = get_reader()
    print(_ascii_safe(f"3. FASTA reader available: {reader_name or 'NONE (need pyfaidx or pysam)'}"))
    if reader_name in (None, "biopython"):
        print("   Install pyfaidx (pip install pyfaidx) for indexed random access; stopping the")
        print("   coordinate checks here since cheap random access is unavailable.")
        return 3

    try:
        fa = opener(fa_path)
    except Exception as e:
        print(_ascii_safe(f"   reader open FAILED: {type(e).__name__}: {e}"))
        return 3

    # 4. reference contig names
    if reader_name == "pyfaidx":
        contigs = list(fa.keys())[:5]
    else:
        contigs = list(fa.references)[:5]
    print(_ascii_safe(f"4. reference contigs (first 5): {contigs}"))
    ref_has_chr = any(str(c).startswith("chr") for c in contigs)
    print(_ascii_safe(f"   reference uses 'chr' prefix: {ref_has_chr}"))
    line()

    # 5-7. load a cohort sample
    import pandas as pd
    cohort = Path("data/processed/clinvar_grch38_pathfix.parquet")
    if not cohort.exists():
        cohort = Path("data/processed/clinvar_grch38.parquet")
    if not cohort.exists():
        print("ABORT: no cohort parquet to test coordinates against.")
        return 2
    df = pd.read_parquet(cohort, columns=["chrom", "pos", "ref", "alt"])
    n = len(df)
    print(_ascii_safe(f"5. cohort: {cohort.name}  ({n:,} rows)"))
    print(_ascii_safe(f"   cohort chrom sample: {list(pd.Series(df['chrom'].astype(str).unique())[:6])}"))
    coh_has_chr = df["chrom"].astype(str).str.startswith("chr").any()
    print(_ascii_safe(f"   cohort uses 'chr' prefix: {coh_has_chr}"))

    # 7. SNV vs indel scope
    r = df["ref"].astype(str); a = df["alt"].astype(str)
    is_snv = (r.str.len() == 1) & (a.str.len() == 1) & r.str.match("^[ACGT]$") & a.str.match("^[ACGT]$")
    print(_ascii_safe(f"7. single-nucleotide ACGT variants: {int(is_snv.sum()):,} "
                      f"({is_snv.mean()*100:.1f}%); non-SNV/indel: {int((~is_snv).sum()):,}"))
    line()

    # 6. DECISIVE reference-base-match test on a sample of SNVs
    print("6. reference-base-match test (find the convention where fetched base == cohort ref):")
    sample = df[is_snv].dropna(subset=["chrom", "pos", "ref"]).head(200)
    combos = {}
    for chr_mode in ("as_is", "add_chr", "strip_chr"):
        for base_mode in ("1based", "0based"):
            match = 0; total = 0
            for _, row in sample.iterrows():
                c = str(row["chrom"])
                if chr_mode == "add_chr" and not c.startswith("chr"):
                    c = "chr" + c
                if chr_mode == "strip_chr" and c.startswith("chr"):
                    c = c[3:]
                try:
                    pos = int(row["pos"])
                except Exception:
                    continue
                start0 = pos - 1 if base_mode == "1based" else pos
                b = fetch_base(reader_name, fa, c, start0, 1)
                if b is None:
                    continue
                total += 1
                if b.upper() == str(row["ref"]).upper():
                    match += 1
            if total:
                combos[(chr_mode, base_mode)] = (match, total, match / total)
    if not combos:
        print("   NO fetches succeeded -- contig names never resolved. Convention mismatch or")
        print("   unreadable index. This is the blocker to fix before building.")
        line("=")
        print("FEASIBILITY: BLOCKED (no successful reference fetch).")
        return 1
    best = max(combos.items(), key=lambda kv: kv[1][2])
    for (cm, bm), (m, t, frac) in sorted(combos.items(), key=lambda kv: -kv[1][2]):
        flag = "  <== BEST" if (cm, bm) == best[0] else ""
        print(_ascii_safe(f"   chrom={cm:9s} coord={bm:7s}: {m}/{t} match ({frac*100:.1f}%){flag}"))
    line("=")
    (cm, bm), (m, t, frac) = best
    if frac > 0.95:
        print(f"FEASIBILITY: PASS -- convention chrom={cm}, coord={bm} matches ref base "
              f"{frac*100:.1f}% of the time.")
        print("The builder can construct correct windows using this convention. Proceed to build.")
        return 0
    print(f"FEASIBILITY: UNCERTAIN -- best match only {frac*100:.1f}%. Investigate before building")
    print("(possible reference build mismatch, masked bases, or cohort coordinate issues).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
