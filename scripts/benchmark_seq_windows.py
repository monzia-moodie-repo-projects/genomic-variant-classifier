#!/usr/bin/env python
"""benchmark_seq_windows.py (2026-07-10) -- Phase A of the delta-window builder hybrid.

Before committing to the full 4.4M-row precompute, MEASURE and VALIDATE on a real sample:
  1. Time N real pyfaidx fetches + window builds against GRCh38.fa -> project the full-run cost.
  2. Validate correctness on real data: for single-nucleotide variants, the built window's center
     base must equal the cohort ref (and the alt window's center the cohort alt). Report the
     match rate and the poly-fallback breakdown by reason -- nothing hidden.
  3. Recommend a chunk size for the resumable full precompute (Step 3) from the measured rate.

Read-only. Uses indexed random access (never loads the 3.15 GB genome). ASCII-safe.
"""
from __future__ import annotations

import io
import sys
import time
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
    print("PHASE A BENCHMARK -- delta-window build cost + correctness on real GRCh38")
    print("=" * 78)

    # locate reference
    fa_path = None
    for c in ["data/external/grch38/GRCh38.fa", "data/external/grch38/GRCh38.fasta"]:
        if Path(c).exists():
            fa_path = Path(c)
            break
    if fa_path is None:
        print("ABORT: GRCh38 reference FASTA not found.")
        return 2
    try:
        import pyfaidx
    except Exception:
        print("ABORT: pyfaidx not installed (pip install pyfaidx).")
        return 3
    try:
        from genomic_variant_classifier.data.delta_window_builder import build_window, POLY
    except Exception as e:
        print(_ascii_safe(f"ABORT: cannot import delta_window_builder: {e}"))
        return 3

    fa = pyfaidx.Fasta(str(fa_path), rebuild=False)

    def fetch(contig, start0, length):
        try:
            if start0 < 0:
                return None
            return str(fa[contig][start0:start0 + length])
        except Exception:
            return None

    import pandas as pd
    cohort = Path("data/processed/clinvar_grch38_pathfix.parquet")
    if not cohort.exists():
        cohort = Path("data/processed/clinvar_grch38.parquet")
    df = pd.read_parquet(cohort, columns=["chrom", "pos", "ref", "alt"])
    total_rows = len(df)
    print(_ascii_safe(f"cohort: {cohort.name}  ({total_rows:,} rows)"))
    print(_ascii_safe(f"reference: {fa_path}"))
    line()

    # Sample N rows at random (reproducible).
    N = 10000
    sample = df.sample(n=min(N, total_rows), random_state=42).reset_index(drop=True)
    W = 101
    HALF = W // 2

    # Time the build over the sample.
    t0 = time.perf_counter()
    ok = 0
    reasons = Counter()
    snv_center_ok = 0
    snv_total = 0
    for _, row in sample.iterrows():
        r = build_window(fetch, row["chrom"], row["pos"], row["ref"], row["alt"], W)
        if r.ok:
            ok += 1
            ref = str(row["ref"]).upper(); alt = str(row["alt"]).upper()
            if len(ref) == 1 and len(alt) == 1:
                snv_total += 1
                if r.ref_window[HALF] == ref and r.alt_window[HALF] == alt:
                    snv_center_ok += 1
        else:
            reasons[r.reason.split("(")[0]] += 1
    elapsed = time.perf_counter() - t0

    n = len(sample)
    rate = n / elapsed if elapsed > 0 else float("inf")
    print(f"built {n:,} windows in {elapsed:.2f}s  ->  {rate:,.0f} windows/sec")
    proj = total_rows / rate if rate > 0 else float("inf")
    print(f"projected full run ({total_rows:,} rows): {proj:,.0f}s = {proj/60:,.1f} min "
          f"= {proj/3600:.2f} h")
    line()

    print(f"correctness on sample:")
    print(f"  built from real reference (ok): {ok:,}/{n:,} ({ok/n*100:.1f}%)")
    if snv_total:
        print(f"  SNV center-base match (ref & alt): {snv_center_ok:,}/{snv_total:,} "
              f"({snv_center_ok/snv_total*100:.2f}%)")
    if reasons:
        print("  poly-fallback breakdown by reason:")
        for reason, cnt in reasons.most_common():
            print(_ascii_safe(f"    {reason}: {cnt:,} ({cnt/n*100:.1f}%)"))
    line()

    # Chunk-size recommendation for the resumable full precompute.
    # Aim for ~30-60s per chunk so a crash loses little; cap memory.
    per_chunk = max(50000, int(rate * 45))  # ~45s of work, min 50k
    n_chunks = (total_rows + per_chunk - 1) // per_chunk
    print(f"recommended chunk size for full precompute: {per_chunk:,} rows "
          f"(~{n_chunks} chunks, ~45s each)")
    line("=")
    verdict = "PASS" if (snv_total == 0 or snv_center_ok / snv_total > 0.99) else "CHECK"
    print(f"VERDICT: {verdict} -- "
          f"{'builder correct on real data; proceed to full precompute' if verdict=='PASS' else 'SNV center match below 99%; investigate before full run'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
