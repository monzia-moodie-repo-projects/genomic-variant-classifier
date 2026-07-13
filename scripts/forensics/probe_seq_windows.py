#!/usr/bin/env python
"""probe_seq_windows.py (2026-07-10)

Verify the precondition that keeps cnn_1d (one-dimensional convolutional neural network) in the
re-baseline: usable reference/alternate sequence windows. train.py drops cnn_1d when meta_test
lacks REF_WIN_COL/ALT_WIN_COL with >100 non-null values. Since the re-baseline must retain all 13
models (no-model-ever-dropped directive), this probe checks where those windows come from and
whether the inputs can supply them.

Read-only. ASCII-safe on every printed line; stdout forced to replace un-encodable bytes.

Checks:
  1. seq_window_join module: the REF_WIN_COL / ALT_WIN_COL names and attach_delta_windows signature.
  2. The label-corrected cohort (clinvar_grch38_pathfix.parquet) and the plain clinvar parquet:
     do they carry the columns attach_delta_windows needs (e.g. fasta_seq_ref/alt, or the raw
     sequence inputs), and a genomic coordinate to build windows from?
  3. Any existing meta_test.parquet under outputs/: does it already have populated ref/alt windows?
     (informs whether a prior run produced them.)
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(".")


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def show_module_symbols():
    p = ROOT / "src/genomic_variant_classifier/data/seq_window_join.py"
    print("1. seq_window_join module")
    if not p.exists():
        print(_ascii_safe(f"   ABSENT: {p}"))
        line()
        return
    txt = p.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    print(_ascii_safe(f"   {p} : {len(lines)} lines"))
    rx = re.compile(r"REF_WIN_COL|ALT_WIN_COL|def attach_delta_windows|def .*window|"
                    r"fasta_seq|_WIN_COL\s*=|required|raise ", re.IGNORECASE)
    for i, ln in enumerate(lines, 1):
        if rx.search(ln):
            print(_ascii_safe(f"     L{i}: {ln.strip()[:140]}"))
    line()


def parquet_columns(path: Path, want_tokens):
    """Print columns of a parquet (schema only, cheap) and flag wanted tokens present."""
    if not path.exists():
        print(_ascii_safe(f"   ABSENT: {path}"))
        return
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        cols = list(schema.names)
    except Exception:
        try:
            import pandas as pd
            cols = list(pd.read_parquet(path).columns)
        except Exception as e:
            print(_ascii_safe(f"   READ ERROR {path.name}: {type(e).__name__}: {e}"))
            return
    print(_ascii_safe(f"   {path}  ({len(cols)} cols)"))
    hits = [c for c in cols if any(t.lower() in c.lower() for t in want_tokens)]
    print(_ascii_safe(f"     seq/coord-related cols: {hits if hits else 'NONE'}"))


def check_nonnull_windows(path: Path, ref_names, alt_names):
    """If a parquet has ref/alt window cols, report their non-null counts (the >100 gate)."""
    if not path.exists():
        return
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception as e:
        print(_ascii_safe(f"   READ ERROR {path.name}: {type(e).__name__}: {e}"))
        return
    ref = next((c for c in df.columns if c in ref_names), None)
    alt = next((c for c in df.columns if c in alt_names), None)
    if ref and alt:
        nr = int(df[ref].notna().sum())
        na = int(df[alt].notna().sum())
        gate = "PASS (>100 both)" if (nr > 100 and na > 100) else "FAIL (cnn_1d would drop)"
        print(_ascii_safe(f"   {path.name}: {ref} non-null={nr:,}  {alt} non-null={na:,}  -> {gate}"))
    else:
        print(_ascii_safe(f"   {path.name}: ref/alt window cols NOT present (would need attach step)"))


def main() -> int:
    print("=" * 78)
    print("SEQUENCE-WINDOW PRECONDITION PROBE (keep cnn_1d in the re-baseline)")
    print("=" * 78)
    show_module_symbols()

    seq_tokens = ["fasta_seq", "ref_win", "alt_win", "window", "seq_ref", "seq_alt",
                  "chrom", "pos", "ref", "alt"]
    print("2. Input cohort columns (can attach_delta_windows build windows from these?)")
    for rel in ["data/processed/clinvar_grch38_pathfix.parquet",
                "data/processed/clinvar_grch38.parquet"]:
        parquet_columns(ROOT / rel, seq_tokens)
    line()

    print("3. Existing meta_test / val parquets: do they already carry populated ref/alt windows?")
    ref_names = {"fasta_seq_ref", "ref_window", "REF_WIN", "seq_ref_window"}
    alt_names = {"fasta_seq_alt", "alt_window", "ALT_WIN", "seq_alt_window"}
    import glob
    metas = sorted(glob.glob("outputs/**/meta_test.parquet", recursive=True))[:6]
    metas += sorted(glob.glob("data/splits/meta_test.parquet"))
    if not metas:
        print("   no meta_test.parquet found under outputs/ or data/splits/")
    for m in metas:
        check_nonnull_windows(Path(m), ref_names, alt_names)
    line("=")
    print("PROBE COMPLETE. If windows are absent/low, the retrain must run attach_delta_windows so")
    print("cnn_1d is retained; if present >100, the precondition already holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
