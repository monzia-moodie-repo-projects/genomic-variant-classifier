#!/usr/bin/env python3
"""
scripts/diagnose_esm2_coverage.py  (v2)
=======================================
Measure-first probe for the ESM-2 coverage gap (Run 15 scored only ~3,451).
v2 adds the columns ESM-2 actually consumes in this pipeline
(fasta_seq / fasta_seq_ref / fasta_seq_alt) and scans every text column for an
embedded HGVSp (p.Xxx###Yyy), since ClinVar carries the protein change inside
its free-text `Name`, not in a dedicated column.

Reports, without assuming column names:
  - schema + dtypes
  - role columns found (HGVSp / protein_pos / wt_aa / mut_aa / consequence / seq windows)
  - coverage (non-null / non-zero|non-empty) for each
  - missense count
  - of missense: how many have a populated fasta_seq_alt  (== current ESM-2 ceiling)
  - any text column containing an embedded p.HGVSp, with a recoverable-by-parse estimate

Run from repo root:
    python scripts/diagnose_esm2_coverage.py
    python scripts/diagnose_esm2_coverage.py --clinvar data/processed/clinvar_grch38_clean.parquet \
                                             --seq     data/processed/clinvar_grch38_clean_seq.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

PAT_HGVSP   = re.compile(r"hgvs?_?p|protein_?change|p_?dot|aa_?change", re.I)
PAT_PROTPOS = re.compile(r"prot(ein)?_?pos|aa_?pos|residue|codon_?pos", re.I)
PAT_WTAA    = re.compile(r"\b(wt|ref|wild).*aa\b|aa_?ref|ref_?aa", re.I)
PAT_MUTAA   = re.compile(r"\b(mut|alt|var).*aa\b|aa_?alt|alt_?aa", re.I)
PAT_CONSEQ  = re.compile(r"conseq|molecular_?consequence|^mc$|effect|so_?term", re.I)
PAT_GENE    = re.compile(r"gene.*sym|^gene$|symbol", re.I)
PAT_ESM     = re.compile(r"esm2", re.I)
PAT_SEQWIN  = re.compile(r"fasta_?seq|seq_?ref|seq_?alt|seq_?window|aa_?seq", re.I)

# embedded HGVSp inside free text, e.g. "(p.Lys177Glu)" or "p.K177E"
EMBED_HGVSP = re.compile(r"p\.\(?\s*([A-Z][a-z]{2}|[A-Z])\d+([A-Z][a-z]{2}|[A-Z]|Ter|\*|=)")


def _cols(names, pat):
    return [c for c in names if pat.search(c)]


def _pct(n, d):
    return f"{(100.0 * n / d):.2f}%" if d else "n/a"


def _coverage(df, col):
    s = df[col]
    nn = int(s.notna().sum())
    out = f"non-null {nn:,} ({_pct(nn, len(df))})"
    if pd.api.types.is_numeric_dtype(s):
        nz = int((s.fillna(0) != 0).sum())
        out += f" | non-zero {nz:,} ({_pct(nz, len(df))})"
    else:
        ne = int(s.fillna('').astype(str).str.strip().ne('').sum())
        out += f" | non-empty {ne:,} ({_pct(ne, len(df))})"
    return out


def analyze(path: Path) -> None:
    print(f"\n{'='*70}\nFILE: {path}\n{'='*70}")
    if not path.exists():
        print("  MISSING - skipped.")
        return

    names = None
    if pq is not None:
        try:
            names = list(pq.ParquetFile(str(path)).schema.names)
        except Exception as exc:
            print(f"  (pyarrow schema read failed: {exc})")
    if names is None:
        names = list(pd.read_parquet(path).columns)

    print(f"\n-- schema: {len(names)} columns --")
    for c in names:
        print(f"   {c}")

    role = {
        "HGVSp": _cols(names, PAT_HGVSP), "protein_pos": _cols(names, PAT_PROTPOS),
        "wt_aa": _cols(names, PAT_WTAA), "mut_aa": _cols(names, PAT_MUTAA),
        "consequence": _cols(names, PAT_CONSEQ), "gene": _cols(names, PAT_GENE),
        "esm2": _cols(names, PAT_ESM), "seq_window": _cols(names, PAT_SEQWIN),
    }
    print("\n-- candidate columns by role --")
    for r, cols in role.items():
        print(f"   {r:12}: {cols if cols else '(none found)'}")

    # load text-ish + role columns; for the embedded scan we also want any object cols
    df_head = pd.read_parquet(path) if pq is None else None
    obj_cols = []
    if pq is not None:
        sch = pq.ParquetFile(str(path)).schema_arrow
        for f in sch:
            if str(f.type) in ("string", "large_string"):
                obj_cols.append(f.name)
    need = sorted(set(sum(role.values(), []) + obj_cols))
    df = pd.read_parquet(path, columns=need)
    n = len(df)

    print(f"\n-- coverage over {n:,} rows --")
    for c in sorted(set(sum(role.values(), []))):
        print(f"   {c:28}: {_coverage(df, c)}")

    # missense subset
    miss = None
    if role["consequence"]:
        cc = role["consequence"][0]
        miss = df[cc].fillna('').astype(str).str.contains("missense", case=False)
        print(f"\n-- missense (via '{cc}'): {int(miss.sum()):,} ({_pct(int(miss.sum()), n)}) --")

    # ESM-2 ceiling: missense with a populated mutant seq window
    alt_win = [c for c in role["seq_window"] if re.search(r"alt", c, re.I)]
    if miss is not None and alt_win:
        aw = alt_win[0]
        have = df[aw].fillna('').astype(str).str.strip().ne('')
        print(f"   missense WITH {aw}: {int((miss & have).sum()):,}  <- current ESM-2 ceiling")
        print(f"   missense MISSING {aw}: {int((miss & ~have).sum()):,}  <- the gap")
    elif role["seq_window"]:
        for c in role["seq_window"]:
            print(f"   {c}: {_coverage(df, c)}")

    # embedded-HGVSp scan across text columns (sampled for speed)
    print("\n-- embedded p.HGVSp scan (text columns) --")
    samp = df if n <= 300_000 else df.sample(300_000, random_state=0)
    found_any = False
    for c in obj_cols:
        s = samp[c].dropna().astype(str)
        if s.empty:
            continue
        hits = s.str.contains(EMBED_HGVSP)
        h = int(hits.sum())
        if h:
            found_any = True
            ex = s[hits].iloc[0][:80]
            print(f"   {c:28}: {h:,}/{len(s):,} sampled rows carry p.HGVSp  e.g. {ex!r}")
            if miss is not None:
                mhits = samp[c].fillna('').astype(str).str.contains(EMBED_HGVSP) & miss.loc[samp.index]
                print(f"        -> of sampled missense: {int(mhits.sum()):,} parseable (scale by {n/len(samp):.1f}x)")
    if not found_any:
        print("   none - HGVSp is not present in any text column of THIS file")
        print("   (expected for the cleaned parquet; the p.HGVSp lives in the RAW ClinVar 'Name' field)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar", default="data/processed/clinvar_grch38_clean.parquet")
    ap.add_argument("--seq", default="data/processed/clinvar_grch38_clean_seq.parquet")
    args = ap.parse_args()
    analyze(Path(args.clinvar))
    analyze(Path(args.seq))
    print("\nDone. If the cleaned files carry no p.HGVSp, run clinvar_name_probe.py on the RAW variant_summary.\n")


if __name__ == "__main__":
    main()
