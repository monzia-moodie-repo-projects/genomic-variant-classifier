#!/usr/bin/env python
"""probe_indel_convention.py (2026-07-10) -- Step C: pin the indel coordinate convention.

SNV windows are proven 100% correct (ref base at pos-1). Indel ref_mismatches (~7.2%) appear to use
a different anchoring. Before writing indel-aware logic (Step B), determine PRECISELY and with
evidence: for indel variants, at what offset does the genome match the cohort ref, and is that
offset CONSISTENT across indels? Also test the standard VCF left-anchor hypothesis directly.

For each sampled indel, tries a set of candidate alignments and records which one makes the genome
equal the cohort ref:
  - off=0  : genome[pos-1 : pos-1+len(ref)] == ref      (the SNV convention)
  - off=-1 : genome[pos-2 : pos-2+len(ref)] == ref      (pos points one base right of ref start)
  - off=+1 : genome[pos   : pos  +len(ref)] == ref
  - anchor : VCF left-anchored -- ref[0] is a shared anchor base equal to genome[pos-1], and
             ref[1:]/alt[1:] are the actual change. Tests genome[pos-1]==ref[0].
  - trimmed: ClinVar first-changed-base -- genome[pos-1 : ...] == ref but with a leading shared
             base removed; tested via the -1/anchor combinations above.

Reports the DISTRIBUTION of which offset wins, so we know if it is a single consistent rule.
Read-only, indexed access, ASCII-safe.
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


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


def main() -> int:
    print("=" * 78)
    print("INDEL COORDINATE-CONVENTION PROBE (pin the anchor before building indel logic)")
    print("=" * 78)
    fa_path = None
    for c in ["data/external/grch38/GRCh38.fa", "data/external/grch38/GRCh38.fasta"]:
        if Path(c).exists():
            fa_path = Path(c); break
    if fa_path is None:
        print("ABORT: reference not found."); return 2
    try:
        import pyfaidx
    except Exception:
        print("ABORT: pyfaidx not installed."); return 3
    fa = pyfaidx.Fasta(str(fa_path), rebuild=False)

    def gslice(contig, start0, length):
        try:
            if start0 < 0 or length <= 0:
                return None
            return str(fa[str(contig)][start0:start0 + length]).upper()
        except Exception:
            return None

    import pandas as pd
    cohort = Path("data/processed/clinvar_grch38_pathfix.parquet")
    if not cohort.exists():
        cohort = Path("data/processed/clinvar_grch38.parquet")
    df = pd.read_parquet(cohort, columns=["chrom", "pos", "ref", "alt"])
    r = df["ref"].astype(str).str.upper()
    a = df["alt"].astype(str).str.upper()
    is_snv = (r.str.len() == 1) & (a.str.len() == 1) & r.str.match("^[ACGT]$") & a.str.match("^[ACGT]$")
    indels = df[~is_snv].copy()
    indels = indels[indels["ref"].astype(str).str.match("^[ACGT]+$")]  # only clean-base indels
    sample = indels.sample(n=min(5000, len(indels)), random_state=42).reset_index(drop=True)
    print(f"indel sample (clean ACGT ref): {len(sample):,}")
    line()

    win = Counter()
    anchor_ok = 0
    by_type = Counter()          # (vtype, offset) -> count
    type_totals = Counter()      # vtype -> count
    examples = {"off=0": [], "off=-1": [], "off=+1": [], "none": []}
    for _, row in sample.iterrows():
        c = row["chrom"]; ref = str(row["ref"]).upper(); alt = str(row["alt"]).upper()
        try:
            pos = int(row["pos"])
        except Exception:
            continue
        L = len(ref)
        if len(alt) > len(ref):
            vtype = "insertion"
        elif len(alt) < len(ref):
            vtype = "deletion"
        else:
            vtype = "mnv"
        type_totals[vtype] += 1
        got = {
            "off=0":  gslice(c, pos - 1, L),
            "off=-1": gslice(c, pos - 2, L),
            "off=+1": gslice(c, pos, L),
        }
        matched = None
        for k in ("off=0", "off=-1", "off=+1"):
            if got[k] == ref:
                matched = k; break
        g_anchor = gslice(c, pos - 1, 1)
        if g_anchor == ref[0]:
            anchor_ok += 1
        win[matched or "none"] += 1
        by_type[(vtype, matched or "none")] += 1
        if matched in examples and len(examples[matched]) < 5:
            examples[matched].append((c, pos, ref, alt, got["off=0"]))
        elif matched is None and len(examples["none"]) < 5:
            examples["none"].append((c, pos, ref, alt, got["off=0"], got["off=-1"], got["off=+1"]))

    n = len(sample)
    print("which alignment makes genome == cohort ref:")
    for k in ("off=0", "off=-1", "off=+1", "none"):
        cnt = win.get(k, 0)
        print(f"  {k:7s}: {cnt:5,} ({cnt/n*100:5.1f}%)")
    print(f"\nVCF left-anchor (genome[pos-1] == ref[0]): {anchor_ok:,}/{n:,} ({anchor_ok/n*100:.1f}%)")
    line()
    print("offset distribution BY variant type (the deterministic-rule check):")
    for vtype in ("insertion", "deletion", "mnv"):
        tot = type_totals.get(vtype, 0)
        if not tot:
            continue
        parts = []
        for off in ("off=0", "off=-1", "off=+1", "none"):
            cnt = by_type.get((vtype, off), 0)
            if cnt:
                parts.append(f"{off}={cnt}({cnt/tot*100:.0f}%)")
        print(_ascii_safe(f"  {vtype:9s} (n={tot:,}): {', '.join(parts)}"))
    line()
    for k in ("off=0", "off=-1", "off=+1"):
        if examples[k]:
            print(_ascii_safe(f"{k} examples (chrom,pos,ref,alt,genome@pos-1):"))
            for c, p, rf, al, g0 in examples[k]:
                print(_ascii_safe(f"    {c}:{p} ref={rf} alt={al} g@pos-1={g0}"))
    if examples["none"]:
        print("UNMATCHED examples (chrom,pos,ref,alt, g@0, g@-1, g@+1):")
        for c, p, rf, al, g0, gm1, gp1 in examples["none"]:
            print(_ascii_safe(f"    {c}:{p} ref={rf} alt={al}  g0={g0} g-1={gm1} g+1={gp1}"))
    line("=")
    # verdict: is there a dominant consistent rule?
    best = max(win.items(), key=lambda kv: kv[1])
    frac = best[1] / n
    if best[0] != "none" and frac > 0.9:
        print(f"VERDICT: CONSISTENT -- {frac*100:.1f}% of indels align at {best[0]}. Indel logic")
        print(f"can use a single rule ({best[0]}). Proceed to build indel-aware handling (Step B).")
        return 0
    if best[0] != "none" and frac > 0.6:
        print(f"VERDICT: MOSTLY {best[0]} ({frac*100:.1f}%) but not universal. Build for {best[0]}")
        print("with the remainder falling back to poly (transparent). Acceptable.")
        return 0
    print(f"VERDICT: MIXED -- no single dominant offset (best {best[0]} {frac*100:.1f}%). Indels")
    print("use varied representations; recovery is more complex. Review examples before Step B.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
