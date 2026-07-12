"""
diagnose_identity_join.py  (2026-07-09)
==========================================================================
Explain, with RAW DATA rather than reasoning, why harden_recovery_identity.py accepted
0 of 3,950 wide-window recoveries -- a result that is almost certainly a silent join
failure, not a true signal.

For a sample of RECOVER rows from the disposition, dump side by side:
  (1) cohort: variant_id, chrom, pos, gene_symbol, pathogenicity
  (2) disposition: recovered variation_id (with exact repr + python type), vcf_pos, and
      the RECOMPUTED true minimal offset (closest VCF pos, not first-in-window)
  (3) variant_summary rows for that chrom within +/-30 of pos: VariationID, Start, Gene,
      Type  -> reveals whether Start == pos / pos+1, VariationID format, gene spelling
  (4) raw VCF lines for that chrom within +/-30 of pos: POS, ID, REF, ALT, and the FULL
      INFO string -> reveals the actual INFO keys (GENEINFO? CLNSIG? other) and ID format

Also computes, across ALL recovered rows, the true minimal-offset distribution so we know
how many are really pos/pos+1 (offset<=1) vs genuinely wide.

Read-only. Writes outputs/identity_join_diagnosis.txt (+ prints a summary).

USAGE
  python scripts/diagnose_identity_join.py \
      --disposition     outputs/alleleless_final_disposition.tsv \
      --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --raw-vcf         data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --assembly GRCh38 --sample 25
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import sys
from pathlib import Path

import pandas as pd


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _index_vcf(path: Path):
    """chrom -> sorted list of (pos, id, ref, alt, info)."""
    by = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            c = _norm_chrom(f[0])
            try:
                p = int(f[1])
            except ValueError:
                continue
            by.setdefault(c, []).append((p, f[2], f[3], f[4], f[7]))
    for c in by:
        by[c].sort()
    return by


def _load_vs(path: Path, assembly: str) -> pd.DataFrame:
    want = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start", "Stop"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


def _closest_offset(vcf_by_chrom, chrom, pos, win=25):
    c = _norm_chrom(chrom)
    rows = vcf_by_chrom.get(c)
    if not rows:
        return None
    positions = [r[0] for r in rows]
    lo = bisect.bisect_left(positions, pos - win)
    hi = bisect.bisect_right(positions, pos + win)
    best = None
    for (p, *_rest) in rows[lo:hi]:
        d = abs(p - pos)
        if best is None or d < best:
            best = d
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disposition", default="outputs/alleleless_final_disposition.tsv")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--win", type=int, default=30)
    ap.add_argument("--out", default="outputs/identity_join_diagnosis.txt")
    a = ap.parse_args(argv)

    disp = pd.read_csv(a.disposition, sep="\t")
    rec = disp[disp["bucket"] == "RECOVER"].copy()
    coh = pd.read_parquet(a.cohort)
    gene_by_id = dict(zip(coh["variant_id"], coh.get("gene_symbol", pd.Series(dtype=str))))
    path_by_id = dict(zip(coh["variant_id"], coh.get("pathogenicity", pd.Series(dtype=str))))

    print("indexing raw VCF ...")
    vcf = _index_vcf(Path(a.raw_vcf))
    print("loading variant_summary ...")
    vs = _load_vs(Path(a.variant_summary), a.assembly)
    vs["_c"] = vs["Chromosome"].map(_norm_chrom)
    vs["_start_int"] = pd.to_numeric(vs["Start"], errors="coerce")

    # recompute true minimal offset for every recovered row
    rec["_true_offset"] = [
        _closest_offset(vcf, c, int(p), a.win)
        for c, p in zip(rec["chrom"], rec["pos"].astype(int))
    ]
    off = rec["_true_offset"].dropna()
    off_dist = {
        "offset==0": int((off == 0).sum()),
        "offset==1": int((off == 1).sum()),
        "offset<=1": int((off <= 1).sum()),
        "offset 2..5": int(((off >= 2) & (off <= 5)).sum()),
        "offset 6..25": int(((off >= 6) & (off <= 25)).sum()),
        "no_vcf_within_win": int(rec["_true_offset"].isna().sum()),
    }

    lines = []
    lines.append("IDENTITY-JOIN DIAGNOSIS  (2026-07-09)")
    lines.append("=" * 60)
    lines.append(f"RECOVER rows: {len(rec):,}")
    lines.append(f"TRUE minimal-offset distribution (closest VCF pos, not first-in-window):")
    for k, v in off_dist.items():
        lines.append(f"    {k:20s}: {v:,}")
    lines.append("")
    lines.append("If offset<=1 is MUCH larger than the 466 'exempt' reported by the harden")
    lines.append("tool, then the harden tool used a stale first-in-window vcf_pos -> BUG.")
    lines.append("")

    # sample wide rows (true offset >=2) for the side-by-side dump
    wide = rec[rec["_true_offset"].fillna(99) >= 2].head(a.sample)
    lines.append(f"--- SAMPLE OF {len(wide)} WIDE ROWS (true offset >=2), RAW DATA ---")
    for _, r in wide.iterrows():
        vid = r["variant_id"]; chrom = _norm_chrom(r["chrom"]); pos = int(r["pos"])
        lines.append("")
        lines.append(f"cohort   : {vid}  gene={gene_by_id.get(vid)!r}  "
                     f"path={path_by_id.get(vid)!r}")
        lines.append(f"disp     : recovered variation_id={r['variation_id']!r} "
                     f"(type={type(r['variation_id']).__name__})  vcf_pos={r['vcf_pos']}  "
                     f"true_offset={r['_true_offset']}")
        near = vs[(vs["_c"] == chrom) & (vs["_start_int"].between(pos - a.win, pos + a.win))]
        if len(near):
            lines.append(f"  variant_summary rows near {chrom}:{pos} (VariationID | Start | Gene | Type):")
            for _, x in near.head(6).iterrows():
                lines.append(f"     {x.get('VariationID')!r:>12} | {x.get('Start')!r:>10} | "
                             f"{x.get('GeneSymbol')!r:>12} | {x.get('Type')!r}")
        else:
            lines.append(f"  variant_summary: NO rows within +/-{a.win} of {chrom}:{pos} "
                         f"(=> Start-key join could never match here)")
        rows = vcf.get(chrom, [])
        positions = [rr[0] for rr in rows]
        lo = bisect.bisect_left(positions, pos - a.win)
        hi = bisect.bisect_right(positions, pos + a.win)
        lines.append(f"  raw VCF lines near {chrom}:{pos} (POS | ID | REF | ALT | INFO[:80]):")
        for (p, vid_, ref_, alt_, info_) in rows[lo:hi][:6]:
            lines.append(f"     {p:>10} | {vid_!r:>10} | {ref_[:12]!r} | {alt_[:12]!r} | "
                         f"{info_[:80]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:14]))
    print(f"\nfull side-by-side dump -> {a.out}")
    print("\nINTERPRETATION KEYS:")
    print("  * if variant_summary Start != pos and != pos+1 for these rows -> the identity-1")
    print("    lookup key is wrong (Start uses a different coordinate than cohort pos).")
    print("  * if recovered variation_id is a float like 12345.0 -> type mismatch vs the")
    print("    string VariationID in variant_summary -> equality always False.")
    print("  * if the VCF INFO has no GENEINFO=/CLNSIG= -> identity-2 regex finds nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
