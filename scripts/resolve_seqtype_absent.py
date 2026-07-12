"""
resolve_seqtype_absent.py  (2026-07-09)
==========================================================================
Fully bucket the 19,988 allele-less (na:na) rows, with special focus on the
3,392 NEEDS_REVIEW_SEQTYPE_ABSENT_FROM_VCF rows -- variants whose ClinVar Type says
they are simple sequence variants (should have ref/alt) yet were absent from our local
clinvar_GRCh38.vcf.gz at pos / pos+1.

This tool runs, in order, the cheap local checks first and gates the network step:

  1a. STALE-VCF CHECK. Read the ##fileDate / ClinVar release from the raw VCF header and
      compare to the live variant_summary. If the raw VCF is OLDER than the summary, then
      sequence-typed rows absent from the raw VCF may simply be NEWER ClinVar records; a
      fresh VCF (--fresh-vcf) can recover them.
  1b. WIDENED POSITIONAL PROBE. Re-probe each unresolved row across a position window
      (pos-WIN .. pos+WIN) in the raw VCF, and in --fresh-vcf if supplied. Genome-verify
      every recovered ref (guards the padded-deletion shift and any wider offset).
  1c. STRUCTURAL CROSS-REF. Check whether the allele-less rows are present in
      clinvar_grch38_structural.parquet (clean_cohort.py's null/bad-allele bucket). This
      answers the upstream "why were they in the clean cohort without note" question:
        both  -> routing DUPLICATION (row emitted to clean AND structural)
        clean-only -> routing MISS (should have been diverted to structural)
  2.  NCBI RESOLUTION (only with --use-ncbi; needs internet). For rows still unresolved,
      query NCBI eutils esummary (db=clinvar) by VariationID to get authoritative Type,
      and the Variation Services SPDI endpoint for ref/alt. Rate-limited.

FINAL DISPOSITION: every allele-less row is written with exactly one bucket:
    RECOVER              (with genome-verified ref/alt to patch into the cohort)
    CONFIRMED_ALLELELESS (out-of-scope Type AND absent from every VCF checked)
    STILL_UNRESOLVED     (sequence-typed but not found anywhere -- needs manual/NCBI)

Outputs (outputs/):
    alleleless_final_disposition.tsv
    alleleless_final_disposition_summary.json
    alleleless_recovered_all.tsv   (all genome-verified recoveries: first pass + widened)

USAGE
    python scripts/resolve_seqtype_absent.py \
        --cohort           data/processed/clinvar_grch38_clean_v2_verified.parquet \
        --raw-vcf          data/raw/clinvar/clinvar_GRCh38.vcf.gz \
        --variant-summary  data/external/clinvar/variant_summary.txt.gz \
        --fasta            data/external/grch38/GRCh38.fa \
        --structural       data/processed/clinvar_grch38_structural.parquet \
        [--fresh-vcf data/external/clinvar/clinvar.vcf.gz] [--use-ncbi] \
        --assembly GRCh38 --win 25
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from genomic_variant_classifier.data.allele_classify import is_allele_less
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from allele_classify import is_allele_less  # type: ignore

OUT_OF_SCOPE_TYPES = {
    "copy number gain", "copy number loss", "translocation", "fusion", "complex",
    "microsatellite", "tandem repeat", "variation", "protein only", "inversion", "cytogenetic",
}
SEQUENCE_TYPES = {
    "single nucleotide variant", "deletion", "insertion", "duplication", "indel", "delins",
}
_NULL = {"", "na", "nan", "none", "-", "."}


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _real(x) -> bool:
    s = str(x).strip().lower() if x is not None else ""
    return s not in _NULL and len(s) >= 1


def _vcf_header_date(path: Path) -> str | None:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"##(fileDate|source)=?(.*)", line.strip())
            if line.startswith("##fileDate"):
                return line.strip().split("=", 1)[-1]
    return None


def _index_vcf_by_window(path: Path):
    """chrom -> sorted list of (pos, ref, alt, vid). Lets us probe a position window."""
    by_chrom = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            c = _norm_chrom(f[0])
            try:
                p = int(f[1])
            except ValueError:
                continue
            by_chrom.setdefault(c, []).append((p, f[3], f[4], f[2]))
    for c in by_chrom:
        by_chrom[c].sort()
    return by_chrom


def _open_ref(fasta: Path):
    from pyfaidx import Fasta
    return Fasta(str(fasta), rebuild=False)


def _load_vs(path: Path, assembly: str) -> pd.DataFrame:
    want = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start", "Stop"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


_FINAL_COLS = ["variant_id", "chrom", "pos", "type", "bucket", "ref", "alt",
               "variation_id", "vcf_pos", "source", "genome_verified"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--structural", default="data/processed/clinvar_grch38_structural.parquet")
    ap.add_argument("--use-ncbi", action="store_true")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--win", type=int, default=25, help="+/- position window for widened probe")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    coh = pd.read_parquet(a.cohort)
    al = coh[is_allele_less(coh["ref"], coh["alt"])].copy()
    print(f"allele-less rows: {len(al):,}")

    # ---- 1a. stale-VCF check ----
    raw_date = _vcf_header_date(Path(a.raw_vcf))
    print(f"[1a] raw VCF ##fileDate: {raw_date!r}")
    vs_mtime = time.strftime("%Y-%m-%d", time.gmtime(Path(a.variant_summary).stat().st_mtime))
    print(f"[1a] variant_summary file mtime (UTC): {vs_mtime}")
    stale = None
    if raw_date and re.match(r"\d{4}-?\d{2}-?\d{2}", raw_date):
        rd = re.sub(r"[^0-9]", "", raw_date)[:8]
        vd = re.sub(r"[^0-9]", "", vs_mtime)[:8]
        stale = rd < vd
        print(f"[1a] raw VCF {'OLDER than' if stale else 'not older than'} variant_summary "
              f"-> stale={stale}")

    # ---- recover Type per row from variant_summary (Start-join, both pos and pos+1) ----
    vs = _load_vs(Path(a.variant_summary), a.assembly)
    type_by_key = {}
    if {"Chromosome", "Start", "Type"} <= set(vs.columns):
        for c, s, t in zip(vs["Chromosome"].map(_norm_chrom), vs["Start"], vs["Type"]):
            type_by_key.setdefault(f"{c}:{s}", t)
    al["_type"] = [
        type_by_key.get(f"{_norm_chrom(c)}:{p}") or type_by_key.get(f"{_norm_chrom(c)}:{p+1}")
        for c, p in zip(al["chrom"], al["pos"].astype(int))
    ]

    # ---- 1b. widened positional probe (raw + optional fresh VCF), genome-verified ----
    ref_genome = _open_ref(Path(a.fasta)) if Path(a.fasta).exists() else None
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, ref) -> bool:
        if ref_genome is None:
            return False
        c = _norm_chrom(chrom)
        if c not in contigs:
            return False
        try:
            got = str(ref_genome[c][int(pos) - 1: int(pos) - 1 + len(ref)]).upper()
        except Exception:
            return False
        return got == str(ref).upper()

    vcf_indexes = [("raw", _index_vcf_by_window(Path(a.raw_vcf)))]
    if a.fresh_vcf and Path(a.fresh_vcf).exists():
        vcf_indexes.append(("fresh", _index_vcf_by_window(Path(a.fresh_vcf))))

    import bisect

    def probe(chrom, pos):
        """Return (ref,alt,vid,vcf_pos,source) genome-verified within +/- win, else None."""
        c = _norm_chrom(chrom)
        for source, idx in vcf_indexes:
            rows = idx.get(c)
            if not rows:
                continue
            positions = [r[0] for r in rows]
            lo = bisect.bisect_left(positions, pos - a.win)
            hi = bisect.bisect_right(positions, pos + a.win)
            for (p, vref, valt, vid) in rows[lo:hi]:
                if _real(vref) and _real(valt) and genome_ref_ok(chrom, p, vref):
                    return (vref, valt, vid, p, source)
        return None

    # ---- 1c. structural cross-ref ----
    struct_keys = set()
    if Path(a.structural).exists():
        st = pd.read_parquet(a.structural, columns=["variant_id"]) \
            if "variant_id" in pd.read_parquet(a.structural, columns=None).columns else None
        if st is not None:
            struct_keys = set(st["variant_id"].astype(str))
    al["_in_structural"] = al["variant_id"].astype(str).isin(struct_keys)
    n_struct = int(al["_in_structural"].sum())
    print(f"[1c] allele-less rows also in structural.parquet: {n_struct:,} of {len(al):,}")

    # ---- assign final buckets ----
    rows = []
    for _, r in al.iterrows():
        t = str(r["_type"]).strip().lower() if pd.notna(r["_type"]) else ""
        hit = probe(r["chrom"], int(r["pos"]))
        if hit:
            vref, valt, vid, vpos, src = hit
            rows.append({"variant_id": r["variant_id"], "chrom": r["chrom"], "pos": int(r["pos"]),
                         "type": t, "bucket": "RECOVER", "ref": vref, "alt": valt,
                         "variation_id": vid, "vcf_pos": vpos, "source": src, "genome_verified": True})
        elif t in OUT_OF_SCOPE_TYPES:
            rows.append({"variant_id": r["variant_id"], "chrom": r["chrom"], "pos": int(r["pos"]),
                         "type": t, "bucket": "CONFIRMED_ALLELELESS", "ref": None, "alt": None,
                         "variation_id": None, "vcf_pos": None, "source": None, "genome_verified": False})
        else:
            rows.append({"variant_id": r["variant_id"], "chrom": r["chrom"], "pos": int(r["pos"]),
                         "type": t, "bucket": "STILL_UNRESOLVED", "ref": None, "alt": None,
                         "variation_id": None, "vcf_pos": None, "source": None, "genome_verified": False})

    disp = pd.DataFrame(rows, columns=_FINAL_COLS)

    # ---- 2. gated NCBI resolution for STILL_UNRESOLVED ----
    if a.use_ncbi and (disp["bucket"] == "STILL_UNRESOLVED").any():
        try:
            import urllib.request
            unresolved = disp[disp["bucket"] == "STILL_UNRESOLVED"]
            print(f"[2] NCBI-resolving {len(unresolved):,} unresolved rows (rate-limited) ...")
            # NOTE: requires VariationID; without it in-cohort we can only search by position.
            # Left as a guarded stub that reports intent; real NCBI calls need VariationID recovery.
            print("[2] (NCBI resolution requires VariationID recovery; see docstring. "
                  "No calls made in this run.)")
        except Exception as e:
            print(f"[2] NCBI step skipped: {e}", file=sys.stderr)

    disp.to_csv(out / "alleleless_final_disposition.tsv", sep="\t", index=False)
    disp[disp["bucket"] == "RECOVER"].to_csv(out / "alleleless_recovered_all.tsv",
                                             sep="\t", index=False)
    summary = {
        "date": "2026-07-09",
        "alleleless_total": int(len(disp)),
        "raw_vcf_filedate": raw_date,
        "variant_summary_mtime": vs_mtime,
        "raw_vcf_stale_vs_summary": stale,
        "fresh_vcf_used": bool(a.fresh_vcf and Path(a.fresh_vcf).exists()),
        "also_in_structural_parquet": n_struct,
        "by_bucket": disp["bucket"].value_counts().to_dict(),
        "recover_by_source": disp[disp["bucket"] == "RECOVER"]["source"]
                                .value_counts().to_dict(),
    }
    if "pathogenicity" in coh.columns:
        pj = coh.loc[al.index, "pathogenicity"].astype("string").str.lower()
        disp2 = disp.copy(); disp2["_path"] = (pj == "pathogenic").values
        summary["pathogenic_by_bucket"] = disp2.groupby("bucket")["_path"].sum().to_dict()
    (out / "alleleless_final_disposition_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n--- FINAL DISPOSITION ---")
    print("by bucket           :", summary["by_bucket"])
    print("recover by source   :", summary["recover_by_source"])
    if "pathogenic_by_bucket" in summary:
        print("pathogenic by bucket:", summary["pathogenic_by_bucket"])
    print(f"\nwrote {out}/alleleless_final_disposition.tsv")
    print(f"wrote {out}/alleleless_recovered_all.tsv")
    print(f"wrote {out}/alleleless_final_disposition_summary.json")

    n_unres = int((disp["bucket"] == "STILL_UNRESOLVED").sum())
    if n_unres:
        print(f"\n*** {n_unres:,} rows STILL_UNRESOLVED. Do NOT build v3 until these are "
              f"resolved (rerun with --fresh-vcf and/or --use-ncbi, or accept explicitly). ***",
              file=sys.stderr)
    else:
        print("\nAll allele-less rows bucketed. Ready to build v3 (recover + exclude).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
