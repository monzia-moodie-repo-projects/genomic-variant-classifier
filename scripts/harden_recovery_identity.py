"""
harden_recovery_identity.py  (2026-07-09)
==========================================================================
Re-verify the WIDE-WINDOW allele recoveries by VariationID identity before they are
allowed into the cohort.

WHY: resolve_seqtype_absent.py widened the VCF probe to +/-25bp and recovered 4,416 rows.
But a genome-verified ref match within 25bp is NOT proof that the matched VCF row is the
SAME ClinVar variant as the cohort row -- it could be a nearby DIFFERENT variant. Attaching
a neighbor's ref/alt would mislabel real training data. The pos/pos+1 recoveries are safe
(that offset IS the documented padded-deletion anchor convention). Only recoveries with
|vcf_pos - cohort_pos| >= 2 need a stronger identity gate.

IDENTITY GATE for each wide-window (|offset|>=2) recovery:
  IDENTITY-1 (strong): look up the cohort row's ClinVar VariationID from variant_summary by
     (Chromosome, Start==pos OR Start==pos+1, GeneSymbol); require the recovered VCF row's
     ID (== VariationID) to EQUAL it.
  IDENTITY-2 (fallback): require the recovered VCF row's INFO GENEINFO gene to equal the
     cohort gene_symbol AND its INFO CLNSIG to be consistent with the cohort pathogenicity
     label (both pathogenic-ish, or both benign-ish).
  ACCEPT iff IDENTITY-1 or IDENTITY-2; else RE-BUCKET the row to STILL_UNRESOLVED.

pos/pos+1 recoveries (|offset|<=1) are ACCEPTED as-is (already genome-verified + convention).

INPUTS
  --disposition   outputs/alleleless_final_disposition.tsv  (from resolve_seqtype_absent.py)
  --cohort        the cohort (for gene_symbol + pathogenicity per variant_id)
  --raw-vcf/--fresh-vcf  the VCF(s) used for recovery (to read ID + INFO of the matched row)
  --variant-summary      for VariationID lookup

OUTPUTS (outputs/)
  alleleless_disposition_hardened.tsv          (RECOVER now only identity-verified)
  alleleless_recovery_identity_audit.tsv       (per wide row: id1/id2 pass/fail + reason)
  alleleless_disposition_hardened_summary.json

USAGE
  python scripts/harden_recovery_identity.py \
      --disposition     outputs/alleleless_final_disposition.tsv \
      --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --raw-vcf         data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --fresh-vcf       data/external/clinvar/clinvar.vcf.gz \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --assembly GRCh38
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

_NULL = {"", "na", "nan", "none", "-", "."}
_PATHOGENIC_TOKENS = {"pathogenic", "likely_pathogenic", "likely pathogenic"}
_BENIGN_TOKENS = {"benign", "likely_benign", "likely benign"}


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _label_class(s: str) -> str:
    s = str(s).strip().lower().replace("/", " ")
    if any(t in s for t in _PATHOGENIC_TOKENS):
        return "pathogenic"
    if any(t in s for t in _BENIGN_TOKENS):
        return "benign"
    return "other"


def _index_vcf_rows(path: Path):
    """chrom -> {pos -> list of (vid, ref, alt, geneinfo, clnsig)}. Parses INFO for GENEINFO/CLNSIG."""
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
            info = f[7]
            gene = None
            m = re.search(r"GENEINFO=([^;]+)", info)
            if m:
                gene = m.group(1).split(":")[0]
            clnsig = None
            m = re.search(r"CLNSIG=([^;]+)", info)
            if m:
                clnsig = m.group(1)
            by.setdefault(c, {}).setdefault(p, []).append((f[2], f[3], f[4], gene, clnsig))
    return by


def _load_vs_varid(path: Path, assembly: str):
    """(chrom,start,gene) -> VariationID, from variant_summary."""
    want = ["VariationID", "GeneSymbol", "Assembly", "Chromosome", "Start"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    idx = {}
    if {"Chromosome", "Start", "VariationID"} <= set(vs.columns):
        gene_col = vs["GeneSymbol"] if "GeneSymbol" in vs.columns else pd.Series([""] * len(vs))
        for c, s, g, vid in zip(vs["Chromosome"].map(_norm_chrom), vs["Start"],
                                gene_col, vs["VariationID"]):
            idx.setdefault((c, str(s), str(g)), vid)
            idx.setdefault((c, str(s), None), vid)   # gene-agnostic fallback
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disposition", default="outputs/alleleless_final_disposition.tsv")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    disp = pd.read_csv(a.disposition, sep="\t")
    coh = pd.read_parquet(a.cohort, columns=["variant_id", "gene_symbol", "pathogenicity"]) \
        if set(["gene_symbol", "pathogenicity"]).issubset(
            pd.read_parquet(a.cohort, columns=None).columns) \
        else pd.read_parquet(a.cohort)
    gene_by_id = dict(zip(coh["variant_id"], coh.get("gene_symbol", pd.Series(dtype=str))))
    path_by_id = dict(zip(coh["variant_id"], coh.get("pathogenicity", pd.Series(dtype=str))))

    vcf_idx = _index_vcf_rows(Path(a.raw_vcf))
    if a.fresh_vcf and Path(a.fresh_vcf).exists():
        fresh = _index_vcf_rows(Path(a.fresh_vcf))
        for c, d in fresh.items():
            for p, rows in d.items():
                vcf_idx.setdefault(c, {}).setdefault(p, []).extend(rows)

    vs_varid = _load_vs_varid(Path(a.variant_summary), a.assembly)

    recover = disp[disp["bucket"] == "RECOVER"].copy()
    recover["_offset"] = (recover["vcf_pos"].fillna(recover["pos"]).astype(float)
                          - recover["pos"].astype(float)).abs()

    audit = []
    hardened_bucket = {}
    for _, r in recover.iterrows():
        vid = r["variant_id"]
        offset = r["_offset"]
        if offset <= 1:
            hardened_bucket[vid] = "RECOVER"           # pos/pos+1 exempt
            continue
        # wide-window: require identity
        chrom = _norm_chrom(r["chrom"]); pos = int(r["pos"])
        rec_vcf_id = str(r["variation_id"]) if pd.notna(r["variation_id"]) else None

        # IDENTITY-1: cohort VariationID via variant_summary vs recovered VCF ID
        gene = str(gene_by_id.get(vid, "")) or None
        coh_varid = (vs_varid.get((chrom, str(pos), gene))
                     or vs_varid.get((chrom, str(pos + 1), gene))
                     or vs_varid.get((chrom, str(pos), None))
                     or vs_varid.get((chrom, str(pos + 1), None)))
        id1 = (rec_vcf_id is not None and coh_varid is not None
               and str(rec_vcf_id) == str(coh_varid))

        # IDENTITY-2: gene + clinsig agreement from the matched VCF row
        id2 = False
        vpos = int(r["vcf_pos"]) if pd.notna(r["vcf_pos"]) else pos
        cand = vcf_idx.get(chrom, {}).get(vpos, [])
        vrow = next((x for x in cand if str(x[0]) == str(rec_vcf_id)), None)
        if vrow is not None:
            _, _, _, vgene, vclnsig = vrow
            gene_ok = (vgene is not None and gene is not None
                       and str(vgene).upper() == str(gene).upper())
            clnsig_ok = (_label_class(vclnsig) != "other"
                         and _label_class(vclnsig) == _label_class(path_by_id.get(vid, "")))
            id2 = bool(gene_ok and clnsig_ok)

        accept = bool(id1 or id2)
        hardened_bucket[vid] = "RECOVER" if accept else "STILL_UNRESOLVED"
        audit.append({"variant_id": vid, "chrom": r["chrom"], "pos": pos,
                      "vcf_pos": r["vcf_pos"], "offset": offset,
                      "recovered_vcf_id": rec_vcf_id, "cohort_variation_id": coh_varid,
                      "identity1_varid": id1, "identity2_gene_clnsig": id2,
                      "decision": "ACCEPT" if accept else "REBUCKET_UNRESOLVED"})

    # apply hardened decisions
    disp["bucket_hardened"] = disp.apply(
        lambda r: hardened_bucket.get(r["variant_id"], r["bucket"]), axis=1)
    # wide rows rebucketed lose their recovered alleles
    for col in ["ref", "alt", "variation_id", "vcf_pos", "source"]:
        disp.loc[disp["bucket_hardened"] != "RECOVER", col] = None

    disp.to_csv(out / "alleleless_disposition_hardened.tsv", sep="\t", index=False)
    pd.DataFrame(audit).to_csv(out / "alleleless_recovery_identity_audit.tsv",
                               sep="\t", index=False)

    n_wide = len(audit)
    n_accept = sum(1 for x in audit if x["decision"] == "ACCEPT")
    summary = {
        "date": "2026-07-09",
        "recover_before": int((disp["bucket"] == "RECOVER").sum()),
        "recover_after_hardening": int((disp["bucket_hardened"] == "RECOVER").sum()),
        "pos_or_pos1_exempt": int((disp["bucket"] == "RECOVER").sum() - n_wide),
        "wide_window_checked": n_wide,
        "wide_window_accepted": n_accept,
        "wide_window_rebucketed": n_wide - n_accept,
        "by_bucket_hardened": disp["bucket_hardened"].value_counts().to_dict(),
    }
    (out / "alleleless_disposition_hardened_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("--- IDENTITY HARDENING ---")
    print(f"RECOVER before                 : {summary['recover_before']:,}")
    print(f"  pos/pos+1 exempt (safe)      : {summary['pos_or_pos1_exempt']:,}")
    print(f"  wide-window checked          : {summary['wide_window_checked']:,}")
    print(f"    accepted (identity proven) : {summary['wide_window_accepted']:,}")
    print(f"    rebucketed -> UNRESOLVED   : {summary['wide_window_rebucketed']:,}")
    print(f"RECOVER after hardening        : {summary['recover_after_hardening']:,}")
    print(f"by bucket (hardened)           : {summary['by_bucket_hardened']}")
    print(f"\nwrote {out}/alleleless_disposition_hardened.tsv")
    print(f"wrote {out}/alleleless_recovery_identity_audit.tsv")
    print(f"wrote {out}/alleleless_disposition_hardened_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
