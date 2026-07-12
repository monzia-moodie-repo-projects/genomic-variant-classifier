"""
verify_alleleless_provenance.py  (2026-07-09)
==========================================================================
Close the provenance question for the 19,988 allele-less (na:na) cohort rows by
cross-checking them against LIVE NCBI ClinVar source files, per the research
memo ALLELELESS_PROVENANCE_2026-07-09.md.

It does NOT modify the cohort. It produces:
  * outputs/alleleless_verdict.tsv   -- one row per allele-less record with its
        ClinVar Type, live-allele status, VCF presence, pattern (na:na vs partial),
        and a verdict: LEGITIMATELY_ALLELELESS | RECOVERABLE | NEEDS_REVIEW
  * outputs/alleleless_provenance_summary.json -- counts by verdict/Type/pathogenicity
  * outputs/alleleless_exclude_ids.txt  -- variant_ids confirmed out-of-scope,
        the ONLY rows rebuild_cohort_v3.py will exclude.

DECISION RULE (per research memo Section 5):
  LEGITIMATELY_ALLELELESS <=> Type in OUT_OF_SCOPE_TYPES AND (absent from
      clinvar.vcf OR no live VCF alleles).
  RECOVERABLE             <=> pattern is partial (one real allele) OR Type in
      SEQUENCE_TYPES with live non-na VCF alleles present.
  NEEDS_REVIEW            <=> anything the rule cannot decide (e.g., no ClinVar
      match at all) -- surfaced explicitly, never silently dropped.

INPUTS (download live, matching the build assembly):
  --variant-summary  variant_summary.txt.gz
  --clinvar-vcf      clinvar.vcf.gz  (optional but strengthens the verdict)

USAGE
  python scripts/verify_alleleless_provenance.py \
      --cohort data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --clinvar-vcf     data/external/clinvar/clinvar.vcf.gz \
      --assembly GRCh38
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from genomic_variant_classifier.data.allele_classify import is_allele_less
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from allele_classify import is_allele_less  # type: ignore

# ClinVar `Type` vocabulary buckets (lowercased for matching)
OUT_OF_SCOPE_TYPES = {
    "copy number gain", "copy number loss", "translocation", "fusion", "complex",
    "microsatellite", "tandem repeat", "variation", "protein only", "inversion",
    "cytogenetic",
}
SEQUENCE_TYPES = {
    "single nucleotide variant", "deletion", "insertion", "duplication",
    "indel", "delins",
}
_NULL = {"", "na", "nan", "none", "-", "."}


def _real_allele(x) -> bool:
    s = str(x).strip().lower() if x is not None else ""
    return s not in _NULL and all(ch in "acgtn" for ch in s) and len(s) >= 1


def _source_alleles_from_variant_id(vid: str):
    """Return (ref_token, alt_token) from clinvar:CHROM:POS:REF:ALT (n=4 split)."""
    parts = str(vid).split(":", 4)
    if len(parts) >= 5:
        return parts[3], parts[4]
    return None, None


def _pattern(ref_tok, alt_tok) -> str:
    r_ok = _real_allele(ref_tok)
    a_ok = _real_allele(alt_tok)
    if not r_ok and not a_ok:
        return "na:na"
    if r_ok ^ a_ok:
        return "partial"          # CTG:na or na:GG -- recoverable indel fingerprint
    return "both"                 # should not occur among allele-less rows


def _load_variant_summary(path: Path, assembly: str) -> pd.DataFrame:
    cols = None
    # read only the columns we need to keep memory bounded
    want = ["#AlleleID", "AlleleID", "Type", "Assembly", "Chromosome",
            "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF",
            "ReferenceAllele", "AlternateAllele", "VariationID", "Start", "Stop"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    present = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=present)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


def _vcf_positions(path: Path) -> set:
    """Set of 'CHROM:POS' present in clinvar.vcf.gz (fast line scan)."""
    positions = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 2)
            if len(f) >= 2:
                positions.add(f[0].lstrip("chr") + ":" + f[1])
    return positions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--clinvar-vcf", default=None)
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--out-verdict", default="outputs/alleleless_verdict.tsv")
    ap.add_argument("--out-summary", default="outputs/alleleless_provenance_summary.json")
    ap.add_argument("--out-exclude", default="outputs/alleleless_exclude_ids.txt")
    a = ap.parse_args(argv)

    cohort = pd.read_parquet(a.cohort)
    al = cohort[is_allele_less(cohort["ref"], cohort["alt"])].copy()
    print(f"allele-less rows in cohort: {len(al):,}")
    if not len(al):
        print("no allele-less rows; nothing to verify.")
        return 0

    # pattern from the ORIGINAL variant_id
    toks = al["variant_id"].map(_source_alleles_from_variant_id)
    al["_ref_tok"] = [t[0] for t in toks]
    al["_alt_tok"] = [t[1] for t in toks]
    al["pattern"] = [_pattern(r, a_) for r, a_ in zip(al["_ref_tok"], al["_alt_tok"])]

    # join to live variant_summary
    vs = _load_variant_summary(Path(a.variant_summary), a.assembly)
    vs_type = None
    if "VariationID" in vs.columns and "variation_id" in al.columns:
        vs_small = vs.drop_duplicates("VariationID").set_index("VariationID")
        al["_vtype"] = al["variation_id"].astype(str).map(vs_small["Type"]) if "Type" in vs.columns else None
        al["_live_ref"] = al["variation_id"].astype(str).map(
            vs_small["ReferenceAlleleVCF"]) if "ReferenceAlleleVCF" in vs.columns else None
        al["_live_alt"] = al["variation_id"].astype(str).map(
            vs_small["AlternateAlleleVCF"]) if "AlternateAlleleVCF" in vs.columns else None
    else:
        # fall back to chrom:pos join
        if {"Chromosome", "PositionVCF", "Type"} <= set(vs.columns):
            vs["_k"] = vs["Chromosome"].astype(str) + ":" + vs["PositionVCF"].astype(str)
            vk = vs.drop_duplicates("_k").set_index("_k")
            al["_k"] = al["chrom"].astype(str) + ":" + al["pos"].astype(str)
            al["_vtype"] = al["_k"].map(vk["Type"])
            al["_live_ref"] = al["_k"].map(vk.get("ReferenceAlleleVCF"))
            al["_live_alt"] = al["_k"].map(vk.get("AlternateAlleleVCF"))
        else:
            al["_vtype"] = None; al["_live_ref"] = None; al["_live_alt"] = None

    # VCF presence
    vcf_pos = _vcf_positions(Path(a.clinvar_vcf)) if a.clinvar_vcf else None
    if vcf_pos is not None:
        al["_in_vcf"] = (al["chrom"].astype(str) + ":" + al["pos"].astype(str)).isin(vcf_pos)
    else:
        al["_in_vcf"] = pd.NA

    def verdict(row) -> str:
        t = str(row["_vtype"]).strip().lower() if pd.notna(row["_vtype"]) else ""
        live_ref_ok = _real_allele(row["_live_ref"])
        live_alt_ok = _real_allele(row["_live_alt"])
        in_vcf = row["_in_vcf"]
        # recoverable: partial pattern, or sequence-type with live alleles present
        if row["pattern"] == "partial":
            return "RECOVERABLE"
        if (live_ref_ok and live_alt_ok) or (t in SEQUENCE_TYPES and in_vcf is True):
            return "RECOVERABLE"
        # legitimately allele-less: out-of-scope type AND (absent from vcf OR no live alleles)
        if t in OUT_OF_SCOPE_TYPES and (in_vcf is False or in_vcf is pd.NA
                                        or not (live_ref_ok and live_alt_ok)):
            return "LEGITIMATELY_ALLELELESS"
        return "NEEDS_REVIEW"

    al["verdict"] = al.apply(verdict, axis=1)

    # write per-row verdict
    keep = ["variant_id", "chrom", "pos", "pattern", "_vtype", "_live_ref",
            "_live_alt", "_in_vcf", "verdict"]
    if "pathogenicity" in al.columns:
        keep.append("pathogenicity")
    Path(a.out_verdict).parent.mkdir(parents=True, exist_ok=True)
    al[keep].to_csv(a.out_verdict, sep="\t", index=False)

    # exclusion list = ONLY confirmed legitimately allele-less
    excl = al.loc[al["verdict"] == "LEGITIMATELY_ALLELELESS", "variant_id"]
    Path(a.out_exclude).write_text("\n".join(excl.astype(str)), encoding="utf-8")

    # summary
    summary = {
        "date": "2026-07-09",
        "allele_less_total": int(len(al)),
        "by_verdict": al["verdict"].value_counts().to_dict(),
        "by_pattern": al["pattern"].value_counts().to_dict(),
        "type_top15": al["_vtype"].astype("string").fillna("<no-clinvar-match>")
                        .value_counts().head(15).to_dict(),
        "exclude_count": int(len(excl)),
        "recoverable_count": int((al["verdict"] == "RECOVERABLE").sum()),
        "needs_review_count": int((al["verdict"] == "NEEDS_REVIEW").sum()),
    }
    if "pathogenicity" in al.columns:
        summary["pathogenic_by_verdict"] = (
            al[al["pathogenicity"].astype("string").str.lower() == "pathogenic"]
            ["verdict"].value_counts().to_dict())
    Path(a.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("verdict counts    :", summary["by_verdict"])
    print("pattern counts    :", summary["by_pattern"])
    print("exclude (confirmed):", summary["exclude_count"])
    print("recoverable       :", summary["recoverable_count"])
    print("needs review      :", summary["needs_review_count"])
    if "pathogenic_by_verdict" in summary:
        print("pathogenic split  :", summary["pathogenic_by_verdict"])
    print(f"\nwrote {a.out_verdict}, {a.out_summary}, {a.out_exclude}")
    # guardrail: escalate if many sequence-type rows have live alleles
    rec = summary["recoverable_count"]
    if rec > 0.5 * len(al):
        print("\n*** ESCALATION: >50% appear RECOVERABLE -- likely a pipeline parsing "
              "loss, not legitimately allele-less. Re-examine the cohort build before "
              "excluding anything. ***", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
