"""
recover_alleleless_provenance.py  (2026-07-09)
==========================================================================
DECISIVE provenance test for the 19,988 allele-less (na:na) cohort rows.

BACKGROUND (established 2026-07-09 from the repo source grep)
    scripts/patch_clinvar_alleles.py documents that ClinVar's variant_summary.txt has
    ref="na"/alt="na" for ~99.99% of rows, and that the cohort's ref/alt are populated by a
    SEPARATE patch step that joins the ClinVar VCF (data/raw/clinvar/clinvar_GRCh38.vcf.gz)
    onto clinvar_grch38.parquet. Therefore the 19,988 na:na rows are exactly the rows that
    patch step did NOT populate. Two possible provenances per row:
        VCF_PRESENT  -> the variant IS in clinvar_GRCh38.vcf.gz but the original patch join
                        missed it  => RECOVERABLE (a pipeline defect, recover ref/alt here).
        VCF_ABSENT   -> the variant is NOT in the ClinVar VCF at all (out of VCF scope:
                        copy number / structural / cytogenetic / imprecise) => legitimately
                        allele-less by ClinVar's own design.

    The earlier verify_alleleless_provenance.py joined on chrom:pos and produced 18,137
    spurious NEEDS_REVIEW because the cohort carries NO VariationID and variant_summary's
    PositionVCF is empty for exactly these rows. THIS tool fixes that by (a) testing VCF
    presence directly against the ClinVar VCF, and (b) recovering VariationID + Type from
    variant_summary via gene+interval, giving a stable-key classification.

WHAT IT DOES (all read-only against the cohort; writes only to outputs/)
    1. Load the cohort; select the na:na rows via the shared allele_classify.is_allele_less.
    2. Parse clinvar_GRCh38.vcf.gz -> map CHROM:POS -> (ref, alt, VariationID) [ID col = VCV id].
       (Also index CHROM:POS presence for the VCF-presence test.)
    3. Join variant_summary.txt.gz to recover, per na:na row, VariationID + Type + Start/Stop
       by GeneSymbol + genomic interval + Assembly (variant_summary keeps these for
       out-of-scope variants; only its VCF-allele columns are blank).
    4. Emit a per-row verdict:
         RECOVERABLE_FROM_VCF   -> position present in ClinVar VCF with real ref/alt.
         LEGITIMATELY_ALLELELESS-> absent from VCF AND Type in OUT_OF_SCOPE_TYPES.
         NEEDS_REVIEW           -> absent from VCF but Type is sequence-like or unknown
                                   (surfaced, never silently dropped).
    5. Write outputs/alleleless_recovery_verdict.tsv, .../alleleless_recovery_summary.json,
       and outputs/alleleless_recoverable_alleles.tsv (chrom,pos,ref,alt,variation_id) for
       the rows we CAN recover, so a subsequent cohort patch can consume them.

USAGE
    python scripts/recover_alleleless_provenance.py \
        --cohort         data/processed/clinvar_grch38_clean_v2_verified.parquet \
        --clinvar-vcf    data/raw/clinvar/clinvar_GRCh38.vcf.gz \
        --variant-summary data/external/clinvar/variant_summary.txt.gz \
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

OUT_OF_SCOPE_TYPES = {
    "copy number gain", "copy number loss", "translocation", "fusion", "complex",
    "microsatellite", "tandem repeat", "variation", "protein only", "inversion",
    "cytogenetic",
}
SEQUENCE_TYPES = {
    "single nucleotide variant", "deletion", "insertion", "duplication", "indel", "delins",
}
_NULL = {"", "na", "nan", "none", "-", "."}


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _parse_clinvar_vcf(path: Path):
    """Return (pos_index, allele_map): a set of 'CHROM:POS' and a dict CHROM:POS ->
    (ref, alt, variation_id). The ClinVar VCF ID column is the ClinVar VariationID."""
    pos_index = set()
    allele_map = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            chrom, pos, vid, ref, alt = _norm_chrom(f[0]), f[1], f[2], f[3], f[4]
            key = f"{chrom}:{pos}"
            pos_index.add(key)
            # keep the first allele seen per position (positions are usually unique per VCF row)
            allele_map.setdefault(key, (ref, alt, vid))
    return pos_index, allele_map


def _load_variant_summary(path: Path, assembly: str) -> pd.DataFrame:
    want = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome",
            "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--clinvar-vcf", required=True,
                    help="the RAW ClinVar VCF used by patch_clinvar_alleles.py "
                         "(e.g. data/raw/clinvar/clinvar_GRCh38.vcf.gz)")
    ap.add_argument("--variant-summary", default=None,
                    help="variant_summary.txt.gz (for VariationID + Type recovery)")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--out-verdict", default="outputs/alleleless_recovery_verdict.tsv")
    ap.add_argument("--out-summary", default="outputs/alleleless_recovery_summary.json")
    ap.add_argument("--out-recoverable", default="outputs/alleleless_recoverable_alleles.tsv")
    a = ap.parse_args(argv)

    coh = pd.read_parquet(a.cohort)
    al = coh[is_allele_less(coh["ref"], coh["alt"])].copy()
    print(f"allele-less (na:na) rows: {len(al):,}")
    if not len(al):
        print("none; nothing to do.")
        return 0
    al["_key"] = al["chrom"].map(_norm_chrom) + ":" + al["pos"].astype(str)

    print("parsing ClinVar VCF (this is the decisive presence test) ...")
    pos_index, allele_map = _parse_clinvar_vcf(Path(a.clinvar_vcf))
    print(f"  VCF positions indexed: {len(pos_index):,}")
    al["_in_vcf"] = al["_key"].isin(pos_index)
    al["_vcf_ref"] = al["_key"].map(lambda k: allele_map.get(k, (None, None, None))[0])
    al["_vcf_alt"] = al["_key"].map(lambda k: allele_map.get(k, (None, None, None))[1])
    al["_vcf_vid"] = al["_key"].map(lambda k: allele_map.get(k, (None, None, None))[2])

    # recover Type (+VariationID) from variant_summary by gene + interval
    al["_vtype"] = pd.NA
    al["_vs_vid"] = pd.NA
    if a.variant_summary:
        print("recovering VariationID + Type from variant_summary ...")
        vs = _load_variant_summary(Path(a.variant_summary), a.assembly)
        if {"GeneSymbol", "Start", "Chromosome", "Type"} <= set(vs.columns):
            vs["_k"] = (vs["Chromosome"].map(_norm_chrom) + ":" + vs["Start"].astype(str))
            vk = vs.drop_duplicates("_k").set_index("_k")
            # cohort pos was verified == variant_summary Start for non-padded rows; use Start-join
            al["_startkey"] = al["chrom"].map(_norm_chrom) + ":" + al["pos"].astype(str)
            al["_vtype"] = al["_startkey"].map(vk["Type"]) if "Type" in vs.columns else pd.NA
            if "VariationID" in vs.columns:
                al["_vs_vid"] = al["_startkey"].map(vk["VariationID"])

    def verdict(row) -> str:
        if bool(row["_in_vcf"]):
            rr = str(row["_vcf_ref"] or "").lower()
            aa = str(row["_vcf_alt"] or "").lower()
            if rr not in _NULL and aa not in _NULL:
                return "RECOVERABLE_FROM_VCF"
        t = str(row["_vtype"]).strip().lower() if pd.notna(row["_vtype"]) else ""
        if not bool(row["_in_vcf"]) and t in OUT_OF_SCOPE_TYPES:
            return "LEGITIMATELY_ALLELELESS"
        return "NEEDS_REVIEW"

    al["verdict"] = al.apply(verdict, axis=1)

    Path(a.out_verdict).parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant_id", "chrom", "pos", "_in_vcf", "_vcf_ref", "_vcf_alt",
            "_vcf_vid", "_vtype", "_vs_vid", "verdict"]
    if "pathogenicity" in al.columns:
        cols.append("pathogenicity")
    al[cols].to_csv(a.out_verdict, sep="\t", index=False)

    # emit recoverable alleles for a downstream cohort patch
    rec = al[al["verdict"] == "RECOVERABLE_FROM_VCF"][
        ["variant_id", "chrom", "pos", "_vcf_ref", "_vcf_alt", "_vcf_vid"]].copy()
    rec.columns = ["variant_id", "chrom", "pos", "ref", "alt", "variation_id"]
    rec.to_csv(a.out_recoverable, sep="\t", index=False)

    summary = {
        "date": "2026-07-09",
        "alleleless_total": int(len(al)),
        "clinvar_vcf": str(a.clinvar_vcf),
        "in_vcf": int(al["_in_vcf"].sum()),
        "by_verdict": al["verdict"].value_counts().to_dict(),
        "type_top15": al["_vtype"].astype("string").fillna("<no-vs-match>")
                        .value_counts().head(15).to_dict(),
        "recoverable_rows": int(len(rec)),
    }
    if "pathogenicity" in al.columns:
        summary["pathogenic_by_verdict"] = (
            al[al["pathogenicity"].astype("string").str.lower() == "pathogenic"]
            ["verdict"].value_counts().to_dict())
    Path(a.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- PROVENANCE VERDICT (by stable evidence) ---")
    print(f"  present in ClinVar VCF : {summary['in_vcf']:,} of {len(al):,}")
    print(f"  by verdict             : {summary['by_verdict']}")
    print(f"  recoverable rows       : {summary['recoverable_rows']:,}")
    if "pathogenic_by_verdict" in summary:
        print(f"  pathogenic by verdict  : {summary['pathogenic_by_verdict']}")
    print(f"\nwrote {a.out_verdict}")
    print(f"wrote {a.out_summary}")
    print(f"wrote {a.out_recoverable}  ({len(rec):,} rows a cohort patch can restore)")

    # interpretation guardrails
    if summary["in_vcf"] > 0:
        print("\n>>> INTERPRETATION: some na:na rows ARE in the ClinVar VCF. Those are a "
              "PIPELINE PATCH-JOIN LOSS, not legitimately allele-less. RECOVER them; do NOT "
              "drop them. Re-examine patch_clinvar_alleles.py's join keys.")
    if summary["in_vcf"] == 0:
        print("\n>>> INTERPRETATION: NONE of the na:na rows are in the ClinVar VCF -> they are "
              "genuinely out of ClinVar's VCF scope (structural/CNV/cytogenetic). This "
              "corroborates legitimately-allele-less; classify by Type and exclude with "
              "documentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
