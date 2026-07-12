#!/usr/bin/env python
"""
classify_alleleless_by_type.py  (2026-07-09)
==========================================================================
FINAL, type-aware disposition of the allele-less rows, keyed on each row's OWN source_id.

BACKGROUND (established 2026-07-09 across the allele-less arc)
The 19,988 allele-less (na:na) cohort rows are, per ClinVar variant_summary Type,
overwhelmingly STRUCTURAL variants -- copy number gain/loss, large Deletions/Duplications,
Microsatellite repeat expansions, Translocations, Inversions -- which ClinVar represents in
variant_summary but NOT in the allele-level VCF, because they have no simple ref/alt. A
direct probe (2026-07-09) confirmed the sampled source_ids are absent from both the raw and
fresh ClinVar VCF, and their Names are cytogenetic/HGVS structural descriptions. Only 12
rows are Type 'single nucleotide variant', and all 12 are likewise absent from the VCF
(withdrawn/re-versioned/transcript-only ids); their alleles exist only as coding HGVS in the
Name, which is strand-relative and must not be hand-parsed into genomic alleles.

The prior pipeline's "544 recoveries" were spurious: recover_identity_first resolved each
structural variant to a CO-LOCATED SNV's VariationID (which IS in the VCF) via a locus
lookup, and borrowed that SNV's point allele -- attaching a wrong single-base allele to a
megabase structural variant. Recovering strictly by each row's OWN source_id yields 0,
because the true variants have no allele in the VCF.

THIS TOOL therefore CLASSIFIES every allele-less row by its own source_id + ClinVar Type:
  CONFIRMED_ALLELELESS_SV             structural type (see STRUCTURAL_TYPES); excluded.
  CONFIRMED_ALLELELESS_SNV_NOT_IN_VCF Type is SNV but source_id absent from both VCFs.
  RECOVER_BY_SID_{RAW,FRESH}          Type is SNV/small and source_id IS in a VCF with a
                                      genome-verified allele  (0 for current data;
                                      future-proof if ClinVar later adds the id).
  CONFIRMED_ALLELELESS_NO_VS          source_id not found in variant_summary at all.
Every row records its ClinVar Type and a reason. The unique per-row key is
(source_id, chrom, pos).

USAGE
  python scripts/classify_alleleless_by_type.py \
      --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --raw-vcf         data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --fresh-vcf       data/external/clinvar/clinvar.vcf.gz \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --fasta           data/external/grch38/GRCh38.fa \
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

# ClinVar Types that are structural / have no simple allele in the VCF
STRUCTURAL_TYPES = {
    "copy number gain", "copy number loss", "Deletion", "Duplication",
    "Insertion", "Indel", "Microsatellite", "Translocation", "Inversion",
    "Complex", "Variation", "Tandem duplication", "fusion",
}
# Types we WILL attempt a per-id VCF allele recovery for
RECOVERABLE_TYPES = {"single nucleotide variant"}


def _norm_chrom(c) -> str:
    return str(c).strip().lstrip("chr")


def _clean_id(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def _real(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    return str(x).strip().lower() not in {"", "na", "nan", "none", "-", ".", "<na>"}


def _find_in_vcf(path, want):
    found = {}
    if not path or not Path(path).exists():
        return found
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 5)
            if len(p) < 5:
                continue
            vid = _clean_id(p[2])
            if vid in want:
                pos = int(p[1]) if p[1].isdigit() else None
                found[vid] = (_norm_chrom(p[0]), pos, p[3], p[4])
                if len(found) == len(want):
                    break
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)

    print("=== classify_alleleless_by_type START ===", flush=True)
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    coh = pd.read_parquet(a.cohort)
    al = coh[is_allele_less(coh["ref"], coh["alt"])].copy()
    al["source_id"] = al["source_id"].map(_clean_id)
    al["chrom"] = al["chrom"].astype(str)
    al["pos"] = al["pos"].astype(int)
    n_al = len(al)
    print(f"allele-less rows: {n_al:,}", flush=True)

    n_no_sid = int((~al["source_id"].map(_real)).sum())
    print(f"rows lacking a usable source_id: {n_no_sid}", flush=True)

    # ClinVar Type per source_id (types agree across assemblies; take first seen)
    vs = pd.read_csv(a.variant_summary, sep="\t", dtype=str, compression="gzip",
                     usecols=lambda c: c in {"VariationID", "Type", "Assembly"})
    vs["VariationID"] = vs["VariationID"].map(_clean_id)
    type_by_id = {}
    for vid, t in zip(vs["VariationID"], vs["Type"]):
        type_by_id.setdefault(vid, t)
    print(f"variant_summary ids indexed: {len(type_by_id):,}", flush=True)

    al["vs_type"] = al["source_id"].map(type_by_id)

    # For SNV/recoverable-typed ids, attempt a strict VCF-by-own-id lookup
    snv_ids = set(al[al["vs_type"].isin(RECOVERABLE_TYPES)]["source_id"])
    raw_hit = _find_in_vcf(a.raw_vcf, snv_ids)
    fresh_hit = _find_in_vcf(a.fresh_vcf, snv_ids) if a.fresh_vcf else {}
    print(f"SNV-typed allele-less ids: {len(snv_ids):,}  "
          f"VCF hits: raw={len(raw_hit)}, fresh={len(fresh_hit)}", flush=True)

    ref_genome = None
    if Path(a.fasta).exists():
        from pyfaidx import Fasta
        ref_genome = Fasta(str(a.fasta), rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, ref):
        if ref_genome is None or pos is None:
            return None
        c = _norm_chrom(chrom)
        if c not in contigs:
            return None
        try:
            got = str(ref_genome[c][int(pos) - 1:int(pos) - 1 + len(ref)]).upper()
        except Exception:
            return None
        return got == str(ref).upper()

    rows = []
    for _, r in al.iterrows():
        sid = r["source_id"]
        vtype = r["vs_type"]
        rec_pos = rec_ref = rec_alt = rec_source = None
        gver = None

        if not _real(sid):
            verdict, reason = "NO_SOURCE_ID", "row has no source_id"
        elif vtype is None or (isinstance(vtype, float) and pd.isna(vtype)):
            verdict, reason = "CONFIRMED_ALLELELESS_NO_VS", "source_id not in variant_summary"
        elif vtype in RECOVERABLE_TYPES:
            hit = raw_hit.get(sid)
            src = "raw"
            if hit is None:
                hit = fresh_hit.get(sid)
                src = "fresh"
            if hit is None:
                verdict = "CONFIRMED_ALLELELESS_SNV_NOT_IN_VCF"
                reason = "SNV-typed but source_id absent from both VCFs"
            else:
                vchrom, vpos, vref, valt = hit
                if not (_real(vref) and _real(valt)):
                    verdict = "CONFIRMED_ALLELELESS_SNV_NULL_VCF"
                    reason = "VCF record for source_id has null allele"
                else:
                    gver = genome_ref_ok(vchrom, vpos, vref)
                    if gver:
                        rec_pos, rec_ref, rec_alt, rec_source = vpos, vref, valt, src
                        verdict = f"RECOVER_BY_SID_{src.upper()}"
                        reason = "SNV recovered by own source_id, genome-verified"
                    else:
                        verdict = "SID_GENOME_MISMATCH"
                        reason = "VCF allele for source_id does not match genome"
        else:
            # structural type
            verdict = "CONFIRMED_ALLELELESS_SV"
            reason = f"structural ClinVar Type '{vtype}' has no simple allele"

        rows.append({
            "variant_id": r["variant_id"], "chrom": r["chrom"], "pos": r["pos"],
            "source_id": sid, "vs_type": vtype, "verdict": verdict, "reason": reason,
            "rec_pos": rec_pos, "rec_ref": rec_ref, "rec_alt": rec_alt,
            "rec_source": rec_source, "genome_ok": gver,
        })

    res = pd.DataFrame(rows)
    res.to_csv(out / "alleleless_recovery_by_sid_full.tsv", sep="\t", index=False)

    recovered = res[res["verdict"].str.startswith("RECOVER_BY_SID_")].copy()
    recovered[["variant_id", "chrom", "pos", "source_id", "rec_pos", "rec_ref", "rec_alt",
               "rec_source", "verdict"]].to_csv(
        out / "alleleless_recovered_by_sid.tsv", sep="\t", index=False)

    # fail-loud: recovered key uniqueness
    key_dupe = int(recovered.duplicated(subset=["source_id", "chrom", "pos"]).sum())

    counts = res["verdict"].value_counts().to_dict()
    type_counts = res["vs_type"].value_counts(dropna=False).to_dict()
    summary = {
        "date": "2026-07-09",
        "cohort": a.cohort,
        "allele_less_rows": int(n_al),
        "rows_lacking_source_id": int(n_no_sid),
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "clinvar_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "recovered_rows": int(len(recovered)),
        "recovered_key_duplicates": key_dupe,
        "excluded_total": int(n_al - len(recovered)),
        "note": ("All allele-less rows are structural or SNV-not-in-VCF; the prior '544 "
                 "recoveries' were spurious neighbor-SNV borrowing. Recovery by each row's "
                 "own source_id yields the recovered count above."),
    }
    (out / "alleleless_recovery_by_sid_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- TYPE-AWARE DISPOSITION (per row) ---", flush=True)
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:36s}: {counts[k]:,}", flush=True)
    print("\n--- ClinVar Type breakdown ---", flush=True)
    for k in sorted(type_counts, key=lambda x: -type_counts[x]):
        print(f"  {str(k):30s}: {type_counts[k]:,}", flush=True)
    print(f"\nrecovered rows: {len(recovered):,}  excluded: {n_al - len(recovered):,}", flush=True)
    print(f"(source_id,chrom,pos) key duplicates among recovered: {key_dupe}", flush=True)

    if key_dupe != 0:
        print("ABORT-WORTHY: recovered rows not unique on (source_id,chrom,pos).", flush=True)
        return 3
    if n_no_sid != 0:
        print(f"WARN: {n_no_sid} allele-less rows lack a source_id (unexpected).", flush=True)
    print("=== classify_alleleless_by_type DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
