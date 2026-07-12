"""
diagnose_and_recover_alleleless.py  (2026-07-09)
==========================================================================
Second-stage handling of the 19,988 allele-less (na:na) rows, after
recover_alleleless_provenance.py established three populations by ClinVar-VCF evidence:
    RECOVERABLE_FROM_VCF    1,829 (1,209 pathogenic)  -- present in ClinVar VCF; patch missed
    LEGITIMATELY_ALLELELESS 14,248 (4,563 pathogenic)  -- absent from VCF, out-of-scope Type
    NEEDS_REVIEW            3,911 (1,956 pathogenic)  -- unresolved

This tool does FOUR things, all read-only against the cohort:

  (1) RECOVER + GENOME-VERIFY the 1,829. It re-reads the ClinVar VCF row for each recoverable
      position and, crucially, GENOME-VERIFIES the recovered ref against GRCh38. This guards
      against the padded-deletion off-by-one: cohort pos for a padded deletion = VCF pos - 1,
      so a naive chrom:pos match could read the wrong VCF row. Only genome-confirmed recoveries
      are emitted to the recovery table; mismatches are quarantined for inspection.

  (2) DIAGNOSE the root cause of the patch miss: tabulate the recoverable rows by whether they
      are padded-deletion-shaped, multi-allelic in the VCF, chromosome distribution, and
      variant_summary Type -- so the fix goes into patch_clinvar_alleles.py's join, not just a
      backfill.

  (3) TIGHTEN NEEDS_REVIEW: re-join variant_summary on BOTH pos and pos+1 (defeating the
      padded-deletion pos shift that made the first Start-key join miss), pull Type, and
      reclassify: out-of-scope Type -> LEGITIMATELY_ALLELELESS; present-in-VCF sequence Type
      -> RECOVERABLE; else stays NEEDS_REVIEW.

  (4) INSPECT the present-but-null-allele rows (in VCF but ALT is '.'/empty): dump them fully.

Outputs (outputs/):
    alleleless_recovered_verified.tsv   -- genome-confirmed chrom,pos,ref,alt,variation_id
    alleleless_recovery_quarantine.tsv  -- recoverable-but-genome-mismatch rows (inspect)
    alleleless_patch_miss_diagnosis.json-- root-cause tabulation
    alleleless_needsreview_reclassified.tsv
    alleleless_null_allele_inspect.tsv  -- the ~13 present-but-null rows

USAGE
    python scripts/diagnose_and_recover_alleleless.py \
        --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
        --clinvar-vcf     data/raw/clinvar/clinvar_GRCh38.vcf.gz \
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
    from genomic_variant_classifier.data.allele_classify import (
        is_allele_less, is_padded_deletion)
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from allele_classify import is_allele_less, is_padded_deletion  # type: ignore

OUT_OF_SCOPE_TYPES = {
    "copy number gain", "copy number loss", "translocation", "fusion", "complex",
    "microsatellite", "tandem repeat", "variation", "protein only", "inversion", "cytogenetic",
}
SEQUENCE_TYPES = {
    "single nucleotide variant", "deletion", "insertion", "duplication", "indel", "delins",
}
_NULL = {"", "na", "nan", "none", "-", "."}

_REC_COLS = ["variant_id", "chrom", "cohort_pos", "vcf_pos", "ref", "alt",
             "variation_id", "genome_verified", "multi_allelic", "pos_shifted"]


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _real(x) -> bool:
    s = str(x).strip().lower() if x is not None else ""
    return s not in _NULL and len(s) >= 1


def _parse_vcf_full(path: Path):
    """CHROM:POS -> list of (ref, alt, vid); keep ALL rows per pos (multi-allelic aware)."""
    by_pos = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            chrom, pos, vid, ref, alt = _norm_chrom(f[0]), f[1], f[2], f[3], f[4]
            by_pos.setdefault(f"{chrom}:{pos}", []).append((ref, alt, vid))
    return by_pos


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--clinvar-vcf", required=True)
    ap.add_argument("--variant-summary", default=None)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    coh = pd.read_parquet(a.cohort)
    al = coh[is_allele_less(coh["ref"], coh["alt"])].copy()
    al["_key"] = al["chrom"].map(_norm_chrom) + ":" + al["pos"].astype(str)
    al["_key1"] = al["chrom"].map(_norm_chrom) + ":" + (al["pos"].astype(int) + 1).astype(str)
    print(f"allele-less rows: {len(al):,}")

    print("parsing ClinVar VCF (multi-allelic aware) ...")
    by_pos = _parse_vcf_full(Path(a.clinvar_vcf))

    ref_genome = _open_ref(Path(a.fasta)) if Path(a.fasta).exists() else None
    if ref_genome is None:
        print(f"WARNING: FASTA not found at {a.fasta}; genome verification DISABLED "
              f"(recoveries will be marked UNVERIFIED).", file=sys.stderr)
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

    # ---- (1)+(2) recover + genome-verify the rows present in the VCF ----
    recovered, quarantine, diag_rows = [], [], []
    for _, r in al.iterrows():
        # a recoverable row is present at its own pos OR at pos+1 (padded-deletion shift)
        cand = []
        for key, kpos in ((r["_key"], int(r["pos"])), (r["_key1"], int(r["pos"]) + 1)):
            for (vref, valt, vid) in by_pos.get(key, []):
                cand.append((kpos, vref, valt, vid))
        # keep only candidates with real alleles
        cand = [c for c in cand if _real(c[1]) and _real(c[2])]
        if not cand:
            continue
        # prefer a candidate whose ref genome-verifies at the VCF pos
        chosen = None
        for (kpos, vref, valt, vid) in cand:
            if genome_ref_ok(r["chrom"], kpos, vref):
                chosen = (kpos, vref, valt, vid, True)
                break
        if chosen is None:
            kpos, vref, valt, vid = cand[0]
            chosen = (kpos, vref, valt, vid, False)
        kpos, vref, valt, vid, verified = chosen
        rec = {"variant_id": r["variant_id"], "chrom": r["chrom"],
               "cohort_pos": int(r["pos"]), "vcf_pos": kpos, "ref": vref, "alt": valt,
               "variation_id": vid, "genome_verified": verified,
               "multi_allelic": ("," in valt), "pos_shifted": (kpos != int(r["pos"]))}
        (recovered if verified else quarantine).append(rec)
        diag_rows.append(rec)

    rec_df = pd.DataFrame(recovered, columns=_REC_COLS)
    quar_df = pd.DataFrame(quarantine, columns=_REC_COLS)
    diag_df = pd.DataFrame(diag_rows, columns=_REC_COLS)
    rec_df.to_csv(out / "alleleless_recovered_verified.tsv", sep="\t", index=False)
    quar_df.to_csv(out / "alleleless_recovery_quarantine.tsv", sep="\t", index=False)

    diagnosis = {
        "date": "2026-07-09",
        "recoverable_candidates": int(len(diag_df)),
        "genome_verified_recovered": int(len(rec_df)),
        "genome_mismatch_quarantined": int(len(quar_df)),
        "of_recovered_pos_shifted": int(diag_df["pos_shifted"].sum()) if len(diag_df) else 0,
        "of_recovered_multi_allelic": int(diag_df["multi_allelic"].sum()) if len(diag_df) else 0,
        "chrom_distribution": (diag_df["chrom"].astype(str).value_counts().to_dict()
                               if len(diag_df) else {}),
    }
    print(f"  recoverable candidates          : {diagnosis['recoverable_candidates']:,}")
    print(f"  genome-verified recovered       : {diagnosis['genome_verified_recovered']:,}")
    print(f"  genome-mismatch quarantined     : {diagnosis['genome_mismatch_quarantined']:,}")
    print(f"  of recovered, pos-shifted (pd)  : {diagnosis['of_recovered_pos_shifted']:,}")
    print(f"  of recovered, multi-allelic     : {diagnosis['of_recovered_multi_allelic']:,}")

    # ---- (3) tighten NEEDS_REVIEW via variant_summary on pos AND pos+1 ----
    if a.variant_summary:
        vs = _load_vs(Path(a.variant_summary), a.assembly)
        if {"Chromosome", "Start", "Type"} <= set(vs.columns):
            vs["_k"] = vs["Chromosome"].map(_norm_chrom) + ":" + vs["Start"].astype(str)
            vk = vs.drop_duplicates("_k").set_index("_k")["Type"]
            al["_type0"] = al["_key"].map(vk)
            al["_type1"] = al["_key1"].map(vk)
            al["_type"] = al["_type0"].fillna(al["_type1"])
            recovered_keys = set(rec_df["variant_id"]) | set(quar_df["variant_id"]) if len(diag_df) else set()

            def reclass(row):
                if row["variant_id"] in recovered_keys:
                    return "RECOVERABLE_FROM_VCF"
                t = str(row["_type"]).strip().lower() if pd.notna(row["_type"]) else ""
                if t in OUT_OF_SCOPE_TYPES:
                    return "LEGITIMATELY_ALLELELESS"
                if t in SEQUENCE_TYPES:
                    return "NEEDS_REVIEW_SEQTYPE_ABSENT_FROM_VCF"  # suspicious: should be in VCF
                return "NEEDS_REVIEW_UNRESOLVED"

            al["verdict2"] = al.apply(reclass, axis=1)
            al[["variant_id", "chrom", "pos", "_type", "verdict2"]].to_csv(
                out / "alleleless_needsreview_reclassified.tsv", sep="\t", index=False)
            diagnosis["reclassified_counts"] = al["verdict2"].value_counts().to_dict()
            print("  reclassified NEEDS_REVIEW        :", diagnosis["reclassified_counts"])

    # ---- (4) inspect present-but-null-allele rows ----
    null_rows = []
    for _, r in al.iterrows():
        for key in (r["_key"], r["_key1"]):
            for (vref, valt, vid) in by_pos.get(key, []):
                if not (_real(vref) and _real(valt)):
                    null_rows.append({"variant_id": r["variant_id"], "chrom": r["chrom"],
                                      "pos": int(r["pos"]), "vcf_ref": vref, "vcf_alt": valt,
                                      "variation_id": vid})
    _NULL_COLS = ["variant_id", "chrom", "pos", "vcf_ref", "vcf_alt", "variation_id"]
    null_df = (pd.DataFrame(null_rows, columns=_NULL_COLS).drop_duplicates("variant_id")
               if null_rows else pd.DataFrame(columns=_NULL_COLS))
    null_df.to_csv(out / "alleleless_null_allele_inspect.tsv", sep="\t", index=False)
    diagnosis["present_but_null_allele"] = int(len(null_df))
    print(f"  present-but-null-allele rows     : {len(null_df):,}")

    (out / "alleleless_patch_miss_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/alleleless_recovered_verified.tsv "
          f"({len(rec_df):,} genome-confirmed recoveries)")
    print(f"wrote {out}/alleleless_recovery_quarantine.tsv ({len(quar_df):,})")
    print(f"wrote {out}/alleleless_needsreview_reclassified.tsv")
    print(f"wrote {out}/alleleless_null_allele_inspect.tsv ({len(null_df):,})")
    print(f"wrote {out}/alleleless_patch_miss_diagnosis.json")

    if ref_genome is None:
        print("\n*** genome verification was DISABLED (no FASTA) -- rerun with --fasta so "
              "recoveries are genome-confirmed before any cohort patch. ***", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
