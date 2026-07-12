#!/usr/bin/env python
"""
recover_by_sourceid.py  (2026-07-09)
==========================================================================
GROUND-UP re-key of the allele-less recovery onto each row's OWN source_id.

BACKGROUND / WHY THIS EXISTS
The previous recovery (recover_identity_first.py) resolved a variant's identity by a
LOCUS lookup: resolve_varid(chrom, pos, gene) returned whichever ClinVar VariationID
variant_summary listed first at that coordinate. For the many loci where several distinct
ClinVar variants share a start position (copy-number variants of differing extent, plus
co-located SNVs/indels), that lookup returned a DIFFERENT VariationID than the cohort
row's own. The recovery then fetched and genome-verified THAT neighbor's allele. The
allele verified (it is a real allele at the position) but belonged to the wrong variant.
A probe on 2026-07-09 showed this affected ALL 544 previously-"recovered" rows: in every
case the resolved id differed from the row's own source_id.

THE COHORT ALREADY CARRIES THE CORRECT IDENTITY in its `source_id` column, proven to be
the ClinVar VariationID (all sampled source_ids matched a VariationID at the row's exact
locus in variant_summary). The correct design is therefore to STOP re-deriving identity by
locus and READ source_id directly, fetching each row's allele strictly by its own
VariationID.

THE TRUE PER-ROW KEY is the triple (source_id, chrom, pos): source_id alone is not unique
(14 VariationIDs map to two loci each, e.g. pseudoautosomal-region genes present on both X
and Y), and variant_id (clinvar:CHR:POS:None:None) is degenerate (collapses distinct
co-located variants). The triple is unique across all allele-less rows.

WHAT THIS TOOL DOES  (over ALL allele-less rows, not just the old 544)
  For each allele-less cohort row:
    sid = row.source_id                      (its true ClinVar VariationID)
    Look sid up in the raw ClinVar VCF by ID, then the fresh VCF by ID.
    Classify:
      RECOVER_BY_SID_RAW / _FRESH   sid's VCF record has a real ref/alt that
                                    genome-verifies at the VCF pos AND variant_summary
                                    places sid at this row's (chrom,pos) locus.
      RECOVER_SID_LOCUS_MISMATCH    allele genome-verifies but variant_summary does NOT
                                    place sid at this row's locus  (quarantine).
      SID_GENOME_MISMATCH           sid's ref does not match the genome at its VCF pos
                                    (quarantine).
      CONFIRMED_ALLELELESS_CNV      sid IS present in ClinVar but its own record is
                                    allele-less (CNV/structural, na ref/alt) -> genuinely
                                    has no simple allele; exclude with an explicit reason.
      SID_NOT_IN_VCF_TRY_NCBI       sid absent from both VCFs -> NCBI fallback (as before).
      NO_SOURCE_ID                  row lacks a usable source_id (should be zero; fail-loud
                                    count reported).

OUTPUTS (all in --outdir)
  alleleless_recovery_by_sid_full.tsv     per-row disposition, KEY=(source_id,chrom,pos)
  alleleless_recovered_by_sid.tsv         only RECOVER_BY_SID_* rows, for the rebuild
  alleleless_recovery_by_sid_summary.json verdict counts + old-vs-new comparison
  alleleless_recovery_old_vs_new.tsv      per-row: old locus-resolved id vs source_id, and
                                          old allele vs new allele (quantifies the defect)

USAGE
  python scripts/recover_by_sourceid.py \
      --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --raw-vcf         data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --fresh-vcf       data/external/clinvar/clinvar.vcf.gz \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --fasta           data/external/grch38/GRCh38.fa \
      --old-recovered   outputs/alleleless_recovered_by_id.tsv \
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


# ------------------------------------------------------------------ helpers
def _norm_chrom(c) -> str:
    return str(c).strip().lstrip("chr")


def _clean_id(x) -> str:
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def _real(x) -> bool:
    """True if x is a real allele string (not null / na / '.')."""
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    return str(x).strip().lower() not in {"", "na", "nan", "none", "-", ".", "<na>"}


def _index_vcf_by_id(path: Path) -> dict:
    """VariationID -> (chrom, pos, ref, alt).  ClinVar VCF ID column is the VariationID."""
    by_id = {}
    if not path or not path.exists():
        return by_id
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t", 5)
            if len(p) < 5:
                continue
            pos = int(p[1]) if p[1].isdigit() else None
            by_id[_clean_id(p[2])] = (_norm_chrom(p[0]), pos, p[3], p[4])
    return by_id


def _load_vs_at(path: Path, assembly: str) -> dict:
    """(chrom, str(start)) -> set of VariationIDs at that locus (for locus cross-check)."""
    at = {}
    if not path or not path.exists():
        return at
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip",
                     usecols=lambda c: c in {"VariationID", "Assembly", "Chromosome", "Start"})
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    for c, s, vid in zip(vs["Chromosome"].map(_norm_chrom), vs["Start"].astype(str),
                         vs["VariationID"]):
        at.setdefault((c, s), set()).add(_clean_id(vid))
    return at


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--old-recovered", default="outputs/alleleless_recovered_by_id.tsv")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)

    print("=== recover_by_sourceid START ===", flush=True)
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

    # fail-loud: every allele-less row must carry a usable source_id
    n_no_sid = int((~al["source_id"].map(_real)).sum())
    print(f"allele-less rows lacking a usable source_id: {n_no_sid}", flush=True)

    raw_by_id = _index_vcf_by_id(Path(a.raw_vcf))
    fresh_by_id = _index_vcf_by_id(Path(a.fresh_vcf)) if a.fresh_vcf else {}
    print(f"raw VCF ids: {len(raw_by_id):,}  fresh VCF ids: {len(fresh_by_id):,}", flush=True)

    vs_at = _load_vs_at(Path(a.variant_summary), a.assembly)
    print(f"variant_summary loci indexed: {len(vs_at):,}", flush=True)

    ref_genome = None
    if Path(a.fasta).exists():
        from pyfaidx import Fasta
        ref_genome = Fasta(str(a.fasta), rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, refallele):
        if ref_genome is None or pos is None:
            return None
        c = _norm_chrom(chrom)
        if c not in contigs:
            return None
        try:
            got = str(ref_genome[c][int(pos) - 1:int(pos) - 1 + len(refallele)]).upper()
        except Exception:
            return None
        return got == str(refallele).upper()

    # old locus-resolved recovery, for the old-vs-new comparison
    old_by_key = {}
    if Path(a.old_recovered).exists():
        old = pd.read_csv(a.old_recovered, sep="\t", dtype=str)
        for _, r in old.iterrows():
            old_by_key[str(r.get("variant_id"))] = (
                _clean_id(r.get("cohort_varid")), r.get("rec_ref"), r.get("rec_alt"))

    rows = []
    cmp_rows = []
    for _, r in al.iterrows():
        vid = str(r["variant_id"])
        chrom = _norm_chrom(r["chrom"])
        pos = int(r["pos"])
        sid = r["source_id"]

        rec_pos = rec_ref = rec_alt = rec_source = None
        locus_ok = None
        gver = None

        if not _real(sid):
            verdict = "NO_SOURCE_ID"
        else:
            hit = None
            src = None
            h = raw_by_id.get(sid)
            if h is not None:
                hit, src = h, "raw"
            elif fresh_by_id.get(sid) is not None:
                hit, src = fresh_by_id.get(sid), "fresh"

            if hit is None:
                verdict = "SID_NOT_IN_VCF_TRY_NCBI"
            else:
                vchrom, vpos, vref, valt = hit
                if not (_real(vref) and _real(valt)):
                    # sid's own ClinVar record is itself allele-less (CNV/structural)
                    verdict = "CONFIRMED_ALLELELESS_CNV"
                else:
                    rec_pos, rec_ref, rec_alt, rec_source = vpos, vref, valt, src
                    gver = genome_ref_ok(vchrom, vpos, vref)
                    locus_ok = sid in vs_at.get((chrom, str(pos)), set()) \
                        or sid in vs_at.get((chrom, str(pos + 1)), set())
                    if gver and locus_ok:
                        verdict = f"RECOVER_BY_SID_{src.upper()}"
                    elif gver and not locus_ok:
                        verdict = "RECOVER_SID_LOCUS_MISMATCH"
                    else:
                        verdict = "SID_GENOME_MISMATCH"

        rows.append({
            "variant_id": vid, "chrom": r["chrom"], "pos": pos, "source_id": sid,
            "rec_pos": rec_pos, "rec_ref": rec_ref, "rec_alt": rec_alt,
            "rec_source": rec_source, "locus_ok": locus_ok, "genome_ok": gver,
            "verdict": verdict,
        })

        # old-vs-new comparison
        old = old_by_key.get(vid)
        if old is not None:
            old_id, old_ref, old_alt = old
            cmp_rows.append({
                "variant_id": vid, "source_id": sid, "old_resolved_id": old_id,
                "id_differs": (old_id != sid),
                "old_allele": f"{old_ref}>{old_alt}",
                "new_allele": (f"{rec_ref}>{rec_alt}" if rec_ref is not None else None),
                "allele_differs": (f"{old_ref}>{old_alt}" != (f"{rec_ref}>{rec_alt}" if rec_ref is not None else None)),
                "new_verdict": verdict,
            })

    res = pd.DataFrame(rows)
    res.to_csv(out / "alleleless_recovery_by_sid_full.tsv", sep="\t", index=False)

    recovered = res[res["verdict"].str.startswith("RECOVER_BY_SID_")].copy()
    recovered[["variant_id", "chrom", "pos", "source_id", "rec_pos", "rec_ref", "rec_alt",
               "rec_source", "verdict"]].to_csv(
        out / "alleleless_recovered_by_sid.tsv", sep="\t", index=False)

    cmp = pd.DataFrame(cmp_rows)
    if len(cmp):
        cmp.to_csv(out / "alleleless_recovery_old_vs_new.tsv", sep="\t", index=False)

    # ---- verify uniqueness of the (source_id, chrom, pos) key among recovered rows
    key_dupe = int(recovered.duplicated(subset=["source_id", "chrom", "pos"]).sum())

    counts = res["verdict"].value_counts().to_dict()
    summary = {
        "date": "2026-07-09",
        "cohort": a.cohort,
        "allele_less_rows": int(n_al),
        "rows_lacking_source_id": int(n_no_sid),
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "recovered_rows": int(len(recovered)),
        "recovered_unique_source_ids": int(recovered["source_id"].nunique()),
        "recovered_key_duplicates_(source_id,chrom,pos)": key_dupe,
        "old_vs_new": (
            {
                "compared_rows": int(len(cmp)),
                "old_rows_with_id_differing_from_source_id": int(cmp["id_differs"].sum()),
                "old_rows_with_allele_differing_from_new": int(cmp["allele_differs"].sum()),
            } if len(cmp) else {}
        ),
    }
    (out / "alleleless_recovery_by_sid_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report
    print("\n--- SOURCE_ID-KEYED DISPOSITION (all rows) ---", flush=True)
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:32s}: {counts[k]:,}", flush=True)
    print(f"\nrecovered rows: {len(recovered):,}  "
          f"(unique source_ids: {recovered['source_id'].nunique():,})", flush=True)
    print(f"(source_id,chrom,pos) key duplicates among recovered: {key_dupe}", flush=True)
    if len(cmp):
        print(f"\nold-vs-new: of {len(cmp):,} old recovered rows, "
              f"{int(cmp['id_differs'].sum()):,} had a resolved id != the row's source_id; "
              f"{int(cmp['allele_differs'].sum()):,} got a DIFFERENT allele under the correct id.",
              flush=True)
    # fail-loud on the invariant that matters for the rebuild
    if key_dupe != 0:
        print("ABORT-WORTHY: recovered rows are not unique on (source_id,chrom,pos).", flush=True)
        return 3
    print("=== recover_by_sourceid DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
