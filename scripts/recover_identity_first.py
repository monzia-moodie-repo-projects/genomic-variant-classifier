"""
recover_identity_first.py  (2026-07-09)
==========================================================================
Full-population, identity-FIRST recovery of the allele-less (na:na) rows, replacing the
invalidated positional-probe recovery.

Background: probe_identity_first_recovery.py showed (32/32 sampled) that the +/-25bp
positional probe attached NEIGHBOR variants' alleles -- it genome-verified the wrong
variant's ref. The correct key is the cohort row's OWN ClinVar VariationID, which
variant_summary carries at Start==cohort_pos. This tool recovers strictly BY VariationID.

For EVERY allele-less cohort row:
  1. Resolve cohort_varid = variant_summary.VariationID where Start==pos (gene-consistent
     if possible; else Start==pos+1 for the padded-deletion anchor; else gene-agnostic).
  2. Look up cohort_varid BY ID in the raw VCF and the fresh VCF. Take the VCF row's own
     POS/REF/ALT and GENOME-VERIFY ref at that POS against GRCh38. (True identity+genome
     verification -- the varid uniquely identifies the ClinVar variant.)
  3. Classify:
       RECOVER_BY_ID_RAW / _FRESH : varid found by ID, real ref/alt, ref genome-verifies.
       REPEAT_RECOVER_BY_ID       : cohort_type repeat/microsat AND varid found by ID with
                                    a real allele -> recover the TRUE repeat allele.
       REPEAT_ALLELELESS          : repeat/microsat AND varid not in any VCF -> allele-less.
       STALE_MISS_TRY_NCBI        : non-repeat, varid not in raw/fresh -> NCBI residual.
       NO_VARID_AT_POS            : variant_summary has nothing at Start==pos/pos+1.
  4. Compare to the OLD positional recovery: probe_was_wrong = (probe_varid != cohort_varid)
     and probe_correct = (probe_varid == cohort_varid). Reported across the FULL set so the
     decision to void the positional bucket rests on all rows, not a sample.

Outputs (outputs/):
  alleleless_identity_recovery_full.tsv     (per-row: cohort_varid, verdict, recovered ref/alt)
  alleleless_identity_recovery_summary.json (corrected disposition + probe-wrong rate)
  alleleless_recovered_by_id.tsv            (genome-verified by-ID recoveries to patch in)

Recovers alleles to TSV; writes NO cohort. The cohort build stays separately gated.

USAGE
  python scripts/recover_identity_first.py \
      --disposition     outputs/alleleless_final_disposition.tsv \
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

REPEAT_TYPES = {"microsatellite", "tandem repeat"}
_NULL = {"", "na", "nan", "none", "-", ".", "<na>"}


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _clean_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _real(x) -> bool:
    return str(x).strip().lower() not in _NULL and len(str(x).strip()) >= 1


def _index_vcf_by_id(path: Path):
    """VariationID(str) -> (chrom, pos, ref, alt)."""
    by_id = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            by_id[_clean_id(f[2])] = (_norm_chrom(f[0]), int(f[1]) if f[1].isdigit() else None,
                                      f[3], f[4])
    return by_id


def _load_vs(path: Path, assembly: str) -> pd.DataFrame:
    want = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disposition", default="outputs/alleleless_final_disposition.tsv")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args(argv)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    coh = pd.read_parquet(a.cohort)
    al = coh[is_allele_less(coh["ref"], coh["alt"])].copy()
    gene_by_id = dict(zip(coh["variant_id"], coh.get("gene_symbol", pd.Series(dtype=str))))
    print(f"allele-less rows: {len(al):,}")

    # old positional recovery (for the probe-wrong comparison)
    probe_varid_by_id = {}
    if Path(a.disposition).exists():
        disp = pd.read_csv(a.disposition, sep="\t")
        for _, r in disp[disp["bucket"] == "RECOVER"].iterrows():
            probe_varid_by_id[r["variant_id"]] = _clean_id(r.get("variation_id"))

    raw_by_id = _index_vcf_by_id(Path(a.raw_vcf))
    fresh_by_id = _index_vcf_by_id(Path(a.fresh_vcf)) if a.fresh_vcf and Path(a.fresh_vcf).exists() else {}

    vs = _load_vs(Path(a.variant_summary), a.assembly)
    vs["_c"] = vs["Chromosome"].map(_norm_chrom)
    vs["_s"] = vs["Start"].astype(str)
    vs_at = {}
    genecol = vs["GeneSymbol"] if "GeneSymbol" in vs.columns else pd.Series([""] * len(vs))
    for c, s, vid, t, g in zip(vs["_c"], vs["_s"], vs["VariationID"], vs["Type"], genecol):
        vs_at.setdefault((c, s), []).append((_clean_id(vid), str(t), str(g)))

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

    def resolve_varid(chrom, pos, gene):
        for key in ((chrom, str(pos)), (chrom, str(pos + 1))):
            entries = vs_at.get(key)
            if not entries:
                continue
            if gene:
                for (v, t, g) in entries:
                    if g and g.upper() == gene.upper():
                        return v, t
            return entries[0][0], entries[0][1]
        return None, None

    rows = []
    for _, r in al.iterrows():
        vid = r["variant_id"]; chrom = _norm_chrom(r["chrom"]); pos = int(r["pos"])
        gene = str(gene_by_id.get(vid, "")) or None
        cohort_varid, cohort_type = resolve_varid(chrom, pos, gene)
        is_repeat = (cohort_type or "").strip().lower() in REPEAT_TYPES

        rec = None; src = None
        if cohort_varid:
            hit = raw_by_id.get(_clean_id(cohort_varid))
            if hit and _real(hit[2]) and _real(hit[3]):
                rec, src = hit, "raw"
            elif fresh_by_id.get(_clean_id(cohort_varid)):
                h2 = fresh_by_id[_clean_id(cohort_varid)]
                if _real(h2[2]) and _real(h2[3]):
                    rec, src = h2, "fresh"

        rref = ralt = None; gver = None; vpos = None
        if rec is not None:
            vchrom, vpos, rref, ralt = rec
            gver = genome_ref_ok(vchrom, vpos, rref)

        if cohort_varid is None:
            verdict = "NO_VARID_AT_POS"
        elif rec is not None and gver:
            verdict = "REPEAT_RECOVER_BY_ID" if is_repeat else f"RECOVER_BY_ID_{src.upper()}"
        elif rec is not None and gver is False:
            verdict = "RECOVER_BY_ID_GENOME_MISMATCH"   # found by id but ref!=genome: quarantine
        elif is_repeat:
            verdict = "REPEAT_ALLELELESS"
        else:
            verdict = "STALE_MISS_TRY_NCBI"

        probe_vid = probe_varid_by_id.get(vid)
        rows.append({
            "variant_id": vid, "chrom": r["chrom"], "pos": pos, "gene": gene,
            "cohort_varid": cohort_varid, "cohort_type": cohort_type, "verdict": verdict,
            "rec_pos": vpos,   # the VCF's OWN verified coordinate for the recovered allele
            "rec_ref": rref, "rec_alt": ralt, "rec_source": src, "genome_ok": gver,
            "probe_varid": probe_vid,
            "probe_was_wrong": (probe_vid is not None and cohort_varid is not None
                                and _clean_id(probe_vid) != _clean_id(cohort_varid)),
            "probe_correct": (probe_vid is not None and cohort_varid is not None
                              and _clean_id(probe_vid) == _clean_id(cohort_varid)),
        })

    res = pd.DataFrame(rows)
    res.to_csv(out / "alleleless_identity_recovery_full.tsv", sep="\t", index=False)
    recovered = res[res["verdict"].isin(
        ["RECOVER_BY_ID_RAW", "RECOVER_BY_ID_FRESH", "REPEAT_RECOVER_BY_ID"])]
    recovered[["variant_id", "chrom", "pos", "rec_pos", "cohort_varid", "rec_ref", "rec_alt",
               "rec_source", "verdict"]].to_csv(
        out / "alleleless_recovered_by_id.tsv", sep="\t", index=False)

    probe_rows = res[res["probe_varid"].notna()]
    summary = {
        "date": "2026-07-09",
        "alleleless_total": int(len(res)),
        "corrected_disposition": res["verdict"].value_counts().to_dict(),
        "recovered_by_id_total": int(len(recovered)),
        "recovered_by_source": recovered["rec_source"].value_counts().to_dict(),
        "positional_probe_rows_scored": int(len(probe_rows)),
        "positional_probe_was_wrong": int(probe_rows["probe_was_wrong"].sum()),
        "positional_probe_correct": int(probe_rows["probe_correct"].sum()),
    }
    if "pathogenicity" in coh.columns:
        pth = coh.loc[al.index, "pathogenicity"].astype("string").str.lower().values
        res2 = res.copy(); res2["_p"] = (pth == "pathogenic")
        summary["pathogenic_by_verdict"] = res2.groupby("verdict")["_p"].sum().to_dict()
    (out / "alleleless_identity_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n--- CORRECTED IDENTITY-FIRST DISPOSITION (ALL rows) ---")
    for k, v in summary["corrected_disposition"].items():
        print(f"  {k:32s}: {v:,}")
    print(f"\nrecovered by-ID (genome-verified): {summary['recovered_by_id_total']:,}  "
          f"{summary['recovered_by_source']}")
    print(f"\nPOSITIONAL PROBE audit over {summary['positional_probe_rows_scored']:,} scored rows:")
    print(f"  probe attached WRONG variant : {summary['positional_probe_was_wrong']:,}")
    print(f"  probe attached CORRECT variant: {summary['positional_probe_correct']:,}")
    if summary["positional_probe_correct"] == 0:
        print("  => the positional RECOVER bucket is DEFINITIVELY void (0 correct of all scored).")
    if "pathogenic_by_verdict" in summary:
        print(f"\npathogenic by verdict: {summary['pathogenic_by_verdict']}")
    print(f"\nwrote {out}/alleleless_identity_recovery_full.tsv")
    print(f"wrote {out}/alleleless_recovered_by_id.tsv ({len(recovered):,})")
    print(f"wrote {out}/alleleless_identity_recovery_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
