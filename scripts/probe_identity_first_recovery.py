"""
probe_identity_first_recovery.py  (2026-07-09)
==========================================================================
DECISION-SUPPORT ONLY. Recovers nothing; changes nothing. It shows, on a stratified
sample of the RECOVER rows, what IDENTITY-FIRST recovery would decide versus what the
positional-window probe actually did -- so the recovery strategy can be chosen on evidence.

Motivation: diagnose_identity_join.py revealed that the +/-25bp positional probe attached
NEIGHBOR variants' alleles (e.g. OAT 10:124404664 is VariationID 174 at Start==pos, but the
probe grabbed 1213865 from 18bp away). The correct key is the cohort row's OWN VariationID,
which variant_summary carries at Start==cohort_pos.

For each sampled row this computes:
  cohort_varid   : variant_summary VariationID where Start==pos (gene-consistent if possible)
  cohort_type    : that row's ClinVar Type
  in_raw_by_id   : is cohort_varid present in the raw VCF, keyed by ID? (+ its ref/alt)
  in_fresh_by_id : same against the fresh VCF
  probe_varid    : the VariationID the positional probe attached (from the disposition)
  probe_was_wrong: probe_varid != cohort_varid  (the mis-attachment flag)
  identity_first_verdict:
     RECOVER_BY_ID        cohort_varid found in a VCF by ID  (the safe recovery)
     STALE_MISS_TRY_NCBI  cohort_varid exists but in neither VCF (stale -> NCBI)
     REPEAT_NO_SEQ_ALLELE cohort_type is microsatellite/repeat -> no simple SNV allele
     NO_VARID_AT_POS      variant_summary has no row at Start==pos

Sampling is stratified by true offset band (0, 1, 2-5, 6-25) and mixes Types.

USAGE
  python scripts/probe_identity_first_recovery.py \
      --disposition     outputs/alleleless_final_disposition.tsv \
      --cohort          data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --raw-vcf         data/raw/clinvar/clinvar_GRCh38.vcf.gz \
      --fresh-vcf       data/external/clinvar/clinvar.vcf.gz \
      --variant-summary data/external/clinvar/variant_summary.txt.gz \
      --fasta           data/external/grch38/GRCh38.fa \
      --assembly GRCh38 --per-band 8
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import sys
from pathlib import Path

import pandas as pd

REPEAT_TYPES = {"microsatellite", "tandem repeat"}


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _clean_id(x) -> str:
    """Normalize a VariationID that may arrive as float '123.0', int, or str."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _index_vcf_by_id_and_pos(path: Path):
    """Return (by_id, by_chrom_pos): id -> (chrom,pos,ref,alt); chrom -> sorted [pos]."""
    by_id, by_chrom = {}, {}
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
            by_id[_clean_id(f[2])] = (c, p, f[3], f[4])
            by_chrom.setdefault(c, []).append(p)
    for c in by_chrom:
        by_chrom[c].sort()
    return by_id, by_chrom


def _load_vs(path: Path, assembly: str) -> pd.DataFrame:
    want = ["VariationID", "Type", "GeneSymbol", "Assembly", "Chromosome", "Start"]
    head = pd.read_csv(path, sep="\t", nrows=0, dtype=str, compression="gzip")
    cols = [c for c in want if c in head.columns]
    vs = pd.read_csv(path, sep="\t", dtype=str, compression="gzip", usecols=cols)
    if "Assembly" in vs.columns:
        vs = vs[vs["Assembly"].isin([assembly, "na"])]
    return vs


def _closest_offset(by_chrom, chrom, pos, win=25):
    rows = by_chrom.get(_norm_chrom(chrom))
    if not rows:
        return None
    lo = bisect.bisect_left(rows, pos - win)
    hi = bisect.bisect_right(rows, pos + win)
    best = None
    for p in rows[lo:hi]:
        d = abs(p - pos)
        if best is None or d < best:
            best = d
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disposition", default="outputs/alleleless_final_disposition.tsv")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--raw-vcf", required=True)
    ap.add_argument("--fresh-vcf", default=None)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--assembly", default="GRCh38")
    ap.add_argument("--per-band", type=int, default=8)
    ap.add_argument("--out", default="outputs/identity_first_sample.tsv")
    a = ap.parse_args(argv)

    disp = pd.read_csv(a.disposition, sep="\t")
    rec = disp[disp["bucket"] == "RECOVER"].copy()
    coh = pd.read_parquet(a.cohort)
    gene_by_id = dict(zip(coh["variant_id"], coh.get("gene_symbol", pd.Series(dtype=str))))

    raw_by_id, raw_by_chrom = _index_vcf_by_id_and_pos(Path(a.raw_vcf))
    fresh_by_id = {}
    if a.fresh_vcf and Path(a.fresh_vcf).exists():
        fresh_by_id, _ = _index_vcf_by_id_and_pos(Path(a.fresh_vcf))

    vs = _load_vs(Path(a.variant_summary), a.assembly)
    vs["_c"] = vs["Chromosome"].map(_norm_chrom)
    vs["_s"] = vs["Start"].astype(str)
    # index: (chrom, start) -> list of (VariationID, Type, Gene)
    vs_at = {}
    for c, s, vid, t, g in zip(vs["_c"], vs["_s"], vs["VariationID"], vs["Type"],
                               vs.get("GeneSymbol", pd.Series([""] * len(vs)))):
        vs_at.setdefault((c, s), []).append((_clean_id(vid), t, g))

    ref_genome = None
    if Path(a.fasta).exists():
        from pyfaidx import Fasta
        ref_genome = Fasta(str(a.fasta), rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, refallele):
        if ref_genome is None:
            return None
        c = _norm_chrom(chrom)
        if c not in contigs:
            return None
        try:
            got = str(ref_genome[c][int(pos) - 1:int(pos) - 1 + len(refallele)]).upper()
        except Exception:
            return None
        return got == str(refallele).upper()

    rec["_true_offset"] = [
        _closest_offset(raw_by_chrom, c, int(p)) for c, p in zip(rec["chrom"], rec["pos"].astype(int))
    ]

    def band(o):
        if o is None:
            return "no_vcf"
        if o == 0:
            return "A_off0"
        if o == 1:
            return "B_off1"
        if o <= 5:
            return "C_off2_5"
        return "D_off6_25"

    rec["_band"] = rec["_true_offset"].map(band)
    sample = pd.concat([d.head(a.per_band) for _, d in rec.groupby("_band")],
                       ignore_index=True) if len(rec) else rec

    out_rows = []
    for _, r in sample.iterrows():
        vid = r["variant_id"]; chrom = _norm_chrom(r["chrom"]); pos = int(r["pos"])
        gene = str(gene_by_id.get(vid, "")) or None
        at = vs_at.get((chrom, str(pos)), [])
        # prefer a gene-consistent entry, else first
        cohort_varid, cohort_type = None, None
        for (v, t, g) in at:
            if gene and g and str(g).upper() == gene.upper():
                cohort_varid, cohort_type = v, t
                break
        if cohort_varid is None and at:
            cohort_varid, cohort_type = at[0][0], at[0][1]

        in_raw = raw_by_id.get(_clean_id(cohort_varid)) if cohort_varid else None
        in_fresh = fresh_by_id.get(_clean_id(cohort_varid)) if cohort_varid else None
        probe_varid = _clean_id(r["variation_id"])
        probe_wrong = (cohort_varid is not None and probe_varid != _clean_id(cohort_varid))

        if cohort_varid is None:
            verdict = "NO_VARID_AT_POS"
            rec_ref = rec_alt = None; gver = None
        elif (cohort_type or "").strip().lower() in REPEAT_TYPES:
            verdict = "REPEAT_NO_SEQ_ALLELE"
            rec_ref = rec_alt = None; gver = None
        elif in_raw or in_fresh:
            src = in_raw or in_fresh
            _, vpos, rec_ref, rec_alt = src
            gver = genome_ref_ok(chrom, vpos, rec_ref)
            verdict = "RECOVER_BY_ID"
        else:
            verdict = "STALE_MISS_TRY_NCBI"
            rec_ref = rec_alt = None; gver = None

        out_rows.append({
            "variant_id": vid, "band": r["_band"], "true_offset": r["_true_offset"],
            "gene": gene, "cohort_varid": cohort_varid, "cohort_type": cohort_type,
            "identity_first_verdict": verdict,
            "id_first_ref": rec_ref, "id_first_alt": rec_alt, "id_first_genome_ok": gver,
            "probe_varid": probe_varid, "probe_ref": r.get("ref"), "probe_alt": r.get("alt"),
            "probe_was_wrong": probe_wrong,
        })

    res = pd.DataFrame(out_rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, sep="\t", index=False)

    print("IDENTITY-FIRST vs POSITIONAL-PROBE  (decision-support sample, 2026-07-09)")
    print("=" * 68)
    for b in ["A_off0", "B_off1", "C_off2_5", "D_off6_25", "no_vcf"]:
        sub = res[res["band"] == b]
        if not len(sub):
            continue
        wrong = int(sub["probe_was_wrong"].sum())
        print(f"\nband {b}  (n={len(sub)}, probe_was_wrong={wrong}/{len(sub)}):")
        print(sub[["variant_id", "cohort_varid", "cohort_type",
                   "identity_first_verdict", "probe_varid", "probe_was_wrong"]]
              .to_string(index=False))
    print(f"\nfull sample table -> {a.out}")
    # overall mis-attach rate by band on the FULL recover set (not just sample)
    print("\n--- projected mis-attach exposure (full RECOVER set) ---")
    print(rec["_band"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
