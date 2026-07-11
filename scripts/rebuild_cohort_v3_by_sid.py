#!/usr/bin/env python
"""
rebuild_cohort_v3_by_sid.py  (2026-07-09)
==========================================================================
Build cohort v3 from the canonical v2, merging recovered allele-less alleles keyed on the
UNIQUE per-row triple (source_id, chrom, pos) -- never the degenerate variant_id.

WHY THE TRIPLE
  * variant_id = clinvar:CHR:POS:None:None is a placeholder that COLLAPSES distinct
    co-located ClinVar variants (e.g. 91 different 22q11 CNVs share one variant_id).
    Merging by it splatters one allele across distinct variants.
  * source_id (the true ClinVar VariationID) is unique per row EXCEPT for 14 ids that map
    to two loci (pseudoautosomal-region genes on X and Y, multi-mapping variants).
  * (source_id, chrom, pos) is unique across every allele-less row. It is the correct key.

INPUTS
  --cohort-v2         canonical v2 parquet (MD5-checked)
  --recovered-by-sid  alleleless_recovered_by_sid.tsv from recover_by_sourceid.py
                      (columns: variant_id, chrom, pos, source_id, rec_pos, rec_ref,
                       rec_alt, rec_source, verdict)
  --disposition       alleleless_recovery_by_sid_full.tsv (all rows + verdicts, for
                      excluded-row provenance)
  --fasta             GRCh38 for independent genome re-verification

MERGE SEMANTICS (per recovered row)
  Match the ONE cohort row where (source_id, chrom, pos) == (rec.source_id, rec.chrom,
  rec.pos). Rewrite that row: pos -> rec_pos, ref -> rec_ref, alt -> rec_alt,
  variant_id -> clinvar:chrom:rec_pos:ref:alt. Every non-recovered allele-less row is
  excluded (dropped) with its verdict recorded.

GUARDS (all fail-loud; any failure aborts with a distinct return code, no v3 written)
  rc 2  refuse to overwrite an existing --out unless --force
  rc 3  --cohort-v2 MD5 != expected canonical (unless --skip-md5-check)
  rc 4  a recovered allele fails independent genome re-verification at rec_pos
  rc 5  a recovered (source_id,chrom,pos) triple is not present in the allele-less set
  rc 6  row-count reconciliation fails: kept + excluded != total allele-less rows
  rc 7  kept and excluded triples overlap
  rc 9  a rewritten canonical variant_id collides with an existing cohort row
  rc 10 duplicate variant_id in the final v3
  rc 8  any na:na row remains in v3 after merge
"""

from __future__ import annotations

import argparse
import hashlib
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

EXPECTED_MD5 = "F3152F671E920A0A0C19A696563002E0"


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


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-v2", required=True)
    ap.add_argument("--recovered-by-sid", required=True)
    ap.add_argument("--disposition", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-md5-check", action="store_true")
    a = ap.parse_args(argv)

    print("=== rebuild_cohort_v3_by_sid START ===", flush=True)
    out_path = Path(a.out)

    # GUARD 2: refuse overwrite
    if out_path.exists() and not a.force:
        print(f"ABORT: {out_path} exists; pass --force to overwrite.", file=sys.stderr)
        return 2

    # GUARD 3: canonical MD5
    md5 = _md5(Path(a.cohort_v2))
    if not a.skip_md5_check and md5 != EXPECTED_MD5:
        print(f"ABORT: cohort-v2 MD5 {md5} != expected {EXPECTED_MD5}.", file=sys.stderr)
        return 3

    coh = pd.read_parquet(a.cohort_v2)
    coh["source_id"] = coh["source_id"].map(_clean_id)
    coh["chrom"] = coh["chrom"].astype(str)
    coh["pos"] = coh["pos"].astype(int)
    print(f"v2 rows: {len(coh):,}  (MD5 {md5})", flush=True)

    al_mask = is_allele_less(coh["ref"], coh["alt"])
    al = coh[al_mask]
    n_al = len(al)
    al_triples = set(zip(al["source_id"], al["chrom"], al["pos"]))
    print(f"allele-less rows: {n_al:,}  unique triples: {len(al_triples):,}", flush=True)

    # ---- load recoveries, keyed on the triple
    rec = pd.read_csv(a.recovered_by_sid, sep="\t")
    rec["source_id"] = rec["source_id"].map(_clean_id)
    rec["chrom"] = rec["chrom"].astype(str)
    rec["pos"] = rec["pos"].astype(int)
    rec_map = {}   # (source_id, chrom, pos) -> (rec_pos, ref, alt, source)
    for _, r in rec.iterrows():
        rpos = int(r["rec_pos"]) if pd.notna(r.get("rec_pos")) else int(r["pos"])
        rec_map[(r["source_id"], _norm_chrom(r["chrom"]), int(r["pos"]))] = (
            rpos, str(r["rec_ref"]), str(r["rec_alt"]), str(r.get("rec_source", "vcf")))
    print(f"recovered rows: {len(rec_map):,}", flush=True)

    # GUARD 3b (genome re-verify at rec_pos)
    ref_genome = None
    if Path(a.fasta).exists():
        from pyfaidx import Fasta
        ref_genome = Fasta(str(a.fasta), rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, ref):
        if ref_genome is None:
            return None
        c = _norm_chrom(chrom)
        if c not in contigs:
            return False
        try:
            got = str(ref_genome[c][int(pos) - 1:int(pos) - 1 + len(ref)]).upper()
        except Exception:
            return False
        return got == str(ref).upper()

    failed = []
    for (sid, chrom, pos), (rpos, ref, alt, src) in rec_map.items():
        if not (_real(ref) and _real(alt)):
            failed.append((sid, "null_allele")); continue
        ok = genome_ref_ok(chrom, rpos, ref)
        if ok is None:
            failed.append((sid, "no_genome")); continue
        if not ok:
            failed.append((sid, "genome_mismatch"))
    if failed:
        print(f"ABORT: {len(failed)} recoveries failed genome re-verification at rec_pos. "
              f"First: {failed[:5]}", file=sys.stderr)
        return 4

    # GUARD 5: recovered triples must be a subset of allele-less triples
    kept_triples = set(rec_map.keys())
    stray = kept_triples - al_triples
    if stray:
        print(f"ABORT: {len(stray)} recovered triples not in the allele-less set. "
              f"First: {list(stray)[:5]}", file=sys.stderr)
        return 5

    excluded_triples = al_triples - kept_triples

    # GUARD 6: per-ROW reconciliation
    if len(kept_triples) + len(excluded_triples) != n_al:
        print(f"ABORT: kept({len(kept_triples)}) + excluded({len(excluded_triples)}) "
              f"!= allele-less rows({n_al}).", file=sys.stderr)
        return 6
    # GUARD 7: disjoint
    if kept_triples & excluded_triples:
        print("ABORT: kept and excluded triples overlap.", file=sys.stderr)
        return 7

    # ---- apply merge, matched on the triple (exactly one row each)
    coh = coh.copy()
    triple = list(zip(coh["source_id"], coh["chrom"], coh["pos"]))
    coh["_triple"] = triple
    existing_ids = set(coh["variant_id"])
    collisions = []
    excl_triples_set = excluded_triples

    # build new column values only for kept rows
    new_vid = coh["variant_id"].copy()
    new_pos = coh["pos"].copy()
    new_ref = coh["ref"].copy()
    new_alt = coh["alt"].copy()
    idx_by_triple = {}
    for i, t in enumerate(coh["_triple"]):
        # a triple can appear once (unique among allele-less); for non-allele-less rows the
        # triple may repeat but those are never in rec_map, so this is safe.
        idx_by_triple.setdefault(t, []).append(i)

    for (sid, chrom, pos), (rpos, ref, alt, src) in rec_map.items():
        idxs = idx_by_triple.get((sid, chrom, pos), [])
        if len(idxs) != 1:
            print(f"ABORT: recovered triple ({sid},{chrom},{pos}) matches {len(idxs)} "
                  f"cohort rows (expected 1).", file=sys.stderr)
            return 5
        i = idxs[0]
        canon = f"clinvar:{chrom}:{rpos}:{ref}:{alt}"
        if canon in existing_ids and canon != coh.iat[i, coh.columns.get_loc("variant_id")]:
            collisions.append((sid, canon))
        new_vid.iat[i] = canon
        new_pos.iat[i] = rpos
        new_ref.iat[i] = ref
        new_alt.iat[i] = alt

    if collisions:
        print(f"ABORT: {len(collisions)} recovered ids collide with existing rows. "
              f"First: {collisions[:5]}", file=sys.stderr)
        return 9

    coh["variant_id"] = new_vid
    coh["pos"] = new_pos
    coh["ref"] = new_ref
    coh["alt"] = new_alt

    # drop excluded allele-less rows (those whose triple is in excluded set)
    drop_mask = coh["_triple"].isin(excl_triples_set)
    v3 = coh[~drop_mask].drop(columns=["_triple"]).reset_index(drop=True)

    # GUARD 8: no na:na remain
    remaining = int(is_allele_less(v3["ref"], v3["alt"]).sum())
    if remaining != 0:
        print(f"ABORT: {remaining} na:na rows remain in v3.", file=sys.stderr)
        return 8

    # GUARD 10: no duplicate variant_id
    dup = int(v3["variant_id"].duplicated().sum())
    if dup:
        print(f"ABORT: {dup} duplicate variant_ids in v3.", file=sys.stderr)
        return 10

    # ---- write outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v3.to_parquet(out_path, index=False)
    new_md5 = _md5(out_path)

    disp = pd.read_csv(a.disposition, sep="\t")
    disp_verdict = {}
    if "verdict" in disp.columns:
        disp["source_id"] = disp["source_id"].map(_clean_id)
        disp["chrom"] = disp["chrom"].astype(str)
        disp["pos"] = disp["pos"].astype(int)
        for _, r in disp.iterrows():
            disp_verdict[(r["source_id"], _norm_chrom(r["chrom"]), int(r["pos"]))] = r["verdict"]

    excl = pd.DataFrame(
        [{"source_id": s, "chrom": c, "pos": p,
          "verdict": disp_verdict.get((s, c, p), "CONFIRMED_ALLELELESS")}
         for (s, c, p) in sorted(excluded_triples)])
    excl.to_csv(out_path.parent / "cohort_v3_excluded_alleleless.tsv", sep="\t", index=False)

    recon = {
        "date": "2026-07-09",
        "v2_md5": md5,
        "v3_md5": new_md5,
        "v2_rows": int(len(coh)),
        "v3_rows": int(len(v3)),
        "allele_less_rows": int(n_al),
        "recovered_merged": int(len(kept_triples)),
        "excluded_total": int(len(excluded_triples)),
        "na_na_remaining": remaining,
        "reconciliation_ok": (len(kept_triples) + len(excluded_triples) == n_al),
        "key": "(source_id, chrom, pos)",
    }
    (out_path.parent / "cohort_v3_reconciliation.json").write_text(json.dumps(recon, indent=2))

    print(f"\nv3 rows          : {len(v3):,}", flush=True)
    print(f"recovered merged : {len(kept_triples):,}", flush=True)
    print(f"excluded         : {len(excluded_triples):,}", flush=True)
    print(f"na:na remaining  : {remaining}", flush=True)
    print(f"v3 MD5           : {new_md5}", flush=True)
    print("=== rebuild_cohort_v3_by_sid DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
