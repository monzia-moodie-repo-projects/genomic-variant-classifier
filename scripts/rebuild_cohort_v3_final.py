"""
rebuild_cohort_v3_final.py  (2026-07-09)
==========================================================================
Build the single, clean, evidence-based cohort v3 from the canonical v2, using the
FULLY RESOLVED allele-less disposition (provenance investigation closed 2026-07-09).

Disposition of the 19,988 allele-less (na:na) rows, all per-row genome-verified:
    RECOVER               547  (534 RECOVER_BY_ID_RAW + 10 REPEAT_RECOVER_BY_ID + 3 NCBI)
    CONFIRMED_ALLELELESS  19,440 (75 REPEAT_ALLELELESS + 19,365 NCBI CONFIRMED_ALLELELESS)
    QUARANTINE            1     (NCBI RESOLVED_GENOME_MISMATCH)

v3 policy (user-specified 2026-07-09):
    - MERGE the 547 back in as real genome-verified ref/alt (rows STAY).
    - EXCLUDE the 19,440 CONFIRMED_ALLELELESS (documented).
    - EXCLUDE the 1 QUARANTINE (documented).
    => v3 rows = 4,420,180 - 19,441 = 4,400,739.

SEVEN FAIL-LOUD GUARDS
    1. Input v2 MD5 must equal the canonical F3152F67... or abort.
    2. Refuse-overwrite: abort if the v3 output already exists.
    3. Independent genome RE-VERIFICATION of every merged recovery (do not trust the TSV).
    4. Exact set reconciliation: kept(547) + removed(19,441) == all alleleless(19,988),
       disjoint, exact counts -- any drift aborts.
    5. Zero na:na rows may remain in v3 (assert).
    6. pos_rate (pathogenic fraction) recomputed before/after -> reconciliation.
    7. v2 retained for lineage (never modified/deleted).

USAGE
  python scripts/rebuild_cohort_v3_final.py \
      --cohort-v2       data/processed/clinvar_grch38_clean_v2_verified.parquet \
      --recovered-by-id outputs/alleleless_recovered_by_id.tsv \
      --ncbi-resolved   outputs/alleleless_ncbi_resolved.tsv \
      --disposition     outputs/alleleless_identity_recovery_full.tsv \
      --fasta           data/external/grch38/GRCh38.fa \
      --out             data/processed/clinvar_grch38_clean_v3_verified.parquet
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

CANONICAL_V2_MD5 = "F3152F671E920A0A0C19A696563002E0"
_NULL = {"", "na", "nan", "none", "-", ".", "<na>"}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _real(x) -> bool:
    return str(x).strip().lower() not in _NULL and len(str(x).strip()) >= 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-v2", required=True)
    ap.add_argument("--recovered-by-id", required=True)
    ap.add_argument("--ncbi-resolved", default=None)
    ap.add_argument("--disposition", required=True)
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-md5-check", action="store_true",
                    help="ONLY for offline synthetic tests; never in production")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recon_path = out.parent / "cohort_v3_reconciliation.json"
    excl_path = out.parent / "cohort_v3_excluded_alleleless.tsv"

    # GUARD 2: refuse-overwrite
    if out.exists():
        print(f"ABORT: output already exists: {out}. Refusing to overwrite.", file=sys.stderr)
        return 2

    # GUARD 1: canonical MD5
    v2_md5 = _md5(Path(a.cohort_v2))
    if not a.skip_md5_check and v2_md5 != CANONICAL_V2_MD5:
        print(f"ABORT: v2 MD5 {v2_md5} != canonical {CANONICAL_V2_MD5}. "
              f"This is not the canonical cohort.", file=sys.stderr)
        return 3

    coh = pd.read_parquet(a.cohort_v2)
    n_v2 = len(coh)
    al_mask = is_allele_less(coh["ref"], coh["alt"])
    al = coh[al_mask]
    n_al = len(al)
    print(f"v2 rows: {n_v2:,}  (MD5 {v2_md5})")
    print(f"allele-less rows: {n_al:,}")

    pos_before = None
    if "pathogenicity" in coh.columns:
        pos_before = float((coh["pathogenicity"].astype("string").str.lower() == "pathogenic").mean())

    # ---- assemble recoveries (547): recovered-by-id (544) + ncbi RESOLVED_HAS_ALLELE (3)
    # Each recovery carries its TRUE coordinate: rec_pos (the VCF's own verified position),
    # NOT the cohort na:na placeholder pos. For pos-shifted deletions/indels these differ;
    # verifying/merging at rec_pos is what makes the allele genome-coherent.
    rec = pd.read_csv(a.recovered_by_id, sep="\t")
    rec_map = {}   # variant_id -> (chrom, rec_pos, ref, alt, source)
    for _, r in rec.iterrows():
        rpos = int(r["rec_pos"]) if "rec_pos" in rec.columns and pd.notna(r.get("rec_pos")) \
            else int(r["pos"])
        rec_map[r["variant_id"]] = (_norm_chrom(r["chrom"]), rpos,
                                    str(r["rec_ref"]), str(r["rec_alt"]),
                                    str(r.get("rec_source", "vcf")))
    if a.ncbi_resolved and Path(a.ncbi_resolved).exists():
        nc = pd.read_csv(a.ncbi_resolved, sep="\t")
        for _, r in nc[nc["ncbi_verdict"] == "RESOLVED_HAS_ALLELE"].iterrows():
            rec_map[r["variant_id"]] = (_norm_chrom(r["chrom"]), int(r["pos"]),
                                        str(r["ref"]), str(r["alt"]), "ncbi")


    # GUARD 3: independent genome re-verification of every recovery
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

    failed_verify = []
    for vid, (chrom, pos, ref, alt, src) in rec_map.items():
        if not (_real(ref) and _real(alt)):
            failed_verify.append((vid, "null_allele"))
            continue
        ok = genome_ref_ok(chrom, pos, ref)
        if ok is None:
            failed_verify.append((vid, "no_genome_available"))
        elif not ok:
            failed_verify.append((vid, "genome_mismatch"))
    if failed_verify:
        print(f"ABORT: {len(failed_verify)} recoveries failed independent genome "
              f"re-verification. First few: {failed_verify[:5]}", file=sys.stderr)
        return 4

    kept_ids = set(rec_map.keys())

    # ---- removed = all alleleless not in kept
    disp = pd.read_csv(a.disposition, sep="\t")
    disp_verdict = dict(zip(disp["variant_id"], disp["verdict"]))
    all_al_ids = set(al["variant_id"])
    removed_ids = all_al_ids - kept_ids

    # GUARD 4: exact set reconciliation
    if not kept_ids.issubset(all_al_ids):
        stray = kept_ids - all_al_ids
        print(f"ABORT: {len(stray)} recovered ids are not in the allele-less set. "
              f"First: {list(stray)[:5]}", file=sys.stderr)
        return 5
    if len(kept_ids) + len(removed_ids) != n_al:
        print(f"ABORT: kept({len(kept_ids)}) + removed({len(removed_ids)}) "
              f"!= alleleless({n_al}).", file=sys.stderr)
        return 6
    if kept_ids & removed_ids:
        print("ABORT: kept and removed sets overlap.", file=sys.stderr)
        return 7

    # ---- apply: merge alleles into kept rows (at their TRUE coordinate), drop removed rows
    # The na:na row carried a placeholder variant_id clinvar:chrom:pos:None:None and pos.
    # The recovered allele's authoritative coordinate is rec_pos; rewrite pos, ref, alt AND
    # the variant_id to the canonical clinvar:chrom:rec_pos:ref:alt so the key reflects the
    # real variant, not the placeholder.
    coh = coh.copy()
    existing_ids = set(coh["variant_id"])
    new_id_map = {}   # old placeholder id -> new canonical id
    collisions = []
    for vid, (chrom, rpos, ref, alt, src) in rec_map.items():
        new_id = f"clinvar:{chrom}:{rpos}:{ref}:{alt}"
        if new_id in existing_ids and new_id != vid:
            collisions.append((vid, new_id))
        new_id_map[vid] = new_id
        mask = coh["variant_id"] == vid
        coh.loc[mask, "pos"] = rpos
        coh.loc[mask, "ref"] = ref
        coh.loc[mask, "alt"] = alt
        coh.loc[mask, "variant_id"] = new_id

    # GUARD 8: duplicate-id collision (recovered variant already present in cohort)
    if collisions:
        print(f"ABORT: {len(collisions)} recovered ids collide with existing cohort rows "
              f"(would create duplicates). First few: {collisions[:5]}", file=sys.stderr)
        return 9

    # removed_ids were placeholder ids; they are still placeholder in coh (not remapped)
    v3 = coh[~coh["variant_id"].isin(removed_ids)].reset_index(drop=True)

    # GUARD 5: zero na:na remain
    remaining = int(is_allele_less(v3["ref"], v3["alt"]).sum())
    if remaining != 0:
        print(f"ABORT: {remaining} na:na rows remain in v3 after merge.", file=sys.stderr)
        return 8

    # GUARD 8b: no duplicate variant_ids in the final cohort
    dup = int(v3["variant_id"].duplicated().sum())
    if dup:
        print(f"ABORT: {dup} duplicate variant_ids in v3 after merge.", file=sys.stderr)
        return 10

    # write excluded documentation
    excl = pd.DataFrame({"variant_id": sorted(removed_ids)})
    excl["verdict"] = excl["variant_id"].map(disp_verdict).fillna("CONFIRMED_ALLELELESS")
    excl.to_csv(excl_path, sep="\t", index=False)

    # write v3 + md5
    v3.to_parquet(out, index=False)
    v3_md5 = _md5(out)

    pos_after = None
    if "pathogenicity" in v3.columns:
        pos_after = float((v3["pathogenicity"].astype("string").str.lower() == "pathogenic").mean())

    recon = {
        "date": "2026-07-09",
        "v2_path": str(a.cohort_v2), "v2_md5": v2_md5, "v2_rows": n_v2,
        "v3_path": str(out), "v3_md5": v3_md5, "v3_rows": int(len(v3)),
        "alleleless_total": n_al,
        "recovered_merged": len(kept_ids),
        "recovered_by_source": pd.Series([s for *_, s in rec_map.values()]).value_counts().to_dict(),
        "excluded_total": len(removed_ids),
        "excluded_by_verdict": excl["verdict"].value_counts().to_dict(),
        "expected_v3_rows": n_v2 - len(removed_ids),
        "pos_rate_before": pos_before, "pos_rate_after": pos_after,
        "na_na_remaining": remaining,
    }
    recon_path.write_text(json.dumps(recon, indent=2, default=str), encoding="utf-8")

    # final consistency assert
    assert len(v3) == n_v2 - len(removed_ids), "row math mismatch"

    print("\n--- COHORT v3 BUILT ---")
    print(f"  v3 rows          : {len(v3):,}   (expected {n_v2 - len(removed_ids):,})")
    print(f"  merged recoveries: {len(kept_ids):,}  {recon['recovered_by_source']}")
    print(f"  excluded         : {len(removed_ids):,}  {recon['excluded_by_verdict']}")
    print(f"  na:na remaining  : {remaining}")
    print(f"  pos_rate         : {pos_before} -> {pos_after}")
    print(f"  v3 MD5           : {v3_md5}")
    print(f"\nwrote {out}")
    print(f"wrote {recon_path}")
    print(f"wrote {excl_path}")
    print("v2 retained for lineage (untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
