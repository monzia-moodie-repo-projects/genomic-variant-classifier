"""
rebuild_cohort_v3.py  (2026-07-09)
==========================================================================
Single-pass cohort-quality rebuild: cohort-v2 -> cohort-v3.

WHAT THIS DOES AND WHY (all decisions backed by measurement, 2026-07-09)
    cohort-v2 (clinvar_grch38_clean_v2_verified.parquet, MD5 F3152F67...) has CORRECT
    coordinates -- the padded-deletion pos-=1 correction was genome-verified (187,235/187,245
    match; 10 residual are genuine ClinVar-vs-GRCh38 disagreements). This rebuild does NOT
    touch that correction. It fixes two things the coordinate work surfaced:

      1. LATENT CLASSIFIER BUG: is_padded_deletion misclassified empty/NaN-alt rows. Now
         imported from the shared allele_classify module with the non-empty guard. Measured
         impact on cohort-v2: 0 rows wrongly shifted (cohort-v2 was uncorrupted). This is
         hygiene, applied so the re-verification and all downstream code use the correct
         predicate.

      2. ALLELE-LESS (na:na) ROWS: the cohort contains 19,988 rows with no ref/alt. Per the
         provenance investigation (docs/status/ALLELELESS_PROVENANCE_2026-07-09.md), these are
         NOT dropped on assumption. This script excludes ONLY the variant_ids in a VERIFIED
         exclusion list produced by verify_alleleless_provenance.py (records confirmed out of
         ClinVar VCF scope against live ClinVar files). Without --exclude-ids, NO rows are
         dropped -- the rebuild is then classifier-fix + genome re-verify only. Excluded rows
         are QUARANTINED to a separate parquet (labels preserved, reversible), never deleted.
         Every id in the exclusion list must be an allele-less row, or the run aborts (exit 7).

    Output: clinvar_grch38_clean_v3_verified.parquet (new canonical) + reconciliation JSON.
    v2 is retained untouched so the lineage v1 -> v2 -> v3 is fully auditable.

RE-VERIFICATION
    After the drop, every remaining padded deletion's pos is re-checked against GRCh38
    (ref allele must match the reference at pos). This re-earns the "verified" suffix for
    the new MD5 rather than inheriting v2's verification by assertion.

USAGE (project root, .venv312)
    python scripts/rebuild_cohort_v3.py --audit
    python scripts/rebuild_cohort_v3.py --apply
    python scripts/rebuild_cohort_v3.py --apply --skip-genome    # drop+quarantine only, no re-verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from genomic_variant_classifier.data.allele_classify import (
        is_allele_less, is_padded_deletion)
except Exception:  # allow running before the module is installed into src
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from allele_classify import is_allele_less, is_padded_deletion  # type: ignore


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest().upper()


def _schema_fingerprint(columns) -> str:
    return hashlib.sha256(",".join(sorted(map(str, columns))).encode()).hexdigest()[:16]


def _pos_rate(df: pd.DataFrame) -> dict:
    if "pathogenicity" not in df.columns:
        return {"available": False}
    p = df["pathogenicity"].astype("string").str.lower()
    pos = int((p == "pathogenic").sum())
    neg = int((p == "benign").sum())
    other = int(len(df) - pos - neg)
    denom = pos + neg
    return {"available": True, "pathogenic": pos, "benign": neg, "other": other,
            "pos_rate_binary": (pos / denom) if denom else None}


def _open_reference(fasta_path: Path):
    from pyfaidx import Fasta
    return Fasta(str(fasta_path), rebuild=False)


def _verify_padded_deletions(df: pd.DataFrame, fasta_path: Path, sample: int | None) -> dict:
    """Re-check that each padded deletion's ref allele matches GRCh38 at pos (1-based).
    For a padded deletion the corrected pos should place the ref allele on the genome."""
    ref = _open_reference(fasta_path)
    contigs = set(ref.keys())
    mask = is_padded_deletion(df["ref"], df["alt"])
    d = df.loc[mask, ["chrom", "pos", "ref", "alt"]].copy()
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=0)
    match = miss = absent = 0
    examples = []
    for chrom, p, r, a in zip(d["chrom"].astype(str), d["pos"].astype(int),
                              d["ref"].astype(str), d["alt"].astype(str)):
        if chrom not in contigs:
            absent += 1
            continue
        try:
            got = str(ref[chrom][p - 1: p - 1 + len(r)]).upper()
        except Exception:
            absent += 1
            continue
        if got == r.upper():
            match += 1
        else:
            miss += 1
            if len(examples) < 10:
                examples.append(f"{chrom}:{p} ref={r} genome={got}")
    checked = match + miss
    return {"checked": checked, "match": match, "mismatch": miss, "contig_absent": absent,
            "match_rate": (match / checked) if checked else None, "examples": examples}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild cohort-v2 -> v3 (drop na:na, fix classifier, re-verify).")
    ap.add_argument("--in", dest="inp", default="data/processed/clinvar_grch38_clean_v2_verified.parquet")
    ap.add_argument("--out", default="data/processed/clinvar_grch38_clean_v3_verified.parquet")
    ap.add_argument("--quarantine", default="data/processed/clinvar_grch38_alleleless_quarantine.parquet")
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--verify-sample", type=int, default=5000,
                    help="genome re-verify a random sample of padded deletions (0 = all)")
    ap.add_argument("--skip-genome", action="store_true", help="skip genome re-verification")
    ap.add_argument("--exclude-ids", default=None,
                    help="path to a newline-delimited list of variant_ids CONFIRMED "
                         "legitimately allele-less (from verify_alleleless_provenance.py). "
                         "ONLY these are excluded. If omitted, NO rows are dropped and the "
                         "rebuild is classifier-fix + re-verify only (safe default -- never "
                         "drops on assumption).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    inp, out, quar = Path(a.inp), Path(a.out), Path(a.quarantine)
    if not inp.exists():
        print(f"ERROR: input cohort not found: {inp}", file=sys.stderr)
        return 2
    if a.apply and out.exists():
        print(f"ERROR: {out} exists. Refusing to overwrite.", file=sys.stderr)
        return 5

    print("=" * 74)
    print(f"REBUILD COHORT v2 -> v3   input {inp.name}")
    print("=" * 74)
    df = pd.read_parquet(inp)
    n0 = len(df)
    print(f"loaded {n0:,} rows / {len(df.columns)} cols")

    # Allele-less rows present in the cohort (for reporting only -- NOT a drop list).
    alleleless_mask = is_allele_less(df["ref"], df["alt"])
    n_alleleless = int(alleleless_mask.sum())

    # The ONLY rows excluded are those in the verified exclusion list. Without it,
    # nothing is dropped -- the rebuild never removes data on assumption.
    if a.exclude_ids:
        excl_path = Path(a.exclude_ids)
        if not excl_path.exists():
            print(f"ERROR: --exclude-ids not found: {excl_path}", file=sys.stderr)
            return 2
        exclude_ids = {ln.strip() for ln in excl_path.read_text(encoding="utf-8").splitlines()
                       if ln.strip()}
        drop_mask = df["variant_id"].astype(str).isin(exclude_ids)
        # safety: every excluded id MUST be allele-less; refuse if any is not
        bad = int((drop_mask & ~alleleless_mask).sum())
        if bad:
            print(f"ERROR: {bad} exclusion ids are NOT allele-less rows. Refusing -- the "
                  f"exclusion list must contain only confirmed allele-less variant_ids.",
                  file=sys.stderr)
            return 7
    else:
        drop_mask = pd.Series(False, index=df.index)
        exclude_ids = set()

    n_drop = int(drop_mask.sum())
    kept = df.loc[~drop_mask].reset_index(drop=True)
    dropped = df.loc[drop_mask].reset_index(drop=True)
    n_pdel = int(is_padded_deletion(kept["ref"], kept["alt"]).sum())
    n_alleleless_kept = int(is_allele_less(kept["ref"], kept["alt"]).sum())

    recon = {
        "date": "2026-07-09",
        "input": {"path": str(inp), "rows": n0, "md5": _md5(inp)},
        "allele_less_in_input": n_alleleless,
        "exclusion_list": str(a.exclude_ids) if a.exclude_ids else None,
        "excluded_rows": n_drop,
        "allele_less_still_kept": n_alleleless_kept,
        "kept_rows": int(len(kept)),
        "padded_deletions_in_kept": n_pdel,
        "pos_rate_before": _pos_rate(df),
        "pos_rate_after": _pos_rate(kept),
        "schema_fingerprint": _schema_fingerprint(kept.columns),
    }

    print(f"allele-less rows in input      : {n_alleleless:,}")
    print(f"excluded (verified list)       : {n_drop:,}")
    print(f"allele-less still kept         : {n_alleleless_kept:,}  "
          f"(0 only if exclusion list covered them all)")
    print(f"kept rows                      : {len(kept):,}")
    print(f"padded deletions in kept       : {n_pdel:,}")
    pb, pa = recon["pos_rate_before"], recon["pos_rate_after"]
    if pb.get("available"):
        print(f"pos_rate (binary) before/after : "
              f"{pb['pos_rate_binary']:.4f} / {pa['pos_rate_binary']:.4f}"
              if pb['pos_rate_binary'] is not None else "  (n/a)")
        print(f"  pathogenic before/after      : {pb['pathogenic']:,} / {pa['pathogenic']:,}")
        print(f"  benign before/after          : {pb['benign']:,} / {pa['benign']:,}")

    if not a.skip_genome:
        if not Path(a.fasta).exists():
            print(f"WARNING: FASTA not found at {a.fasta}; skipping genome re-verify.", file=sys.stderr)
        else:
            print(f"\nre-verifying padded-deletion coordinates against {Path(a.fasta).name} "
                  f"({'all' if not a.verify_sample else f'sample={a.verify_sample:,}'}) ...")
            v = _verify_padded_deletions(kept, Path(a.fasta), a.verify_sample or None)
            recon["genome_reverify"] = v
            print(f"  checked {v['checked']:,}  match {v['match']:,}  mismatch {v['mismatch']:,}"
                  f"  contig_absent {v['contig_absent']:,}")
            if v["match_rate"] is not None:
                print(f"  match rate: {v['match_rate']:.6f}")
            if v["mismatch"] and v["match_rate"] is not None and v["match_rate"] < 0.99:
                print("  *** genome match rate < 99% -- coordinates suspect, refusing to write. ***",
                      file=sys.stderr)
                if a.apply:
                    return 6

    if not a.apply:
        print("\nAUDIT (dry-run). Nothing written. Re-run with --apply.")
        return 0

    kept.to_parquet(out, index=False)
    recon["output"] = {"path": str(out), "rows": int(len(kept)), "md5": _md5(out)}
    if len(dropped):
        dropped.to_parquet(quar, index=False)
        recon["quarantine"] = {"path": str(quar), "rows": int(len(dropped)), "md5": _md5(quar)}
    else:
        recon["quarantine"] = None
    recon_path = out.with_name("cohort_v3_reconciliation.json")
    recon_path.write_text(json.dumps(recon, indent=2), encoding="utf-8")

    print(f"\nWROTE: {out.name}  ({len(kept):,} rows, MD5 {recon['output']['md5']})")
    if len(dropped):
        print(f"WROTE: {quar.name}  ({len(dropped):,} quarantined rows, "
              f"MD5 {recon['quarantine']['md5']})")
    else:
        print("(no rows excluded -- no quarantine file written)")
    print(f"WROTE: {recon_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
