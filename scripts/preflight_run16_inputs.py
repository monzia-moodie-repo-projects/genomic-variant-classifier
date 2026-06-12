#!/usr/bin/env python3
"""preflight_run16_inputs.py -- Run-16 INPUT preflight: fail loudly BEFORE a regen
if the cohort or mandatory annotation files would silently yield inert features.

Exit: 0 all-pass / 2 any-fail / 3 usage-or-env (matches run_schema_drift_check.py).

Checks:
  1. --clinvar cohort exists AND carries POPULATED fasta_seq_ref/fasta_seq_alt.
     Without these the 1D-CNN and maxentscan_delta degrade SILENTLY to inert
     (CNN dropped to placeholders; maxentscan_delta all-zero). real_data_prep
     never adds fasta_seq*; the columns ride only from this input cohort.
  2. --esm2-uniprot-index exists (else ESM-2 falls back to slow live REST per gene).
  3. --alphamissense exists (else ProteinCoordConnector gets no source -> protein_pos
     empty -> esm2 deadzone).
  4. EXPECTED_TABULAR_FEATURE_COUNT == 81 (esm2_llr + maxentscan_delta registered).

This is the DATA/CONFIG half of the run gate; SSH/instance/key launch vars are a
separate operational layer. Author: Monzia Moodie."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

POLY101 = "A" * 101
EXPECTED_COUNT = 81


def check_exists(label, path, required=True):
    if path is None:
        return (False if required else None), f"{label}: {'FAIL (not supplied)' if required else 'SKIP (not supplied)'}"
    ok = Path(path).exists()
    return ok, f"{label}: {'PASS' if ok else 'FAIL'} ({path})"


def check_cohort_ref_alt(clinvar_path, sample_rows=4000, min_real_frac=0.5):
    p = Path(clinvar_path)
    if not p.exists():
        return False, f"clinvar cohort: FAIL (not found: {clinvar_path})"
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, "clinvar cohort: ENV (pyarrow not importable)"
    pf = pq.ParquetFile(p)
    cols = set(pf.schema.names)
    missing = {"fasta_seq_ref", "fasta_seq_alt"} - cols
    if missing:
        return False, (f"clinvar cohort: FAIL (missing {sorted(missing)} -- CNN + "
                       f"maxentscan_delta would be INERT; use clinvar_grch38_clean_seq.parquet)")
    batch = next(pf.iter_batches(batch_size=sample_rows, columns=["fasta_seq_ref", "fasta_seq_alt"]))
    ref = batch.column("fasta_seq_ref").to_pylist()
    n = len(ref)
    real = sum(1 for s in ref if s and str(s) != POLY101 and len(str(s)) >= 50)
    frac = real / max(n, 1)
    ok = frac >= min_real_frac
    return ok, (f"clinvar cohort: {'PASS' if ok else 'FAIL'} (ref/alt present; "
                f"{real}/{n} = {frac:.1%} real windows in sample, need >= {min_real_frac:.0%})")


def check_feature_count():
    try:
        from genomic_variant_classifier.models.variant_ensemble import (
            EXPECTED_TABULAR_FEATURE_COUNT as C,
        )
    except Exception as e:
        return None, f"feature count: ENV (cannot import variant_ensemble: {e})"
    ok = C == EXPECTED_COUNT
    return ok, f"feature count: {'PASS' if ok else 'FAIL'} (EXPECTED_TABULAR_FEATURE_COUNT={C}, want {EXPECTED_COUNT})"


def check_gnomad_constraint(path, min_mb=1.0):
    if path is None:
        return False, "gnomad constraint: FAIL (not supplied -- gene_constraint_oe would deadzone via stub mode)"
    p = Path(path)
    if not p.exists():
        return False, f"gnomad constraint: FAIL (not found: {path} -- stub mode -> gene_constraint_oe deadzones)"
    mb = p.stat().st_size / 1e6
    ok = mb >= min_mb
    return ok, (f"gnomad constraint: {'PASS' if ok else 'FAIL'} "
                f"({path}, {mb:.1f} MB, need >= {min_mb} MB)")


def aggregate(results):
    any_fail = any(ok is False for ok, _ in results)
    any_env = any(ok is None for ok, _ in results)
    if any_fail:
        return 2
    if any_env:
        return 3
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run-16 input preflight (data/config gate).")
    ap.add_argument("--clinvar", default="data/processed/clinvar_grch38_clean_seq.parquet")
    ap.add_argument("--esm2-uniprot-index", default="data/external/uniprot/uniprot_human_reviewed.parquet")
    ap.add_argument("--alphamissense", default=None, help="AlphaMissense scores parquet (required)")
    ap.add_argument("--gnomad-constraint",
                    default="data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv",
                    help="gnomAD v4.1 constraint TSV (revives gene_constraint_oe via loeuf)")
    args = ap.parse_args(argv)

    results = [
        check_cohort_ref_alt(args.clinvar),
        check_exists("esm2 uniprot index", args.esm2_uniprot_index),
        check_exists("alphamissense", args.alphamissense),
        check_gnomad_constraint(args.gnomad_constraint),
        check_feature_count(),
    ]
    print("=== Run-16 input preflight ===")
    for ok, msg in results:
        print("  " + msg)
    code = aggregate(results)
    print({0: "RESULT: PASS -- input cohort + mandatory files OK.",
           2: "RESULT: FAIL -- do NOT launch Run 16 until every check passes.",
           3: "RESULT: ENV/USAGE -- could not run all checks."}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
