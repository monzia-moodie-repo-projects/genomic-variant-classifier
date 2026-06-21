#!/usr/bin/env python3
"""regen_splits_local.py -- Author: Monzia Moodie

Prep-ONLY local pre-flight for Run 17. Runs the SAME DataPrepPipeline.run() that
run_phase2_eval.py invokes at launch (identical AnnotationConfig wiring), writing
the gene-aware train/val/test splits to --output/splits, then STOPS -- no model
training, no GNN stage. Purpose: cheaply confirm that prep.run() under the current
code + current data revives the stale feature families (the "must-revive" columns)
BEFORE committing GPU hours to the full run.

WHY THE STUB (read this): DataPrepPipeline._annotate_scores runs TWO CPU-prohibitive
deep pipelines UNCONDITIONALLY -- step 14 ProteinStructurePipeline (AlphaFold REST /
structure) and step 16 ESM2Connector (loads facebook/esm2_* and runs a transformer
forward pass over every missense variant). On a GPU box these are fast; on a CPU
laptop ESM-2 alone is a ~31-hour grind with NO progress output -- it looks frozen.
These two steps ONLY populate expected-zero columns (alphafold_plddt,
dist_to_active_site, solvent_accessibility, secondary_structure_context,
esm2_delta_norm, esm2_llr), and both are stubbed today pending the AlphaFold cache /
HGVSp parser. The feature builder fills those columns with the SAME constant defaults
when they are absent (alphafold_plddt=50.0, solvent_accessibility=0.5,
dist_to_active_site=100.0, secondary_structure_context=0, esm2_*=0.0). So skipping
them is provably equivalent for these columns -- and removes the hang.

By default this driver STUBS steps 14+16 (no model load, no network) so the local
prep is tractable. The RNA pipeline (step 13) stays ON -- it populates four
must-revive columns (maxentscan_score, dist_to_splice_site, exon_number,
is_canonical_splice) and is lightweight. Pass --run-protein-esm2 ONLY on a GPU box
to run the real protein/ESM-2 forward passes.

Validate the result with:
    python scripts/split_health_gate.py --splits-dir <out>/splits --prep-only

COST (with the stub, CPU laptop): the tabular + RNA annotation over ~1.49M variants
is still RAM-heavy on the AlphaMissense/dbNSFP joins -- budget roughly 20-60 minutes,
not seconds. This is a deliberate, accepted cost; far below the full ~10-20h training
run, and it is the cheap gate before the GPU launch.

Run from the repo root with .venv312 active. Pass only sources you actually have on
disk; a missing path makes that connector return defaults (logged loudly).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="prep-only local split regeneration")
    p.add_argument("--clinvar", required=True)
    p.add_argument("--gnomad", default=None)
    p.add_argument("--spliceai", default=None)
    p.add_argument("--alphamissense", default=None)
    p.add_argument("--gnomad-constraint", default=None)
    p.add_argument("--dbnsfp-path", default=None)
    p.add_argument("--gtex-path", default=None)
    p.add_argument("--gtex-genes", nargs="*", default=[])
    p.add_argument(
        "--clingen-path",
        default=None,
        help="ClinGen Gene-Disease Validity CSV; when omitted, clingen_validity_score defaults to 0.",
    )

    p.add_argument("--reactome-path", default=None)
    p.add_argument("--rnaseq-path", default=None)
    p.add_argument("--kg", default=None)
    p.add_argument("--finngen-path", default=None)
    p.add_argument("--lovd-path", default=None)
    p.add_argument("--esm2-uniprot-index", default=None)
    p.add_argument("--min-review-tier", type=int, default=3)
    p.add_argument("--output", default="outputs/run17_prepcheck/full")
    p.add_argument("--run-protein-esm2", action="store_true",
                   help="GPU-ONLY: run the real ProteinStructure + ESM-2 forward passes "
                        "(CPU-prohibitive ~31h). Default: stub them (safe; see module docstring).")
    return p.parse_args(argv)


def _install_cpu_stubs() -> None:
    """Replace the two CPU-prohibitive pipelines with no-ops BEFORE prep.run().

    _annotate_scores imports both classes at CALL time, so swapping the module
    attribute here takes effect. The no-op __init__ means the ESM-2 model is never
    loaded and the AlphaFold REST path is never hit. The feature builder then fills
    the affected columns with its constant defaults (verified)."""
    import genomic_variant_classifier.data.esm2 as _esm2_mod
    import genomic_variant_classifier.pipelines.protein_pipeline as _pp_mod

    class _NoOpESM2:
        def __init__(self, *a, **k): pass
        def annotate_dataframe(self, df): return df
        def annotate_llr(self, df): return df

    class _NoOpProtein:
        def __init__(self, *a, **k): pass
        def annotate_dataframe(self, df): return df

    _esm2_mod.ESM2Connector = _NoOpESM2
    _pp_mod.ProteinStructurePipeline = _NoOpProtein
    print("[regen] STUBBED ESM-2 (step 16) + protein-structure (step 14): no model load, "
          "no network. These populate only expected-zero columns; the feature builder "
          "fills constant defaults (alphafold_plddt=50.0, solvent_accessibility=0.5, "
          "dist_to_active_site=100.0, secondary_structure_context=0, esm2_*=0.0).")


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    clinvar = Path(args.clinvar)
    if not clinvar.exists():
        print(f"ClinVar parquet not found: {clinvar.resolve()} -- STOP.")
        return 2

    outdir = Path(args.output)
    (outdir / "splits").mkdir(parents=True, exist_ok=True)

    if not args.run_protein_esm2:
        _install_cpu_stubs()
    else:
        print("[regen] --run-protein-esm2 set: running REAL protein/ESM-2 forward passes "
              "(GPU strongly recommended; CPU ~31h).")

    from genomic_variant_classifier.data.real_data_prep import (
        AnnotationConfig, DataPrepConfig, DataPrepPipeline,
    )

    _esm2_index = None
    if args.esm2_uniprot_index:
        _esm2_index = Path(args.esm2_uniprot_index)
        if not _esm2_index.exists():
            print(f"ESM-2 UniProt index not found: {_esm2_index} -- STOP "
                  "(omit --esm2-uniprot-index to leave ESM-2 stubbed).")
            return 2

    # EXACT mirror of run_phase2_eval.main()'s AnnotationConfig wiring (the wired subset).
    ann = AnnotationConfig(
        spliceai_path=Path(args.spliceai) if args.spliceai else None,
        esm2_uniprot_index_path=_esm2_index,
        alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
        gtex_genes=args.gtex_genes or [],
        gtex_path=Path(args.gtex_path) if args.gtex_path else None,
        kg_path=Path(args.kg) if args.kg else None,
        gnomad_constraint_path=(
            Path(args.gnomad_constraint) if args.gnomad_constraint else None),
        lovd_path=Path(args.lovd_path) if args.lovd_path else None,
        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,
        reactome_path=Path(args.reactome_path) if args.reactome_path else None,
        clingen_path=Path(args.clingen_path) if args.clingen_path else None,
        rnaseq_path=Path(args.rnaseq_path) if args.rnaseq_path else None,
        finngen_path=Path(args.finngen_path) if args.finngen_path else None,
    )
    prep = DataPrepPipeline(
        config=DataPrepConfig(
            min_review_tier=args.min_review_tier,
            output_dir=outdir / "splits",
        ),
        annotation_config=ann,
    )

    print(f"[regen] prep-only run -> splits at {(outdir / 'splits').resolve()}")
    print(f"[regen] clinvar={clinvar}  min_review_tier={args.min_review_tier}")
    for label, val in [("gnomad", args.gnomad), ("spliceai", args.spliceai),
                       ("alphamissense", args.alphamissense), ("dbnsfp", args.dbnsfp_path),
                       ("gtex", args.gtex_path), ("reactome", args.reactome_path),
                       ("rnaseq", args.rnaseq_path), ("kg", args.kg),
                       ("finngen", args.finngen_path), ("lovd", args.lovd_path)]:
        if val and not Path(val).exists():
            print(f"[regen] WARNING: --{label} path does not exist ({val}) -- that "
                  f"connector will return DEFAULTS (its column(s) will be degenerate).")

    X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test = prep.run(
        clinvar_path=str(clinvar), gnomad_path=args.gnomad,
    )
    dt = time.perf_counter() - t0
    print(f"[regen] DONE in {dt/60:.1f} min  "
          f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
          f"features={X_train.shape[1]}")
    print("[regen] NEXT: validate with")
    print(f"  python scripts/split_health_gate.py --splits-dir {outdir / 'splits'} --prep-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
