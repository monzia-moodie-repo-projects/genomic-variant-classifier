#!/usr/bin/env python3
"""regen_splits_local.py -- Author: Monzia Moodie

Prep-ONLY local pre-flight for Run 17. Runs the SAME DataPrepPipeline.run() that
run_phase2_eval.py invokes at launch (identical AnnotationConfig wiring), writing
the gene-aware train/val/test splits to --output/splits, then STOPS -- no model
training, no GNN stage. Purpose: cheaply confirm that prep.run() under the current
code + current data revives the stale feature families (the "must-revive" columns)
BEFORE committing GPU hours to the full run.

WHY THE STUB (read this): DataPrepPipeline._annotate_scores runs a CPU-prohibitive deep
pipeline UNCONDITIONALLY -- step 16 ESM2Connector (loads facebook/esm2_* and runs a transformer
forward pass over every missense variant). On a GPU box it is fast; on a CPU laptop it is a
~31-hour grind with NO progress output -- it looks frozen. It populates esm2_delta_norm and
esm2_llr, and when those columns are absent the feature builder sets both to 0.0 (measured
2026-09-23, variant_ensemble.engineer_features) -- so stubbing it changes nothing in those
columns, and removes the hang.

Step 14 (protein structure) is QUARANTINED since 2026-09-22 (quarantine_policy.py): real_data_prep
no longer calls it and its producer refuses, so its four columns are no longer produced at all. It
is deliberately NOT stubbed -- a no-op stand-in would mask that refusal if the call ever returned.

By default this driver STUBS step 16 (no model load, no network) so the local prep is tractable.
The RNA pipeline (step 13) stays ON -- it populates four must-revive columns (maxentscan_score,
dist_to_splice_site, exon_number, is_canonical_splice) and is lightweight. Pass --run-protein-esm2
ONLY on a GPU box to run the real ESM-2 forward pass (the flag keeps its historical name).

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
import logging
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
    p.add_argument("--omim-path", default=None,
                   help="OMIM mim2gene file; when omitted, omim_* default to 0.")
    p.add_argument("--omim-genemap2-path", default=None,
                   help="OMIM genemap2.txt; REQUIRED for omim_n_diseases/"
                        "omim_n_diseases_molecular/omim_is_autosomal_dominant "
                        "(when omitted, all three default to 0).")
    p.add_argument("--phylop-path", default=None,
                   help="PhyloP conservation source (.bw/.parquet); when omitted, phylop_score=0.0.")
    p.add_argument("--dbsnp-path", default=None,
                   help="dbSNP allele-frequency parquet; when omitted, dbsnp_af=0.0.")

    p.add_argument("--reactome-path", default=None)
    p.add_argument("--rnaseq-path", default=None)
    p.add_argument("--kg", default=None)
    p.add_argument("--finngen-path", default=None)
    p.add_argument("--lovd-path", default=None)
    p.add_argument("--esm2-uniprot-index", default=None)
    p.add_argument("--eve-path", default=None)
    p.add_argument("--eve-entry-map", default=None)
    p.add_argument("--min-review-tier", type=int, default=3)
    p.add_argument(
        "--no-scale-features",
        action="store_true",
        help=(
            "Write the feature matrices WITHOUT standardisation. "
            "DataPrepConfig.scale_features defaults to True and this driver "
            "did not expose it, so every prep-only run so far has written "
            "STANDARDIZED matrices: a column named gnomad_af holds z-scores "
            "fitted on the TRAINING partition, not allele frequencies. "
            "MEASURED in real_data_prep.py: _scale runs at line 502 and "
            "_save_splits at 504, so the saved splits are post-scaling, and "
            "NEITHER the original matrices NOR the fitted scaler is "
            "persisted by that path. Pass this flag to obtain a parent whose "
            "values carry their source meaning, and fit learned preprocessing "
            "inside each experiment's own fitting boundary instead."
        ),
    )
    p.add_argument("--output", default="outputs/run17_prepcheck/full")
    p.add_argument("--run-protein-esm2", action="store_true",
                   help="GPU-ONLY: run the real ESM-2 forward pass (CPU-prohibitive ~31h). Protein structure "
                        "is quarantined and never runs. Default: stub ESM-2 (safe; see module docstring).")
    return p.parse_args(argv)


def _install_cpu_stubs() -> None:
    """Replace the CPU-prohibitive ESM-2 pipeline with a no-op BEFORE prep.run().

    Protein structure (step 14) is QUARANTINED (quarantine_policy.py): real_data_prep no longer calls it and
    its producer refuses. It is deliberately NOT replaced with a no-op, which would mask that refusal.

    _annotate_scores imports the class at CALL time, so swapping the module attribute here
    takes effect. The no-op __init__ means the ESM-2 model is never loaded; the feature
    builder then sets esm2_delta_norm and esm2_llr to 0.0 (measured 2026-09-23)."""
    import genomic_variant_classifier.data.esm2 as _esm2_mod

    class _NoOpESM2:
        def __init__(self, *a, **k): pass
        def annotate_dataframe(self, df): return df
        def annotate_llr(self, df): return df

    _esm2_mod.ESM2Connector = _NoOpESM2
    print("[regen] STUBBED ESM-2 (step 16): no model load, no network. Protein structure (step 14) is "
          "QUARANTINED and never runs; it is not stubbed, because a no-op stand-in would mask its refusal.")


def main(argv=None) -> int:
    # Stream the library's per-step coverage logs (every "Score annotation N/17"
    # line, ProteinCoord/EVE coverage, etc.) to stderr. DataPrepPipeline emits these
    # via logging.getLogger(__name__) at INFO; without a basicConfig here Python's
    # last-resort handler drops INFO (keeping only WARNING+), so prep coverage was
    # invisible and had to be reverse-engineered from parquet columns. The script
    # (not the library) owns logging config, per the 'logging out of library
    # modules' convention. Honour an existing root config if one is already set.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
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
        print("[regen] --run-protein-esm2 set: running the REAL ESM-2 forward pass (GPU strongly "
              "recommended; CPU ~31h). Protein structure is quarantined and never runs.")

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
        # Run 17 wiring parity with run_phase2_eval (omim/phylop/dbsnp were missing)
        omim_path=Path(args.omim_path) if args.omim_path else None,
        omim_genemap2_path=Path(args.omim_genemap2_path) if args.omim_genemap2_path else None,
        phylop_path=Path(args.phylop_path) if args.phylop_path else None,
        dbsnp_path=Path(args.dbsnp_path) if args.dbsnp_path else None,
        eve_path=Path(args.eve_path) if args.eve_path else None,
        eve_entry_map_path=Path(args.eve_entry_map) if args.eve_entry_map else None,
    )
    prep = DataPrepPipeline(
        config=DataPrepConfig(
            min_review_tier=args.min_review_tier,
            output_dir=outdir / "splits",
            scale_features=not args.no_scale_features,
        ),
        annotation_config=ann,
    )

    print(f"[regen] prep-only run -> splits at {(outdir / 'splits').resolve()}")
    print(f"[regen] clinvar={clinvar}  min_review_tier={args.min_review_tier}")
    # RECORD THE EFFECTIVE VALUE, not the flag. A reader of this transcript
    # must be able to tell whether the saved matrices are standardized without
    # reconstructing the argument parsing.
    print(f"[regen] scale_features={not args.no_scale_features}  "
          f"(saved matrices are "
          f"{'STANDARDIZED' if not args.no_scale_features else 'SEMANTIC -- unscaled'})")
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
