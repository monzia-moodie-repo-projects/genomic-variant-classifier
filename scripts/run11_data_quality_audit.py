"""
Run 11 — Integration 1: Data-Quality Audit Script
====================================================
Systematic audit of all features in the training matrix to identify which
are contributing real signal vs. silently zero.

Usage:
    python scripts/run11_data_quality_audit.py \
        --splits-dir outputs/run11/regen/splits \
        --output-dir outputs/run11

Outputs:
    data_quality_audit.csv    — per-feature statistics
    data_quality_audit.json   — machine-readable summary
    (console)                 — grouped summary by zero-fraction tier

This script uses Polars for fast scanning. Falls back to pandas if Polars
is not installed (slower but functional).

Author: Claude Opus 4.6 for Monzia Moodie
Date: 2026-05-24
Run: 11 (pre-training baseline audit)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_quality_audit")

# ---------------------------------------------------------------------------
# Connector → feature attribution map
# ---------------------------------------------------------------------------
FEATURE_CONNECTOR_MAP = {
    # Allele / variant properties (derived, always present)
    "af_raw": "gnomAD", "af_log10": "gnomAD", "af_bin": "gnomAD",
    "ref_len": "derived", "alt_len": "derived", "len_diff": "derived",
    "is_snv": "derived", "is_insertion": "derived", "is_deletion": "derived",
    "is_indel": "derived",
    # Consequence (derived from VEP annotation)
    "consequence_severity": "VEP", "is_loss_of_function": "VEP",
    "is_missense": "VEP", "is_synonymous": "VEP", "is_splice": "VEP",
    "in_coding": "VEP",
    # Functional scores
    "cadd_phred": "dbNSFP/CADD", "cadd_high": "dbNSFP/CADD",
    "sift_score": "dbNSFP/SIFT", "sift_deleterious": "dbNSFP/SIFT",
    "polyphen2_score": "dbNSFP/PolyPhen", "polyphen_probably_damaging": "dbNSFP/PolyPhen",
    "revel_score": "dbNSFP/REVEL", "revel_pathogenic": "dbNSFP/REVEL",
    "phylop_score": "PhyloP", "gerp_score": "dbNSFP/GERP",
    "n_tools_pathogenic": "derived(composite)",
    # SpliceAI
    "splice_ai_score": "SpliceAI",
    "splice_ai_max_ds": "SpliceAI",
    "splice_ai_ds_ag": "SpliceAI", "splice_ai_ds_al": "SpliceAI",
    "splice_ai_ds_dg": "SpliceAI", "splice_ai_ds_dl": "SpliceAI",
    # AlphaMissense
    "alphamissense_score": "AlphaMissense",
    # Gene-level
    "n_pathogenic_in_gene": "ClinVar(derived)", "gene_has_known_disease": "ClinVar(derived)",
    "gene_constraint_oe": "gnomAD-constraint", "gene_is_constrained": "gnomAD-constraint",
    "loeuf": "gnomAD-constraint", "syn_z": "gnomAD-constraint",
    "mis_z": "gnomAD-constraint", "pli_score": "gnomAD-constraint",
    # Protein features
    "has_uniprot_annotation": "UniProt",
    "n_known_pathogenic_protein_variants": "UniProt",
    # Chromosome features
    "is_autosome": "derived", "is_sex_chrom": "derived", "is_mitochondrial": "derived",
    # LOVD
    "lovd_variant_class": "LOVD",
    # DbNSFP aggregate
    "dbnsfp_rank_score": "dbNSFP",
    # FinnGen
    "finngen_af_fin": "FinnGen", "finngen_af_nfsee": "FinnGen",
    "finngen_enrichment": "FinnGen",
    # ESM-2 (stub until HGVSp parser)
    "esm2_delta_norm": "ESM-2(stub)",
    # EVE (stub until HGVSp parser)
    "eve_score": "EVE(stub)",
    # GTEx
    "gtex_median_tpm": "GTEx",
    # VEP extended
    "codon_position": "VEP(codon)",
    # OMIM
    "omim_has_phenotype": "OMIM",
    # ClinGen
    "clingen_gene_validity": "ClinGen",
    # dbSNP
    "dbsnp_build": "dbSNP",
    # RNA splice features
    "rna_splice_score": "RNA-splice",
    # Protein structure
    "protein_structure_score": "ProteinStructure",
    # GNN score (post-training addition)
    "gnn_score": "GNN",
    # PrimateAI-3D (Run 11 addition)
    "primateai3d_score": "PrimateAI-3D",
}


def identify_connector(feature_name: str) -> str:
    """Map feature name to its source connector."""
    return FEATURE_CONNECTOR_MAP.get(feature_name, "UNKNOWN")


def audit_with_polars(splits_dir: Path) -> list[dict]:
    """Audit using Polars (preferred — fast lazy scanning)."""
    import polars as pl

    logger.info("Using Polars backend for audit")
    dfs = []
    for name in ["X_train", "X_val", "X_test"]:
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            dfs.append(pl.read_parquet(path))
            logger.info("  Loaded %s: %d rows × %d cols", name, dfs[-1].height, dfs[-1].width)
    if not dfs:
        logger.error("No split files found in %s", splits_dir)
        sys.exit(1)

    X_all = pl.concat(dfs)
    logger.info("Combined: %d rows × %d cols", X_all.height, X_all.width)

    results = []
    for col in X_all.columns:
        series = X_all[col]
        n = series.len()
        n_null = series.null_count()
        n_zero = (series == 0).sum()
        n_nonzero = n - n_zero - n_null

        results.append({
            "feature": col,
            "n_total": n,
            "n_zero": int(n_zero),
            "n_null": int(n_null),
            "n_nonzero": int(n_nonzero),
            "zero_fraction": round(float(n_zero) / n, 6) if n > 0 else 0.0,
            "null_fraction": round(float(n_null) / n, 6) if n > 0 else 0.0,
            "nonzero_fraction": round(float(n_nonzero) / n, 6) if n > 0 else 0.0,
            "nunique": int(series.n_unique()),
            "min": float(series.min()) if series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] else None,
            "max": float(series.max()) if series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] else None,
            "mean": float(series.mean()) if series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] else None,
            "std": float(series.std()) if series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] else None,
            "dtype": str(series.dtype),
            "source_connector": identify_connector(col),
        })

    return sorted(results, key=lambda r: r["zero_fraction"], reverse=True)


def audit_with_pandas(splits_dir: Path) -> list[dict]:
    """Fallback audit using pandas (slower but always available)."""
    import pandas as pd

    logger.info("Using pandas backend for audit (Polars not available)")
    dfs = []
    for name in ["X_train", "X_val", "X_test"]:
        path = splits_dir / f"{name}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
            logger.info("  Loaded %s: %d rows × %d cols", name, dfs[-1].shape[0], dfs[-1].shape[1])
    if not dfs:
        logger.error("No split files found in %s", splits_dir)
        sys.exit(1)

    X_all = pd.concat(dfs, ignore_index=True)
    logger.info("Combined: %d rows × %d cols", X_all.shape[0], X_all.shape[1])

    results = []
    for col in X_all.columns:
        series = X_all[col]
        n = len(series)
        n_null = int(series.isnull().sum())
        n_zero = int((series == 0).sum())
        n_nonzero = n - n_zero - n_null

        results.append({
            "feature": col,
            "n_total": n,
            "n_zero": n_zero,
            "n_null": n_null,
            "n_nonzero": n_nonzero,
            "zero_fraction": round(n_zero / n, 6) if n > 0 else 0.0,
            "null_fraction": round(n_null / n, 6) if n > 0 else 0.0,
            "nonzero_fraction": round(n_nonzero / n, 6) if n > 0 else 0.0,
            "nunique": int(series.nunique()),
            "min": float(series.min()) if series.dtype.kind in "iuf" else None,
            "max": float(series.max()) if series.dtype.kind in "iuf" else None,
            "mean": float(series.mean()) if series.dtype.kind in "iuf" else None,
            "std": float(series.std()) if series.dtype.kind in "iuf" else None,
            "dtype": str(series.dtype),
            "source_connector": identify_connector(col),
        })

    return sorted(results, key=lambda r: r["zero_fraction"], reverse=True)


def print_summary(results: list[dict]) -> dict:
    """Print grouped summary and return summary dict."""
    tiers = {
        "DEAD (100% zero)": [],
        "MOSTLY DEAD (>90% zero)": [],
        "WEAK (50-90% zero)": [],
        "MODERATE (10-50% zero)": [],
        "HEALTHY (<10% zero)": [],
    }

    for r in results:
        zf = r["zero_fraction"]
        if zf >= 1.0:
            tiers["DEAD (100% zero)"].append(r)
        elif zf >= 0.9:
            tiers["MOSTLY DEAD (>90% zero)"].append(r)
        elif zf >= 0.5:
            tiers["WEAK (50-90% zero)"].append(r)
        elif zf >= 0.1:
            tiers["MODERATE (10-50% zero)"].append(r)
        else:
            tiers["HEALTHY (<10% zero)"].append(r)

    print("\n" + "=" * 80)
    print("DATA-QUALITY AUDIT SUMMARY")
    print("=" * 80)

    for tier_name, tier_features in tiers.items():
        print(f"\n--- {tier_name}: {len(tier_features)} features ---")
        for r in tier_features:
            print(f"  {r['feature']:40s} zero={r['zero_fraction']:.4f}  "
                  f"nonzero={r['n_nonzero']:>10,}  source={r['source_connector']}")

    summary = {
        "total_features": len(results),
        "dead_features": len(tiers["DEAD (100% zero)"]),
        "mostly_dead_features": len(tiers["MOSTLY DEAD (>90% zero)"]),
        "weak_features": len(tiers["WEAK (50-90% zero)"]),
        "moderate_features": len(tiers["MODERATE (10-50% zero)"]),
        "healthy_features": len(tiers["HEALTHY (<10% zero)"]),
        "dead_connectors": list(set(r["source_connector"] for r in tiers["DEAD (100% zero)"])),
    }

    print(f"\n{'TOTAL':40s} {len(results)} features")
    print(f"{'DEAD (100% zero)':40s} {summary['dead_features']}")
    print(f"{'HEALTHY (<10% zero)':40s} {summary['healthy_features']}")
    print(f"\nDead connectors: {', '.join(summary['dead_connectors'])}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run 11 Data-Quality Audit")
    parser.add_argument("--splits-dir", required=True, help="Path to splits directory")
    parser.add_argument("--output-dir", default="outputs/run11", help="Output directory")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try Polars first, fall back to pandas
    try:
        import polars  # noqa: F401
        results = audit_with_polars(splits_dir)
    except ImportError:
        results = audit_with_pandas(splits_dir)

    # Save CSV
    csv_path = output_dir / "data_quality_audit.csv"
    try:
        import polars as pl
        pl.DataFrame(results).write_csv(csv_path)
    except ImportError:
        import pandas as pd
        pd.DataFrame(results).to_csv(csv_path, index=False)
    logger.info("Audit CSV saved to %s", csv_path)

    # Print and save summary
    summary = print_summary(results)
    summary["results"] = results
    json_path = output_dir / "data_quality_audit.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Audit JSON saved to %s", json_path)


if __name__ == "__main__":
    main()
