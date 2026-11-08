"""
scripts/validate_lovd_enriched.py
==================================
Enrich LOVD variants with gnomAD AF + gene features, then run model validation.

Enrichment steps (in priority order):
  1. gnomAD v4 allele frequency join (variant_id = chrom:pos:ref:alt)
  2. Gene-level features from gene_summary parquet
  3. Consequence: the cohort's OWN `consequence` column. Inferring it from allele lengths is refused unless
     --allow-inferred-consequence is given, and then it is recorded in metrics.json as a heuristic input.

Output: outputs/lovd_validation_full/metrics.json  +  predictions.parquet

Package 1, set 1c (2026-11-08): the score is the selected predictor's (predict_proba = the stored convex blend).
Corrected here: `consequence` -- a model input -- was FABRICATED for every row from the allele lengths (every SNV
"missense_variant", every length change "frameshift_variant", in-frame ones included), so the validation measured the
model on inputs the cohort never had; that now requires an explicit flag, uses in-frame classes where the length change
is a multiple of three, and is recorded. The fills the enrichment applies (absent gnomAD frequency -> 0.0, absent
AlphaMissense -> 0.5) are COUNTED in metrics.json; a hard-coded copy into `G:/My Drive` (a cloud mount the project
never bulk-writes through) is removed; the unbound `metadata.val_auroc` is replaced by the predictor's identity and
bound evaluation evidence; nothing is rounded; metrics.json is strict JSON.

Usage:
    python scripts/validate_lovd_enriched.py
    python scripts/validate_lovd_enriched.py --model models/phase4_pipeline_calibrated.joblib
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consequence inference from variant_id pattern
# ---------------------------------------------------------------------------

def _infer_consequence(variant_id: str) -> str:
    """
    HEURISTIC consequence from variant_id 'chrom:pos:ref:alt' -- used ONLY with --allow-inferred-consequence.

    It is a guess, not an annotation: an SNV is assumed missense (it may be synonymous, nonsense, splice or
    non-coding); a length change is frameshift unless the net change is a multiple of three (in-frame). An unknown
    allele ('?') or a malformed identifier is "unknown_consequence" -- never a guessed coding class.
    """
    parts = variant_id.split(":")
    if len(parts) != 4:
        return "unknown_consequence"
    _, _, ref, alt = parts
    if ref == "?" or alt == "?" or not ref or not alt:
        return "unknown_consequence"
    if len(ref) == 1 and len(alt) == 1:
        return "missense_variant"
    change = len(alt) - len(ref)
    if change == 0:
        return "missense_variant"          # a multi-nucleotide substitution
    if change % 3 == 0:
        return "inframe_insertion" if change > 0 else "inframe_deletion"
    return "frameshift_variant"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("data/external/lovd/lovd_all_variants.parquet"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/phase4_pipeline.joblib"),
    )
    parser.add_argument(
        "--gnomad",
        type=Path,
        default=Path("data/processed/gnomad_v4_exomes.parquet"),
    )
    parser.add_argument(
        "--gene-summary",
        type=Path,
        default=Path("data/processed/gene_summary.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lovd_validation_full"),
    )
    parser.add_argument(
        "--allow-inferred-consequence", action="store_true",
        help="when the cohort has no `consequence` column, GUESS it from allele lengths (recorded as heuristic)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from genomic_variant_classifier.api.pipeline import InferencePipeline

    # ---- load cohort ----
    logger.info("Loading LOVD cohort from %s ...", args.cohort)
    df = pd.read_parquet(args.cohort)
    if set(pd.unique(df["label"])) != {0, 1}:
        logger.error("The cohort's `label` must be exactly 0/1 with both classes present; found %s",
                     sorted(map(str, pd.unique(df["label"]))))
        return 2
    if "consequence" not in df.columns and not args.allow_inferred_consequence:
        logger.error("REFUSED: the cohort has no `consequence` column, and `consequence` is a model input. Annotate "
                     "it (for example with VEP), or pass --allow-inferred-consequence to accept a recorded guess "
                     "from allele lengths.")
        return 2
    enrichment: dict[str, dict] = {}
    logger.info(
        "Cohort: %d variants  (%d pathogenic, %d benign)",
        len(df), int((df["label"] == 1).sum()), int((df["label"] == 0).sum()),
    )

    # ---- enrich: gnomAD allele frequency ----
    if args.gnomad.exists():
        logger.info("Loading gnomAD AF from %s ...", args.gnomad)
        gnomad = pd.read_parquet(args.gnomad)[["variant_id", "allele_freq"]]
        gnomad = gnomad.rename(columns={"allele_freq": "gnomad_af"})
        df = df.merge(gnomad, on="variant_id", how="left")
        df["allele_freq"] = df["gnomad_af"].fillna(0.0)
        n_matched = df["gnomad_af"].notna().sum()
        enrichment["allele_freq"] = {"source": str(args.gnomad), "n_matched": int(n_matched),
                                     "n_filled_with_0": int(len(df) - n_matched)}
        logger.info(
            "gnomAD match: %d / %d variants (%.1f%%)",
            n_matched, len(df), 100 * n_matched / len(df),
        )
        df = df.drop(columns=["gnomad_af"])
    else:
        logger.warning("gnomAD parquet not found; allele_freq will default to 0.0")
        df["allele_freq"] = 0.0
        enrichment["allele_freq"] = {"source": None, "n_matched": 0, "n_filled_with_0": int(len(df))}

    # ---- enrich: gene-level features ----
    if args.gene_summary.exists():
        logger.info("Joining gene summary features ...")
        gs = pd.read_parquet(args.gene_summary)
        df = df.merge(gs, on="gene_symbol", how="left")
        logger.info("Gene summary joined.")
    else:
        logger.warning("gene_summary parquet not found.")

    # ---- enrich: AlphaMissense ----
    am_path = Path("data/external/alphamissense/am_lovd_genes.parquet")
    if am_path.exists():
        logger.info("Joining AlphaMissense scores from %s ...", am_path)
        am = pd.read_parquet(am_path)
        df = df.merge(am, on="variant_id", how="left")
        n_am = df["am_pathogenicity"].notna().sum()
        logger.info(
            "AlphaMissense match: %d / %d SNV variants (%.1f%%)",
            n_am, len(df), 100 * n_am / len(df),
        )
        # engineer_features() reads alphamissense_score; fill missing with 0.5 (not covered) -- COUNTED below
        enrichment["alphamissense_score"] = {"source": str(am_path), "n_matched": int(n_am),
                                             "n_filled_with_0.5": int(len(df) - n_am)}
        df["alphamissense_score"] = df["am_pathogenicity"].fillna(0.5)
        df = df.drop(columns=["am_pathogenicity"])
    else:
        logger.warning("AlphaMissense parquet not found; alphamissense_score defaults to 0.5")
        enrichment["alphamissense_score"] = {"source": None, "n_matched": 0,
                                             "note": "absent: engineer_features applies its own default"}

    # ---- consequence: the cohort's own, or an explicitly accepted and recorded guess ----
    if "consequence" in df.columns:
        enrichment["consequence"] = {"source": "cohort column"}
    else:
        df["consequence"] = df["variant_id"].apply(_infer_consequence)
        enrichment["consequence"] = {"source": "HEURISTIC from allele lengths (--allow-inferred-consequence)",
                                     "counts": {str(k): int(v) for k, v in
                                                df["consequence"].value_counts().items()}}
    logger.info(
        "Consequence distribution:\n%s",
        df["consequence"].value_counts().to_string(),
    )

    # ---- load model ----
    logger.info("Loading model from %s ...", args.model)
    from genomic_variant_classifier.model_admission import repository_registry_path
    pipeline = InferencePipeline.load(args.model, consumer="scripts/validate_lovd_enriched.py",
                                      registry_path=repository_registry_path())
    logger.info(
        "Model loaded: combiner %s  features=%d  evaluation evidence %s",
        pipeline.combiner.digest, pipeline.metadata.n_features,
        pipeline.metadata.evaluation_evidence.get("status"),
    )

    # ---- score ----
    logger.info("Scoring %d variants ...", len(df))
    scores_chunks = []
    batch_size = 2048
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        scores_chunks.append(pipeline.predict_proba(chunk))
    y_prob = np.concatenate(scores_chunks)
    y_true = df["label"].astype(int).values

    # ---- overall metrics ----
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    logger.info("Overall: AUROC=%.4f  AUPRC=%.4f  Brier=%.4f", auroc, auprc, brier)

    # ---- per-gene metrics ----
    df["_prob"] = y_prob
    gene_metrics: dict[str, dict] = {}
    for gene, grp in df.groupby("gene_symbol"):
        if grp["label"].nunique() < 2:
            continue
        ga = roc_auc_score(grp["label"].values, grp["_prob"].values)
        gene_metrics[gene] = {
            "auroc": float(ga),
            "n": len(grp),
            "n_pathogenic": int((grp["label"] == 1).sum()),
            "n_benign": int((grp["label"] == 0).sum()),
        }
        logger.info(
            "  %-8s  AUROC=%.4f  n=%d  (%d path / %d benign)",
            gene, ga, len(grp),
            gene_metrics[gene]["n_pathogenic"],
            gene_metrics[gene]["n_benign"],
        )

    # ---- SNV-only subset ----
    snv_mask = (df["ref"].notna()) & (df["ref"] != "?") & (df["alt"].notna()) & (df["alt"] != "?")
    if snv_mask.sum() >= 10 and df.loc[snv_mask, "label"].nunique() == 2:
        snv_auroc = roc_auc_score(
            df.loc[snv_mask, "label"].values,
            df.loc[snv_mask, "_prob"].values,
        )
        logger.info(
            "SNV-only subset (n=%d): AUROC=%.4f", snv_mask.sum(), snv_auroc
        )
    else:
        snv_auroc = None

    # ---- save ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation_type": "lovd_external_enriched",
        "genes": sorted(df["gene_symbol"].unique().tolist()),
        "n_variants": int(len(df)),
        "n_pathogenic": int((y_true == 1).sum()),
        "n_benign": int((y_true == 0).sum()),
        "n_snv": int(snv_mask.sum()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "auroc_snv_only": float(snv_auroc) if snv_auroc is not None else None,
        "per_gene": gene_metrics,
        "model": args.model.name,
        "enrichment": enrichment,
        "predictor": {"combiner_sha256": pipeline.combiner.digest,
                      "predictor_contract_sha256": pipeline.predictor_contract_sha256(),
                      "evaluation_evidence": pipeline.metadata.evaluation_evidence},
    }

    from genomic_variant_classifier.evaluation.serialization import dump_strict_json
    out_metrics = args.output_dir / "metrics.json"
    out_metrics.write_text(dump_strict_json(metrics, artifact="metrics.json") + "\n", encoding="utf-8", newline="\n")
    logger.info("Metrics saved to %s", out_metrics)

    out_preds = args.output_dir / "predictions.parquet"
    df["predicted_proba"] = y_prob
    df.to_parquet(out_preds, index=False)
    logger.info("Predictions saved to %s", out_preds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
