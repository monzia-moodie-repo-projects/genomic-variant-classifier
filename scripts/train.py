"""
scripts/train.py
=================
End-to-end training script for the Genomic Variant Classifier.

Takes processed ClinVar parquet and produces:
  1. A trained ensemble saved to <out-dir>/ensemble_v1.joblib
  2. A JSON metrics file at <out-dir>/metrics_v1.json
  3. A README-ready Markdown table at <out-dir>/METRICS.md

Run:
    python scripts/train.py

Or with explicit paths:
    python scripts/train.py \
        --clinvar  data/processed/clinvar_grch38.parquet \
        --gnomad   data/processed/gnomad_v4_exomes.parquet \
        --out-dir  models/v1

With optional annotation sources:
    python scripts/train.py \
        --clinvar        data/processed/clinvar_grch38.parquet \
        --gnomad         data/processed/gnomad_v4_exomes.parquet \
        --alphamissense  data/raw/cache/alphamissense_scores_hg38.parquet \
        --lovd-path      data/external/lovd/lovd_all_variants.parquet \
        --out-dir        models/v1 \
        --skip-svm

The first run takes 5-15 minutes depending on dataset size and CPU count.

CHANGES FROM PHASE 1:
  - Was a bare string literal (Bug 3 fixed -- now a real executable script).
  - Imported VariantEnsemble from genomic_variant_classifier.models.ensemble but that module only
    exports EnsembleClassifier. VariantEnsemble is in variant_ensemble.py
    (Bug 5 fixed).
  - logging.basicConfig is now called only here, before any other imports
    that might emit log messages (Issue L).
  - from __future__ import annotations added (Issue N).

CHANGES -- LOVD + AlphaMissense integration:
  - --alphamissense-path argument added; wired into AnnotationConfig
  - --lovd-path argument added; wired into AnnotationConfig
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -- Logging must be configured before any pipeline imports -----------------
Path("logs").mkdir(exist_ok=True)
def _force_utf8_stdio(streams=None):
    """Make stdout/stderr UTF-8 so Unicode in reports/logs does not crash a
    Windows cp1252 console. No-op where .reconfigure is unavailable."""
    if streams is None:
        streams = (sys.stdout, sys.stderr)
    for _stream in streams:
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            try:
                _reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_stdio()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="w"),
    ],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Genomic Variant Classifier ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--clinvar",
        default="data/processed/clinvar_grch38.parquet",
        help="Processed ClinVar parquet (output of database_connectors.py)",
    )
    p.add_argument(
        "--gnomad",
        default=None,
        help="Optional processed gnomAD parquet for AF enrichment",
    )
    p.add_argument(
        "--uniprot",
        default=None,
        help="Optional processed UniProt parquet for protein features",
    )
    p.add_argument(
        "--alphamissense",
        default=None,
        metavar="PATH",
        help=(
            "AlphaMissense scores parquet (alphamissense_scores_hg38.parquet). "
            "Improves AUROC for missense-heavy genes (PTEN, TP53, MSH2). "
            "Default: None (stub mode, score=0.5 for all variants)."
        ),
    )
    p.add_argument(
        "--lovd-path",
        default=None,
        metavar="PATH",
        help=(
            "LOVD all-variants parquet (data/external/lovd/lovd_all_variants.parquet). "
            "Adds lovd_variant_class feature (ordinal 0-4). "
            "Default: None (stub mode, lovd_variant_class=0 for all variants)."
        ),
    )
    p.add_argument(
        "--finngen-path",
        default=None,
        metavar="PATH",
        help=(
            "FinnGen R10 annotated variants TSV (gzipped). "
            "Adds finngen_af_fin, finngen_af_nfsee, finngen_enrichment features. "
            "Default: None (stub mode, finngen_af_fin=0.0 for all variants)."
        ),
    )
    p.add_argument(
        "--dbnsfp-path",
        default=None,
        metavar="PATH",
        help=(
            "dbNSFP4 TSV file (raw or gzipped, GRCh38 build). "
            "Adds SIFT, PolyPhen-2, REVEL, MutationTaster, FATHMM, CADD scores. "
            "Default: None (stub mode, all scores receive population-median defaults)."
        ),
    )
    p.add_argument(
        "--gnomad-constraint",
        default=None,
        metavar="PATH",
        help=(
            "gnomAD v4.1 gene-constraint TSV (gnomad.v4.1.constraint_metrics.tsv.bgz). "
            "Adds pli_score, loeuf, syn_z, mis_z; loeuf revives gene_constraint_oe "
            "(Run-15 #2 feature). Default: None (stub mode -> constant defaults -> the "
            "feature silently deadzones)."
        ),
    )
    p.add_argument(
        "--skip-svm",
        action="store_true",
        default=False,
        help="Exclude SVM from ensemble (recommended for datasets > 100K samples).",
    )
    p.add_argument(
        "--out-dir",
        default="models/v1",
        help="Output directory for saved models, metrics, and reports",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Cross-validation folds for stacking meta-learner",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="50-estimator fast mode for development iteration",
    )
    p.add_argument(
        "--skip-nn", action="store_true",
        help="Skip neural network models (faster without TensorFlow/GPU)",
    )
    p.add_argument(
        "--esm2-model",
        default="esm2_t6_8M_UR50D",
        metavar="NAME",
        help=(
            "ESM-2 model for esm2_delta_norm / esm2_llr (HuggingFace facebook/<NAME>). "
            "Default esm2_t6_8M_UR50D (fast, for smoke tests). Set "
            "esm2_t33_650M_UR50D for production runs."
        ),
    )
    p.add_argument(
        "--esm2-uniprot-index",
        default=None,
        metavar="PATH",
        help=(
            "Local UniProt sequence index parquet for ESM-2 (e.g. "
            "data/external/uniprot/uniprot_human_reviewed.parquet). When set, "
            "ESM-2 resolves sequences offline with NO run-time UniProt REST. "
            "Default None -> live REST per gene (slow; not for large regens)."
        ),
    )
    p.add_argument(
        "--esm2-cache",
        default=None,
        metavar="PATH",
        help="SQLite cache path for ESM-2 sequences/embeddings. Default None -> connector default.",
    )
    p.add_argument(
        "--esm2-device",
        default=None,
        metavar="DEV",
        help="Device for ESM-2 ('cpu','cuda','auto'). Default None -> cuda if available else cpu.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -- 1. Data preparation ------------------------------------------------
    logger.info("PHASE 1: Data Preparation")

    from genomic_variant_classifier.data.real_data_prep import (
        AnnotationConfig,
        DataPrepConfig,
        DataPrepPipeline,
        BENIGN_TERMS,
        PATHOGENIC_TERMS,
        enrich_gene_counts,
    )

    logger.info("Loading ClinVar from %s", args.clinvar)
    raw_df = pd.read_parquet(args.clinvar)

    # Assign binary labels and compute gene-level pathogenic counts on the
    # full dataset BEFORE splitting, to avoid label leakage in the feature.
    raw_df["label"] = np.nan
    raw_df.loc[raw_df["clinical_sig"].isin(PATHOGENIC_TERMS), "label"] = 1
    raw_df.loc[raw_df["clinical_sig"].isin(BENIGN_TERMS),     "label"] = 0
    raw_df = raw_df[raw_df["label"].notna()].copy()
    raw_df["label"] = raw_df["label"].astype(int)
    raw_df = enrich_gene_counts(raw_df)

    enriched_path = out_dir / "clinvar_enriched.parquet"
    raw_df.to_parquet(enriched_path, index=False)

    config = DataPrepConfig(
        min_review_tier=3,
        test_fraction=0.20,
        random_state=42,
        scale_features=True,
        output_dir=out_dir / "splits",
    )

    annotation_config = AnnotationConfig(
        alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,
        lovd_path=Path(args.lovd_path) if args.lovd_path else None,
        finngen_path=Path(args.finngen_path) if args.finngen_path else None,
        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,
        gnomad_constraint_path=Path(args.gnomad_constraint) if args.gnomad_constraint else None,
        esm2_model_name=args.esm2_model,
        esm2_uniprot_index_path=Path(args.esm2_uniprot_index) if args.esm2_uniprot_index else None,
        esm2_cache_path=Path(args.esm2_cache) if args.esm2_cache else None,
        esm2_device=args.esm2_device,
    )

    pipeline = DataPrepPipeline(config=config, annotation_config=annotation_config)
    X_train, X_val, X_test, y_train, y_val, y_test, meta_val, meta_test = pipeline.run(
        clinvar_path=str(enriched_path),
        gnomad_path=args.gnomad,
        uniprot_path=args.uniprot,
    )

    logger.info(
        "Train: %d variants (%d pathogenic, %.1f%%)",
        len(X_train), int(y_train.sum()), y_train.mean() * 100,
    )
    logger.info(
        "Test:  %d variants (%d pathogenic, %.1f%%)",
        len(X_test), int(y_test.sum()), y_test.mean() * 100,
    )

    class_weights = pipeline.get_class_weights(y_train)
    logger.info("Class weights: %s", class_weights)
    feature_names = list(X_train.columns)
    logger.info("Feature dimensionality: %d", len(feature_names))

    # -- 2. Build and configure ensemble ------------------------------------
    logger.info("PHASE 2: Model Configuration")

    from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble

    ensemble_config = EnsembleConfig(
        n_folds=args.n_folds,
        random_state=42,
        calibrate=True,
        class_weight="balanced",
        n_jobs=-1,
        model_dir=out_dir,
        skip_svm=args.skip_svm,
    )
    ensemble = VariantEnsemble(config=ensemble_config)

    if args.fast:
        _apply_fast_mode(ensemble)
        logger.info("Fast mode: 50 estimators per tree model.")

    if args.skip_nn:
        for key in ("tabular_nn", "cnn_1d", "mc_dropout", "deep_ensemble"):
            ensemble.base_estimators.pop(key, None)
        logger.info("Skipping neural network models (--skip-nn).")

    logger.info("Models to train: %s", list(ensemble.base_estimators.keys()))

    # -- 3. Handle sequence data for CNN ------------------------------------
    # PM11d fix: sequences MUST be realigned to the gene-aware split, not
    # sliced positionally from the pre-split raw_df. meta_test carries the
    # split-aligned fasta_seq directly (df.iloc[test_idx]); the old
    # raw_df["fasta_seq"].iloc[:len(y_test)] paired sequences with the WRONG
    # labels after the shuffling GroupShuffleSplit. See
    # docs/incidents/INCIDENT_2026-05-30_train-sequence-misalignment.md
    # CNN sequence input: the live 2-column [fasta_seq_ref, fasta_seq_alt] delta
    # windows. The legacy single 'fasta_seq' column is deprecated and empty.
    from genomic_variant_classifier.data.seq_window_join import (
        attach_delta_windows,
        REF_WIN_COL,
        ALT_WIN_COL,
    )

    has_sequences = (
        REF_WIN_COL in meta_test.columns
        and ALT_WIN_COL in meta_test.columns
        and meta_test[REF_WIN_COL].notna().sum() > 100
    )
    if not has_sequences:
        logger.info("No usable ref/alt sequence windows -- removing CNN from ensemble.")
        ensemble.base_estimators.pop("cnn_1d", None)
        # CNN is the only sequence consumer; with it removed these placeholders
        # satisfy the seq-aware fit/evaluate/predict signatures but are unused.
        X_seq_train = pd.Series(["A" * 101] * len(y_train))
        X_seq_test  = pd.Series(["A" * 101] * len(y_test))
    else:
        # Test side: meta_test carries ref/alt, structurally split-aligned to X_test.
        X_seq_test, _n_unmapped_test = attach_delta_windows(meta_test)
        # Train side: meta_train is persisted by _save_splits, gene-split-aligned to
        # X_train (both df.iloc[train_idx].reset_index). Read it -- no run() change.
        meta_train_path = config.output_dir / "meta_train.parquet"
        if not meta_train_path.exists():
            raise FileNotFoundError(
                f"meta_train.parquet not found at {meta_train_path}; required for "
                "CNN train-side sequences (DataPrepPipeline._save_splits writes it)."
            )
        _meta_train = pd.read_parquet(meta_train_path)
        if len(_meta_train) != len(y_train):
            raise ValueError(
                f"meta_train rows ({len(_meta_train)}) != y_train ({len(y_train)}); "
                "split misalignment -- aborting to avoid PM11d-style label mismatch."
            )
        X_seq_train, _n_unmapped_train = attach_delta_windows(_meta_train)
        logger.info(
            "CNN sequences active (delta mode): train=%d (unmapped=%d), "
            "test=%d (unmapped=%d).",
            len(X_seq_train), _n_unmapped_train, len(X_seq_test), _n_unmapped_test,
        )

    # -- 4. Train -----------------------------------------------------------
    logger.info("PHASE 3: Training")
    ensemble.fit(X_train, X_seq_train, y_train)

    # -- 5. Evaluate --------------------------------------------------------
    logger.info("PHASE 4: Evaluation")
    metrics_df = ensemble.evaluate(X_test, X_seq_test, y_test)
    logger.info("\n%s", metrics_df.to_string())

    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator
    evaluator = ClinicalEvaluator()
    ensemble_proba = ensemble.predict_proba(X_test, X_seq_test)[:, 1]
    # meta_test drives the per-consequence and per-gene breakdowns; both
    # silently return [] if their required columns are absent, so verify
    # row-alignment and column names explicitly rather than fail quietly.
    assert len(meta_test) == len(y_test) == len(ensemble_proba), (
        f"meta_test ({len(meta_test)}) not aligned to "
        f"y_test ({len(y_test)}) / proba ({len(ensemble_proba)})"
    )
    _missing = [c for c in ("consequence", "gene_symbol")
                if c not in meta_test.columns]
    if _missing:
        logger.warning(
            "meta_test missing %s; those breakdowns will be empty. "
            "Columns present: %s", _missing, list(meta_test.columns),
        )
    eval_report = evaluator.evaluate(
        y_true=y_test,
        y_proba=ensemble_proba,
        meta=meta_test,
        model_name="EnsembleStacker",
    )
    if not eval_report.consequence_breakdown:
        logger.warning(
            "consequence_breakdown EMPTY - check meta_test 'consequence' column"
        )
    if not eval_report.gene_errors:
        logger.warning(
            "gene_errors EMPTY - check meta_test 'gene_symbol' column"
        )
    evaluator.save_report(eval_report, out_dir / "eval_report.json")

    # -- 6. Save (before feature importance so a crash there never loses the model)
    logger.info("PHASE 6: Saving")
    ensemble.save(out_dir / "ensemble_v1.joblib")

    # -- 7. Feature importance ----------------------------------------------
    logger.info("PHASE 5: Feature Importance")
    fi_df = compute_feature_importance(ensemble, feature_names)
    if fi_df is not None:
        logger.info("Top 10 features:\n%s", fi_df.head(10).to_string(index=False))
        fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # Save validation split for InterpretabilityAgent (SHAP audit)
    _val_df = X_val.copy() if hasattr(X_val, "copy") else pd.DataFrame(X_val, columns=feature_names)
    _val_df["pathogenicity_class"] = y_val.values if hasattr(y_val, "values") else y_val
    _val_df.to_parquet(out_dir / "val.parquet", index=False)
    logger.info("Validation split saved to %s", out_dir / "val.parquet")

    # Save CatBoost native .cbm file for SHAP and InterpretabilityAgent
    _cb_key = next(
        (k for k in ensemble.trained_models_ if "catboost" in k.lower()), None
    )
    if _cb_key is not None:
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        _cb_wrapper = ensemble.trained_models_[_cb_key]
        if isinstance(_cb_wrapper, CatBoostVariantClassifier):
            _cbm_path = out_dir / "catboost_model.cbm"
            _cb_wrapper.save_catboost_model(_cbm_path)
            logger.info("CatBoost model saved to %s", _cbm_path)
        else:
            logger.warning("CatBoost model is not a CatBoostVariantClassifier -- .cbm not written.")
    else:
        logger.warning("No CatBoost model found in trained_models_ -- .cbm not written.")

    metrics_dict = (
        metrics_df
        .reset_index()
        .rename(columns={"index": "model"})
        .to_dict(orient="records")
    )
    metrics_path = out_dir / "metrics_v1.json"
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "run_timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_train":             int(len(y_train)),
                "n_test":              int(len(y_test)),
                "n_pathogenic_train":  int(y_train.sum()),
                "n_pathogenic_test":   int(y_test.sum()),
                "n_features":          len(feature_names),
                "feature_names":       feature_names,
                "annotation_sources":  {
                    "alphamissense": str(args.alphamissense) if args.alphamissense else None,
                    "finngen": str(args.finngen_path) if args.finngen_path else None,
                    "dbnsfp": str(args.dbnsfp_path) if args.dbnsfp_path else None,
                    "gnomad_constraint": str(args.gnomad_constraint) if args.gnomad_constraint else None,
                    "esm2_model": args.esm2_model,
                    "esm2_uniprot_index": str(args.esm2_uniprot_index) if args.esm2_uniprot_index else None,
                    "lovd":         str(args.lovd_path) if args.lovd_path else None,
                },
                "metrics":             metrics_dict,
                "training_time_sec":   round(time.time() - t0, 1),
            },
            fh, indent=2,
        )
    logger.info("Metrics saved to %s", metrics_path)

    # -- 8. Markdown summary ------------------------------------------------
    md_path = out_dir / "METRICS.md"
    write_metrics_markdown(metrics_df, y_train, y_test, md_path)
    logger.info("README-ready markdown saved to %s", md_path)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)
    logger.info(
        "Best model: %s  AUROC=%.4f",
        metrics_df["auroc"].idxmax(), metrics_df["auroc"].max(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_fast_mode(ensemble: "VariantEnsemble") -> None:
    """Reduce estimator counts for rapid development iteration."""
    for name, model in ensemble.base_estimators.items():
        if hasattr(model, "n_estimators"):
            model.set_params(n_estimators=50)
        if hasattr(model, "estimator") and hasattr(model.estimator, "n_estimators"):
            model.estimator.set_params(n_estimators=50)


def compute_feature_importance(
    ensemble: "VariantEnsemble",
    feature_names: list[str],
) -> "pd.DataFrame | None":
    """
    Aggregate feature importances from all tree-based ensemble members.

    Returns:
        DataFrame with columns [feature, importance, importance_std, n_models],
        sorted descending by importance.
        None if no tree models have feature_importances_ set.
    """
    importance_records: list[pd.Series] = []
    model_map = getattr(ensemble, "trained_models_", ensemble.base_estimators)

    for name, model in model_map.items():
        base = getattr(model, "estimator", model)
        base = getattr(base, "base_estimator", base)
        fi = getattr(base, "feature_importances_",
             getattr(model, "feature_importances_", None))
        # feature_importances_ may be a method (e.g. CatBoostVariantClassifier)
        # rather than a property — call it and unwrap if needed.
        if callable(fi):
            try:
                fi = fi()
            except Exception:
                fi = None
        # Unwrap list-of-tuples [(name, score), ...] from CatBoost wrapper
        if fi is not None and isinstance(fi, (list, tuple)) and fi and isinstance(fi[0], (list, tuple)):
            try:
                fi = [v for _, v in fi]
            except Exception:
                fi = None
        if fi is not None:
            try:
                fi = np.asarray(fi, dtype=float)
            except Exception:
                fi = None
        if fi is not None and fi.ndim == 1 and len(fi) == len(feature_names):
            importance_records.append(pd.Series(fi, index=feature_names, name=name))

    if not importance_records:
        logger.warning("No feature importances available from tree models.")
        return None

    fi_df = pd.concat(importance_records, axis=1)
    return pd.DataFrame({
        "feature":        feature_names,
        "importance":     fi_df.mean(axis=1).values,
        "importance_std": fi_df.std(axis=1).values,
        "n_models":       fi_df.notna().sum(axis=1).values,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def write_metrics_markdown(
    metrics_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    path: Path,
) -> None:
    """Write a clean Markdown table for the repository README."""
    best_auroc = metrics_df["auroc"].max()
    lines = [
        "## Model Performance\n",
        f"Evaluated on **{len(y_test):,} held-out variants** "
        f"({int(y_test.sum()):,} pathogenic, "
        f"{int((y_test == 0).sum()):,} benign) "
        f"using a **gene-aware train/test split** "
        f"(no gene appears in both train and test).\n",
        "",
        "| Model | AUROC | AUPRC | F1 (macro) | MCC | Brier |",
        "|-------|-------|-------|-----------|-----|-------|",
    ]
    for model_name, row in metrics_df.iterrows():
        bold = "**" if row["auroc"] == best_auroc else ""
        lines.append(
            f"| {bold}{model_name}{bold} "
            f"| {bold}{row['auroc']:.4f}{bold} "
            f"| {row['auprc']:.4f} "
            f"| {row['f1_macro']:.4f} "
            f"| {row['mcc']:.4f} "
            f"| {row['brier']:.4f} |"
        )
    lines += [
        "",
        "### Notes",
        "- **AUROC**: Area under ROC curve; primary metric for imbalanced classification.",
        "- **AUPRC**: Precision-Recall AUC; measures rare-class detection quality.",
        "- **MCC**: Matthews Correlation Coefficient; most informative single metric "
          "for imbalanced binary classification.",
        "- **Brier Score**: Calibration quality; lower is better (0 = perfect).",
        f"- Training set: {len(y_train):,} variants ({y_train.mean()*100:.1f}% pathogenic).",
        "",
        "_Generated by scripts/train.py_",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()