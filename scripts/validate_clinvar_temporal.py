"""
scripts/validate_clinvar_temporal.py
=====================================
ClinVar temporal holdout validation -- Step 7A.

Evaluates the InferencePipeline on ClinVar variants submitted *after* the
training data cutoff date.  This is the fastest available form of external
validation: it uses the same database but a genuinely held-out time window.

Limitations vs. full external validation
-----------------------------------------
- Same label source (ClinVar) -- shares submitter and review biases
- Reviews from the same clinical centres may share systematic errors
- Treats this as a weak external validation; LOVD and UK Biobank are stronger

Metrics produced
----------------
  AUROC, AUPRC, Brier score, ECE (15 bins)
  Per-threshold sensitivity, specificity, PPV, NPV, F1
  Label distribution of new variants (new pathogenic / benign / VUS)
  Overlap with training set (fraction of new variants already in training)

Usage
-----
  python scripts/validate_clinvar_temporal.py \\
      --clinvar    data/processed/clinvar_grch38.parquet \\
      --splits-dir outputs/phase2_with_gnomad/splits \\
      --model      models/phase2_pipeline.joblib \\
      --cutoff     2024-01-01 \\
      --output     outputs/temporal_validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_clinvar_temporal")


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    bins = np.linspace(0, 1, n_bins + 1)
    counts = np.histogram(y_proba, bins=bins)[0]
    ece = sum((c / len(y_true)) * abs(fp - mp)
              for fp, mp, c in zip(frac_pos, mean_pred, counts))
    return float(ece)


def _threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    f1   = 2 * ppv * sens / max(ppv + sens, 1e-9)
    return dict(threshold=threshold, sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def main() -> None:
    p = argparse.ArgumentParser(description="ClinVar temporal holdout validation.")
    p.add_argument("--clinvar",    type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True,
                   help="Directory containing X_train.parquet, y_train.parquet, etc.")
    p.add_argument("--model",      type=Path, default=Path("models/phase2_pipeline.joblib"))
    p.add_argument("--cutoff",     type=str,  default="2024-01-01",
                   help="ISO date; variants submitted after this date form the test set.")
    p.add_argument("--output",     type=Path, default=Path("outputs/temporal_validation"))
    args = p.parse_args()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    logger.info("Loading model from %s ...", args.model)
    from src.api.pipeline import InferencePipeline
    pipeline = InferencePipeline.load(args.model)
    logger.info("Model val_auroc=%.4f, n_features=%d",
                pipeline.metadata.val_auroc, pipeline.metadata.n_features)

    # -----------------------------------------------------------------------
    # Load ClinVar and filter to post-cutoff variants
    # -----------------------------------------------------------------------
    logger.info("Loading ClinVar from %s ...", args.clinvar)
    clinvar = pd.read_parquet(args.clinvar)
    logger.info("Total ClinVar variants: %d", len(clinvar))

    # Identify training set variant_ids to exclude overlap
    train_ids: set[str] = set()
    for split in ("X_train", "X_val"):
        f = args.splits_dir / f"{split}.parquet"
        if f.exists():
            split_df = pd.read_parquet(f)
            if "variant_id" in split_df.columns:
                train_ids.update(split_df["variant_id"].astype(str))
            else:
                train_ids.update(split_df.index.astype(str))
    logger.info("Training+val set size (for deduplication): %d", len(train_ids))

    # Date filter -- try common ClinVar date columns
    date_col = None
    for col in ("LastEvaluated", "last_evaluated", "SubmissionDate", "date_last_evaluated"):
        if col in clinvar.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning(
            "No date column found in ClinVar parquet. "
            "Cannot perform temporal split. Available columns: %s",
            list(clinvar.columns)[:20],
        )
        new_variants = clinvar
    else:
        clinvar[date_col] = pd.to_datetime(clinvar[date_col], errors="coerce")
        new_variants = clinvar[clinvar[date_col] > args.cutoff].copy()
        logger.info("Variants submitted after %s: %d", args.cutoff, len(new_variants))

    # Remove overlap with training set
    if "variant_id" in new_variants.columns:
        id_col = "variant_id"
    else:
        id_col = None

    if id_col and train_ids:
        before = len(new_variants)
        new_variants = new_variants[~new_variants[id_col].astype(str).isin(train_ids)]
        logger.info("After removing training overlap: %d (removed %d)", len(new_variants), before - len(new_variants))

    # Label column
    label_col = None
    for col in ("label", "acmg_label", "pathogenic"):
        if col in new_variants.columns:
            label_col = col
            break

    if label_col is None:
        logger.error("No label column found. Expected one of: label, acmg_label, pathogenic.")
        raise SystemExit(1)

    labeled = new_variants[new_variants[label_col].isin([0, 1])].copy()
    if len(labeled) < 50:
        logger.error("Only %d labeled variants after cutoff -- too few to evaluate.", len(labeled))
        raise SystemExit(1)

    logger.info(
        "Labeled temporal holdout: %d total, %d pathogenic (%.1f%%), %d benign",
        len(labeled),
        int(labeled[label_col].sum()),
        100 * labeled[label_col].mean(),
        int((labeled[label_col] == 0).sum()),
    )

    # -----------------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------------
    logger.info("Scoring %d variants ...", len(labeled))
    y_proba = pipeline.predict_proba(labeled)
    y_true  = labeled[label_col].values

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    ece   = _ece(y_true, y_proba)

    logger.info("Temporal holdout AUROC: %.4f", auroc)
    logger.info("Temporal holdout AUPRC: %.4f", auprc)
    logger.info("Temporal holdout Brier: %.4f", brier)
    logger.info("Temporal holdout ECE:   %.4f", ece)

    threshold_results = [
        _threshold_metrics(y_true, y_proba, t)
        for t in [0.30, 0.50, 0.70, 0.90]
    ]

    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    prec_vals, rec_vals, pr_thresh = [], [], []
    try:
        from sklearn.metrics import precision_recall_curve
        prec_vals, rec_vals, pr_thresh = precision_recall_curve(y_true, y_proba)
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = {
        "validation_type": "clinvar_temporal_holdout",
        "cutoff_date": args.cutoff,
        "n_variants": int(len(labeled)),
        "n_pathogenic": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "model_val_auroc": pipeline.metadata.val_auroc,
        "threshold_metrics": threshold_results,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(args.output / "roc_curve.parquet", index=False)

    labeled_out = labeled.copy()
    labeled_out["predicted_proba"] = y_proba
    labeled_out.to_parquet(args.output / "predictions.parquet", index=False)

    logger.info("Results saved to %s", args.output)
    logger.info(
        "\n=== Temporal Holdout Summary ===\n"
        "  AUROC: %.4f  (training val: %.4f)\n"
        "  AUPRC: %.4f\n"
        "  Brier: %.4f\n"
        "  ECE:   %.4f\n"
        "  N:     %d (%d pathogenic)",
        auroc, pipeline.metadata.val_auroc,
        auprc, brier, ece,
        len(labeled), int(y_true.sum()),
    )


if __name__ == "__main__":
    main()
