"""
scripts/validate_external.py
==============================
Phase 8.1 — External validation cohort evaluation.

Runs the trained InferencePipeline against an external validation cohort
(ClinVar holdout, gnomAD de-novo, or a custom labelled VCF) and writes a
comprehensive performance report.

Metrics computed
----------------
  AUROC          Area under ROC curve
  AUPRC          Area under precision-recall curve
  ECE            Expected calibration error (15 equal-width bins)
  MCE            Maximum calibration error
  Operating points (only with --decision-policy): counts and typed rates at each operating point of a VERIFIED
                 decision policy chosen for THIS predictor (evaluation/decision_policy.py)

Output files (--output-dir)
----------------------------
  metrics.json          All scalar metrics, the predictor's identity and the decision policy's status
  roc_curve.parquet     fpr, tpr, threshold columns (scikit-learn's own thresholds; the first is +inf)
  pr_curve.parquet      precision, recall, threshold columns (the final point has no threshold: null)
  calibration.parquet   mean_prob, fraction_positives per bin
  predictions.parquet   variant_id, label, score (full precision) and, with a policy, its band

Package 1, set 1c (2026-11-08): the score is the selected predictor's (predict_proba = the stored convex blend). The
five "ACMG-tier" thresholds are gone: they came from api/schemas.CLASSIFICATION_THRESHOLDS, loaded at import from a
path relative to the current directory with every error swallowed, and they were passed to the threshold metrics as
(lo, hi) TUPLES, so `y_prob >= thr` compared each score with a pair. A decision policy is now an explicit argument,
verified and refused for any other predictor; without one no threshold is applied and the report says so. The ROC
frame no longer appends a threshold (its arrays already have equal length, so the old frame raised), the PR frame no
longer invents 1.0 for its final point, nothing is rounded, the unbound `pipeline_val_auroc` is replaced by the
pipeline's predictor identity and bound evaluation evidence, and metrics.json is strict JSON.

Usage
-----
  python scripts/validate_external.py \
      --pipeline   models/phase2_pipeline.joblib \
      --cohort     data/external/validation_cohort.parquet \
      --output-dir outputs/external_validation \
      [--decision-policy models/decision_policy.json]

  # cohort parquet must have columns:
  #   - All feature columns accepted by /predict (chrom, pos, ref, alt, …)
  #   - label  (1=pathogenic, 0=benign)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute ECE and MCE using equal-width probability bins.

    Returns
    -------
    ece : float
    mce : float
    cal_df : pd.DataFrame  columns: bin_lower, bin_upper, mean_prob, fraction_pos, count
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    n   = len(y_true)

    for _b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Top bin CLOSED to [lo, 1.0]; see the note in scripts/calibrate_thresholds.py.
        # Repaired 2026-07-20.
        if _b == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos  = float(y_true[mask].mean())
        mean_prob = float(y_prob[mask].mean())
        cnt       = int(mask.sum())
        gap       = abs(mean_prob - frac_pos)
        ece      += gap * cnt / n
        mce       = max(mce, gap)
        rows.append({
            "bin_lower":       round(lo, 4),
            "bin_upper":       round(hi, 4),
            "mean_prob":       round(mean_prob, 4),
            "fraction_pos":    round(frac_pos, 4),
            "count":           cnt,
        })

    return round(ece, 4), round(mce, 4), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Operating points of a verified decision policy
# ---------------------------------------------------------------------------

def _policy_metrics(policy, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Counts and typed rates at each operating point of `policy` on this cohort (refusal-aware: an undefined rate is
    a status with a reason, never a fabricated 0.0)."""
    from genomic_variant_classifier.evaluation.thresholds import (
        ConfusionCounts,
        ThresholdOperator,
        ThresholdParameters,
        ThresholdSource,
        metrics_from_counts,
    )

    out = {}
    for name, point in policy.record["operating_points"].items():
        values = y_prob if point["score_space"] == "selected_score" else 1.0 - y_prob
        target = y_true if point["score_space"] == "selected_score" else 1 - y_true
        calls = values >= point["threshold"] if point["operator"] == ">=" else values > point["threshold"]
        counts = ConfusionCounts(true_positive=int((calls & (target == 1)).sum()),
                                 false_positive=int((calls & (target == 0)).sum()),
                                 false_negative=int((~calls & (target == 1)).sum()),
                                 true_negative=int((~calls & (target == 0)).sum()))
        parameters = ThresholdParameters(threshold=point["threshold"], operator=ThresholdOperator(point["operator"]),
                                         source=ThresholdSource.CALIBRATED)
        out[name] = {"score_space": point["score_space"], "floor_chosen_for": point["floor"],
                     **metrics_from_counts(counts, parameters).to_dict()}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="External cohort validation for the variant pathogenicity pipeline."
    )
    parser.add_argument("--pipeline",   required=True, help="Path to InferencePipeline .joblib")
    parser.add_argument("--cohort",     required=True, help="Labelled cohort parquet (must have 'label' column)")
    parser.add_argument("--output-dir", default="outputs/external_validation",
                        help="Directory for output files (default: outputs/external_validation)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Scoring batch size (default: 2048)")
    parser.add_argument("--decision-policy", default=None,
                        help="a gvc.decision-policy.v1 record (scripts/calibrate_thresholds.py) chosen for THIS "
                             "predictor; without it no threshold is applied")
    parser.add_argument("--log-level",  default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pipeline ---
    logger.info("Loading pipeline from %s …", args.pipeline)
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from genomic_variant_classifier.model_admission import repository_registry_path
    pipeline = InferencePipeline.load(args.pipeline, consumer="scripts/validate_external.py",
                                      registry_path=repository_registry_path())
    policy = None
    if args.decision_policy:
        from genomic_variant_classifier.evaluation.decision_policy import load_decision_policy
        policy = load_decision_policy(args.decision_policy)
        policy.require_predictor(pipeline.combiner.digest)      # refused BEFORE any cohort is read or scored

    # --- Load cohort ---
    logger.info("Loading cohort from %s …", args.cohort)
    df = pd.read_parquet(args.cohort)
    if "label" not in df.columns:
        logger.error("Cohort parquet must contain a 'label' column (1=pathogenic, 0=benign).")
        return 1

    y_true = df["label"].astype(int).values
    n_pos  = int(y_true.sum())
    n_neg  = int((y_true == 0).sum())
    logger.info("Cohort: %d variants (%d pathogenic, %d benign).", len(df), n_pos, n_neg)

    if set(np.unique(y_true)) != {0, 1}:
        logger.error("'label' column must be binary (0/1). Found: %s", np.unique(y_true).tolist())
        return 1

    # --- Score in batches ---
    logger.info("Scoring %d variants (batch_size=%d) …", len(df), args.batch_size)
    scores = []
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start : start + args.batch_size]
        proba = pipeline.predict_proba(chunk)
        scores.append(proba)
    y_prob = np.concatenate(scores)

    # --- AUROC / AUPRC ---
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    logger.info("AUROC=%.4f  AUPRC=%.4f", auroc, auprc)

    # --- Calibration ---
    ece, mce, cal_df = expected_calibration_error(y_true, y_prob)
    logger.info("ECE=%.4f  MCE=%.4f", ece, mce)

    # --- Operating points: only those of a verified policy chosen for this predictor ---
    if policy is not None:
        decision_policy = {"status": "applied", "record_sha256": policy.record_sha256,
                           "file_sha256": policy.file_sha256, "policy_status": policy.record["status"],
                           "certification_blockers": policy.record["certification_blockers"],
                           "operating_points": _policy_metrics(policy, y_true, y_prob)}
    else:
        decision_policy = {"status": "unavailable",
                           "reason": "no --decision-policy was given; no threshold is applied to the selected score"}

    # --- ROC / PR curves ---
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)

    # --- Predictions table ---
    pred_df = pd.DataFrame({
        "variant_id":    df.get("variant_id", pd.Series(
            [f"{r.chrom}:{r.pos}:{r.ref}:{r.alt}" for _, r in df.iterrows()],
            index=df.index,
        )),
        "label":         y_true,
        "score":         y_prob,
    })
    if policy is not None:
        pred_df["decision_band"] = policy.classify(np.asarray(y_prob, dtype=np.float64))

    # --- Write outputs ---
    from genomic_variant_classifier.evaluation.serialization import dump_strict_json
    metrics = {
        "n_variants":    len(df),
        "n_pathogenic":  n_pos,
        "n_benign":      n_neg,
        "auroc":         float(auroc),
        "auprc":         float(auprc),
        "ece":           ece,
        "mce":           mce,
        "decision_policy": decision_policy,
        "predictor": {"combiner_sha256": pipeline.combiner.digest,
                      "predictor_contract_sha256": pipeline.predictor_contract_sha256(),
                      "evaluation_evidence": pipeline.metadata.evaluation_evidence},
    }

    (output_dir / "metrics.json").write_text(dump_strict_json(metrics, artifact="metrics.json") + "\n",
                                             encoding="utf-8", newline="\n")

    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_parquet(
        output_dir / "roc_curve.parquet", index=False
    )
    # precision_recall_curve returns one more (precision, recall) point than thresholds: the final point (recall 0) is
    # reached by no threshold, so its threshold is null rather than an invented 1.0.
    pd.DataFrame({"precision": prec, "recall": rec, "threshold": np.append(pr_thr, np.nan)}).to_parquet(
        output_dir / "pr_curve.parquet", index=False
    )
    cal_df.to_parquet(output_dir / "calibration.parquet", index=False)
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)

    logger.info("Outputs written to %s/", output_dir)
    logger.info(
        "Summary: AUROC=%.4f  AUPRC=%.4f  ECE=%.4f  n=%d",
        auroc, auprc, ece, len(df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
