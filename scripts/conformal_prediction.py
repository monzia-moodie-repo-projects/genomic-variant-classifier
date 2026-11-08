"""
scripts/conformal_prediction.py
=================================
Phase 8.3 — Conformal prediction intervals for /predict responses.

Implements split conformal prediction (Papadopoulos et al. 2002) to produce
prediction sets with guaranteed marginal coverage at a user-specified error
level α.

Package 1, set 1c (2026-11-08) -- two defects of this script, measured and repaired:
  * THE QUANTILE. It took `np.quantile(scores, ceil((n+1)(1-α))/n)`. Linear
    interpolation at that level lands between the k-th and (k+1)-th smallest
    scores (position (n-1)k/n), so it was not the conformal threshold (it was
    conservative there); and when k > n it clipped the level to 1 and returned
    the LARGEST score where the guarantee requires every label in the set --
    anti-conservative. It now delegates to the package's tested
    `conformal.split.conformal_quantile` (the k-th smallest score; +inf when
    k > n, recorded as unbounded).
  * THE OUTPUT. For binary labels the conformal object is the prediction SET
    {y : r(x, y) <= q̂}; the old "interval" [s - q̂, s + q̂] was not a conformal
    quantity. `--apply` now writes which labels the set contains, and the
    score at full precision.
The score is the pipeline's predict_proba -- the selected convex blend.

Algorithm (split conformal)
---------------------------
1. Score a calibration split with the trained pipeline → s_i = f(x_i).
2. Define non-conformity scores as residuals:
     r_i = |y_i - s_i|        for regression-style intervals
   or for classification:
     r_i = 1 - s_i  if y_i = 1   (pathogenic)
     r_i = s_i      if y_i = 0   (benign)
3. Compute the (1-α)(1 + 1/n) quantile q̂ of {r_i}.
4. At test time, the prediction set is:
     Ŷ = {y : r(x, y) ≤ q̂}
   which for binary labels is: pathogenic in the set iff 1 - s <= q̂, and
   benign in the set iff s <= q̂ (both, one, or -- rarely -- neither).

Coverage guarantee: P(y_true ∈ Ŷ) ≥ 1 - α for exchangeable data.

Output
------
  conformal_config.json   q̂ values for α ∈ {0.01, 0.05, 0.10, 0.20}
                          + the coverage on the calibration rows themselves
                          (in-sample: >= 1 - α by construction, not an
                          estimate of coverage on new variants)

  Serving does not load this file: response schema 3.0.0 reports the coverage
  interval as unavailable.

Usage
-----
  python scripts/conformal_prediction.py \
      --pipeline  models/phase2_pipeline.joblib \
      --cal-split data/splits/val.parquet \
      --output    models/conformal_config.json \
      --alpha     0.05 0.10

  # Add --apply to score a test parquet and write intervals:
  python scripts/conformal_prediction.py \
      --pipeline  models/phase2_pipeline.joblib \
      --cal-split data/splits/val.parquet \
      --apply     data/splits/test.parquet \
      --output    outputs/test_with_intervals.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ALPHAS = [0.01, 0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Core conformal calibration
# ---------------------------------------------------------------------------

def calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float,
) -> float:
    """
    The conformal threshold q̂ for error level α: the k-th smallest
    non-conformity score, k = ceil((n+1)(1-α)), or +inf when k > n.

    Non-conformity score (least-ambiguous-set score of the true label):
        r_i = 1 - y_prob_i  if y_i = 1
        r_i = y_prob_i      if y_i = 0

    Delegates to conformal.split.conformal_quantile (set 1c): the previous
    np.quantile form interpolated between the k-th and (k+1)-th order
    statistics, and returned the largest score instead of +inf when k > n.
    """
    from genomic_variant_classifier.conformal.split import conformal_quantile

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or not np.isin(y_true, (0, 1)).all():
        raise ValueError("one binary label per probability required")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    scores = np.where(y_true == 1, 1.0 - y_prob, y_prob)
    return conformal_quantile(scores, alpha)


def prediction_set(
    y_prob: np.ndarray,
    q_hat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (contains_pathogenic, contains_benign): the binary conformal prediction set
    {y : r(x, y) <= q̂} as two boolean arrays. q̂ = +inf contains both labels.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    return (1.0 - y_prob) <= q_hat, y_prob <= q_hat


def empirical_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    q_hat: float,
) -> float:
    """
    The fraction of rows whose true label is in its prediction set. On the
    calibration rows themselves this is >= 1 - α by construction.
    """
    contains_pathogenic, contains_benign = prediction_set(y_prob, q_hat)
    covered = np.where(np.asarray(y_true) == 1, contains_pathogenic, contains_benign)
    return float(covered.mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conformal prediction interval calibration."
    )
    parser.add_argument("--pipeline",  required=True, help="InferencePipeline .joblib")
    parser.add_argument("--cal-split", required=True, help="Calibration parquet (must have 'label' column)")
    parser.add_argument("--output",    default="models/conformal_config.json",
                        help="Output JSON path (default: models/conformal_config.json)")
    parser.add_argument("--alpha",     type=float, nargs="+", default=_DEFAULT_ALPHAS,
                        help="Error levels to calibrate (default: 0.01 0.05 0.10 0.20)")
    parser.add_argument("--apply",     default=None,
                        help="Optional: path to test parquet to annotate with intervals")
    parser.add_argument("--apply-alpha", type=float, default=0.05,
                        help="α level to use when --apply is set (default: 0.05)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load pipeline ---
    logger.info("Loading pipeline from %s …", args.pipeline)
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from genomic_variant_classifier.model_admission import repository_registry_path
    pipeline = InferencePipeline.load(args.pipeline, consumer="scripts/conformal_prediction.py",
                                      registry_path=repository_registry_path())

    # --- Load calibration split ---
    logger.info("Loading calibration split from %s …", args.cal_split)
    cal_df  = pd.read_parquet(args.cal_split)
    if "label" not in cal_df.columns:
        logger.error("Calibration parquet must have a 'label' column.")
        return 1

    y_true  = cal_df["label"].astype(int).values
    y_prob  = pipeline.predict_proba(cal_df)
    n_cal   = len(y_true)
    logger.info("Calibration set: %d variants (%d pathogenic).", n_cal, int(y_true.sum()))

    # --- Calibrate for each α ---
    config: dict = {"n_calibration": n_cal, "alphas": {}}

    for alpha in sorted(args.alpha):
        q_hat    = calibrate(y_true, y_prob, alpha)
        coverage = empirical_coverage(y_true, y_prob, q_hat)
        config["alphas"][str(alpha)] = {
            # Full precision; +inf (alpha < 1/(n+1)) is recorded as unbounded, never as the largest score.
            "q_hat":             q_hat if np.isfinite(q_hat) else None,
            "q_hat_status":      "finite" if np.isfinite(q_hat) else "unbounded_every_label_in_the_set",
            "calibration_set_coverage_in_sample": coverage,
            "target_coverage":   1 - alpha,
        }
        logger.info(
            "α=%.2f  q̂=%s  in-sample coverage=%.4f  (target=%.4f)",
            alpha, q_hat, coverage, 1 - alpha,
        )

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, allow_nan=False)
    logger.info("Conformal config written to %s", output_path)

    # --- Optional: annotate a test set ---
    if args.apply:
        apply_alpha = args.apply_alpha
        entry = config["alphas"].get(str(apply_alpha))
        if entry is None:
            logger.error("α=%.2f not in calibrated alphas.", apply_alpha)
            return 1
        q_hat = entry["q_hat"] if entry["q_hat"] is not None else np.inf

        logger.info("Annotating %s with α=%.2f prediction sets …", args.apply, apply_alpha)
        test_df = pd.read_parquet(args.apply)
        test_prob = pipeline.predict_proba(test_df)
        contains_pathogenic, contains_benign = prediction_set(test_prob, q_hat)

        test_df = test_df.copy()
        test_df["pathogenicity_score"] = test_prob             # full precision (no rounding)
        test_df["set_contains_pathogenic"] = contains_pathogenic
        test_df["set_contains_benign"] = contains_benign

        apply_out = Path(args.apply).with_suffix("").name + "_with_prediction_sets.parquet"
        apply_path = output_path.parent / apply_out
        test_df.to_parquet(apply_path, index=False)
        logger.info("Annotated test set written to %s", apply_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
