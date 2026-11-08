"""
scripts/calibrate_thresholds.py
================================
Research operating points on the SELECTED predictor's score, chosen on one split of a COMPLETED run.

Author: Monzia Moodie

    python scripts/calibrate_thresholds.py --run-dir outputs/run16/full --role val \\
        --output models/decision_policy.json

What it does (package 1, set 1c, 2026-11-08 -- evaluation/decision_policy.py holds the method and its rationale)
  1. Reads `<role>_predictions.parquet` of a completed run through prediction_artifacts.
     load_completed_split_predictions: the completion manifest and the prediction blocks are verified and the stored
     convex blend's `selected_score` is read with the identities of the predictor, run, file and rows. Nothing is
     re-scored, so no model is loaded.
  2. Chooses four operating points with the repository's exact sweep and Objective A (the most sensitive point whose
     predictive value clears a floor): two positive-predictive-value floors on `selected_score` and two
     negative-predictive-value floors on `1 - selected_score`.
  3. Writes a strict, digest-bearing gvc.decision-policy.v1 record. Consumers verify it and refuse it for any other
     predictor (DecisionPolicy.require_predictor).

What it no longer does, and why
  * It no longer scores the logistic meta-learner (removed from the inference pipeline in set 1a: it is not the
    selected predictor).
  * It no longer fits Platt or temperature scaling, picks one by its error on the same rows, and thresholds the
    transformed score. A strictly increasing transformation cannot change which rows clear a threshold, so it cannot
    change a predictive-value operating point; it only moved the thresholds onto a scale serving never produces.
  * It no longer returns the highest qualifying threshold while documenting the lowest, and it no longer falls back
    to the best-precision point when a floor is unreachable: it refuses (exit 1) and writes nothing.
  * Its bands carry neutral names, not ACMG/AMP tier labels: they are research operating points.

Exit codes
    0  policy written
    1  refused: a floor is unreachable, or the bands would be inconsistent or overlap (nothing written)
    2  the input is not a verified completed run with the requested split (nothing written)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("calibrate_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research operating points of the selected predictor, chosen on a "
                                            "completed run's split predictions.")
    p.add_argument("--run-dir", required=True, type=Path, help="a completed run directory (completion manifest)")
    p.add_argument("--role", default="val", choices=("val", "test", "holdout", "train"),
                   help="the split whose predictions choose the operating points (default: val)")
    p.add_argument("--output", required=True, type=Path, help="where the gvc.decision-policy.v1 record is written")
    p.add_argument("--positive-high-floor", type=float, default=0.95,
                   help="minimum positive predictive value of the positive_high_ppv band (default 0.95)")
    p.add_argument("--positive-moderate-floor", type=float, default=0.80,
                   help="minimum positive predictive value of every positive call (default 0.80)")
    p.add_argument("--negative-high-floor", type=float, default=0.95,
                   help="minimum negative predictive value of the negative_high_npv band (default 0.95)")
    p.add_argument("--negative-moderate-floor", type=float, default=0.80,
                   help="minimum negative predictive value of every negative call (default 0.80)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    from genomic_variant_classifier.evaluation.decision_policy import (
        DecisionPolicyRefused,
        build_decision_policy,
        write_decision_policy,
    )
    from genomic_variant_classifier.evaluation.prediction_artifacts import (
        IncompleteRunError,
        load_completed_split_predictions,
    )

    try:
        bound = load_completed_split_predictions(args.run_dir, args.role)
    except (IncompleteRunError, FileNotFoundError) as exc:
        logger.error("NOT a verified completed run with %s predictions: %s", args.role, exc)
        return 2
    logger.info("Bound input: %d %s rows (%d positive) of run %s/%s; predictor %s; file sha256 %s",
                len(bound.frame), args.role, int(bound.labels.sum()), bound.run_id, bound.ablation,
                bound.predictor["combiner_sha256"], bound.predictions_sha256)
    try:
        record = build_decision_policy(bound, positive_high=args.positive_high_floor,
                                       positive_moderate=args.positive_moderate_floor,
                                       negative_high=args.negative_high_floor,
                                       negative_moderate=args.negative_moderate_floor)
    except DecisionPolicyRefused as exc:
        logger.error("REFUSED -- nothing written: %s", exc)
        for name, entry in exc.outcomes.items():
            logger.error("  %s: %s", name, json.dumps(entry["outcome"]["selection"], sort_keys=True))
        return 1
    for name, point in record["operating_points"].items():
        metrics = point["outcome"]["metrics"]["metrics"]
        logger.info("  %-18s %s %s %.17g  (floor %.4f; sensitivity %s; predictive value %s)", name,
                    point["score_space"], point["operator"], point["threshold"], point["floor"],
                    metrics["sensitivity"]["value"], metrics["positive_predictive_value"]["value"])
    file_sha256 = write_decision_policy(record, args.output)
    logger.info("Decision policy written: %s (record_sha256 %s, file sha256 %s). Certification blockers: %s",
                args.output, record["record_sha256"], file_sha256, record["certification_blockers"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
