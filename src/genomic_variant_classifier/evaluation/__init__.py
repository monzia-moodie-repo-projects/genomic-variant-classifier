"""
src/genomic_variant_classifier/evaluation
==============
Clinical evaluation package for the Genomic Variant Classifier.

RESTORED 2026-07-08. Commit 87e32ad replaced this file with a two-line stub, silently
removing every re-export below. Restored verbatim from 87e32ad^; the metric-stack names
are APPENDED, not substituted.
"""

from __future__ import annotations

from genomic_variant_classifier.evaluation.evaluator import (
    ClinicalEvaluator,
    ConsequenceBreakdown,
    EvaluationReport,
    GeneErrorAnalysis,
    OperatingPoint,
    compare_models,
)

__all__ = [
    "ClinicalEvaluator",
    "ConsequenceBreakdown",
    "EvaluationReport",
    "GeneErrorAnalysis",
    "OperatingPoint",
    "compare_models",
]
from genomic_variant_classifier.evaluation.prediction_artifacts import RunArtifactWriter

# --- metric stack (2026-07-08) ----------------------------------------------------
# Additive. Nothing above is changed. `RunArtifactWriter` was imported but omitted from
# __all__ in the original -- a pre-existing inconsistency, corrected here.
from genomic_variant_classifier.evaluation import metrics  # noqa: E402,F401
from genomic_variant_classifier.evaluation.metrics import (  # noqa: E402
    auroc,
    auprc,
    no_skill_auprc,
    brier_score,
    expected_calibration_error,
    calibration_slope_intercept,
    bootstrap_ci,
    evaluate,
    stratified_evaluate,
    compute_classification_metrics,
    ModelEvaluator,
)

__all__ += [
    "RunArtifactWriter",
    "metrics",
    "auroc",
    "auprc",
    "no_skill_auprc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope_intercept",
    "bootstrap_ci",
    "evaluate",
    "stratified_evaluate",
    "compute_classification_metrics",
    "ModelEvaluator",
]
