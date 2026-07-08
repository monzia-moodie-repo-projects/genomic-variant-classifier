"""
src/genomic_variant_classifier/evaluation
==============
Clinical evaluation package for the Genomic Variant Classifier.

=============================================================================
IMPORT CONTRACT — DO NOT IMPORT metrics.py FROM THIS FILE.

  This package MUST import cleanly with scikit-learn absent. The contract is
  locked by tests/unit/test_evaluator_phase5.py::test_module_imports_without_sklearn
  and (since 2026-07-08) tests/unit/test_evaluation_metrics.py::test_package_imports_without_sklearn.

      evaluator.py            lazy-loads sklearn via _ensure_sklearn()
      prediction_artifacts.py imports sklearn / shap INSIDE functions only
      metrics.py              imports sklearn AT MODULE LEVEL  <-- the trap

  History. Commit 87e32ad replaced this file with a two-line stub, deleting every
  re-export below. The restore in 015ff94 brought them back but ALSO added
  `from ... import metrics`, which pulled sklearn eagerly and broke the Phase-5
  contract — trading one silent failure for another. This file is now byte-exact
  with 87e32ad^ below the docstring.

  Import the metric stack directly, where sklearn is expected to be present:

      from genomic_variant_classifier.evaluation.metrics import auroc, auprc, evaluate

  KNOWN, DELIBERATELY UNCHANGED: `RunArtifactWriter` is imported but absent from
  __all__, and its import sits below the __all__ assignment. Both are pre-existing.
  Correcting them is a behaviour-visible change to `import *` and belongs in its own
  commit, not in a restore.
=============================================================================
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
