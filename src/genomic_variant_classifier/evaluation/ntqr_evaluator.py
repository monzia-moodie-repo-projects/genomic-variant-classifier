"""
src/genomic_variant_classifier/evaluation/ntqr_evaluator.py
=============================================================
NTQR (No-Test-Quorums Required) evaluator for variant pathogenicity
classifiers.

NTQR theory (Platzer & Schmidhuber 2023) provides algebraically exact
classifier-accuracy bounds without a labelled test set.  For variant
classifiers this is clinically important: it gives model-accuracy estimates
that are valid even for VUS (Variants of Uncertain Significance), where
ground-truth labels are unavailable.

This module wraps ``ntqr.r2.Evaluator2`` and integrates with
``ClinicalEvaluator`` so that NTQR bounds appear alongside standard
AUROC/AUPRC in evaluation reports.

Standing Rule #31 compliance
-----------------------------
ntqr must pass the bounded Start-Job smoke test before being added to
requirements.txt.  Run ``docs/preflight/ntqr_sr31_check.ps1`` and confirm
SR31_PASS.  When ntqr is absent the evaluator runs in stub mode: all bounds
are ``None`` and ``ntqr_available=False``.

⚠ PHASE_2_FEATURES: SR #31 smoke test is a hard prerequisite.

API contract (ntqr >= 0.3.0)
-----------------------------
    from ntqr.r2 import Evaluator2
    ev = Evaluator2(n_0, n_1)          # n_0=benign count, n_1=pathogenic count
    bounds = ev.classifier_accuracy_bounds(q_00, q_01, q_10, q_11)
    # bounds: {0: (lower_0, upper_0), 1: (lower_1, upper_1)}

Design rules
------------
- No logging.basicConfig at module level.
- from __future__ import annotations (standing rule).
- Stub mode is fully functional (no AttributeError on missing bounds).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ntqr import — stub mode when unavailable
# ---------------------------------------------------------------------------
_NTQR_AVAILABLE = False
try:
    from ntqr.r2 import Evaluator2 as _Evaluator2  # type: ignore[import]
    _NTQR_AVAILABLE = True
except ImportError:
    logger.warning(
        "ntqr not installed — NTQREvaluator will return stub bounds (None).  "
        "Install: pip install ntqr --break-system-packages  "
        "(run docs/preflight/ntqr_sr31_check.ps1 first)."
    )


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class NTQRBounds:
    """Algebraically exact accuracy bounds from the NTQR r2 evaluator."""

    n_benign:     int
    n_pathogenic: int

    # Confusion-matrix quadrants (at the chosen decision threshold)
    q_00: int  # TN — predicted benign   & true benign
    q_01: int  # FP — predicted pathogen & true benign
    q_10: int  # FN — predicted benign   & true pathogen
    q_11: int  # TP — predicted pathogen & true pathogen

    # Per-class accuracy bounds (None when ntqr unavailable)
    benign_accuracy_lower:      Optional[float]
    benign_accuracy_upper:      Optional[float]
    pathogenic_accuracy_lower:  Optional[float]
    pathogenic_accuracy_upper:  Optional[float]

    ntqr_available: bool

    # ── Convenience aliases ─────────────────────────────────────────────────

    @property
    def sensitivity_lower(self) -> Optional[float]:
        """Lower bound on sensitivity (pathogenic recall)."""
        return self.pathogenic_accuracy_lower

    @property
    def sensitivity_upper(self) -> Optional[float]:
        """Upper bound on sensitivity."""
        return self.pathogenic_accuracy_upper

    @property
    def specificity_lower(self) -> Optional[float]:
        """Lower bound on specificity (benign recall)."""
        return self.benign_accuracy_lower

    @property
    def specificity_upper(self) -> Optional[float]:
        """Upper bound on specificity."""
        return self.benign_accuracy_upper

    def to_dict(self) -> dict:
        return {
            "n_benign":                     self.n_benign,
            "n_pathogenic":                 self.n_pathogenic,
            "q_00":                         self.q_00,
            "q_01":                         self.q_01,
            "q_10":                         self.q_10,
            "q_11":                         self.q_11,
            "benign_accuracy_lower":        self.benign_accuracy_lower,
            "benign_accuracy_upper":        self.benign_accuracy_upper,
            "pathogenic_accuracy_lower":    self.pathogenic_accuracy_lower,
            "pathogenic_accuracy_upper":    self.pathogenic_accuracy_upper,
            "ntqr_available":               self.ntqr_available,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class NTQREvaluator:
    """
    Wraps ``ntqr.r2.Evaluator2`` for binary variant pathogenicity classifiers.

    Parameters
    ----------
    threshold : float
        Decision threshold for converting predicted probabilities to
        {0, 1} class predictions.  Defaults to 0.5.
        For clinical use, set to the high-PPV threshold from
        ``ClinicalEvaluator.find_high_confidence_threshold()``.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"NTQREvaluator threshold must be in (0, 1), got {threshold}."
            )
        self.threshold = threshold

    # ── Public entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        y_true:  np.ndarray,
        y_proba: np.ndarray,
    ) -> NTQRBounds:
        """
        Compute NTQR accuracy bounds for a binary classifier output.

        Parameters
        ----------
        y_true  : shape (n,), integer 0/1 ground-truth labels.
        y_proba : shape (n,), predicted pathogenic probabilities in [0, 1].

        Returns
        -------
        NTQRBounds dataclass.  When ntqr is not installed, bounds fields are
        ``None`` and ``ntqr_available`` is ``False`` (stub mode).
        """
        y_true  = np.asarray(y_true,  dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_true.shape != y_proba.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} != y_proba shape {y_proba.shape}"
            )

        y_pred = (y_proba >= self.threshold).astype(int)

        n_0  = int((y_true == 0).sum())   # benign
        n_1  = int((y_true == 1).sum())   # pathogenic
        q_00 = int(((y_pred == 0) & (y_true == 0)).sum())  # TN
        q_01 = int(((y_pred == 1) & (y_true == 0)).sum())  # FP
        q_10 = int(((y_pred == 0) & (y_true == 1)).sum())  # FN
        q_11 = int(((y_pred == 1) & (y_true == 1)).sum())  # TP

        # Sanity check — counts must be self-consistent.
        assert q_00 + q_01 == n_0, "Benign confusion-matrix rows do not sum to n_0"
        assert q_10 + q_11 == n_1, "Pathogenic confusion-matrix rows do not sum to n_1"

        if not _NTQR_AVAILABLE:
            return NTQRBounds(
                n_benign=n_0,     n_pathogenic=n_1,
                q_00=q_00,        q_01=q_01,
                q_10=q_10,        q_11=q_11,
                benign_accuracy_lower=None,    benign_accuracy_upper=None,
                pathogenic_accuracy_lower=None, pathogenic_accuracy_upper=None,
                ntqr_available=False,
            )

        try:
            ev     = _Evaluator2(n_0, n_1)
            bounds = ev.classifier_accuracy_bounds(q_00, q_01, q_10, q_11)
            # ntqr >= 0.3.0 returns {0: (lo, hi), 1: (lo, hi)}
            b0_lo, b0_hi = bounds[0]
            b1_lo, b1_hi = bounds[1]
        except Exception as exc:
            logger.error(
                "ntqr.r2.Evaluator2 raised %s: %s.  "
                "Check ntqr version compatibility (requires >= 0.3.0).  "
                "Falling back to stub mode.",
                type(exc).__name__, exc,
            )
            return NTQRBounds(
                n_benign=n_0,     n_pathogenic=n_1,
                q_00=q_00,        q_01=q_01,
                q_10=q_10,        q_11=q_11,
                benign_accuracy_lower=None,    benign_accuracy_upper=None,
                pathogenic_accuracy_lower=None, pathogenic_accuracy_upper=None,
                ntqr_available=False,
            )

        result = NTQRBounds(
            n_benign=n_0,       n_pathogenic=n_1,
            q_00=q_00,          q_01=q_01,
            q_10=q_10,          q_11=q_11,
            benign_accuracy_lower=float(b0_lo),     benign_accuracy_upper=float(b0_hi),
            pathogenic_accuracy_lower=float(b1_lo), pathogenic_accuracy_upper=float(b1_hi),
            ntqr_available=True,
        )
        logger.info(
            "NTQR bounds (threshold=%.3f): "
            "benign=[%.4f, %.4f]  pathogenic=[%.4f, %.4f]  "
            "TP=%d FP=%d FN=%d TN=%d",
            self.threshold,
            result.benign_accuracy_lower,    result.benign_accuracy_upper,
            result.pathogenic_accuracy_lower, result.pathogenic_accuracy_upper,
            q_11, q_01, q_10, q_00,
        )
        return result
