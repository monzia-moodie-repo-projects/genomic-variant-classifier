"""Confidence-Based Performance Estimation (CBPE) -- estimate model performance WITHOUT labels.

Created 2026-07-13 (roadmap 6.19).

WHAT THIS IS FOR
----------------
A new ClinVar release lands. Its variants have not been adjudicated yet -- there are no
ground-truth labels, and there will not be for months. The question we need answered *now* is:

    "Is the classifier still performing, on THIS data, or has the world moved under it?"

Confidence-Based Performance Estimation answers exactly that. It takes the model's PREDICTED
PROBABILITIES on the new, unlabelled data, and -- calibrated against a labelled reference
period -- estimates ROC AUC (Receiver Operating Characteristic, Area Under Curve), accuracy,
and related metrics, with confidence bounds. If the estimate collapses, the model is degrading
on real data and we know before anyone gets a wrong variant call.

WHY THIS FILE EXISTS AT ALL
---------------------------
`requirements.in` has claimed since 2026-05 that drift monitoring includes
"nannyml (CBPE)". On 2026-07-13 an audit found:

  * `nannyml` was declared in requirements.in but ABSENT from requirements.txt -- the ONLY
    file Continuous Integration, Docker and every launch script install. It was therefore
    never installed anywhere except the developer's laptop.
  * On the laptop it COULD NOT IMPORT AT ALL: it optionally imports `pyspark`, and
    pyspark 3.5.x's pandas module calls `np.NaN`, which NumPy REMOVED in 2.0.
  * And no code anywhere called it. It was imported by exactly one file --
    `scripts/verify_drift_libs.py` -- which prints its version number.

So the capability was claimed, never built, never installed, and could not have run. This
module is the capability, built for real.

THE DEPENDENCY PROBLEM, AND WHY THIS RUNS IN A SEPARATE ENVIRONMENT
-------------------------------------------------------------------
nannyml 0.13.0 AND 0.13.1 both require:

    lightgbm >=3.3,<4.6      -- the ensemble TRAINS on lightgbm 4.6.0
    plotly   >=5.6,<6.0      -- the project runs plotly 6.6.0

There is NO release of nannyml compatible with lightgbm 4.6.0. LightGBM is a BASE MODEL of
this classifier. Downgrading it to satisfy a MONITORING library would change the science to
suit the instrument -- and it would move the very model whose silent positional-column-mapping
behaviour was measured on 2026-07-13 (0.855 probability swing on mis-ordered columns; see
tests/unit/test_feature_name_contract.py). That trade is refused.

Instead, the drift monitor runs in its OWN environment: `requirements-drift.txt`, with
lightgbm 4.5.0, plotly 5.24.1, NO pyspark, and the shared-data contract (numpy 2.4.4,
pandas 2.3.3, scikit-learn 1.8.0) PINNED TO MATCH the training stack. Verified 2026-07-13:
pip check clean, nannyml imports, CBPE estimates ROC AUC on unlabelled data.

    HARD BOUNDARY: this environment holds LightGBM 4.5.0 and XGBoost 2.1.4, NOT the 4.6.0 /
    3.2.0 the ensemble is trained with. IT MUST NEVER UNPICKLE A TRAINED MODEL. Deserialising
    a 4.6.0 booster into a 4.5.0 runtime is either a warning nobody reads or silently wrong
    deserialisation -- root pattern (d), a green result from a mutated environment.

    CBPE consumes PREDICTED PROBABILITIES, not the model, so the boundary costs nothing. It is
    asserted by a test in tests_drift/, not left to memory.

WHERE THE REFERENCE SET COMES FROM
----------------------------------
`VariantEnsemble` already persists everything needed:

    oof_predictions_   out-of-fold probabilities, one column per base model
    oof_fit_indices_   the row indices those predictions correspond to
    oof_model_names_   which model each column belongs to

The out-of-fold probabilities are honest, unleaked estimates of what the ensemble would say
about data it has not seen -- which is exactly the right reference distribution. Pair them
with the cohort labels at `oof_fit_indices_` and the reference set is free.

NOTHING FAILS SILENTLY HERE
---------------------------
If nannyml is unavailable, this module RAISES, with the exact remediation. It does NOT
`try/except ImportError` into a `logger.warning` -- which is what `_export_evidently` used to
do, telling the operator "Evidently AI not installed. Run: pip install evidently" while
Evidently WAS installed and the real fault was that the API had been deleted. The code
misdiagnosed its own failure, and the drift report was silently never produced, for months.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# The column names CBPE is given. They are OUR names, not nannyml's -- the reference and
# analysis frames are built by us, so we control them.
COL_PROBA = "y_pred_proba"
COL_PRED = "y_pred"
COL_TRUE = "y_true"

DEFAULT_METRICS: tuple[str, ...] = ("roc_auc", "accuracy", "f1", "precision", "recall")

# nannyml warns "The resulting number of chunks is too low" below ~5 chunks, and its sampling
# error estimates degrade on small chunks. 10 chunks is the floor at which the confidence
# bands mean anything.
MIN_CHUNKS = 10


class NannyMLUnavailableError(RuntimeError):
    """nannyml could not be imported. RAISED, never swallowed."""


def _import_nannyml():
    """Import nannyml, or FAIL LOUD with the exact remediation.

    The two ways this fails are BOTH diagnosed explicitly, because the generic
    `except ImportError -> "not installed"` message is how the Evidently export lied about
    itself for months.
    """
    try:
        import nannyml as nml  # type: ignore[import]
    except ImportError as exc:
        raise NannyMLUnavailableError(
            "nannyml is NOT INSTALLED in this environment.\n"
            "\n"
            "Confidence-Based Performance Estimation runs in a SEPARATE environment, because\n"
            "nannyml requires lightgbm<4.6 while the ensemble TRAINS on lightgbm 4.6.0 (a base\n"
            "model). It is deliberately absent from requirements.txt.\n"
            "\n"
            "  python -m venv .venv-drift\n"
            "  .venv-drift/Scripts/pip install -r requirements-drift.txt\n"
            "  .venv-drift/Scripts/pip install -e . --no-deps\n"
            "\n"
            "See requirements-drift.txt and roadmap 6.19."
        ) from exc
    except AttributeError as exc:
        # THE ACTUAL FAILURE SEEN ON THE DEVELOPER'S LAPTOP, 2026-07-13.
        if "np.NaN" in str(exc) or "NaN" in str(exc):
            raise NannyMLUnavailableError(
                f"nannyml is installed but CANNOT IMPORT: {exc}\n"
                "\n"
                "ROOT CAUSE: nannyml OPTIONALLY imports pyspark. pyspark 3.5.x's pandas module\n"
                "calls `np.NaN`, which NumPy REMOVED in 2.0. So on any machine where pyspark is\n"
                "importable, `import nannyml` dies -- and it dies inside pyspark, not nannyml.\n"
                "\n"
                "This is why nannyml never imported on the developer's laptop, and it is not\n"
                "fixed by any nannyml version.\n"
                "\n"
                "FIX: the drift environment (requirements-drift.txt) deliberately does NOT\n"
                "install pyspark. If you need Spark here, first move to pyspark 4.x (which\n"
                "supports NumPy 2) and RE-VERIFY that nannyml still imports."
            ) from exc
        raise
    return nml


@dataclass
class PerformanceEstimate:
    """The result of a CBPE run. Plain data -- no library objects leak out of this module."""

    metrics: list[str]
    chunk_size: int
    n_reference: int
    n_analysis: int
    #: one row per chunk, columns: metric, period, estimate, lower/upper confidence, alert
    estimates: pd.DataFrame
    #: metric -> the estimated value on the ANALYSIS period (mean across analysis chunks)
    analysis_summary: dict[str, float] = field(default_factory=dict)
    #: metric -> True if ANY analysis chunk raised nannyml's alert flag
    alerts: dict[str, bool] = field(default_factory=dict)

    @property
    def any_alert(self) -> bool:
        return any(self.alerts.values())


def build_reference_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred_proba: Sequence[float] | np.ndarray,
    features: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Assemble the LABELLED reference frame CBPE calibrates against.

    Use the ensemble's OUT-OF-FOLD probabilities here, not its in-sample ones. Out-of-fold
    predictions are honest estimates of what the model says about data it has not seen; the
    in-sample ones are optimistic, and calibrating CBPE against them would make every future
    estimate look worse than it is.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    if y_true.shape != y_pred_proba.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred_proba {y_pred_proba.shape} must be the same "
            f"length. A mismatch here silently mis-calibrates every downstream estimate."
        )
    if not np.all((y_pred_proba >= 0.0) & (y_pred_proba <= 1.0)):
        raise ValueError(
            "y_pred_proba contains values outside [0, 1]. CBPE requires CALIBRATED "
            "probabilities; passing raw scores or logits produces confident nonsense."
        )

    frame = pd.DataFrame(
        {
            COL_TRUE: y_true.astype(int),
            COL_PROBA: y_pred_proba,
            COL_PRED: (y_pred_proba >= threshold).astype(int),
        }
    )
    if features is not None:
        frame = pd.concat([features.reset_index(drop=True), frame], axis=1)
    return frame


def build_analysis_frame(
    y_pred_proba: Sequence[float] | np.ndarray,
    features: Optional[pd.DataFrame] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Assemble the UNLABELLED analysis frame -- the new ClinVar release.

    Deliberately has NO y_true column. That is the entire point: the labels do not exist yet.
    """
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    frame = pd.DataFrame(
        {
            COL_PROBA: y_pred_proba,
            COL_PRED: (y_pred_proba >= threshold).astype(int),
        }
    )
    if features is not None:
        frame = pd.concat([features.reset_index(drop=True), frame], axis=1)
    return frame


def estimate_performance(
    reference: pd.DataFrame,
    analysis: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
    chunk_size: Optional[int] = None,
) -> PerformanceEstimate:
    """Estimate performance on UNLABELLED `analysis` data, calibrated on labelled `reference`.

    Raises
    ------
    NannyMLUnavailableError
        If nannyml cannot be imported -- with the exact remediation. NEVER swallowed.
    ValueError
        If the frames do not carry what CBPE needs. Fail at the door, not deep inside a
        third-party library with a message about a column name we chose ourselves.
    """
    nml = _import_nannyml()

    missing_ref = {COL_TRUE, COL_PROBA, COL_PRED} - set(reference.columns)
    if missing_ref:
        raise ValueError(
            f"reference frame is missing {sorted(missing_ref)}. Build it with "
            f"build_reference_frame() -- the reference MUST carry labels, or CBPE has nothing "
            f"to calibrate against."
        )
    missing_ana = {COL_PROBA, COL_PRED} - set(analysis.columns)
    if missing_ana:
        raise ValueError(
            f"analysis frame is missing {sorted(missing_ana)}. Build it with "
            f"build_analysis_frame()."
        )
    if COL_TRUE in analysis.columns:
        raise ValueError(
            "analysis frame CONTAINS y_true. It must not: the whole purpose of "
            "Confidence-Based Performance Estimation is to estimate performance where the "
            "labels DO NOT EXIST YET. If you have labels, measure the metric directly -- an "
            "estimate is strictly worse than a measurement."
        )

    if chunk_size is None:
        # Size chunks so the confidence bands mean something. nannyml warns below ~5 chunks;
        # 10 is the floor at which its sampling-error estimate is worth reading.
        chunk_size = max(1, min(len(reference), len(analysis)) // MIN_CHUNKS)

    n_chunks = min(len(reference), len(analysis)) // max(chunk_size, 1)
    if n_chunks < MIN_CHUNKS:
        logger.warning(
            "CBPE will produce only ~%d chunks (chunk_size=%d, reference=%d, analysis=%d). "
            "nannyml's sampling-error estimate degrades below %d chunks and the confidence "
            "bands become unreliable. Consider a smaller chunk_size or more data.",
            n_chunks, chunk_size, len(reference), len(analysis), MIN_CHUNKS,
        )

    estimator = nml.CBPE(
        y_pred_proba=COL_PROBA,
        y_pred=COL_PRED,
        y_true=COL_TRUE,
        problem_type="classification_binary",
        metrics=list(metrics),
        chunk_size=chunk_size,
    )
    estimator.fit(reference)
    result = estimator.estimate(analysis)
    df = result.to_df()

    # nannyml returns a MultiIndex column frame: (metric, field). Flatten it into something a
    # human -- and a JSON report -- can read, and surface the ANALYSIS period only, because the
    # reference period is by definition the thing we calibrated on.
    rows: list[dict[str, Any]] = []
    summary: dict[str, float] = {}
    alerts: dict[str, bool] = {}

    period = df[("chunk", "period")] if ("chunk", "period") in df.columns else None
    is_analysis = (period == "analysis") if period is not None else pd.Series(True, index=df.index)

    for metric in metrics:
        if (metric, "value") not in df.columns:
            logger.warning("CBPE returned no '%s' column; skipping it.", metric)
            continue
        vals = df.loc[is_analysis, (metric, "value")].astype(float)
        alert_col = (metric, "alert")
        alert = bool(df.loc[is_analysis, alert_col].any()) if alert_col in df.columns else False
        summary[metric] = float(vals.mean()) if len(vals) else float("nan")
        alerts[metric] = alert

        for idx in df.index[is_analysis]:
            rows.append(
                {
                    "metric": metric,
                    "chunk": df.loc[idx, ("chunk", "key")] if ("chunk", "key") in df.columns else idx,
                    "period": "analysis",
                    "estimate": float(df.loc[idx, (metric, "value")]),
                    "lower": float(df.loc[idx, (metric, "lower_confidence_boundary")])
                    if (metric, "lower_confidence_boundary") in df.columns else float("nan"),
                    "upper": float(df.loc[idx, (metric, "upper_confidence_boundary")])
                    if (metric, "upper_confidence_boundary") in df.columns else float("nan"),
                    "alert": bool(df.loc[idx, alert_col]) if alert_col in df.columns else False,
                }
            )

    estimates = pd.DataFrame(rows)

    if alerts and any(alerts.values()):
        flagged = sorted(m for m, a in alerts.items() if a)
        logger.error(
            "CBPE ALERT: estimated performance on the UNLABELLED analysis data has crossed "
            "nannyml's threshold for %s. Estimated values: %s. The model may be degrading on "
            "real data. Investigate BEFORE the labels arrive.",
            ", ".join(flagged),
            {m: round(v, 4) for m, v in summary.items()},
        )
    else:
        logger.info(
            "CBPE: no alerts. Estimated performance on unlabelled data: %s",
            {m: round(v, 4) for m, v in summary.items()},
        )

    return PerformanceEstimate(
        metrics=list(metrics),
        chunk_size=chunk_size,
        n_reference=len(reference),
        n_analysis=len(analysis),
        estimates=estimates,
        analysis_summary=summary,
        alerts=alerts,
    )
