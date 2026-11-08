"""Research operating points on the SELECTED predictor's score, bound to the completed run they were chosen on.

Author: Monzia Moodie

Package 1, set 1c (2026-11-08). This replaces the private threshold search of scripts/calibrate_thresholds.py, which
(measured on a synthetic split of 20,000 rows, prevalence 0.15) returned the TOP score for every target -- its loop
walked `reversed(precision[:-1])` and stopped at the first, i.e. the HIGHEST, qualifying threshold while its docstring
promised the lowest -- so the "Pathogenic" and "Likely pathogenic" boundaries coincided at 1.0 and the second tier was
empty. It also scored the logistic meta-learner (not the selected predictor), fitted Platt and temperature scaling,
chose between them and placed thresholds on the SAME rows, fell back silently to the best-precision point when a
target was unreachable, and wrote thresholds on a calibrated scale that the serving score never passes through.

WHAT THIS MODULE DOES INSTEAD
  * Input: one split of a COMPLETED run (prediction_artifacts.load_completed_split_predictions) -- the stored convex
    blend's `selected_score`, the labels it was evaluated against, and the identities of the predictor, run, file and
    rows. Nothing is re-scored.
  * Selection: the repository's exact operating-point machinery (evaluation.thresholds): the exact O(n log n) sweep
    over every achievable threshold and Objective A, "the most sensitive operating point whose positive predictive
    value clears a floor", with its declared total order. No grid, no traversal-order tie rule, no fallback: an
    unreachable floor is a refusal.
  * Score stage: the RAW selected score -- the value serving returns. The positive predictive value and the negative
    predictive value at a threshold depend only on which rows clear it, and a strictly increasing transformation
    (Platt scaling with a positive slope, temperature scaling) never changes which rows clear the image of the
    threshold. A calibration transformation therefore cannot change these operating points; it can only change the
    number written down. Thresholds are written on the stage serving produces, so no consumer can apply them to a
    different stage. (Probability calibration itself is a separate question with its own estimand.)
  * The negative side is the same machinery on (1 - label, 1 - score): its score space is declared as
    `one_minus_selected_score` and a consumer evaluates it on `1.0 - selected_score` computed in binary64, exactly as
    here.
  * Bands carry neutral names. They are research operating points, NOT ACMG/AMP classifications: response schema
    3.0.0 removed the five-tier labels from serving because a classifier threshold is not an evidence-strength
    calibration. Every operating point carries the certification blockers its own selection implies (chosen and
    reported on the same rows; no held-out validation).

The record is strict JSON with its own digest (`record_sha256`); `load_decision_policy` refuses any record whose digest,
schema or band geometry does not hold, and `DecisionPolicy.require_predictor` refuses use with another predictor.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.serialization import dump_strict_json
from genomic_variant_classifier.evaluation.thresholds import (
    OperatingPointOutcome,
    ThresholdOperator,
    select_max_sensitivity_at_ppv_floor,
    sweep_thresholds,
)
from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest, require_sha256

logger = logging.getLogger(__name__)

__all__ = [
    "BAND_ORDER",
    "DECISION_POLICY_SCHEMA",
    "DecisionPolicy",
    "DecisionPolicyRefused",
    "NEGATIVE_SCORE_SPACE",
    "POSITIVE_SCORE_SPACE",
    "build_decision_policy",
    "load_decision_policy",
    "write_decision_policy",
]

DECISION_POLICY_SCHEMA = "gvc.decision-policy.v1"
POSITIVE_SCORE_SPACE = "selected_score"
NEGATIVE_SCORE_SPACE = "one_minus_selected_score"
SCORE_STAGE = "selected_convex_blend_positive_probability"
POLICY_STATUS = "research_operating_points_not_clinically_qualified"

#: (operating point, score space, the band it opens). Order is the classification precedence within each side.
_POINTS = (
    ("positive_high", POSITIVE_SCORE_SPACE, "positive_high_ppv"),
    ("positive_moderate", POSITIVE_SCORE_SPACE, "positive_moderate_ppv"),
    ("negative_high", NEGATIVE_SCORE_SPACE, "negative_high_npv"),
    ("negative_moderate", NEGATIVE_SCORE_SPACE, "negative_moderate_npv"),
)
BAND_ORDER = ("positive_high_ppv", "positive_moderate_ppv", "indeterminate", "negative_moderate_npv",
              "negative_high_npv")
_OPERATORS = {ThresholdOperator.GREATER_OR_EQUAL.value: ThresholdOperator.GREATER_OR_EQUAL,
              ThresholdOperator.GREATER.value: ThresholdOperator.GREATER}


class DecisionPolicyRefused(ContractError):
    """No policy is produced: a floor is unreachable, the floors are inconsistent, or the bands would overlap.
    `outcomes` holds every operating-point outcome that WAS computed, so the refusal can be reported in full."""

    def __init__(self, message: str, outcomes: Mapping[str, dict] | None = None) -> None:
        super().__init__(message)
        self.outcomes = dict(outcomes or {})


def _floor(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{name} must be a finite number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ContractError(f"{name} must lie in (0, 1], got {value}")
    return value


def _space_values(scores: np.ndarray, space: str) -> np.ndarray:
    if space == POSITIVE_SCORE_SPACE:
        return scores
    if space == NEGATIVE_SCORE_SPACE:
        return 1.0 - scores
    raise ContractError(f"unknown score space {space!r}")


def _member(values: np.ndarray, threshold: float, operator: str) -> np.ndarray:
    if operator == ThresholdOperator.GREATER_OR_EQUAL.value:
        return values >= threshold
    if operator == ThresholdOperator.GREATER.value:
        return values > threshold
    raise ContractError(f"unknown threshold operator {operator!r}")


def _require_scores(scores: Any) -> np.ndarray:
    p = np.asarray(scores)
    if p.ndim != 1 or p.dtype != np.float64 or not len(p):
        raise ContractError("a nonempty one-dimensional float64 selected-score vector is required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("selected scores must be finite and in [0, 1]; nothing is clipped")
    return p


def _point_of(outcome: OperatingPointOutcome) -> dict:
    parameters = outcome.metrics.parameters
    return {"threshold": parameters.threshold, "threshold_binary64_hex": float(parameters.threshold).hex(),
            "operator": parameters.operator.value}


def _check_geometry(points: Mapping[str, dict]) -> None:
    """The bands partition [0, 1]: each side's high band lies inside its moderate band, and the two sides never meet.

    The positive region {p >= t} (or > t) is an up-set; the negative region {fl(1 - p) >= u} is a down-set because
    correctly rounded subtraction is monotone. They meet exactly when the SMALLEST positive-region score is a
    negative-region member, so that one point decides it -- for every possible score, not only the fitted rows.
    """
    for side, space in (("positive", POSITIVE_SCORE_SPACE), ("negative", NEGATIVE_SCORE_SPACE)):
        high, moderate = points[f"{side}_high"], points[f"{side}_moderate"]
        smallest_high = (high["threshold"] if high["operator"] == ">=" else
                         float(np.nextafter(high["threshold"], 2.0)))
        if not bool(_member(np.array([smallest_high]), moderate["threshold"], moderate["operator"])[0]):
            raise DecisionPolicyRefused(f"the {side} high band is not inside the {side} moderate band")
    positive, negative = points["positive_moderate"], points["negative_moderate"]
    smallest_positive = (positive["threshold"] if positive["operator"] == ">=" else
                         float(np.nextafter(positive["threshold"], 2.0)))
    if smallest_positive <= 1.0 and bool(_member(_space_values(np.array([smallest_positive]), NEGATIVE_SCORE_SPACE),
                                                 negative["threshold"], negative["operator"])[0]):
        raise DecisionPolicyRefused(
            f"the positive and negative bands overlap: selected_score {positive['operator']} "
            f"{positive['threshold']!r} clears the positive_moderate floor and 1 - selected_score "
            f"{negative['operator']} {negative['threshold']!r} clears the negative_moderate floor, so a score "
            "between them would be both. Each side's predictive value is a property of its whole call set, so "
            "assigning the overlap to either side (or to neither) changes a call set and voids its selected "
            "operating point; no tie rule is chosen here. Raise the moderate floors, or see the open ruling "
            "DECISION-POLICY-OVERLAP-1.")


def build_decision_policy(bound: Any, *, positive_high: float = 0.95, positive_moderate: float = 0.80,
                          negative_high: float = 0.95, negative_moderate: float = 0.80) -> dict:
    """The decision-policy record for one bound split (prediction_artifacts.BoundSplitPredictions), or a refusal.

    Floors are predictive values: a positive floor is a minimum positive predictive value of the calls the band
    makes; a negative floor is a minimum negative predictive value (the positive predictive value of `1 - label`
    under `1 - score`). The moderate floor of a side may not exceed its high floor.
    """
    floors = {"positive_high": _floor(positive_high, "positive_high"),
              "positive_moderate": _floor(positive_moderate, "positive_moderate"),
              "negative_high": _floor(negative_high, "negative_high"),
              "negative_moderate": _floor(negative_moderate, "negative_moderate")}
    for side in ("positive", "negative"):
        if floors[f"{side}_moderate"] > floors[f"{side}_high"]:
            raise ContractError(f"the {side} moderate floor exceeds the {side} high floor")
    identity = bound.identity()
    require_sha256(identity["combiner_sha256"], "combiner_sha256")
    scores = _require_scores(bound.scores)
    labels = np.asarray(bound.labels)
    if labels.shape != scores.shape or labels.dtype != np.int64 or not np.isin(labels, (0, 1)).all():
        raise ContractError("one int64 label in {0, 1} per score required")
    population = EvaluationPopulation.full(
        len(scores), scope=f"{bound.role}_split_predictions_of_a_completed_run",
        source_id=f"gvc-run-predictions:sha256:{identity['predictions_sha256']}")
    sweeps = {POSITIVE_SCORE_SPACE: sweep_thresholds(labels, scores, population=population),
              NEGATIVE_SCORE_SPACE: sweep_thresholds(1 - labels, _space_values(scores, NEGATIVE_SCORE_SPACE),
                                                     population=population)}
    outcomes: dict[str, dict] = {}
    points: dict[str, dict] = {}
    refused = []
    for name, space, band in _POINTS:
        outcome = select_max_sensitivity_at_ppv_floor(sweeps[space], positive_predictive_value_floor=floors[name])
        outcomes[name] = {"score_space": space, "floor": floors[name], "opens_band": band,
                          "outcome": outcome.to_dict()}
        if outcome.is_ok:
            points[name] = {"score_space": space, **_point_of(outcome)}
            outcomes[name].update(points[name])
        else:
            refused.append(f"{name} (floor {floors[name]}): {outcome.reason}")
    if refused:
        raise DecisionPolicyRefused("no decision policy: " + "; ".join(refused), outcomes)
    try:
        _check_geometry(points)
    except DecisionPolicyRefused as exc:
        raise DecisionPolicyRefused(str(exc), outcomes) from None
    blockers = sorted({b for o in outcomes.values() for b in o["outcome"]["certification_blockers"]})
    record = {
        "schema": DECISION_POLICY_SCHEMA,
        "status": POLICY_STATUS,
        "predictor": dict(bound.predictor),
        "fitted_on": {**identity, "population": {"scope": population.scope, "n": population.n,
                                                 "membership_fingerprint": population.membership_fingerprint},
                      "n_positive": int(labels.sum())},
        "score": {"column": "selected_score", "stage": SCORE_STAGE, "transform": "identity",
                  "negative_space_definition": "1.0 - selected_score, computed in IEEE 754 binary64",
                  "why_no_calibration_transform": (
                      "predictive values at a threshold depend only on which rows clear it, and a strictly increasing "
                      "transformation never changes that set; thresholds are written on the stage serving returns")},
        "selection": {"objective": "maximize_sensitivity_at_positive_predictive_value_floor",
                      "sweep": "exact_achievable_thresholds", "fallback": "none: an unreachable floor refuses"},
        "operating_points": outcomes,
        "bands": [
            {"band": "positive_high_ppv", "rule": "positive_high point clears"},
            {"band": "positive_moderate_ppv", "rule": "positive_moderate clears and positive_high does not"},
            {"band": "indeterminate", "rule": "neither side's moderate point clears"},
            {"band": "negative_moderate_npv", "rule": "negative_moderate clears and negative_high does not"},
            {"band": "negative_high_npv", "rule": "negative_high point clears"}],
        "certification_blockers": blockers,
        "clinical_qualification": "none: not ACMG/AMP evidence and not an evidence-strength calibration",
    }
    record["record_sha256"] = canonical_digest(DECISION_POLICY_SCHEMA, record)
    return record


def write_decision_policy(record: Mapping[str, Any], path: Path | str) -> str:
    """Atomic strict-JSON write (temporary sibling, fsync, rename). Returns the SHA-256 of the bytes written."""
    path = Path(path)
    payload = (dump_strict_json(dict(record), artifact=str(path)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, eq=False)
class DecisionPolicy:
    """A loaded, verified decision-policy record."""

    record: Mapping[str, Any]
    file_sha256: str

    @property
    def combiner_sha256(self) -> str:
        return self.record["predictor"]["combiner_sha256"]

    @property
    def record_sha256(self) -> str:
        return self.record["record_sha256"]

    def point(self, name: str) -> dict:
        return self.record["operating_points"][name]

    def require_predictor(self, combiner_sha256: str) -> None:
        """Refuse use with any predictor other than the one the operating points were chosen for."""
        if combiner_sha256 != self.combiner_sha256:
            raise ContractError(f"this decision policy was chosen for predictor {self.combiner_sha256}, not "
                                f"{combiner_sha256}; its operating points say nothing about another predictor")

    def classify(self, scores: Any) -> np.ndarray:
        """The band of each selected score, as an object array of BAND_ORDER names."""
        p = _require_scores(scores)
        bands = np.full(len(p), "indeterminate", dtype=object)
        for name, space, band in reversed(_POINTS):          # moderate first, high overrides on its own side
            point = self.point(name)
            bands[_member(_space_values(p, space), point["threshold"], point["operator"])] = band
        return bands


def load_decision_policy(path: Path | str) -> DecisionPolicy:
    """Read, verify and return a decision policy; refuse anything that does not verify."""
    data = Path(path).read_bytes()
    try:
        record = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: not a JSON decision policy ({exc})") from exc
    if not isinstance(record, dict) or record.get("schema") != DECISION_POLICY_SCHEMA:
        raise ContractError(f"{path}: not a {DECISION_POLICY_SCHEMA} record")
    claimed = record.get("record_sha256")
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if claimed != canonical_digest(DECISION_POLICY_SCHEMA, body):
        raise ContractError(f"{path}: the record does not reproduce its record_sha256 (edited or truncated)")
    require_sha256((record.get("predictor") or {}).get("combiner_sha256"), "combiner_sha256")
    points = {}
    for name, space, _band in _POINTS:
        point = (record.get("operating_points") or {}).get(name)
        if not isinstance(point, dict) or point.get("score_space") != space or point.get("operator") not in _OPERATORS:
            raise ContractError(f"{path}: operating point {name} is missing or malformed")
        threshold = point.get("threshold")
        try:
            exact = float.fromhex(point.get("threshold_binary64_hex"))
        except (TypeError, ValueError):
            exact = None
        if type(threshold) is not float or not 0.0 <= threshold <= 1.0 or exact != threshold:
            raise ContractError(f"{path}: operating point {name} has no exact binary64 threshold")
        points[name] = point
    _check_geometry(points)
    return DecisionPolicy(record=record, file_sha256=hashlib.sha256(data).hexdigest())
