"""The derived execution assessment, the release decision and the reference-evaluator boundary (owner ruling 2026-10-09).

THE BOUNDARY IS IN EXECUTION, NOT IN PRESENTATION
=================================================
"Withheld" must not mean merely hidden in the final report: a feasibility path that loaded reference labels, computed recovery and then
hid it could still leak reference membership through logs, explanation tables or cached intermediates. So the components are separate:

    scientific worker     frozen inputs, method configuration, qualified environment -> scores, exposure outcomes, coverage, trace
                          (it receives NO reference input at all)
    admission (here)      the admitted run intent + the worker's evidence -> a DERIVED ExecutionAssessment and release decision
    reference evaluator   (here) opened ONLY when the admitted intents and the derived assessment permit evaluation
    report renderer       admitted evidence and evaluation results -> human-readable report

DERIVED, NEVER ASSERTED
=======================
No caller supplies a success flag. assess_execution takes the admitted intent, the frozen PairPlan (its digest must be the intent's),
the recorder directory and the exact score bytes, and runs the EXISTING validators -- read_exposure_trace, classify_exposures,
coverage_report -- plus the coverage rule (exposure_outcomes.score_coverage). evaluate() does the same itself: it accepts no assessment
object, so a caller cannot hand it internally consistent claims that were never derived from the evidence. Whether reference evidence
may be opened is read from the ONE release table (exposure_outcomes.endpoint_release), with the stage taken from the admitted intent.

The derived fields (ruling 2026-10-09 section 5) are kept apart, never collapsed:

    execution_integrity       passed | refused
    method_completion         complete | incomplete | not_determined   (not_determined: integrity refused, so completion is unknown --
                                                                         a refinement of the ruling's complete_or_incomplete, recorded)
    evaluation_permission     permitted | withheld
    scientific_interpretation not_established_by_these_fields

"evaluated" means the specified calculation occurred -- never "validated". A callback test verifies control flow; it does not prove
filesystem isolation of the worker.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import (
    PRIMARY_COMPLETE, PRIMARY_REFUSED, PRIMARY_WITHHELD, PairPlan, classify_exposures, coverage_report, endpoint_release,
    exposure_trace_sha256, read_exposure_trace, score_coverage)
from genomic_variant_classifier.inference.ranking import read_score_matrix
from genomic_variant_classifier.inference.run_intent import RunIntent, Stage, admit_evaluation_intent, admit_run_intent

logger = logging.getLogger(__name__)

__all__ = ["ASSESSMENT_SCHEMA", "ExecutionAssessment", "assess_execution", "release_decision", "evaluate", "SCIENTIFIC_INTERPRETATION"]

ASSESSMENT_SCHEMA = "gvc.execution-assessment"
SCIENTIFIC_INTERPRETATION = "not_established_by_these_fields"
_COMPLETION = {PRIMARY_COMPLETE: "complete", PRIMARY_WITHHELD: "incomplete", PRIMARY_REFUSED: "not_determined"}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


@dataclass(frozen=True)
class ExecutionAssessment:
    """What admission DERIVED about one execution. Internally consistent by construction (checked here); never accepted as input by the
    evaluator, which re-derives it."""

    run_intent_sha256: str
    stage: str
    plan_sha256: str
    score_artifact_sha256: str
    exposure_trace_sha256: str
    outcomes: tuple                 # ((exposure, status), ...) sorted by exposure
    primary_status: str
    reasons: tuple
    exposure_completion: str | None
    execution_integrity: str
    method_completion: str
    evaluation_permission: str
    release: tuple                  # endpoint_release(primary_status, stage) as sorted (key, value) pairs

    def __post_init__(self) -> None:
        _require(self.primary_status in _COMPLETION, "assessment_primary_status")
        _require(self.execution_integrity == ("refused" if self.primary_status == PRIMARY_REFUSED else "passed"), "assessment_inconsistent")
        _require(self.method_completion == _COMPLETION[self.primary_status], "assessment_inconsistent")
        _require(bool(self.reasons) == (self.primary_status != PRIMARY_COMPLETE), "assessment_inconsistent")
        release = dict(self.release)
        _require(release.get("stage") == self.stage and release.get("primary_status") == self.primary_status, "assessment_inconsistent")
        _require(self.evaluation_permission == ("permitted" if release["reference_recovery"] == "released" else "withheld"),
                 "assessment_inconsistent")

    def as_dict(self) -> dict:
        return {"schema": ASSESSMENT_SCHEMA, "schema_version": 1, "run_intent_sha256": self.run_intent_sha256, "stage": self.stage,
                "plan_sha256": self.plan_sha256, "score_artifact_sha256": self.score_artifact_sha256,
                "exposure_trace_sha256": self.exposure_trace_sha256, "outcomes": dict(self.outcomes),
                "primary_status": self.primary_status, "reasons": list(self.reasons), "exposure_completion": self.exposure_completion,
                "execution_integrity": self.execution_integrity, "method_completion": self.method_completion,
                "evaluation_permission": self.evaluation_permission, "scientific_interpretation": SCIENTIFIC_INTERPRETATION,
                "release": dict(self.release)}

    def render(self) -> bytes:
        return (json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.render()).hexdigest()


def assess_execution(intent, intent_sha256: str, plan: PairPlan, trace_dir, score_bytes: bytes) -> ExecutionAssessment:
    """DERIVE the assessment of one execution from its admitted evidence (ruling 2026-10-09 section 3).

    intent / intent_sha256: an ADMITTED RunIntent and its admitted digest (run_intent.admit_run_intent); plan: the frozen PairPlan, whose
    digest must be the intent's; trace_dir: the exposure recorder's directory; score_bytes: the exact score artifact (mat_p.tsv).
    Malformed or inconsistent evidence raises InferenceError (the existing validators' refusals, unchanged); well-formed evidence yields
    an assessment whose status the coverage rule derived."""
    _require(type(intent) is RunIntent and intent.sha256 == intent_sha256, "assessment_intent")
    _require(type(plan) is PairPlan and plan.sha256 == intent.plan_sha256, "assessment_plan_not_the_intents",
             "the plan frozen before execution is the one the intent binds")
    _require(type(score_bytes) is bytes, "score_artifact_bytes")
    events = read_exposure_trace(trace_dir)
    trace_digest = exposure_trace_sha256(trace_dir)
    statuses = classify_exposures(plan.exposures, plan.exclusions_dict(), plan.usable_dict(), events, predicted_support=plan.support_dict())
    grid = plan.as_dict()
    primary, reasons = score_coverage(grid, statuses, read_score_matrix(score_bytes))
    # The completion COUNT is reported from coverage_report (one definition of exposure completion). The coverage rule and
    # coverage_report agree on every non-refused status by construction -- measured on 400 random plans by the property test
    # test_PROPERTY_the_coverage_rule_agrees_with_coverage_report_on_consistent_evidence; a runtime cross-check here could never fire
    # (a seeded-defect run of 2026-10-09 showed it survived removal), so it is not kept as dead code.
    completion = None if "no_eligible_exposures" in reasons else coverage_report(grid, statuses)["exposure_completion"]
    release = endpoint_release(primary, intent.stage.value)
    return ExecutionAssessment(
        intent_sha256, intent.stage.value, plan.sha256, hashlib.sha256(score_bytes).hexdigest(), trace_digest,
        tuple(sorted((x, s["status"].value) for x, s in statuses.items())), primary, tuple(reasons), completion,
        "refused" if primary == PRIMARY_REFUSED else "passed", _COMPLETION[primary],
        "permitted" if release["reference_recovery"] == "released" else "withheld", tuple(sorted(release.items())))


def release_decision(*, run_intent_bytes: bytes, admitted_run_intent_sha256: str, plan: PairPlan, trace_dir, score_bytes: bytes) -> dict:
    """The derived release decision of a run under ITS OWN admitted intent. It has no reference parameter: the scientific worker and this
    admission step never receive reference evidence (ruling 2026-10-09 section 1). For a feasibility intent, reference recovery and
    Delta H(20) are withheld whatever the completion."""
    intent = admit_run_intent(run_intent_bytes, admitted_run_intent_sha256)
    assessment = assess_execution(intent, admitted_run_intent_sha256, plan, trace_dir, score_bytes)
    return {"run_intent_sha256": admitted_run_intent_sha256, "stage": intent.stage.value, "assessment": assessment.as_dict(),
            "assessment_sha256": assessment.sha256, "reference_evaluation": None}


def evaluate(*, evaluation_intent_bytes: bytes, admitted_evaluation_intent_sha256: str, run_intent_bytes: bytes,
             admitted_run_intent_sha256: str, plan: PairPlan, trace_dir, score_bytes: bytes, read_reference, calculate_endpoint) -> dict:
    """THE REFERENCE-EVALUATOR BOUNDARY (ruling 2026-10-09 sections 2-3). Every check below happens BEFORE `read_reference` can be
    called, and it is called only when the release table permits a reference recovery for the admitted evaluation stage:

        both intents admitted against their persisted digests and this implementation's release policy
        the evaluation intent names THIS computation, and THESE exact score bytes
        a confirmatory evaluation of a feasibility computation lists that computation in informed_by (history disclosed)
        the assessment is RE-DERIVED here from the evidence (never accepted from a caller)
        refused evidence -> "refused"; feasibility -> "withheld_feasibility_stage"; incomplete -> "withheld_incomplete_exposures"
        otherwise: read the reference, require its digest to be the intent's, compute the endpoint -> "evaluated"

    read_reference() -> bytes; calculate_endpoint(score_bytes, reference_bytes) -> dict."""
    evaluation = admit_evaluation_intent(evaluation_intent_bytes, admitted_evaluation_intent_sha256)
    run = admit_run_intent(run_intent_bytes, admitted_run_intent_sha256)
    _require(evaluation.run_intent_sha256 == admitted_run_intent_sha256, "evaluation_run_intent_mismatch")
    if run.stage is Stage.FEASIBILITY and evaluation.stage is Stage.CONFIRMATORY:
        _require(admitted_run_intent_sha256 in evaluation.informed_by, "evaluation_history_undisclosed",
                 "a confirmatory evaluation of a feasibility computation lists it in informed_by")
    _require(type(score_bytes) is bytes and hashlib.sha256(score_bytes).hexdigest() == evaluation.score_artifact_sha256,
             "evaluation_score_bytes_mismatch", "refused before any reference access")
    _require(callable(read_reference) and callable(calculate_endpoint), "evaluation_callables")
    assessment = assess_execution(run, admitted_run_intent_sha256, plan, trace_dir, score_bytes)
    release = endpoint_release(assessment.primary_status, evaluation.stage.value)
    decision = {"evaluation_intent_sha256": admitted_evaluation_intent_sha256, "run_intent_sha256": admitted_run_intent_sha256,
                "assessment_sha256": assessment.sha256, "score_artifact_sha256": evaluation.score_artifact_sha256,
                "stage": evaluation.stage.value, "release": release, "scientific_interpretation": SCIENTIFIC_INTERPRETATION,
                "reference_sha256": None, "endpoint": None}
    if release["reference_recovery"] != "released":
        return {**decision, "status": release["reference_recovery"]}
    # The FIRST point at which reference evidence may be opened.
    reference = read_reference()
    _require(type(reference) is bytes and hashlib.sha256(reference).hexdigest() == evaluation.reference_sha256,
             "evaluation_reference_bytes_mismatch")
    endpoint = calculate_endpoint(score_bytes, reference)
    _require(type(endpoint) is dict, "evaluation_endpoint_type")
    return {**decision, "status": "evaluated", "reference_sha256": evaluation.reference_sha256, "endpoint": endpoint}
