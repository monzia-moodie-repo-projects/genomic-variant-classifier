"""The derived execution assessment, the release decision and the reference-evaluator boundary (owner ruling 2026-10-09).

The six FORBIDDEN-CALL cases the ruling requires use a reference loader that RAISES if it is ever called: complete feasibility,
incomplete confirmatory and refused evidence never call it; changed score bytes are refused before any reference access; a wrong
reference digest is refused after the read and before any endpoint; matching bindings compute the endpoint exactly once. A callback
test verifies CONTROL FLOW; it does not prove filesystem isolation of the worker (the worker and release_decision have no reference
parameter at all, which is checked by signature).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from genomic_variant_classifier.inference import evaluation_boundary as eb
from genomic_variant_classifier.inference import run_intent as ri
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import EXPOSURE_RECORDER_VERSION, PairPlan, release_policy_sha256

GENES = ("A1", "A2", "A3", "A4")
BURDEN = {"A1": "0x1p-4", "A2": "0x1p-3", "A3": "0x1p-2", "A4": "0x1p-1"}
PLAN = PairPlan(("X1", "X2", "X3"), GENES,
                frozenset({(g, "X1") for g in GENES} | {(g, "X2") for g in GENES[:3]}),
                (("X3", "fewer_than_2_valid_genes"),), (("X1", 4, 4), ("X2", 4, 3), ("X3", 4, 1)),
                (("X1", GENES), ("X2", GENES[:3])))
VALID = dict(pi0a="0x1.8p-1", pi0b="0x1.cp-1", wg1="0x1p-3", wg2="0x1p-4", wg3="0x1.4p-1", wg_sum="0x1.ep-1")
SCORES = {("A1", "X1"): "0x1p-5", ("A2", "X1"): "0x1p-6", ("A3", "X1"): "0x1p-2", ("A4", "X1"): "0x1.8p-2",
          ("A1", "X2"): "0x1p-7", ("A2", "X2"): "0x1p-3", ("A3", "X2"): "0x1.4p-2"}
REFERENCE = b"A2\n"


def reason(fn):
    with pytest.raises(InferenceError) as exc:
        fn()
    return exc.value.code


def event(x, outcome, hits, n_valid, burden=None, **values):
    doc = {"exposure_id": x, "outcome": outcome, "last_guard_reached": hits, "n_trans": 4, "n_valid": n_valid,
           "pi0a": None, "pi0b": None, "wg1": None, "wg2": None, "wg3": None, "wg_sum": None, "burden_input": burden,
           "observation_kind": "actual_call_trace", "recorder_version": EXPOSURE_RECORDER_VERSION}
    doc.update(values)
    return json.dumps(doc, separators=(",", ":"))


def write_run(root, x2="scored", scores=SCORES):
    """A worker's evidence: the recorder directory and the score artifact. x2 selects X2's outcome."""
    trace = root / "trace_exposures"
    trace.mkdir(parents=True)
    lines = [event("X1", "scored", 4, 4, "burden-0001", **VALID)]
    if x2 == "scored":
        lines.append(event("X2", "scored", 4, 3, "burden-0002", **VALID))
    elif x2 == "mixture":
        lines.append(event("X2", "mixture_estimate_invalid", 3, 3, "burden-0002", pi0a="-0x1p-7", pi0b="0x1.cp-1"))
    elif x2 == "abnormal":
        lines.append(event("X2", "abnormal_exit", 3, 3, "burden-0002", pi0a="0x1.8p-1", pi0b="0x1.cp-1"))
    lines.append(event("X3", "fewer_than_2_valid_genes", 2, 1))
    (trace / "exposures.jsonl").write_bytes("".join(x + "\n" for x in lines).encode("ascii"))
    (trace / "burden-0001.tsv").write_bytes("".join("{}\t{}\n".format(g, BURDEN[g]) for g in GENES).encode("ascii"))
    if x2 != "missing":
        (trace / "burden-0002.tsv").write_bytes("".join("{}\t{}\n".format(g, BURDEN[g]) for g in GENES[:3]).encode("ascii"))
    cells = {(g, x): scores.get((g, x), "NA") for x in PLAN.exposures for g in GENES}
    if x2 in ("mixture", "abnormal", "missing"):
        cells.update({(g, "X2"): "NA" for g in GENES})
    score_bytes = "".join("{}\t{}\t{}\n".format(g, x, v) for (g, x), v in sorted(cells.items())).encode("ascii")
    return trace, score_bytes


def run_bytes(root, stage=ri.Stage.FEASIBILITY, plan=PLAN):
    """Seal a run intent BEFORE execution; return its persisted bytes and the digest recorded at sealing."""
    intent = ri.RunIntent("run-1", stage, "c" * 64, plan.sha256, (ri.InputIdentity("burden", "1" * 64, 10),), "e" * 64, "a" * 40,
                          release_policy_sha256(), "Prominent asthma genes are known to the analyst.", ())
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("run_intent_" + stage.value + ".json")
    digest = ri.seal_intent(path, intent)
    return path.read_bytes(), digest


def seal_evaluation(root, run_digest, score_bytes, stage=ri.Stage.CONFIRMATORY, reference=REFERENCE, informed_by=None):
    ev = ri.EvaluationIntent("eval-1", stage, run_digest, hashlib.sha256(score_bytes).hexdigest(),
                             None if stage is ri.Stage.FEASIBILITY else hashlib.sha256(reference).hexdigest(), release_policy_sha256(),
                             "The feasibility outputs were inspected before this evaluation was specified.",
                             tuple(sorted({run_digest} if informed_by is None else informed_by)))
    path = root / ("evaluation_intent_" + stage.value + ".json")
    digest = ri.seal_intent(path, ev)
    return path.read_bytes(), digest


class Loader:
    """A reference loader that counts its calls and, when forbidden, RAISES if called at all."""

    def __init__(self, payload=REFERENCE, forbidden=False):
        self.payload, self.forbidden, self.calls = payload, forbidden, 0

    def __call__(self):
        self.calls += 1
        if self.forbidden:
            raise AssertionError("reference access was forbidden")
        return self.payload


class Endpoint:
    def __init__(self, forbidden=False, result=None):
        self.forbidden, self.calls, self.seen, self.result = forbidden, 0, None, result

    def __call__(self, score_bytes, reference_bytes):
        self.calls += 1
        if self.forbidden:
            raise AssertionError("the endpoint must not be computed")
        self.seen = (score_bytes, reference_bytes)
        return {"delta_h20": 1} if self.result is None else self.result


def evaluate(tmp_path, *, run_stage=ri.Stage.FEASIBILITY, eval_stage=ri.Stage.CONFIRMATORY, x2="scored", loader=None, endpoint=None,
             informed_by=None, tamper_scores=False, reference=REFERENCE):
    trace, score_bytes = write_run(tmp_path / "w", x2=x2)
    run_raw, run_digest = run_bytes(tmp_path, run_stage)
    ev_raw, ev_digest = seal_evaluation(tmp_path, run_digest, score_bytes, eval_stage, reference=reference, informed_by=informed_by)
    if tamper_scores:
        score_bytes = score_bytes.replace(b"0x1p-5", b"0x1p-4")
    return eb.evaluate(evaluation_intent_bytes=ev_raw, admitted_evaluation_intent_sha256=ev_digest, run_intent_bytes=run_raw,
                       admitted_run_intent_sha256=run_digest, plan=PLAN, trace_dir=trace, score_bytes=score_bytes,
                       read_reference=loader or Loader(), calculate_endpoint=endpoint or Endpoint())


# ------------------------------------------------------------------------------------------------------------------ the assessment
def assessed(tmp_path, stage=ri.Stage.FEASIBILITY, x2="scored", scores=SCORES):
    trace, score_bytes = write_run(tmp_path / "w", x2=x2, scores=scores)
    raw, digest = run_bytes(tmp_path, stage)
    return eb.assess_execution(ri.admit_run_intent(raw, digest), digest, PLAN, trace, score_bytes)


@pytest.mark.parametrize("stage, x2, scores, expected", [
    (ri.Stage.FEASIBILITY, "scored", SCORES, ("complete", (), "passed", "complete", "withheld", "withheld_feasibility_stage", "2/2")),
    (ri.Stage.CONFIRMATORY, "scored", SCORES, ("complete", (), "passed", "complete", "permitted", "released", "2/2")),
    (ri.Stage.CONFIRMATORY, "mixture", SCORES, ("withheld", ("incomplete_eligible_exposures",), "passed", "incomplete", "withheld",
                                                "withheld_incomplete_exposures", "1/2")),
    (ri.Stage.CONFIRMATORY, "abnormal", SCORES, ("refused", ("execution_or_classification_failure",), "refused", "not_determined",
                                                 "withheld", "refused", "1/2")),
    (ri.Stage.CONFIRMATORY, "scored", {k: v for k, v in SCORES.items() if k != ("A3", "X2")},   # "scored" X2 silently omits A3
     ("refused", ("scores_disagree_with_exposure_outcomes",), "refused", "not_determined", "withheld", "refused", "2/2")),
])
def test_the_assessment_is_derived_from_the_evidence(tmp_path, stage, x2, scores, expected):
    a = assessed(tmp_path, stage, x2, scores)
    assert (a.primary_status, a.reasons, a.execution_integrity, a.method_completion, a.evaluation_permission,
            dict(a.release)["reference_recovery"], a.exposure_completion) == expected
    d = a.as_dict()
    assert d["scientific_interpretation"] == "not_established_by_these_fields" and d["stage"] == stage.value
    assert d["plan_sha256"] == PLAN.sha256 and a.sha256 == hashlib.sha256(a.render()).hexdigest()


def test_an_empty_eligible_set_is_refused(tmp_path):
    plan = PairPlan(("X3",), GENES, frozenset(), (("X3", "fewer_than_2_valid_genes"),), (("X3", 4, 1),), ())
    trace = tmp_path / "w"
    trace.mkdir()
    (trace / "exposures.jsonl").write_bytes((event("X3", "fewer_than_2_valid_genes", 2, 1) + "\n").encode("ascii"))
    score_bytes = "".join("{}\tX3\tNA\n".format(g) for g in GENES).encode("ascii")
    raw, digest = run_bytes(tmp_path, plan=plan)
    a = eb.assess_execution(ri.admit_run_intent(raw, digest), digest, plan, trace, score_bytes)
    assert (a.primary_status, a.reasons, a.exposure_completion, a.execution_integrity) == ("refused", ("no_eligible_exposures",), None, "refused")


def test_the_plan_must_be_the_one_the_intent_froze(tmp_path):
    trace, score_bytes = write_run(tmp_path / "w")
    raw, digest = run_bytes(tmp_path)
    other = PairPlan(PLAN.exposures, PLAN.genes, PLAN.scored - {("A4", "X1")}, PLAN.exclusions, (("X1", 4, 3), ("X2", 4, 3), ("X3", 4, 1)),
                     (("X1", GENES[:3]), ("X2", GENES[:3])))
    intent = ri.admit_run_intent(raw, digest)
    assert reason(lambda: eb.assess_execution(intent, digest, other, trace, score_bytes)) == "assessment_plan_not_the_intents"
    assert reason(lambda: eb.assess_execution(intent, "0" * 64, PLAN, trace, score_bytes)) == "assessment_intent"


def test_an_assessment_cannot_be_constructed_incoherently(tmp_path):
    a = assessed(tmp_path)
    fields = dict(a.__dict__)
    for change in ({"execution_integrity": "refused"}, {"method_completion": "incomplete"}, {"evaluation_permission": "permitted"},
                   {"reasons": ("x",)}, {"stage": "confirmatory"}):
        with pytest.raises(InferenceError) as exc:
            eb.ExecutionAssessment(**dict(fields, **change))
        assert exc.value.code == "assessment_inconsistent"


def test_release_decision_has_no_reference_input_and_derives_the_stage(tmp_path):
    assert not [p for p in inspect.signature(eb.release_decision).parameters if "reference" in p]
    assert not [p for p in inspect.signature(eb.assess_execution).parameters if "reference" in p]
    assert "assessment" not in inspect.signature(eb.evaluate).parameters        # the evaluator re-derives it; none is accepted
    trace, score_bytes = write_run(tmp_path / "w")
    raw, digest = run_bytes(tmp_path)
    d = eb.release_decision(run_intent_bytes=raw, admitted_run_intent_sha256=digest, plan=PLAN, trace_dir=trace, score_bytes=score_bytes)
    assert d["stage"] == "feasibility" and d["reference_evaluation"] is None
    assert d["assessment"]["release"]["reference_recovery"] == "withheld_feasibility_stage"
    assert d["assessment"]["release"]["release_policy_sha256"] == release_policy_sha256()
    relabelled = raw.replace(b'"feasibility"', b'"confirmatory"')
    assert reason(lambda: eb.release_decision(run_intent_bytes=relabelled, admitted_run_intent_sha256=digest, plan=PLAN, trace_dir=trace,
                                              score_bytes=score_bytes)) == "intent_binding_mismatch"


# ------------------------------------------------------------------------------------------------------------------ forbidden calls
def test_FORBIDDEN_1_a_complete_feasibility_evaluation_never_opens_the_reference(tmp_path):
    loader, endpoint = Loader(forbidden=True), Endpoint(forbidden=True)
    d = evaluate(tmp_path, eval_stage=ri.Stage.FEASIBILITY, loader=loader, endpoint=endpoint)
    assert d["status"] == "withheld_feasibility_stage" and d["endpoint"] is None and d["reference_sha256"] is None
    assert loader.calls == 0 and endpoint.calls == 0


def test_FORBIDDEN_2_an_incomplete_confirmatory_evaluation_never_opens_the_reference(tmp_path):
    loader, endpoint = Loader(forbidden=True), Endpoint(forbidden=True)
    d = evaluate(tmp_path, run_stage=ri.Stage.CONFIRMATORY, x2="mixture", loader=loader, endpoint=endpoint)
    assert d["status"] == "withheld_incomplete_exposures" and loader.calls == 0 and endpoint.calls == 0


def test_FORBIDDEN_3_refused_evidence_never_opens_the_reference(tmp_path):
    loader, endpoint = Loader(forbidden=True), Endpoint(forbidden=True)
    d = evaluate(tmp_path, run_stage=ri.Stage.CONFIRMATORY, x2="abnormal", loader=loader, endpoint=endpoint)
    assert d["status"] == "refused" and loader.calls == 0 and endpoint.calls == 0


def test_FORBIDDEN_4_changed_score_bytes_are_refused_before_any_reference_access(tmp_path):
    loader, endpoint = Loader(forbidden=True), Endpoint(forbidden=True)
    assert reason(lambda: evaluate(tmp_path, run_stage=ri.Stage.CONFIRMATORY, loader=loader, endpoint=endpoint, tamper_scores=True)) == \
        "evaluation_score_bytes_mismatch"
    assert loader.calls == 0 and endpoint.calls == 0


def test_FORBIDDEN_5_a_wrong_reference_digest_is_refused_before_the_endpoint(tmp_path):
    loader, endpoint = Loader(payload=b"A1\n"), Endpoint(forbidden=True)
    assert reason(lambda: evaluate(tmp_path, run_stage=ri.Stage.CONFIRMATORY, loader=loader, endpoint=endpoint)) == \
        "evaluation_reference_bytes_mismatch"
    assert loader.calls == 1 and endpoint.calls == 0


def test_FORBIDDEN_6_matching_bindings_compute_the_endpoint_exactly_once(tmp_path):
    loader, endpoint = Loader(), Endpoint()
    d = evaluate(tmp_path, run_stage=ri.Stage.CONFIRMATORY, loader=loader, endpoint=endpoint)
    assert d["status"] == "evaluated" and d["endpoint"] == {"delta_h20": 1} and loader.calls == 1 and endpoint.calls == 1
    assert endpoint.seen[1] == REFERENCE and hashlib.sha256(endpoint.seen[0]).hexdigest() == d["score_artifact_sha256"]
    assert d["reference_sha256"] == hashlib.sha256(REFERENCE).hexdigest() and d["scientific_interpretation"] == "not_established_by_these_fields"


def test_a_confirmatory_evaluation_may_reuse_feasibility_scores_only_with_the_history_disclosed(tmp_path):
    loader = Loader(forbidden=True)
    assert reason(lambda: evaluate(tmp_path / "a", run_stage=ri.Stage.FEASIBILITY, informed_by=(), loader=loader)) == \
        "evaluation_history_undisclosed"
    assert loader.calls == 0
    d = evaluate(tmp_path / "b", run_stage=ri.Stage.FEASIBILITY)            # informed_by names the feasibility computation
    assert d["status"] == "evaluated"            # a NEW evaluation identity; the computation itself stays feasibility


def test_the_evaluation_must_name_this_computation(tmp_path):
    trace, score_bytes = write_run(tmp_path / "w")
    run_raw, run_digest = run_bytes(tmp_path, ri.Stage.CONFIRMATORY)
    ev_raw, ev_digest = seal_evaluation(tmp_path, "9" * 64, score_bytes, informed_by=())
    loader = Loader(forbidden=True)
    assert reason(lambda: eb.evaluate(evaluation_intent_bytes=ev_raw, admitted_evaluation_intent_sha256=ev_digest, run_intent_bytes=run_raw,
                                      admitted_run_intent_sha256=run_digest, plan=PLAN, trace_dir=trace, score_bytes=score_bytes,
                                      read_reference=loader, calculate_endpoint=Endpoint(forbidden=True))) == "evaluation_run_intent_mismatch"
    assert loader.calls == 0


def test_the_reference_and_the_endpoint_must_have_their_types(tmp_path):
    assert reason(lambda: evaluate(tmp_path / "a", run_stage=ri.Stage.CONFIRMATORY, loader=Loader(payload="A2\n"))) == \
        "evaluation_reference_bytes_mismatch"
    assert reason(lambda: evaluate(tmp_path / "b", run_stage=ri.Stage.CONFIRMATORY, endpoint=Endpoint(result=[1]))) == "evaluation_endpoint_type"
    assert reason(lambda: evaluate(tmp_path / "c", run_stage=ri.Stage.CONFIRMATORY, loader="not callable")) == "evaluation_callables"


def test_a_stage_change_after_sealing_is_refused_by_the_evaluator_too(tmp_path):
    trace, score_bytes = write_run(tmp_path / "w")
    run_raw, run_digest = run_bytes(tmp_path, ri.Stage.FEASIBILITY)
    ev_raw, ev_digest = seal_evaluation(tmp_path, run_digest, score_bytes)
    relabelled = run_raw.replace(b'"feasibility"', b'"confirmatory"')
    assert reason(lambda: eb.evaluate(evaluation_intent_bytes=ev_raw, admitted_evaluation_intent_sha256=ev_digest,
                                      run_intent_bytes=relabelled, admitted_run_intent_sha256=run_digest, plan=PLAN, trace_dir=trace,
                                      score_bytes=score_bytes, read_reference=Loader(forbidden=True),
                                      calculate_endpoint=Endpoint(forbidden=True))) == "intent_binding_mismatch"
