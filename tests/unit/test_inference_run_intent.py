"""Pre-execution run and evaluation intents (owner ruling 2026-10-09): canonical identity, exclusive sealing, strict admission against
the persisted digest and the implementation's release policy, and the rule that changing ONLY the stage breaks the binding.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json

import pytest

from genomic_variant_classifier.inference import run_intent as ri
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import STAGES, release_policy_sha256

POLICY = release_policy_sha256()


def reason(fn):
    with pytest.raises(InferenceError) as exc:
        fn()
    return exc.value.code


def run_intent(**over):
    kw = dict(run_id="asthma-feasibility-1", stage=ri.Stage.FEASIBILITY, contract_sha256="c" * 64, plan_sha256="d" * 64,
              inputs=(ri.InputIdentity("burden", "1" * 64, 10), ri.InputIdentity("trans", "2" * 64, 20)), environment_sha256="e" * 64,
              implementation_tree="a" * 40, release_policy_sha256=POLICY,
              prior_knowledge="The analyst knows several prominent asthma genes; software withholding does not blind that knowledge.",
              informed_by=())
    kw.update(over)
    return ri.RunIntent(**kw)


def evaluation_intent(**over):
    kw = dict(evaluation_id="asthma-confirmatory-eval-1", stage=ri.Stage.CONFIRMATORY, run_intent_sha256="3" * 64,
              score_artifact_sha256="4" * 64, reference_sha256="5" * 64, release_policy_sha256=POLICY,
              prior_knowledge="Feasibility run 3333 was inspected before this evaluation was specified.", informed_by=("3" * 64,))
    kw.update(over)
    return ri.EvaluationIntent(**kw)


# ------------------------------------------------------------------------------------------------------------------ identity
def test_the_stage_vocabulary_is_the_release_tables():
    assert sorted(s.value for s in ri.Stage) == sorted(STAGES)


def test_the_run_intent_renders_canonically_and_round_trips():
    intent = run_intent()
    raw = intent.render()
    doc = json.loads(raw)
    assert raw == (json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    assert doc["schema"] == ri.RUN_INTENT_SCHEMA and doc["schema_version"] == 1 and doc["stage"] == "feasibility"
    assert ri.RunIntent.parse(raw) == intent and intent.sha256 == hashlib.sha256(raw).hexdigest()


def test_changing_only_the_stage_changes_the_identity():
    feasibility, confirmatory = run_intent(), run_intent(stage=ri.Stage.CONFIRMATORY)
    assert feasibility.render().replace(b'"stage":"feasibility"', b'"stage":"confirmatory"') == confirmatory.render()
    assert feasibility.sha256 != confirmatory.sha256


# ------------------------------------------------------------------------------------------------------------------ sealing and admission
def test_sealing_is_exclusive_and_admission_requires_the_sealed_bytes(tmp_path):
    intent = run_intent()
    path = tmp_path / "run_intent.json"
    digest = ri.seal_intent(path, intent)
    assert digest == intent.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reason(lambda: ri.seal_intent(path, run_intent(stage=ri.Stage.CONFIRMATORY))) == "intent_exists"
    assert path.read_bytes() == intent.render()                       # the sealed intent was not replaced
    assert ri.admit_run_intent(path.read_bytes(), digest) == intent


def test_THE_RULING_TEST_changing_only_the_stage_breaks_the_persisted_binding(tmp_path):
    """Ruling 2026-10-09: the admitted digest comes from the persisted pre-execution admission; a newly supplied intent is never
    recomputed and trusted."""
    digest = ri.seal_intent(tmp_path / "run_intent.json", run_intent())
    relabelled = run_intent(stage=ri.Stage.CONFIRMATORY).render()
    assert reason(lambda: ri.admit_run_intent(relabelled, digest)) == "intent_binding_mismatch"
    assert reason(lambda: ri.admit_run_intent(relabelled, hashlib.sha256(relabelled).hexdigest()[:63] + "x")) == "intent_admitted_digest"


def test_an_intent_bound_to_another_release_policy_is_refused(tmp_path):
    other = run_intent(release_policy_sha256="f" * 64)
    digest = ri.seal_intent(tmp_path / "i.json", other)
    assert reason(lambda: ri.admit_run_intent(other.render(), digest)) == "intent_release_policy_not_this_implementation"
    ev = evaluation_intent(release_policy_sha256="f" * 64)
    assert reason(lambda: ri.admit_evaluation_intent(ev.render(), ev.sha256)) == "evaluation_intent_release_policy_not_this_implementation"


def test_every_parse_case_really_changes_the_bytes():
    """A replacement that matches nothing would test the canonical intent instead of the damage (measured 2026-10-09: the first
    version of the missing-stage case looked for a trailing comma the last key does not have)."""
    for raw, _ in PARSE_CASES:
        assert raw != run_intent().render()


PARSE_CASES = [
    (run_intent().render().replace(b',"stage":"feasibility"', b""), "intent_keys"),                     # a MISSING stage refuses
    (run_intent().render().replace(b'"feasibility"', b'"exploratory"'), "intent_stage"),                 # an UNKNOWN stage refuses
    (run_intent().render().replace(b'"feasibility"', b"null"), "intent_stage"),
    (run_intent().render().replace(b'"feasibility"', b'"Feasibility"'), "intent_stage"),
    (run_intent().render().replace(b'{"contract', b'{"extra":1,"contract'), "intent_keys"),
    (run_intent().render().replace(b'"schema_version":1', b'"schema_version":true'), "intent_schema"),
    (run_intent().render().replace(b'"schema_version":1', b'"schema_version":2'), "intent_schema"),
    (run_intent().render().replace(b'gvc.analysis-run-intent', b'gvc.analysis-run-intent-x'), "intent_schema"),
    (run_intent().render().replace(b'"size_bytes":10', b'"size_bytes":10.0'), "intent_json"),
    (run_intent().render().replace(b'"size_bytes":10', b'"size_bytes":true'), "intent_input_size"),
    (run_intent().render().replace(b'{"contract', b'{"run_id":"x","contract'), "intent_json"),           # duplicate key
    (b"\xef\xbb\xbf" + run_intent().render(), "intent_json"),
    (b"[]\n", "intent_json"),
    ((json.dumps(json.loads(run_intent().render()), indent=2, sort_keys=True) + "\n").encode(), "intent_not_canonical"),
    (run_intent().render()[:-1], "intent_not_canonical"),
    (run_intent().render().replace(b"\n", b"\r\n"), "intent_not_canonical"),
]


@pytest.mark.parametrize("raw, code", PARSE_CASES)
def test_parsing_is_strict(raw, code):
    assert reason(lambda: ri.RunIntent.parse(raw)) == code


@pytest.mark.parametrize("over, code", [
    ({"run_id": ""}, "intent_run_id"), ({"run_id": "has space"}, "intent_run_id"), ({"run_id": "-lead"}, "intent_run_id"),
    ({"stage": "feasibility"}, "intent_stage"), ({"stage": None}, "intent_stage"),
    ({"contract_sha256": "C" * 64}, "intent_contract_sha256"), ({"plan_sha256": "d" * 63}, "intent_plan_sha256"),
    ({"environment_sha256": None}, "intent_environment_sha256"), ({"release_policy_sha256": ""}, "intent_release_policy_sha256"),
    ({"inputs": ()}, "intent_inputs"), ({"inputs": [ri.InputIdentity("a", "1" * 64, 1)]}, "intent_inputs"),
    ({"inputs": (ri.InputIdentity("b", "1" * 64, 1), ri.InputIdentity("a", "1" * 64, 1))}, "intent_inputs"),
    ({"inputs": (ri.InputIdentity("a", "1" * 64, 1), ri.InputIdentity("a", "2" * 64, 1))}, "intent_inputs"),
    ({"implementation_tree": "a" * 39}, "intent_implementation_tree"), ({"implementation_tree": "A" * 40}, "intent_implementation_tree"),
    ({"prior_knowledge": "  "}, "intent_prior_knowledge"), ({"prior_knowledge": "x" * 4001}, "intent_prior_knowledge"),
    ({"informed_by": ("2" * 64, "1" * 64)}, "intent_informed_by"), ({"informed_by": ("1" * 64, "1" * 64)}, "intent_informed_by"),
    ({"informed_by": ["1" * 64]}, "intent_informed_by"), ({"informed_by": ("1" * 63,)}, "intent_informed_by"),
])
def test_the_run_intent_fields_are_checked(over, code):
    assert reason(lambda: run_intent(**over)) == code


@pytest.mark.parametrize("args, code", [(("", "1" * 64, 1), "intent_input_name"), (("a", "1" * 64, 0), "intent_input_size"),
                                        (("a", "1" * 64, True), "intent_input_size"), (("a", "g" * 64, 1), "intent_input_sha256")])
def test_input_identities_are_checked(args, code):
    assert reason(lambda: ri.InputIdentity(*args)) == code


# ------------------------------------------------------------------------------------------------------------------ evaluation intents
def test_a_feasibility_evaluation_names_no_reference_and_a_confirmatory_one_must():
    assert reason(lambda: evaluation_intent(stage=ri.Stage.FEASIBILITY)) == "evaluation_intent_feasibility_reference_forbidden"
    assert evaluation_intent(stage=ri.Stage.FEASIBILITY, reference_sha256=None).reference_sha256 is None
    assert reason(lambda: evaluation_intent(reference_sha256=None)) == "evaluation_intent_reference_sha256"


def test_the_evaluation_intent_round_trips_and_is_admitted_only_against_its_digest(tmp_path):
    ev = evaluation_intent()
    assert ri.EvaluationIntent.parse(ev.render()) == ev
    digest = ri.seal_intent(tmp_path / "evaluation_intent.json", ev)
    assert ri.admit_evaluation_intent(ev.render(), digest) == ev
    other = evaluation_intent(reference_sha256="6" * 64).render()
    assert reason(lambda: ri.admit_evaluation_intent(other, digest)) == "evaluation_intent_binding_mismatch"
    assert reason(lambda: ri.EvaluationIntent.parse(ev.render().replace(b'"confirmatory"', b'"exploratory"'))) == "evaluation_intent_stage"
    assert reason(lambda: ri.EvaluationIntent.parse(ev.render()[:-1])) == "evaluation_intent_not_canonical"
    assert reason(lambda: ri.seal_intent(tmp_path / "x.json", ev.render())) == "intent_type"


@pytest.mark.parametrize("over, code", [
    ({"evaluation_id": "a b"}, "evaluation_intent_id"), ({"stage": "confirmatory"}, "evaluation_intent_stage"),
    ({"run_intent_sha256": "3" * 65}, "evaluation_intent_run_intent_sha256"),
    ({"score_artifact_sha256": None}, "evaluation_intent_score_artifact_sha256"),
    ({"reference_sha256": "5" * 63}, "evaluation_intent_reference_sha256"),
    ({"prior_knowledge": ""}, "evaluation_intent_prior_knowledge"), ({"informed_by": ("3" * 64, "3" * 64)}, "evaluation_intent_informed_by"),
])
def test_the_evaluation_intent_fields_are_checked(over, code):
    assert reason(lambda: evaluation_intent(**over)) == code
