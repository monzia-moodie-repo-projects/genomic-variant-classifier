"""Per-exposure outcomes, coverage and the primary / diagnostic split (owner ruling 2026-10-08g).

The trace lines below are the REAL output of scripts/dandelion/dandelion_exposure_recorder.R on fixture T2 (DANDELION 0.1.0, R 4.3.3,
development sandbox), copied verbatim; the rest are constructed to exercise each refusal.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

import pytest

from genomic_variant_classifier.inference import exposure_outcomes as xo
from genomic_variant_classifier.inference.analysis_contract import ExposureFailurePolicy, FailureDiagnostic
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.ranking import Pair, State, rank_genes

S = xo.ExposureStatus
REAL_T2 = [
    '{"exposure_id":"E1","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":30,"pi0a":"0x1.bfd3c9c5623ap-1","pi0b":"0x1.c86b0bd2cbe9bp-1","wg1":"0x1.84ec47bbf77f9p-4","wg2":"0x1.c9a6582744fd1p-4","wg3":"0x1.8f3640cde34a1p-1","wg_sum":"0x1.f90894ca4ad9ap-1","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/1"}',
    '{"exposure_id":"E2","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":30,"pi0a":"0x1.f716a749f10cbp-1","pi0b":"0x1.c86b0bd2cbe9bp-1","wg1":"0x1.b4ea5b90194dbp-4","wg2":"0x1.fc6bfeb781ad4p-7","wg3":"0x1.c0795bd7ede3p-1","wg_sum":"0x1.ff085744cf137p-1","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/1"}',
    '{"exposure_id":"E3","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":9,"pi0a":"0x1.996dbad302264p-1","pi0b":"0x1.05ab77eb75ad7p-1","wg1":"0x1.905c5a168261fp-2","wg2":"0x1.a35f511da5c15p-4","wg3":"0x1.a27f1b8f81ea9p-2","wg_sum":"0x1.cdd9a4f6b6de6p-1","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/1"}',
    '{"exposure_id":"E4","outcome":"fewer_than_2_valid_genes","last_guard_reached":2,"n_trans":31,"n_valid":1,"pi0a":null,"pi0b":null,"wg1":null,"wg2":null,"wg3":null,"wg_sum":null,"observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/1"}',
    '{"exposure_id":"E5","outcome":"mixture_estimate_invalid","last_guard_reached":3,"n_trans":31,"n_valid":31,"pi0a":"-0x1.e7bab1690b2p-7","pi0b":"0x1.b8eb92c088c7dp-1","wg1":null,"wg2":null,"wg3":null,"wg_sum":null,"observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/1"}',
]
T2_EXPOSURES = ("E1", "E2", "E3", "E4", "E5", "E6")
T2_EXCLUDED = {"E4": "fewer_than_2_valid_genes", "E6": "exposure_not_annotated"}
T2_USABLE = {"E1": (31, 30), "E2": (31, 30), "E3": (31, 9), "E4": (31, 1), "E5": (31, 31)}


def reason(fn):
    with pytest.raises(InferenceError) as exc:
        fn()
    return exc.value.code


def write_trace(directory, lines):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "exposures.jsonl").write_bytes("".join(x + "\n" for x in lines).encode("ascii"))
    return directory


def line(**over):
    """A scored event, edited by keyword."""
    doc = json.loads(REAL_T2[0])
    doc.update(over)
    return json.dumps(doc, separators=(",", ":"))


def t2_statuses(tmp_path):
    return xo.classify_exposures(T2_EXPOSURES, T2_EXCLUDED, T2_USABLE, xo.read_exposure_trace(write_trace(tmp_path / "x", REAL_T2)))


# ------------------------------------------------------------------------------------------------------------------ reading the trace
def test_the_real_trace_parses_exactly(tmp_path):
    events = xo.read_exposure_trace(write_trace(tmp_path / "x", REAL_T2))
    assert [(e.exposure_id, e.outcome, e.last_guard_reached, e.n_trans, e.n_valid) for e in events] == [
        ("E1", "scored", 4, 31, 30), ("E2", "scored", 4, 31, 30), ("E3", "scored", 4, 31, 9),
        ("E4", "fewer_than_2_valid_genes", 2, 31, 1), ("E5", "mixture_estimate_invalid", 3, 31, 31)]
    e5 = events[4]
    assert e5.pi0a == Fraction(float.fromhex("-0x1.e7bab1690b2p-7")) < 0 and e5.wg_sum is None
    assert events[0].pi0b == events[1].pi0b != events[4].pi0b          # the burden side follows the VALID gene set


def test_an_empty_trace_is_zero_events_not_a_missing_trace(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "exposures.jsonl").write_bytes(b"")
    assert xo.read_exposure_trace(tmp_path / "x") == ()
    assert reason(lambda: xo.read_exposure_trace(tmp_path / "nothing")) == "exposure_trace_missing"


def test_a_na_estimate_is_kept_explicitly_and_is_a_mixture_failure(tmp_path):
    na = line(exposure_id="E9", outcome="mixture_estimate_invalid", last_guard_reached=3, pi0a="NA", wg1=None, wg2=None, wg3=None, wg_sum=None)
    (event,) = xo.read_exposure_trace(write_trace(tmp_path / "x", [na]))
    assert event.pi0a == "NA" and event.pi0b > 0


@pytest.mark.parametrize("lines, code", [
    ([REAL_T2[0], REAL_T2[0]], "exposure_trace_duplicate_exposure"),
    ([REAL_T2[0].replace('"n_trans":31', '"n_trans":31,"extra":1')], "exposure_trace_keys"),
    ([REAL_T2[0].replace('"wg_sum":"0x1.f90894ca4ad9ap-1",', '')], "exposure_trace_keys"),
    ([REAL_T2[0].replace('"n_trans":31', '"n_trans":31,"n_trans":31')], "exposure_trace_duplicate_key"),
    ([REAL_T2[0].replace('"n_trans":31', '"n_trans":31.0')], "exposure_trace_number"),
    ([REAL_T2[0].replace('"n_trans":31', '"n_trans":NaN')], "exposure_trace_number"),
    ([REAL_T2[0][:-5]], "exposure_trace_json"),
    ([line(recorder_version="gvc.dandelion-exposure-recorder/0")], "exposure_trace_value"),
    ([line(observation_kind="replay")], "exposure_trace_value"),
    ([line(outcome="vanished")], "exposure_trace_value"),
    ([line(last_guard_reached=5)], "exposure_trace_value"),
    ([line(exposure_id="")], "exposure_trace_value"),
    ([line(n_valid=-1)], "exposure_trace_value"),
    ([line(n_valid=True)], "exposure_trace_value"),
    ([line(pi0a="0.5")], "exposure_trace_value"),
    ([line(pi0a="Inf")], "exposure_trace_value"),
    ([line(pi0a="0x1.00000000000001p-1")], "exposure_trace_value_not_binary64"),
    ([line(last_guard_reached=3)], "exposure_trace_inconsistent"),                        # "scored" needs all four guards
    ([line(outcome="mixture_estimate_invalid", pi0a="-0x1p-7")], "exposure_trace_inconsistent"),   # the pi0 guard returns before weights
    ([line(n_valid=32)], "exposure_trace_inconsistent"),                                   # more valid than trans genes
    ([line(pi0a="-0x1p-7")], "exposure_trace_inconsistent"),                               # scores with an invalid estimate
    ([line(wg_sum="0x0p+0")], "exposure_trace_inconsistent"),                              # scores with a zero weight sum
    ([line(outcome="nonpositive_weight_sum")], "exposure_trace_inconsistent"),            # positive sum yet NULL
    ([line(outcome="mixture_estimate_invalid", last_guard_reached=3, wg1=None, wg2=None, wg3=None, wg_sum=None)],
     "exposure_trace_inconsistent"),                                                       # "invalid" with valid estimates
    ([line(outcome="mixture_estimate_invalid", last_guard_reached=3)], "exposure_trace_inconsistent"),   # weights never reached
    ([line(outcome="fewer_than_2_valid_genes", last_guard_reached=2, pi0a=None, pi0b=None, wg1=None, wg2=None, wg3=None, wg_sum=None)],
     "exposure_trace_inconsistent"),                                                       # 30 valid genes is not "fewer than 2"
    ([line(outcome="no_trans_genes", last_guard_reached=1, n_valid=None, pi0a=None, pi0b=None, wg1=None, wg2=None, wg3=None, wg_sum=None)],
     "exposure_trace_inconsistent"),                                                       # 31 trans genes is not "none"
    ([line(outcome="fewer_than_2_valid_genes", last_guard_reached=2, n_valid=1)], "exposure_trace_inconsistent"),  # estimate recorded
])
def test_the_trace_is_strict(tmp_path, lines, code):
    assert reason(lambda: xo.read_exposure_trace(write_trace(tmp_path / "x", lines))) == code


def test_the_trace_refuses_bytes_it_did_not_write(tmp_path):
    d = write_trace(tmp_path / "x", REAL_T2[:1])
    (d / "stray.txt").write_bytes(b"x")
    assert reason(lambda: xo.read_exposure_trace(d)) == "exposure_trace_unexpected_files"
    (d / "stray.txt").unlink()
    (d / "exposures.jsonl").write_bytes(REAL_T2[0].encode() + b"\r\n")
    assert reason(lambda: xo.read_exposure_trace(d)) == "exposure_trace_truncated"
    (d / "exposures.jsonl").write_bytes(REAL_T2[0].encode())
    assert reason(lambda: xo.read_exposure_trace(d)) == "exposure_trace_truncated"
    (d / "exposures.jsonl").write_bytes(REAL_T2[0].replace("E1", "É1").encode() + b"\n")
    assert reason(lambda: xo.read_exposure_trace(d)) == "exposure_trace_encoding"


# ------------------------------------------------------------------------------------------------------------------ classification
def test_the_real_t2_trace_classifies_one_status_per_planned_exposure(tmp_path):
    st = t2_statuses(tmp_path)
    assert {x: s["status"] for x, s in st.items()} == {
        "E1": S.SCORED, "E2": S.SCORED, "E3": S.SCORED, "E4": S.STRUCTURALLY_INELIGIBLE, "E5": S.MIXTURE_ESTIMATE_INVALID,
        "E6": S.STRUCTURALLY_INELIGIBLE}
    assert st["E6"]["event"] is None and st["E6"]["reason"] == "exposure_not_annotated"


def _classify(tmp_path, lines, excluded=None, usable=None, exposures=T2_EXPOSURES):
    events = xo.read_exposure_trace(write_trace(tmp_path / "c", lines))
    return xo.classify_exposures(exposures, T2_EXCLUDED if excluded is None else excluded, T2_USABLE if usable is None else usable, events)


def test_a_missing_call_is_unclassified_never_silently_dropped(tmp_path):
    st = _classify(tmp_path, REAL_T2[:4])                                   # E5 never observed
    assert st["E5"]["status"] is S.UNCLASSIFIED_MISSING_EXPOSURE
    st = _classify(tmp_path / "b", [REAL_T2[i] for i in (0, 1, 2, 4)])      # predicted-structural E4 never observed
    assert st["E4"]["status"] is S.UNCLASSIFIED_MISSING_EXPOSURE


def test_an_abnormal_or_unclassified_exit_is_an_infrastructure_error(tmp_path):
    bad = line(exposure_id="E5", outcome="abnormal_exit", last_guard_reached=3, n_valid=31, wg1=None, wg2=None, wg3=None, wg_sum=None)
    assert _classify(tmp_path, REAL_T2[:4] + [bad])["E5"]["status"] is S.INFRASTRUCTURE_ERROR
    odd = line(exposure_id="E4", outcome="unclassified", last_guard_reached=2, n_valid=1, pi0a=None, pi0b=None, wg1=None, wg2=None,
               wg3=None, wg_sum=None)
    assert _classify(tmp_path / "b", REAL_T2[:3] + [odd, REAL_T2[4]])["E4"]["status"] is S.INFRASTRUCTURE_ERROR


@pytest.mark.parametrize("mutate, code", [
    # an exposure predicted eligible that the implementation calls structural: the frozen rule is wrong -- refused, never relabelled
    (lambda lines: lines[:4] + [line(exposure_id="E5", outcome="fewer_than_2_valid_genes", last_guard_reached=2, n_valid=1, pi0a=None,
                                     pi0b=None, wg1=None, wg2=None, wg3=None, wg_sum=None)], "usable_pair_count_mismatch"),
    (lambda lines: lines[:4] + [line(exposure_id="E5", outcome="no_trans_genes", last_guard_reached=1, n_trans=0, n_valid=None, pi0a=None,
                                     pi0b=None, wg1=None, wg2=None, wg3=None, wg_sum=None)], "usable_pair_count_mismatch"),
    (lambda lines: lines[:3] + [line(exposure_id="E4", n_valid=1)] + lines[4:], "exposure_trace_inconsistent"),
    (lambda lines: lines + [line(exposure_id="E6")], "eligibility_rule_mismatch"),
    (lambda lines: lines + [line(exposure_id="E7")], "exposure_trace_unplanned"),
    (lambda lines: [line(n_valid=29)] + lines[1:], "usable_pair_count_mismatch"),
    (lambda lines: [line(n_trans=30, n_valid=30)] + lines[1:], "usable_pair_count_mismatch"),
])
def test_disagreement_with_the_frozen_rule_refuses(tmp_path, mutate, code):
    assert reason(lambda: _classify(tmp_path, mutate(list(REAL_T2)))) == code


def test_a_structural_outcome_for_a_predicted_eligible_exposure_is_a_rule_mismatch(tmp_path):
    usable = dict(T2_USABLE, E5=(31, 1))           # counts agree; the CATEGORY does not
    lines = REAL_T2[:4] + [line(exposure_id="E5", outcome="fewer_than_2_valid_genes", last_guard_reached=2, n_valid=1, pi0a=None, pi0b=None,
                                wg1=None, wg2=None, wg3=None, wg_sum=None)]
    assert reason(lambda: _classify(tmp_path, lines, usable=usable)) == "eligibility_rule_mismatch"
    usable = dict(T2_USABLE, E4=(31, 30))           # predicted structural (fewer than 2 valid genes), observed SCORED
    lines = REAL_T2[:3] + [line(exposure_id="E4"), REAL_T2[4]]
    assert reason(lambda: _classify(tmp_path / "b", lines, excluded=T2_EXCLUDED, usable=usable)) == "eligibility_rule_mismatch"


def test_an_unannotated_exposure_must_never_reach_the_method(tmp_path):
    assert reason(lambda: _classify(tmp_path, REAL_T2 + [line(exposure_id="E6")])) == "eligibility_rule_mismatch"
    # counts are predicted for exactly the annotated exposures: one for the unannotated E6 (or none for E5) is a malformed prediction
    assert reason(lambda: _classify(tmp_path / "b", REAL_T2, usable=dict(T2_USABLE, E6=(31, 30)))) == "usable_pair_prediction_scope"
    usable = {k: v for k, v in T2_USABLE.items() if k != "E5"}
    assert reason(lambda: _classify(tmp_path / "c", REAL_T2, usable=usable)) == "usable_pair_prediction_scope"


@pytest.mark.parametrize("args, code", [
    (((), {}, {}), "exposure_plan"),
    ((("E1", "E1"), {}, {"E1": (1, 1)}), "exposure_plan"),
    ((("E1",), {"E1": "too_few"}, {"E1": (1, 1)}), "eligibility_rule_unknown"),
    ((("E1",), {"E2": "no_trans_genes"}, {"E1": (1, 1)}), "eligibility_rule_unknown"),
    ((("E1", "E2"), {}, {"E1": (1, 1)}), "usable_pair_prediction_scope"),
])
def test_the_classification_inputs_are_checked(args, code):
    assert reason(lambda: xo.classify_exposures(*args, ())) == code


def test_the_events_must_be_parsed_events():
    assert reason(lambda: xo.classify_exposures(("E1",), {}, {"E1": (1, 1)}, ("not an event",))) == "exposure_event_type"


# ------------------------------------------------------------------------------------------------------------------ coverage
def t2_plan():
    """T2's plan: G01-G09 planned for E1, E2, E3, E5; G10-G30 for E1, E2, E5; G31 for E5 only; E4 and E6 structural."""
    genes = ["G%02d" % i for i in range(1, 32)]
    plan = {}
    for i, g in enumerate(genes, 1):
        for x in T2_EXPOSURES:
            plan[(g, x)] = (x == "E5" or (i <= 30 and x in ("E1", "E2")) or (i <= 9 and x == "E3"))
    return plan


def test_t2_coverage_completion_and_the_withheld_primary(tmp_path):
    cov = xo.coverage_report(t2_plan(), t2_statuses(tmp_path))
    assert cov["exposure_completion"] == "3/4" and cov["planned_eligible_exposures"] == 4 and cov["successfully_scored_eligible_exposures"] == 3
    assert cov["primary_status"] == xo.PRIMARY_WITHHELD and cov["failed_eligible_exposures"] == {"E5": "mixture_estimate_invalid"}
    assert cov["gene_coverage_distribution"] == {"0/1": 1, "2/3": 21, "3/4": 9}
    assert cov["structurally_ineligible_exposures"] == T2_EXCLUDED and cov["genes_outside_planned_universe"] == []
    assert cov["per_gene_coverage"]["G31"] == {"planned_pairs": 1, "scored_pairs": 0}


def _statuses(**kinds):
    return {x: {"status": k, "reason": k.value, "event": None} for x, k in kinds.items()}


def test_the_distribution_is_ordered_numerically_and_genes_without_a_plan_are_named():
    plan = {("g%02d" % i, "x%02d" % j): (j <= i) for i in range(1, 13) for j in range(1, 13)}
    plan[("lonely", "x01")] = False
    st = _statuses(**{"x%02d" % j: S.SCORED for j in range(1, 13)})
    st["x12"] = {"status": S.NONPOSITIVE_WEIGHT_SUM, "reason": "nonpositive_weight_sum", "event": None}
    cov = xo.coverage_report(plan, st)
    assert list(cov["gene_coverage_distribution"]) == ["1/1", "2/2", "3/3", "4/4", "5/5", "6/6", "7/7", "8/8", "9/9", "10/10", "11/11", "11/12"]
    assert cov["genes_outside_planned_universe"] == ["lonely"] and cov["genes_with_incomplete_coverage"] == ["g12"]


def test_the_primary_status_table():
    plan = {("g", "a"): True, ("g", "b"): True, ("h", "a"): True, ("h", "c"): False}
    assert xo.coverage_report(plan, _statuses(a=S.SCORED, b=S.SCORED, c=S.STRUCTURALLY_INELIGIBLE))["primary_status"] == xo.PRIMARY_COMPLETE
    assert xo.coverage_report(plan, _statuses(a=S.SCORED, b=S.MIXTURE_ESTIMATE_INVALID, c=S.STRUCTURALLY_INELIGIBLE))["primary_status"] == xo.PRIMARY_WITHHELD
    for refusal in (S.UNCLASSIFIED_MISSING_EXPOSURE, S.INFRASTRUCTURE_ERROR):
        st = _statuses(a=S.SCORED, b=S.MIXTURE_ESTIMATE_INVALID, c=S.STRUCTURALLY_INELIGIBLE)
        st["a"] = {"status": refusal, "reason": "r", "event": None}
        assert xo.coverage_report(plan, st)["primary_status"] == xo.PRIMARY_REFUSED


@pytest.mark.parametrize("plan, statuses, code", [
    ({}, _statuses(a=S.SCORED), "empty_plan"),
    ({("g", "a"): 1}, _statuses(a=S.SCORED), "invalid_plan"),
    ({("g", "a"): True}, {"a": {"status": "scored", "reason": "", "event": None}}, "exposure_status_type"),
    ({("g", "a"): True, ("g", "b"): False}, _statuses(a=S.SCORED), "exposure_status_missing"),
    ({("g", "a"): True}, _statuses(a=S.STRUCTURALLY_INELIGIBLE), "plan_status_mismatch"),
    ({("g", "a"): False}, _statuses(a=S.STRUCTURALLY_INELIGIBLE), "no_eligible_exposure"),
])
def test_coverage_inputs_are_checked(plan, statuses, code):
    assert reason(lambda: xo.coverage_report(plan, statuses)) == code


def test_exposure_records_preserve_what_the_ruling_lists(tmp_path):
    rec = xo.exposure_records(t2_plan(), t2_statuses(tmp_path))
    e5 = rec["E5"]
    assert e5["status"] == "mixture_estimate_invalid" and e5["invalid_side"] == "trans" and e5["planned_genes"] == 31
    assert e5["genes_it_would_cover"] == ["G%02d" % i for i in range(1, 32)]
    assert e5["observed"] == {"outcome": "mixture_estimate_invalid", "last_guard_reached": 3, "n_trans": 31, "usable_pairs": 31,
                              "pi0a": "-0x1.e7bab1690b2p-7", "pi0b": "0x1.b8eb92c088c7dp-1", "wg1": None, "wg2": None, "wg3": None, "wg_sum": None}
    assert "genes_it_would_cover" not in rec["E1"] and rec["E1"]["observed"]["wg_sum"] == "0x1.f90894ca4ad9ap-1"
    assert rec["E6"] == {"status": "structurally_ineligible", "reason": "exposure_not_annotated", "planned_genes": 0, "observed": None}


@pytest.mark.parametrize("pi0a, pi0b, side", [("-0x1p-7", "0x1p-1", "trans"), ("0x1p-1", "-0x1p-9", "burden"), ("NA", "-0x1p-9", "both")])
def test_the_invalid_side_is_named(tmp_path, pi0a, pi0b, side):
    bad = line(exposure_id="E5", outcome="mixture_estimate_invalid", last_guard_reached=3, n_valid=31, pi0a=pi0a, pi0b=pi0b, wg1=None, wg2=None,
               wg3=None, wg_sum=None)
    st = _classify(tmp_path, REAL_T2[:4] + [bad])
    assert xo.exposure_records(t2_plan(), st)["E5"]["invalid_side"] == side


def test_the_render_is_r_hex_form():
    assert xo._render(Fraction(1, 2)) == "0x1p-1" and xo._render(Fraction(0)) == "0x0p+0" and xo._render(Fraction(-3, 8)) == "-0x1.8p-2"
    assert xo._render("NA") == "NA" and xo._render(None) is None


# ------------------------------------------------------------------------------------------------------------------ partial ranking
F = Fraction


def t2_pairs(plan, failed=("E5",), relabel=False, score_failed=False):
    rows = []
    for (g, x), planned in sorted(plan.items()):
        if not planned:
            rows.append(Pair(g, x, State.STRUCTURAL))
        elif x in failed:
            rows.append(Pair(g, x, State.SCORED, F(1, 7), False) if score_failed else Pair(g, x, State.STRUCTURAL if relabel else State.FAILED))
        else:
            rows.append(Pair(g, x, State.SCORED, F(int(g[1:]) * 3 + "E1E2E3".index(x) + 1, 1000), False))
    return rows


def test_the_partial_ranking_keeps_the_universe_and_marks_unscored(tmp_path):
    plan = t2_plan()
    out = xo.partial_ranking(t2_pairs(plan), plan, t2_statuses(tmp_path), ExposureFailurePolicy())
    assert out["status"] == "exploratory_conditional_on_estimability" and out["definition"] == xo.PARTIAL_RANKING_DEFINITION
    assert out["unscored"] == ["G31"] and out["planned_universe"] == ["G%02d" % i for i in range(1, 32)]
    assert out["missing_score_representation"] == "unscored" and out["conditioned_on_exposures"] == ["E1", "E2", "E3"]
    assert [r.gene for r in out["ranked"]] == ["G%02d" % i for i in range(1, 31)]          # G31 never receives a score
    assert all(r.score < 1 for r in out["ranked"])
    g01 = out["ranked"][0]
    # E4 and E6 are structural for G01; E5 FAILED -- counted separately, never as structural
    assert (g01.tested_pairs, g01.structural_pairs) == (3, 2) and out["estimation_failed_pairs_per_gene"]["G01"] == 1
    assert out["estimation_failed_pairs_per_gene"] == {"G%02d" % i: 1 for i in range(1, 32)}


@pytest.mark.parametrize("kw, code", [({"relabel": True}, "failed_exposure_relabelled"), ({"score_failed": True}, "failed_exposure_has_scores")])
def test_a_failed_exposure_is_never_relabelled_or_scored(tmp_path, kw, code):
    plan = t2_plan()
    assert reason(lambda: xo.partial_ranking(t2_pairs(plan, **kw), plan, t2_statuses(tmp_path), ExposureFailurePolicy())) == code


def test_the_partial_ranking_is_refused_when_the_run_is_refused_or_not_preregistered(tmp_path):
    plan = t2_plan()
    st = t2_statuses(tmp_path)
    st["E3"] = {"status": S.INFRASTRUCTURE_ERROR, "reason": "abnormal_exit", "event": None}
    assert reason(lambda: xo.partial_ranking(t2_pairs(plan), plan, st, ExposureFailurePolicy())) == "partial_ranking_refused"
    stricter = ExposureFailurePolicy(diagnostic=FailureDiagnostic(partial_ranking_allowed=False))
    assert reason(lambda: xo.partial_ranking(t2_pairs(plan), plan, t2_statuses(tmp_path / "b"), stricter)) == "partial_ranking_not_preregistered"
    assert reason(lambda: xo.partial_ranking(t2_pairs(plan), plan, t2_statuses(tmp_path / "c"), None)) == "policy_type"


def test_the_partial_ranking_requires_every_plan_cell_exactly_once(tmp_path):
    plan, st, pol = t2_plan(), t2_statuses(tmp_path), ExposureFailurePolicy()
    rows = t2_pairs(plan)
    assert reason(lambda: xo.partial_ranking(rows[:-1], plan, st, pol)) == "missing_pair_record"
    assert reason(lambda: xo.partial_ranking(rows + rows[:1], plan, st, pol)) == "duplicate_pair"
    assert reason(lambda: xo.partial_ranking(rows + [Pair("G99", "E1", State.STRUCTURAL)], plan, st, pol)) == "unexpected_pair"
    assert reason(lambda: xo.partial_ranking(rows[:-1] + [("G31", "E6")], plan, st, pol)) == "pair_type"
    broken = [Pair(p.gene, p.exposure, State.FAILED) if (p.gene, p.exposure) == ("G01", "E1") else p for p in rows]
    assert reason(lambda: xo.partial_ranking(broken, plan, st, pol)) == "incomplete_method_execution"   # a SCORED exposure lost a score


def test_when_every_eligible_exposure_fails_every_gene_is_unscored():
    plan = {("g", "a"): True, ("h", "a"): True}
    st = _statuses(a=S.MIXTURE_ESTIMATE_INVALID)
    out = xo.partial_ranking([Pair("g", "a", State.FAILED), Pair("h", "a", State.FAILED)], plan, st, ExposureFailurePolicy())
    assert out["ranked"] == () and out["unscored"] == ["g", "h"] and out["conditioned_on_exposures"] == []


def test_PROPERTY_the_partial_ranking_is_the_rule_restricted_to_scored_exposures():
    """On 300 random plans and outcomes: the ranked genes are exactly those with a planned score in a SCORED exposure, each with its
    minimum over those exposures; every other planned gene is "unscored"; nothing from a failed exposure ever enters."""
    rng = random.Random(20261008)
    pol = ExposureFailurePolicy()
    for _ in range(300):
        exposures = ["x%d" % j for j in range(rng.randint(1, 5))]
        kinds = {x: rng.choice([S.SCORED, S.SCORED, S.MIXTURE_ESTIMATE_INVALID, S.NONPOSITIVE_WEIGHT_SUM, S.STRUCTURALLY_INELIGIBLE])
                 for x in exposures}
        if all(k is S.STRUCTURALLY_INELIGIBLE for k in kinds.values()):
            kinds[exposures[0]] = S.SCORED
        genes = ["g%d" % i for i in range(rng.randint(1, 8))]
        plan = {(g, x): (kinds[x] is not S.STRUCTURALLY_INELIGIBLE and rng.random() < 0.7) for g in genes for x in exposures}
        if not any(plan.values()):
            continue
        rows, best = [], {}
        for (g, x), planned in plan.items():
            if not planned:
                rows.append(Pair(g, x, State.STRUCTURAL))
            elif kinds[x] is S.SCORED:
                s = F(rng.randint(1, 50), 50)
                rows.append(Pair(g, x, State.SCORED, s, rng.random() < 0.5))
                best[g] = min(best.get(g, (2, "")), (s, x))
            else:
                rows.append(Pair(g, x, State.FAILED))
        out = xo.partial_ranking(rows, plan, _statuses(**kinds), pol)
        assert {r.gene: (r.score, r.best_exposure) for r in out["ranked"]} == best
        universe = {g for (g, _), p in plan.items() if p}
        assert out["planned_universe"] == sorted(universe) and out["unscored"] == sorted(universe - set(best))
        assert [r.gene for r in out["ranked"]] == sorted(best, key=lambda g: (best[g][0], g))
        if best:
            sub = {c: v for c, v in plan.items() if kinds[c[1]] is not S.MIXTURE_ESTIMATE_INVALID
                   and kinds[c[1]] is not S.NONPOSITIVE_WEIGHT_SUM and c[0] in best}
            assert out["ranked"] == rank_genes([r for r in rows if (r.gene, r.exposure) in sub], sub)


# ------------------------------------------------------------------------------------------------------------------ release
@pytest.mark.parametrize("primary, stage, ranking, evaluation, partial", [
    (xo.PRIMARY_COMPLETE, "confirmatory", "released", "released", "exploratory"),
    (xo.PRIMARY_COMPLETE, "feasibility", "computed_not_evaluated", "withheld_feasibility_stage", "exploratory"),
    (xo.PRIMARY_WITHHELD, "confirmatory", "withheld_incomplete_exposures", "withheld_incomplete_exposures", "exploratory"),
    (xo.PRIMARY_WITHHELD, "feasibility", "withheld_incomplete_exposures", "withheld_incomplete_exposures", "exploratory"),
    (xo.PRIMARY_REFUSED, "confirmatory", "refused", "refused", "refused"),
    (xo.PRIMARY_REFUSED, "feasibility", "refused", "refused", "refused"),
])
def test_endpoint_release_table(primary, stage, ranking, evaluation, partial):
    r = xo.endpoint_release(primary, stage)
    assert (r["primary_ranking"], r["reference_recovery"], r["delta_h20"], r["partial_ranking"]) == (ranking, evaluation, evaluation, partial)
    assert r["policy_selection"] == "not_permitted" and r["stage"] == stage


def test_endpoint_release_refuses_unknown_inputs():
    assert reason(lambda: xo.endpoint_release("partial", "feasibility")) == "primary_status"
    assert reason(lambda: xo.endpoint_release(xo.PRIMARY_COMPLETE, "exploratory")) == "stage"
