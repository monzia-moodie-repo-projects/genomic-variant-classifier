"""Per-exposure outcomes, coverage and the primary / diagnostic split (owner rulings 2026-10-08g, 2026-10-09).

The trace lines and burden-input files below are the REAL output of scripts/dandelion/dandelion_exposure_recorder.R (version 2) on
fixture T2 (DANDELION 0.1.0, R 4.3.3, development sandbox, 2026-10-09), copied verbatim -- their file digests are pinned below; every
mixture quantity equals the version-1 output of 2026-10-08 bit for bit (the recorder change did not touch the computation). The rest
are constructed to exercise each refusal.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from fractions import Fraction

import pytest

from genomic_variant_classifier.inference import exposure_outcomes as xo
from genomic_variant_classifier.inference.analysis_contract import ExposureFailurePolicy, FailureDiagnostic
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.ranking import Pair, State, rank_genes

S = xo.ExposureStatus
REAL_T2 = [
    '{"exposure_id":"E1","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":30,"pi0a":"0x1.bfd3c9c5623ap-1","pi0b":"0x1.c86b0bd2cbe9bp-1","wg1":"0x1.84ec47bbf77f9p-4","wg2":"0x1.c9a6582744fd1p-4","wg3":"0x1.8f3640cde34a1p-1","wg_sum":"0x1.f90894ca4ad9ap-1","burden_input":"burden-0001","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/2"}',
    '{"exposure_id":"E2","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":30,"pi0a":"0x1.f716a749f10cbp-1","pi0b":"0x1.c86b0bd2cbe9bp-1","wg1":"0x1.b4ea5b90194dbp-4","wg2":"0x1.fc6bfeb781ad4p-7","wg3":"0x1.c0795bd7ede3p-1","wg_sum":"0x1.ff085744cf137p-1","burden_input":"burden-0001","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/2"}',
    '{"exposure_id":"E3","outcome":"scored","last_guard_reached":4,"n_trans":31,"n_valid":9,"pi0a":"0x1.996dbad302264p-1","pi0b":"0x1.05ab77eb75ad7p-1","wg1":"0x1.905c5a168261fp-2","wg2":"0x1.a35f511da5c15p-4","wg3":"0x1.a27f1b8f81ea9p-2","wg_sum":"0x1.cdd9a4f6b6de6p-1","burden_input":"burden-0002","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/2"}',
    '{"exposure_id":"E4","outcome":"fewer_than_2_valid_genes","last_guard_reached":2,"n_trans":31,"n_valid":1,"pi0a":null,"pi0b":null,"wg1":null,"wg2":null,"wg3":null,"wg_sum":null,"burden_input":null,"observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/2"}',
    '{"exposure_id":"E5","outcome":"mixture_estimate_invalid","last_guard_reached":3,"n_trans":31,"n_valid":31,"pi0a":"-0x1.e7bab1690b2p-7","pi0b":"0x1.b8eb92c088c7dp-1","wg1":null,"wg2":null,"wg3":null,"wg_sum":null,"burden_input":"burden-0003","observation_kind":"actual_call_trace","recorder_version":"gvc.dandelion-exposure-recorder/2"}',
]
#: T2's burden p-values after clamp_p, gene by gene, exactly as the recorder wrote them (the three real files are prefixes of this list)
BURDEN_31 = [("G01", "0x1.0c6f7a0b5ed8dp-20"), ("G02", "0x1.a36e2eb1c432dp-14"), ("G03", "0x1.0624dd2f1a9fcp-10"), ("G04", "0x1.47ae147ae147bp-7"),
             ("G05", "0x1.838b6be9f1d25p-5"), ("G06", "0x1.5a95a95a95a95p-4"), ("G07", "0x1.f3659cc032698p-4"), ("G08", "0x1.461ac812e794ep-3"),
             ("G09", "0x1.9282c1c5b5f5p-3"), ("G10", "0x1.deeabb7884551p-3"), ("G11", "0x1.15a95a95a95a9p-2"), ("G12", "0x1.3bdd576f108aap-2"),
             ("G13", "0x1.6211544877babp-2"), ("G14", "0x1.88455121deeacp-2"), ("G15", "0x1.ae794dfb461acp-2"), ("G16", "0x1.d4ad4ad4ad4adp-2"),
             ("G17", "0x1.fae147ae147aep-2"), ("G18", "0x1.108aa243bdd57p-1"), ("G19", "0x1.23a4a0b0716d8p-1"), ("G20", "0x1.36be9f1d25058p-1"),
             ("G21", "0x1.49d89d89d89d8p-1"), ("G22", "0x1.5cf29bf68c359p-1"), ("G23", "0x1.700c9a633fcd9p-1"), ("G24", "0x1.832698cff365ap-1"),
             ("G25", "0x1.9640973ca6fdap-1"), ("G26", "0x1.a95a95a95a95ap-1"), ("G27", "0x1.bc7494160e2dbp-1"), ("G28", "0x1.cf8e9282c1c5bp-1"),
             ("G29", "0x1.e2a890ef755dbp-1"), ("G30", "0x1.f5c28f5c28f5cp-1"), ("G31", "0x1.fae147ae147aep-1")]


def burden_bytes(rows) -> bytes:
    return "".join("{}\t{}\n".format(g, v) for g, v in rows).encode("ascii")


REAL_BURDEN = {"burden-0001": burden_bytes(BURDEN_31[:30]), "burden-0002": burden_bytes(BURDEN_31[:9]), "burden-0003": burden_bytes(BURDEN_31)}
#: SHA-256 of the three files the real run wrote (measured with sha256sum, 2026-10-09)
REAL_BURDEN_SHA256 = {"burden-0001": "df548f07ab63ea75865edd433def38847a837d310ef16232de9521bcd7b8d83e",
                      "burden-0002": "933c95cfe37d0b9dc9d32e7fe85b3e51161e0f4125b51d6eb1ae238e1d9f7608",
                      "burden-0003": "b3b5779dd4e45d0526ec3df4e9ee5863be62fdb941686e1156f5f4b5b4269572"}
T2_EXPOSURES = ("E1", "E2", "E3", "E4", "E5", "E6")
T2_EXCLUDED = {"E4": "fewer_than_2_valid_genes", "E6": "exposure_not_annotated"}
T2_USABLE = {"E1": (31, 30), "E2": (31, 30), "E3": (31, 9), "E4": (31, 1), "E5": (31, 31)}
T2_SUPPORT = {"E1": tuple("G%02d" % i for i in range(1, 31)), "E2": tuple("G%02d" % i for i in range(1, 31)),
              "E3": tuple("G%02d" % i for i in range(1, 10)), "E5": tuple("G%02d" % i for i in range(1, 32))}


def reason(fn):
    with pytest.raises(InferenceError) as exc:
        fn()
    return exc.value.code


def write_trace(directory, lines, burden=None):
    """The recorder directory: the lines, and the burden files they reference (by default the real ones; `burden` overrides)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "exposures.jsonl").write_bytes("".join(x + "\n" for x in lines).encode("ascii"))
    files = burden if burden is not None else {
        ref: REAL_BURDEN[ref] for x in lines for ref in re.findall(r'"burden_input":"(burden-\d+)"', x) if ref in REAL_BURDEN}
    for name, raw in files.items():
        (directory / (name + ".tsv")).write_bytes(raw)
    return directory


def line(**over):
    """A scored event, edited by keyword. A call that stops before the pi0 guard carries no burden input unless one is given."""
    doc = json.loads(REAL_T2[0])
    doc.update(over)
    if "burden_input" not in over and doc["last_guard_reached"] < 3:
        doc["burden_input"] = None
    return json.dumps(doc, separators=(",", ":"))


def t2_statuses(tmp_path):
    return xo.classify_exposures(T2_EXPOSURES, T2_EXCLUDED, T2_USABLE, xo.read_exposure_trace(write_trace(tmp_path / "x", REAL_T2)),
                                 predicted_support=T2_SUPPORT)


# ------------------------------------------------------------------------------------------------------------------ reading the trace
def test_the_real_trace_parses_exactly(tmp_path):
    events = xo.read_exposure_trace(write_trace(tmp_path / "x", REAL_T2))
    assert [(e.exposure_id, e.outcome, e.last_guard_reached, e.n_trans, e.n_valid) for e in events] == [
        ("E1", "scored", 4, 31, 30), ("E2", "scored", 4, 31, 30), ("E3", "scored", 4, 31, 9),
        ("E4", "fewer_than_2_valid_genes", 2, 31, 1), ("E5", "mixture_estimate_invalid", 3, 31, 31)]
    e5 = events[4]
    assert e5.pi0a == Fraction(float.fromhex("-0x1.e7bab1690b2p-7")) < 0 and e5.wg_sum is None
    assert events[0].pi0b == events[1].pi0b != events[4].pi0b          # the burden side follows the VALID gene set
    # the effective burden inputs: E1 and E2 share ONE file; E3 and E5 have their own; E4 never reached the estimate
    assert [e.burden_input and e.burden_input.input_id for e in events] == ["burden-0001", "burden-0001", "burden-0002", None, "burden-0003"]
    assert events[0].burden_input == events[1].burden_input and events[4].burden_input.genes == T2_SUPPORT["E5"]
    assert {e.burden_input.input_id: e.burden_input.sha256 for e in events if e.burden_input} == REAL_BURDEN_SHA256
    assert events[2].burden_input.values == tuple(Fraction(float.fromhex(v)) for _, v in BURDEN_31[:9])


def test_the_test_constants_are_the_real_recorder_bytes():
    assert {k: hashlib.sha256(v).hexdigest() for k, v in REAL_BURDEN.items()} == REAL_BURDEN_SHA256


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


def _classify(tmp_path, lines, excluded=None, usable=None, exposures=T2_EXPOSURES, support=None):
    events = xo.read_exposure_trace(write_trace(tmp_path / "c", lines))
    return xo.classify_exposures(exposures, T2_EXCLUDED if excluded is None else excluded, T2_USABLE if usable is None else usable, events,
                                 predicted_support=T2_SUPPORT if support is None else support)


def test_a_missing_call_is_unclassified_never_silently_dropped(tmp_path):
    st = _classify(tmp_path, REAL_T2[:4])                                   # E5 never observed
    assert st["E5"]["status"] is S.UNCLASSIFIED_MISSING_EXPOSURE
    st = _classify(tmp_path / "b", [REAL_T2[i] for i in (0, 1, 2, 4)])      # predicted-structural E4 never observed
    assert st["E4"]["status"] is S.UNCLASSIFIED_MISSING_EXPOSURE


def test_an_abnormal_or_unclassified_exit_is_an_infrastructure_error(tmp_path):
    bad = line(exposure_id="E5", outcome="abnormal_exit", last_guard_reached=3, n_valid=31, wg1=None, wg2=None, wg3=None, wg_sum=None,
               burden_input="burden-0003")
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
    (lambda lines: [line(n_trans=30, n_valid=30)] + lines[1:], "usable_pair_count_mismatch"),
])
def test_disagreement_with_the_frozen_rule_refuses(tmp_path, mutate, code):
    assert reason(lambda: _classify(tmp_path, mutate(list(REAL_T2)))) == code


def test_a_usable_count_that_differs_from_the_prediction_refuses(tmp_path):
    """A self-consistent call (29 recorded genes, n_valid 29) whose count differs from the prediction (30): the frozen rule does not
    describe the call. (With recorder version 2 an event claiming 29 while its burden file holds 30 genes is refused earlier, by the
    reader, as internally inconsistent.)"""
    lines = [line(n_valid=29)] + REAL_T2[2:]
    burden = dict(REAL_BURDEN, **{"burden-0001": burden_bytes(BURDEN_31[:29])})
    events = xo.read_exposure_trace(write_trace(tmp_path / "c", lines, burden=burden))
    assert reason(lambda: xo.classify_exposures(T2_EXPOSURES, T2_EXCLUDED, T2_USABLE, events, predicted_support=T2_SUPPORT)) == \
        "usable_pair_count_mismatch"
    assert reason(lambda: xo.read_exposure_trace(write_trace(tmp_path / "d", [line(n_valid=29)]))) == "exposure_trace_inconsistent"


def test_a_structural_outcome_for_a_predicted_eligible_exposure_is_a_rule_mismatch(tmp_path):
    usable = dict(T2_USABLE, E5=(31, 1))           # counts agree; the CATEGORY does not
    lines = REAL_T2[:4] + [line(exposure_id="E5", outcome="fewer_than_2_valid_genes", last_guard_reached=2, n_valid=1, pi0a=None, pi0b=None,
                                wg1=None, wg2=None, wg3=None, wg_sum=None)]
    support = dict(T2_SUPPORT, E5=("G01",))
    assert reason(lambda: _classify(tmp_path, lines, usable=usable, support=support)) == "eligibility_rule_mismatch"
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
    assert reason(lambda: xo.classify_exposures(*args, (), predicted_support={})) == code


def test_the_events_must_be_parsed_events():
    assert reason(lambda: xo.classify_exposures(("E1",), {}, {"E1": (1, 1)}, ("not an event",), predicted_support={"E1": ("g",)})) == \
        "exposure_event_type"


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
               wg3=None, wg_sum=None, burden_input="burden-0003")
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


# ------------------------------------------------------------------------------------------------------------------ recorder v2 (2026-10-09)
B30 = REAL_BURDEN["burden-0001"]


def _rows_with(index, replacement):
    rows = [list(r) for r in BURDEN_31[:30]]
    rows[index] = replacement
    return "".join("\t".join(r) + "\n" for r in rows).encode("ascii")


@pytest.mark.parametrize("lines, burden, code", [
    ([line(burden_input=7)], {}, "exposure_trace_value"),
    ([line(burden_input="burden-1")], {}, "exposure_trace_value"),
    ([line(burden_input=None)], {}, "exposure_trace_inconsistent"),                           # the pi0 guard was reached: input required
    ([line(outcome="fewer_than_2_valid_genes", last_guard_reached=2, n_valid=1, pi0a=None, pi0b=None, wg1=None, wg2=None, wg3=None,
           wg_sum=None, burden_input="burden-0001")], {"burden-0001": B30}, "exposure_trace_inconsistent"),   # never reached it: none allowed
    ([line(burden_input="burden-0002")], {"burden-0002": B30}, "exposure_trace_burden_sequence"),   # numbered in order of first reference
    ([line()], {}, "exposure_trace_burden_missing"),
    ([line()], {"burden-0001": B30, "burden-0002": REAL_BURDEN["burden-0002"]}, "exposure_trace_burden_orphan"),
    ([line(), line(exposure_id="E3", burden_input="burden-0002")], {"burden-0001": B30, "burden-0002": B30}, "exposure_trace_burden_duplicate"),
    ([line()], {"burden-0001": burden_bytes(BURDEN_31[:29])}, "exposure_trace_inconsistent"),        # 29 genes for 30 usable pairs
    ([line()], {"burden-0001": B30.replace(b"\n", b"\r\n")}, "exposure_trace_burden_file"),
    ([line()], {"burden-0001": B30[:-1]}, "exposure_trace_burden_file"),
    ([line()], {"burden-0001": b""}, "exposure_trace_burden_file"),
    ([line()], {"burden-0001": _rows_with(3, ["G04", "0x1p-1", "x"])}, "exposure_trace_burden_file"),
    ([line()], {"burden-0001": _rows_with(3, ["", "0x1p-1"])}, "exposure_trace_burden_file"),
    ([line()], {"burden-0001": _rows_with(3, ["G01", "0x1p-1"])}, "exposure_trace_burden_file"),      # a duplicate gene
    ([line()], {"burden-0001": _rows_with(3, ["G04", "NA"])}, "exposure_trace_burden_value"),
    ([line()], {"burden-0001": _rows_with(3, ["G04", "0x1p+0"])}, "exposure_trace_burden_value"),      # 1 is outside clamp_p's range
    ([line()], {"burden-0001": _rows_with(3, ["G04", "0x0p+0"])}, "exposure_trace_burden_value"),
    ([line()], {"burden-0001": _rows_with(3, ["G04", "0.5"])}, "exposure_trace_value"),
    ([line()], {"burden-0001": _rows_with(3, ["G04", "0x1.00000000000001p-1"])}, "exposure_trace_value_not_binary64"),
    ([line()], {"burden-0001": B30.replace(b"G04", "Gö4".encode("utf-8"))}, "exposure_trace_encoding"),
])
def test_the_burden_input_record_is_strict(tmp_path, lines, burden, code):
    assert reason(lambda: xo.read_exposure_trace(write_trace(tmp_path / "x", lines, burden=burden))) == code


@pytest.mark.parametrize("name", ["burden-1.tsv", "other.tsv", "burden-0001.txt"])
def test_a_file_the_recorder_does_not_write_is_refused(tmp_path, name):
    d = write_trace(tmp_path / "x", REAL_T2)
    (d / name).write_bytes(B30)
    assert reason(lambda: xo.read_exposure_trace(d)) == "exposure_trace_unexpected_files"


def test_the_recorded_support_must_be_the_planned_one(tmp_path):
    """Equal counts are not equal genes: the plan's ordered valid genes must be EXACTLY the actual call's (order included)."""
    for changed in (tuple(reversed(T2_SUPPORT["E1"])), T2_SUPPORT["E1"][:29] + ("G31",)):
        support = dict(T2_SUPPORT, E1=changed)
        assert reason(lambda: _classify(tmp_path / str(hash(changed)), REAL_T2, support=support)) == "burden_support_mismatch"
    assert reason(lambda: _classify(tmp_path / "a", REAL_T2, support={k: v for k, v in T2_SUPPORT.items() if k != "E5"})) == "support_prediction_scope"
    assert reason(lambda: _classify(tmp_path / "b", REAL_T2, support=dict(T2_SUPPORT, E4=("G01",)))) == "support_prediction_scope"
    assert reason(lambda: _classify(tmp_path / "c", REAL_T2, support=dict(T2_SUPPORT, E3=T2_SUPPORT["E3"][:8]))) == "support_prediction_scope"
    assert reason(lambda: _classify(tmp_path / "d", REAL_T2, support=dict(T2_SUPPORT, E3=list(T2_SUPPORT["E3"])))) == "support_prediction_scope"


def test_the_trace_identity_covers_every_file(tmp_path):
    d = write_trace(tmp_path / "x", REAL_T2)
    xo.read_exposure_trace(d)
    first = xo.exposure_trace_sha256(d)
    assert first == xo.exposure_trace_sha256(d) and len(first) == 64
    (d / "burden-0002.tsv").write_bytes(REAL_BURDEN["burden-0002"].replace(b"G09", b"G99"))
    assert xo.exposure_trace_sha256(d) != first
    (d / "sub").mkdir()
    assert reason(lambda: xo.exposure_trace_sha256(d)) == "exposure_trace_unexpected_files"


# ------------------------------------------------------------------------------------------------------------------ the frozen plan
def t2_pair_plan(**over):
    kw = dict(exposures=T2_EXPOSURES, genes=tuple("G%02d" % i for i in range(1, 32)),
              scored=frozenset(c for c, v in t2_plan().items() if v), exclusions=tuple(sorted(T2_EXCLUDED.items())),
              usable=tuple(sorted((x, t, v) for x, (t, v) in T2_USABLE.items())), support=tuple(sorted(T2_SUPPORT.items())))
    kw.update(over)
    return xo.PairPlan(**kw)


#: the canonical identity of T2's plan (regression pin: a change of the rendering or of the plan is visible here)
T2_PLAN_SHA256 = "3cf569ff60d5c174fd3f2200eeabde93411e810ad80a8861fe9ce363703b3f11"


def test_the_pair_plan_is_the_plan_with_one_identity():
    plan = t2_pair_plan()
    assert plan.as_dict() == t2_plan() and plan.support_dict() == T2_SUPPORT and plan.usable_dict() == T2_USABLE
    assert plan.exclusions_dict() == T2_EXCLUDED and plan.sha256 == T2_PLAN_SHA256
    doc = json.loads(plan.render())
    assert plan.render() == (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode() and doc["schema"] == "gvc.dandelion-pair-plan"
    moved = t2_pair_plan(scored=plan.scored - {("G31", "E5")}, support=tuple(sorted(dict(T2_SUPPORT, E5=T2_SUPPORT["E5"][:30]).items())),
                         usable=tuple(sorted((x, t, v) for x, (t, v) in dict(T2_USABLE, E5=(31, 30)).items())))
    assert moved.sha256 != plan.sha256


@pytest.mark.parametrize("over, code", [
    ({"exposures": ()}, "pair_plan_exposures"),
    ({"exposures": T2_EXPOSURES + ("E1",)}, "pair_plan_exposures"),
    ({"exposures": list(T2_EXPOSURES)}, "pair_plan_exposures"),
    ({"genes": ()}, "pair_plan_genes"),
    ({"scored": set()}, "pair_plan_scored"),
    ({"scored": frozenset({("G99", "E1")})}, "pair_plan_scored"),
    ({"exclusions": (("E6", "exposure_not_annotated"), ("E4", "fewer_than_2_valid_genes"))}, "pair_plan_exclusions"),
    ({"exclusions": (("E4", "too_few"), ("E6", "exposure_not_annotated"))}, "pair_plan_exclusions"),
    ({"usable": tuple(r for r in sorted((x, t, v) for x, (t, v) in T2_USABLE.items()) if r[0] != "E3")}, "pair_plan_usable"),
    ({"usable": tuple(sorted((x, t, v) for x, (t, v) in dict(T2_USABLE, E4=(31, 2)).items()))}, "pair_plan_usable"),
    ({"usable": tuple(sorted((x, t, v) for x, (t, v) in dict(T2_USABLE, E1=(29, 30)).items()))}, "pair_plan_usable"),
    ({"exclusions": (("E4", "no_trans_genes"), ("E6", "exposure_not_annotated"))}, "pair_plan_usable"),
    ({"support": tuple(sorted(dict(T2_SUPPORT, E4=("G01",)).items()))}, "pair_plan_support"),
    ({"support": tuple(sorted(dict(T2_SUPPORT, E3=T2_SUPPORT["E3"][:8] + ("G10",)).items()))}, "pair_plan_support"),
    ({"support": tuple(sorted(dict(T2_SUPPORT, E3=T2_SUPPORT["E3"][:8]).items()))}, "pair_plan_support"),
    ({"support": tuple(sorted({k: v for k, v in T2_SUPPORT.items() if k != "E2"}.items()))}, "pair_plan_support"),
])
def test_the_pair_plan_invariants(over, code):
    assert reason(lambda: t2_pair_plan(**over)) == code


# ------------------------------------------------------------------------------------------------------------------ the coverage rule
SMALL = {("g", "a"): True, ("h", "a"): True, ("g", "b"): True, ("h", "b"): False, ("g", "c"): False, ("h", "c"): False}
F_ = Fraction(1, 10)


def _scores(**cells):
    out = {c: None for c in SMALL}
    for key, value in cells.items():
        out[tuple(key.split("_"))] = value
    return out


COMPLETE_SCORES = _scores(g_a=F_, h_a=F_, g_b=F_)


@pytest.mark.parametrize("kinds, scores, expected", [
    # the ruling's acceptance table
    ({"a": S.SCORED, "b": S.SCORED, "c": S.STRUCTURALLY_INELIGIBLE}, COMPLETE_SCORES, ("complete", ())),
    ({"a": S.SCORED, "b": S.UNCLASSIFIED_MISSING_EXPOSURE, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_a=F_),
     ("refused", ("execution_or_classification_failure",))),
    ({"a": S.SCORED, "b": S.INFRASTRUCTURE_ERROR, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_a=F_),
     ("refused", ("execution_or_classification_failure",))),
    ({"a": S.SCORED, "b": S.SCORED, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, g_b=F_),        # "scored" a silently omits h
     ("refused", ("scores_disagree_with_exposure_outcomes",))),
    ({"a": S.SCORED, "b": S.SCORED, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_a=F_, g_b=F_, h_b=F_),   # h_b is excluded
     ("refused", ("unexpected_scored_pair",))),
    ({"a": S.SCORED, "b": S.MIXTURE_ESTIMATE_INVALID, "c": S.STRUCTURALLY_INELIGIBLE}, COMPLETE_SCORES,   # a failed b has a score
     ("refused", ("scores_disagree_with_exposure_outcomes",))),
    ({"a": S.SCORED, "b": S.MIXTURE_ESTIMATE_INVALID, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_a=F_),
     ("withheld", ("incomplete_eligible_exposures",))),
    ({"a": S.SCORED, "b": S.NONPOSITIVE_WEIGHT_SUM, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_a=F_),
     ("withheld", ("incomplete_eligible_exposures",))),
    ({"a": S.SCORED, "b": S.SCORED, "c": S.STRUCTURALLY_INELIGIBLE}, {c: v for c, v in COMPLETE_SCORES.items() if c != ("h", "c")},
     ("refused", ("score_matrix_cells",))),
    ({"a": S.SCORED, "b": S.SCORED, "c": S.STRUCTURALLY_INELIGIBLE}, {**COMPLETE_SCORES, ("z", "a"): F_},
     ("refused", ("score_matrix_cells", "unexpected_scored_pair"))),
    # every applicable reason, in order
    ({"a": S.SCORED, "b": S.INFRASTRUCTURE_ERROR, "c": S.STRUCTURALLY_INELIGIBLE}, _scores(g_a=F_, h_b=F_),
     ("refused", ("execution_or_classification_failure", "unexpected_scored_pair", "scores_disagree_with_exposure_outcomes"))),
])
def test_the_coverage_rule(kinds, scores, expected):
    assert xo.score_coverage(SMALL, _statuses(**kinds), scores) == expected


def test_an_empty_eligible_set_is_refused_never_complete():
    plan = {("g", "c"): False}
    assert xo.score_coverage(plan, _statuses(c=S.STRUCTURALLY_INELIGIBLE), {("g", "c"): None}) == ("refused", ("no_eligible_exposures",))


@pytest.mark.parametrize("statuses, scores, code", [
    (_statuses(a=S.SCORED, b=S.SCORED, c=S.STRUCTURALLY_INELIGIBLE, d=S.SCORED), COMPLETE_SCORES, "exposure_status_scope"),
    (_statuses(a=S.SCORED, b=S.SCORED, c=S.STRUCTURALLY_INELIGIBLE), [], "score_matrix_type"),
    (_statuses(a=S.SCORED, b=S.SCORED, c=S.STRUCTURALLY_INELIGIBLE), {**COMPLETE_SCORES, ("g", "a"): 0.1}, "score_matrix_type"),
    (_statuses(a=S.SCORED, b=S.SCORED), COMPLETE_SCORES, "exposure_status_missing"),
])
def test_the_coverage_rule_refuses_malformed_inputs(statuses, scores, code):
    assert reason(lambda: xo.score_coverage(SMALL, statuses, scores)) == code


def test_PROPERTY_the_coverage_rule_agrees_with_coverage_report_on_consistent_evidence():
    """On 400 random plans and outcomes whose scores are exactly where they belong, the derived status IS coverage_report's (one
    completion definition), and removing or adding one score always refuses."""
    rng = random.Random(20261009)
    kinds_all = [S.SCORED, S.SCORED, S.MIXTURE_ESTIMATE_INVALID, S.NONPOSITIVE_WEIGHT_SUM, S.STRUCTURALLY_INELIGIBLE,
                 S.UNCLASSIFIED_MISSING_EXPOSURE, S.INFRASTRUCTURE_ERROR]
    checked = 0
    for _ in range(400):
        exposures = ["x%d" % j for j in range(rng.randint(1, 5))]
        kinds = {x: rng.choice(kinds_all) for x in exposures}
        genes = ["g%d" % i for i in range(rng.randint(1, 6))]
        plan = {(g, x): kinds[x] is not S.STRUCTURALLY_INELIGIBLE and rng.random() < 0.7 for g in genes for x in exposures}
        if not any(plan.values()):
            continue
        st = _statuses(**kinds)
        scores = {c: (Fraction(rng.randint(1, 9), 10) if p and kinds[c[1]] is S.SCORED else None) for c, p in plan.items()}
        status, reasons = xo.score_coverage(plan, st, scores)
        assert status == xo.coverage_report(plan, st)["primary_status"]
        if status != "refused":
            planned = [c for c, p in plan.items() if p]
            c = rng.choice(planned)
            flipped = {**scores, c: None if scores[c] is not None else Fraction(1, 2)}
            assert xo.score_coverage(plan, st, flipped)[0] == "refused"
            checked += 1
    assert checked > 50


# ------------------------------------------------------------------------------------------------------------------ the release policy
#: The release-policy identity a run intent binds. Changing the table or its rules changes this digest: that is an AMENDMENT (a new
#: policy identity), never an edit of this pin to make a test pass.
RELEASE_POLICY_SHA256 = "b56f33dcec5f68f55633487cf917be31bda0a677bbddf0f07cdc2c43e42f3abe"


def test_the_release_policy_has_its_registered_identity():
    assert xo.release_policy_sha256() == RELEASE_POLICY_SHA256
    doc = json.loads(xo.render_release_policy())
    assert doc == xo.release_policy() and doc["schema"] == "gvc.endpoint-release-policy" and doc["policy_selection"] == "not_permitted"
    assert {(d["primary_status"], d["stage"]) for d in doc["decisions"]} == {(p, s) for p in ("complete", "withheld", "refused") for s in xo.STAGES}
    assert len(doc["decisions"]) == 6 and all(d["reference_recovery"] == d["delta_h20"] for d in doc["decisions"])


def test_endpoint_release_reads_the_one_table_and_names_its_identity():
    for d in xo.release_policy()["decisions"]:
        r = xo.endpoint_release(d["primary_status"], d["stage"])
        assert {k: r[k] for k in d} == d and r["release_policy_sha256"] == RELEASE_POLICY_SHA256
    assert type(xo._RELEASE_TABLE) is tuple and all(type(row) is tuple for row in xo._RELEASE_TABLE)
    # feasibility NEVER releases a reference recovery, whatever the completion
    assert all(d["reference_recovery"] != "released" for d in xo.release_policy()["decisions"] if d["stage"] == "feasibility")


# ------------------------------------------------------------------------------------------------------------------ shared burden inputs
ENV = "e" * 64


def test_the_real_t2_burden_summary(tmp_path):
    s = xo.burden_input_summary(t2_plan(), t2_statuses(tmp_path), ENV)
    by = {tuple(g["exposures"]): g for g in s["groups"]}
    assert set(by) == {("E1", "E2"), ("E3",), ("E5",)}
    shared = by[("E1", "E2")]
    assert (shared["valid_genes"], shared["exposure_count"], shared["burden_estimate"], shared["agreement"], shared["burden_side"],
            shared["outcomes"], shared["failed_exposures"], shared["genes_losing_coverage_count"], shared["burden_input_files"]) == (
        30, 2, ["0x1.c86b0bd2cbe9bp-1"], "identical", "estimate_valid", {"scored": 2}, [], 0, ["burden-0001"])
    e5 = by[("E5",)]
    assert e5["outcomes"] == {"mixture_estimate_invalid": 1} and e5["burden_side"] == "estimate_valid"
    assert e5["failure_counts"] == {"burden_side_invalid": 0, "trans_side_only_invalid": 1, "nonpositive_weight_sum": 0, "other_unscored": 0}
    assert e5["genes_losing_coverage"] == ["G%02d" % i for i in range(1, 32)]
    assert by[("E3",)]["valid_genes"] == 9
    assert s["exposures_not_reaching_burden_estimation"] == {"E4": "structurally_ineligible", "E6": "structurally_ineligible"}
    assert [v["support_groups"] for v in s["numerical_inputs"]] == [1, 1, 1] and s["investigate"] == []
    assert s["totals"] == {"support_groups": 3, "exposures_reaching_burden_estimation": 4, "burden_side_failure_groups": 0,
                           "exposures_with_burden_side_failure": 0, "genes_losing_coverage_through_burden_side_failure": 0,
                           "exposures_unscored_after_reaching_the_estimate": 1, "genes_losing_coverage_any_cause_among_these": 31}
    assert s["preprocessing"] == xo.BURDEN_PREPROCESSING and s["estimator"] == xo.BURDEN_ESTIMATOR and s["environment_sha256"] == ENV


def _two_exposure_trace(tmp_path, second_genes, second_pi0b="0x1.c86b0bd2cbe9bp-1", first_pi0b="0x1.c86b0bd2cbe9bp-1", second_file=None):
    first = line(exposure_id="E1", pi0b=first_pi0b)
    second = line(exposure_id="E2", pi0b=second_pi0b, burden_input=second_file or "burden-0001")
    files = {"burden-0001": B30}
    if second_file:
        files[second_file] = burden_bytes([(g, v) for g, (_, v) in zip(second_genes, BURDEN_31[:30])])
    events = xo.read_exposure_trace(write_trace(tmp_path / "t", [first, second], burden=files))
    plan = {(g, x): True for x, genes in (("E1", BURDEN_31[:30]), ("E2", [(g, None) for g in second_genes])) for g, _ in genes}
    for x in ("E1", "E2"):
        for g in {g for g, _ in plan}:
            plan.setdefault((g, x), False)
    support = {"E1": tuple(g for g, _ in BURDEN_31[:30]), "E2": tuple(second_genes)}
    statuses = xo.classify_exposures(("E1", "E2"), {}, {"E1": (31, 30), "E2": (31, 30)}, events, predicted_support=support)
    return plan, statuses


def test_identical_values_with_different_genes_are_one_numerical_problem_but_two_supports(tmp_path):
    other = ["H%02d" % i for i in range(1, 31)]
    plan, st = _two_exposure_trace(tmp_path, other, second_file="burden-0002")
    s = xo.burden_input_summary(plan, st, ENV)
    assert len(s["groups"]) == 2 and s["numerical_inputs"] == [{"numeric_input_sha256": s["groups"][0]["numeric_input_sha256"],
                                                                "support_groups": 2, "exposures": 2}]
    assert s["groups"][0]["support_sha256"] != s["groups"][1]["support_sha256"]


def test_a_disagreement_between_identical_inputs_is_flagged_never_averaged(tmp_path):
    genes = [g for g, _ in BURDEN_31[:30]]
    plan, st = _two_exposure_trace(tmp_path, genes, second_pi0b="0x1.8p-1")
    s = xo.burden_input_summary(plan, st, ENV)
    (group,) = s["groups"]
    assert group["agreement"] == "disagreement_investigate" and group["burden_side"] == "undetermined_disagreement"
    assert group["burden_estimate"] == ["0x1.8p-1", "0x1.c86b0bd2cbe9bp-1"] and s["investigate"] == [group["support_sha256"]]


def test_a_shared_burden_side_failure_is_one_cause_with_several_consequences(tmp_path):
    genes = [g for g, _ in BURDEN_31[:30]]
    first = line(exposure_id="E1", outcome="mixture_estimate_invalid", last_guard_reached=3, pi0b="-0x1p-9", wg1=None, wg2=None, wg3=None, wg_sum=None)
    second = line(exposure_id="E2", outcome="mixture_estimate_invalid", last_guard_reached=3, pi0b="-0x1p-9", wg1=None, wg2=None, wg3=None, wg_sum=None)
    events = xo.read_exposure_trace(write_trace(tmp_path / "t", [first, second], burden={"burden-0001": B30}))
    plan = {(g, x): True for x in ("E1", "E2") for g in genes}
    st = xo.classify_exposures(("E1", "E2"), {}, {"E1": (31, 30), "E2": (31, 30)}, events, predicted_support={"E1": tuple(genes), "E2": tuple(genes)})
    s = xo.burden_input_summary(plan, st, ENV)
    (group,) = s["groups"]
    assert group["burden_side"] == "estimate_invalid" and group["failure_counts"]["burden_side_invalid"] == 2
    assert s["totals"]["burden_side_failure_groups"] == 1 and s["totals"]["exposures_with_burden_side_failure"] == 2
    assert s["totals"]["genes_losing_coverage_through_burden_side_failure"] == 30


def test_the_two_identities_separate_what_they_should():
    b = xo.BurdenInput("burden-0001", "a" * 64, ("A", "B"), (Fraction(1, 4), Fraction(1, 2)))
    renamed = xo.BurdenInput("burden-0002", "b" * 64, ("A", "C"), b.values)
    revalued = xo.BurdenInput("burden-0003", "c" * 64, b.genes, (Fraction(1, 4), Fraction(3, 4)))
    reordered = xo.BurdenInput("burden-0004", "d" * 64, ("B", "A"), (Fraction(1, 2), Fraction(1, 4)))
    assert xo.burden_support_sha256(b) == xo.burden_support_sha256(xo.BurdenInput("burden-0009", "f" * 64, b.genes, b.values))
    assert xo.burden_support_sha256(renamed) != xo.burden_support_sha256(b)
    assert xo.burden_numeric_input_sha256(renamed, ENV) == xo.burden_numeric_input_sha256(b, ENV)
    assert xo.burden_numeric_input_sha256(b, "f" * 64) != xo.burden_numeric_input_sha256(b, ENV)
    for other in (revalued, reordered):          # ORDER is part of the input: the estimator sums in order
        assert xo.burden_numeric_input_sha256(other, ENV) != xo.burden_numeric_input_sha256(b, ENV)
    assert reason(lambda: xo.burden_numeric_input_sha256(b, "E" * 64)) == "environment_digest"
    assert reason(lambda: xo.burden_support_sha256("burden-0001")) == "burden_input_type"
    assert reason(lambda: xo.burden_input_summary(t2_plan(), {}, ENV)) == "exposure_status_type"
