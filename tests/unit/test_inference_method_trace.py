"""Predetermined DANDELION method fixtures and the adjustment-to-endpoint trace (owner rulings 2026-10-08e, 2026-10-08f section 6,
2026-10-08g).

Pure-Python tests build SYNTHETIC runs (files shaped exactly as scripts/dandelion/method_fixtures.R writes them) so the judge is
exercised everywhere; the integration test runs the real fixtures through R when GVC_METHOD_FIXTURE_RSCRIPT and
GVC_METHOD_FIXTURE_RLIBS name an Rscript and a library holding DANDELION (and qvalue), and skips with that reason otherwise.

Author: Monzia Moodie
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from genomic_variant_classifier.inference import method_trace as mt
from genomic_variant_classifier.inference.backend_trace import RECORDER_VERSION
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import EXPOSURE_RECORDER_VERSION, ExposureStatus, coverage_report
from genomic_variant_classifier.inference.ranking import GeneRank

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "tests" / "fixtures" / "dandelion" / "method_fixtures_v2.json"
#: The pre-registration identity: the specification was frozen before any qualified run. An edit must show up here. (v1, c34617fe...25d1,
#: was frozen on 2026-10-08 and superseded by v2 before ANY run of it: ruling 2026-10-08g changed what the judge must report.)
SPEC_SHA256 = "07af75116dc767a4b791b6f99038a3df143a2264bd703929f075cbcecf70ea98"
COMMIT = "f471153bfa3c0069cd68a67565000889c7cdf5d1"


def rhex(x: float) -> str:
    """R's sprintf("%a") form (no trailing zero digits)."""
    mant, exp = float(x).hex().split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return mant + "p" + exp


def r_bh(p) -> list:
    """R's p.adjust(method = "BH") in R's own binary64 operation order: pmin(1, cummin(n / i * p[o]))[ro]."""
    n = len(p)
    o = sorted(range(n), key=lambda i: p[i], reverse=True)
    out, running = [None] * n, float("inf")
    for k, idx in enumerate(o):
        running = min(running, n / (n - k) * p[idx])
        out[idx] = min(1.0, running)
    return out


def reason(fn):
    with pytest.raises(InferenceError) as exc:
        fn()
    return exc.value.code


# ------------------------------------------------------------------------------------------------------------------ exact values
@pytest.mark.parametrize("text, value", [("0x1p-1", Fraction(1, 2)), ("0x1.8p+1", Fraction(3)), ("0x0p+0", Fraction(0)), ("NA", None),
                                         ("0x0.0000000000001p-1022", Fraction(1, 2 ** 1074)), ("-0x1p-2", Fraction(-1, 4))])
def test_parse_hex_is_exact(text, value):
    assert mt.parse_hex(text) == value


@pytest.mark.parametrize("text", ["0.1", "1", "0x1", "0X1P-1", "0x1p-1\n", "0x1.00000000000001p-1", "0x1.8P+1", ""])
def test_parse_hex_refuses_other_grammar_and_inexact_values(text):
    assert reason(lambda: mt.parse_hex(text)) in ("fixture_value", "fixture_value_not_binary64")


def test_exact_bh_hand_example_ties_and_cap():
    q = mt.exact_bh([Fraction(1, 100), Fraction(4, 100), Fraction(3, 100), Fraction(1, 2)])
    assert q == (Fraction(4, 100), Fraction(4, 75), Fraction(4, 75), Fraction(1, 2))
    assert mt.exact_bh([Fraction(1, 5)] * 3) == (Fraction(1, 5),) * 3
    assert mt.exact_bh([Fraction(9, 10), Fraction(19, 20)]) == (Fraction(19, 20), Fraction(19, 20))


def test_PROPERTY_exact_bh_equals_its_definition():
    """q_i = min(1, min over k >= position of i in ascending order of m p_(k) / k), on 400 random rational vectors with ties."""
    rng = random.Random(20261008)
    for _ in range(400):
        m = rng.randint(1, 12)
        p = [Fraction(rng.randint(0, 20), 20) for _ in range(m)]
        s = sorted(p)
        want = [min([Fraction(1)] + [m * s[k] / (k + 1) for k in range(m) if s[k] >= v]) for v in p]
        assert list(mt.exact_bh(p)) == want


def test_r_floating_bh_stays_within_the_frozen_tolerance_on_random_vectors():
    """The tolerance claim (2^-52 per element, frozen at 2^-50) checked against R's operation order replicated in binary64."""
    tol = Fraction(mt.load_spec(SPEC.read_bytes())["bh_relative_tolerance"])
    rng = random.Random(7)
    worst = Fraction(0)
    for _ in range(300):
        p = [rng.random() or 0.5 for _ in range(rng.randint(1, 60))]
        exact = mt.exact_bh([Fraction(x) for x in p])
        for o, e in zip(r_bh(p), exact):
            worst = max(worst, abs(Fraction(o) - e) / e)
    assert worst <= Fraction(1, 2 ** 52) < tol


def test_significance_is_decided_in_binary64_as_r_decides_it():
    """R compares q <= 0.1 with the DOUBLE 0.1, which exceeds one tenth: a q equal to that double is significant, one ulp above is not."""
    double = Fraction(0.1)
    assert double > Fraction(1, 10)
    assert mt.significant(double, "0.1") and mt.significant(Fraction(1, 10), "0.1")
    assert not mt.significant(Fraction(float.fromhex("0x1.999999999999bp-4")), "0.1")


# ------------------------------------------------------------------------------------------------------------------ the specification
def test_the_frozen_specification_has_its_registered_identity_and_loads():
    raw = SPEC.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SPEC_SHA256
    spec = mt.load_spec(raw)
    assert [f["id"] for f in spec["route_fixtures"]] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    routes = {f["id"]: (f["predicted"]["backend"], f["predicted"]["fallback_reason"]) for f in spec["route_fixtures"]}
    assert routes == {"R1": ("qvalue", None), "R2": ("BH", "qvalue_error"), "R3": ("BH", "fewer_than_10_values"),
                      "R4": ("BH", "fewer_than_4_distinct_values"), "R5": ("qvalue", None), "R6": ("BH", "qvalue_error")}
    assert [t["id"] for t in spec["trace_fixtures"]] == ["T1", "T2"]
    assert [t["predicted"]["primary_status"] for t in spec["trace_fixtures"]] == ["complete", "withheld"]


def test_v2_keeps_every_v1_input_unchanged():
    """v2 supersedes the never-run v1: the six route fixtures and T1's inputs are the v1 inputs, unchanged (only predictions were added)."""
    spec = mt.load_spec(SPEC.read_bytes())
    t1 = spec["trace_fixtures"][0]
    assert hashlib.sha256(json.dumps(spec["route_fixtures"], sort_keys=True).encode()).hexdigest() == V1_ROUTE_FIXTURES_SHA256
    digest = hashlib.sha256(json.dumps({k: t1[k] for k in ("genes", "exposures", "trans", "wes", "ref_table", "k", "reference_positives")},
                                       sort_keys=True).encode()).hexdigest()
    assert digest == T1_INPUT_SHA256


def test_the_fixture_values_obey_the_stated_route_conditions():
    """Each prediction follows from the stated rule (n >= 10, >= 4 distinct post-clamp values, max >= 0.95 for the native route)."""
    spec = mt.load_spec(SPEC.read_bytes())
    xmin, top = Fraction(2) ** -1022, Fraction(1 - 1e-15)
    for f in spec["route_fixtures"]:
        v = [mt.parse_hex(x) for x in f["values"]]
        clamped = [xmin if x <= 0 else top if x >= 1 else x for x in v]
        n, distinct, mx = len(clamped), len(set(clamped)), max(clamped)
        if n < 10:
            rule = ("BH", "fewer_than_10_values")
        elif distinct < 4:
            rule = ("BH", "fewer_than_4_distinct_values")
        elif mx >= Fraction(0.95):
            rule = ("qvalue", None)
        else:
            rule = ("BH", "qvalue_error")
        assert rule == (f["predicted"]["backend"], f["predicted"]["fallback_reason"]), f["id"]
        assert (n, distinct) == (f["predicted"]["n_values"], f["predicted"]["n_distinct"]), f["id"]


#: v1's six route fixtures (inputs AND predictions), canonical JSON
V1_ROUTE_FIXTURES_SHA256 = "9f2b0c4a2945ba9981d2da38a0e810cda6c267f276e09b5fb663149807283c4c"
#: T1's inputs as frozen in v1 (canonical JSON of genes, exposures, trans, wes, ref_table, k, reference_positives)
T1_INPUT_SHA256 = "dbe1f389a0a6e255b840e0104d0da0d4fae8578e5315c9ae2af9ffb8c28a8d86"


def _spec_doc():
    return json.loads(SPEC.read_bytes())


def _raw(doc):
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


@pytest.mark.parametrize("mutate, code", [
    (lambda d: d.__setitem__("extra", 1), "spec_keys"),
    (lambda d: d["route_fixtures"][0]["predicted"].__setitem__("n_values", 3), "spec_prediction_n"),
    (lambda d: d["route_fixtures"][0]["predicted"].__setitem__("fallback_reason", "qvalue_error"), "spec_route"),
    (lambda d: d["route_fixtures"][1].__setitem__("id", "R1"), "spec_fixture_id"),
    (lambda d: d["route_fixtures"][0]["values"].append("0x1p+1"), "spec_values"),
    (lambda d: d["trace_fixtures"][0]["predicted"]["integrated_top_k"].reverse(), "spec_prediction_order"),
    (lambda d: d["trace_fixtures"][0]["predicted"]["excluded_exposures"].__setitem__("E4", "vanished"), "spec_exclusion"),
    (lambda d: d["trace_fixtures"][0]["wes"].pop(), "spec_matrix_shape"),
    (lambda d: d.__setitem__("bh_relative_tolerance", "1/1000"), "spec_tolerance"),
    (lambda d: d["trace_fixtures"][1].__setitem__("id", "T1"), "spec_fixture_id"),
    (lambda d: d["trace_fixtures"][1].__setitem__("id", "R1"), "spec_fixture_id"),
    (lambda d: d["trace_fixtures"][1].__setitem__("purpose", " "), "spec_purpose"),
    (lambda d: d.__setitem__("trace_fixtures", []), "spec_trace_fixtures"),
    (lambda d: d["trace_fixtures"][1]["predicted"].__setitem__("delta_h", 0), "spec_primary_only"),          # a withheld endpoint has no value
    (lambda d: d["trace_fixtures"][1]["predicted"].__setitem__("explained_genes", []), "spec_primary_only"),
    (lambda d: d["trace_fixtures"][0]["predicted"].__setitem__("integrated_top_k", None), "spec_primary_only"),
    (lambda d: d["trace_fixtures"][0]["predicted"].__setitem__("partial_top_k", None), "spec_primary_only"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["exposure_outcomes"].pop("E6"), "spec_exposure_outcomes"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["exposure_outcomes"].__setitem__("E5", "skipped"), "spec_exposure_outcomes"),
    # a mixture failure may never be PREDICTED as structural ineligibility either
    (lambda d: d["trace_fixtures"][1]["predicted"]["exposure_outcomes"].__setitem__("E5", "structurally_ineligible"), "spec_exposure_outcomes"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["mixture_failure_side"].clear(), "spec_mixture_side"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["mixture_failure_side"].__setitem__("E5", "left"), "spec_mixture_side"),
    (lambda d: d["trace_fixtures"][1]["predicted"].__setitem__("primary_status", "partial"), "spec_primary_status"),
    (lambda d: d["trace_fixtures"][1]["predicted"].__setitem__("exposure_completion", "0.75"), "spec_completion"),
    (lambda d: d["trace_fixtures"][1]["predicted"].__setitem__("exposure_completion", "3/0"), "spec_completion"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["gene_coverage_distribution"].__setitem__("2/3", 0), "spec_coverage"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["gene_coverage_distribution"].__setitem__("most", 1), "spec_coverage"),
    (lambda d: d["trace_fixtures"][1]["predicted"]["unscored_genes"].insert(0, "G99"), "spec_prediction_order"),
])
def test_the_specification_is_strict(mutate, code):
    d = _spec_doc()
    mutate(d)
    assert reason(lambda: mt.load_spec(_raw(d))) == code


@pytest.mark.parametrize("raw, code", [(b'{"a": 1.5}', "spec_number"), (b'{"a": 1, "a": 2}', "spec_duplicate_key"),
                                       (b"\xef\xbb\xbf{}", "spec_bytes"), (b"{", "spec_json")])
def test_the_specification_parse_is_strict(raw, code):
    assert reason(lambda: mt.load_spec(raw)) == code


# ------------------------------------------------------------------------------------------------------------------ the pair plan
def test_the_plan_of_the_real_trace_fixtures():
    t1, t2 = mt.load_spec(SPEC.read_bytes())["trace_fixtures"]
    plan, excluded, order = mt.pair_plan(t1)
    assert excluded == {"E4": "fewer_than_2_valid_genes"}
    assert sum(plan.values()) == 30 + 30 + 9 and len(plan) == 30 * 4
    assert list(order) == ["E1", "E2", "E3"] and order["E3"] == tuple("G%02d" % i for i in range(1, 10))
    assert mt.usable_pairs(t1) == {"E1": (30, 30), "E2": (30, 30), "E3": (30, 9), "E4": (30, 1)}
    plan, excluded, order = mt.pair_plan(t2)
    # E5 (every trans p-value 1) is ELIGIBLE: its failure is not predictable from the inputs, so it is planned like any other
    assert excluded == {"E4": "fewer_than_2_valid_genes", "E6": "exposure_not_annotated"}
    assert sum(plan.values()) == 30 + 30 + 9 + 31 and len(plan) == 31 * 6 and list(order) == ["E1", "E2", "E3", "E5"]
    assert mt.usable_pairs(t2) == {"E1": (31, 30), "E2": (31, 30), "E3": (31, 9), "E4": (31, 1), "E5": (31, 31)}


def test_the_coverage_predictions_follow_from_the_plan_and_the_predicted_outcomes():
    """Completion, the per-gene coverage distribution and the unscored genes are DERIVED (the plan from the inputs, the outcomes as
    predicted) -- not tuned: computing them from those two sources reproduces every frozen value."""
    for t in mt.load_spec(SPEC.read_bytes())["trace_fixtures"]:
        plan = mt.pair_plan(t)[0]
        p = t["predicted"]
        statuses = {x: {"status": ExposureStatus(v), "reason": v, "event": None} for x, v in p["exposure_outcomes"].items()}
        cov = coverage_report(plan, statuses)
        assert (cov["exposure_completion"], cov["primary_status"], cov["gene_coverage_distribution"]) == (
            p["exposure_completion"], p["primary_status"], p["gene_coverage_distribution"]), t["id"]
        assert sorted(g for g, r in cov["per_gene_coverage"].items() if r["scored_pairs"] == 0) == p["unscored_genes"], t["id"]


def _mini_trace(**over):
    genes = ["A1", "A2", "A3", "A4"]
    t = {"id": "T9", "genes": genes, "exposures": ["X1"], "trans": {"X1": [rhex(0.01), rhex(0.2), rhex(0.3), rhex(0.4)]},
         "wes": [rhex(0.1)] * 4, "ref_table": [[g, "protein_coding", "chr1", 20_000_000 * (i + 1), 20_000_000 * (i + 1) + 10]
                                                for i, g in enumerate(genes)] + [["X1", "protein_coding", "chr2", 1000, 2000]]}
    t.update(over)
    return t


def test_the_plan_applies_the_cis_window_annotation_and_missingness_rules():
    t = _mini_trace(ref_table=_mini_trace()["ref_table"][:4] + [["X1", "protein_coding", "chr1", 40_000_000, 40_000_010]])
    plan, excluded, order = mt.pair_plan(t)                       # A2 sits at 40 Mb: within 5 Mb of X1 -> cis -> structural
    assert plan[("A2", "X1")] is False and order["X1"] == ("A1", "A3", "A4") and not excluded
    t = _mini_trace(trans={"X1": [rhex(0.01), "NA", "NA", "NA"]})
    assert mt.pair_plan(t)[1] == {"X1": "fewer_than_2_valid_genes"}
    t = _mini_trace(ref_table=_mini_trace()["ref_table"][:4])        # the exposure has no annotation
    assert mt.pair_plan(t)[1] == {"X1": "exposure_not_annotated"}
    t = _mini_trace(ref_table=[[g, "pseudogene", "chr1", 1, 2] for g in ["A1", "A2", "A3", "A4"]] + [["X1", "protein_coding", "chr2", 1, 2]])
    plan, excluded, _ = mt.pair_plan(t)                            # no gene survives the annotation filter
    assert plan == {} and excluded == {"X1": "no_trans_genes"}
    t = _mini_trace(wes=[rhex(0.1), "NA", rhex(0.1), rhex(0.1)])
    assert mt.pair_plan(t)[0][("A2", "X1")] is False


# ------------------------------------------------------------------------------------------------------------------ synthetic runs
GENES = ["A%d" % i for i in range(1, 13)]


def _synthetic_trace(tid: str) -> dict:
    """T1: 12 genes; X1 native, X2 BH, X3 excluded (fewer than 2 valid genes). T2: T1 plus X4, a mixture-estimation failure, and gene
    A13 that only X4 would cover."""
    two = tid == "T2"
    genes = GENES + (["A13"] if two else [])
    exposures = ["X1", "X2", "X3"] + (["X4"] if two else [])
    pad = ["NA"] if two else []
    trans = {"X1": [rhex(0.01)] * 12 + pad, "X2": [rhex(0.02)] * 12 + pad, "X3": [rhex(0.5)] + ["NA"] * 11 + pad}
    if two:
        trans["X4"] = [rhex(1.0)] * 13
    return {"id": tid, "purpose": "synthetic " + tid, "genes": genes, "exposures": exposures, "trans": trans,
            "wes": [rhex(0.001 * (i + 1)) for i in range(12)] + ([rhex(0.5)] if two else []),
            "ref_table": [[g, "protein_coding", "chr1", 20_000_000 * (i + 1), 20_000_000 * (i + 1) + 10] for i, g in enumerate(genes)]
                         + [[x, "protein_coding", "chr%d" % (j + 2), 1000, 2000] for j, x in enumerate(exposures)],
            "k": 3, "reference_id": "ref", "reference_positives": ["A9"], "reference_note": "n", "predicted": {}}


def synthetic_spec():
    """A small spec whose predictions are consistent with build_runs() below."""
    route_values = [rhex(x) for x in [0.001 * (i + 1) for i in range(11)] + [0.97]]
    d = {"schema": mt.SPEC_SCHEMA, "target_fdr": "0.1", "bh_relative_tolerance": "1/1125899906842624", "tolerance_justification": "t",
         "prediction_basis": "synthetic", "route_fixtures": [
             {"id": "R1", "purpose": "native", "values": route_values,
              "predicted": {"backend": "qvalue", "fallback_reason": None, "n_values": 12, "n_distinct": 12}},
             {"id": "R2", "purpose": "small", "values": route_values[:5],
              "predicted": {"backend": "BH", "fallback_reason": "fewer_than_10_values", "n_values": 5, "n_distinct": 5}}],
         "trace_fixtures": [_synthetic_trace("T1"), _synthetic_trace("T2")]}
    return d


# DANDELION p-values for the synthetic trace (any values in (0, 1); the judge never recomputes p_dact)
P_DACT = {"X1": [0.0001, 0.0003, 0.05, 0.25, 0.001, 0.3, 0.35, 0.4, 0.002, 0.5, 0.6, 0.9],
          "X2": [0.0002, 0.0004, 0.15, 0.22, 0.05, 0.31, 0.36, 0.41, 0.003, 0.51, 0.61, 0.91]}
PI0 = 0.5          # the synthetic "native" route scales BH by pi0 (qvalue's form with pi0 = 0.5)


def _write(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(x + "\n" for x in lines).encode("ascii"))


def _put(path: Path, text: str) -> None:
    """Write EXACTLY these bytes. Text mode translates "\n" to "\r\n" on Windows (measured 2026-10-09: 30 tests failed there with
    run_line_endings / exposure_trace_truncated), so no test in this module ever writes a file in text mode."""
    path.write_bytes(text.encode("ascii"))


def _get(path: Path) -> str:
    return path.read_bytes().decode("ascii")


def _event(call, exposure, backend, reason_, n, distinct, entered):
    step = 3 if reason_ in ("fewer_than_10_values", "fewer_than_4_distinct_values") else 4
    return json.dumps({"call": call, "exposure_id": exposure, "backend": backend, "fallback_reason": reason_, "last_step_reached": step,
                       "n_values": n, "n_distinct": distinct, "qvalue_entered": entered, "p_adjust_inside_qvalue": 0,
                       "observation_kind": "actual_call_trace", "runtime_instrumented": True, "method_commit": COMMIT,
                       "recorder_version": RECORDER_VERSION}, separators=(",", ":"))


def _xevent(exposure, outcome, hits, n_trans, n_valid, pi0a=None, pi0b=None, wg=None):
    """One line as scripts/dandelion/dandelion_exposure_recorder.R writes it."""
    wg = wg or (None, None, None, None)
    return json.dumps({"exposure_id": exposure, "outcome": outcome, "last_guard_reached": hits, "n_trans": n_trans, "n_valid": n_valid,
                       "pi0a": pi0a, "pi0b": pi0b, "wg1": wg[0], "wg2": wg[1], "wg3": wg[2], "wg_sum": wg[3],
                       "observation_kind": "actual_call_trace", "recorder_version": EXPOSURE_RECORDER_VERSION}, separators=(",", ":"))


def _exposure_lines(t: dict) -> list:
    n = len(t["genes"])
    scored = ("0x1.8p-1", "0x1.cp-1", ("0x1p-3", "0x1p-4", "0x1.4p-1", "0x1.ep-1"))
    lines = [_xevent("X1", "scored", 4, n, 12, *scored), _xevent("X2", "scored", 4, n, 12, *scored),
             _xevent("X3", "fewer_than_2_valid_genes", 2, n, 1)]
    if "X4" in t["exposures"]:
        lines.append(_xevent("X4", "mixture_estimate_invalid", 3, n, 13, "-0x1p-7", "0x1.cp-1"))
    return lines


def _common(run: Path, mode: str):
    _write(run / "rng.txt", ["1 2 3", "1 2 3", rhex(0.25), rhex(0.5), rhex(0.75)])
    (run / "warnings.txt").write_bytes(b"")
    _write(run / "environment.tsv", ["R\tR version 4.6.1 (synthetic)", "platform\tx", "rng\tMersenne-Twister Inversion Rejection", "libpaths\t/l",
                                     "DANDELION\t0.1.0\t/nonexistent/DESCRIPTION", "qvalue\t2.44.0\t/nonexistent/DESCRIPTION", "mode\t" + mode])


def build_runs(spec: dict, root: Path):
    """Runs for synthetic_spec(): files exactly as the R runner writes them, for both modes."""
    for f in spec["route_fixtures"]:
        p = [float(mt.parse_hex(v)) for v in f["values"]]
        bh = r_bh(p)
        native = f["predicted"]["backend"] == "qvalue"
        q = [PI0 * x for x in bh] if native else bh
        for mode in ("off", "on"):
            run = root / f["id"] / mode
            _write(run / "output.txt", [rhex(x) for x in q])
            _write(run / "oracle_bh.txt", [rhex(x) for x in bh])
            _write(run / "oracle_qvalue_pi0_1.txt", [rhex(x) for x in bh])
            _common(run, mode)
            if mode == "on":
                tr = run / "trace"
                _write(tr / "events.jsonl", [_event("call-0001", f["id"], f["predicted"]["backend"], f["predicted"]["fallback_reason"],
                                                    len(p), len(set(p)), native)])
                _write(tr / "call-0001.input.txt", [rhex(x) for x in p])
                _write(tr / "call-0001.clamped.txt", [rhex(x) for x in p])
                _write(tr / "call-0001.output.txt", [rhex(x) for x in q])
    q = {"X1": [PI0 * x for x in r_bh(P_DACT["X1"])], "X2": r_bh(P_DACT["X2"])}
    for t in spec["trace_fixtures"]:
        genes = t["genes"]

        def p_of(e, i):
            return rhex(P_DACT[e][i]) if e in P_DACT and i < 12 else "NA"

        def sig_of(e, i):
            return int(q[e][i] <= 0.1) if e in q and i < 12 else 0
        for mode in ("off", "on"):
            run = root / t["id"] / mode
            _write(run / "gene1.txt", ["X1", "X2"])
            _write(run / "mat_p.tsv", ["{}\t{}\t{}".format(g, e, p_of(e, i)) for e in t["exposures"] for i, g in enumerate(genes)])
            _write(run / "mat_sig.tsv", ["{}\t{}\t{}".format(g, e, sig_of(e, i)) for e in t["exposures"] for i, g in enumerate(genes)])
            _write(run / "nominations.tsv", ["{}\t{}\t{}".format(e, g, rhex(P_DACT[e][i])) for e in ("X1", "X2")
                                             for i, g in enumerate(GENES) if q[e][i] <= 0.1])
            _common(run, mode)
            if mode == "on":
                tr = run / "trace"
                _write(tr / "events.jsonl", [_event("call-0001", "X1", "qvalue", None, 12, 12, True),
                                             _event("call-0002", "X2", "BH", "qvalue_error", 12, 12, True)])
                for n, e in ((1, "X1"), (2, "X2")):
                    _write(tr / "call-{:04d}.input.txt".format(n), [rhex(x) for x in P_DACT[e]])
                    _write(tr / "call-{:04d}.clamped.txt".format(n), [rhex(x) for x in P_DACT[e]])
                    _write(tr / "call-{:04d}.output.txt".format(n), [rhex(x) for x in q[e]])
                _write(run / "trace_exposures" / "exposures.jsonl", _exposure_lines(t))


def predicted_for(tid: str = "T1"):
    """The trace predictions that make the synthetic run agree (derived from the synthetic construction)."""
    q = {"X1": [PI0 * x for x in r_bh(P_DACT["X1"])], "X2": r_bh(P_DACT["X2"])}
    nom = sorted([e, g] for e in q for i, g in enumerate(GENES) if q[e][i] <= 0.1)
    counter = sorted([e, g] for e in q for i, g in enumerate(GENES)
                     if mt.exact_bh([Fraction(x) for x in P_DACT[e]])[i] <= Fraction(0.1))
    best = {g: min(P_DACT["X1"][i], P_DACT["X2"][i]) for i, g in enumerate(GENES)}
    integrated = sorted(GENES, key=lambda g: (best[g], g))[:3]
    burden = GENES[:3]
    p = {"gene1": ["X1", "X2"], "excluded_exposures": {"X3": "fewer_than_2_valid_genes"},
         "routes": {"X1": {"backend": "qvalue", "fallback_reason": None}, "X2": {"backend": "BH", "fallback_reason": "qvalue_error"}},
         "nominated_pairs": nom, "counterfactual_bh_nominated_pairs": counter, "integrated_top_k": sorted(integrated),
         "burden_top_k": sorted(burden), "delta_h": int("A9" in integrated) - int("A9" in burden),
         "explained_genes": sorted(set(integrated) ^ set(burden)),
         "exposure_outcomes": {"X1": "scored", "X2": "scored", "X3": "structurally_ineligible"}, "mixture_failure_side": {},
         "exposure_completion": "2/2", "primary_status": "complete", "gene_coverage_distribution": {"2/2": 12}, "unscored_genes": [],
         "partial_top_k": sorted(integrated)}
    if tid == "T2":
        p.update({"integrated_top_k": None, "delta_h": None, "explained_genes": None,
                  "exposure_outcomes": {**p["exposure_outcomes"], "X4": "mixture_estimate_invalid"}, "mixture_failure_side": {"X4": "trans"},
                  "exposure_completion": "2/3", "primary_status": "withheld", "gene_coverage_distribution": {"0/1": 1, "2/3": 12},
                  "unscored_genes": ["A13"]})
    return p


@pytest.fixture
def synthetic(tmp_path):
    spec = synthetic_spec()
    for t in spec["trace_fixtures"]:
        t["predicted"] = predicted_for(t["id"])
    raw = _raw(spec)
    build_runs(spec, tmp_path / "runs")
    return {"raw": raw, "sha": hashlib.sha256(raw).hexdigest(), "runs": tmp_path / "runs", "spec": spec}


def judged(s):
    return mt.judge(s["raw"], s["sha"], s["runs"])


def test_a_consistent_synthetic_run_passes_and_reports_every_layer(synthetic):
    r = judged(synthetic)
    assert r["fixtures_passed"] is True and r["failures"] == []
    assert r["schema"] == mt.REPORT_SCHEMA and r["exposure_failure_policy"]["primary"]["on_failure"] == "withhold_primary_ranking_and_delta_h20"
    t = r["trace_fixtures"][0]
    assert set(t["exposures"]) == {"X1", "X2"}
    for layer in ("input", "adjustment", "pair_decision"):
        assert layer in t["exposures"]["X1"]
    assert t["gene_aggregation"]["basis"] == "primary"
    assert t["gene_aggregation"]["ranking_scores_unchanged_under_exact_bh"] is True
    assert t["endpoint"]["unchanged_under_exact_bh"] is True
    assert t["nominations"]["changed_by_adjustment"]                      # pi0 < 1 nominates more than BH would
    assert {row["gene"] for row in t["explanation_rows"]} == set(synthetic["spec"]["trace_fixtures"][0]["predicted"]["explained_genes"])
    assert t["endpoint_release"]["confirmatory"]["delta_h20"] == "released"
    assert t["endpoint_release"]["feasibility"]["delta_h20"] == "withheld_feasibility_stage"
    assert mt.render_report(r) == mt.render_report(judged(synthetic))     # deterministic


def test_a_mixture_failure_withholds_the_primary_and_lets_the_diagnostics_finish(synthetic):
    """Ruling 2026-10-08g: the failed exposure keeps its category, usable-pair count and estimates; the successful exposures keep every
    layer; the endpoint is withheld (never computed on the evaluable subset); the partial ranking is exploratory and keeps A13 unscored."""
    t2 = judged(synthetic)["trace_fixtures"][1]
    assert t2["coverage"]["primary_status"] == "withheld" and t2["coverage"]["exposure_completion"] == "2/3"
    x4 = t2["exposure_outcomes"]["X4"]
    assert (x4["status"], x4["invalid_side"], x4["observed"]["usable_pairs"], x4["observed"]["pi0a"]) == (
        "mixture_estimate_invalid", "trans", 13, "-0x1p-7")
    assert x4["genes_it_would_cover"] == sorted(GENES + ["A13"])
    assert set(t2["exposures"]) == {"X1", "X2"} and t2["consistency"] == {
        "scores_exactly_where_planned_and_scored": True, "nominations_equal_mat_sig_pairs": True, "gene1_equals_scored_exposures": True}
    assert set(t2["endpoint"]) == {"withheld"} and t2["explanation_rows"] is None
    assert t2["endpoint_release"]["confirmatory"]["delta_h20"] == "withheld_incomplete_exposures"
    pr = t2["partial_ranking"]
    assert pr["status"] == "exploratory_conditional_on_estimability" and pr["unscored"] == ["A13"]
    assert pr["planned_universe"] == sorted(GENES + ["A13"]) and pr["estimation_failed_pairs_per_gene"] == {g: 1 for g in GENES + ["A13"]}
    assert all(row["structural_pairs"] == 1 for row in pr["ranked"])        # X3 only: the failed X4 pair is NOT counted as structural
    assert t2["gene_aggregation"]["basis"] == "partial_exploratory"


def test_PROPERTY_an_adjustment_change_never_reaches_the_min_score_endpoint(synthetic):
    """Scores come from mat.p, written before safe_qvalues runs: the counterfactual BH adjustment leaves every ranking score and the
    endpoint unchanged, while it may change significance counts and DANDELION's own nominations."""
    t1, t2 = judged(synthetic)["trace_fixtures"]
    assert t1["gene_aggregation"]["ranking_scores_unchanged_under_exact_bh"] and t1["endpoint"]["unchanged_under_exact_bh"]
    assert t1["gene_aggregation"]["significant_pair_counts_changed"]
    assert t2["gene_aggregation"]["ranking_scores_unchanged_under_exact_bh"] and t2["gene_aggregation"]["significant_pair_counts_changed"]


def _scale_both(fixture: Path, names, factor: float):
    """The same perturbation in BOTH modes (so recorder equivalence still holds) -- an algorithm difference, not a recorder effect."""
    for mode in ("off", "on"):
        for name in names:
            path = fixture / mode / name
            _put(path, "".join(rhex(min(1.0, float(mt.parse_hex(x)) * factor)) + "\n" for x in _get(path).split()))
        if mode == "on" and "output.txt" in names:
            call = fixture / "on" / "trace" / "call-0001.output.txt"
            _put(call, _get(fixture / "on" / "output.txt"))


def _drop_second_call(trace: Path):
    _put(trace / "events.jsonl", _get(trace / "events.jsonl").split("\n")[0] + "\n")
    for f in trace.glob("call-0002.*"):
        f.unlink()


def _flip(path: Path, line_no: int, new: str):
    lines = _get(path).split("\n")[:-1]
    lines[line_no] = new
    _put(path, "".join(x + "\n" for x in lines))


def _edit_exposure(run: Path, exposure: str, drop: bool = False, **over):
    path = run / "trace_exposures" / "exposures.jsonl"
    out = []
    for text in _get(path).split("\n")[:-1]:
        doc = json.loads(text)
        if doc["exposure_id"] == exposure:
            if drop:
                continue
            doc.update(over)
        out.append(json.dumps(doc, separators=(",", ":")))
    _put(path, "".join(x + "\n" for x in out))


def _unpredicted_failure(s):
    """T1's X2 returns NULL at the weight-sum guard, as DANDELION would leave it: no mat.p column, no safe_qvalues call, not in gene1."""
    for m in ("off", "on"):
        run = s["runs"] / "T1" / m
        for name, value in (("mat_p.tsv", "NA"), ("mat_sig.tsv", "0")):
            lines = [x if x.split("\t")[1] != "X2" else "\t".join(x.split("\t")[:2] + [value]) for x in _get(run / name).split("\n")[:-1]]
            _put(run / name, "".join(x + "\n" for x in lines))
        noms = [x for x in _get(run / "nominations.tsv").split("\n")[:-1] if not x.startswith("X2\t")]
        _put(run / "nominations.tsv", "".join(x + "\n" for x in noms))
        _put(run / "gene1.txt", "X1\n")
    _drop_second_call(s["runs"] / "T1" / "on" / "trace")
    _edit_exposure(s["runs"] / "T1" / "on", "X2", outcome="nonpositive_weight_sum", wg_sum="0x0p+0")


def test_an_unpredicted_failure_still_finishes_the_diagnostics(synthetic):
    _unpredicted_failure(synthetic)
    r = judged(synthetic)
    t1 = r["trace_fixtures"][0]
    assert t1["coverage"]["primary_status"] == "withheld" and t1["consistency"] == {
        "scores_exactly_where_planned_and_scored": True, "nominations_equal_mat_sig_pairs": True, "gene1_equals_scored_exposures": True}
    assert set(t1["exposures"]) == {"X1"} and t1["partial_ranking"]["conditioned_on_exposures"] == ["X1"]
    assert "T1:prediction:delta_h" in r["failures"] and not any(f.startswith("T2:") for f in r["failures"])


@pytest.mark.parametrize("damage, finding", [
    (lambda s: _put(s["runs"] / "R1" / "on" / "output.txt", rhex(0.5) + "\n" + "".join(
        x + "\n" for x in _get(s["runs"] / "R1" / "on" / "output.txt").split("\n")[1:-1])), "R1:recorder_changed_science"),
    (lambda s: _flip(s["runs"] / "R2" / "on" / "rng.txt", 1, "9 9 9"), "R2:random_stream_consumed"),
    (lambda s: _flip(s["runs"] / "T1" / "on" / "rng.txt", 1, "9 9 9"), "T1:random_stream_consumed"),
    (lambda s: _flip(s["runs"] / "R2" / "on" / "oracle_bh.txt", 0, rhex(0.4)), "R2:output_is_r_bh_bitwise"),
    (lambda s: _flip(s["runs"] / "T1" / "on" / "mat_sig.tsv", 0, "A1\tX1\t0"), "T1:decisions_equal_mat_sig"),
    (lambda s: _flip(s["runs"] / "T1" / "on" / "trace" / "call-0001.input.txt", 2, rhex(0.2000001)), "T1:recorder_input_equals_mat_p"),
    (lambda s: _flip(s["runs"] / "T1" / "on" / "nominations.tsv", 0, "X1\tA12\t" + rhex(0.9)), "T1:nominations_equal_mat_sig_pairs"),
    (lambda s: _flip(s["runs"] / "T1" / "on" / "mat_p.tsv", 24, "A1\tX3\t" + rhex(0.5)), "T1:scores_exactly_where_planned_and_scored"),
    (lambda s: [_put(s["runs"] / "T1" / m / "gene1.txt", "X1\nX2\nX3\n") for m in ("off", "on")], "T1:gene1_equals_scored_exposures"),
    (lambda s: [_put(s["runs"] / x / m / "warnings.txt", "a warning\n") for x in ("R1",) for m in ("off", "on")], "R1:warnings_raised"),
    (lambda s: [_put(s["runs"] / "T1" / m / "warnings.txt", "a warning\n") for m in ("off", "on")], "T1:warnings_raised"),
    (lambda s: [_put(s["runs"] / "T2" / m / "warnings.txt", "a warning\n") for m in ("off", "on")], "T2:warnings_raised"),
    (lambda s: _scale_both(s["runs"] / "R2", ("output.txt", "oracle_bh.txt"), 1 + 2 ** -47), "R2:r_bh_within_tolerance_of_exact_bh"),
    (lambda s: _scale_both(s["runs"] / "R1", ("oracle_qvalue_pi0_1.txt",), 1 + 2 ** -47), "R1:qvalue_pi0_1_within_tolerance_of_exact_bh"),
    (lambda s: _scale_both(s["runs"] / "R1", ("output.txt",), 2 / PI0 + 1), "R1:native_not_above_exact_bh"),
    # an exposure without an identified cause, or an infrastructure failure: the primary is REFUSED and the run is reported
    (lambda s: _edit_exposure(s["runs"] / "T2" / "on", "X4", drop=True), "T2:primary_refused:X4:unclassified_missing_exposure"),
    (lambda s: _edit_exposure(s["runs"] / "T2" / "on", "X4", outcome="abnormal_exit"), "T2:primary_refused:X4:infrastructure_error"),
    (lambda s: _edit_exposure(s["runs"] / "T1" / "on", "X3", drop=True), "T1:primary_refused:X3:unclassified_missing_exposure"),
    # an UNPREDICTED estimation failure (consistent evidence: no scores, no adjustment call) is a finding, never a refusal or a pass
    (lambda s: _unpredicted_failure(s), "T1:prediction:exposure_outcomes"),
    (lambda s: _unpredicted_failure(s), "T1:prediction:primary_status"),
])
def test_damaged_evidence_is_reported_as_a_finding(synthetic, damage, finding):
    damage(synthetic)
    r = judged(synthetic)
    assert r["fixtures_passed"] is False and finding in r["failures"]


def test_a_refused_primary_withholds_the_partial_ranking_too(synthetic):
    _edit_exposure(synthetic["runs"] / "T2" / "on", "X4", drop=True)
    t2 = judged(synthetic)["trace_fixtures"][1]
    assert t2["coverage"]["primary_status"] == "refused" and t2["partial_ranking"] is None
    assert t2["endpoint_release"]["feasibility"]["partial_ranking"] == "refused"
    assert t2["prediction"]["observed"]["partial_top_k"] is None and t2["gene_aggregation"]["basis"] is None


def test_a_failed_prediction_is_a_finding_never_a_silent_pass(synthetic):
    spec = copy.deepcopy(synthetic["spec"])
    spec["route_fixtures"][0]["predicted"]["backend"], spec["route_fixtures"][0]["predicted"]["fallback_reason"] = "BH", "qvalue_error"
    spec["trace_fixtures"][0]["predicted"]["delta_h"] += 1
    spec["trace_fixtures"][1]["predicted"]["gene_coverage_distribution"] = {"2/3": 13}
    raw = _raw(spec)
    r = mt.judge(raw, hashlib.sha256(raw).hexdigest(), synthetic["runs"])
    assert {"R1:route_prediction", "T1:prediction:delta_h", "T2:prediction:gene_coverage_distribution"} <= set(r["failures"])
    assert r["fixtures_passed"] is False


def _score_failed_exposure(s):
    for m in ("off", "on"):
        path = s["runs"] / "T2" / m / "mat_p.tsv"
        lines = _get(path).split("\n")[:-1]
        i = lines.index("A1\tX4\tNA")
        _flip(path, i, "A1\tX4\t" + rhex(0.3))


def _adjust_failed_exposure(s):
    tr = s["runs"] / "T2" / "on" / "trace"
    _put(tr / "events.jsonl", _get(tr / "events.jsonl") + _event("call-0003", "X4", "BH", "qvalue_error", 12, 1, True) + "\n")
    for kind in ("input", "clamped", "output"):
        _write(tr / "call-0003.{}.txt".format(kind), [rhex(0.5)] * 12)


@pytest.mark.parametrize("damage, code", [
    (lambda s: _flip(s["runs"] / "T1" / "on" / "mat_p.tsv", 0, "A1\tX1\tNA"), "incomplete_method_execution"),   # planned score missing
    (lambda s: _flip(s["runs"] / "T2" / "on" / "mat_p.tsv", 0, "A1\tX1\tNA"), "incomplete_method_execution"),   # ... under a withheld primary
    (lambda s: _drop_second_call(s["runs"] / "T1" / "on" / "trace"), "trace_incomplete"),
    (lambda s: _put(s["runs"] / "R1" / "on" / "trace" / "stray.txt", "x"), "trace_unexpected_files"),
    (lambda s: (s["runs"] / "R1" / "off" / "oracle_qvalue_pi0_1.txt").unlink(), "run_file_presence"),
    (lambda s: (s["runs"] / "T1" / "on" / "mat_sig.tsv").write_bytes(b"A1\tX1\t2\n"), "run_mat_sig"),
    (lambda s: (s["runs"] / "R2" / "on" / "output.txt").write_bytes(b"0x1p-1\r\n"), "run_line_endings"),
    (lambda s: (s["runs"] / "R2" / "off" / "trace").mkdir(), "run_off_recorded"),
    (lambda s: (s["runs"] / "T1" / "off" / "trace_exposures").mkdir(), "run_off_recorded"),
    (lambda s: (s["runs"] / "R1" / "on" / "trace_exposures").mkdir(), "run_unexpected_recorder"),
    (lambda s: (s["runs"] / "T1" / "on" / "trace_exposures" / "exposures.jsonl").unlink(), "exposure_trace_missing"),
    (lambda s: _edit_exposure(s["runs"] / "T1" / "on", "X1", n_valid=11), "usable_pair_count_mismatch"),
    (lambda s: _edit_exposure(s["runs"] / "T2" / "on", "X4", n_trans=14), "usable_pair_count_mismatch"),
    (_score_failed_exposure, "failed_exposure_has_scores"),
    (lambda s: _adjust_failed_exposure(s), "trace_incomplete"),        # a NULL-returning exposure can never reach safe_qvalues
    (lambda s: _flip(s["runs"] / "R2" / "off" / "environment.tsv", 5, "qvalue\t2.45.0\t/nonexistent/DESCRIPTION"), "run_environments_differ"),
    (lambda s: _flip(s["runs"] / "T2" / "off" / "environment.tsv", 3, "libpaths\t/other"), "run_environments_differ"),
    (lambda s: _flip(s["runs"] / "R2" / "off" / "environment.tsv", 6, "mode\ton"), "run_environment"),
])
def test_malformed_or_incomplete_evidence_refuses(synthetic, damage, code):
    damage(synthetic)
    assert reason(lambda: judged(synthetic)) == code


def test_the_specification_must_be_the_bound_one(synthetic):
    assert reason(lambda: mt.judge(synthetic["raw"], "0" * 64, synthetic["runs"])) == "spec_digest"


def test_prepared_scenarios_hold_inputs_only(tmp_path):
    spec = mt.load_spec(SPEC.read_bytes())
    ids = mt.prepare_scenarios(spec, tmp_path / "s")
    assert ids == ("R1", "R2", "R3", "R4", "R5", "R6", "T1", "T2")
    for f in (tmp_path / "s").rglob("*"):
        if f.is_file():
            text = _get(f)
            assert "predicted" not in text and "qvalue_error" not in text and "fallback" not in text
            assert "mixture" not in text and "withheld" not in text and "scored" not in text
    assert _get(tmp_path / "s" / "R1" / "input.txt").split() == spec["route_fixtures"][0]["values"]
    assert _get(tmp_path / "s" / "T2" / "exposures.txt").split() == ["E1", "E2", "E3", "E4", "E5", "E6"]
    with pytest.raises(FileExistsError):
        mt.prepare_scenarios(spec, tmp_path / "s")


# ------------------------------------------------------------------------------------------------------------------ explanation rows
def test_explanation_rows_name_entrants_and_leavers_with_their_evidence():
    F = Fraction
    integrated = (GeneRank("g1", F(1, 100), "e1", 2, 1, 0), GeneRank("g3", F(2, 100), "e2", 1, 0, 1), GeneRank("g2", F(5, 100), "e1", 2, 0, 0))
    plan = {("g1", "e1"): True, ("g1", "e2"): True, ("g2", "e1"): True, ("g2", "e2"): True, ("g3", "e1"): False, ("g3", "e2"): True}
    rows = mt.explanation_rows(integrated, ("g1", "g2", "g3"), 2, frozenset({"g3"}), "ref", {"e2": {"backend": "BH"}}, plan)
    assert [(r["gene"], r["change"], r["burden_rank"], r["integrated_rank"]) for r in rows] == [("g2", "left", 2, 3), ("g3", "entered", 3, 2)]
    g3 = rows[1]
    assert (g3["responsible_exposure"], g3["responsible_exposure_route"], g3["tested_exposures"], g3["structural_exposures"],
            g3["planned_exposures"], g3["reference_member"]) == ("e2", {"backend": "BH"}, 1, 1, 2, True)
    assert g3["responsible_score"] == float(F(2, 100)).hex() and not g3["on_integrated_top_k_tie"]


def test_a_tie_needs_a_second_gene_at_the_cutoff():
    F = Fraction
    tied = (GeneRank("a", F(1, 10), "e", 1, 0), GeneRank("b", F(2, 10), "e", 1, 0), GeneRank("c", F(2, 10), "e", 1, 0))
    plan = {(g, "e"): True for g in "abc"}
    rows = mt.explanation_rows(tied, ("a", "c", "b"), 2, frozenset(), "ref", {}, plan)
    assert [(r["gene"], r["on_integrated_top_k_tie"]) for r in rows] == [("b", True), ("c", True)]
    untied = (GeneRank("a", F(1, 10), "e", 1, 0), GeneRank("b", F(2, 10), "e", 1, 0), GeneRank("c", F(3, 10), "e", 1, 0))
    rows = mt.explanation_rows(untied, ("a", "c", "b"), 2, frozenset(), "ref", {}, plan)
    assert [(r["gene"], r["on_integrated_top_k_tie"]) for r in rows] == [("b", False), ("c", False)]


# ------------------------------------------------------------------------------------------------------------------ integration (R)
@pytest.mark.skipif(not (os.environ.get("GVC_METHOD_FIXTURE_RSCRIPT") and os.environ.get("GVC_METHOD_FIXTURE_RLIBS")),
                    reason="set GVC_METHOD_FIXTURE_RSCRIPT and GVC_METHOD_FIXTURE_RLIBS to an Rscript and a library holding DANDELION and qvalue")
def test_the_real_fixtures_meet_their_frozen_predictions(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts" / "dandelion"))
    try:
        import run_method_fixtures
    finally:
        sys.path.pop(0)
    code = run_method_fixtures.main(["--spec", str(SPEC), "--spec-sha256", SPEC_SHA256, "--rscript", os.environ["GVC_METHOD_FIXTURE_RSCRIPT"],
                                     "--r-libs", os.environ["GVC_METHOD_FIXTURE_RLIBS"], "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    report = json.loads(_get(tmp_path / "out" / "report.json"))
    assert code == 0 and report["fixtures_passed"] is True, out
    t1, t2 = report["trace_fixtures"]
    assert t1["nominations"]["changed_by_adjustment"] == [["E1", "G05"], ["E1", "G06"]]
    # T2: the mixture-estimation failure, observed in the actual call with its invalid estimate preserved
    e5 = t2["exposure_outcomes"]["E5"]
    assert (e5["status"], e5["invalid_side"], e5["observed"]["usable_pairs"]) == ("mixture_estimate_invalid", "trans", 31)
    assert t2["coverage"]["primary_status"] == "withheld" and t2["partial_ranking"]["unscored"] == ["G31"]
    # ONLY the declared libraries, the run's empty user / site library and R's own library are searched
    declared = {Path(x).resolve().as_posix() for x in os.environ["GVC_METHOD_FIXTURE_RLIBS"].split(os.pathsep)}
    searched = report["environment"]["libpaths"].split(" | ")
    rest = [x for x in searched if x not in declared and x != (tmp_path / "out" / "empty_library").resolve().as_posix()]
    assert len(rest) == 1 and rest[0].endswith("/library"), searched


_RECORDER_GUARDS_R = r'''
args <- commandArgs(trailingOnly = TRUE); source(args[1]); tmp <- args[2]
suppressPackageStartupMessages(library(DANDELION))
d <- file.path(tmp, "a"); exposure_recorder_start(d); invisible(exposure_recorder_stop())
stopifnot(identical(list.files(d, all.files = TRUE, no.. = TRUE), "exposures.jsonl"), file.size(file.path(d, "exposures.jsonl")) == 0)
r <- tryCatch({ exposure_recorder_start(d); "started" }, error = function(e) conditionMessage(e))
stopifnot(grepl("not empty", r))
ns <- asNamespace("DANDELION"); f <- get("run_dandelion_for_exposure", envir = ns); b <- body(f); b[[18]] <- quote(NULL); body(f) <- b
unlockBinding("run_dandelion_for_exposure", ns); assign("run_dandelion_for_exposure", f, envir = ns)
r <- tryCatch({ exposure_recorder_start(file.path(tmp, "b")); "started" }, error = function(e) conditionMessage(e))
stopifnot(grepl("refusing to trace an unmeasured implementation", r), !dir.exists(file.path(tmp, "b")))
cat("RECORDER GUARDS OK\n")
'''


@pytest.mark.skipif(not (os.environ.get("GVC_METHOD_FIXTURE_RSCRIPT") and os.environ.get("GVC_METHOD_FIXTURE_RLIBS")),
                    reason="set GVC_METHOD_FIXTURE_RSCRIPT and GVC_METHOD_FIXTURE_RLIBS to an Rscript and a library holding DANDELION")
def test_the_exposure_recorder_guards_in_r(tmp_path):
    """What the fixtures cannot show (each trace fixture calls the method): the trace file exists, EMPTY, from the start (zero calls is
    a result, not a missing trace); a used destination is refused; a DANDELION body that differs at a measured guard is refused."""
    script = tmp_path / "guards.R"
    _put(script, _RECORDER_GUARDS_R)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("R_", "RENV_"))}
    env["R_LIBS"] = os.environ["GVC_METHOD_FIXTURE_RLIBS"]
    (tmp_path / "empty").mkdir()
    env["R_LIBS_USER"] = env["R_LIBS_SITE"] = str(tmp_path / "empty")
    done = subprocess.run([os.environ["GVC_METHOD_FIXTURE_RSCRIPT"], "--vanilla", str(script),
                           str(ROOT / "scripts" / "dandelion" / "dandelion_exposure_recorder.R"), str(tmp_path / "w")],
                          capture_output=True, text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert done.returncode == 0 and "RECORDER GUARDS OK" in done.stdout, done.stderr[-2000:]


def test_these_test_modules_never_use_text_mode_file_io():
    """REGRESSION GUARD (2026-10-09, measured on Windows): text-mode writes turn "\n" into "\r\n" there, which the strict readers
    rightly refuse, so these modules use byte-exact helpers only. Any write_text / read_text call or open() in these modules fails here
    on EVERY platform, not only on Windows."""
    import ast
    for name in ("test_inference_method_trace.py", "test_inference_exposure_outcomes.py"):
        tree = ast.parse((ROOT / "tests" / "unit" / name).read_bytes().decode("utf-8"))
        bad = [(name, node.lineno) for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ((isinstance(node.func, ast.Attribute) and node.func.attr in ("write_text", "read_text", "open"))
                    or (isinstance(node.func, ast.Name) and node.func.id == "open"))]
        assert bad == [], bad
