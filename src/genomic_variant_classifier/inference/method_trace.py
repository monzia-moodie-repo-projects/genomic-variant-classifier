"""Predetermined DANDELION method fixtures and the adjustment-to-endpoint trace (owner rulings 2026-10-08e, 2026-10-08f section 6,
2026-10-08g).

WHY
===
A q-value backend change (native qvalue versus a Benjamini-Hochberg fallback) is a change in the ADJUSTMENT layer. Whether it reaches
the endpoint depends on the path, and the pinned code has two (DANDELION 0.1.0, R/DANDELION.R, read in full):

    pair scores -> adjusted values -> significance decisions -> nominations          (mat.sig, calc_pair.gene: DANDELION's own output)
    pair scores -> gene aggregation -> ranking -> top k -> reference recovery        (mat.p: the extended ranking, inference/ranking.py)

run_dandelion_for_exposure computes p_dact and stores it in mat.p BEFORE safe_qvalues is called; q-values only set mat.sig. So under
minimum-raw-score aggregation an adjustment change cannot move a ranking score -- but it can change nominations. A changed adjustment
method must never be described as a changed Delta H(20) without the intermediate layers demonstrating it; this module records them.

LAYERS (ruling 2026-10-08f section 6)
=====================================
    input            exact ordered p-values (the recorder's input and post-clamp files), exposure identity, input digests
    adjustment       the backend and fallback reason the ACTUAL call took (backend_trace), the qvalue / DANDELION identities
    pair decision    the adjusted values and the decisions q <= target FDR (compared in binary64, exactly as R compares)
    gene aggregation tested exposures, the minimum-score rule, the pair plan (eligibility), the tie rule (score, then gene)
    endpoint         top-k membership, reference membership, Delta H(k) against burden-only, and an EXPLANATION ROW for every gene
                     entering or leaving the integrated top k -- diagnostic, never an additional primary endpoint

EXPOSURE FAILURE (ruling 2026-10-08g, inference/exposure_outcomes.py)
=====================================================================
Every exposure gets ONE status from the actual call (the exposure recorder) checked against what the inputs predict before execution:
structurally ineligible, scored, a mixture-estimation failure, unclassified-missing or an infrastructure error. The PRIMARY ranking
(and with it the endpoint and the explanation rows) exists only when every eligible exposure is scored; otherwise it is WITHHELD and
the run still finishes: every successful exposure keeps its layers, the failed exposure keeps its category, usable-pair count and
mixture estimates, coverage is reported per gene, and an EXPLORATORY partial ranking keeps the planned gene universe with "unscored"
genes. Unclassified-missing and infrastructure outcomes refuse the primary analysis and the partial ranking, and are reported.

FIXTURES are frozen with their predictions BEFORE any qualified run (tests/fixtures/dandelion/method_fixtures_v2.json); a prediction
that fails is a FINDING reported in the report, never a reason to edit the fixture. Malformed evidence REFUSES (InferenceError).
The counterfactual "every exposure adjusted by BH" is computed EXACTLY here (rational arithmetic on the recorded post-clamp values) --
an independent oracle, not a second run of the package.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from genomic_variant_classifier.inference.analysis_contract import ExposureFailurePolicy
from genomic_variant_classifier.inference.backend_trace import admissible_events, read_trace
from genomic_variant_classifier.inference.endpoints import EndpointContract, recovery_contrast
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import (
    PRIMARY_COMPLETE, PRIMARY_REFUSED, PRIMARY_WITHHELD, ExposureStatus, classify_exposures, coverage_report, endpoint_release,
    exposure_records, partial_ranking, read_exposure_trace)
from genomic_variant_classifier.inference.ranking import Pair, State, audit_top_k, rank_genes

logger = logging.getLogger(__name__)

__all__ = ["SPEC_SCHEMA", "REPORT_SCHEMA", "CIS_DISTANCE", "parse_hex", "exact_bh", "significant", "load_spec", "pair_plan", "usable_pairs",
           "prepare_scenarios", "judge", "explanation_rows", "render_report"]

SPEC_SCHEMA = "gvc.dandelion-method-fixtures/2"        # 2: several trace fixtures; per-exposure outcomes and coverage (ruling 2026-10-08g)
REPORT_SCHEMA = "gvc.dandelion-method-report/2"
CIS_DISTANCE = 5_000_000            # med_gene's default `dist`, the value the runner uses
_HEX = re.compile(r"(?P<sign>-?)0x(?P<whole>[0-9a-f])(?:\.(?P<frac>[0-9a-f]+))?p(?P<exp>[+-]\d+)")
_SCIENCE_FILES = {"route": ("output.txt", "oracle_bh.txt", "oracle_qvalue_pi0_1.txt", "rng.txt", "warnings.txt"),
                  "trace": ("gene1.txt", "mat_p.tsv", "mat_sig.tsv", "nominations.tsv", "rng.txt", "warnings.txt")}
_RECORDER_DIRS = ("trace", "trace_exposures")
_STATUSES = {s.value for s in ExposureStatus}
_PRIMARY = {PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED}
_PRIMARY_ONLY = ("integrated_top_k", "delta_h", "explained_genes")      # exist only when the primary ranking is complete
_SIDES = {"trans", "burden", "both"}
_BACKENDS = {"qvalue", "BH"}
_REASONS = {None, "fewer_than_10_values", "fewer_than_4_distinct_values", "qvalue_not_installed", "qvalue_warning", "qvalue_error"}
_EXCLUSIONS = {"fewer_than_2_valid_genes", "no_trans_genes", "exposure_not_annotated"}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


# ---------------------------------------------------------------------------------------------------- exact values
def parse_hex(text: str):
    """R's sprintf("%a") text -> the EXACT binary64 value as a Fraction ("NA" -> None). Zero is admitted here (a post-clamp vector may
    not hold it, but an input may); the value must round-trip through float exactly (no rounding is ever accepted)."""
    _require(type(text) is str, "fixture_value")
    if text == "NA":
        return None
    m = _HEX.fullmatch(text)
    _require(m is not None, "fixture_value", repr(text[:40]))
    frac = m["frac"] or ""
    exact = Fraction(int(m["whole"] + frac, 16), 16 ** len(frac)) * Fraction(2) ** int(m["exp"])
    exact = -exact if m["sign"] else exact
    _require(Fraction(float.fromhex(text)) == exact, "fixture_value_not_binary64", text)
    return exact


def exact_bh(values) -> tuple:
    """Benjamini-Hochberg adjusted values in EXACT rational arithmetic: q_i = min(1, min over j with p_(j) >= p_i of m p_(j) / rank_j).
    Tie-invariant (tied values receive the same q); the independent oracle for R's floating p.adjust(method = "BH")."""
    values = tuple(values)
    _require(len(values) > 0 and all(type(v) is Fraction and 0 <= v <= 1 for v in values), "bh_input")
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    out, running = [None] * m, Fraction(1)
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, values[i] * m / rank)
        out[i] = running
    return tuple(out)


def _binary64(decimal_text: str) -> Fraction:
    """The binary64 value R holds for a decimal literal -- R compares q <= target.fdr in binary64, so must we."""
    return Fraction(float(decimal_text))


def significant(q: Fraction, target_fdr: str) -> bool:
    """DANDELION's decision `q_dact <= target.fdr`, evaluated EXACTLY as R evaluates it: both sides in binary64 (0.1 is the double
    0.1000000000000000055511151231257827..., not one tenth)."""
    return q <= _binary64(target_fdr)


# ---------------------------------------------------------------------------------------------------- the specification
def _strict(raw: bytes):
    def pairs(items):
        out = {}
        for k, v in items:
            _require(k not in out, "spec_duplicate_key", k)
            out[k] = v
        return out

    def refuse(token):
        raise InferenceError("spec_number", token)
    _require(type(raw) is bytes and not raw.startswith(b"\xef\xbb\xbf"), "spec_bytes")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError("spec_json", str(exc)) from None


def _keys(doc, keys, what):
    _require(type(doc) is dict and set(doc) == set(keys), "spec_keys", "{}: {}".format(what, sorted(set(doc) ^ set(keys)) if type(doc) is dict else "not an object"))


def _route(pred, what):
    _keys(pred, ("backend", "fallback_reason"), what)
    _require(pred["backend"] in _BACKENDS and pred["fallback_reason"] in _REASONS
             and (pred["backend"] == "qvalue") == (pred["fallback_reason"] is None), "spec_route", what)


def load_spec(raw: bytes) -> dict:
    """The frozen fixture specification, strictly validated (unknown or missing keys, floats, duplicates refused)."""
    spec = _strict(raw)
    _keys(spec, ("schema", "target_fdr", "bh_relative_tolerance", "tolerance_justification", "prediction_basis", "route_fixtures",
                 "trace_fixtures"), "spec")
    _require(spec["schema"] == SPEC_SCHEMA, "spec_schema")
    _binary64(spec["target_fdr"])
    tol = Fraction(spec["bh_relative_tolerance"])
    _require(0 < tol < Fraction(1, 2 ** 40), "spec_tolerance")
    ids = set()
    for f in spec["route_fixtures"]:
        _keys(f, ("id", "purpose", "values", "predicted"), "route fixture")
        _require(type(f["id"]) is str and re.fullmatch(r"R\d+", f["id"]) is not None and f["id"] not in ids, "spec_fixture_id")
        ids.add(f["id"])
        _require(type(f["values"]) is list and len(f["values"]) > 0, "spec_values", f["id"])
        for v in f["values"]:
            x = parse_hex(v)
            _require(x is not None and 0 <= x <= 1, "spec_values", f["id"])
        _keys(f["predicted"], ("backend", "fallback_reason", "n_values", "n_distinct"), f["id"])
        _route({k: f["predicted"][k] for k in ("backend", "fallback_reason")}, f["id"])
        _require(f["predicted"]["n_values"] == len(f["values"]), "spec_prediction_n", f["id"])
    _require(type(spec["trace_fixtures"]) is list and len(spec["trace_fixtures"]) > 0, "spec_trace_fixtures")
    for t in spec["trace_fixtures"]:
        _load_trace(t, ids)
    return spec


def _load_trace(t: dict, ids: set) -> None:
    _keys(t, ("id", "purpose", "genes", "exposures", "trans", "wes", "ref_table", "k", "reference_id", "reference_positives",
              "reference_note", "predicted"), "trace fixture")
    _require(type(t["id"]) is str and re.fullmatch(r"T\d+", t["id"]) is not None and t["id"] not in ids, "spec_fixture_id")
    ids.add(t["id"])
    _require(type(t["purpose"]) is str and bool(t["purpose"].strip()), "spec_purpose", t["id"])
    _require(len(set(t["genes"])) == len(t["genes"]) > 1 and len(set(t["exposures"])) == len(t["exposures"]) > 0, "spec_identities")
    _require(set(t["trans"]) == set(t["exposures"]) and all(len(t["trans"][e]) == len(t["genes"]) for e in t["exposures"])
             and len(t["wes"]) == len(t["genes"]), "spec_matrix_shape")
    for column in list(t["trans"].values()) + [t["wes"]]:
        for v in column:
            x = parse_hex(v)
            _require(x is None or 0 <= x <= 1, "spec_values", t["id"])
    for row in t["ref_table"]:
        _require(type(row) is list and len(row) == 5 and all(type(x) is str for x in row[:3]) and all(type(x) is int for x in row[3:]),
                 "spec_ref_table")
    _require(type(t["k"]) is int and 1 <= t["k"] <= len(t["genes"]), "spec_k")
    _require(set(t["reference_positives"]) <= set(t["genes"]), "spec_reference")
    p = t["predicted"]
    _keys(p, ("gene1", "excluded_exposures", "routes", "nominated_pairs", "counterfactual_bh_nominated_pairs", "integrated_top_k",
              "burden_top_k", "delta_h", "explained_genes", "exposure_outcomes", "mixture_failure_side", "exposure_completion",
              "primary_status", "gene_coverage_distribution", "unscored_genes", "partial_top_k"), "trace prediction")
    _require(set(p["excluded_exposures"].values()) <= _EXCLUSIONS, "spec_exclusion")
    for e, r in p["routes"].items():
        _route(r, e)
    _require(type(p["exposure_outcomes"]) is dict and set(p["exposure_outcomes"]) == set(t["exposures"])
             and set(p["exposure_outcomes"].values()) <= _STATUSES, "spec_exposure_outcomes", t["id"])
    _require({e for e, s in p["exposure_outcomes"].items() if s == ExposureStatus.STRUCTURALLY_INELIGIBLE.value} == set(p["excluded_exposures"]),
             "spec_exposure_outcomes", "structural outcomes must be exactly the predicted exclusions")
    _require(type(p["mixture_failure_side"]) is dict and set(p["mixture_failure_side"].values()) <= _SIDES
             and set(p["mixture_failure_side"]) == {e for e, s in p["exposure_outcomes"].items() if s == ExposureStatus.MIXTURE_ESTIMATE_INVALID.value},
             "spec_mixture_side", t["id"])
    _require(p["primary_status"] in _PRIMARY, "spec_primary_status")
    _require(type(p["exposure_completion"]) is str and re.fullmatch(r"\d+/[1-9]\d*", p["exposure_completion"]) is not None, "spec_completion")
    _require(type(p["gene_coverage_distribution"]) is dict and all(re.fullmatch(r"\d+/\d+", k) and type(v) is int and v > 0
                                                                  for k, v in p["gene_coverage_distribution"].items()), "spec_coverage")
    complete = p["primary_status"] == PRIMARY_COMPLETE
    for key in _PRIMARY_ONLY:
        _require((p[key] is not None) == complete, "spec_primary_only", key)       # never a value for a withheld endpoint
    partial = p["primary_status"] != PRIMARY_REFUSED
    _require((p["partial_top_k"] is not None) == partial, "spec_primary_only", "partial_top_k")
    for key in ("gene1", "nominated_pairs", "counterfactual_bh_nominated_pairs", "integrated_top_k", "burden_top_k", "explained_genes",
                "unscored_genes", "partial_top_k"):
        _require(p[key] is None or (type(p[key]) is list and p[key] == sorted(p[key])), "spec_prediction_order", key)  # sorted lists
    _require(p["delta_h"] is None or type(p["delta_h"]) is int, "spec_prediction_delta")


# ---------------------------------------------------------------------------------------------------- the pair plan (eligibility)
def _kept_annotation(ref_table) -> dict:
    """med_gene's annotation filter: lincRNA / protein_coding, not chrM / chrX / chrY, first row per gene name."""
    kept = {}
    for name, kind, chrom, start, end in ref_table:
        if kind in ("lincRNA", "protein_coding") and chrom not in ("chrM", "chrX", "chrY") and name not in kept:
            kept[name] = (chrom, start, end)
    return kept


def _exposure_sets(trace: dict) -> tuple:
    """(common genes, {exposure: None when not annotated, else (trans genes, valid genes)}) by the pinned code's structural rules."""
    genes = trace["genes"]
    kept = _kept_annotation(trace["ref_table"])
    wes = {g: parse_hex(v) for g, v in zip(genes, trace["wes"])}
    common = [g for g in genes if g in kept]                     # intersect(rownames(p.trans), names(p.wes)) then the annotation
    sets = {}
    for e in trace["exposures"]:
        if e not in kept:
            sets[e] = None
            continue
        chrom, start, end = kept[e]
        lo, hi = start - CIS_DISTANCE, end + CIS_DISTANCE
        cis = {g for g, (c, s, t) in kept.items() if c == chrom and ((lo < s < hi) or (lo < t < hi) or (s <= lo and t >= hi))}
        trans_genes = [g for g in common if g not in cis]
        column = {g: parse_hex(v) for g, v in zip(genes, trace["trans"][e])}
        sets[e] = (trans_genes, [g for g in trans_genes if column[g] is not None and wes[g] is not None])
    return common, sets


def pair_plan(trace: dict) -> tuple:
    """The plan DERIVED FROM THE INPUTS before any score exists, mirroring the pinned code's structural rules for gene exposures:
    returns (plan {(gene, exposure): True scored / False structural} over the genes med_gene keeps (rownames(p.trans) that are named
    in p.wes and in the filtered annotation -- the rows of mat.p), {exposure: exclusion reason}, {exposure: ordered scored genes}).
    A computation-dependent NULL (an invalid pi0 estimate or a non-positive mixture weight sum) is NOT predictable here and is never
    folded into the exclusions: it is a mixture-estimation failure of an ELIGIBLE exposure (ruling 2026-10-08g), observed by the
    exposure recorder -- the primary ranking is withheld and the diagnostics finish (inference/exposure_outcomes.py)."""
    common, sets = _exposure_sets(trace)
    plan, excluded, order = {}, {}, {}
    for e in trace["exposures"]:
        if sets[e] is None:
            excluded[e] = "exposure_not_annotated"
            plan.update({(g, e): False for g in common})
            continue
        trans_genes, valid = sets[e]
        if not trans_genes:
            excluded[e] = "no_trans_genes"
        elif len(valid) < 2:
            excluded[e] = "fewer_than_2_valid_genes"
        scored = set(valid) if e not in excluded else set()
        plan.update({(g, e): g in scored for g in common})
        if scored:
            order[e] = tuple(valid)
    return plan, excluded, order


def usable_pairs(trace: dict) -> dict:
    """{exposure: (trans genes, valid genes)} counted from the inputs for every ANNOTATED exposure -- the counts the exposure recorder
    must observe at its first two guards (an unannotated exposure never reaches run_dandelion_for_exposure; measured)."""
    return {e: (len(v[0]), len(v[1])) for e, v in _exposure_sets(trace)[1].items() if v is not None}


# ---------------------------------------------------------------------------------------------------- scenarios
def _write(path: Path, text: str) -> None:
    with open(path, "xb") as stream:
        stream.write(text.encode("ascii"))


def prepare_scenarios(spec: dict, root) -> tuple:
    """One directory per fixture holding ONLY its inputs (the runner never sees a prediction). Returns the scenario ids in order."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    ids = []
    for f in spec["route_fixtures"]:
        d = root / f["id"]
        d.mkdir()
        _write(d / "scenario.dcf", "Kind: route\nFixture: {}\nTargetFDR: {}\n".format(f["id"], spec["target_fdr"]))
        _write(d / "input.txt", "".join(v + "\n" for v in f["values"]))
        ids.append(f["id"])
    for t in spec["trace_fixtures"]:
        d = root / t["id"]
        d.mkdir()
        _write(d / "scenario.dcf", "Kind: trace\nFixture: {}\nTargetFDR: {}\n".format(t["id"], spec["target_fdr"]))
        _write(d / "genes.txt", "".join(g + "\n" for g in t["genes"]))
        _write(d / "exposures.txt", "".join(e + "\n" for e in t["exposures"]))
        for e in t["exposures"]:
            _write(d / "trans_{}.txt".format(e), "".join(v + "\n" for v in t["trans"][e]))
        _write(d / "wes.txt", "".join(v + "\n" for v in t["wes"]))
        _write(d / "ref_table.tsv", "gene_name\ttype\tChromosome\tstart\tend\n" + "".join("\t".join(map(str, r)) + "\n" for r in t["ref_table"]))
        ids.append(t["id"])
    return tuple(ids)


def _lines(path: Path) -> list:
    raw = path.read_bytes()
    _require(b"\r" not in raw, "run_line_endings", path.name)
    text = raw.decode("ascii")
    _require(text == "" or text.endswith("\n"), "run_truncated", path.name)
    return text.split("\n")[:-1]


def _science_equal(kind: str, off: Path, on: Path) -> dict:
    """Recorder-off and recorder-on scientific outputs, byte for byte (ruling: equality of the science, nominations included). The
    recorder-off run must hold no trace at all (neither the backend nor the exposure recorder's)."""
    for name in _RECORDER_DIRS:
        _require(not (off / name).exists(), "run_off_recorded", str(off / name))
    out = {}
    for name in _SCIENCE_FILES[kind]:
        a, b = off / name, on / name
        _require(a.is_file() == b.is_file(), "run_file_presence", name)
        out[name] = (a.read_bytes() == b.read_bytes()) if a.is_file() else None
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(observed: Fraction, exact: Fraction, tol: Fraction) -> bool:
    return abs(observed - exact) <= tol * exact


# ---------------------------------------------------------------------------------------------------- judging
def _judge_route(f: dict, spec: dict, off: Path, on: Path) -> dict:
    tol = Fraction(spec["bh_relative_tolerance"])
    _require(not (on / "trace_exposures").exists(), "run_unexpected_recorder", str(on))      # safe_qvalues alone: no exposure call
    event = admissible_events(read_trace(on / "trace", expected_exposures=[f["id"]]))[0]
    observed = {"backend": event.backend, "fallback_reason": event.fallback_reason, "n_values": event.n_values, "n_distinct": event.n_distinct}
    clamped = [parse_hex(x) for x in _lines(on / "trace" / (event.call + ".clamped.txt"))]
    output = [parse_hex(x) for x in _lines(on / "output.txt")]
    _require(len(clamped) == len(output) == len(f["values"]), "run_shape", f["id"])
    exact = exact_bh(clamped)
    r_bh = [parse_hex(x) for x in _lines(on / "oracle_bh.txt")]
    checks = {"r_bh_within_tolerance_of_exact_bh": all(_within(o, e, tol) for o, e in zip(r_bh, exact))}
    q1 = on / "oracle_qvalue_pi0_1.txt"
    if q1.is_file():                                     # qvalue installed: with pi0 = 1 it IS Benjamini-Hochberg (independent oracle)
        checks["qvalue_pi0_1_within_tolerance_of_exact_bh"] = all(_within(parse_hex(x), e, tol) for x, e in zip(_lines(q1), exact))
    if event.backend == "BH":
        checks["output_is_r_bh_bitwise"] = (on / "output.txt").read_bytes() == (on / "oracle_bh.txt").read_bytes()
    else:
        _require(q1.is_file(), "run_oracle_missing", f["id"])
        checks["native_not_above_exact_bh"] = all(o <= e * (1 + tol) for o, e in zip(output, exact))      # pi0 <= 1 scales BH down
    science = _science_equal("route", off, on)
    rng = _lines(on / "rng.txt")
    return {"fixture": f["id"], "purpose": f["purpose"],
            "input": {"n": len(f["values"]), "input_sha256": event.input_sha256, "clamped_sha256": event.clamped_sha256},
            "adjustment": {"predicted": f["predicted"], "observed": observed, "agrees": observed == f["predicted"]},
            "pair_decision": {"output_sha256": event.output_sha256, "checks": checks},
            "recorder_equivalence": {"science_files_equal": science, "all_equal": all(v is not False for v in science.values())},
            "random_stream": {"state_unchanged_by_science": rng[0] == rng[1]},
            "warnings": _lines(on / "warnings.txt")}


def explanation_rows(integrated, burden_order, k: int, reference: frozenset, reference_id: str, routes: dict, plan: dict) -> tuple:
    """One DIAGNOSTIC row per gene entering or leaving the integrated top k relative to burden-only (ruling 2026-10-08f section 6):
    both ranks, the score and exposure responsible for the integrated rank, tested-exposure coverage, the responsible exposure's
    adjustment route (it changes that pair's SIGNIFICANCE, never its score), whether the gene sits on a top-k tie, and reference
    membership under the independently defined reference."""
    integrated = tuple(integrated)
    by_gene = {r.gene: (i + 1, r) for i, r in enumerate(integrated)}
    burden_rank = {g: i + 1 for i, g in enumerate(burden_order)}
    top_i, top_b = {r.gene for r in integrated[:k]}, set(burden_order[:k])
    eligible = frozenset(by_gene) & frozenset(burden_rank)
    tied = audit_top_k({r.gene: r.score for r in integrated}, eligible, reference & eligible, k)["tied_genes"]
    ties_i = set(tied) if len(tied) > 1 else set()          # the cutoff gene always "ties" itself; a tie needs a second gene
    rows = []
    for gene, change in sorted([(g, "entered") for g in top_i - top_b] + [(g, "left") for g in top_b - top_i]):
        rank, r = by_gene[gene]
        planned = sorted(e for (g, e), scored in plan.items() if g == gene)
        rows.append({"gene": gene, "change": change, "burden_rank": burden_rank[gene], "integrated_rank": rank,
                     "responsible_exposure": r.best_exposure, "responsible_score": float(r.score).hex(),
                     "tested_exposures": r.tested_pairs, "structural_exposures": r.structural_pairs, "planned_exposures": len(planned),
                     "significant_pairs": r.significant_pairs, "responsible_exposure_route": routes.get(r.best_exposure),
                     "on_integrated_top_k_tie": gene in ties_i, "reference_member": gene in reference, "reference_id": reference_id})
    return tuple(rows)


def _ranked_rows(ranked) -> list:
    return [{"gene": r.gene, "score": r.score, "best_exposure": r.best_exposure, "tested_pairs": r.tested_pairs,
             "significant_pairs": r.significant_pairs, "structural_pairs": r.structural_pairs} for r in ranked]


def _judge_trace(t: dict, spec: dict, off: Path, on: Path, policy: ExposureFailurePolicy) -> dict:
    plan, excluded, order = pair_plan(t)
    # ONE status per planned exposure, from the ACTUAL call, checked against what the inputs predicted before execution (ruling 2026-10-08g)
    statuses = classify_exposures(t["exposures"], excluded, usable_pairs(t), read_exposure_trace(on / "trace_exposures"))
    scored_x = sorted(x for x, s in statuses.items() if s["status"] is ExposureStatus.SCORED)
    refused_x = sorted(x for x, s in statuses.items() if s["status"] in (ExposureStatus.UNCLASSIFIED_MISSING_EXPOSURE, ExposureStatus.INFRASTRUCTURE_ERROR))
    coverage = coverage_report(plan, statuses)
    primary = coverage["primary_status"]
    gene1 = _lines(on / "gene1.txt")
    # safe_qvalues runs only after mat.p is written, so ONLY a scored exposure reaches it (an exposure whose outcome is unknown may)
    events = admissible_events(read_trace(on / "trace"))
    seen = [e.exposure_id for e in events]
    _require(len(set(seen)) == len(seen) and set(scored_x) <= set(seen) <= set(scored_x) | set(refused_x), "trace_incomplete",
             "scored {} / backend calls {}".format(scored_x, seen))
    by_exposure = {e.exposure_id: e for e in events}
    mat_p, mat_sig = {}, {}
    for line in _lines(on / "mat_p.tsv"):
        g, e, v = line.split("\t")
        mat_p[(g, e)] = parse_hex(v)
    for line in _lines(on / "mat_sig.tsv"):
        g, e, v = line.split("\t")
        _require(v in ("0", "1"), "run_mat_sig", line)
        mat_sig[(g, e)] = v == "1"
    _require(set(mat_p) == set(mat_sig) == set(plan), "run_matrix_cells")
    # COMPLETENESS, NOT SILENCE: a score exactly where the plan expects one AND the exposure produced scores
    scored_cells = {c for c, v in mat_p.items() if v is not None}
    planned_cells = {c for c, s in plan.items() if s}
    expected_cells = {c for c in planned_cells if c[1] in scored_x}
    layers, actual_nom, counter_nom, routes = {}, set(), set(), {}
    for e in scored_x:
        ev = by_exposure[e]
        genes = order[e]
        clamped = [parse_hex(x) for x in _lines(on / "trace" / (ev.call + ".clamped.txt"))]
        recorded_input = [parse_hex(x) for x in _lines(on / "trace" / (ev.call + ".input.txt"))]
        q = [parse_hex(x) for x in _lines(on / "trace" / (ev.call + ".output.txt"))]
        _require(len(genes) == len(clamped) == len(q) == len(recorded_input), "run_shape", e)
        # the recorder's input IS the mat.p column the ranking consumes (bitwise, in the plan's gene order)
        input_is_mat_p = all(recorded_input[i] == mat_p[(g, e)] for i, g in enumerate(genes))
        decisions = {g: significant(q[i], spec["target_fdr"]) for i, g in enumerate(genes)}
        bh = exact_bh(clamped)
        counter = {g: significant(bh[i], spec["target_fdr"]) for i, g in enumerate(genes)}
        routes[e] = {"backend": ev.backend, "fallback_reason": ev.fallback_reason}
        actual_nom |= {(e, g) for g, d in decisions.items() if d}
        counter_nom |= {(e, g) for g, d in counter.items() if d}
        layers[e] = {"input": {"n_values": ev.n_values, "n_distinct": ev.n_distinct, "input_sha256": ev.input_sha256,
                               "clamped_sha256": ev.clamped_sha256, "recorder_input_equals_mat_p": input_is_mat_p},
                     "adjustment": routes[e],
                     "pair_decision": {"output_sha256": ev.output_sha256,
                                       "decisions_equal_mat_sig": all(decisions[g] == mat_sig[(g, e)] for g in genes),
                                       "significant": sorted(g for g, d in decisions.items() if d),
                                       "significant_under_exact_bh": sorted(g for g, d in counter.items() if d)}}
    nominations = set()
    for line in _lines(on / "nominations.tsv"):
        e, g, _ = line.split("\t")
        nominations.add((e, g))
    # gene aggregation: the SAME scores whatever the adjustment (mat.p precedes safe_qvalues); significance is descriptive only
    pairs = []
    for (g, e), scored in sorted(plan.items()):
        if not scored:
            pairs.append(Pair(g, e, State.STRUCTURAL))
        elif mat_p[(g, e)] is None:
            pairs.append(Pair(g, e, State.FAILED))
        else:
            pairs.append(Pair(g, e, State.SCORED, mat_p[(g, e)], mat_sig[(g, e)]))
    counter_pairs = [Pair(p.gene, p.exposure, p.state, p.score, ((p.exposure, p.gene) in counter_nom) if p.state is State.SCORED else None)
                     for p in pairs]
    genes_with_scores = {g for (g, e), s in plan.items() if s}
    ranking_plan = {c: s for c, s in plan.items() if c[0] in genes_with_scores}
    wes = {g: parse_hex(v) for g, v in zip(t["genes"], t["wes"])}
    eligible = frozenset(g for g in genes_with_scores if wes[g] is not None)
    burden_order = tuple(sorted(eligible, key=lambda g: (wes[g], g)))
    reference = frozenset(t["reference_positives"]) & eligible
    integrated = counterfactual = partial = partial_counter = None
    if primary == PRIMARY_COMPLETE:
        integrated = rank_genes([p for p in pairs if p.gene in genes_with_scores], ranking_plan)
        counterfactual = rank_genes([p for p in counter_pairs if p.gene in genes_with_scores], ranking_plan)
    if primary != PRIMARY_REFUSED:
        partial = partial_ranking(pairs, plan, statuses, policy)
        partial_counter = partial_ranking(counter_pairs, plan, statuses, policy)
    basis, ranked, ranked_counter = ((None, None, None) if partial is None else
                                     ("primary", integrated, counterfactual) if integrated is not None else
                                     ("partial_exploratory", partial["ranked"], partial_counter["ranked"]))
    if integrated is not None:
        contract = EndpointContract(t["k"], eligible, reference, t["reference_id"], "not_applicable_fixture")
        integrated_order = tuple(r.gene for r in integrated if r.gene in eligible)
        contrast = recovery_contrast(integrated_order, burden_order, contract)
        counter_contrast = recovery_contrast(tuple(r.gene for r in counterfactual if r.gene in eligible), burden_order, contract)
        explained = explanation_rows([r for r in integrated if r.gene in eligible], burden_order, t["k"], reference, t["reference_id"],
                                     routes, ranking_plan)
        endpoint = {"k": t["k"], "reference_id": t["reference_id"],
                    "contrast": {k: list(v) if type(v) is tuple else v for k, v in contrast.items()},
                    "unchanged_under_exact_bh": contrast == counter_contrast}
    else:
        integrated_order, contrast, explained = None, None, None
        endpoint = {"withheld": "primary ranking {}: exposure completion {} ({})".format(
            primary, coverage["exposure_completion"], ", ".join("{} {}".format(x, s) for x, s in coverage["failed_eligible_exposures"].items()))}
    records = exposure_records(plan, statuses)
    observed = {"gene1": sorted(gene1), "excluded_exposures": dict(sorted(excluded.items())), "routes": routes,
                "nominated_pairs": sorted(map(list, nominations)), "counterfactual_bh_nominated_pairs": sorted(map(list, counter_nom)),
                "integrated_top_k": None if integrated_order is None else sorted(integrated_order[:t["k"]]),
                "burden_top_k": sorted(burden_order[:t["k"]]), "delta_h": None if contrast is None else contrast["delta"],
                "explained_genes": None if explained is None else sorted(r["gene"] for r in explained),
                "exposure_outcomes": {x: s["status"].value for x, s in sorted(statuses.items())},
                "mixture_failure_side": {x: r["invalid_side"] for x, r in records.items() if "invalid_side" in r},
                "exposure_completion": coverage["exposure_completion"], "primary_status": primary,
                "gene_coverage_distribution": coverage["gene_coverage_distribution"],
                "unscored_genes": sorted(g for g, r in coverage["per_gene_coverage"].items() if r["scored_pairs"] == 0),
                "partial_top_k": None if partial is None else sorted(r.gene for r in partial["ranked"][:t["k"]])}
    p = t["predicted"]
    agreement = {key: observed[key] == p[key] for key in p}
    science = _science_equal("trace", off, on)
    rng = _lines(on / "rng.txt")
    return {"fixture": t["id"], "purpose": t["purpose"], "exposure_outcomes": records, "coverage": coverage,
            "endpoint_release": {stage: endpoint_release(primary, stage) for stage in ("confirmatory", "feasibility")},
            "exposures": layers,
            "consistency": {"scores_exactly_where_planned_and_scored": scored_cells == expected_cells,
                            "nominations_equal_mat_sig_pairs": nominations == actual_nom,
                            "gene1_equals_scored_exposures": sorted(gene1) == scored_x},
            "gene_aggregation": {"rule": "minimum raw DANDELION p over planned pairs; ties by gene identity", "basis": basis,
                                 "planned_pairs": len(planned_cells), "structural_pairs": len(plan) - len(planned_cells),
                                 "ranking_scores_unchanged_under_exact_bh": None if ranked is None else
                                 [(r.gene, r.score) for r in ranked] == [(r.gene, r.score) for r in ranked_counter],
                                 "significant_pair_counts_changed": None if ranked is None else
                                 sorted(a.gene for a, b in zip(ranked, ranked_counter) if a.significant_pairs != b.significant_pairs)},
            "endpoint": endpoint,
            "partial_ranking": None if partial is None else {**{k: v for k, v in partial.items() if k != "ranked"},
                                                             "ranked": _ranked_rows(partial["ranked"])},
            "nominations": {"actual": sorted(map(list, nominations)), "under_exact_bh": sorted(map(list, counter_nom)),
                            "changed_by_adjustment": sorted(map(list, nominations ^ counter_nom))},
            "explanation_rows": None if explained is None else list(explained),
            "prediction": {"predicted": p, "observed": observed, "agrees": agreement},
            "recorder_equivalence": {"science_files_equal": science, "all_equal": all(v is not False for v in science.values())},
            "random_stream": {"state_unchanged_by_science": rng[0] == rng[1]},
            "warnings": _lines(on / "warnings.txt")}


def _environment(run: Path, mode: str) -> dict:
    out = {}
    for line in _lines(run / "environment.tsv"):
        key, _, value = line.partition("\t")
        _require(key not in out, "run_environment_key", key)
        out[key] = value
    _require(set(out) == {"R", "platform", "rng", "libpaths", "DANDELION", "qvalue", "mode"} and out.pop("mode") == mode, "run_environment", str(run))
    for pkg in ("DANDELION", "qvalue"):
        version, _, path = out.get(pkg, "").partition("\t")
        out[pkg] = {"version": version, "description_sha256": _sha(Path(path)) if path and Path(path).is_file() else None}
    return out


def judge(spec_bytes: bytes, spec_sha256: str, runs) -> dict:
    """The report over every fixture: runs/<id>/off and runs/<id>/on as written by the runner. Disagreements are FINDINGS in the
    report (fixtures_passed false), never exceptions; malformed or incomplete evidence refuses. The exposure-failure policy applied is
    the ruling's own (ExposureFailurePolicy defaults, ruling 2026-10-08g); a primary analysis refused for an unclassified-missing or
    infrastructure outcome is always a finding."""
    _require(hashlib.sha256(spec_bytes).hexdigest() == spec_sha256, "spec_digest")
    spec = load_spec(spec_bytes)
    policy = ExposureFailurePolicy()
    runs = Path(runs)
    routes = [_judge_route(f, spec, runs / f["id"] / "off", runs / f["id"] / "on") for f in spec["route_fixtures"]]
    traces = [_judge_trace(t, spec, runs / t["id"] / "off", runs / t["id"] / "on", policy) for t in spec["trace_fixtures"]]
    # ONE environment for every run (R, platform, random-number kind, DANDELION and qvalue identities), each run in its stated mode
    envs = [_environment(runs / i / m, m) for i in [f["id"] for f in spec["route_fixtures"]] + [t["id"] for t in spec["trace_fixtures"]]
            for m in ("off", "on")]
    _require(all(e == envs[0] for e in envs), "run_environments_differ")
    failures = []
    for r in routes:
        if not r["adjustment"]["agrees"]:
            failures.append(r["fixture"] + ":route_prediction")
        failures += [r["fixture"] + ":" + k for k, v in r["pair_decision"]["checks"].items() if not v]
        if not r["recorder_equivalence"]["all_equal"]:
            failures.append(r["fixture"] + ":recorder_changed_science")
        if not r["random_stream"]["state_unchanged_by_science"]:
            failures.append(r["fixture"] + ":random_stream_consumed")
        if r["warnings"]:
            failures.append(r["fixture"] + ":warnings_raised")
    for trace in traces:
        i = trace["fixture"]
        failures += [i + ":prediction:" + k for k, v in trace["prediction"]["agrees"].items() if not v]
        failures += [i + ":" + k for k, v in trace["consistency"].items() if not v]
        failures += [i + ":" + k for e in trace["exposures"].values() for k, v in
                     (("recorder_input_equals_mat_p", e["input"]["recorder_input_equals_mat_p"]),
                      ("decisions_equal_mat_sig", e["pair_decision"]["decisions_equal_mat_sig"])) if not v]
        failures += ["{}:primary_refused:{}:{}".format(i, x, r["status"]) for x, r in trace["exposure_outcomes"].items()
                     if r["status"] in (ExposureStatus.UNCLASSIFIED_MISSING_EXPOSURE.value, ExposureStatus.INFRASTRUCTURE_ERROR.value)]
        if not trace["recorder_equivalence"]["all_equal"]:
            failures.append(i + ":recorder_changed_science")
        if not trace["random_stream"]["state_unchanged_by_science"]:
            failures.append(i + ":random_stream_consumed")
        if trace["warnings"]:
            failures.append(i + ":warnings_raised")
    return {"schema": REPORT_SCHEMA, "spec_sha256": spec_sha256, "environment": envs[0], "exposure_failure_policy": asdict(policy),
            "route_fixtures": routes, "trace_fixtures": traces, "failures": sorted(failures), "fixtures_passed": not failures}


def render_report(report: dict) -> bytes:
    def default(value):
        if type(value) is Fraction:
            return float(value).hex() if Fraction(float(value)) == value else "{}/{}".format(value.numerator, value.denominator)
        raise TypeError(type(value).__name__)
    return (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, default=default) + "\n").encode("ascii")
