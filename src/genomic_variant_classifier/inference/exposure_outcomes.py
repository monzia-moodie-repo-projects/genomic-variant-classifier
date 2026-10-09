"""Per-exposure outcomes, exposure completion, per-gene scoring coverage, and the primary / diagnostic split (owner ruling 2026-10-08g).

THE RULE (refuse the primary endpoint, let computation and diagnosis finish)
===========================================================================
    exposure fails a frozen structural rule           -> STRUCTURALLY_INELIGIBLE: excluded before the planned eligible set exists
    eligible exposure produces scores                 -> SCORED
    eligible exposure: invalid mixture estimates      -> MIXTURE_ESTIMATE_INVALID  } primary ranking INCOMPLETE: withhold it and
    eligible exposure: non-positive weight sum        -> NONPOSITIVE_WEIGHT_SUM    } Delta H(20); keep every successful output
    eligible exposure disappears, no identified cause -> UNCLASSIFIED_MISSING_EXPOSURE  } REFUSE
    infrastructure or implementation error            -> INFRASTRUCTURE_ERROR           } (never "biological non-estimability")

A mixture failure is NEVER relabelled structural after execution: the structural reason and the usable-pair counts are predicted from
the inputs BEFORE execution (method_trace.pair_plan, method_trace.usable_pairs) and the observed outcome
(scripts/dandelion/dandelion_exposure_recorder.R, an actual-call trace) must AGREE with them, or the evidence is refused. A failed
exposure's planned pairs stay "computation failed" everywhere -- in the partial ranking they are COUNTED as estimation-failed pairs,
never folded into the structural count. This is distinct from the q-value fallback, which happens downstream of existing scores and
can change significance decisions, never scores.

MEASURED (DANDELION 0.1.0 source and actual calls; nonnullPropEst; R 4.3.3): after the "< 2 valid genes" guard every p-value is clamped
to a finite z-score, so a NA mixture estimate is unreachable; wg.sum = 1 - (1 - pi0a)(1 - pi0b) for pi0 in [0, 1], so a non-positive
weight sum needs pi0a = pi0b = 0 exactly; the reachable failure is a NEGATIVE pi0 estimate (every trans p-value equal to 1 gives
pi0a = -0x1.e7bab1690b2p-7, about -0.0149, for 30 and for 31 genes). pi0b (the burden side) is estimated from the burden p-values of
the exposure's VALID genes: exposures with the same valid gene set share it bit for bit (E1 and E2 of the fixtures: 0x1.c86b0bd2cbe9bp-1),
so with near-complete data a burden-side failure would be near-systematic across exposures.

REPORTED: exposure completion = scored eligible / planned eligible (exact counts), and the DISTRIBUTION of per-gene coverage (scored /
planned pairs per gene over the planned gene universe) -- a small exposure-failure fraction can still matter for particular genes, and
no universal "under 5 % is fine" threshold exists. The partial ranking is EXPLORATORY, conditional on estimability: it keeps the
planned gene universe and lists genes without any score as "unscored" -- never deleted, never given a manufactured score. In the
FEASIBILITY stage reference recovery and Delta H(20) are withheld even when every exposure is scored (endpoint_release).

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path

from genomic_variant_classifier.inference.analysis_contract import ExposureFailurePolicy
from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.ranking import Pair, State, rank_genes

logger = logging.getLogger(__name__)

__all__ = ["EXPOSURE_RECORDER_VERSION", "PARTIAL_RANKING_DEFINITION", "ExposureStatus", "ExposureEvent", "read_exposure_trace",
           "classify_exposures", "planned_universe", "coverage_report", "exposure_records", "partial_ranking", "endpoint_release",
           "PRIMARY_COMPLETE", "PRIMARY_WITHHELD", "PRIMARY_REFUSED", "STAGES"]

EXPOSURE_RECORDER_VERSION = "gvc.dandelion-exposure-recorder/1"
PARTIAL_RANKING_DEFINITION = ("Ranking conditional on exposures for which DANDELION produced scores; exploratory and not a substitute "
                              "for the complete-exposure primary endpoint.")
PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED = "complete", "withheld", "refused"
STAGES = ("feasibility", "confirmatory")
_KEYS = {"exposure_id", "outcome", "last_guard_reached", "n_trans", "n_valid", "pi0a", "pi0b", "wg1", "wg2", "wg3", "wg_sum",
         "observation_kind", "recorder_version"}
_HEX = re.compile(r"(?P<sign>-?)0x(?P<whole>[0-9a-f])(?:\.(?P<frac>[0-9a-f]+))?p(?P<exp>[+-]\d{1,4})")
# outcome -> the guard the call must have reached (number of tracer hits); None: any (an abnormal or unclassified exit)
_OUTCOMES = {"no_trans_genes": 1, "fewer_than_2_valid_genes": 2, "mixture_estimate_invalid": 3, "nonpositive_weight_sum": 4,
             "scored": 4, "abnormal_exit": None, "unclassified": None}
_STRUCTURAL = {"no_trans_genes", "fewer_than_2_valid_genes"}
_PREDICTABLE = _STRUCTURAL | {"exposure_not_annotated"}


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


class ExposureStatus(str, Enum):
    STRUCTURALLY_INELIGIBLE = "structurally_ineligible"
    SCORED = "scored"
    MIXTURE_ESTIMATE_INVALID = "mixture_estimate_invalid"
    NONPOSITIVE_WEIGHT_SUM = "nonpositive_weight_sum"
    UNCLASSIFIED_MISSING_EXPOSURE = "unclassified_missing_exposure"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


_ESTIMATION_FAILURES = frozenset({ExposureStatus.MIXTURE_ESTIMATE_INVALID, ExposureStatus.NONPOSITIVE_WEIGHT_SUM})
_REFUSALS = frozenset({ExposureStatus.UNCLASSIFIED_MISSING_EXPOSURE, ExposureStatus.INFRASTRUCTURE_ERROR})


def _value(text):
    """An exact recorded number: hexadecimal binary64 -> Fraction, "NA" -> "NA" (kept explicitly), JSON null -> None (not reached)."""
    if text is None or text == "NA":
        return text
    m = _HEX.fullmatch(text) if type(text) is str else None
    _require(m is not None, "exposure_trace_value", repr(text)[:40])
    frac = m["frac"] or ""
    exact = Fraction(int(m["whole"] + frac, 16), 16 ** len(frac)) * Fraction(2) ** int(m["exp"])
    exact = -exact if m["sign"] else exact
    try:
        stored = Fraction(float.fromhex(text))
    except OverflowError:
        raise InferenceError("exposure_trace_value", text) from None
    _require(stored == exact, "exposure_trace_value_not_binary64", text)      # no rounding is ever accepted
    return exact


@dataclass(frozen=True)
class ExposureEvent:
    exposure_id: str
    outcome: str
    last_guard_reached: int
    n_trans: int | None
    n_valid: int | None
    pi0a: object        # Fraction, "NA", or None (guard not reached)
    pi0b: object
    wg1: object
    wg2: object
    wg3: object
    wg_sum: object


def read_exposure_trace(directory) -> tuple:
    """Strictly parse <dir>/exposures.jsonl (the recorder creates it empty at start): exact keys, one event per exposure, every recorded
    quantity consistent with the guard the call reached (a structural outcome carries no estimate; an estimation failure carries its
    invalid estimate; scores need valid ones). Zero events is a valid trace (no exposure entered run_dandelion_for_exposure)."""
    directory = Path(directory)
    path = directory / "exposures.jsonl"
    _require(path.is_file(), "exposure_trace_missing", str(directory))
    stray = sorted(p.name for p in directory.iterdir() if p.name != "exposures.jsonl")
    _require(not stray, "exposure_trace_unexpected_files", repr(stray[:5]))
    raw = path.read_bytes()
    _require(b"\r" not in raw and (raw == b"" or raw.endswith(b"\n")), "exposure_trace_truncated")

    def strict(pairs):
        out = {}
        for k, v in pairs:
            _require(k not in out, "exposure_trace_duplicate_key", k)
            out[k] = v
        return out

    def refuse(token):
        raise InferenceError("exposure_trace_number", token)
    try:
        lines = raw.decode("ascii").split("\n")[:-1]
    except UnicodeDecodeError:
        raise InferenceError("exposure_trace_encoding") from None
    events, seen = [], set()
    for n, line in enumerate(lines, 1):
        try:
            doc = json.loads(line, object_pairs_hook=strict, parse_float=refuse, parse_constant=refuse)
        except json.JSONDecodeError:
            raise InferenceError("exposure_trace_json", "line {}".format(n)) from None
        _require(type(doc) is dict and set(doc) == _KEYS, "exposure_trace_keys", "line {}".format(n))
        _require(doc["observation_kind"] == "actual_call_trace" and doc["recorder_version"] == EXPOSURE_RECORDER_VERSION
                 and doc["outcome"] in _OUTCOMES and type(doc["exposure_id"]) is str and bool(doc["exposure_id"])
                 and type(doc["last_guard_reached"]) is int and 1 <= doc["last_guard_reached"] <= 4, "exposure_trace_value",
                 "line {}".format(n))
        _require(doc["exposure_id"] not in seen, "exposure_trace_duplicate_exposure", doc["exposure_id"])
        seen.add(doc["exposure_id"])
        hits, outcome = doc["last_guard_reached"], doc["outcome"]
        expected = _OUTCOMES[outcome]
        _require(expected is None or hits == expected, "exposure_trace_inconsistent", "line {}".format(n))
        for k in ("n_trans", "n_valid"):
            _require(doc[k] is None or (type(doc[k]) is int and doc[k] >= 0), "exposure_trace_value", k)
        e = ExposureEvent(doc["exposure_id"], outcome, hits, doc["n_trans"], doc["n_valid"],
                          *(_value(doc[k]) for k in ("pi0a", "pi0b", "wg1", "wg2", "wg3", "wg_sum")))
        _require((e.n_trans is not None) == (hits >= 1) and (e.n_valid is not None) == (hits >= 2)
                 and all((v is not None) == (hits >= 3) for v in (e.pi0a, e.pi0b))
                 and all((v is not None) == (hits >= 4) for v in (e.wg1, e.wg2, e.wg3, e.wg_sum)), "exposure_trace_inconsistent", e.exposure_id)
        _require(e.n_valid is None or e.n_valid <= e.n_trans, "exposure_trace_inconsistent", e.exposure_id)
        invalid_pi0 = hits >= 3 and any(v == "NA" or v < 0 for v in (e.pi0a, e.pi0b))
        if outcome == "no_trans_genes":
            _require(e.n_trans == 0, "exposure_trace_inconsistent", e.exposure_id)
        elif outcome == "fewer_than_2_valid_genes":
            _require(e.n_trans > 0 and e.n_valid < 2, "exposure_trace_inconsistent", e.exposure_id)
        elif outcome == "mixture_estimate_invalid":
            _require(e.n_valid >= 2 and invalid_pi0, "exposure_trace_inconsistent", e.exposure_id)
        elif outcome in ("nonpositive_weight_sum", "scored"):
            _require(e.n_valid >= 2 and not invalid_pi0 and e.wg_sum != "NA", "exposure_trace_inconsistent", e.exposure_id)
            _require((e.wg_sum <= 0) == (outcome == "nonpositive_weight_sum"), "exposure_trace_inconsistent", e.exposure_id)
        events.append(e)
    return tuple(events)


def classify_exposures(exposures, predicted_exclusions: dict, predicted_usable: dict, events) -> dict:
    """One status per PLANNED exposure (ruling: require_one_status_per_planned_exposure).

    predicted_exclusions: {exposure: structural reason}, and predicted_usable: {exposure: (n_trans, n_valid)} for every annotated
    exposure -- both derived from the inputs BEFORE execution. The observed call must agree with them: an exposure predicted eligible
    that the method reports structural (or the reverse), or a usable-pair count that differs, means the frozen rule does not describe
    the implementation -- refused, never reconciled after the fact."""
    exposures = tuple(exposures)
    _require(bool(exposures) and len(set(exposures)) == len(exposures) and all(type(x) is str and x for x in exposures), "exposure_plan")
    _require(set(predicted_exclusions) <= set(exposures) and set(predicted_exclusions.values()) <= _PREDICTABLE, "eligibility_rule_unknown",
             repr(sorted(set(predicted_exclusions.values()) - _PREDICTABLE)[:3]))
    annotated = {x for x in exposures if predicted_exclusions.get(x) != "exposure_not_annotated"}
    _require(set(predicted_usable) == annotated, "usable_pair_prediction_scope", "counts are predicted for exactly the annotated exposures")
    events = tuple(events)
    _require(all(type(e) is ExposureEvent for e in events), "exposure_event_type")
    by_id = {e.exposure_id: e for e in events}
    _require(len(by_id) == len(events), "exposure_trace_duplicate_exposure")
    _require(set(by_id) <= set(exposures), "exposure_trace_unplanned", repr(sorted(set(by_id) - set(exposures))[:5]))
    out = {}
    for x in exposures:
        event, predicted = by_id.get(x), predicted_exclusions.get(x)
        if predicted == "exposure_not_annotated":                       # med_gene never calls run_dandelion_for_exposure for it (measured)
            _require(event is None, "eligibility_rule_mismatch", "{} predicted unannotated, observed {}".format(x, event and event.outcome))
            out[x] = {"status": ExposureStatus.STRUCTURALLY_INELIGIBLE, "reason": predicted, "event": None}
            continue
        if event is not None:
            n_trans, n_valid = predicted_usable[x]
            _require(event.n_trans == n_trans and (event.n_valid is None or event.n_valid == n_valid), "usable_pair_count_mismatch",
                     "{}: predicted {} trans / {} valid, observed {} / {}".format(x, n_trans, n_valid, event.n_trans, event.n_valid))
        if predicted in _STRUCTURAL:
            if event is None:
                out[x] = {"status": ExposureStatus.UNCLASSIFIED_MISSING_EXPOSURE, "reason": "no observed call (predicted " + predicted + ")",
                          "event": None}
            elif event.outcome in ("abnormal_exit", "unclassified"):
                out[x] = {"status": ExposureStatus.INFRASTRUCTURE_ERROR, "reason": event.outcome, "event": event}
            else:
                _require(event.outcome == predicted, "eligibility_rule_mismatch", "{} predicted {}, observed {}".format(x, predicted, event.outcome))
                out[x] = {"status": ExposureStatus.STRUCTURALLY_INELIGIBLE, "reason": predicted, "event": event}
        elif event is None:
            out[x] = {"status": ExposureStatus.UNCLASSIFIED_MISSING_EXPOSURE, "reason": "no observed call", "event": None}
        elif event.outcome in _STRUCTURAL:
            raise InferenceError("eligibility_rule_mismatch", "{} predicted eligible, observed {}".format(x, event.outcome))
        elif event.outcome in ("abnormal_exit", "unclassified"):
            out[x] = {"status": ExposureStatus.INFRASTRUCTURE_ERROR, "reason": event.outcome, "event": event}
        else:
            out[x] = {"status": ExposureStatus(event.outcome), "reason": event.outcome, "event": event}
    return out


def _check_plan(plan: dict) -> None:
    _require(type(plan) is dict and bool(plan), "empty_plan")
    for key, scored in plan.items():
        _require(type(key) is tuple and len(key) == 2 and all(type(x) is str and x for x in key) and type(scored) is bool, "invalid_plan")


def _check_statuses(plan: dict, statuses: dict) -> None:
    _require(type(statuses) is dict and bool(statuses) and all(type(s.get("status")) is ExposureStatus for s in statuses.values()),
             "exposure_status_type")
    _require({x for _, x in plan} <= set(statuses), "exposure_status_missing")
    for (g, x), scored in plan.items():
        # a structurally ineligible exposure has no planned score by construction; finding one means plan and statuses disagree
        _require(not (scored and statuses[x]["status"] is ExposureStatus.STRUCTURALLY_INELIGIBLE), "plan_status_mismatch", x)


def planned_universe(plan: dict) -> tuple:
    """The genes the PRIMARY ranking ranks: every gene with at least one planned score (method_trace ranks exactly these)."""
    _check_plan(plan)
    return tuple(sorted({g for (g, _), scored in plan.items() if scored}))


def coverage_report(plan: dict, statuses: dict) -> dict:
    """Exposure completion and the per-gene coverage distribution over the PLANNED universe, and the primary status."""
    _check_plan(plan)
    _check_statuses(plan, statuses)
    eligible = sorted(x for x, s in statuses.items() if s["status"] is not ExposureStatus.STRUCTURALLY_INELIGIBLE)
    _require(bool(eligible), "no_eligible_exposure")
    scored = [x for x in eligible if statuses[x]["status"] is ExposureStatus.SCORED]
    universe = planned_universe(plan)
    per_gene = {g: {"planned_pairs": 0, "scored_pairs": 0} for g in universe}
    for (g, x), planned in plan.items():
        if planned:
            per_gene[g]["planned_pairs"] += 1
            per_gene[g]["scored_pairs"] += statuses[x]["status"] is ExposureStatus.SCORED
    distribution = {}
    for row in per_gene.values():
        key = "{}/{}".format(row["scored_pairs"], row["planned_pairs"])
        distribution[key] = distribution.get(key, 0) + 1
    if any(s["status"] in _REFUSALS for s in statuses.values()):
        primary = PRIMARY_REFUSED
    elif len(scored) == len(eligible):
        primary = PRIMARY_COMPLETE
    else:
        primary = PRIMARY_WITHHELD
    return {"planned_eligible_exposures": len(eligible), "successfully_scored_eligible_exposures": len(scored),
            "exposure_completion": "{}/{}".format(len(scored), len(eligible)), "primary_status": primary,
            "failed_eligible_exposures": {x: statuses[x]["status"].value for x in eligible if x not in scored},
            "structurally_ineligible_exposures": {x: s["reason"] for x, s in sorted(statuses.items())
                                                  if s["status"] is ExposureStatus.STRUCTURALLY_INELIGIBLE},
            "gene_coverage_distribution": dict(sorted(distribution.items(), key=lambda kv: tuple(map(int, kv[0].split("/"))))),
            "per_gene_coverage": per_gene,
            "genes_with_incomplete_coverage": sorted(g for g, r in per_gene.items() if r["scored_pairs"] < r["planned_pairs"]),
            "genes_outside_planned_universe": sorted({g for g, _ in plan} - set(universe))}


def _render(value):
    if type(value) is Fraction:             # exact: _value admitted only binary64 values; written in R's "%a" form
        mantissa, exponent = float(value).hex().split("p")
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return mantissa + "p" + exponent
    return value                            # "NA" or None


def exposure_records(plan: dict, statuses: dict) -> dict:
    """What the qualified run preserves per exposure (ruling): its exact category, the guard reached, the usable-pair counts, the mixture
    quantities AS ESTIMATED (invalid values explicit; null = never reached), and the genes whose scoring coverage it would contribute."""
    _check_plan(plan)
    _check_statuses(plan, statuses)
    out = {}
    for x, s in sorted(statuses.items()):
        e = s["event"]
        genes = sorted(g for (g, y), scored in plan.items() if y == x and scored)
        row = {"status": s["status"].value, "reason": s["reason"], "planned_genes": len(genes)}
        if s["status"] in _ESTIMATION_FAILURES or s["status"] in _REFUSALS:
            row["genes_it_would_cover"] = genes
        if e is None:
            row["observed"] = None
        else:
            row["observed"] = {"outcome": e.outcome, "last_guard_reached": e.last_guard_reached, "n_trans": e.n_trans,
                               "usable_pairs": e.n_valid, **{k: _render(getattr(e, k)) for k in ("pi0a", "pi0b", "wg1", "wg2", "wg3", "wg_sum")}}
            if s["status"] is ExposureStatus.MIXTURE_ESTIMATE_INVALID:
                bad = [side for side, v in (("trans", e.pi0a), ("burden", e.pi0b)) if v == "NA" or v < 0]
                row["invalid_side"] = "both" if len(bad) == 2 else bad[0]
        out[x] = row
    return out


def partial_ranking(pairs, plan: dict, statuses: dict, policy: ExposureFailurePolicy) -> dict:
    """EXPLORATORY ranking conditional on the exposures for which DANDELION produced scores (ruling 2026-10-08g): the same minimum-score
    rule over SCORED exposures only. The planned universe is kept; a gene with no score is listed as "unscored". A failed exposure's
    planned pairs must be rows in the FAILED state (never relabelled structural) and are reported per gene as estimation-failed pairs.
    Refused when the sealed policy does not pre-register a partial ranking, or when any exposure is unclassified or failed for an
    infrastructure reason (that is a refusal of the run, not a diagnostic)."""
    _require(type(policy) is ExposureFailurePolicy, "policy_type")
    _require(policy.diagnostic.partial_ranking_allowed, "partial_ranking_not_preregistered")
    universe = planned_universe(plan)
    _check_statuses(plan, statuses)
    _require(not any(s["status"] in _REFUSALS for s in statuses.values()), "partial_ranking_refused",
             repr(sorted(x for x, s in statuses.items() if s["status"] in _REFUSALS)[:5]))
    rows = tuple(pairs)
    _require(all(type(p) is Pair for p in rows), "pair_type")
    keys = [(p.gene, p.exposure) for p in rows]
    _require(len(set(keys)) == len(keys), "duplicate_pair")
    _require(set(keys) == set(plan), "missing_pair_record" if set(keys) < set(plan) else "unexpected_pair")
    failed = {x for x, s in statuses.items() if s["status"] in _ESTIMATION_FAILURES}
    failed_pairs = {}
    for p in rows:
        if p.exposure in failed:
            planned = plan[(p.gene, p.exposure)]
            _require(p.state is not State.SCORED and p.score is None and p.significant is None, "failed_exposure_has_scores", p.exposure)
            _require(p.state is (State.FAILED if planned else State.STRUCTURAL), "failed_exposure_relabelled", "{} {}".format(p.gene, p.exposure))
            if planned:
                failed_pairs[p.gene] = failed_pairs.get(p.gene, 0) + 1
    sub_plan = {c: v for c, v in plan.items() if c[1] not in failed}
    rankable = {g for (g, _), v in sub_plan.items() if v}
    kept = [p for p in rows if p.exposure not in failed and p.gene in rankable]
    ranked = rank_genes(kept, {c: v for c, v in sub_plan.items() if c[0] in rankable}) if rankable else ()
    return {"status": policy.diagnostic.status, "definition": PARTIAL_RANKING_DEFINITION, "ranked": ranked,
            "unscored": sorted(set(universe) - rankable), "missing_score_representation": policy.diagnostic.missing_score_representation,
            "planned_universe": list(universe), "conditioned_on_exposures": sorted(x for x, s in statuses.items() if s["status"] is ExposureStatus.SCORED),
            "estimation_failed_pairs_per_gene": dict(sorted(failed_pairs.items()))}


def endpoint_release(primary_status: str, stage: str) -> dict:
    """What may be released (ruling 2026-10-08g). FEASIBILITY: reference recovery and Delta H(20) are withheld whatever the completion
    (benchmark diagnostics stay development information); the complete ranking may be computed but is not evaluated. CONFIRMATORY: the
    primary ranking, reference recovery and Delta H(20) are released only when every eligible exposure is scored. Never: a performance-
    based choice among exposure-failure policies -- a revised policy is a new contract version."""
    _require(primary_status in (PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED), "primary_status")
    _require(stage in STAGES, "stage")
    if primary_status == PRIMARY_REFUSED:
        primary = evaluation = "refused"
        partial = "refused"
    else:
        partial = "exploratory"
        if primary_status == PRIMARY_WITHHELD:
            primary = evaluation = "withheld_incomplete_exposures"
        elif stage == "feasibility":
            primary, evaluation = "computed_not_evaluated", "withheld_feasibility_stage"
        else:
            primary = evaluation = "released"
    return {"stage": stage, "primary_status": primary_status, "primary_ranking": primary, "reference_recovery": evaluation,
            "delta_h20": evaluation, "partial_ranking": partial, "policy_selection": "not_permitted"}
