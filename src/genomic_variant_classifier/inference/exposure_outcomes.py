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

OWNER RULING 2026-10-09 (added here, the authoritative owner of exposure outcomes and coverage)
==============================================================================================
    PairPlan          the plan FROZEN BEFORE EXECUTION as one typed object with a canonical rendering and a digest: the planned
                      exposures, the pair grid, the structural exclusions, the usable-pair counts and each eligible exposure's ORDERED
                      valid genes. A run intent binds its digest, so eligibility cannot move after outcomes are seen.
    score_coverage    the ruling's coverage rule over ADMITTED evidence (statuses from classify_exposures, scores from
                      ranking.read_score_matrix): an empty eligible set, an unclassified or infrastructure outcome, a missing or extra
                      grid cell, a score for an unplanned pair, or scores that disagree with the exposure outcomes REFUSE; a recorded
                      mixture failure WITHHOLDS; otherwise COMPLETE. Completeness is DERIVED -- no caller-supplied success flag exists.
    RELEASE POLICY    endpoint_release reads ONE declared, immutable decision table whose canonical rendering has a digest
                      (release_policy_sha256). A run intent binds that digest: changing release semantics changes the identity, so an
                      earlier intent can never silently acquire a new meaning.
    RECORDER v2       the exposure recorder also writes the EFFECTIVE BURDEN INPUT of every call that reaches the pi0 guard -- the
                      ordered gene identifiers and the exact post-clamp burden p-values (names(p_b), p_b) -- once per DISTINCT input
                      (burden-NNNN.tsv), each exposure line naming its file. classify_exposures checks the recorded genes against the
                      plan's ordered valid genes (a difference refuses: the frozen rule does not describe the actual call).
    burden_input_summary
                      exposures grouped by their COMPLETE effective burden-input identity, never by a tolerance: biological support
                      (ordered gene IDs + exact values + preprocessing) and numerical input (exact ordered values + estimator +
                      environment). Per group: exposure and valid-gene counts, the exact burden estimate, outcome and failure counts,
                      agreement across identical inputs (a disagreement is a determinism finding to investigate, never averaged) and
                      the genes that lose scoring coverage -- one failure CAUSE kept apart from its downstream CONSEQUENCES. Exposures
                      sharing a group are not independent replications.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
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

__all__ = ["EXPOSURE_RECORDER_VERSION", "PARTIAL_RANKING_DEFINITION", "ExposureStatus", "ExposureEvent", "BurdenInput",
           "read_exposure_trace", "exposure_trace_sha256", "classify_exposures", "PairPlan", "planned_universe", "coverage_report",
           "score_coverage", "exposure_records", "partial_ranking", "endpoint_release", "release_policy", "render_release_policy",
           "release_policy_sha256", "BURDEN_PREPROCESSING", "BURDEN_ESTIMATOR", "burden_support_sha256", "burden_numeric_input_sha256",
           "burden_input_summary", "COVERAGE_REASONS", "PRIMARY_COMPLETE", "PRIMARY_WITHHELD", "PRIMARY_REFUSED", "STAGES"]

EXPOSURE_RECORDER_VERSION = "gvc.dandelion-exposure-recorder/2"     # 2: the effective burden input at the pi0 guard (ruling 2026-10-09)
PARTIAL_RANKING_DEFINITION = ("Ranking conditional on exposures for which DANDELION produced scores; exploratory and not a substitute "
                              "for the complete-exposure primary endpoint.")
PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED = "complete", "withheld", "refused"
STAGES = ("feasibility", "confirmatory")
_KEYS = {"exposure_id", "outcome", "last_guard_reached", "n_trans", "n_valid", "pi0a", "pi0b", "wg1", "wg2", "wg3", "wg_sum",
         "burden_input", "observation_kind", "recorder_version"}
_BURDEN_ID = re.compile(r"burden-(?P<n>\d{4,})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: What turns the planned burden p-values into the estimator's input -- measured in the installed DANDELION 0.1.0
#: run_dandelion_for_exposure body (positions 4-15) on 2026-10-09. The environment identity (DANDELION version and DESCRIPTION digest)
#: binds the implementation; these strings name the steps so an identity states what it identifies.
BURDEN_PREPROCESSING = ("dandelion-0.1.0:run_dandelion_for_exposure: p_b = p.wes.new[gene.trans]; keep genes with non-missing trans AND "
                        "burden p-values, in gene.trans order; p_b = clamp_p(p_b) (p <= 0 -> .Machine$double.xmin, p >= 1 -> 1 - 1e-15); "
                        "names(p_b) = gene.trans")
BURDEN_ESTIMATOR = "dandelion-0.1.0:pi0b = 1 - nonnullPropEst(qnorm(p_b, lower.tail = FALSE), u = 0, sigma = 1)"
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
class BurdenInput:
    """The EFFECTIVE burden input of an actual call, as the recorder wrote it: the ordered gene identifiers and the exact post-clamp
    burden p-values at the pi0 guard. `input_id` is the recorder's file stem (burden-NNNN); `sha256` is the digest of that file's exact
    bytes. One file per DISTINCT input: exposures sharing it name the same file."""

    input_id: str
    sha256: str
    genes: tuple
    values: tuple       # Fractions: exact binary64 values in (0, 1)


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
    burden_input: BurdenInput | None     # present exactly when the pi0 guard was reached


def _read_burden_file(path: Path) -> BurdenInput:
    """One burden-NNNN.tsv: LF-terminated ASCII lines "gene<TAB>value", value R's "%a" text of an exact binary64 in (0, 1) (clamp_p maps
    every burden p-value into [double.xmin, 1 - 1e-15]); at least one line; unique, non-empty gene identifiers."""
    raw = path.read_bytes()
    _require(raw != b"" and b"\r" not in raw and raw.endswith(b"\n"), "exposure_trace_burden_file", path.name)
    try:
        lines = raw.decode("ascii").split("\n")[:-1]
    except UnicodeDecodeError:
        raise InferenceError("exposure_trace_encoding", path.name) from None
    genes, values = [], []
    for n, line in enumerate(lines, 1):
        fields = line.split("\t")
        _require(len(fields) == 2 and fields[0] != "", "exposure_trace_burden_file", "{} line {}".format(path.name, n))
        value = _value(fields[1])
        _require(type(value) is Fraction and 0 < value < 1, "exposure_trace_burden_value", "{} line {}".format(path.name, n))
        genes.append(fields[0])
        values.append(value)
    _require(len(set(genes)) == len(genes), "exposure_trace_burden_file", path.name + ": duplicate gene")
    return BurdenInput(path.stem, hashlib.sha256(raw).hexdigest(), tuple(genes), tuple(values))


def read_exposure_trace(directory) -> tuple:
    """Strictly parse a recorder directory: <dir>/exposures.jsonl (the recorder creates it empty at start) and the burden-NNNN.tsv files
    it references. Exact keys, one event per exposure, every recorded quantity consistent with the guard the call reached (a structural
    outcome carries no estimate and no burden input; an estimation failure carries its invalid estimate; scores need valid ones). Burden
    files: numbered 1, 2, ... in order of first reference, each referenced at least once, no two with the same bytes (one file per
    DISTINCT input), and as many genes as the referencing call's usable pairs. Zero events is a valid trace (no exposure entered
    run_dandelion_for_exposure)."""
    directory = Path(directory)
    path = directory / "exposures.jsonl"
    _require(path.is_file(), "exposure_trace_missing", str(directory))
    names = sorted(p.name for p in directory.iterdir())
    burden_files = {n[:-4] for n in names if n.endswith(".tsv") and _BURDEN_ID.fullmatch(n[:-4])}
    stray = [n for n in names if n != "exposures.jsonl" and not (n.endswith(".tsv") and n[:-4] in burden_files)]
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
    events, seen, burden, order = [], set(), {}, []
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
        reference = doc["burden_input"]
        _require(reference is None or (type(reference) is str and _BURDEN_ID.fullmatch(reference) is not None), "exposure_trace_value",
                 "burden_input line {}".format(n))
        _require((reference is not None) == (hits >= 3), "exposure_trace_inconsistent", doc["exposure_id"])
        if reference is not None and reference not in burden:
            # numbered in order of FIRST reference, exactly as the recorder assigns them; the file must exist
            _require(reference == "burden-{:04d}".format(len(order) + 1), "exposure_trace_burden_sequence", reference)
            _require(reference in burden_files, "exposure_trace_burden_missing", reference)
            burden[reference] = _read_burden_file(directory / (reference + ".tsv"))
            order.append(reference)
        e = ExposureEvent(doc["exposure_id"], outcome, hits, doc["n_trans"], doc["n_valid"],
                          *(_value(doc[k]) for k in ("pi0a", "pi0b", "wg1", "wg2", "wg3", "wg_sum")),
                          None if reference is None else burden[reference])
        _require(e.burden_input is None or len(e.burden_input.genes) == e.n_valid, "exposure_trace_inconsistent", e.exposure_id)
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
    _require(set(order) == burden_files, "exposure_trace_burden_orphan", repr(sorted(burden_files - set(order))[:5]))
    digests = [b.sha256 for b in burden.values()]
    _require(len(set(digests)) == len(digests), "exposure_trace_burden_duplicate", "one file per DISTINCT input")
    return tuple(events)


def exposure_trace_sha256(directory) -> str:
    """The identity of a recorder directory AS A WHOLE: SHA-256 over every file's name and exact-bytes SHA-256, sorted by name. Call it
    after read_exposure_trace has admitted the directory (that refuses any file the recorder did not write)."""
    directory = Path(directory)
    lines = []
    for p in sorted(directory.iterdir(), key=lambda q: q.name):
        _require(p.is_file() and not p.is_symlink(), "exposure_trace_unexpected_files", p.name)
        lines.append("{}\t{}\n".format(p.name, hashlib.sha256(p.read_bytes()).hexdigest()))
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def classify_exposures(exposures, predicted_exclusions: dict, predicted_usable: dict, events, *, predicted_support: dict) -> dict:
    """One status per PLANNED exposure (ruling: require_one_status_per_planned_exposure).

    predicted_exclusions: {exposure: structural reason}, predicted_usable: {exposure: (n_trans, n_valid)} for every annotated
    exposure, and predicted_support: {exposure: ordered valid genes} for every ELIGIBLE exposure (annotated, not excluded) -- all derived
    from the inputs BEFORE execution. The observed call must agree with them: an exposure predicted eligible that the method reports
    structural (or the reverse), a usable-pair count that differs, or a recorded burden input whose ordered genes differ from the
    predicted valid genes (ruling 2026-10-09: the effective input of the ACTUAL call) means the frozen rule does not describe the
    implementation -- refused, never reconciled after the fact."""
    exposures = tuple(exposures)
    _require(bool(exposures) and len(set(exposures)) == len(exposures) and all(type(x) is str and x for x in exposures), "exposure_plan")
    _require(set(predicted_exclusions) <= set(exposures) and set(predicted_exclusions.values()) <= _PREDICTABLE, "eligibility_rule_unknown",
             repr(sorted(set(predicted_exclusions.values()) - _PREDICTABLE)[:3]))
    annotated = {x for x in exposures if predicted_exclusions.get(x) != "exposure_not_annotated"}
    _require(set(predicted_usable) == annotated, "usable_pair_prediction_scope", "counts are predicted for exactly the annotated exposures")
    eligible = {x for x in exposures if x not in predicted_exclusions}
    _require(type(predicted_support) is dict and set(predicted_support) == eligible, "support_prediction_scope",
             "ordered valid genes are predicted for exactly the eligible exposures")
    for x, genes in predicted_support.items():
        _require(type(genes) is tuple and len(genes) == predicted_usable[x][1] and len(set(genes)) == len(genes)
                 and all(type(g) is str and g for g in genes), "support_prediction_scope", x)
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
            if event.burden_input is not None and x in predicted_support:
                # equal counts are not equal genes: the ACTUAL call's ordered support must be the predicted one (ruling 2026-10-09). A
                # predicted-structural exposure that reached the estimator is refused below as an eligibility-rule mismatch.
                _require(event.burden_input.genes == predicted_support[x], "burden_support_mismatch", x)
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


def _canonical_bytes(doc) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class PairPlan:
    """The plan FROZEN BEFORE EXECUTION (ruling 2026-10-09: completeness is derived from ADMITTED exposure and pair evidence; a run intent
    binds this plan's digest, so eligibility cannot move once outcomes exist).

        exposures   the planned exposures, in their planned order (DANDELION's gene1.list)
        genes       the planned gene rows (the rows of mat.p), in order
        scored      frozenset of (gene, exposure) pairs that MUST produce a score; every other grid pair is structurally excluded
        exclusions  ((exposure, structural reason), ...) sorted by exposure
        usable      ((exposure, n_trans, n_valid), ...) for every ANNOTATED exposure, sorted
        support     ((exposure, ordered valid genes), ...) for every ELIGIBLE exposure (not excluded), sorted by exposure

    The invariants tie them together: the reasons agree with the counts, an eligible exposure's planned pairs are exactly its support,
    and an excluded exposure plans none. Built from the inputs by the method-specific planner (method_trace.plan_for for the fixture
    inputs); this type only states and checks the plan."""

    exposures: tuple
    genes: tuple
    scored: frozenset
    exclusions: tuple
    usable: tuple
    support: tuple

    def __post_init__(self) -> None:
        for name, values in (("exposures", self.exposures), ("genes", self.genes)):
            _require(type(values) is tuple and bool(values) and all(type(v) is str and v for v in values) and len(set(values)) == len(values),
                     "pair_plan_" + name)
        _require(type(self.scored) is frozenset and all(type(c) is tuple and len(c) == 2 for c in self.scored), "pair_plan_scored")
        _require({g for g, _ in self.scored} <= set(self.genes) and {x for _, x in self.scored} <= set(self.exposures), "pair_plan_scored")
        _require(type(self.exclusions) is tuple and all(type(r) is tuple and len(r) == 2 for r in self.exclusions)
                 and [x for x, _ in self.exclusions] == sorted({x for x, _ in self.exclusions}), "pair_plan_exclusions")
        excluded = dict(self.exclusions)
        _require(set(excluded) <= set(self.exposures) and set(excluded.values()) <= _PREDICTABLE, "pair_plan_exclusions")
        _require(type(self.usable) is tuple and all(type(r) is tuple and len(r) == 3 for r in self.usable)
                 and [x for x, _, _ in self.usable] == sorted({x for x, _, _ in self.usable}), "pair_plan_usable")
        usable = {x: (t, v) for x, t, v in self.usable}
        annotated = {x for x in self.exposures if excluded.get(x) != "exposure_not_annotated"}
        _require(set(usable) == annotated, "pair_plan_usable", "counts for exactly the annotated exposures")
        for x, (t, v) in usable.items():
            _require(type(t) is int and type(v) is int and 0 <= v <= t, "pair_plan_usable", x)
            reason = excluded.get(x)
            _require((reason == "no_trans_genes") == (t == 0) and (reason == "fewer_than_2_valid_genes") == (t > 0 and v < 2),
                     "pair_plan_usable", "{}: the structural reason disagrees with the counts".format(x))
        _require(type(self.support) is tuple and all(type(r) is tuple and len(r) == 2 for r in self.support)
                 and [x for x, _ in self.support] == sorted({x for x, _ in self.support}), "pair_plan_support")
        support = dict(self.support)
        _require(set(support) == {x for x in self.exposures if x not in excluded}, "pair_plan_support", "for exactly the eligible exposures")
        for x in self.exposures:
            planned = {g for g, y in self.scored if y == x}
            genes = support.get(x, ())
            _require(type(genes) is tuple and len(set(genes)) == len(genes) and len(genes) == (usable[x][1] if x in support else 0)
                     and set(genes) == planned, "pair_plan_support", x)

    def as_dict(self) -> dict:
        """{(gene, exposure): True must score / False structurally excluded} over the whole planned grid."""
        return {(g, x): (g, x) in self.scored for x in self.exposures for g in self.genes}

    def exclusions_dict(self) -> dict:
        return dict(self.exclusions)

    def usable_dict(self) -> dict:
        return {x: (t, v) for x, t, v in self.usable}

    def support_dict(self) -> dict:
        return dict(self.support)

    def render(self) -> bytes:
        return _canonical_bytes({"schema": "gvc.dandelion-pair-plan", "schema_version": 1, "exposures": list(self.exposures),
                                 "genes": list(self.genes), "scored": sorted([g, x] for g, x in self.scored),
                                 "exclusions": dict(self.exclusions), "usable": {x: [t, v] for x, t, v in self.usable},
                                 "support": {x: list(g) for x, g in self.support}})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.render()).hexdigest()


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


def _eligible(statuses: dict) -> list:
    """The planned ELIGIBLE exposures: every planned exposure not structurally ineligible (ONE definition, used by coverage_report and
    score_coverage)."""
    return sorted(x for x, s in statuses.items() if s["status"] is not ExposureStatus.STRUCTURALLY_INELIGIBLE)


def coverage_report(plan: dict, statuses: dict) -> dict:
    """Exposure completion and the per-gene coverage distribution over the PLANNED universe, and the primary status."""
    _check_plan(plan)
    _check_statuses(plan, statuses)
    eligible = _eligible(statuses)
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


#: score_coverage's refusal reasons, in the order they are reported
COVERAGE_REASONS = ("no_eligible_exposures", "execution_or_classification_failure", "score_matrix_cells", "unexpected_scored_pair",
                    "scores_disagree_with_exposure_outcomes")


def score_coverage(plan: dict, statuses: dict, scores: dict) -> tuple:
    """The coverage rule of owner ruling 2026-10-09 over ADMITTED evidence -> (primary status, reasons).

    plan: {(gene, exposure): must score} (PairPlan.as_dict); statuses: classify_exposures' one status per planned exposure (its
    validators stay authoritative); scores: ranking.read_score_matrix of the score artifact, {(gene, exposure): exact score or None}.

        no eligible exposure                                        REFUSED  (all([]) must never read as "complete")
        an unclassified-missing or infrastructure outcome           REFUSED
        the score grid is not exactly the planned grid              REFUSED
        a score for a pair the plan excludes                        REFUSED
        scored pairs != the planned pairs of SCORED exposures       REFUSED  (a "scored" exposure that silently omits a planned score,
                                                                              or a failed exposure that nevertheless has scores)
        otherwise, a recorded mixture-estimation failure            WITHHELD (diagnostics retained)
        otherwise                                                   COMPLETE (eligible for the later release checks -- not a release)

    Every applicable refusal reason is reported, in COVERAGE_REASONS order. Malformed inputs raise InferenceError."""
    _check_plan(plan)
    _check_statuses(plan, statuses)
    _require(set(statuses) == {x for _, x in plan}, "exposure_status_scope", "one status for exactly the planned exposures")
    _require(type(scores) is dict and all(type(c) is tuple and len(c) == 2 for c in scores)
             and all(v is None or type(v) is Fraction for v in scores.values()), "score_matrix_type")
    reasons = []
    if not _eligible(statuses):
        reasons.append("no_eligible_exposures")
    if any(s["status"] in _REFUSALS for s in statuses.values()):
        reasons.append("execution_or_classification_failure")
    if set(scores) != set(plan):
        reasons.append("score_matrix_cells")
    scored = {c for c, v in scores.items() if v is not None}
    planned = {c for c, p in plan.items() if p}
    if scored - planned:
        reasons.append("unexpected_scored_pair")
    expected = {c for c in planned if statuses[c[1]]["status"] is ExposureStatus.SCORED}
    if scored & planned != expected:
        reasons.append("scores_disagree_with_exposure_outcomes")
    if reasons:
        return PRIMARY_REFUSED, tuple(reasons)
    if any(s["status"] in _ESTIMATION_FAILURES for s in statuses.values()):
        return PRIMARY_WITHHELD, ("incomplete_eligible_exposures",)
    return PRIMARY_COMPLETE, ()


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


#: THE RELEASE POLICY (rulings 2026-10-08g and 2026-10-09): ONE immutable decision table, read by endpoint_release and identified by
#: the digest of its canonical rendering. A run intent binds that digest, so a change of release semantics is a new policy identity --
#: an explicit amendment -- and an earlier intent can never silently acquire a new meaning. Rows: (primary status, stage) ->
#: (primary ranking, reference recovery and Delta H(20), partial ranking).
_RELEASE_TABLE = (
    ((PRIMARY_COMPLETE, "confirmatory"), ("released", "released", "exploratory")),
    ((PRIMARY_COMPLETE, "feasibility"), ("computed_not_evaluated", "withheld_feasibility_stage", "exploratory")),
    ((PRIMARY_WITHHELD, "confirmatory"), ("withheld_incomplete_exposures", "withheld_incomplete_exposures", "exploratory")),
    ((PRIMARY_WITHHELD, "feasibility"), ("withheld_incomplete_exposures", "withheld_incomplete_exposures", "exploratory")),
    ((PRIMARY_REFUSED, "confirmatory"), ("refused", "refused", "refused")),
    ((PRIMARY_REFUSED, "feasibility"), ("refused", "refused", "refused")),
)
_RELEASE_RULES = (
    "the stage is taken ONLY from the admitted pre-execution run intent or evaluation intent -- never a command-line option, a report "
    "setting or a default; a missing or unknown stage refuses",
    "a feasibility run stays feasibility whatever its completion; there is no automatic promotion",
    "a confirmatory evaluation is a NEW evaluation intent; unchanged admitted scores may be reused, nothing is relabelled, and the "
    "feasibility history it follows is disclosed",
    "reference evidence is opened only after admission permits evaluation, and only for a released reference recovery",
    "a performance-based choice among exposure-failure or release policies is never permitted; a revised policy is a new identity",
    "evaluated means the specified calculation occurred; it is not a scientific validation",
)


def release_policy() -> dict:
    """The release policy as a document (the table above, with its rules)."""
    return {"schema": "gvc.endpoint-release-policy", "schema_version": 1, "stages": list(STAGES),
            "primary_statuses": [PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED],
            "decisions": [{"primary_status": p, "stage": s, "primary_ranking": r, "reference_recovery": e, "delta_h20": e, "partial_ranking": q}
                          for (p, s), (r, e, q) in _RELEASE_TABLE],
            "policy_selection": "not_permitted", "rules": list(_RELEASE_RULES)}


def render_release_policy() -> bytes:
    return _canonical_bytes(release_policy())


def release_policy_sha256() -> str:
    """The release-policy identity a run intent binds (canonical JSON, sorted keys, no whitespace, LF)."""
    return hashlib.sha256(render_release_policy()).hexdigest()


def endpoint_release(primary_status: str, stage: str) -> dict:
    """What may be released (rulings 2026-10-08g, 2026-10-09), read from the ONE release table. FEASIBILITY: reference recovery and
    Delta H(20) are withheld whatever the completion (benchmark diagnostics stay development information); the complete ranking may be
    computed but is not evaluated. CONFIRMATORY: the primary ranking, reference recovery and Delta H(20) are released only when every
    eligible exposure is scored. Never: a performance-based choice among exposure-failure policies -- a revised policy is a new contract
    version. Its PRODUCTION caller (inference/evaluation_boundary.py) takes the stage from the admitted intent only."""
    _require(primary_status in (PRIMARY_COMPLETE, PRIMARY_WITHHELD, PRIMARY_REFUSED), "primary_status")
    _require(stage in STAGES, "stage")
    (row,) = [decision for key, decision in _RELEASE_TABLE if key == (primary_status, stage)]
    primary, evaluation, partial = row
    return {"stage": stage, "primary_status": primary_status, "primary_ranking": primary, "reference_recovery": evaluation,
            "delta_h20": evaluation, "partial_ranking": partial, "policy_selection": "not_permitted",
            "release_policy_sha256": release_policy_sha256()}


# ---------------------------------------------------------------------------------------------------- shared burden inputs (ruling 2026-10-09)
def _hex_list(values) -> list:
    return [_render(v) for v in values]


def burden_support_sha256(burden: BurdenInput) -> str:
    """BIOLOGICAL-SUPPORT identity: the ordered gene identifiers, their exact effective values and the preprocessing that produced them."""
    _require(type(burden) is BurdenInput, "burden_input_type")
    return hashlib.sha256(_canonical_bytes({"schema": "gvc.burden-support", "schema_version": 1, "preprocessing": BURDEN_PREPROCESSING,
                                            "genes": list(burden.genes), "values": _hex_list(burden.values)})).hexdigest()


def burden_numeric_input_sha256(burden: BurdenInput, environment_sha256: str) -> str:
    """NUMERICAL-INPUT identity: the exact ordered effective values, the estimator and the environment -- WITHOUT gene identifiers. Two
    exposures can share it while their biological support differs; the estimate is then the same numerical problem, not the same
    biology."""
    _require(type(burden) is BurdenInput, "burden_input_type")
    _require(type(environment_sha256) is str and _SHA256.fullmatch(environment_sha256) is not None, "environment_digest")
    return hashlib.sha256(_canonical_bytes({"schema": "gvc.burden-numeric-input", "schema_version": 1, "estimator": BURDEN_ESTIMATOR,
                                            "environment_sha256": environment_sha256, "values": _hex_list(burden.values)})).hexdigest()


def _invalid_estimate(value) -> bool:
    return value == "NA" or value < 0


def burden_input_summary(plan: dict, statuses: dict, environment_sha256: str) -> dict:
    """Which exposures share the SAME effective burden-side estimation problem, from the ACTUAL-CALL trace (ruling 2026-10-09 sections
    3-4) -- a derived feasibility table, never a forecast and never an input to eligibility, methods or release.

    Groups are the COMPLETE identity (biological support, numerical input); no tolerance ever merges two inputs. Per group: the exposures,
    the valid-gene count, the exact burden estimate (pi0b as estimated), agreement across the group's actual calls (identical inputs in
    one environment must give bit-identical estimates -- a disagreement is listed under "investigate", never averaged), whether the
    burden side failed the pi0 guard, the outcome counts, and the genes whose planned scores the group's failed exposures lose. The
    numerical view counts how many support groups share each numerical input. Exposures that never reached the burden estimate
    (structural, or no observed call) are listed with their status."""
    _check_plan(plan)
    _check_statuses(plan, statuses)
    _require(type(environment_sha256) is str and _SHA256.fullmatch(environment_sha256) is not None, "environment_digest")
    groups, unreached = {}, {}
    for x, s in sorted(statuses.items()):
        event = s["event"]
        burden = None if event is None else event.burden_input
        if burden is None:
            unreached[x] = s["status"].value
            continue
        key = (burden_support_sha256(burden), burden_numeric_input_sha256(burden, environment_sha256))
        groups.setdefault(key, []).append((x, s, burden))
    rows, numeric, investigate = [], {}, []
    lost_any, lost_burden, burden_failed = set(), set(), []
    for (support, number), members in sorted(groups.items()):
        estimates = {s["event"].pi0b for _, s, _ in members}
        if len(estimates) == 1:
            (estimate,) = estimates
            agreement = "identical"
            side = "estimate_invalid" if _invalid_estimate(estimate) else "estimate_valid"
        else:
            agreement, side = "disagreement_investigate", "undetermined_disagreement"
            investigate.append(support)
        failed = sorted(x for x, s, _ in members if s["status"] is not ExposureStatus.SCORED)
        lost = sorted({g for (g, y), p in plan.items() if p and y in failed})
        mixture = [(x, s) for x, s, _ in members if s["status"] is ExposureStatus.MIXTURE_ESTIMATE_INVALID]
        burden_side = sorted(x for x, s in mixture if _invalid_estimate(s["event"].pi0b))
        outcomes = {}
        for _, s, _ in members:
            outcomes[s["status"].value] = outcomes.get(s["status"].value, 0) + 1
        lost_any |= set(lost)
        lost_burden |= {g for (g, y), p in plan.items() if p and y in burden_side}
        burden_failed += burden_side
        rows.append({"support_sha256": support, "numeric_input_sha256": number, "burden_input_files": sorted({b.input_id for _, _, b in members}),
                     "valid_genes": len(members[0][2].genes), "exposures": [x for x, _, _ in members], "exposure_count": len(members),
                     "burden_estimate": [_render(e) for e in sorted(estimates, key=lambda v: (v == "NA", v if v != "NA" else 0))],
                     "agreement": agreement, "burden_side": side, "outcomes": dict(sorted(outcomes.items())),
                     "failure_counts": {"burden_side_invalid": len(burden_side),
                                        "trans_side_only_invalid": len(mixture) - len(burden_side),
                                        "nonpositive_weight_sum": sum(s["status"] is ExposureStatus.NONPOSITIVE_WEIGHT_SUM for _, s, _ in members),
                                        "other_unscored": sum(s["status"] in _REFUSALS for _, s, _ in members)},
                     "failed_exposures": failed, "genes_losing_coverage": lost, "genes_losing_coverage_count": len(lost)})
        view = numeric.setdefault(number, {"numeric_input_sha256": number, "support_groups": 0, "exposures": 0})
        view["support_groups"] += 1
        view["exposures"] += len(members)
    return {"definition": ("exposures grouped by their COMPLETE effective burden-input identity from the actual-call trace: biological "
                           "support (ordered gene IDs + exact values + preprocessing) and numerical input (exact ordered values + estimator "
                           "+ environment); no tolerance"),
            "preprocessing": BURDEN_PREPROCESSING, "estimator": BURDEN_ESTIMATOR, "environment_sha256": environment_sha256,
            "groups": rows, "numerical_inputs": sorted(numeric.values(), key=lambda v: v["numeric_input_sha256"]),
            "exposures_not_reaching_burden_estimation": unreached,
            "totals": {"support_groups": len(rows), "exposures_reaching_burden_estimation": sum(r["exposure_count"] for r in rows),
                       "burden_side_failure_groups": sum(r["burden_side"] == "estimate_invalid" for r in rows),
                       "exposures_with_burden_side_failure": len(burden_failed),
                       "genes_losing_coverage_through_burden_side_failure": len(lost_burden),
                       "exposures_unscored_after_reaching_the_estimate": sum(len(r["failed_exposures"]) for r in rows),
                       "genes_losing_coverage_any_cause_among_these": len(lost_any)},
            "investigate": investigate,
            "independence": "exposures in one group share one burden-side estimation problem: they are not independent replications"}
