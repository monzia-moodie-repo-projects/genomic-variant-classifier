"""
Clinical Evaluator
==================
Comprehensive evaluation framework for variant pathogenicity classifiers.

Goes beyond basic AUROC to measure what matters in clinical genomics:
  - Calibration quality (are predicted probabilities trustworthy?)
  - Performance at clinically-relevant operating points (90%/95% sensitivity,
    ≥80% PPV for confident reporting to clinicians)
  - Per-consequence-class breakdown (LoF vs. missense vs. synonymous)
  - Bootstrap confidence intervals on all primary metrics
  - Gene-level error analysis (which genes drive the most errors?)

CHANGES FROM PHASE 1:
  - Was a bare string literal (Bug 3 fixed — now a real .py file).
  - _gene_error_analysis used itertuples() and then **row._asdict().
    This is fragile: pandas renames columns whose names conflict with
    Python keywords or NamedTuple internals (e.g., "index", "_fields").
    Fixed by using DataFrame.to_dict(orient="records") which returns
    plain dicts that unpack cleanly with ** (Issue S).
  - Module-level logging.basicConfig removed (Issue L).
  - from __future__ import annotations added (Issue N).

Usage:
    from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator

    evaluator = ClinicalEvaluator()
    report = evaluator.evaluate(
        y_true=y_test,
        y_proba=ensemble_proba,
        meta=meta_test,
        model_name="EnsembleStacker",
    )
    evaluator.print_report(report)
    evaluator.save_report(report, path="models/v1/eval_report.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

# Both of these are scikit-learn FREE and are therefore safe at module level.
# `capabilities` imports only dataclasses, enum and typing; `cluster_resolution`
# imports numpy, pandas and capabilities. Verified 2026-07-26 by importing each
# in a subprocess with scikit-learn absent. The metric KERNEL is a different
# matter -- metrics.py imports scikit-learn at module level, so the bootstrap
# dispatcher is imported lazily inside evaluate(), never here. See
# `test_evaluator_phase5.py::test_module_imports_without_sklearn`.
from genomic_variant_classifier.evaluation.capabilities import BootstrapUnit, MetricResult, MetricStatus
from genomic_variant_classifier.evaluation.cluster_resolution import (
    ClusterResolution,
    resolve_gene_clusters,
)
from genomic_variant_classifier.evaluation.input_validation import (
    validate_probabilities, validate_ranking_scores, validate_reference_labels)
from genomic_variant_classifier.evaluation.legacy_projection import project_legacy_fields
from genomic_variant_classifier.evaluation.population import EvaluationPopulation
from genomic_variant_classifier.evaluation.registry import (
    MetricContext, evaluate_registered)
from genomic_variant_classifier.evaluation.serialization import dump_strict_json

# Schema version of the persisted evaluation report.
#
#   1  auroc_ci_lo / auroc_ci_hi were always finite floats, with no record of
#      which resampling design produced them. Historical artifacts under
#      outputs/run10/, outputs/run16/ and the ablation arms are version 1. A
#      version-1 interval MUST NOT be read as certified: it predates the
#      distinction, and the row-level bootstrap that produced it is
#      anti-conservative by a measured factor of 2.935 (ratchet entry 2055).
#   2  endpoints are nullable, and every interval carries its own status,
#      resampling unit, stratification, cluster provenance, replicate
#      accounting and finding.
#   3  the report carries `metric_results`, the typed registry results, and each
#      serialised result carries its own `result_kind`. A version-3 report MUST
#      have a non-empty mapping; a version-2 report MUST have an empty one.
#      Version-2 artifacts are NEVER given synthesised typed results: status,
#      reason, applicability, population fingerprint, threshold provenance and
#      result kind were all absent when they were written, and reconstructing
#      them from bare floats would fabricate provenance that never existed.
EVALUATION_REPORT_SCHEMA_VERSION = 2

# The version this codebase WRITES once the report is a projection of the typed
# results. Introduced in commit 3a as a serialisation capability; `evaluate()`
# does not emit it until commit 3b makes the report a pure projection, so that
# schema introduction and computational retirement remain independently
# falsifiable.
EVALUATION_REPORT_SCHEMA_VERSION_TYPED = 3

# Versions this codebase can READ.
SUPPORTED_REPORT_SCHEMA_VERSIONS = (1, 2, 3)

# PHASE5: lazy sklearn loader + first-class F1
# sklearn is imported on first use (not at module import) so this module imports cleanly in
# minimal environments without scikit-learn. _ensure_sklearn() binds the seven symbols into module
# globals once; the two methods that use them call it at entry, so all call sites below stay
# byte-identical (bare names resolve as module globals after the first call).
_SKLEARN_LOADED = False


def _ensure_sklearn() -> None:
    """Import the sklearn symbols once and bind them into module globals (idempotent)."""
    global _SKLEARN_LOADED
    if _SKLEARN_LOADED:
        return
    from sklearn.calibration import calibration_curve as _calibration_curve
    from sklearn.metrics import (
        average_precision_score as _average_precision_score,
        f1_score as _f1_score,
        matthews_corrcoef as _matthews_corrcoef,
        roc_auc_score as _roc_auc_score,
        roc_curve as _roc_curve,
        precision_recall_curve as _precision_recall_curve,
    )
    globals().update({
        "calibration_curve": _calibration_curve,
        "average_precision_score": _average_precision_score,
        "f1_score": _f1_score,
        "matthews_corrcoef": _matthews_corrcoef,
        "roc_auc_score": _roc_auc_score,
        "roc_curve": _roc_curve,
        "precision_recall_curve": _precision_recall_curve,
    })
    _SKLEARN_LOADED = True

logger = logging.getLogger(__name__)


def derive_seed(base_seed: int, namespace: str) -> int:
    """A reproducible seed for one named quantity, derived from a base seed.

    WHY THIS EXISTS. Until 2026-07-26 this class held ONE mutable generator,
    'self.rng', and both bootstrap calls drew from it in sequence. Three
    consequences, all measured:

      * the interval for the area under the precision-recall curve depended on
        the interval for the area under the receiver operating characteristic
        curve having been computed first, because it inherited the advanced
        stream;
      * calling evaluate() twice on one evaluator returned DIFFERENT intervals
        for identical inputs;
      * adding, removing or reordering any bootstrap anywhere in the method
        silently changed every interval after it.

    A derived seed removes all three. Each metric addresses its own independent
    stream, and that stream is a pure function of (base seed, name), so it is
    stable across runs, across call order, and across processes.

    hashlib rather than the builtin hash(): PYTHONHASHSEED randomises string
    hashing per process, so builtin hash() would make the "reproducible" seed
    vary between interpreter invocations -- the exact defect being removed.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string naming the quantity")
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{namespace}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2 ** 32)


def format_ci(
    lower: Optional[float],
    upper: Optional[float],
    *,
    status: MetricStatus,
    finding: Optional[str] = None,
) -> str:
    """Render one interval for human reading, in ONE place.

    Every rendering site calls this. The alternative -- an 'if value is None'
    at each of the three call sites -- is how three sites come to disagree.

    An unavailable interval renders as words, never as '[nan, nan]'. A
    numeric-looking rendering of an absent result is the failure mode this
    replaces: it looks like arithmetic went wrong rather than like no estimate
    was made, and it is easy to skim past in a printed report.
    """
    if status is MetricStatus.OK:
        if lower is None or upper is None:
            # Unreachable through EvaluationReport, whose __post_init__ refuses
            # this combination. Guarded anyway because this helper is public and
            # a caller may assemble the arguments by hand.
            raise ValueError(
                "format_ci: status is OK but an endpoint is None. An available "
                "interval must carry both endpoints.")
        return f"[{lower:.4f}, {upper:.4f}]"
    if finding:
        return f"unavailable ({finding})"
    return "unavailable"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    """Model performance at a specific probability threshold."""

    threshold:    float
    sensitivity:  float   # TPR / recall
    specificity:  float   # TNR
    ppv:          float   # precision / positive predictive value
    npv:          float   # negative predictive value
    f1:           float
    n_flagged:    int     # total predicted positive (TP + FP)
    n_tp:         int
    n_fp:         int
    n_fn:         int
    n_tn:         int


@dataclass
class ConsequenceBreakdown:
    """Per-consequence-class performance metrics."""

    consequence:  str
    n_total:      int
    n_pathogenic: int
    auroc:        float
    auprc:        float
    prevalence:   float


@dataclass
class GeneErrorAnalysis:
    """Error analysis for a single gene."""

    gene_symbol:        str
    n_variants:         int
    n_false_positives:  int
    n_false_negatives:  int
    total_errors:       int
    error_rate:         float


def _validate_ci_fields(
    metric: str,
    *,
    lower: Optional[float],
    upper: Optional[float],
    status: MetricStatus,
    unit: Optional[BootstrapUnit],
    stratified: Optional[bool],
    cluster_source: Optional[str],
    partition_verified: bool,
    certification_eligible: bool,
    n_requested: int,
    n_valid: int,
    n_degenerate: int,
    finding: Optional[str],
) -> None:
    """Refuse a persisted interval that cannot be true.

    Enforced in __post_init__ rather than at read time, for the reason
    capabilities.py gives for CapabilityEvidence: a check at decision time can
    be skipped by a caller who forgets to consult it, while an invariant in the
    constructor has no path around it. An impossible artifact -- an available
    interval with no endpoints, a null interval that still claims certification,
    a variant-level interval marked certifiable -- cannot be built, so no
    downstream reader has to defend against one.
    """
    if not isinstance(status, MetricStatus):
        raise TypeError(
            f"{metric}: ci status must be a MetricStatus, got "
            f"{type(status).__name__}. A bare string cannot be compared against "
            "the enum and would silently miss every branch below.")

    if status is MetricStatus.OK:
        if lower is None or upper is None:
            raise ValueError(
                f"{metric}: an available interval requires both endpoints; got "
                f"lower={lower!r}, upper={upper!r}.")
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(
                f"{metric}: available interval endpoints must be finite; got "
                f"[{lower}, {upper}]. A non-finite endpoint is an absent "
                "estimate wearing a number's clothes.")
        if lower > upper:
            raise ValueError(
                f"{metric}: interval lower endpoint {lower} exceeds upper {upper}.")
        if unit is None:
            raise ValueError(
                f"{metric}: an available interval requires a resampling unit. An "
                "interval whose inferential design is unrecorded cannot be "
                "interpreted, and would be indistinguishable from a certified one.")
        if n_requested <= 0:
            raise ValueError(
                f"{metric}: an available interval requires n_requested > 0, got "
                f"{n_requested}.")
        if n_valid <= 0:
            raise ValueError(
                f"{metric}: an available interval requires n_valid > 0, got {n_valid}.")
        if n_valid + n_degenerate != n_requested:
            raise ValueError(
                f"{metric}: replicate accounting does not balance: "
                f"{n_valid} valid + {n_degenerate} degenerate != "
                f"{n_requested} requested.")
    else:
        if lower is not None or upper is not None:
            raise ValueError(
                f"{metric}: status {status.value} must not carry endpoints; got "
                f"lower={lower!r}, upper={upper!r}. Endpoints beside a non-OK "
                "status are the ambiguity this schema exists to remove.")
        if certification_eligible:
            raise ValueError(
                f"{metric}: status {status.value} cannot be certification "
                "eligible. An interval that was not produced cannot be admissible.")
        if not finding:
            raise ValueError(
                f"{metric}: status {status.value} requires a machine-readable "
                "finding. 'Not available' without a reason forces every reader to "
                "guess whether the identifier was missing, the columns "
                "disagreed, or the replicates were too few.")

    if certification_eligible:
        if status is not MetricStatus.OK:
            raise ValueError(
                f"{metric}: certification requires status OK, got {status.value}.")
        if unit is not BootstrapUnit.GENE:
            raise ValueError(
                f"{metric}: only gene-cluster intervals may be certification "
                f"eligible; got unit "
                f"{unit.value if unit is not None else None}. Row-level "
                "resampling assumes variants are independent, which they are not.")
        if not cluster_source:
            raise ValueError(
                f"{metric}: a certified interval must name the column its gene "
                "clusters came from.")

    if partition_verified and cluster_source not in {"cluster_id", "gene_id+gene_symbol"}:
        raise ValueError(
            f"{metric}: partition_verified is only meaningful when two cluster "
            f"labelings were compared or a canonical column was supplied; got "
            f"cluster_source={cluster_source!r}.")

    if stratified is not None and not isinstance(stratified, bool):
        raise TypeError(f"{metric}: stratified must be a bool or None, got {stratified!r}")


# --------------------------------------------------------------------------- #
# Typed-result serialisation, schema version 3
#
# `result_kind` is written into the ARTIFACT even though it lives on the
# descriptor and not on `MetricResult`. Commit 2b-2 ruled it must never enter
# result metadata -- that would perturb every already-serialised result -- but an
# artifact that cannot say what kind of quantity it recorded is not
# self-describing, and a future registry revision could silently reinterpret it.
#
# So it is written from the descriptor at serialisation time and VERIFIED on
# read. A disagreement between the artifact and the current registry is a
# schema-or-registry version conflict and is raised; it is never resolved by
# overwriting the recorded value with today's opinion, because the artifact is
# the evidence and the registry is the interpreter.
#
# `asdict()` cannot do this: it walks the dataclass and bypasses `to_dict()`
# entirely, so anything added to `MetricResult.to_dict` would never reach a file.
# --------------------------------------------------------------------------- #
# Flat report fields whose declared type is an enum. JSON flattens these to their
# string values, and the report REFUSES a bare string -- correctly: "a bare
# string cannot be compared against the enum and would silently miss every branch
# below." A deserialiser that did not restore them would either crash on every
# round trip or, worse, be "fixed" by relaxing the report's type check, which is
# the guard that stops an interval status from being silently misread.
_ENUM_REPORT_FIELDS = {
    "auroc_ci_status": MetricStatus,
    "auprc_ci_status": MetricStatus,
    "auroc_ci_resampling_unit": BootstrapUnit,
    "auprc_ci_resampling_unit": BootstrapUnit,
}


def _restore_enum_fields(payload: dict) -> dict:
    out = dict(payload)
    for name, enum_type in _ENUM_REPORT_FIELDS.items():
        value = out.get(name)
        if isinstance(value, str):
            try:
                out[name] = enum_type(value)
            except ValueError as exc:
                raise ValueError(
                    f"{name}: {value!r} is not a member of "
                    f"{enum_type.__name__}; the artifact was written by an "
                    "incompatible version and must not be coerced") from exc
    return out


def serialize_metric_results(metric_results: Mapping) -> dict:
    from genomic_variant_classifier.evaluation.registry import by_name

    out = {}
    for name, result in metric_results.items():
        payload = result.to_dict()
        # A NON-FINITE VALUE IS WRITTEN AS null, NOT AS NaN.
        #
        # `dump_strict_json` refuses NaN by design -- "an absent estimate wearing
        # a number's clothes" -- and `MetricResult.from_dict` already documents
        # that it reads a null back as NaN. The READER therefore implements a
        # contract the WRITER does not: every refused result is unpersistable as
        # `to_dict()` emits it. Normalising here completes the round trip for the
        # report without touching `MetricResult.to_dict`, whose five other call
        # sites are Family B representation probes that legitimately produce
        # non-finite results and are outside this commit's subject. The
        # underlying asymmetry is recorded as a carried roadmap item.
        #
        # The STATUS and REASON carry the meaning; the null is only the absence
        # of a number, which is exactly what a refusal is.
        if not math.isfinite(payload.get("value", float("nan"))):
            payload["value"] = None
        try:
            payload["result_kind"] = by_name(name).result_kind.value
        except KeyError as exc:
            raise ValueError(
                f"cannot serialise metric_results[{name!r}]: no descriptor is "
                "registered under that name, so the artifact would record a "
                "quantity nothing can interpret") from exc
        out[name] = payload
    return out


def deserialize_metric_results(payload: Mapping) -> dict:
    from genomic_variant_classifier.evaluation.registry import by_name

    out = {}
    for name, entry in payload.items():
        recorded = entry.get("result_kind")
        if recorded is None:
            raise ValueError(
                f"serialised result {name!r} carries no result_kind; a "
                "version-3 artifact must be self-describing")
        try:
            current = by_name(name).result_kind.value
        except KeyError as exc:
            raise ValueError(
                f"serialised result {name!r} has no descriptor in the current "
                "registry; this artifact was written by a different registry "
                "version and cannot be interpreted here") from exc
        if recorded != current:
            raise ValueError(
                f"result_kind conflict for {name!r}: the artifact records "
                f"{recorded!r} and this registry declares {current!r}. That is a "
                "schema or registry version conflict requiring an explicit "
                "decision; it is NOT resolved by preferring today's registry, "
                "because the artifact is the evidence.")
        out[name] = MetricResult.from_dict(
            {k: v for k, v in entry.items() if k != "result_kind"})
    return out


@dataclass
class EvaluationReport:
    """Full evaluation report for one model.

    SCHEMA VERSION 2 (2026-07-26). Interval endpoints are nullable, and each
    interval carries the design that produced it. See
    EVALUATION_REPORT_SCHEMA_VERSION for what changed and why a version-1
    interval must never be read as certified.
    """

    schema_version: int

    model_name:   str
    n_samples:    int
    n_pathogenic: int
    n_benign:     int
    prevalence:   float

    # Core discriminative metrics
    auroc:       float
    auroc_ci_lo: Optional[float]
    auroc_ci_hi: Optional[float]
    auprc:       float
    auprc_ci_lo: Optional[float]
    auprc_ci_hi: Optional[float]
    mcc:         float
    f1:          float
    brier_score: float

    # Interval provenance, per metric. Deliberately NOT shared: the two metrics
    # can fail differently on the same cohort. On a heavily imbalanced split the
    # area under the precision-recall curve degenerates in resamples that the
    # area under the receiver operating characteristic curve survives, so their
    # valid-replicate counts diverge and one can be available while the other is
    # not. One shared status would have to pick a winner.
    auroc_ci_status:                 MetricStatus
    auroc_ci_resampling_unit:        Optional[BootstrapUnit]
    auroc_ci_stratified:             Optional[bool]
    auroc_ci_cluster_source:         Optional[str]
    auroc_ci_partition_verified:     bool
    auroc_ci_certification_eligible: bool
    auroc_ci_n_requested:            int
    auroc_ci_n_valid:                int
    auroc_ci_n_degenerate:           int
    auroc_ci_finding:                Optional[str]

    auprc_ci_status:                 MetricStatus
    auprc_ci_resampling_unit:        Optional[BootstrapUnit]
    auprc_ci_stratified:             Optional[bool]
    auprc_ci_cluster_source:         Optional[str]
    auprc_ci_partition_verified:     bool
    auprc_ci_certification_eligible: bool
    auprc_ci_n_requested:            int
    auprc_ci_n_valid:                int
    auprc_ci_n_degenerate:           int
    auprc_ci_finding:                Optional[str]

    # Calibration
    calibration_ece: float  # Expected Calibration Error
    calibration_mce: float  # Maximum Calibration Error

    # Clinical operating points
    at_sensitivity_90: Optional[OperatingPoint] = None
    at_sensitivity_95: Optional[OperatingPoint] = None
    at_high_ppv:       Optional[OperatingPoint] = None

    # Breakdowns
    consequence_breakdown: list = field(default_factory=list)
    gene_errors:           list = field(default_factory=list)

    # Curves for downstream plotting
    fpr_curve:               list = field(default_factory=list)
    tpr_curve:               list = field(default_factory=list)
    precision_curve:         list = field(default_factory=list)
    recall_curve:            list = field(default_factory=list)
    calibration_frac_pos:    list = field(default_factory=list)
    calibration_mean_pred:   list = field(default_factory=list)

    # --- the typed registry results, schema version 3 (2026-07-28) ----------
    # The canonical layer. The flat scalar fields above become PROJECTIONS of
    # this mapping in commit 3b; until then they remain independently computed
    # and this mapping is empty, so schema introduction and computational
    # retirement stay independently falsifiable.
    #
    # EMPTY IS MEANINGFUL, NOT MISSING. A version-2 artifact deserialises with an
    # empty mapping and is never given synthesised results: status, reason,
    # applicability, population fingerprint, threshold provenance and result kind
    # were all absent when it was written, and reconstructing them from bare
    # floats would fabricate provenance rather than recover it.
    metric_results: Mapping = field(default_factory=dict)

    def to_serializable(self) -> dict:
        """Dictionary form suitable for strict JSON.

        Uses `asdict` for the flat surface, then REPLACES the typed mapping,
        because `asdict` bypasses `MetricResult.to_dict()` and would omit the
        `result_kind` the artifact must carry.
        """
        payload = asdict(self)
        payload["metric_results"] = serialize_metric_results(self.metric_results)
        return payload

    @classmethod
    def from_serialized(cls, payload: Mapping) -> "EvaluationReport":
        """Read an artifact of any supported version, dispatching on its own
        recorded version rather than on what the reader hopes to find."""
        version = payload.get("schema_version")
        if version is None:
            raise ValueError("serialized report carries no schema_version")
        version = int(version)
        if version < EVALUATION_REPORT_SCHEMA_VERSION_TYPED:
            return cls.from_serialized_v2(payload)
        known = {f.name for f in dataclasses.fields(cls)}
        accepted = {k: v for k, v in payload.items()
                    if k in known and k != "metric_results"}
        accepted = _restore_enum_fields(accepted)
        accepted["schema_version"] = version
        accepted["metric_results"] = deserialize_metric_results(
            payload.get("metric_results") or {})
        return cls(**accepted)

    # ------------------------------------------------------------------ #
    # Schema, typed results, and construction
    # ------------------------------------------------------------------ #
    def _validate_schema_and_typed_results(self) -> None:
        """The version and the typed mapping must agree, in both directions.

        A version-3 report with an empty mapping would claim a typed surface it
        does not have. A version-2 report with a populated one would imply that
        historical artifacts carry provenance they never recorded. Either would
        make the version unreadable as evidence of what a file actually
        contains, which is the only thing a schema version is for.
        """
        if self.schema_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported report schema version {self.schema_version!r}; "
                f"this codebase reads {SUPPORTED_REPORT_SCHEMA_VERSIONS}")
        if not isinstance(self.metric_results, Mapping):
            raise TypeError(
                f"metric_results must be a mapping, got "
                f"{type(self.metric_results).__name__}")
        for name, result in self.metric_results.items():
            if not isinstance(name, str) or not name:
                raise TypeError("metric_results keys must be non-empty strings")
            if not isinstance(result, MetricResult):
                raise TypeError(
                    f"metric_results[{name!r}] must be a MetricResult, got "
                    f"{type(result).__name__}; a bare float carries no status, "
                    "no reason and no population, and admitting one here would "
                    "reintroduce exactly the untyped surface this layer replaces")
        if self.schema_version >= EVALUATION_REPORT_SCHEMA_VERSION_TYPED:
            if not self.metric_results:
                raise ValueError(
                    f"a version-{self.schema_version} report requires a "
                    "non-empty metric_results mapping: the version asserts that "
                    "the typed surface is present")
        elif self.metric_results:
            raise ValueError(
                f"a version-{self.schema_version} report must have an EMPTY "
                "metric_results mapping. Versions 1 and 2 predate the typed "
                "surface, so a populated mapping on one of them is either a "
                "mislabelled artifact or synthesised provenance.")

    @classmethod
    def from_metric_results(cls, *, metric_results: Mapping,
                            **fields) -> "EvaluationReport":
        """Build a version-3 report around the typed results.

        The canonical constructor once the report is a projection. Direct
        dataclass construction remains available because historical
        deserialisation needs it, and because making the fields `init=False`
        would break every existing caller; consistency is enforced in
        `__post_init__` rather than by removing the door.
        """
        if not metric_results:
            raise ValueError(
                "from_metric_results() requires at least one typed result; an "
                "empty mapping means a version-2 report, which is built by "
                "from_serialized_v2() or by direct construction")
        fields.pop("schema_version", None)
        return cls(schema_version=EVALUATION_REPORT_SCHEMA_VERSION_TYPED,
                   metric_results=dict(metric_results), **fields)

    @classmethod
    def from_serialized_v2(cls, payload: Mapping) -> "EvaluationReport":
        """Read a historical version-1 or version-2 artifact.

        The typed mapping is left EMPTY. It is never synthesised from the flat
        scalars, however tempting: an `OK` result manufactured from a bare float
        would assert a population scope, a support count, an applicability
        verdict and a certification eligibility that the artifact never
        recorded. A report that cannot say what it does not know is more useful
        than one that guesses.
        """
        version = payload.get("schema_version")
        if version is None:
            raise ValueError("serialized report carries no schema_version")
        if int(version) >= EVALUATION_REPORT_SCHEMA_VERSION_TYPED:
            raise ValueError(
                f"from_serialized_v2() refuses schema version {version}; use "
                "from_serialized() so the typed results are read rather than "
                "discarded")
        known = {f.name for f in dataclasses.fields(cls)}
        accepted = {k: v for k, v in payload.items()
                    if k in known and k != "metric_results"}
        accepted = _restore_enum_fields(accepted)
        accepted["schema_version"] = int(version)
        accepted["metric_results"] = {}
        return cls(**accepted)

    def __post_init__(self) -> None:
        self._validate_schema_and_typed_results()
        for metric in ("auroc", "auprc"):
            _validate_ci_fields(
                metric,
                lower=getattr(self, f"{metric}_ci_lo"),
                upper=getattr(self, f"{metric}_ci_hi"),
                status=getattr(self, f"{metric}_ci_status"),
                unit=getattr(self, f"{metric}_ci_resampling_unit"),
                stratified=getattr(self, f"{metric}_ci_stratified"),
                cluster_source=getattr(self, f"{metric}_ci_cluster_source"),
                partition_verified=getattr(self, f"{metric}_ci_partition_verified"),
                certification_eligible=getattr(self, f"{metric}_ci_certification_eligible"),
                n_requested=getattr(self, f"{metric}_ci_n_requested"),
                n_valid=getattr(self, f"{metric}_ci_n_valid"),
                n_degenerate=getattr(self, f"{metric}_ci_n_degenerate"),
                finding=getattr(self, f"{metric}_ci_finding"),
            )


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------
class ClinicalEvaluator:
    """
    Computes a full suite of clinical evaluation metrics for a binary
    variant pathogenicity classifier.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        # The BASE seed, not a generator. Until 2026-07-26 this was a single
        # mutable np.random.Generator shared by both bootstrap calls, which made
        # every interval depend on call order and on how many intervals had been
        # drawn before it. Each interval now derives its own stream from this
        # value via derive_seed(); see that function for the three defects.
        self.random_state = int(random_state)

    # ── Public entry point ─────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        meta: Optional[pd.DataFrame] = None,
        model_name: str = "model",
        source_id: Optional[str] = None,
        *,
        scores: Optional[np.ndarray] = None,
    ) -> EvaluationReport:
        """
        Full evaluation pipeline.

        Args:
            y_true:     Binary ground-truth labels (1=pathogenic, 0=benign).
            y_proba:    Predicted probabilities in [0, 1].
            meta:       Canonical variant DataFrame aligned with y_true/y_proba.
                        Required for per-gene and per-consequence analysis.
            model_name: Label for this model in report output.

        Returns:
            EvaluationReport with all metrics populated.
        """
        y = np.asarray(y_true)
        p = np.asarray(y_proba)
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos

        logger.info(
            "Evaluating %s: n=%d, pos=%d (%.1f%%)",
            model_name, n, n_pos, n_pos / n * 100,
        )

        _ensure_sklearn()  # PHASE5: load sklearn symbols into module globals

        # --- INPUT GATES, BEFORE ANY LIBRARY CALL (2026-07-28, CI-t) ----------
        #
        # Five scikit-learn calls consume this pair and disagree about what is
        # invalid. Measured: non-finite probabilities make `roc_curve` raise and
        # `calibration_curve` RETURN a degenerate one-point curve carrying NaN;
        # values outside the unit interval make `calibration_curve` raise and
        # `roc_curve` return happily. Letting the library decide which defect
        # becomes which status means the status is decided by a library that does
        # not agree with itself.
        #
        # REFUSAL IS COMPONENT-LEVEL. A corrupt probability array must not abort
        # the whole report: the typed registry results, the population, the model
        # identity and the prevalence are all still valid, and a report-wide
        # exception would discard scientifically sound information.
        #
        # THE TWO CURVES ARE GATED SEPARATELY because they are different
        # functions with different prerequisites -- `roc_curve` at one line and
        # `precision_recall_curve` at the next -- and on an all-positive cohort
        # they already diverge, one warning while the other returns normally.
        label_check = validate_reference_labels(y, n_expected=n)
        probability_check = validate_probabilities(p, n_expected=n)

        # THE RANKING CHANNEL (2026-07-28, CI-t).
        #
        # `scores` is where a genuine ranking score belongs: a decision-function
        # output, a log-odds, an ensemble margin. It is validated WITHOUT a range
        # restriction, because a score is an ordering and not a magnitude on any
        # particular scale.
        #
        # When it is absent the ranking quantities fall back to `y_proba`, which
        # is then validated AS A PROBABILITY. That asymmetry is the point: an
        # out-of-range array supplied as `y_proba` is invalid model output, while
        # the same array supplied as `scores` is a perfectly good ordering. The
        # caller declares which they meant, and the report stops having to guess.
        if scores is not None:
            candidate = np.asarray(scores, dtype=float)
            ranking_check = validate_ranking_scores(candidate, n_expected=n)
            # REFUSED MEANS NOT FORWARDED. A first version validated the array
            # and passed it to the registry anyway, so a mis-sized `scores`
            # raised a ValueError from the context's own length check -- turning
            # a refusal this gate exists to make graceful back into an exception
            # three layers down.
            ranking_values = candidate if ranking_check.ok else None
        else:
            # NO FALLBACK WHEN THE PROBABILITY IS INVALID.
            #
            # A first version set `ranking_values = p` unconditionally here, and
            # that left the seam open: the registry received the out-of-range
            # array in `y_score`, ranked it happily, and reported auroc 1.0 while
            # the curve computed from the same values was withheld. One input,
            # two layers, opposite verdicts -- which is the incoherence this
            # commit exists to remove, surviving one level down.
            #
            # An out-of-range array supplied as `y_proba` is invalid model
            # output. The caller who meant a ranking score has `scores`.
            ranking_check = probability_check
            ranking_values = p if probability_check.ok else None

        # THE CURVES ARE GATED ON THE PROBABILITY CHANNEL, NOT THE RANKING ONE.
        #
        # A first version gated them on ranking, reasoning that an out-of-range
        # array still ranks perfectly well -- which is true of a SCORE. But this
        # array arrived through a parameter named `y_proba`. Letting it through
        # to `roc_curve` while `calibration_curve` refuses it would preserve
        # exactly the incoherent contract this commit removes:
        #
        #     the same array, invalid as a probability for calibration
        #                     yet accepted as a probability for the receiver
        #                     operating characteristic curve
        #
        # `roc_curve` accepts it because it consumes SCORES, not because the
        # array is a valid probability. When a `scores` channel exists, ranking
        # quantities will compute from it; until then, a caller who supplied an
        # out-of-range array in `y_proba` supplied invalid model output.
        ranking_usable = label_check.ok and ranking_check.ok
        probability_usable = label_check.ok and probability_check.ok

        # THE REGISTRY IS NOW THE ONLY COMPUTATION PATH (2026-07-28, commit 3b-2).
        #
        # Until this commit the eight scalar fields were computed HERE --
        # roc_auc_score, average_precision_score, matthews_corrcoef at a
        # hard-coded 0.5, f1_score at the same hidden threshold, an inline Brier
        # expression and a private calibration loop -- while the registry computed
        # them again independently. Two implementations of one quantity is how
        # they come to disagree, and this stack found three such disagreements by
        # measurement before the duplication was removed.
        #
        # The flat fields are DERIVED VIEWS. `project_legacy_fields` is the single
        # projection, and the abstract-syntax-tree guard refuses any direct kernel
        # call or threshold comparison in this path.
        #
        # THE POPULATION. `evaluate` receives arrays, not a canonical table, so it
        # has no source identity unless the caller supplies one. Attribution is
        # OPTIONAL and never faked: with `source_id` the population is attributed
        # and carries a membership fingerprint; without it it is unattributed, has
        # NO fingerprint, and comparison against any other population returns
        # UNKNOWN rather than a false equality.
        population = EvaluationPopulation.full(
            n, scope="attempted_cohort", source_id=source_id)
        # THE TWO REGISTRY CHANNELS RECEIVE DIFFERENT ARRAYS (2026-07-28, CI-t).
        #
        # Until this commit both received `p`. The registry has always drawn the
        # right distinction -- `auroc` and `auprc` consume `y_score` and rank
        # scale-free, while every probability-dependent metric refuses an
        # out-of-range array with `values_are_not_probabilities` -- but feeding
        # one array into both channels meant an out-of-range `y_proba` was ranked
        # as though it were a score, while the report path withheld the curve
        # computed from the same values. Two layers, one input, opposite
        # verdicts.
        #
        # `ranking_values` is `scores` when the caller supplied it and `p`
        # otherwise, so the registry now receives what the caller actually meant.
        metric_results = evaluate_registered(MetricContext(
            y_true=y.astype(float), y_prob=p.astype(float),
            y_score=(np.asarray(ranking_values, dtype=float)
                     if ranking_values is not None else None),
            population=population))
        legacy = project_legacy_fields(metric_results)

        # Gene clusters, resolved ONCE and shared by both intervals, from the
        # same frame the breakdowns use. Deriving them separately per metric
        # would reintroduce the alignment-defect class the canonical seam exists
        # to prevent.
        cluster = resolve_gene_clusters(meta)
        if not cluster.usable:
            logger.warning(
                "certified bootstrap withheld: %s (%s). Point metrics are "
                "unaffected.", cluster.status.value, cluster.finding)
        else:
            logger.info(
                "gene clusters resolved from %s: %d clusters over %d rows "
                "(partition_verified=%s)",
                cluster.source, cluster.n_clusters, cluster.n_rows,
                cluster.partition_verified)

        ci_fields: dict = {}
        ci_fields.update(self._interval_fields("auroc", y, p, roc_auc_score, cluster))
        ci_fields.update(self._interval_fields("auprc", y, p, average_precision_score, cluster))

        if ranking_usable:
            fpr, tpr, _ = roc_curve(y, ranking_values)
        else:
            fpr, tpr = np.array([]), np.array([])
            logger.warning(
                "receiver operating characteristic curve withheld: %s (%s)",
                (label_check.reason or ranking_check.reason),
                (label_check.detail or ranking_check.detail or ""))

        if ranking_usable:
            prec, rec, _ = precision_recall_curve(y, ranking_values)
        else:
            prec, rec = np.array([]), np.array([])

        # Calibration is gated on the PROBABILITY channel, not the ranking one.
        # This is the call that would otherwise ship a NaN-poisoned artifact,
        # since it neither raises nor warns on non-finite input.
        if probability_usable:
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=10, strategy="quantile")
        else:
            frac_pos, mean_pred = np.array([]), np.array([])
            logger.warning(
                "calibration curve withheld: %s (%s). The reliability diagram "
                "is omitted rather than computed over an undeclared cohort.",
                (label_check.reason or probability_check.reason),
                (label_check.detail or probability_check.detail or ""))
        # The expected and maximum calibration errors are projections now. The
        # private loop that produced them has been deleted along with the rest of
        # the duplicate computation.

        # Operating points
        op_90  = self._find_operating_point(y, p, target_sensitivity=0.90)
        op_95  = self._find_operating_point(y, p, target_sensitivity=0.95)
        op_ppv = self._find_high_ppv_point(y, p, min_ppv=0.80)

        # Breakdowns (require meta)
        consequence_rows: list = []
        gene_error_rows:  list = []
        if meta is not None:
            consequence_rows = self._consequence_breakdown(y, p, meta)
            gene_error_rows  = self._gene_error_analysis(y, p, meta, top_n=20)

        # THE TYPED SURFACE IS NOW EMITTED (commit 3b-2).
        #
        # Commit 3a introduced schema version 3 as a CAPABILITY and stated that
        # `evaluate` would not emit it until the report became a pure projection.
        # It is one now: every flat scalar above is `project_legacy_fields`
        # output, so the typed results that produced them are carried alongside
        # rather than discarded. A report whose scalars are projections of a
        # mapping it does not include would force every consumer wanting status,
        # reason, population or certification to recompute them.
        report = EvaluationReport(
            schema_version=EVALUATION_REPORT_SCHEMA_VERSION_TYPED,
            metric_results=metric_results,
            model_name=model_name,
            n_samples=n, n_pathogenic=n_pos, n_benign=n_neg,
            # DERIVED VIEWS, every one. None is computed here. The rounding lives
            # in the projection policy, per field -- prevalence at four decimals
            # and the rest at five -- EXTRACTED from what this constructor used to
            # do rather than chosen.
            prevalence=legacy["prevalence"],
            auroc=legacy["auroc"],
            auprc=legacy["auprc"],
            mcc=legacy["mcc"],
            f1=legacy["f1"],
            brier_score=legacy["brier_score"],
            calibration_ece=legacy["calibration_ece"],
            calibration_mce=legacy["calibration_mce"],
            at_sensitivity_90=op_90,
            at_sensitivity_95=op_95,
            at_high_ppv=op_ppv,
            consequence_breakdown=consequence_rows,
            gene_errors=gene_error_rows,
            fpr_curve=fpr.tolist(),
            tpr_curve=tpr.tolist(),
            precision_curve=prec.tolist(),
            recall_curve=rec.tolist(),
            calibration_frac_pos=frac_pos.tolist(),
            calibration_mean_pred=mean_pred.tolist(),
            **ci_fields,
        )
        self.print_report(report)
        return report

    # ── Metric helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _nan_safe(metric_fn):
        """Return a metric that yields NaN, rather than raising, on a degenerate resample.

        scikit-learn's roc_auc_score raises ValueError when a resample contains
        one class; the kernel's resampling loops test np.isfinite on the returned
        value and never catch. Gene-cluster resampling makes single-class draws
        entirely reachable -- a cohort of thirty genes can easily draw thirty
        all-benign clusters -- so without this the certified path would crash on
        exactly the cohorts it exists to serve.

        Wrapping the SAME function the point estimate uses, rather than
        substituting the kernel's own rank implementation, keeps the interval
        bounding precisely the quantity the report states.
        """
        def wrapped(y_i, s_i):
            try:
                return float(metric_fn(y_i, s_i))
            except ValueError:
                return float("nan")
        return wrapped

    def _interval_fields(
        self,
        metric: str,
        y: np.ndarray,
        p: np.ndarray,
        metric_fn,
        cluster: ClusterResolution,
    ) -> dict:
        """Compute one certified interval and flatten it into report fields.

        n_requested CONTRACT, stated explicitly because the number is otherwise
        ambiguous: it is 0 when no bootstrap was attempted -- there was no usable
        cluster identifier, so nothing was ever requested -- and the configured
        replicate count when one WAS attempted, whatever its outcome. A reader
        can therefore distinguish "we never asked" from "we asked and got too
        few back" without consulting the finding string.
        """
        prefix = f"{metric}_ci_"

        # GATED 2026-07-28 (CI-t). `roc_auc_score` and `average_precision_score`
        # both RAISE on non-finite input, so an unusable prediction array would
        # abort the whole report from inside the bootstrap -- after the point
        # metrics had already been computed successfully.
        #
        # Refused here as INSUFFICIENT_SUPPORT rather than raising, using the
        # same shape the unusable-cluster branch below already returns, so a
        # reader sees one vocabulary for "no interval, and why".
        # Gated on the PROBABILITY channel for the same reason as the curves:
        # this array arrived as `y_proba`, so out-of-range values are invalid
        # model output rather than legitimate ranking scores.
        ranking_check = validate_probabilities(p, n_expected=len(y))
        label_check = validate_reference_labels(y, n_expected=len(y))
        if not (ranking_check.ok and label_check.ok):
            reason = ranking_check.reason or label_check.reason
            logger.warning(
                "certified interval for %s withheld: %s (%s). Point metrics are "
                "unaffected.", metric, reason,
                ranking_check.detail or label_check.detail or "")
            return {
                prefix + "lo": None,
                prefix + "hi": None,
                prefix + "status": MetricStatus.FAILED,
                prefix + "resampling_unit": None,
                prefix + "stratified": None,
                prefix + "cluster_source": cluster.source,
                prefix + "partition_verified": cluster.partition_verified,
                prefix + "certification_eligible": False,
                prefix + "n_requested": 0,
                prefix + "n_valid": 0,
                prefix + "n_degenerate": 0,
                prefix + "finding": reason,
            }

        if not cluster.usable:
            return {
                prefix + "lo": None,
                prefix + "hi": None,
                prefix + "status": cluster.status,
                prefix + "resampling_unit": None,
                prefix + "stratified": None,
                prefix + "cluster_source": cluster.source,
                prefix + "partition_verified": cluster.partition_verified,
                prefix + "certification_eligible": False,
                prefix + "n_requested": 0,
                prefix + "n_valid": 0,
                prefix + "n_degenerate": 0,
                prefix + "finding": cluster.finding,
            }

        # Imported HERE, not at module scope: metrics.py imports scikit-learn at
        # module level, and this module must import without it.
        from genomic_variant_classifier.evaluation.metrics import bootstrap_metric

        result = bootstrap_metric(
            self._nan_safe(metric_fn), y, p,
            clusters=cluster.values,
            unit=BootstrapUnit.GENE,
            n_boot=self.n_bootstrap,
            seed=derive_seed(self.random_state, metric),
        )
        available = result.status is MetricStatus.OK
        return {
            prefix + "lo": round(float(result.lower), 5) if available else None,
            prefix + "hi": round(float(result.upper), 5) if available else None,
            prefix + "status": result.status,
            prefix + "resampling_unit": result.resampling_unit if available else None,
            prefix + "stratified": result.stratified if available else None,
            prefix + "cluster_source": cluster.source,
            prefix + "partition_verified": cluster.partition_verified,
            prefix + "certification_eligible": result.certification_eligible,
            prefix + "n_requested": result.n_requested,
            prefix + "n_valid": result.n_valid,
            prefix + "n_degenerate": result.n_degenerate,
            prefix + "finding": result.finding,
        }

    def _find_operating_point(
        self,
        y: np.ndarray,
        p: np.ndarray,
        target_sensitivity: float,
    ) -> Optional[OperatingPoint]:
        """Find the threshold closest to the target sensitivity (recall).

        GATED 2026-07-28 (CI-t), after measuring what this sweep does to an
        unusable prediction.

        `preds = (p >= t)` evaluates FALSE for a NaN, so every non-finite
        probability silently became a PREDICTED NEGATIVE. Measured on a cohort
        where 100 of 200 true positives had unusable predictions: the operating
        point moved from (threshold 0.6366, sensitivity 0.90, specificity 1.00,
        positive predictive value 1.0000) to (threshold 0.0000, sensitivity 0.50,
        specificity 0.00, positive predictive value 0.3333). No exception, no
        warning -- a plausible clinical decision threshold describing a cohort
        nobody declared.

        The sweep also assumes the probability SCALE: it walks thresholds across
        [0, 1], so an array outside that range would place every row on one side
        of every threshold. Both conditions are refused here rather than
        producing a number.
        """
        validation = validate_probabilities(p, n_expected=len(y))
        if not validation.ok:
            logger.warning(
                "operating point at sensitivity %.2f withheld: %s (%s). A "
                "threshold sweep over unusable predictions would report a "
                "decision threshold for a cohort nobody declared.",
                target_sensitivity, validation.reason, validation.detail or "")
            return None
        label_validation = validate_reference_labels(y, n_expected=len(y))
        if not label_validation.ok:
            logger.warning(
                "operating point at sensitivity %.2f withheld: %s (%s)",
                target_sensitivity, label_validation.reason,
                label_validation.detail or "")
            return None

        best: Optional[OperatingPoint] = None
        best_diff = float("inf")
        for t in np.linspace(0, 1, 1000):
            preds = (p >= t).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            n_pos = tp + fn
            n_neg = fp + tn
            if n_pos == 0:
                continue
            sensitivity = tp / n_pos
            diff = abs(sensitivity - target_sensitivity)
            if diff < best_diff:
                best_diff = diff
                specificity = tn / n_neg if n_neg > 0 else 0.0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                f1  = (2 * ppv * sensitivity / (ppv + sensitivity)
                       if (ppv + sensitivity) > 0 else 0.0)
                best = OperatingPoint(
                    threshold=round(float(t), 4),
                    sensitivity=round(sensitivity, 4),
                    specificity=round(specificity, 4),
                    ppv=round(ppv, 4),
                    npv=round(npv, 4),
                    f1=round(f1, 4),
                    n_flagged=int(tp + fp),
                    n_tp=tp, n_fp=fp, n_fn=fn, n_tn=tn,
                )
        return best

    def _find_high_ppv_point(
        self,
        y: np.ndarray,
        p: np.ndarray,
        min_ppv: float = 0.80,
    ) -> Optional[OperatingPoint]:
        """
        Highest-sensitivity threshold where PPV ≥ min_ppv.

        Walk thresholds from HIGH->LOW (conservative->liberal).
        Track the last threshold seen where ppv >= min_ppv -- that is the
        most permissive threshold that never drops below min_ppv.

        GATED 2026-07-28 (CI-t), for the same reason as the sensitivity sweep and
        found the same way. Gating the two sensitivity targets alone left THIS
        one still reporting a decision threshold on a cohort where 100 of 400
        predictions were unusable: sensitivity 0.5, specificity 0.875, positive
        predictive value 0.8 -- entirely plausible, entirely wrong, because
        `p >= t` counts every non-finite prediction as a predicted negative.

        Three operating points, three call sites, one contract. Gating two of
        three would have read as though the class were closed.
        """
        validation = validate_probabilities(p, n_expected=len(y))
        if not validation.ok:
            logger.warning(
                "high positive-predictive-value operating point withheld: %s "
                "(%s)", validation.reason, validation.detail or "")
            return None
        label_validation = validate_reference_labels(y, n_expected=len(y))
        if not label_validation.ok:
            logger.warning(
                "high positive-predictive-value operating point withheld: %s "
                "(%s)", label_validation.reason, label_validation.detail or "")
            return None
        thresholds = np.sort(np.unique(p))[::-1]  # high → low
        best: Optional[OperatingPoint] = None

        for t in thresholds:
            preds = (p >= t).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            n_pos = tp + fn
            n_neg = tp + fp  # n_flagged

            if n_neg == 0 or n_pos == 0:
                continue

            ppv = tp / n_neg
            if ppv < min_ppv:
                # Once PPV drops below target, stop — prior iteration was the best
                break

            sensitivity = tp / n_pos
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1          = (2 * ppv * sensitivity / (ppv + sensitivity)
                           if (ppv + sensitivity) > 0 else 0.0)

            best = OperatingPoint(
                threshold   = round(float(t), 4),
                sensitivity = round(sensitivity, 4),
                specificity = round(specificity, 4),
                ppv         = round(ppv, 4),
                npv         = round(npv, 4),
                f1          = round(f1, 4),
                n_flagged   = int(n_neg),
                n_tp=tp, n_fp=fp, n_fn=fn, n_tn=tn,
            )

        return best

    # ── Breakdown helpers ──────────────────────────────────────────────────

    def _consequence_breakdown(
        self,
        y: np.ndarray,
        p: np.ndarray,
        meta: pd.DataFrame,
    ) -> list[ConsequenceBreakdown]:
        """AUROC and AUPRC broken down by coarsened consequence category."""
        _ensure_sklearn()  # PHASE5: load sklearn symbols into module globals
        # GATED 2026-07-28. THIS PATH WAS MISSED BY CI-t.
        #
        # CI-t enumerated ten call sites and reported the class closed. It was
        # not: the subgroup breakdowns call `roc_auc_score` and
        # `average_precision_score` directly, and they are reached ONLY when
        # `meta` is supplied. Every corrupt-model test written for CI-t passed
        # `meta=None`, so the fixture shape hid the gap -- the same failure that
        # hid the calibration binning defect for seventeen days.
        #
        # Found by the very next measurement, which supplied `meta` in order to
        # exercise clustered bootstrap intervals.
        validation = validate_probabilities(p, n_expected=len(y))
        if not validation.ok:
            logger.warning(
                "consequence breakdown withheld: %s (%s). A subgroup area under "
                "the curve computed over unusable predictions would describe a "
                "cohort nobody declared.", validation.reason, validation.detail or "")
            return []

        if "consequence" not in meta.columns:
            return []

        meta = meta.reset_index(drop=True)
        consequence = meta["consequence"].fillna("unknown")

        def coarsen(c: str) -> str:
            c = str(c).lower()
            if any(t in c for t in [
                "stop_gained", "frameshift", "splice_donor",
                "splice_acceptor", "start_lost",
            ]):
                return "loss_of_function"
            if "missense"   in c: return "missense"
            if "synonymous" in c: return "synonymous"
            if "splice"     in c: return "splice_region"
            if "inframe"    in c: return "inframe_indel"
            return "other"

        consequence_coarse = consequence.map(coarsen)
        rows: list[ConsequenceBreakdown] = []

        for cat in sorted(consequence_coarse.unique()):
            mask = (consequence_coarse == cat).values
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            rows.append(ConsequenceBreakdown(
                consequence=cat,
                n_total=int(mask.sum()),
                n_pathogenic=int(y[mask].sum()),
                auroc=round(float(roc_auc_score(y[mask], p[mask])), 4),
                auprc=round(float(average_precision_score(y[mask], p[mask])), 4),
                prevalence=round(float(y[mask].mean()), 4),
            ))
        return rows

    def _gene_error_analysis(
        self,
        y: np.ndarray,
        p: np.ndarray,
        meta: pd.DataFrame,
        top_n: int = 20,
        threshold: float = 0.5,
    ) -> list[GeneErrorAnalysis]:
        """
        Identify genes contributing most to false positives and negatives.

        CHANGE: The original code used itertuples() and then **row._asdict().
        pandas renames columns that collide with NamedTuple reserved names
        (e.g., "index", "_fields") which causes KeyError on unpack.
        Using .to_dict(orient="records") returns plain dicts that unpack
        cleanly and are immune to column-name collisions (Issue S).
        """
        if "gene_symbol" not in meta.columns:
            return []

        meta = meta.reset_index(drop=True).copy()
        preds = (p >= threshold).astype(int)

        meta["_fp"] = ((preds == 1) & (y == 0)).astype(int)
        meta["_fn"] = ((preds == 0) & (y == 1)).astype(int)

        gene_errors = (
            meta.groupby("gene_symbol")
            .agg(
                n_variants=("_fp", "count"),
                n_false_positives=("_fp", "sum"),
                n_false_negatives=("_fn", "sum"),
            )
            .reset_index()
        )
        gene_errors["total_errors"] = (
            gene_errors["n_false_positives"] + gene_errors["n_false_negatives"]
        )
        gene_errors["error_rate"] = (
            gene_errors["total_errors"] / gene_errors["n_variants"]
        ).round(4)

        gene_errors = (
            gene_errors
            .sort_values("total_errors", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        # CHANGE: to_dict → plain dicts → unpack cleanly (Issue S)
        return [
            GeneErrorAnalysis(**row)
            for row in gene_errors.to_dict(orient="records")
        ]

    # ── Output ─────────────────────────────────────────────────────────────

    def print_report(self, r: EvaluationReport) -> None:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT: {r.model_name}")
        print(sep)
        print(
            f"  Dataset: {r.n_samples:,} variants  "
            f"({r.n_pathogenic:,} pathogenic = {r.prevalence*100:.1f}%)"
        )
        print()
        print(
            f"  AUROC   : {r.auroc:.4f}  95% CI: "
            + format_ci(r.auroc_ci_lo, r.auroc_ci_hi,
                        status=r.auroc_ci_status, finding=r.auroc_ci_finding)
        )
        print(
            f"  AUPRC   : {r.auprc:.4f}  95% CI: "
            + format_ci(r.auprc_ci_lo, r.auprc_ci_hi,
                        status=r.auprc_ci_status, finding=r.auprc_ci_finding)
        )
        if r.auroc_ci_status is MetricStatus.OK:
            print(
                f"            resampling unit: {r.auroc_ci_resampling_unit.value}"
                f"  certified: {str(r.auroc_ci_certification_eligible).lower()}"
                f"  clusters from: {r.auroc_ci_cluster_source}"
            )
        print(f"  MCC     : {r.mcc:.4f}")
        print(f"  F1      : {r.f1:.4f}")
        print(
            f"  Brier   : {r.brier_score:.4f}  "
            f"(ECE: {r.calibration_ece:.4f}, MCE: {r.calibration_mce:.4f})"
        )

        for label, op in [
            ("@ Sensitivity ≥ 90%", r.at_sensitivity_90),
            ("@ Sensitivity ≥ 95%", r.at_sensitivity_95),
            ("@ PPV ≥ 80%",         r.at_high_ppv),
        ]:
            if op:
                print()
                print(f"  {label}  (threshold={op.threshold:.3f}):")
                print(
                    f"    Sens: {op.sensitivity:.3f}  Spec: {op.specificity:.3f}  "
                    f"PPV: {op.ppv:.3f}  NPV: {op.npv:.3f}  Flagged: {op.n_flagged:,}"
                )

        if r.consequence_breakdown:
            print()
            print(f"  {'Consequence':<22} {'N':>7} {'%Path':>7} {'AUROC':>8} {'AUPRC':>8}")
            print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
            for cb in sorted(r.consequence_breakdown, key=lambda x: x.auroc, reverse=True):
                print(
                    f"  {cb.consequence:<22} {cb.n_total:>7,} "
                    f"{cb.prevalence*100:>6.1f}% "
                    f"{cb.auroc:>8.4f} {cb.auprc:>8.4f}"
                )

        print(sep + "\n")

    def save_report(self, report: EvaluationReport, path: str | Path) -> None:
        """Serialize the full report to JSON (curves included for downstream plotting).

        STRICT since 2026-07-26. The previous implementation passed
        `default=str`, which silently rendered any NumPy integer as a JSON
        STRING, and left `allow_nan` at its default, which wrote bare `NaN`
        literals that are not valid JSON. Both are corrections to persisted
        evidence, not formatting preferences. See evaluation/serialization.py.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = dump_strict_json(report.to_serializable(), artifact=str(path))
        path.write_text(text, encoding="utf-8", newline="\n")
        logger.info("Evaluation report saved to %s", path)


# ---------------------------------------------------------------------------
# Multi-model comparison convenience function
# ---------------------------------------------------------------------------
def compare_models(
    y_true: np.ndarray,
    model_probas: dict[str, np.ndarray],
    meta: Optional[pd.DataFrame] = None,
    n_bootstrap: int = 500,
    output_csv: str = "models/model_comparison.csv",
) -> pd.DataFrame:
    """
    Compare multiple models in one call.

    Args:
        y_true:        Ground-truth binary labels.
        model_probas:  {model_name: proba_array}.
        meta:          Optional variant metadata for consequence/gene breakdowns.
        n_bootstrap:   Bootstrap iterations for CI estimation.
        output_csv:    Where to save the comparison table.

    Returns:
        DataFrame with one row per model, sorted by AUROC descending.
    """
    evaluator = ClinicalEvaluator(n_bootstrap=n_bootstrap)
    records: list[dict] = []

    for name, proba in model_probas.items():
        r = evaluator.evaluate(y_true, proba, meta=meta, model_name=name)
        records.append({
            "model":             name,
            "auroc":             r.auroc,
            "auroc_95ci":        format_ci(r.auroc_ci_lo, r.auroc_ci_hi,
                                                status=r.auroc_ci_status,
                                                finding=r.auroc_ci_finding),
            "auroc_ci_certified": r.auroc_ci_certification_eligible,
            "auprc":             r.auprc,
            "mcc":               r.mcc,
            "f1":                r.f1,
            "brier":             r.brier_score,
            "ece":               r.calibration_ece,
            "sens_at_90_spec":   r.at_sensitivity_90.specificity if r.at_sensitivity_90 else None,
            "ppv_at_90_sens":    r.at_sensitivity_90.ppv         if r.at_sensitivity_90 else None,
        })

    df = pd.DataFrame(records).sort_values("auroc", ascending=False).reset_index(drop=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Comparison table saved to %s", out_path)
    return df
