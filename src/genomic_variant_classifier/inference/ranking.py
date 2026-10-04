"""DANDELION with minimum-score gene aggregation; top-k recovery with a tie audit (owner ruling 2026-10-04).

THE RANKING (primary extended method)
=====================================
Genes are ranked by their minimum finite DANDELION pair score across the prespecified scoreable exposures. Scores are
extracted before exposure-to-gene annotation and locus-level presentation. Per-exposure significance flags do not
determine ranking. Exact score ties are resolved by stable gene ID. The aggregate is a RANKING SCORE, not a calibrated
gene-level p-value -- never multiply or divide it by an exposure count and call the result corrected.

COMPLETENESS, NOT SILENCE: the pair plan is frozen at admission (True = must produce a score; False = structurally excluded
before scoring). A failed computation, a missing row or an unexpected row refuses the ranking -- computational failure can
never silently redefine the shared gene universe.

THREE UNIVERSES: fitting (what the method was fitted on), evaluation (the common eligibility rule every comparator ranks
within) and the reference-positive set (independent adjudication; never an input to fitting or ranking). Scores are
projected onto the evaluation universe; the method is not refitted per evaluation change.

THREE METHOD IDENTITIES: the published method, the pinned implementation actually executed, and the benchmark extension.
MEASURED 2026-10-04: the published empirical-null calibration (JCCorrect, Analysis/real_data/run_dandelion_real_data.R) is
defined but never called in any of the repository's 73 commits, so the pinned implementation executes no calibration.

The calculations come from the owner's reference code (ruling generation 05444c27, lines 201-325 and 548-622), transformed
mechanically (generic errors now raise InferenceError with the same codes); the structural-pair count, score transport,
universe and method identities and annotate() are added.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from genomic_variant_classifier.inference.exact_confirmation import InferenceError

logger = logging.getLogger(__name__)

__all__ = ["State", "Pair", "GeneRank", "rank_genes", "top_k", "audit_top_k", "contrast", "score_from_binary64",
           "UniverseKind", "Universe", "project_scores", "Implementation", "Calibration", "MethodIdentity", "PINNED_COMMIT",
           "annotate", "ScoreRange", "topk_audit"]


class State(Enum):
    SCORED = "scored"
    STRUCTURAL = "structurally_not_tested"
    FAILED = "computation_failed"


@dataclass(frozen=True)
class Pair:
    gene: str
    exposure: str
    state: State
    score: Fraction | None = None
    significant: bool | None = None


@dataclass(frozen=True)
class GeneRank:
    gene: str
    score: Fraction
    best_exposure: str
    tested_pairs: int
    significant_pairs: int
    structural_pairs: int = 0


def rank_genes(rows, plan):
    """
    plan[(gene_id, exposure_id)]:
        True  -> this pair must produce a score
        False -> structurally excluded before scoring

    The admission layer must bind this plan to the sealed contract.
    """
    if not plan:
        raise InferenceError("empty_plan")

    for key, expected_score in plan.items():
        if (
            type(key) is not tuple
            or len(key) != 2
            or any(type(x) is not str or not x for x in key)
            or type(expected_score) is not bool
        ):
            raise InferenceError("invalid_plan")

    seen = {}
    by_gene = {gene: [] for gene, exposure in plan}
    structural = {gene: 0 for gene, exposure in plan}

    for row in rows:
        if type(row) is not Pair:
            raise InferenceError("pair_type")

        key = (row.gene, row.exposure)

        if key in seen:
            raise InferenceError("duplicate_pair")
        if key not in plan:
            raise InferenceError("unexpected_pair")

        seen[key] = row

        if row.state is State.FAILED:
            raise InferenceError("incomplete_method_execution")

        if plan[key]:
            if row.state is not State.SCORED:
                raise InferenceError("required_pair_unscored")
            if (
                type(row.score) is not Fraction
                or not 0 < row.score <= 1
            ):
                raise InferenceError("invalid_score")
            if type(row.significant) is not bool:
                raise InferenceError("invalid_significance_flag")

            by_gene[row.gene].append(row)

        else:
            if (
                row.state is not State.STRUCTURAL
                or row.score is not None
                or row.significant is not None
            ):
                raise InferenceError("structural_pair_has_result")
            structural[row.gene] += 1

    if set(seen) != set(plan):
        raise InferenceError("missing_pair_record")

    ranked = []

    for gene, pairs in by_gene.items():
        if not pairs:
            raise InferenceError("gene_has_no_planned_score")

        best = min(
            pairs,
            key=lambda pair: (pair.score, pair.exposure),
        )

        ranked.append(
            GeneRank(
                gene=gene,
                score=best.score,
                best_exposure=best.exposure,
                tested_pairs=len(pairs),
                significant_pairs=sum(
                    pair.significant for pair in pairs
                ),
                structural_pairs=structural[gene],
            )
        )

    # Significance and exposure counts are deliberately not sort keys.
    return tuple(
        sorted(ranked, key=lambda result: (result.score, result.gene))
    )


def top_k(ranked, k):
    if type(k) is not int or not 1 <= k <= len(ranked):
        raise InferenceError("nomination_budget_incomplete")
    return ranked[:k]


def audit_top_k(scores, eligible, reference, k):
    """
    scores: gene -> exact representation of the stored ranking score
    eligible: frozen evaluation universe
    reference: reference positives within that universe

    Lower scores are better.
    """
    if type(k) is not int or not 1 <= k <= len(eligible):
        raise InferenceError("invalid_k")

    if not reference <= eligible:
        raise InferenceError("reference_outside_evaluation_universe")

    if not eligible <= scores.keys():
        raise InferenceError("evaluation_score_missing")

    if any(type(g) is not str or not g for g in eligible):
        raise InferenceError("gene_identity")

    if any(
        type(scores[g]) is not Fraction
        or not 0 < scores[g] <= 1
        for g in eligible
    ):
        raise InferenceError("invalid_score")

    ordered = sorted(
        eligible,
        key=lambda gene: (scores[gene], gene),
    )

    selected = tuple(ordered[:k])
    cutoff = scores[selected[-1]]

    strictly_better = {
        gene for gene in eligible
        if scores[gene] < cutoff
    }
    tied = {
        gene for gene in eligible
        if scores[gene] == cutoff
    }

    remaining_slots = k - len(strictly_better)
    fixed_hits = len(strictly_better & reference)
    tied_positives = len(tied & reference)

    # These are reference nonmembers, NOT biological negatives.
    tied_nonmembers = len(tied - reference)

    return {
        "selected": selected,
        "hits": len(set(selected) & reference),
        "hit_lower": fixed_hits + max(
            0, remaining_slots - tied_nonmembers
        ),
        "hit_upper": fixed_hits + min(
            remaining_slots, tied_positives
        ),
        "cutoff": cutoff,
        "tied_genes": tuple(sorted(tied)),
        "tie_slots": remaining_slots,
    }


def contrast(a, b):
    return {
        "delta": a["hits"] - b["hits"],
        "tie_lower": a["hit_lower"] - b["hit_upper"],
        "tie_upper": a["hit_upper"] - b["hit_lower"],
    }


# ---------------------------------------------------------------------------------------------------- additions

def score_from_binary64(value: float) -> Fraction:
    """The EXACT stored binary64 score as R produced it (ruling: never via a rounded display string)."""
    if type(value) is not float or not math.isfinite(value) or not 0 < value <= 1:
        raise InferenceError("invalid_score")
    return Fraction.from_float(value)


class UniverseKind(str, Enum):
    FITTING = "fitting"          # what the method was fitted on (method inputs and quality rules)
    EVALUATION = "evaluation"    # the common eligibility rule every primary comparator ranks within


@dataclass(frozen=True)
class Universe:
    """A frozen gene set with the rule that defined it. Its identity is derived from its content."""

    kind: UniverseKind
    rule_id: str
    genes: frozenset

    def __post_init__(self):
        if type(self.kind) is not UniverseKind:
            raise InferenceError("universe_kind")
        if type(self.rule_id) is not str or not self.rule_id.strip():
            raise InferenceError("rule_identity_required")
        if type(self.genes) is not frozenset or not self.genes or any(type(g) is not str or not g for g in self.genes):
            raise InferenceError("universe_genes")

    @property
    def identity(self) -> str:
        body = "\n".join([self.kind.value, self.rule_id] + sorted(self.genes)) + "\n"
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


def project_scores(ranked, evaluation: Universe) -> dict:
    """Project a fitted ranking onto the frozen EVALUATION universe (never refit). Every evaluation gene must have a
    score; genes outside the universe are dropped from the projection (and remain in the unrestricted ranking)."""
    if type(evaluation) is not Universe or evaluation.kind is not UniverseKind.EVALUATION:
        raise InferenceError("evaluation_universe_required")
    scores = {r.gene: r.score for r in ranked}
    if len(scores) != len(ranked):
        raise InferenceError("duplicate_gene")
    missing = evaluation.genes - scores.keys()
    if missing:
        raise InferenceError("evaluation_score_missing", repr(sorted(missing)[:5]))
    return {g: scores[g] for g in evaluation.genes}


class Implementation(str, Enum):
    """The executable variants found at mxxptian/DANDELION f471153 (measured 2026-10-04); they differ numerically."""

    ROOT_PACKAGE = "root_package"                # R/DANDELION.R -- what `library(DANDELION)` loads per the root DESCRIPTION
    NESTED_PACKAGE = "nested_package"            # DANDELION/R/DANDELION.R
    REAL_DATA_SCRIPT = "real_data_script"        # Analysis/real_data/run_dandelion_real_data.R (own med_gene/calc_pair)


class Calibration(str, Enum):
    NONE_EXECUTED = "none_executed"                          # what every variant at f471153 executes
    PUBLISHED_EMPIRICAL_NULL = "published_empirical_null"    # the paper's Jin-Cai step -- a SEPARATE, later variant


_COMMIT = re.compile(r"[0-9a-f]{40}")
PINNED_COMMIT = "f471153bfa3c0069cd68a67565000889c7cdf5d1"   # mxxptian/DANDELION, measured HEAD 2026-10-04


@dataclass(frozen=True)
class MethodIdentity:
    """Published method, pinned implementation and benchmark extension -- three identities, never merged."""

    published_method: str            # the citation of the described procedure
    repository: str
    commit: str
    implementation: Implementation
    calibration: Calibration
    qvalue_backend_policy: str       # e.g. "safe_qvalues: BH if n < 10 or < 4 distinct values or on any qvalue warning/error"
    extension: str                   # e.g. "minimum-score gene aggregation v1"

    def __post_init__(self):
        for name in ("published_method", "repository", "qvalue_backend_policy", "extension"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise InferenceError("method_" + name)
        if type(self.commit) is not str or not _COMMIT.fullmatch(self.commit):
            raise InferenceError("method_commit")
        if type(self.implementation) is not Implementation or type(self.calibration) is not Calibration:
            raise InferenceError("method_variant")
        if self.commit == PINNED_COMMIT and self.calibration is not Calibration.NONE_EXECUTED:
            # measured: no variant at the pinned commit executes the published calibration
            raise InferenceError("calibration_not_executed_at_pinned_commit")



def annotate(ranked, annotations) -> tuple:
    """Attach descriptive labels (exposure-to-gene mapping, burden significance, pathway) to a FINISHED ranking.

    The order is returned unchanged by construction: annotations are downstream of scoring and can never be a sort key."""
    if type(annotations) is not dict:
        raise InferenceError("annotations_type")
    return tuple((r, annotations.get(r.gene, {})) for r in ranked)


# ---------------------------------------------------------------------------------------------------- numerical sensitivity audit
# Owner ruling 2026-10-04b, reference code (ruling generation 98ea254a, lines 558-623), transformed mechanically (its generic errors now
# raise InferenceError with the same codes). For each gene, the best and worst rank over every assignment of scores within the supplied
# intervals [low, high] -- smaller scores rank first, stable gene IDs break ties -- and whether its top-k membership is "always_in",
# "always_out" or "sensitive". These are SENSITIVITY bounds, never confidence intervals: an interval may be a point (the primary run), an
# observed range across qualified environments, or a certified numerical bound -- each means something different. Rank is monotone in
# every competitor's score, so the extremes are attained at interval endpoints (verified 2026-10-04 against exhaustive endpoint
# enumeration in 27,482 gene-cutoff checks). It never changes the primary ranking.

@dataclass(frozen=True)
class ScoreRange:
    gene: str
    low: Fraction
    high: Fraction

    def __post_init__(self):
        if not isinstance(self.gene, str) or not self.gene:
            raise InferenceError("gene_id")

        if type(self.low) is not Fraction:
            raise InferenceError("exact_bounds_required")
        if type(self.high) is not Fraction:
            raise InferenceError("exact_bounds_required")

        if not 0 <= self.low <= self.high <= 1:
            raise InferenceError("score_range")


def topk_audit(rows, k):
    """Bounds over the Cartesian product of supplied score intervals.

    These are sensitivity bounds, not statistical confidence intervals.
    Correlated score changes can make the bounds conservative.
    """
    rows = tuple(rows)

    if type(k) is not int or not 1 <= k <= len(rows):
        raise InferenceError("k_range")

    if len({row.gene for row in rows}) != len(rows):
        raise InferenceError("duplicate_gene")

    result = {}

    for target in rows:
        others = [row for row in rows if row.gene != target.gene]

        # Best possible rank: target is at its minimum,
        # and every competing gene is at its maximum.
        best_rank = 1 + sum(
            (other.high, other.gene) < (target.low, target.gene)
            for other in others
        )

        # Worst possible rank: target is at its maximum,
        # and every competing gene is at its minimum.
        worst_rank = 1 + sum(
            (other.low, other.gene) < (target.high, target.gene)
            for other in others
        )

        if worst_rank <= k:
            status = "always_in"
        elif best_rank > k:
            status = "always_out"
        else:
            status = "sensitive"

        result[target.gene] = {
            "best_rank": best_rank,
            "worst_rank": worst_rank,
            "status": status,
        }

    return result
