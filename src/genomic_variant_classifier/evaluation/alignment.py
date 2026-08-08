"""src/genomic_variant_classifier/evaluation/alignment.py

Author: Monzia Moodie
Written 2026-08-07. ALIGNMENT-1, part of GATE-1.

Is the observed score/label pairing credible as the intended join?

WHAT THIS IS NOT. It is not a model-quality gate, and it is not the metric
registry in `evaluation/registry.py`. It answers a DATA INTEGRITY question:
given scores and labels that are supposed to correspond row-for-row, is their
discrimination high enough that the join is plausibly the one intended?

WHY IT EXISTS. Two places asked that question independently with the same
number and no shared definition:

    conformal/calibrate.py            auroc_floor = 0.90, raising
                                      AlignmentError with "the score<->label
                                      join is broken; refusing to calibrate"
    scripts/forensics/
      verify_oof_alignment.py         AUROC_FLOOR = 0.90, "a correctly-joined
                                      base model should be well above chance"

GATE-1's original census counted four AUROC thresholds in this repository. It
missed the forensic one. That is recorded as a CENSUS CORRECTION rather than as
a fifth policy: there are not five gates, there are THREE CLASSES OF DECISION.

    0.90    score<->label alignment integrity        <- this module
    0.97    absolute production-performance floor    <- PromotionPolicy
    0.002   maximum shadow-versus-production drop    <- PromotionPolicy
    0.9842  copied arithmetic; NOT a policy at all   <- deleted 2026-08-07

THE THRESHOLD IS A HEURISTIC, AND THE TYPE SAYS SO. A perfectly sound weak
model can score below 0.90 with a completely correct join. The honest reading
is:

    for this project's expected high-performing out-of-fold substrate, AUROC
    below 0.90 is strong evidence the join MAY be corrupted

and NOT:

    AUROC below 0.90 proves misalignment.

The policy therefore carries `PolicyProvenance` marking its evidence status as
LEGACY_PENDING_JUSTIFICATION. The code keeps enforcing it -- it is a fail-safe
worth having -- but nothing in the repository should be able to mistake
"inherited and never justified" for "established".

WHY THE FORENSIC SCRIPT DOES NOT IMPORT conformal. `verify_oof_alignment.py` is
a general integrity check; making it depend on conformal calibration to obtain
a number would be a backwards dependency. Both consume this module instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from genomic_variant_classifier.monitoring.model_registry import (
    PolicyEvidenceStatus,
    PolicyProvenance,
)

__all__ = [
    "DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY",
    "ScoreLabelAlignmentPolicy",
    "AlignmentVerdict",
]


def _legacy_provenance() -> PolicyProvenance:
    return PolicyProvenance(
        status=PolicyEvidenceStatus.LEGACY_PENDING_JUSTIFICATION,
        source="conformal/calibrate.py and scripts/forensics/"
               "verify_oof_alignment.py, both carrying 0.90 independently "
               "before ALIGNMENT-1 unified them on 2026-08-07")


@dataclass(frozen=True)
class ScoreLabelAlignmentPolicy:
    """The minimum discrimination below which a join is treated as suspect.

    `minimum_auroc` is named for what it measures rather than as a bare
    "floor": this project has cosine alignment, representation alignment and
    cross-modal alignment elsewhere, and a symbol called `alignment_floor`
    would eventually be read as one of those.
    """

    minimum_auroc: float = 0.90
    provenance: PolicyProvenance = field(default_factory=_legacy_provenance)

    def __post_init__(self) -> None:
        if not 0.5 < self.minimum_auroc <= 1.0:
            raise ValueError(
                "a score<->label alignment minimum must lie in (0.5, 1.0]; "
                f"got {self.minimum_auroc}. At or below 0.5 the policy cannot "
                "distinguish a broken join from chance, which is the only "
                "thing it exists to detect.")

    def judge(self, auroc: float) -> "AlignmentVerdict":
        """Evaluate WITHOUT raising, so callers choose their own consequence.

        `calibrate.py` refuses to proceed; the forensic script prints a flag
        and continues. Both are legitimate responses to the same finding, and
        a policy that raised would force one of them.
        """
        return AlignmentVerdict(
            auroc=float(auroc),
            minimum_auroc=self.minimum_auroc,
            plausible=float(auroc) >= self.minimum_auroc)


@dataclass(frozen=True)
class AlignmentVerdict:
    """A judgement with the threshold it was judged against, never a bare bool.

    Reporting `plausible` alone would repeat the defect PROD-1 removed from the
    application programming interface: a verdict detached from the standard it
    was measured by.
    """

    auroc: float
    minimum_auroc: float
    plausible: bool
    detail: Optional[str] = None

    def describe(self) -> str:
        verdict = "plausible" if self.plausible else "SUSPECT"
        return (f"score<->label alignment {verdict}: AUROC {self.auroc:.4f} "
                f"vs minimum {self.minimum_auroc}")


#: The single default both consumers use. Overriding it per call is supported
#: -- `run_conformal_calibration.py` exposes a command-line flag -- but the
#: DEFAULT lives in exactly one place, so the two cannot silently diverge.
DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY = ScoreLabelAlignmentPolicy()
