"""Legal transitions between capability states, enforced rather than advised.

WHY THIS MODULE EXISTS
======================
`capabilities.py` answers "what state is this capability in, and is that state
internally consistent?" It cannot answer "was this state reached legitimately?"
Those are different questions, and only the second one prevents the failure that
matters here.

The specific instruction this module makes structural: DO NOT FLIP A CAPABILITY
FROM NOT_IMPLEMENTED STRAIGHT TO VALIDATED. That is written as guidance in
prose, and prose does not enforce itself -- the same lesson the Continuous
Integration suite-size comment taught when 330 tests could have vanished behind
a hand-maintained floor that said PASS.

A capability that jumps from "not built" to "validated" in one commit has, by
construction, never had a commit in which its output existed but was unverified.
That intermediate state is where the interesting defects live: an artifact that
is produced but wrong, a probe that runs but leaks, a metric that computes but
on the wrong tensor. Skipping it is not efficiency. It is skipping the audit.

THE LADDER
----------
    NOT_IMPLEMENTED        nothing runs
      |                    (build the thing)
    IMPLEMENTED_NO_OUTPUT  it runs, produces nothing durable
      |                    (persist an artifact)
    OUTPUT_AVAILABLE       an artifact exists, unverified
      |                    (verify it against evidence)
    VALIDATED              the artifact is checked and admissible
      |
    DEPRECATED             withdrawn, from any state

Backward moves are ALWAYS legal, from any state to any earlier state, and to
DEPRECATED from anywhere. Discovering that an artifact is wrong must never be
harder to record than claiming it is right. A contract that made retraction
awkward would quietly encourage not retracting.

Forward moves advance EXACTLY ONE RUNG. That is the whole rule.

WORKED EXAMPLE -- PANEL R
-------------------------
Stage one of Panel R registered R3 through R7 as NOT_IMPLEMENTED / ABSENT
because `models/gnn.py` computes `focal_embeddings` and returns only
`classifier(focal_embeddings)`; the representation is discarded at the return
boundary.

That is an OBSERVABILITY gap, not an impossibility, and the distinction was
overstated when stage one shipped: those panels are not unbuildable, they are
blocked on a refactor that has not happened yet. `panel_r_expected_ladder()`
below records the sequence they must climb, so the claim "R3 is validated" can
be checked against the path taken rather than taken on trust.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .capabilities import CapabilityEvidence, CapabilityState, MetricStatus, TargetState

logger = logging.getLogger(__name__)

__all__ = [
    "REASON_REPRESENTATION_NOT_EXPOSED",
    "REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE",
    "REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED",
    "REASON_LOCAL_GEOMETRY_EVALUATOR_ABSENT",
    "REASON_CHECKPOINT_SERIES_UNAVAILABLE",
    "REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE",
    "REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES",
    "CAPABILITY_LADDER",
    "TransitionVerdict",
    "transition_is_legal",
    "assert_transition_legal",
    "panel_r_expected_ladder",
    "IllegalCapabilityTransition",
]

# Reasons naming a SPECIFIC missing thing. "not available" is not a reason; it
# is a restatement of the status. Each of these points at one identifiable
# artifact whose absence a reader can go and check.
REASON_REPRESENTATION_NOT_EXPOSED = "representation_not_exposed_at_return_boundary"
REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE = "no_persisted_representation_artifact"
REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED = "probe_protocol_not_implemented"
REASON_LOCAL_GEOMETRY_EVALUATOR_ABSENT = "local_geometry_evaluator_absent"
REASON_CHECKPOINT_SERIES_UNAVAILABLE = "no_checkpoint_series"
REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE = "no_longitudinal_outcome_table"
REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES = "insufficient_longitudinal_replicates"

# The rungs, in order. DEPRECATED is deliberately NOT on the ladder: it is
# reachable from anywhere and leads nowhere, so treating it as a rung would
# imply it can be climbed past.
CAPABILITY_LADDER: tuple = (
    CapabilityState.NOT_IMPLEMENTED,
    CapabilityState.IMPLEMENTED_NO_OUTPUT,
    CapabilityState.OUTPUT_AVAILABLE,
    CapabilityState.VALIDATED,
)


class IllegalCapabilityTransition(ValueError):
    """Raised when a capability is moved more than one rung forward."""


@dataclass(frozen=True)
class TransitionVerdict:
    """Why a transition was allowed or refused. Never a bare boolean."""

    legal: bool
    from_state: CapabilityState
    to_state: CapabilityState
    direction: str          # "forward" | "backward" | "none" | "withdrawal"
    rungs: int
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.legal and not self.reason:
            raise ValueError("an illegal transition must carry a reason")
        if self.legal and self.reason:
            raise ValueError(
                "a legal transition must not carry a reason; a reason explains a "
                "refusal")


def transition_is_legal(from_state: CapabilityState,
                        to_state: CapabilityState) -> TransitionVerdict:
    """One rung forward, any distance backward, DEPRECATED from anywhere."""
    if not isinstance(from_state, CapabilityState):
        raise TypeError(f"from_state must be a CapabilityState, got {type(from_state).__name__}")
    if not isinstance(to_state, CapabilityState):
        raise TypeError(f"to_state must be a CapabilityState, got {type(to_state).__name__}")

    if to_state is CapabilityState.DEPRECATED:
        # Withdrawal is always available. Making retraction awkward would
        # quietly encourage not retracting.
        return TransitionVerdict(True, from_state, to_state, "withdrawal", 0)

    if from_state is CapabilityState.DEPRECATED:
        return TransitionVerdict(
            False, from_state, to_state, "forward", 0,
            "a DEPRECATED capability cannot be revived in place. Register it "
            "under a new name so the deprecation stays visible in the record.")

    if from_state is to_state:
        return TransitionVerdict(True, from_state, to_state, "none", 0)

    i = CAPABILITY_LADDER.index(from_state)
    j = CAPABILITY_LADDER.index(to_state)
    rungs = j - i

    if rungs < 0:
        return TransitionVerdict(True, from_state, to_state, "backward", rungs)
    if rungs == 1:
        return TransitionVerdict(True, from_state, to_state, "forward", rungs)

    skipped = ", ".join(s.value for s in CAPABILITY_LADDER[i + 1:j])
    return TransitionVerdict(
        False, from_state, to_state, "forward", rungs,
        f"{from_state.value} -> {to_state.value} skips {rungs - 1} rung(s): "
        f"{skipped}. A capability that never had a commit in which its output "
        "existed but was unverified has never been audited in the state where "
        "the interesting defects live -- an artifact produced but wrong, a probe "
        "that runs but leaks, a metric computed on the wrong tensor.")


def assert_transition_legal(from_state: CapabilityState,
                            to_state: CapabilityState) -> TransitionVerdict:
    """Raise on an illegal transition. Use where a refusal must stop a pipeline."""
    verdict = transition_is_legal(from_state, to_state)
    if not verdict.legal:
        raise IllegalCapabilityTransition(verdict.reason)
    return verdict


def panel_r_expected_ladder() -> dict:
    """The rung each Panel R stage must climb, and what unblocks each step.

    Recorded so that a future claim of "R3 is validated" can be checked against
    the path taken rather than accepted on trust. This is a MAP, not a gate: it
    asserts nothing about the present state, only about the order.
    """
    exposed = ("models/gnn.py returns only classifier(focal_embeddings); the "
               "representation is discarded at the return boundary")
    return {
        "panel_r3_norm_angle_decomposition": (
            (CapabilityState.NOT_IMPLEMENTED, REASON_REPRESENTATION_NOT_EXPOSED, exposed),
            (CapabilityState.IMPLEMENTED_NO_OUTPUT, REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE,
             "a typed model output exposes the representation, but nothing persists it"),
            (CapabilityState.OUTPUT_AVAILABLE, REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED,
             "an identified, versioned artifact exists; the probe protocol does not"),
            (CapabilityState.VALIDATED, None,
             "probes fit on TRAIN only, applied unchanged to TUNE/STRUCTURE/TEST"),
        ),
        "panel_r4_conditioning_recoverability": (
            (CapabilityState.NOT_IMPLEMENTED, REASON_REPRESENTATION_NOT_EXPOSED, exposed),
            (CapabilityState.IMPLEMENTED_NO_OUTPUT, REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE,
             "shares R3's infrastructure; built alongside it"),
            (CapabilityState.OUTPUT_AVAILABLE, REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED,
             "a whitening transform fit on STRUCTURE or TEST is leakage even "
             "with no labels; the fit partition is the whole contract. When the "
             "probe protocol is implemented it INHERITS the R3b-validated matched-"
             "null protocol (recovery_protocol.py: TRAIN-fitted, alignment-"
             "sensitive shape recovery, matched-spectrum null, synthetically "
             "calibrated) rather than re-deriving one -- see "
             "R4_CONDITIONING_INHERITANCE"),
            (CapabilityState.VALIDATED, None, "transform ladder fit on TRAIN only, "
             "via the inherited R3b-validated recovery protocol"),
        ),
        "panel_r5_hubness_local_geometry": (
            (CapabilityState.NOT_IMPLEMENTED, REASON_LOCAL_GEOMETRY_EVALUATOR_ABSENT,
             "estimand depends on euclidean-versus-cosine, raw-versus-normalised, "
             "exact-versus-approximate neighbours, duplicate and zero-vector policy"),
            (CapabilityState.IMPLEMENTED_NO_OUTPUT, REASON_REPRESENTATION_ARTIFACT_UNAVAILABLE, ""),
            (CapabilityState.OUTPUT_AVAILABLE, REASON_PROBE_PROTOCOL_NOT_IMPLEMENTED,
             "approximate nearest-neighbour implementations validated against "
             "exact fixtures BEFORE use at project scale"),
            (CapabilityState.VALIDATED, None, ""),
        ),
        "panel_r6_training_trajectory": (
            (CapabilityState.NOT_IMPLEMENTED, REASON_CHECKPOINT_SERIES_UNAVAILABLE,
             "a frozen sentinel cohort must exist first; changing it invalidates "
             "direct longitudinal comparison"),
            (CapabilityState.IMPLEMENTED_NO_OUTPUT, REASON_CHECKPOINT_SERIES_UNAVAILABLE, ""),
            (CapabilityState.OUTPUT_AVAILABLE, REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES,
             "one checkpoint is not a trajectory"),
            (CapabilityState.VALIDATED, None, ""),
        ),
        "panel_r7_downstream_sensitivity": (
            (CapabilityState.NOT_IMPLEMENTED, REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE, ""),
            (CapabilityState.IMPLEMENTED_NO_OUTPUT, REASON_LONGITUDINAL_OUTCOMES_UNAVAILABLE, ""),
            (CapabilityState.OUTPUT_AVAILABLE, REASON_INSUFFICIENT_LONGITUDINAL_REPLICATES,
             "lead-time, change-point and lagged-correlation analysis all need "
             "multiple checkpoints AND multiple seeds"),
            (CapabilityState.VALIDATED, None, ""),
        ),
    }
