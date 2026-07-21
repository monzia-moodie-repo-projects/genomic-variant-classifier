"""The capability-and-evidence contract: what a panel may claim, and when.

WHY THIS MODULE EXISTS
======================
A validation gate over a capability that does not exist does not block. It
PASSES VACUOUSLY, because there is nothing to check and therefore nothing to
fail. A green Panel H would then be cited as evidence that a disease head was
validated, when no disease head exists.

That is the defect this project removed five separate times on 2026-07-21:

  * `assert_data_usable` was well tested and called from nowhere.
  * `PartitionSchema` capped role repetition against an enumerated list that
    was already stale on the morning it was written.
  * The isotonic calibrator was fitted on genes the models had trained on,
    while the cross-validation twenty lines below was carefully gene-disjoint.
  * `calibration_valid` asserted that calibration numbers were sound after
    checking only that the values lay between zero and one.
  * `n_groups_with_multiple_covariate_values` was recorded from the start and
    had exactly one reader in the entire repository: a test asserting it
    equalled zero.

    A check that cannot fail is worse than no check, because it manufactures
    confidence.

THE CONTRACT
------------
Four questions are separated, because collapsing them is what allows a false
green:

    does the capability exist?          CapabilityState
    does it produce an output?          CapabilityState.OUTPUT_AVAILABLE
    is its target scientifically
      admissible?                       TargetState
    has it passed validation?           CapabilityState.VALIDATED

`CapabilityEvidence.__post_init__` then makes a false green STRUCTURALLY
IMPOSSIBLE TO CONSTRUCT rather than merely unlikely to be reported. An OK
status cannot be built without a validated capability, an admissible target and
a named output artifact; and any non-OK status cannot be built without a
machine-readable reason. The object refuses to exist in an inconsistent state,
so no downstream gate has to remember to check.

TWO DELIBERATE DEPARTURES FROM THE ORIGINAL PROPOSAL
----------------------------------------------------
1. `MetricStatus` is defined HERE, ONCE, and imported by clustering_metrics.
   The proposal defined a second class of the same name in a second module.
   Two enums sharing a name is the divergence problem this project removed on
   2026-07-21 in `b8275a0`, when `compute_classification_metrics` was deleted
   rather than wrapped precisely because two evaluation contracts invite drift.
   Status vocabulary is more foundational than any panel, so it lives at the
   bottom of the layering and panels import upward.

2. `(str, Enum)` rather than `StrEnum`. `StrEnum` arrived in Python 3.11, and
   `pyproject.toml` declares `requires-python = ">=3.10"`. The continuous
   integration matrix runs 3.11 and 3.12 only, so an installation on 3.10 would
   fail at IMPORT time with nothing in the pipeline to catch it. The existing
   `(str, Enum)` pattern is 3.10-safe and equally JSON-serialisable.
   `test_capability_contract.py` pins this so the floor cannot drift silently.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MetricStatus(str, Enum):
    """Why a metric does or does not carry a value.

    Inheriting from str keeps these JSON-serialisable without a custom encoder,
    which matters because these records end up in run manifests. See the module
    docstring for why this is not `StrEnum`.

    The first six members are the original vocabulary and their VALUES ARE
    LOAD-BEARING: existing run manifests on disk contain these strings, and
    changing one would silently orphan every historical record.
    """

    # --- original vocabulary, values frozen -------------------------------
    OK = "ok"
    UNDEFINED = "undefined"                                # mathematically undefined
    INSUFFICIENT_SUPPORT = "insufficient_support"          # too few observations/overlap
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"      # package not installed
    COMPUTATIONALLY_DEFERRED = "computationally_deferred"  # refused on cost, before allocating
    FAILED = "failed"                                      # raised during computation

    # --- added 2026-07-21 -------------------------------------------------
    # The evaluator itself does not exist. RESERVED for that case alone: once a
    # panel is implemented but its required head or output is missing,
    # INSUFFICIENT_SUPPORT is more informative, because it says the machinery is
    # ready and the science is not.
    NOT_IMPLEMENTED = "not_implemented"
    # The metric is meaningless for this observation, not merely unmeasurable:
    # a protein-level score for a variant with no protein product.
    NOT_APPLICABLE = "not_applicable"
    # Distinct from INSUFFICIENT_SUPPORT: the cohort is admissible and the
    # machinery ready, but there are too few rows or too few positives to
    # estimate anything. "One positive" is insufficient DATA; "the target is an
    # input feature" is insufficient SUPPORT.
    INSUFFICIENT_DATA = "insufficient_data"


class CapabilityState(str, Enum):
    """How far a capability has progressed, independent of its target.

    IMPLEMENTED_NO_OUTPUT is the state most easily forgotten and the reason a
    two-state model is not enough: a head can exist, be importable, be tested,
    and still have produced nothing to evaluate.
    """

    NOT_IMPLEMENTED = "not_implemented"
    IMPLEMENTED_NO_OUTPUT = "implemented_no_output"
    OUTPUT_AVAILABLE = "output_available"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"


class TargetState(str, Enum):
    """Whether the thing being predicted is scientifically admissible.

    CONTAMINATED is the state that makes this enum necessary. Gene ranking has a
    working capability AND an available target -- `clingen_validity_score` -- and
    is still inadmissible, because that column is one of the model's own input
    features. Ranking genes with a model handed ClinGen's verdict and then
    scoring the ranking against ClinGen is circular. Encoding that in a reason
    STRING, as an earlier draft did, leaves it unenforceable; encoding it as a
    typed state lets the constructor refuse.
    """

    ABSENT = "absent"                        # no target data exists at all
    CONTAMINATED = "contaminated"            # target is, or derives from, an input feature
    INSUFFICIENT_DATA = "insufficient_data"  # target exists but is too sparse
    PROVISIONAL = "provisional"              # target exists; provenance unverified
    ADMISSIBLE = "admissible"                # audited and usable


# Machine-readable reasons. Strings rather than an enum because callers add
# domain-specific reasons, but the shared ones are named so they cannot be
# misspelled into a reason nobody greps for.
REASON_NO_REGRESSION_FRAMEWORK = "no_regression_framework"
REASON_NO_REGRESSION_TARGETS = "no_regression_targets"
REASON_NO_MULTILABEL_HEAD = "no_multilabel_head"
REASON_NO_DISEASE_LABEL_INGESTION = "no_disease_label_ingestion"
REASON_TARGET_IS_AN_INPUT_FEATURE = "target_is_an_input_feature"
REASON_TARGET_PROXY_IS_AN_INPUT_FEATURE = "target_proxy_is_an_input_feature"
REASON_TARGET_PROVENANCE_UNVERIFIED = "target_provenance_unverified"
REASON_NO_TEMPORAL_HOLDOUT = "no_temporal_holdout"
REASON_TARGET_COVERAGE_TOO_LOW = "target_coverage_too_low"
REASON_NO_OUTPUT_ARTIFACT = "no_output_artifact"
REASON_DISEASE_INFORMED_GRAPH = "disease_informed_graph_contaminates_target"
REASON_KNOWLEDGE_CUTOFF_UNSAFE = "knowledge_cutoff_not_safe"


@dataclass(frozen=True)
class CapabilityEvidence:
    """What a panel is entitled to claim, and the evidence for it.

    THE INVARIANT IS ENFORCED IN THE CONSTRUCTOR, not at gate time. An OK
    capability that is not validated, or whose target is not admissible, or
    which names no output artifact, CANNOT BE BUILT. Neither can a non-OK
    capability without a reason.

    That placement is the whole point. A gate that checks at decision time can
    be bypassed by a caller who forgets to consult it -- which is exactly how
    `assert_data_usable` came to be a well-tested function that nothing called.
    An invariant in `__post_init__` cannot be forgotten, because there is no
    path to the object that avoids it.
    """

    capability_name: str
    capability_state: CapabilityState
    target_state: TargetState
    output_artifact: Optional[str]
    target_manifest: Optional[str]
    status: MetricStatus
    reason: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.capability_state, CapabilityState):
            raise TypeError(
                f"capability_state must be a CapabilityState, got "
                f"{type(self.capability_state).__name__}. A bare string cannot be "
                "checked against the enum and would let an unknown state pass.")
        if not isinstance(self.target_state, TargetState):
            raise TypeError(
                f"target_state must be a TargetState, got "
                f"{type(self.target_state).__name__}")
        if not isinstance(self.status, MetricStatus):
            raise TypeError(
                f"status must be a MetricStatus, got {type(self.status).__name__}")
        if not self.capability_name:
            raise ValueError("capability_name must be a non-empty string")

        if self.status is MetricStatus.OK:
            if self.capability_state is not CapabilityState.VALIDATED:
                raise ValueError(
                    f"{self.capability_name}: MetricStatus.OK requires "
                    f"CapabilityState.VALIDATED, got "
                    f"{self.capability_state.value}. A capability that has not "
                    "passed validation cannot report a passing metric, because "
                    "'it did not fail' is not the same as 'it was tested'.")
            if self.target_state is not TargetState.ADMISSIBLE:
                raise ValueError(
                    f"{self.capability_name}: MetricStatus.OK requires "
                    f"TargetState.ADMISSIBLE, got {self.target_state.value}. A "
                    "result measured against an absent, contaminated or "
                    "unverified target is not evidence, however good the number.")
            if not self.output_artifact:
                raise ValueError(
                    f"{self.capability_name}: MetricStatus.OK requires a named "
                    "output artifact. A pass with nothing to point at cannot be "
                    "reproduced or audited.")
        elif not self.reason:
            raise ValueError(
                f"{self.capability_name}: status {self.status.value} requires a "
                "machine-readable reason. 'Not OK' without a reason forces every "
                "reader to guess whether the capability is missing, the target is "
                "contaminated, or the cohort is too small -- three situations "
                "demanding three different responses.")

    def to_dict(self) -> dict:
        return {"capability_name": self.capability_name,
                "capability_state": self.capability_state.value,
                "target_state": self.target_state.value,
                "output_artifact": self.output_artifact,
                "target_manifest": self.target_manifest,
                "status": self.status.value,
                "reason": self.reason}


def release_gate_satisfied(evidence: CapabilityEvidence) -> bool:
    """Whether this capability satisfies a release gate.

    ONLY MetricStatus.OK satisfies. Every other status is UNSATISFIED -- not
    skipped, not waived, not passed. The distinction that matters in a report is
    between NOT SATISFIED and FAILED: a panel awaiting its head has not failed,
    but it has not passed either, and a release summary must not render it green
    or omit it.

    Note this reads only `status`, and it can afford to: the constructor has
    already guaranteed that OK implies validated, admissible and artifacted. The
    checking is done where it cannot be skipped.
    """
    if not isinstance(evidence, CapabilityEvidence):
        raise TypeError(
            f"release_gate_satisfied expects CapabilityEvidence, got "
            f"{type(evidence).__name__}. Passing a dict would compare a string "
            "to an enum and quietly return False for a passing capability.")
    return evidence.status is MetricStatus.OK


def summarize_release(evidences) -> dict:
    """A release summary in which an unsatisfied gate cannot hide.

    Returns counts plus the full list of unsatisfied capabilities with their
    reasons, so a caller cannot report 'all panels green' by iterating only the
    ones that passed.
    """
    items = list(evidences)
    satisfied = [e for e in items if release_gate_satisfied(e)]
    unsatisfied = [e for e in items if not release_gate_satisfied(e)]
    return {
        "n_capabilities": len(items),
        "n_satisfied": len(satisfied),
        "n_unsatisfied": len(unsatisfied),
        "release_complete": len(items) > 0 and not unsatisfied,
        "satisfied": [e.capability_name for e in satisfied],
        "unsatisfied": [{"capability_name": e.capability_name,
                         "capability_state": e.capability_state.value,
                         "target_state": e.target_state.value,
                         "status": e.status.value,
                         "reason": e.reason} for e in unsatisfied],
    }
