"""Whether a feature-drift assessment may happen at all. Not whether it found drift.

Created 2026-08-24.

WHY THIS MODULE EXISTS
======================
`.github/workflows/drift_monitor.yml` ran a monthly cron that invoked
`scripts/run_drift_monitor.py` with neither `--new-data` nor `--new-clinvar`.
MEASURED against `run_drift_monitor.py:313-351`: that combination takes the else
branch and returns `EXIT_NOT_CHECKED`. The job could not reach a verdict by
construction, every month, for the life of the workflow.

The workflow's own label-drift step already demonstrates the honest alternative:
it declares `LABEL DRIFT WAS NOT CHECKED`, calls it "a KNOWN LIMITATION ...
REPORTED, not laundered into a green tick", echoes the runnable command, and
invokes nothing. This module lets the feature-drift step do the same, with a
typed reason rather than prose.

WHAT THIS MODULE DOES NOT DO
============================
It does not assess drift. `drift_detector.py` owns that, and owns
`FeatureDriftResult`, `DriftReport` and the population-stability-index
thresholds with it.

It does not discover an observation population. MEASURED 2026-08-24 across 1,622
tracked files: `ObservationCohort`, `CohortRecord`, `cohort_id`,
`candidate_population`, `new_observation`, `production_cohort`,
`inference_batch` and `CandidatePopulation` occur ZERO times, and the 42 lines
combining a discovery verb with a population noun all glob data-source
artifacts, sequence-window shards, or training splits. Nothing discovers a new
observation population.

It does not reuse `AdaptiveRetrainingInputs`. That is a fail-closed
ADAPTATION gate; coupling "can we measure drift?" to "may we retrain and promote
a replacement?" is backwards, and we want the state where drift is measurable
while retraining stays forbidden.

WHY THE PRESENT STATE IS UNDETERMINED AND NOT NOT_READY
=======================================================
`NOT_READY` says the system evaluated readiness and found the population
unsuitable or absent. **Candidate discovery does not exist, so the readiness
question could not be evaluated at all.**

Encoding `NO_NEW_OBSERVATION_POPULATION` would assert a fact about the world.
`CANDIDATE_DISCOVERY_NOT_IMPLEMENTED` asserts a fact about this repository,
which is the only thing measured. That distinction is the same one
`EXIT_NOT_CHECKED` draws against `0`, one layer up.

OWNERSHIP OF FACTS
==================
    Repository capability code   may state whether capability exists.
    Discovery authority          may state whether an observation exists.
    Admission code               may state whether populations are comparable.
    Assessment code              may state whether drift exists.
    Policy code                  may state whether action is required.

No layer may author a fact owned by a downstream layer. This module owns only
the first line, which is why `NO_NEW_OBSERVATION_POPULATION` exists in the
vocabulary but is emitted by nothing here: only a discovery implementation may
claim it.

The same holds for the four ADMISSION reasons added on 2026-08-27 --
`REFERENCE_REPRESENTATION_UNIDENTIFIED`, `REPRESENTATION_MISMATCH`,
`SOURCE_RELEASE_DIVERGENT` and `POPULATION_UNATTRIBUTED`. They name states the
third layer may report; nothing here emits them, and a test proves each one is
unemitted. Adding them now removes a future migration without asserting a
verdict this layer cannot reach.

DEPENDENCIES
============
STANDARD LIBRARY ONLY, and a test enforces it. `drift_detector.py` imports
numpy, pandas, scipy and scipy.spatial.distance at module level, and
`drift_reference_profile.py` imports numpy and pandas at module level. A
readiness check that must run on a hosted runner should not drag the scientific
stack in to report that it has nothing to measure.

Acronyms: JSON = JavaScript Object Notation; UTC = Coordinated Universal Time.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

__all__ = [
    "READINESS_SCHEMA_VERSION",
    "DriftReadinessStatus",
    "DriftReadinessReason",
    "DriftReadiness",
    "current_feature_drift_readiness",
    "as_document",
    "render_json",
    "render_github_output_lines",
]

#: Versioned from the first commit. Schema drift has already cost this project
#: an evaluation-report migration; a record without a schema owner acquires one
#: retrospectively, which is how a reader ends up guessing.
READINESS_SCHEMA_VERSION = 1

#: The legacy `drift_level` token. `UNKNOWN` conflates "no verdict" with a
#: severity, which is exactly the overload this module exists to stop. It
#: survives ONLY as a compatibility projection at the adapter boundary, and a
#: test asserts it is absent from every domain enumeration.
_LEGACY_DRIFT_LEVEL_WHEN_UNMEASURED = "UNKNOWN"


class DriftReadinessStatus(str, Enum):
    """Whether an assessment may proceed -- never what an assessment found.

    `(str, Enum)` rather than `StrEnum`: MEASURED 2026-08-24 across `src/`,
    79 classes use `(str, Enum)` and 0 use `StrEnum`. A new module follows the
    project, not a preference.
    """

    #: Discovery ran, found an admissible population, and admission passed.
    #: NOT constructible by anything in this module today -- see
    #: `current_feature_drift_readiness`. The member exists because the type is
    #: reused when discovery lands, and a type that must be rewritten to
    #: express success is a type that will be.
    READY = "ready"

    #: Readiness WAS evaluated and the answer is no.
    NOT_READY = "not_ready"

    #: Readiness could not be evaluated. Distinct from NOT_READY, and the
    #: distinction is the whole point.
    UNDETERMINED = "undetermined"


class DriftReadinessReason(str, Enum):
    """Why an assessment may not proceed.

    Each member names the LAYER entitled to emit it. A reason emitted by a
    layer that cannot know it is a fabricated fact.
    """

    #: Emitted by repository-capability code -- this module. States that no
    #: authoritative discovery mechanism exists, NOT that no population does.
    CANDIDATE_DISCOVERY_NOT_IMPLEMENTED = "candidate_discovery_not_implemented"

    #: Emitted ONLY by a discovery authority that looked and found nothing.
    #: Nothing in this module emits it, and a test proves that.
    NO_NEW_OBSERVATION_POPULATION = "no_new_observation_population"

    # --- the ADMISSION layer, added 2026-08-27 ---------------------------
    #
    # Third of the five layers named above: "Admission code may state whether
    # populations are comparable." That layer is reserved and unoccupied, and
    # these members are present for the reason this module already gives for
    # NO_NEW_OBSERVATION_POPULATION -- "its absence from the vocabulary would
    # force a future migration; its emission here would be a claim no layer at
    # this level owns."
    #
    # NOTHING IN THIS MODULE EMITS ANY OF THEM, and a test proves each one.
    #
    # They describe the reference as MEASURED on 2026-08-27:
    # `data/reference/drift/run15_reference_profile.json` is 1,089,400 bytes,
    # format_version 1, 78 features over 1,038,974 rows, and its entire
    # provenance is one field -- `source`, holding a machine-local path,
    # `outputs\run15_rerun_report\full\splits\X_train.parquet`. A path names
    # where a file sat, not what was in it.

    #: Emitted ONLY by admission code. The reference exists and is
    #: PSI-verified against its own matrix, but records no representation: no
    #: ordered feature contract, no preprocessing policy digest, no source
    #: manifest. Nothing can be shown to inhabit the same representation as
    #: something that does not state one.
    REFERENCE_REPRESENTATION_UNIDENTIFIED = "reference_representation_unidentified"

    #: Emitted ONLY by admission code. Both sides state a representation and
    #: the two differ -- a reordered column, a substituted feature, a changed
    #: missingness policy. Distinct from the above: here the comparison is
    #: refusable BECAUSE both sides were identified.
    REPRESENTATION_MISMATCH = "representation_mismatch"

    #: Emitted ONLY by admission code. The populations are comparable and the
    #: SOURCE RELEASES are not -- same variants, a new dbNSFP or gnomAD
    #: release, values that moved because the measurement process changed. A
    #: distribution shift attributable to a release is not population drift,
    #: and reporting it as such would be a scientific error.
    SOURCE_RELEASE_DIVERGENT = "source_release_divergent"

    #: Emitted ONLY by admission code. A population was discovered and cannot
    #: be identified -- no membership fingerprint -- so "the same rows" is
    #: unprovable. `EvaluationPopulation` returns None rather than a digest for
    #: exactly this state, because a digest of nothing would let two
    #: populations of unknown equivalence compare equal.
    POPULATION_UNATTRIBUTED = "population_unattributed"


@dataclass(frozen=True)
class DriftReadiness:
    """A readiness verdict, with the two states kept apart by construction."""

    status: DriftReadinessStatus
    reason: Optional[DriftReadinessReason]

    def __post_init__(self) -> None:
        if self.status is DriftReadinessStatus.READY:
            if self.reason is not None:
                raise ValueError(
                    "READY carries a refusal reason {!r}. A verdict that "
                    "permits assessment cannot also explain why it does not."
                    .format(self.reason.value))
        elif self.reason is None:
            raise ValueError(
                "{} requires a reason. A refusal without one is the state "
                "this module exists to make unrepresentable."
                .format(self.status.value))

    @property
    def checked(self) -> bool:
        """Whether a feature-drift assessment may be performed.

        Never whether one WAS performed: that is a measurement status, and it
        belongs to the assessment layer.
        """
        return self.status is DriftReadinessStatus.READY


def current_feature_drift_readiness() -> DriftReadiness:
    """The measured capability state of this repository.

    Returns UNDETERMINED because candidate discovery is not implemented. This
    does NOT claim that no new observation population exists -- nothing here is
    entitled to know that, and asserting it would be the same class of error as
    reporting `0` for a run that measured nothing.

    When discovery lands, this function is replaced rather than extended, and
    the test naming its reason is deleted and replaced by one proving
    CANDIDATE_DISCOVERY_NOT_IMPLEMENTED is no longer emittable.
    """
    return DriftReadiness(
        status=DriftReadinessStatus.UNDETERMINED,
        reason=DriftReadinessReason.CANDIDATE_DISCOVERY_NOT_IMPLEMENTED,
    )


def as_document(readiness: DriftReadiness) -> dict:
    """The canonical record. Every adapter projects from THIS, never from state.

    One producer, so `checked=false` beside `drift_level=none` cannot arise --
    a combination the previous workflow could author because three fields were
    written independently.
    """
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "readiness_status": readiness.status.value,
        "feature_drift_checked": readiness.checked,
        "not_checked_reason": (
            readiness.reason.value if readiness.reason is not None else None),
    }


def render_json(readiness: DriftReadiness) -> str:
    """Deterministic and diffable, with a terminal newline.

    Sorted keys and fixed indentation are fully deterministic; a compact
    separator form is no more so and turns a durable record into one
    unreadable line.
    """
    return json.dumps(as_document(readiness), indent=2, sort_keys=True,
                      ensure_ascii=True) + "\n"


def render_github_output_lines(readiness: DriftReadiness) -> Tuple[str, ...]:
    """Project the record into the workflow's vocabulary. Adapter, not author.

    `drift_level` receives the legacy `UNKNOWN` token whenever nothing was
    measured. That token is a COMPATIBILITY SPELLING: it exists in the
    workflow's five-value vocabulary and in no domain enumeration here.
    """
    document = as_document(readiness)
    level = ("" if readiness.checked
             else _LEGACY_DRIFT_LEVEL_WHEN_UNMEASURED)
    reason = document["not_checked_reason"] or ""
    return (
        "readiness_status={}".format(document["readiness_status"]),
        "feature_drift_checked={}".format(
            "true" if document["feature_drift_checked"] else "false"),
        "not_checked_reason={}".format(reason),
        "drift_level={}".format(level),
    )


def validate_document(document: Mapping[str, object]) -> None:
    """Refuse a record that is not this schema.

    Not decoration: an adapter that hand-builds "roughly the same" dictionary
    is how two producers of one record diverge, which this project has already
    paid for once.
    """
    expected = {"schema_version", "readiness_status", "feature_drift_checked",
                "not_checked_reason"}
    actual = set(document)
    if actual != expected:
        raise ValueError(
            "readiness document keys {} differ from the schema by {}"
            .format(sorted(actual), sorted(actual ^ expected)))
    if document["schema_version"] != READINESS_SCHEMA_VERSION:
        raise ValueError(
            "readiness document declares schema_version {!r}; this module "
            "owns version {}".format(document["schema_version"],
                                     READINESS_SCHEMA_VERSION))
