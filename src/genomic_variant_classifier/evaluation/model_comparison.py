"""The model-comparison artifact.

TWO CLAIMS, KEPT APART
======================
A comparison needs to state two different things, and one field cannot carry
both:

    1. these models were evaluated on the SAME ROWS within this comparison
    2. those rows are EXTERNALLY IDENTIFIED and reproducible across artifacts

`population_fingerprint` establishes claim 2 when the cohort is attributed. It
must not be overloaded to establish claim 1 when the cohort is unnamed, because
that would mean minting an identity nobody supplied -- the sentinel error already
ruled out for `EvaluationPopulation`, reintroduced one layer up.

So claim 1 is established STRUCTURALLY: `compare_models` constructs one
population and hands the same object to every model. There is no opportunity for
one model to receive a different mask, scope or frame, and the evidence is the
construction history rather than a digest.

WHY `compare_membership` IS NOT USED FOR CLAIM 1
-------------------------------------------------
`EvaluationPopulation.compare_membership` correctly returns UNKNOWN for two
unattributed results. It answers: "can these independently represented
populations be PROVEN equal from their provenance?" -- and they cannot.

The comparison layer knows something stronger from construction: it supplied the
same object to both. Those are different evidence channels, not a contradiction,
and teaching `compare_membership` that two unattributed populations are equal
would destroy the only honest answer it has.

WHY THE RANKING IS REFUSED RATHER THAN FILTERED
------------------------------------------------
Measured 2026-07-28, before this module existed: with one corrupt model,
`compare_models` returned a complete three-row table sorted by area under the
receiver operating characteristic curve. The corrupt model sorted to the bottom
on a NaN comparison, and nothing in the table said it had never been evaluated.
A reader saw `good > fair > corrupt` and could not distinguish "evaluated and
worst" from "not evaluated at all".

Omitting the corrupt model would be no better: a ranking that silently excludes a
submitted model is not a ranking of the models submitted.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "ComparisonBlocker",
    "ComparisonPopulationRelation",
    "ModelComparison",
]

COMPARISON_SCHEMA_VERSION = 1


class ComparisonPopulationRelation(str, Enum):
    """How the populations behind a comparison relate to one another.

    SHARED_BY_CONSTRUCTION and VERIFIED_BY_FINGERPRINT are deliberately
    distinct. The first says one object was handed to every model in one call --
    proof of intra-call sameness that needs no identity. The second says the rows
    are externally identified, which is what lets two SEPARATE artifacts be
    compared. A comparison can have the first without the second, and saying so
    is the whole point.
    """

    SHARED_BY_CONSTRUCTION = "shared_by_construction"
    VERIFIED_BY_FINGERPRINT = "verified_by_fingerprint"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


class ComparisonBlocker(str, Enum):
    """Why a comparison could not be ranked. A CONTROLLED vocabulary.

    `INVALID_RANKING_METRIC` names the direct cause: the metric the ranking is
    built on is not valid for some model. It deliberately does NOT say
    "invalid model output", because a future ranking on a different metric might
    be perfectly possible for the same reports.
    """

    INVALID_RANKING_METRIC = "invalid_ranking_metric"
    DIFFERENT_POPULATIONS = "different_populations"
    POPULATION_IDENTITY_MISSING = "population_identity_missing"
    INCOMPLETE_MODEL_SET = "incomplete_model_set"


# Windows reserves these names with or without an extension, so `nul.anything`
# still addresses the device. Compared case-insensitively on the stem alone.
_WINDOWS_RESERVED = {"nul", "con", "prn", "aux"} | {f"com{i}" for i in range(1, 10)} \
    | {f"lpt{i}" for i in range(1, 10)}


def _is_null_device(path: Path) -> bool:
    """Is this path a null device rather than a file the caller wants?"""
    if str(path) == os.devnull:
        return True
    return path.name.split(".")[0].lower() in _WINDOWS_RESERVED


@dataclass(frozen=True)
class ModelComparison:
    """A comparison, with its own schema rather than columns bolted onto a frame.

    The table alone cannot carry comparison-level facts: duplicating the
    population relation across every model row is not a schema, and a
    comma-separated-values file has nowhere else to put it. Measured 2026-07-28,
    the comparison artifact has NO consumers -- the only reference to `output_csv`
    outside `compare_models` is a test passing the null device -- so the eleven
    legacy columns are preserved on grounds of churn rather than compatibility.
    """

    table: pd.DataFrame
    ranking_metric: str
    comparison_rankable: bool
    comparison_blocked_by: Optional[ComparisonBlocker]
    blocked_models: Tuple[str, ...]
    population_relation: ComparisonPopulationRelation
    comparison_population_key: str
    population_source_id: Optional[str]
    population_fingerprint: Optional[str]
    comparison_is_like_for_like: bool
    population_is_attributed: bool
    comparison_certification_eligible: bool
    n_models: int
    population_n: int
    schema_version: int = COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # THE THREE AXES MAY NOT BE COLLAPSED.
        #
        # An unattributed shared comparison is (True, False, False): internally
        # valid, externally unreproducible. Reporting a single boolean would say
        # the comparison is invalid, which is false and would discourage a
        # perfectly sound exploratory analysis.
        if self.comparison_certification_eligible:
            if not self.comparison_is_like_for_like:
                raise ValueError(
                    "a comparison cannot be certifiable while its models were "
                    "not evaluated over one population")
            if not self.population_is_attributed:
                raise ValueError(
                    "a comparison cannot be certifiable over an unattributed "
                    "population: a certified claim asserts something about a "
                    "NAMED set of rows")
            if not self.comparison_rankable:
                raise ValueError(
                    "a comparison cannot be certifiable while its ranking is "
                    "refused")
        if self.comparison_rankable and self.comparison_blocked_by is not None:
            raise ValueError(
                "a rankable comparison must not name a blocker")
        if not self.comparison_rankable and self.comparison_blocked_by is None:
            raise ValueError(
                "a refused ranking must name WHY; an unexplained refusal is "
                "indistinguishable from a defect")
        if self.population_is_attributed != (self.population_source_id is not None):
            raise ValueError(
                "attribution and the source identity must agree; one without "
                "the other means the artifact contradicts itself")

    def metadata(self) -> dict:
        """The comparison-level facts, for the sidecar.

        Separate from the table because these describe the COMPARISON, not any
        model in it. Duplicating them per row would invite a reader to believe
        they could differ between rows, which they cannot.
        """
        return {
            "comparison_schema_version": self.schema_version,
            "ranking_metric": self.ranking_metric,
            "comparison_rankable": self.comparison_rankable,
            "comparison_blocked_by": (self.comparison_blocked_by.value
                                      if self.comparison_blocked_by else None),
            "blocked_models": list(self.blocked_models),
            "population_relation": self.population_relation.value,
            "comparison_population_key": self.comparison_population_key,
            "population_source_id": self.population_source_id,
            "population_fingerprint": self.population_fingerprint,
            "comparison_is_like_for_like": self.comparison_is_like_for_like,
            "population_is_attributed": self.population_is_attributed,
            "comparison_certification_eligible": self.comparison_certification_eligible,
            "n_models": self.n_models,
            "population_n": self.population_n,
        }

    def write_csv(self, path) -> Tuple[Path, Path]:
        """Write the compatibility table and the versioned metadata sidecar.

        Two files, because they answer different questions and have different
        shapes. The table is per-model; the metadata is per-comparison.
        """
        table_path = Path(path)

        # THE NULL DEVICE TAKES NO SIDECAR (2026-07-28).
        #
        # A caller discarding the table -- `output_csv=os.devnull` -- is asking
        # for no artifact at all, and a metadata file beside a discarded table is
        # meaningless. Worse, on Windows `os.devnull` is `nul`, a RESERVED DEVICE
        # NAME with no suffix, so `with_suffix` produced `nul.metadata.json` in
        # the working directory: an entry that appears in a directory listing and
        # CANNOT BE OPENED. Git saw it, could not index it, and `git add -A`
        # failed with "unable to index file".
        #
        # The full suite passed. Only version control caught it, because a test
        # that writes to the null device never reads back what it wrote.
        if _is_null_device(table_path):
            self.table.to_csv(table_path, index=False)
            logger.debug("comparison written to the null device; no sidecar")
            return table_path, None

        table_path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(table_path, index=False)

        sidecar = table_path.with_suffix(table_path.suffix + ".metadata.json")
        sidecar.write_text(json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        return table_path, sidecar
