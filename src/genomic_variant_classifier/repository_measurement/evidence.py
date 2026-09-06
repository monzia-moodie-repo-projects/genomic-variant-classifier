"""How completely an analysis processed what was selected, and how strong its
evidence is.

ADR-0005.

MEASURED 2026-09-05, the failure this prevents: a false-positive rate
calibrated on EIGHT markdown documents was applied to 1,637 tracked files.
Recorded as `selected 1637, attempted 8`, that cannot masquerade as a complete
census; recorded as prose, it did.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceStrength(str, Enum):
    """Where a statement sits on the inferential ladder.

    DISCOVERY evidence cannot license an absence claim without a separate
    completeness argument. That single rule would have prevented:

        grep found no "import audit_data_tree"
        therefore runtime invocation count = 0

    The grep was useful. The promoted inference was not.
    """

    DIRECT = "direct"
    DERIVED = "derived"
    DISCOVERY = "discovery"


@dataclass(frozen=True)
class EvidenceItem:
    statement: str
    strength: EvidenceStrength
    basis: str

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("evidence statement must be non-empty")
        if not self.basis.strip():
            raise ValueError(
                "evidence basis must be non-empty: a statement with no stated "
                "basis is an assertion, not evidence")


@dataclass(frozen=True)
class AnalysisCoverage:
    """Did analysis successfully process every enumerated member?"""

    selected: int
    attempted: int
    succeeded: int
    failed: int

    def __post_init__(self) -> None:
        for name, value in (("selected", self.selected),
                            ("attempted", self.attempted),
                            ("succeeded", self.succeeded),
                            ("failed", self.failed)):
            if value < 0:
                raise ValueError("{} must be >= 0".format(name))
        if self.succeeded + self.failed != self.attempted:
            raise ValueError(
                "attempted must equal succeeded + failed; got {} != {} + {}"
                .format(self.attempted, self.succeeded, self.failed))
        if self.attempted > self.selected:
            raise ValueError(
                "attempted ({}) cannot exceed selected ({})"
                .format(self.attempted, self.selected))

    @property
    def fully_attempted(self) -> bool:
        return self.attempted == self.selected

    @property
    def complete(self) -> bool:
        return self.attempted == self.selected and self.failed == 0


class IncompleteMeasurementError(RuntimeError):
    """Raised where a universal negative would rest on incomplete analysis."""


def require_complete_census(coverage: AnalysisCoverage) -> None:
    """Call before any universal negative conclusion.

    "No instances found" over an incomplete analysis is not "no instances
    exist".
    """
    if not coverage.complete:
        raise IncompleteMeasurementError(
            "analysis is incomplete -- selected {}, attempted {}, failed {} "
            "-- so a universal absence claim is not licensed."
            .format(coverage.selected, coverage.attempted, coverage.failed))
