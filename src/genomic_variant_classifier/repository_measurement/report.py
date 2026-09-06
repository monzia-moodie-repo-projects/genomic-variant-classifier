"""The complete measurement object, from which both renderings derive.

ADR-0005.

MeasurementMode and Verdict are sited HERE rather than in a `types` module.
The governing ruling's section 1 enumerates the package as six files --
__init__, corpus, evidence, claims, report, serialization -- while its section
11 shows an illustrative import reading `from .types import MeasurementMode,
Verdict`, naming a seventh module the file list does not contain. The explicit
file list governs; the discrepancy is recorded here rather than resolved
silently.

ONE OBJECT, TWO RENDERINGS. A JSON payload and a terminal report must derive
from this object. Independently assembled prose and dict eventually disagree,
and then nobody can say which was the measurement.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .claims import MeasurementClaim
from .corpus import CorpusSnapshot
from .evidence import AnalysisCoverage, EvidenceItem


class MeasurementMode(str, Enum):
    """What kind of question the measurement asked.

    A search can inspect the correct corpus and still support the wrong
    inference, so corpus semantics alone are insufficient.
    """

    CENSUS = "census"
    PREDICATE = "predicate"
    DISCOVERY = "discovery"


class Verdict(str, Enum):
    """An adjudication, where one was requested.

    NOT_JUDGED is not PASS. The coverage probe prints `[ -- ]` for commits it
    does not judge, which is the right idea rendered as a display convention;
    this makes it a type, so it cannot collapse into truthiness.
    """

    PASS = "pass"
    FAIL = "fail"
    NOT_JUDGED = "not_judged"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MeasurementContext:
    """Environment attributes, only where the proposition depends on them.

    Deliberately not an environment dump. A full dependency freeze, hostname
    and hardware inventory add noise and machine identity to a result that
    usually does not depend on them. Context must be justified by the
    proposition being measured.
    """

    attributes: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        keys = [k for k, _v in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate context attribute keys")


@dataclass(frozen=True)
class MeasurementResult:
    """A repository observation, with its scope and its limits attached."""

    corpus: CorpusSnapshot
    mode: MeasurementMode
    claim: MeasurementClaim
    evidence: Tuple[EvidenceItem, ...]

    coverage: AnalysisCoverage | None = None
    verdict: Verdict | None = None
    complete_census: bool = False
    context: MeasurementContext | None = None

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(
                "measurement result must contain evidence; a bare verdict is "
                "an assertion")
        if self.mode is MeasurementMode.PREDICATE and self.verdict is None:
            raise ValueError("predicate measurement requires a verdict")
        if (self.mode is not MeasurementMode.PREDICATE
                and self.verdict in (Verdict.PASS, Verdict.FAIL)):
            raise ValueError(
                "PASS/FAIL belongs to a predicate measurement; a descriptive "
                "census must not fabricate a verdict in order to look "
                "complete")
        if self.complete_census:
            if self.mode is not MeasurementMode.CENSUS:
                raise ValueError(
                    "complete_census is only valid for CENSUS measurements")
            if self.coverage is None or not self.coverage.complete:
                raise ValueError(
                    "complete census requires complete analysis coverage; "
                    "parse failures cannot coexist with a complete census")

    def render(self) -> str:
        """The human view, derived from this object and nothing else."""
        c = self.corpus
        out = ["MEASUREMENT", "  mode          {}".format(self.mode.value), "",
               "CORPUS",
               "  kind          {}".format(c.spec.kind.value),
               "  selector      {}".format(c.spec.selector),
               "  enumerator    {}".format(c.spec.enumerator),
               "  members       {}".format(c.n_members),
               "  identity      {}".format(c.membership_sha256[:16])]
        if c.repository_head is not None:
            label = ("base head" if c.spec.kind.value == "worktree"
                     else "head")
            out.append("  {:<13} {}".format(label, c.repository_head))
        if c.worktree_dirty:
            out.append("  worktree      DIRTY -- uncommitted bytes are not "
                       "part of that commit")
        if c.selection is not None and c.selection.missing_roots:
            out.append("  MISSING ROOTS {}".format(
                ", ".join(c.selection.missing_roots)))
        if self.coverage is not None:
            out += ["", "COVERAGE",
                    "  selected      {}".format(self.coverage.selected),
                    "  attempted     {}".format(self.coverage.attempted),
                    "  succeeded     {}".format(self.coverage.succeeded),
                    "  failed        {}".format(self.coverage.failed),
                    "  complete      {}".format(
                        "yes" if self.coverage.complete else "NO")]
        out += ["", "PROVES"] + ["  - {}".format(p) for p in self.claim.proves]
        out += ["", "DOES NOT PROVE"] + [
            "  - {}".format(p) for p in self.claim.does_not_prove]
        out += ["", "EVIDENCE"] + [
            "  [{}] {}".format(e.strength.value, e.statement)
            for e in self.evidence]
        out += ["", "VERDICT",
                "  {}".format(self.verdict.value if self.verdict is not None
                              else "none -- descriptive measurement")]
        return "\n".join(out)
