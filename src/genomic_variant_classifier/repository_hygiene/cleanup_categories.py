"""Categories A-F, inspected and reported. Application is UNAVAILABLE.

Author: Monzia Moodie

WHAT THIS PRESERVES
===================
The six categories of `scripts/cleanup_apply.py` at
`f3392e9491e71198e20c709ddfa73b80945b117f81a598ab09edb64ca4d7a541`, read
line by line rather than reconstructed from descriptions:

    A  seven named .bak files, each with its live counterpart
       original: committed_head(live) and not tracked(bak)
    B  .gitignore.prebakfix
       original: committed_head(".gitignore") and not tracked(bak)
    C  data/processed/seq_windows/part_*.parquet and *.done
       original: merged seq_windows.parquet exists and is nonempty,
                 and the part is untracked
    D  data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak
       original: the live index exists, is nonempty, and the bak is untracked
    E  install_*.py at the repository root
       original: ignored(p) and not tracked(p)
    F  scripts/dump_*.py and scripts/patch_*.py
       original: not tracked(p)

WHAT THIS REPAIRS
=================
MEASURED 2026-09-08, running the original outside a Git working tree:

    ### (F)  2 eligible (4.0B), 0 SKIPPED ###

`tracked()` returned False whenever Git exited non-zero, so failed inspection
became eligibility. Category E escaped only because it also required
`ignored()`, which fails the same way but in the safe direction. Every
condition is now a tri-state observation, and UNKNOWN is never eligibility.

Paths are anchored to the SELECTED repository root. The original resolved
every path against the process's current directory.

A category predicate passing is NOT authorization. The wording says
"category conditions satisfied; application unavailable", never
"would-delete".

READ-ONLY, VERIFIED
===================
Every write in the original -- `Path(p).unlink()` at line 108,
`Path("outputs").mkdir` at 115 and `write_text` at 116 -- lies inside
`if APPLY`. Discovery, lines 30 to 80, mutates nothing. That was established
by locating each write, not inferred from the banner's "DRY-RUN".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .repository_inspection import (
    InspectionError, RepositoryInspection, TrackingState)


@dataclass(frozen=True)
class HeadObservation:
    """ONE resolution of the committed reference, shared by every category.

    MEASURED 2026-09-08: the read-only owner's docstring claimed HEAD was
    resolved once per inspection. It was not. Category A called
    `head_paths()`, category B called it AGAIN when its candidate existed,
    and `head_paths()` re-runs `rev-parse HEAD` every time -- so A and B
    could inspect different commits. The claim was a comment, not a
    mechanism.

    A FAILURE IS CARRIED, not raised at capture time, so a category can
    retain its discovered candidates with UNKNOWN conditions instead of
    dropping them.

    This freezes the COMMITTED REFERENCE only. The index and the filesystem
    are read separately and are not frozen by it.
    """

    commit: object
    paths: frozenset
    failure: object


def capture_head(inspection) -> HeadObservation:
    try:
        commit, paths = inspection.head_paths()
    except InspectionError as exc:
        return HeadObservation(None, frozenset(), str(exc)[:200])
    return HeadObservation(commit, frozenset(paths), None)


class InspectionState(Enum):
    """Explicit. `not failures` alone would let a category that never ran look
    identical to one that ran and found nothing."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NOT_RUN = "not_run"


class ConditionState(Enum):
    """The condition, distinct from the inspection process's completeness."""

    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Candidate:
    """A DISCOVERED path. Retained even when its conditions are unknown.

    Several category functions previously discovered a candidate, hit a Git
    failure and `continue`d without recording it: the category became
    incomplete, but its candidate count omitted a path already observed.
    Paths discovered and paths whose conditions could be determined are two
    populations, and both are now reported.
    """

    relative_path: str
    conditions: ConditionState
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.conditions, ConditionState):
            raise ValueError(
                "conditions must be a ConditionState, not {!r}".format(
                    self.conditions))


@dataclass(frozen=True)
class CategoryInspection:
    category: str
    state: InspectionState
    candidates: tuple = ()
    failures: tuple = ()

    def __post_init__(self) -> None:
        # TYPE first. MEASURED 2026-09-08: CategoryInspection("A", "complete")
        # constructed successfully because the identity checks below never
        # reject a string, and rendering would then fail at `.value`.
        if not isinstance(self.state, InspectionState):
            raise ValueError(
                "state must be an InspectionState, not {!r}".format(
                    self.state))
        if self.state is InspectionState.COMPLETE and self.failures:
            raise ValueError("a complete inspection cannot carry failures")
        if self.state is InspectionState.INCOMPLETE and not self.failures:
            raise ValueError("an incomplete inspection must name its failure")
        if self.state is InspectionState.NOT_RUN and (self.candidates
                                                      or self.failures):
            raise ValueError("a category that did not run has no results")


#: The seven category-A pairs, verbatim from the original.
CATEGORY_A_PAIRS = (
    ("scripts/train.py.w1bak", "scripts/train.py"),
    ("scripts/train.py.w2b2bak", "scripts/train.py"),
    ("src/genomic_variant_classifier/data/real_data_prep.py.w2b1bak",
     "src/genomic_variant_classifier/data/real_data_prep.py"),
    ("src/genomic_variant_classifier/data/split_protocol_v2.py.w2b1bak",
     "src/genomic_variant_classifier/data/split_protocol_v2.py"),
    ("src/genomic_variant_classifier/models/variant_ensemble.py.w2bak",
     "src/genomic_variant_classifier/models/variant_ensemble.py"),
    ("src/genomic_variant_classifier/evaluation/evaluator.py.bak",
     "src/genomic_variant_classifier/evaluation/evaluator.py"),
    ("src/genomic_variant_classifier/data/database_connectors.py.bak",
     "src/genomic_variant_classifier/data/database_connectors.py"),
)
SEQ_WINDOWS = "data/processed/seq_windows"
DBNSFP_BAK = "data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak"
DBNSFP_LIVE = "data/external/dbnsfp/dbnsfp_full_index.parquet"


def observed_size(path) -> int:
    """The size, or an inspection FAILURE.

    MEASURED 2026-09-08: the previous `_size` returned 0 for an object whose
    `stat()` raised PermissionError, so a category could report that a
    supporting artifact failed the nonempty condition while marking itself
    COMPLETE -- when the observation had never been available at all.
    """
    try:
        return path.stat().st_size
    except OSError as exc:
        raise InspectionError(
            "could not inspect size of {}: {}".format(path, exc)) from exc


def _tracking(inspection, relative, failures):
    """SATISFIED when Git established untracked; UNKNOWN records a failure."""
    observed = inspection.tracking(relative)
    if observed.state is TrackingState.UNKNOWN:
        failures.append("{}: tracking unknown: {}".format(relative,
                                                          observed.detail))
        return None
    return observed.state is TrackingState.UNTRACKED


def _verdict(*flags):
    """SATISFIED only if every flag is True; UNKNOWN if any is None."""
    if any(flag is None for flag in flags):
        return ConditionState.UNKNOWN
    return (ConditionState.SATISFIED if all(flags)
            else ConditionState.NOT_SATISFIED)


def inspect_category_a(root: Path, inspection, head=None) -> CategoryInspection:
    found, failures = [], []
    head = capture_head(inspection) if head is None else head
    if head.failure:
        failures.append("HEAD unavailable: {}".format(head.failure))
    for backup, live in CATEGORY_A_PAIRS:
        if not (root / backup).is_file():
            continue
        untracked = _tracking(inspection, backup, failures)
        # A CARRIED FAILURE keeps the candidate with UNKNOWN conditions.
        committed = None if head.failure else (live in head.paths)
        found.append(Candidate(
            backup, _verdict(committed, untracked),
            "live committed at HEAD={}, backup untracked={}".format(
                committed, untracked)))
    return _result("A", found, failures)


def inspect_category_b(root: Path, inspection, head=None) -> CategoryInspection:
    found, failures = [], []
    name = ".gitignore.prebakfix"
    if (root / name).is_file():
        head = capture_head(inspection) if head is None else head
        if head.failure:
            failures.append("HEAD unavailable: {}".format(head.failure))
        untracked = _tracking(inspection, name, failures)
        committed = None if head.failure else (".gitignore" in head.paths)
        found.append(Candidate(
            name, _verdict(committed, untracked),
            ".gitignore committed at HEAD={}".format(committed)))
    return _result("B", found, failures)


def inspect_category_c(root: Path, inspection, head=None) -> CategoryInspection:
    found, failures = [], []
    directory = root / SEQ_WINDOWS
    if not directory.is_dir():
        return _result("C", found, failures)
    merged = directory / "seq_windows.parquet"
    merged_ok = None
    if merged.is_file():
        try:
            merged_ok = observed_size(merged) > 0
        except InspectionError as exc:
            failures.append(str(exc)[:200])
    else:
        merged_ok = False
    try:
        parts = sorted(directory.glob("part_*.parquet")) + \
            sorted(directory.glob("*.done"))
    except OSError as exc:
        # A FAILED ENUMERATION cannot establish zero matches.
        return CategoryInspection(
            "C", InspectionState.INCOMPLETE,
            failures=("could not enumerate {}: {}".format(directory, exc),))
    for path in parts:
        relative = path.relative_to(root).as_posix()
        untracked = _tracking(inspection, relative, failures)
        found.append(Candidate(
            relative, _verdict(merged_ok, untracked),
            "merged artifact present and nonempty={}".format(merged_ok)))
    return _result("C", found, failures)


def inspect_category_d(root: Path, inspection, head=None) -> CategoryInspection:
    found, failures = [], []
    if (root / DBNSFP_BAK).is_file():
        live = root / DBNSFP_LIVE
        live_ok = None
        if live.is_file():
            try:
                live_ok = observed_size(live) > 0
            except InspectionError as exc:
                failures.append(str(exc)[:200])
        else:
            live_ok = False
        untracked = _tracking(inspection, DBNSFP_BAK, failures)
        found.append(Candidate(
            DBNSFP_BAK, _verdict(live_ok, untracked),
            "live index present and nonempty={}".format(live_ok)))
    return _result("D", found, failures)


def inspect_category_e(root: Path, inspection, head=None) -> CategoryInspection:
    found, failures = [], []
    try:
        candidates = sorted(root.glob("install_*.py"))
    except OSError as exc:
        return CategoryInspection(
            "E", InspectionState.INCOMPLETE,
            failures=("could not enumerate {}: {}".format(root, exc),))
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        observed = inspection.ignored(relative)
        if observed.state is TrackingState.UNKNOWN:
            failures.append("{}: ignore state unknown: {}".format(
                relative, observed.detail))
            is_ignored = None
        else:
            is_ignored = observed.state is TrackingState.TRACKED
        untracked = _tracking(inspection, relative, failures)
        found.append(Candidate(relative, _verdict(is_ignored, untracked),
                               "ignored={}".format(is_ignored)))
    return _result("E", found, failures)


def inspect_category_f(root: Path, inspection, head=None) -> CategoryInspection:
    """The category whose original condition was `ok = not tracked(p)`.

    MEASURED: outside a repository every candidate became eligible. UNKNOWN is
    now a recorded failure and an UNKNOWN condition, never eligibility.
    """
    found, failures = [], []
    scripts = root / "scripts"
    if not scripts.is_dir():
        return _result("F", found, failures)
    try:
        seen = []
        for pattern in ("dump_*.py", "patch_*.py"):
            seen.extend(sorted(scripts.glob(pattern)))
    except OSError as exc:
        return CategoryInspection(
            "F", InspectionState.INCOMPLETE,
            failures=("could not enumerate {}: {}".format(scripts, exc),))
    for path in seen:
        relative = path.relative_to(root).as_posix()
        untracked = _tracking(inspection, relative, failures)
        found.append(Candidate(relative, _verdict(untracked),
                               "untracked={}".format(untracked)))
    return _result("F", found, failures)


def _result(category, found, failures) -> CategoryInspection:
    state = (InspectionState.INCOMPLETE if failures
             else InspectionState.COMPLETE)
    return CategoryInspection(category=category, state=state,
                              candidates=tuple(found),
                              failures=tuple(failures))


#: THE ONE authoritative enumeration. Order, expected identities and report
#: labels all derive from here, so no second list can drift.
CATEGORY_INSPECTORS = (
    ("A", inspect_category_a), ("B", inspect_category_b),
    ("C", inspect_category_c), ("D", inspect_category_d),
    ("E", inspect_category_e), ("F", inspect_category_f),
)


def require_category_coverage(expected, results) -> None:
    actual = tuple(result.category for result in results)
    if len(actual) != len(set(actual)):
        raise InspectionError("duplicate category result: {}".format(actual))
    if set(actual) != set(expected):
        raise InspectionError(
            "missing or unexpected category result: expected {}, got "
            "{}".format(sorted(expected), sorted(actual)))


@dataclass(frozen=True)
class CleanupProposal:
    """An inspection result. NOT an approved plan, and not executable."""

    repository_root: str
    categories: tuple
    application_status: str = "unavailable"
    authorization_status: str = "not_established"

    @property
    def inspection_status(self) -> str:
        if any(row.state is not InspectionState.COMPLETE
               for row in self.categories):
            return "incomplete"
        return "completed"

    @property
    def conditions_determined(self) -> int:
        """Candidates whose conditions Git and the filesystem could settle."""
        return sum(1 for row in self.categories for c in row.candidates
                   if c.conditions is not ConditionState.UNKNOWN)

    @property
    def category_findings(self) -> int:
        """Candidate APPEARANCES across categories. A path may appear twice."""
        return sum(len(row.candidates) for row in self.categories)

    @property
    def unique_candidate_paths(self) -> int:
        return len({c.relative_path for row in self.categories
                    for c in row.candidates})

    @property
    def inspection_failures(self) -> int:
        return sum(len(row.failures) for row in self.categories)


def inspect_cleanup_candidates(inspection: RepositoryInspection
                               ) -> CleanupProposal:
    # ONE capture, passed to every category, so no two can disagree about
    # which commit they examined.
    head = capture_head(inspection)
    results = tuple(function(inspection.root, inspection, head)
                    for _, function in CATEGORY_INSPECTORS)
    require_category_coverage([name for name, _ in CATEGORY_INSPECTORS],
                              results)
    return CleanupProposal(repository_root=str(inspection.root),
                           categories=results)


def render_proposal(proposal: CleanupProposal, stream=None) -> None:
    out = stream if stream is not None else sys.stdout
    print("CLEANUP INSPECTION -- NO DELETION IS POSSIBLE IN THIS VERSION",
          file=out)
    print("  repository            : {}".format(proposal.repository_root),
          file=out)
    print("  inspection_status     : {}".format(proposal.inspection_status),
          file=out)
    print("  category_findings     : {}".format(proposal.category_findings),
          file=out)
    print("  unique_candidate_paths: {}".format(
        proposal.unique_candidate_paths), file=out)
    print("  conditions_determined : {}".format(
        proposal.conditions_determined), file=out)
    print("  inspection_failures   : {}".format(proposal.inspection_failures),
          file=out)
    print("  application_status    : {}".format(proposal.application_status),
          file=out)
    print("  authorization_status  : {}".format(proposal.authorization_status),
          file=out)
    print("  deletion_attempts     : 0", file=out)
    for row in proposal.categories:
        print("", file=out)
        print("  ({}) {} -- {} candidate(s), {} failure(s)".format(
            row.category, row.state.value, len(row.candidates),
            len(row.failures)), file=out)
        for candidate in row.candidates:
            verdict = {
                ConditionState.SATISFIED:
                    "category conditions satisfied; application unavailable",
                ConditionState.NOT_SATISFIED:
                    "category conditions NOT satisfied",
                ConditionState.UNKNOWN:
                    "category conditions UNKNOWN -- inspection failed",
            }[candidate.conditions]
            print("      {:<52} {}".format(candidate.relative_path[:52],
                                           verdict), file=out)
            print("          {}".format(candidate.detail), file=out)
        for failure in row.failures:
            print("      INSPECTION FAILED: {}".format(failure), file=out)
