"""Three counters, one measured quantity.

RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1 and
ROADMAP-PROVENANCE-CLAIM-STALE-1. Created 2026-08-26.

WHAT THIS GUARDS
----------------
The collected test count appears in THREE places:

    tests/EXPECTED_SUITE_SIZE        the bare integer a conftest enforces
    README.md                        the shields.io badge
    docs/ROADMAP.md                  the current-state snapshot table

Every ratchet-moving installer renders all three from ONE measured count. That
is discipline, and discipline is not a binding: nothing FAILS if a future unit
patches two of the three, and the third then states a figure the repository no
longer supports.

MEASURED 2026-08-26 -- and this is not hypothetical in the neighbouring sense.
Eleven consecutive installers patched the roadmap's figure with a same-width
substitution while leaving the paragraph above it claiming *"Every number here
was MEASURED on 2026-08-23 at `f2b93ff`"*. The FIGURE was correct throughout;
the PROVENANCE was false from the second commit onward. A same-width
substitution is invisible to a length check, and nothing read the prose.

That is `TEMPORALCITE-1`'s shape: a citation whose subject moved.

WHY A TEST RATHER THAN CARE
---------------------------
`METHODS-CURRENT-ARCHITECTURE-STALE-1` was found on 2026-08-24 because someone
finally read a file they had claimed to read in full. The roadmap was read for
the first time on 2026-08-26, eleven patches in. Both were found by reading;
neither was found by anything that runs.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RATCHET = ROOT / "tests" / "EXPECTED_SUITE_SIZE"
README = ROOT / "README.md"
ROADMAP = ROOT / "docs" / "ROADMAP.md"

#: The roadmap row, with a thousands separator. The README badge, without one.
_ROADMAP_ROW = re.compile(r"\|\s*Test suite\s*\|\s*\*\*([\d,]+) collected\*\*")
_README_BADGE = re.compile(r"badge/tests-(\d+)-")


def ratchet_value() -> int:
    """The single bare integer. `conftest` rejects any other shape."""
    assert RATCHET.is_file(), RATCHET
    bare = [line.strip() for line in RATCHET.read_text(encoding="utf-8").split("\n")
            if line.strip().isdigit()]
    assert len(bare) == 1, (
        "tests/EXPECTED_SUITE_SIZE holds {} bare integer(s); it must hold "
        "exactly one".format(len(bare)))
    return int(bare[0])


def roadmap_value(text: str) -> int:
    """The snapshot figure, thousands separator stripped.

    Stripping matters: `5,583` and `5583` are one quantity written two ways,
    and an earlier comparison in this programme called two such spellings
    different.
    """
    found = _ROADMAP_ROW.findall(text)
    assert len(found) == 1, (
        "docs/ROADMAP.md states the test-suite row {} time(s); it must state "
        "it exactly once".format(len(found)))
    return int(found[0].replace(",", ""))


def readme_value(text: str) -> int:
    found = _README_BADGE.findall(text)
    assert len(found) == 1, (
        "README.md carries {} test badge(s); it must carry exactly "
        "one".format(len(found)))
    return int(found[0])


def test_the_roadmap_figure_equals_the_ratchet():
    """The roadmap's third column names `tests/EXPECTED_SUITE_SIZE` as its
    source. This is what makes that claim true rather than decorative."""
    stated = roadmap_value(ROADMAP.read_text(encoding="utf-8"))
    assert stated == ratchet_value(), (
        "docs/ROADMAP.md states {} collected; tests/EXPECTED_SUITE_SIZE holds "
        "{}".format(stated, ratchet_value()))


def test_the_readme_badge_equals_the_ratchet():
    stated = readme_value(README.read_text(encoding="utf-8"))
    assert stated == ratchet_value(), (
        "the README badge states {}; tests/EXPECTED_SUITE_SIZE holds "
        "{}".format(stated, ratchet_value()))


def test_all_three_agree():
    """Stated separately so a failure names WHICH pair disagrees, and then
    once more together so a reader sees the quantity is single."""
    values = {
        "tests/EXPECTED_SUITE_SIZE": ratchet_value(),
        "README.md badge": readme_value(README.read_text(encoding="utf-8")),
        "docs/ROADMAP.md row": roadmap_value(ROADMAP.read_text(encoding="utf-8")),
    }
    assert len(set(values.values())) == 1, values


@pytest.mark.parametrize(
    "text,expected",
    [("| Test suite | **5,583 collected** | src |", 5583),
     ("| Test suite | **5583 collected** | src |", 5583),
     ("|  Test suite  |  **12,345 collected**  | src |", 12345)],
    ids=["with-separator", "without-separator", "extra-whitespace"])
def test_the_roadmap_extractor_reads_both_spellings(text, expected):
    """Guards the extractor.

    A reader that understood only one spelling would pass forever on a
    roadmap written the other way -- silently, because there would be nothing
    to compare.
    """
    assert roadmap_value(text) == expected


def test_the_extractor_refuses_a_missing_or_duplicated_row():
    """Zero and two are both failures, and for different reasons: nothing to
    check, versus two claims that may disagree with each other."""
    with pytest.raises(AssertionError):
        roadmap_value("no such row here")
    with pytest.raises(AssertionError):
        roadmap_value("| Test suite | **1 collected** |\n"
                      "| Test suite | **2 collected** |")


def test_the_roadmap_does_not_claim_its_figures_were_frozen_at_one_commit():
    """ROADMAP-PROVENANCE-CLAIM-STALE-1.

    The snapshot table is MAINTAINED, not frozen. A paragraph asserting every
    number was measured at one named commit becomes false the moment an
    installer patches one of them -- which happened eleven times before anyone
    read it.

    The permissive direction matters as much: the roadmap may SAY when the
    table was constructed, and may quote the old claim as history. What it may
    not do is assert, in the present tense, that the figures were measured at
    a commit that is no longer HEAD.
    """
    text = ROADMAP.read_text(encoding="utf-8")
    history = ("previously read", "repaired", "was true when written",
               "STALE", "no longer", "until")
    offenders = []
    for i, line in enumerate(text.split("\n"), 1):
        if not re.search(r"number.{0,40}(was|were) MEASURED", line, re.I):
            continue
        if any(marker in line for marker in history):
            continue
        offenders.append((i, line.strip()[:88]))
    assert not offenders, (
        "these lines assert the snapshot figures were measured at a fixed "
        "point: {}. The table is maintained; the right-hand column names each "
        "figure's source so it can be re-derived.".format(offenders))
