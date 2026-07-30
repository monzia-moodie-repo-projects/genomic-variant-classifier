"""docs/CHANGELOG.md must never carry a double-encoded sequence.

WHY THIS EXISTS
===============
MEASURED 2026-07-30. The changelog held 301 lines of mojibake across THREE
generations of one round trip -- text read as Windows code page 1252 and re-saved
as UTF-8, sometimes twice, sometimes three times:

    135 lines at one generation
    144 lines at two
     22 lines at three

THE FILE WAS VALID UTF-8 THROUGHOUT. That is exactly why nothing caught it for
weeks: no reader ever refused it, and `git diff` showed nothing unusual. It
surfaced only from a character census -- 'a-hat' 506, 'euro' 263, 'A-tilde' 251
-- and the THIRD generation surfaced only from running the repair to a fixed
point. A census counts symptoms; the repair counts causes.

WHAT THIS ASSERTS, AND WHY IT IS THE STRONGEST AVAILABLE FORM
--------------------------------------------------------------
That applying the repair to the file CHANGES NOTHING. If any recoverable
mojibake existed anywhere, the repair would move it and this would fail. That is
stronger than scanning for a list of known markers, because it requires no list
and cannot be defeated by a signature nobody thought of.

The marker scan is kept beside it because a fixed-point failure says only "the
file moved", while the marker scan names the line.

THE cp1252 HOLE
---------------
Five byte values -- 0x81, 0x8D, 0x8F, 0x90, 0x9D -- are UNDEFINED in the code
page. Python's strict codec refuses them in BOTH directions; Windows passes them
through as their Latin-1 equivalents, which is how 120 occurrences of U+009D
reached the file. The helpers below reproduce the Windows behaviour, because
that is what has to be reversed. A naive `s.encode("cp1252")` raises on those
lines and would leave them silently unchecked.

WHY LEGITIMATE NON-ASCII IS SAFE
--------------------------------
An isolated accented character encodes to a byte that is not valid UTF-8 on its
own, so the decode fails and no repair is proposed. Verified 2026-07-30 against
French accented text and against the 25 characters that legitimately survive in
this file, among them the o-umlaut of Nystroem and the capital delta of a metric
difference.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
CHANGELOG = _DOCS / "CHANGELOG.md"
ROADMAP = _DOCS / "ROADMAP.md"

#: Byte values Windows code page 1252 leaves undefined and passes through.
_HOLES = {0x81, 0x8D, 0x8F, 0x90, 0x9D}

#: Two-character signatures of the round trip. TWO characters, not one: a lone
#: accented letter is legitimate text, a pair of these is not.
_MARKERS = ("\u00e2\u20ac", "\u00c3\u00a2", "\u00e2\u2020",
            "\u00e2\u201a", "\u00c3\u0192", "\u00e2\u2030")

_CAP = 8


def _cp1252_decode(raw: bytes) -> str:
    return "".join(chr(b) if b in _HOLES else bytes([b]).decode("cp1252")
                   for b in raw)


def _cp1252_encode(text: str) -> bytes:
    buf = bytearray()
    for ch in text:
        cp = ord(ch)
        if cp in _HOLES:
            buf.append(cp)
        else:
            buf.extend(ch.encode("cp1252"))
    return bytes(buf)


def _repair(text: str):
    """(repaired, generations). Stops when stable or when a step is impossible."""
    cur, gens = text, 0
    while gens < _CAP:
        try:
            cand = _cp1252_encode(cur).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if cand == cur:
            break
        cur, gens = cand, gens + 1
    return cur, gens


@pytest.mark.parametrize("path", [CHANGELOG, ROADMAP],
                         ids=["CHANGELOG.md", "ROADMAP.md"])
def test_the_document_is_valid_utf8(path):
    """A document that cannot be decoded cannot be checked for anything else."""
    assert path.is_file(), "%s is missing" % path
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(
            "%s is not valid UTF-8: byte 0x%02x at offset %d.\n"
            "Context: %r" % (path.name, raw[exc.start], exc.start,
                             raw[max(0, exc.start - 40):exc.start + 40]))


@pytest.mark.parametrize("path", [CHANGELOG, ROADMAP],
                         ids=["CHANGELOG.md", "ROADMAP.md"])
def test_the_repair_is_a_fixed_point(path):
    """THE PRIMARY ASSERTION. Applying the repair must change nothing.

    Stronger than any marker list: if recoverable mojibake exists anywhere in
    the file, under any signature, the repair moves it and this fails.
    """
    text = path.read_bytes().decode("utf-8")
    lines = text.split("\n")
    moved = []
    for i, line in enumerate(lines):
        fixed, gens = _repair(line)
        if gens:
            moved.append((i + 1, gens, line[:90], fixed[:90]))
    assert not moved, (
        "%d line(s) in %s still carry recoverable double-encoding.\n"
        "This is the 2026-07-30 corruption returning, or a new instance of it.\n"
        "The first five:\n%s\n\n"
        "DO NOT relax this test. Repair the file: every repair was proven a "
        "bijection on 2026-07-30, meaning re-corrupting the repaired text "
        "reproduces the original byte for byte, so the operation is lossless."
        % (len(moved), path.name,
           "\n".join("  line %d, %d generation(s)\n    is:     %s\n    should: %s"
                     % m for m in moved[:5])))


def test_the_changelog_carries_no_known_marker():
    """The human-readable companion to the fixed-point test.

    A fixed-point failure says the file moved; this names the line and the
    signature, which is what someone reading the failure actually needs.
    """
    lines = CHANGELOG.read_bytes().decode("utf-8").split("\n")
    hits = [(i + 1, [m for m in _MARKERS if m in l], l[:90])
            for i, l in enumerate(lines)
            if any(m in l for m in _MARKERS)]
    assert not hits, (
        "%d line(s) carry a code-page-1252 round-trip signature:\n%s"
        % (len(hits), "\n".join("  line %d %r\n    %s" % h for h in hits[:5])))


def test_the_repair_leaves_legitimate_characters_alone():
    """The premise that makes the repair safe, pinned so it cannot rot.

    If this ever fails, the repair has become destructive and the tests above
    are no longer safe to act on.
    """
    for sample in ("Nystr\u00f6m `svm` (OOF 0.9804)",
                   "Test AUROC 0.9975 (Run 13 0.9974, \u0394 +0.0001)",
                   "4.40M raw \u2192 1.49M after filtering",
                   "PASS \u22650.9; total 767 s",
                   "~0.50\u20130.51 \u2014 more data does not help",
                   "Beau Travail, r\u00e9alis\u00e9 en 1999"):
        fixed, gens = _repair(sample)
        assert gens == 0, (
            "the repair altered legitimate text %r into %r after %d "
            "generation(s); it is no longer safe to apply"
            % (sample, fixed, gens))
