"""The roadmap must not lie about numbers the code already knows. Enforced, not trusted.

Created 2026-08-23, immediately after the D2c authority succession (`2bfc5b1`).

WHY
---
The predecessor of `docs/ROADMAP.md` reached 466,826 bytes and 7,020 lines, and its
headline facts had gone measurably wrong: it stated "80 features, 13-model ensemble"
against a contract of 95, and "Suite: 862 passed / 1 skipped" against a ratchet of
5,395 -- a factor of six. Its own section 7 names the mechanism, root pattern (a):
a number written down once and never re-derived becomes a lie on a schedule.

D2c replaced it with an 11,275-byte current-state document whose every headline
number was measured from the package and proven by the installer BEFORE it was
written. That proof was a one-time event. Without this file the successor's numbers
are unbound from the moment the installer exits, and it begins rotting on day one --
which is exactly how the predecessor got where it did.

    THE FIX IS NOT TO CORRECT THE NUMBERS. It is to stop keeping a second copy.

This file re-derives the agreement on every test run, and is deliberately modelled on
`tests/unit/test_readme_claims.py`, which was rebuilt three times by its own failures:

  * its feature-count sweep matched "36 of its 78 features CONSTANT ZERO" -- true,
    necessary history -- and was fixed by ENUMERATING claim sites;
  * its test-count check was written as a blanket ban and fired on the README's own
    correction note, forbidding the document from recording its own repair. Fixed by
    exempting blockquote COMMENTARY from body-text CLAIMS;
  * its first equality check used `assert collected - n <= 50`, which let a real
    17-test drift pass. "A tolerance on a number that CAN be exact is not engineering
    judgement; it is a place for rot to live."

All three lessons are applied here rather than re-learned.

WHAT THIS FILE ADDS THAT THE README BINDING HAS NO ANALOGUE FOR
--------------------------------------------------------------
The successor QUOTES the archival proof of the succession: the predecessor's byte
count and its git blob object identifier. If the archive were altered or removed, the
roadmap would go on asserting an archival guarantee that no longer held, and nothing
would notice. So the pointer must resolve and the blob identity must still hold --
recomputed here the way git computes it, not read from a field.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    PHASE_2_FEATURES,
    PHASE_4_FEATURES,
    SEQUENCE_FEATURES,
    TABULAR_FEATURES,
)

ROADMAP = Path("docs/ROADMAP.md")
SUITE_SIZE_FILE = Path("tests/EXPECTED_SUITE_SIZE")
ARCHIVE = Path("docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md")

#: The predecessor, as measured at f2b93ff and preserved by D2c at 2bfc5b1.
ARCHIVED_BLOB_OID = "990088a61365ef3de3a02fd34327c7c5f3134731"
ARCHIVED_BYTES = 466826

#: Every place the roadmap makes a CLAIM about present state, with the live source
#: each must equal.
#:
#: ENUMERATED, NOT SWEPT -- for the reason test_readme_claims.py records at length.
#: A sweep cannot distinguish "the model HAS n features" from "36 of its 78 features
#: WERE zero", and a gate that cannot tell a claim from a historical note is a ban on
#: writing history down, in a project whose entire method is writing history down.
#:
#: If you add a new place that asserts one of these numbers, ADD IT HERE. That is the
#: one manual step, and it is the honest one.
CLAIM_SITES: dict[str, tuple[str, str]] = {
    "snapshot: tabular contract":
        (r"Tabular feature contract \| \*\*(\d+)\*\*", "feature_count"),
    "snapshot: phase-2 features":
        (r"not yet computed\) \| \*\*(\d+)\*\*", "phase2"),
    "snapshot: phase-4 features":
        (r"Phase-4 features \| (\d+) \|", "phase4"),
    "snapshot: sequence features":
        (r"Sequence features \| (\d+) \|", "sequence"),
    "snapshot: base-model roster":
        (r"Base-model roster \| \*\*(\d+)\*\*", "roster"),
    "snapshot: registered agents":
        (r"Registered agents \| \*\*(\d+)\*\*", "agents"),
    "snapshot: suite size":
        (r"Test suite \| \*\*([\d,]+) collected\*\*", "suite"),
    "identity sentence: features":
        (r"(\d+) features, \d+-model ensemble", "feature_count"),
    "identity sentence: models":
        (r"\d+ features, (\d+)-model ensemble", "roster"),
}


@pytest.fixture(scope="module")
def roadmap() -> str:
    if not ROADMAP.is_file():
        pytest.fail("docs/ROADMAP.md not found at {}".format(ROADMAP.resolve()))
    return ROADMAP.read_text(encoding="utf-8")


def _expected_suite_size() -> int:
    """The ratchet's number -- the SAME file tests/conftest.py reads. Never a copy."""
    for line in SUITE_SIZE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return int(line)
    raise AssertionError("no bare integer found in {}".format(SUITE_SIZE_FILE))


def _live_roster() -> list:
    """The roster is BUILT, not declared.

    There is no roster constant. `_build_estimators` produces `base_estimators`,
    which is what `fit()` writes into `ensemble_completeness_["roster"]`. Six guessed
    attribute names failed to find it on 2026-08-23; the repository's own
    test_readme_claims.py had the answer, with a comment saying why a regular
    expression over the source is not an acceptable substitute.
    """
    from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
    return sorted(VariantEnsemble().base_estimators)


def _live_agents() -> list:
    """Built inside a method, so the instance is made WITHOUT __init__ side effects."""
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch._register_agents()
    return sorted(orch._agent_registry)


def _blob_oid(data: bytes) -> str:
    """Git's own blob identity: sha1("blob " + len + "\\0" + content).

    No path is an input, which is precisely why a preserved copy of identical bytes
    carries the identical identifier. Verified against `git hash-object` on
    2026-08-23 for five payloads including empty and binary.
    """
    h = hashlib.sha1()
    h.update("blob {}\0".format(len(data)).encode("ascii"))
    h.update(data)
    return h.hexdigest()


def _measured() -> dict:
    return {
        "feature_count": EXPECTED_TABULAR_FEATURE_COUNT,
        "phase2": len(PHASE_2_FEATURES),
        "phase4": len(PHASE_4_FEATURES),
        "sequence": len(SEQUENCE_FEATURES),
        "roster": len(_live_roster()),
        "agents": len(_live_agents()),
        "suite": _expected_suite_size(),
    }


# ---------------------------------------------------------------------------
# 1. THE CLAIM SITES -- present, and equal to the live source
# ---------------------------------------------------------------------------

def test_every_claim_site_is_still_present(roadmap):
    """A claim site that vanishes must FAIL, not quietly stop being checked.

    Either the roadmap was restructured -- fix the pattern -- or the claim was
    dropped. Deleting the entry to go green is the defect this file prevents.
    """
    missing = [site for site, (pattern, _key) in CLAIM_SITES.items()
               if re.search(pattern, roadmap) is None]
    assert not missing, (
        "these claim sites are no longer found in docs/ROADMAP.md: {}\n\n"
        "Either the roadmap was restructured -- in which case FIX THE PATTERN in "
        "CLAIM_SITES -- or the claim was removed. Do not delete the entry to make "
        "this pass: a check that no longer checks anything is exactly the defect "
        "this file exists to prevent.".format(missing))


def test_every_claim_equals_its_live_source(roadmap):
    """Nine claim sites, seven quantities, EQUALITY throughout.

    No tolerance anywhere. The README binding's first version used
    `assert collected - n <= 50` and let a real 17-test drift pass while reporting
    green.
    """
    measured = _measured()
    wrong = []
    for site, (pattern, key) in CLAIM_SITES.items():
        m = re.search(pattern, roadmap)
        assert m is not None, site      # covered by the test above; belt and braces
        claimed = int(m.group(1).replace(",", ""))
        if claimed != measured[key]:
            wrong.append((site, claimed, measured[key]))
    assert not wrong, (
        "docs/ROADMAP.md states {} wrong value(s):\n".format(len(wrong))
        + "\n".join("    {:34s} says {:>6}   live source says {:>6}".format(s, c, a)
                    for s, c, a in wrong)
        + "\n\nUpdate EVERY site, in the same commit as the change that moved the "
          "number. The predecessor of this document said 80 features against a "
          "contract of 95, and 862 tests against a suite of 5,395.")


def test_the_constant_equals_the_list_it_describes():
    """A constant that has drifted from its own list is the same defect, one level
    down -- and it would make every claim above agree with a wrong number."""
    assert EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES), (
        "EXPECTED_TABULAR_FEATURE_COUNT is {} but len(TABULAR_FEATURES) is {}"
        .format(EXPECTED_TABULAR_FEATURE_COUNT, len(TABULAR_FEATURES)))


# ---------------------------------------------------------------------------
# 2. THE KIND OF NUMBER -- collected, never passing
# ---------------------------------------------------------------------------

def test_the_roadmap_does_not_quote_a_passing_count(roadmap):
    """The roadmap must quote COLLECTED, not PASSING.

    Passing is `collected - skipped`, and the skip set differs by machine -- 15 on
    this Windows tree, a different split on the hosted Linux runner, from the SAME
    collection. Any passing count is therefore true on at most one machine.

    BLOCKQUOTES ARE EXEMPT, AND THAT EXEMPTION IS THE POINT. Lines beginning with
    `>` are the document explaining its own history: the succession notice quotes the
    predecessor's figures, and section 9 quotes the archived plan verbatim. The README
    binding banned a string outright and fired on its own correction note -- twice, in
    two turns. Commentary is not a claim.
    """
    body = "\n".join(line for line in roadmap.splitlines()
                     if not line.lstrip().startswith(">"))
    offenders = [m.group(0) for m in
                 re.finditer(r"\b\d[\d,]{2,6}\s*(?:tests?\s+)?passing", body)
                 if "/" not in m.group(0)]
    assert not offenders, (
        "docs/ROADMAP.md quotes a PASSING test count in body text: {}\n\n"
        "Quote the COLLECTED count instead -- it is environment-independent and "
        "lives in exactly one place, tests/EXPECTED_SUITE_SIZE.".format(offenders))


# ---------------------------------------------------------------------------
# 3. THE ARCHIVAL PROOF -- the succession's guarantee, still true
# ---------------------------------------------------------------------------

def test_the_archive_pointer_resolves(roadmap):
    """The roadmap names its predecessor's address. That address must exist."""
    assert ARCHIVE.as_posix() in roadmap, (
        "docs/ROADMAP.md no longer names its archive at {}. The succession notice "
        "is the only thing telling a reader where the discharged history went."
        .format(ARCHIVE.as_posix()))
    assert ARCHIVE.is_file(), (
        "docs/ROADMAP.md points at {} but that file does not exist. The roadmap is "
        "asserting an archival guarantee that no longer holds."
        .format(ARCHIVE.as_posix()))


def test_the_archived_blob_identity_still_holds():
    """THE SUCCESSION'S PROOF, re-derived rather than trusted.

    D2c preserved the predecessor by CREATE rather than `git mv`, because git blobs
    are content-addressed: identical bytes yield the identical object identifier
    regardless of path. The installer proved it after the commit with
    `git rev-parse HEAD:<archive>`. This re-derives the same identity from the bytes
    on disk, so a later edit to the archive fails here rather than silently voiding
    the guarantee the roadmap states.
    """
    raw = ARCHIVE.read_bytes()
    assert len(raw) == ARCHIVED_BYTES, (
        "the archived predecessor is {:,} bytes; it was preserved at {:,}. An "
        "archive is only evidence while its bytes are the bytes that existed."
        .format(len(raw), ARCHIVED_BYTES))
    actual = _blob_oid(raw)
    assert actual == ARCHIVED_BLOB_OID, (
        "the archived predecessor's git blob object identifier is {} but the "
        "succession preserved {}.\n\nThe roadmap quotes the latter as proof that "
        "the archived bytes are the bytes that were live at f2b93ff. If they "
        "differ, that proof is false.".format(actual, ARCHIVED_BLOB_OID))


def test_the_roadmap_quotes_the_blob_identity_it_can_prove(roadmap):
    """The identifier printed in the document must be the one on disk.

    Otherwise the roadmap states an archival proof that is merely plausible.
    """
    assert ARCHIVED_BLOB_OID in roadmap, (
        "docs/ROADMAP.md no longer quotes the archived blob object identifier {}. "
        "That string is the succession's proof; without it the notice is an "
        "assertion.".format(ARCHIVED_BLOB_OID))
    assert _blob_oid(ARCHIVE.read_bytes()) == ARCHIVED_BLOB_OID


# ---------------------------------------------------------------------------
# 4. THE DOCUMENT ITSELF
# ---------------------------------------------------------------------------

def test_the_roadmap_is_authored_text(roadmap):
    """The AUTHORING policy: this document is written fresh, not preserved.

    The archive is PRESERVED and obeys a different policy -- its 184 non-ASCII bytes
    are a fact about history, and correcting them would falsify the record. That
    asymmetry is ADR-0004 section C, visible in two files in the same repository.
    """
    raw = ROADMAP.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", "docs/ROADMAP.md carries a byte-order mark"
    assert b"\r\n" not in raw, "docs/ROADMAP.md contains CRLF"
    assert not any(b > 0x7F for b in raw), "docs/ROADMAP.md contains non-ASCII"
    assert raw.endswith(b"\n"), "docs/ROADMAP.md has no trailing newline"


def test_the_roadmap_has_not_regrown_into_a_journal(roadmap):
    """A LIVING roadmap carries present state. Discharged history goes to the archive.

    The predecessor became 466,826 bytes by appending roughly forty dated delta
    sections and never discharging any. This is a smoke alarm, not a style rule: it
    fires long before the document becomes unreadable, and the remedy is another
    succession, not a larger threshold.
    """
    deltas = re.findall(r"^#+\s*ROADMAP delta\b", roadmap, re.M)
    assert not deltas, (
        "docs/ROADMAP.md has {} 'ROADMAP delta' section(s). The predecessor grew to "
        "466,826 bytes exactly this way. Discharge them to the archive."
        .format(len(deltas)))
    assert len(roadmap.encode("utf-8")) < 120_000, (
        "docs/ROADMAP.md is {:,} bytes. It was 11,275 after the 2026-08-23 "
        "succession and its predecessor was 466,826. Consider a succession before "
        "raising this bound.".format(len(roadmap.encode("utf-8"))))
