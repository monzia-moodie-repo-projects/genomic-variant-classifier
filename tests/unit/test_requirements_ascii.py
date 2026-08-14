"""Tracked requirements files must be pure ASCII -- REQFILES-NONASCII-1.

WHY
===
Measured 2026-08-13: three EM DASH characters (U+2014) across two tracked
requirements files --

    requirements-dev.txt:1    "# Development and testing dependencies <em> never..."
    requirements.in:43        "# Testing <em> keep separate or use requirements-dev.in"
    requirements.in:109       "# Core ML <em> production deps that were missing..."

all three in section-header comments, all three in the same prose role.

NOTHING IS BROKEN TODAY. pip reads requirements files as UTF-8, and these bytes
sit inside comments. The argument is not aesthetic:

    These files pass through PowerShell Set-Content, Copy-Item and Windows
    editors, where encoding is not always preserved. A mojibake byte in a
    dependency file surfaces during an INSTALL, not during review.

Every guarded installer in this repository already refuses a payload carrying a
non-ASCII byte, for exactly that reason. This applies the same rule to the
files those installers edit.

IT IS ALSO ALREADY THE CONVENTION. requirements.txt uses " -- " in the same
prose role at lines 160, 166 and 177, and is pure ASCII across 236 lines. So
one sibling follows the convention and two did not.

WHAT THIS DOES NOT COVER
Untracked backup artifacts (*.bak_*, *.pre_*) are excluded: they are local
history, ignored by .gitignore, and rewriting them would destroy the record of
what a file looked like before an edit.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import subprocess
import unicodedata
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tracked_requirements_files():
    """Files git actually tracks, so ignored backups are excluded by GIT rather
    than by a filename pattern I invented."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "requirements*"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=30)
    except Exception:                                          # noqa: BLE001
        pytest.skip("git is not available to enumerate tracked files")
    if out.returncode != 0:
        pytest.skip("git ls-files failed: {}".format(out.stderr.strip()[:80]))
    return sorted(n for n in out.stdout.split("\n") if n.strip())


def test_git_reports_at_least_the_known_requirements_files():
    """A guard on the guard: if `git ls-files` returned nothing, every test
    below would pass vacuously -- an empty result masquerading as a clean one,
    which is the failure this repository keeps eliminating."""
    tracked = _tracked_requirements_files()
    assert len(tracked) >= 6, tracked
    for expected in ("requirements.txt", "requirements.in",
                     "requirements-dev.txt", "requirements-api.txt"):
        assert expected in tracked, (expected, tracked)


def test_no_tracked_requirements_file_contains_a_non_ascii_byte():
    offenders = []
    for name in _tracked_requirements_files():
        p = _REPO_ROOT / name
        if not p.exists():
            continue
        raw = io.open(p, "rb").read()
        bad = [i for i, b in enumerate(raw) if b > 0x7F]
        if bad:
            text = raw.decode("utf-8", errors="replace")
            chars = sorted({c for c in text if ord(c) > 127})
            offenders.append((name, len(bad), [
                (c, "U+%04X" % ord(c), unicodedata.name(c, "?")) for c in chars]))
    assert not offenders, (
        "non-ASCII byte(s) in tracked requirements file(s): {}. These files "
        "pass through Set-Content, Copy-Item and Windows editors where "
        "encoding is not always preserved.".format(offenders))


def test_the_ascii_prose_dash_convention_is_in_use():
    """requirements.txt established ' -- '. This asserts the convention exists,
    so a future reader knows the ASCII rule has a positive form and is not
    merely a prohibition."""
    text = io.open(_REPO_ROOT / "requirements.txt", encoding="utf-8").read()
    assert " -- " in text


def test_pytest_anyio_is_not_declared_anywhere():
    """PYTEST-ANYIO-REDIRECT-1.

    MEASURED: pytest-anyio is a 3,559-byte package at version 0.0.0 whose PyPI
    summary reads "The pytest anyio plugin is built into anyio. You don't need
    this package." It requires anyio and pytest, and has never installed here.

    PROVEN, not merely quoted: anyio 4.13.0 declares exactly ONE entry-point
    group -- pytest11, pointing at anyio.pytest_plugin -- and `pytest
    --trace-config` shows that plugin REGISTERED and active. So the capability
    arrives through anyio>=4.0, which is declared one line below where the
    redirect used to sit.
    """
    for name in _tracked_requirements_files():
        p = _REPO_ROOT / name
        if not p.exists():
            continue
        text = io.open(p, encoding="utf-8", errors="replace").read().lower()
        assert "pytest-anyio" not in text, name
        assert "pytest_anyio" not in text, name


def test_anyio_still_supplies_the_pytest_plugin():
    """The claim the removal rests on, asserted rather than assumed.

    If a future anyio drops its pytest11 entry point, this fails -- and the
    removal of pytest-anyio would need revisiting rather than silently leaving
    the suite without an async plugin.
    """
    from importlib.metadata import distribution
    try:
        dist = distribution("anyio")
    except Exception:                                          # noqa: BLE001
        pytest.skip("anyio is not installed in this environment")
    groups = {e.group for e in dist.entry_points}
    assert "pytest11" in groups, sorted(groups)
    p11 = [e for e in dist.entry_points if e.group == "pytest11"]
    assert any("anyio" in e.value for e in p11), [(e.name, e.value) for e in p11]
