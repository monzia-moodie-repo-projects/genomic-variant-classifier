"""Preflight check 9 must confirm a CREDENTIAL, not a variable name.

PREFLIGHT-TOKEN-SUBSTRING-1
===========================
`scripts/preflight_check.py:259` read:

    if "GITHUB_TOKEN=" in content:
        return True, ".env has GITHUB_TOKEN"

A substring search over the WHOLE FILE. It returned True for a commented-out
line, an empty value, a placeholder, and an unrelated variable whose name
merely ENDS with GITHUB_TOKEN.

MEASURED 2026-08-15, TWICE. The literal text `GITHUB_TOKEN=<the real token>`
was written into .env on two separate occasions, and check 9 reported

    GITHUB_TOKEN available somewhere: True  (.env has GITHUB_TOKEN)

both times. Every cloud run is gated on that check, so a run would have
proceeded on a credential that did not exist.

The Windows User-environment branch at line 275 has ALWAYS applied
`len(token) > 10`. Two of three branches disagreed about what "available"
means, and the weaker one was the one people actually use.

WHY A FLOOR AND NOT AN EXACT LENGTH
GitHub personal access tokens are 40 characters; fine-grained tokens are
longer; installation tokens are moving to a stateless ~520-character format
(announced on the Actions page, 2026-08). An equality check would break on the
next format. A floor admits every one and still rejects placeholders.

Author: Monzia Moodie
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "preflight_check.py"


def _load():
    """Import the script by path. It is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("preflight_check", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_check"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pf():
    return _load()


# ---- the placeholders that actually fooled it ---------------------------
def test_the_placeholder_written_twice_today_is_REJECTED(pf):
    """THE DEFECT, as the exact bytes that caused it."""
    assert pf._env_token_value("GITHUB_TOKEN=<the real token>\n") is None


def test_an_empty_value_is_REJECTED(pf):
    assert pf._env_token_value("GITHUB_TOKEN=\n") is None
    assert pf._env_token_value("GITHUB_TOKEN=   \n") is None


def test_a_commented_line_is_REJECTED(pf):
    assert pf._env_token_value("# GITHUB_TOKEN=ghp_realish_value\n") is None
    assert pf._env_token_value("  #GITHUB_TOKEN=ghp_realish_value\n") is None


def test_a_comment_containing_a_WELL_FORMED_assignment_is_REJECTED(pf):
    """The comment skip, made load-bearing.

    Sabotage X4 removed `line.startswith("#")` and went UNDETECTED, because my
    commented examples were `# GITHUB_TOKEN=...` -- and partition("=") gives
    the name "# GITHUB_TOKEN", which the exact-name check rejects anyway. X5
    was doing X4's work, so X4 had no independent effect.

    MEASURED, and the honest answer: with the skip removed, all three of these
    STILL return None, because partition("=") on "# GITHUB_TOKEN=..." yields
    the name "# GITHUB_TOKEN", which fails the exact-name comparison. X5 does
    X4's work entirely, and no input distinguishes them while both rules stand.

    These cases are therefore DOCUMENTATION of intent, not coverage. They are
    kept because the comment skip is the rule a reader expects to be doing this
    work, and a future relaxation of the name check would make it load-bearing
    again -- at which point these fail, which is exactly when they should.
    """
    assert pf._env_token_value("#GITHUB_TOKEN=ghp_commented_out_val\n") is None
    assert pf._env_token_value("   # GITHUB_TOKEN=ghp_commented_out\n") is None
    # A comment line that would otherwise parse as a valid assignment:
    assert pf._env_token_value("#\n#GITHUB_TOKEN=ghp_two_comment_lines\n") is None


def test_the_comment_marker_is_checked_BEFORE_the_name(pf):
    """Order of rules, asserted for the same documentary reason as above:
    inert while the exact-name check stands, load-bearing if it is relaxed."""
    src = "# disabled for now\n#GITHUB_TOKEN=ghp_disabled_token_val\n"
    assert pf._env_token_value(src) is None
    # and the same content UNcommented is found, so the test is not vacuous
    assert pf._env_token_value(
        "# disabled for now\nGITHUB_TOKEN=ghp_disabled_token_val\n"
    ) == "ghp_disabled_token_val"


def test_a_name_that_merely_ENDS_with_the_key_is_REJECTED(pf):
    """The substring search matched `MY_GITHUB_TOKEN=` too."""
    assert pf._env_token_value("MY_GITHUB_TOKEN=ghp_realish_value\n") is None
    assert pf._env_token_value("OLD_GITHUB_TOKEN=ghp_realish_value\n") is None


@pytest.mark.parametrize("value", [
    "<paste yours here>", "$env:GITHUB_TOKEN", "${GITHUB_TOKEN}",
    "your_token_here", "YOUR_TOKEN", "PASTE_TOKEN_HERE",
])
def test_placeholder_shapes_are_REJECTED(pf, value):
    assert pf._env_token_value("GITHUB_TOKEN={}\n".format(value)) is None


# ---- and a real token is accepted --------------------------------------
def test_a_plain_token_is_returned(pf):
    assert pf._env_token_value(
        "GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n") == "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.mark.parametrize("quote", ['"', "'"])
def test_a_quoted_token_is_unquoted(pf, quote):
    src = "GITHUB_TOKEN={q}ghp_quoted_token_value{q}\n".format(q=quote)
    assert pf._env_token_value(src) == "ghp_quoted_token_value"


def test_surrounding_whitespace_is_stripped(pf):
    assert pf._env_token_value(
        "GITHUB_TOKEN=   ghp_padded_token_value   \n") == "ghp_padded_token_value"


def test_the_token_need_not_be_on_the_first_line(pf):
    src = "NCBI_API_KEY=x\n\n# a comment\nGITHUB_TOKEN=ghp_later_line_token\n"
    assert pf._env_token_value(src) == "ghp_later_line_token"


def test_an_empty_file_yields_None(pf):
    assert pf._env_token_value("") is None
    assert pf._env_token_value("\n\n# only comments\n") is None


def test_a_long_modern_token_is_accepted(pf):
    """Installation tokens are moving to a ~520-character format. A floor
    admits them; an equality check would not."""
    long_token = "ghs_" + "a" * 516
    assert pf._env_token_value(
        "GITHUB_TOKEN={}\n".format(long_token)) == long_token


# ---- the floor is applied, and is the SAME floor ------------------------
def test_the_length_floor_is_THIRTY_and_is_a_floor(pf):
    """Derived from two measured lengths, not chosen.

    22  the fragments Add-Content produced when it split a pasted token across
        two lines, twice, on 2026-08-15.
    40  the real token, ghp_ + 36 alphanumeric.

    Every current GitHub format is at least 40. Any floor in [22, 40) works;
    thirty leaves margin on both sides.
    """
    assert pf._MIN_TOKEN_LENGTH == 30
    # The PARSER has no floor -- it reports what is there.
    assert pf._env_token_value("GITHUB_TOKEN=short\n") == "short"


def test_the_22_CHARACTER_FRAGMENT_does_not_satisfy_the_check(pf, tmp_path,
                                                              monkeypatch):
    """THE INCIDENT, as a test.

    PowerShell Add-Content split a 40-character token across two lines, leaving
    22 characters after GITHUB_TOKEN= and the remainder on a bare line. The old
    substring check accepted it. A floor of 10 -- or of 20, which I asserted
    before checking it against the measured 22 -- would also accept it.
    """
    fragment = "ghp_abcdefghijklmnop69"          # exactly 22, as observed
    assert len(fragment) == 22, len(fragment)
    monkeypatch.setattr(pf, "REPO", tmp_path)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN={}\n".format(fragment), encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pf, "IS_WINDOWS", False)
    ok, detail = pf.github_token_available()
    assert ok is False, detail


def test_a_FULL_forty_character_token_satisfies_the_check(pf, tmp_path,
                                                          monkeypatch):
    """The other side of the same boundary: the real thing must pass."""
    token = "ghp_" + "a" * 36                     # exactly 40, as observed
    assert len(token) == 40, len(token)
    monkeypatch.setattr(pf, "REPO", tmp_path)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN={}\n".format(token), encoding="utf-8")
    ok, detail = pf.github_token_available()
    assert ok is True, detail
    assert "40" in detail


@pytest.mark.parametrize("prefix,body", [
    ("ghp_", 36), ("gho_", 36), ("ghu_", 36), ("ghs_", 36),
    ("ghr_", 72), ("github_pat_", 82),
])
def test_every_current_github_format_clears_the_floor(pf, prefix, body):
    """The floor must not reject a real credential. Lengths are the published
    ones: classic, OAuth, user-to-server and server-to-server are 40; refresh
    tokens 76; fine-grained 93."""
    value = prefix + "a" * body
    assert len(value) > pf._MIN_TOKEN_LENGTH, (prefix, len(value))


def test_a_short_value_does_not_satisfy_the_CHECK(pf, tmp_path, monkeypatch):
    """The parser returns it; the CHECK rejects it on length. Both halves
    matter: the parser reports what is there, the check decides usability."""
    monkeypatch.setattr(pf, "REPO", tmp_path)
    (tmp_path / ".env").write_text("GITHUB_TOKEN=short\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pf, "IS_WINDOWS", False)
    ok, detail = pf.github_token_available()
    assert ok is False, detail


def test_a_real_token_in_env_satisfies_the_check(pf, tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "REPO", tmp_path)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", encoding="utf-8")
    ok, detail = pf.github_token_available()
    assert ok is True
    assert "length" in detail
    assert "ghp_" not in detail, "the detail string must NOT echo the token"


def test_the_placeholder_does_NOT_satisfy_the_check(pf, tmp_path, monkeypatch):
    """End to end, through the real function, with the exact failing bytes."""
    monkeypatch.setattr(pf, "REPO", tmp_path)
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN=<the real token>\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pf, "IS_WINDOWS", False)
    ok, detail = pf.github_token_available()
    assert ok is False, detail


# ---- no branch may echo the secret --------------------------------------
def test_no_branch_echoes_the_token_value(pf, tmp_path, monkeypatch):
    """A .env token was printed in full during this session because a display
    split on '=' and a bare line had none. Every branch reports a LENGTH."""
    monkeypatch.setattr(pf, "REPO", tmp_path)
    secret = "ghp_S3CRETVALUEDONOTPRINTaaaaaaaaaaaaaaa"
    (tmp_path / ".env").write_text(
        "GITHUB_TOKEN={}\n".format(secret), encoding="utf-8")
    ok, detail = pf.github_token_available()
    assert ok is True
    assert secret not in detail

    monkeypatch.setattr(pf, "REPO", tmp_path / "nonexistent")
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    ok2, detail2 = pf.github_token_available()
    assert ok2 is True
    assert secret not in detail2


# ---- an unreadable file is not an absent token --------------------------
def test_an_unreadable_env_is_REPORTED_not_swallowed(pf, tmp_path, monkeypatch):
    """`except Exception: pass` made "cannot read" and "no token" the same
    answer. They are different, and one of them is a fault."""
    monkeypatch.setattr(pf, "REPO", tmp_path)
    env = tmp_path / ".env"
    env.write_text("GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(type(env), "read_text", boom, raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(pf, "IS_WINDOWS", False)
    ok, detail = pf.github_token_available()
    assert ok is False
    assert "could not be read" in detail
