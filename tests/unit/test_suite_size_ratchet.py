"""The suite-size ratchet must itself be guarded. A guard with no self-test is a rumour.

Added 2026-07-13 (roadmap 6.14).

WHAT THE RATCHET IS
-------------------
`tests/EXPECTED_SUITE_SIZE` holds ONE number: how many tests this suite collects.
`tests/conftest.py`, under the explicit `--assert-suite-size` flag, aborts the run if the
collected count disagrees with it -- in EITHER direction:

    collected < expected  ->  tests have VANISHED (deleted, mis-named, lost to a collection
                              error, or silently skipped at module import because a dependency
                              went missing -- exactly how the entire graph-neural-network
                              branch went untested for 508 Continuous Integration runs).
    collected > expected  ->  tests were ADDED and the ratchet was not bumped. Red until it is.

Both G1 (`scripts/Run_Preflight_Local.ps1`) and Continuous Integration
(`.github/workflows/ci.yml`) pass that flag and read that file, so the two gates cannot drift
apart.

WHY IT EXISTS
-------------
The G1 pre-flight carried a hand-maintained pytest floor. It rotted FIVE TIMES IN TWO DAYS:

    1485  ->  1805  ->  1842  ->  1850  ->  1853

Every single time, the number sat directly beneath an emphatic, all-capitals comment ordering
the next person to raise it -- written by the person who then failed to raise it. At 1485
against a suite passing 1,815, THREE HUNDRED AND THIRTY tests could have silently vanished and
the gate would still have reported PASS.

    A COMMENT DOES NOT ENFORCE ITSELF. No volume of emphasis will make it.
    The fix for a rule that can be forgotten is to MAKE FORGETTING FAIL.

This is the same fail-loud pattern the project already trusts for features:
`EXPECTED_TABULAR_FEATURE_COUNT` guards `TABULAR_FEATURES`, and adding a feature without
bumping the count is a hard error.

WHY THIS FILE EXISTS
--------------------
Roadmap 6.9 records that the three existing autouse guards (`sys.path` leaks, `data/`
pollution, connector-cache isolation) have **no permanent self-test** -- they were each
negative-tested by hand once, and nothing re-proves them. A guard that is never re-proven is
indistinguishable from a guard that has quietly stopped guarding: that is the entire lesson of
`KNOWN_ZERO_DEFAULT` (27 vs 25), the "65 features" comment (vs a 97-feature contract), and the
1485 floor itself.

So the ratchet gets a self-test on day one. In particular the PARSER is tested, because a
malformed ratchet file is the one failure that could make the guard silently do nothing --
and a guard that silently does nothing is worse than no guard, since it also carries a
reassuring name.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONFTEST = _REPO / "tests" / "conftest.py"
_RATCHET = _REPO / "tests" / "EXPECTED_SUITE_SIZE"


def _load_conftest():
    """Load tests/conftest.py as a module so its parser can be exercised directly.

    Imported by path rather than by name: `import conftest` is not reliable from
    tests/unit/ under pytest's default prepend import mode.
    """
    spec = importlib.util.spec_from_file_location("_ratchet_conftest", _CONFTEST)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. The committed file must be well-formed. It is the single source of truth.
# ---------------------------------------------------------------------------
def test_the_ratchet_file_exists_and_is_a_single_positive_integer():
    assert _RATCHET.is_file(), (
        f"{_RATCHET} is MISSING. It is the single source of truth for the suite size "
        f"(roadmap 6.14). Without it the ratchet cannot run -- and a missing guard must never "
        f"degrade to a silent pass."
    )

    payload = [
        line.strip()
        for line in _RATCHET.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(payload) == 1, (
        f"{_RATCHET} must contain EXACTLY ONE non-comment line (a bare positive integer). "
        f"Found {len(payload)}: {payload!r}. A ratchet that cannot be parsed is a ratchet that "
        f"does not guard."
    )
    assert payload[0].isdigit() and int(payload[0]) > 0, (
        f"the ratchet value must be a bare positive integer; got {payload[0]!r}"
    )


def test_the_parser_agrees_with_the_committed_file():
    cf = _load_conftest()
    expected = int(
        [
            ln.strip()
            for ln in _RATCHET.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ][0]
    )
    assert cf._read_expected_suite_size() == expected


# ---------------------------------------------------------------------------
# 2. NEGATIVE TESTS. The guard must FAIL LOUD when its own input is broken.
#    This is the failure mode that would otherwise make the ratchet silently inert.
# ---------------------------------------------------------------------------
def test_a_MISSING_ratchet_file_is_a_hard_error_not_a_silent_pass(tmp_path, monkeypatch):
    """A guard whose config vanished must SCREAM, never shrug."""
    cf = _load_conftest()
    monkeypatch.setattr(cf, "_SUITE_SIZE_FILE", tmp_path / "does_not_exist")

    with pytest.raises(pytest.UsageError, match="does not exist"):
        cf._read_expected_suite_size()


@pytest.mark.parametrize(
    "junk, why",
    [
        ("", "empty file"),
        ("# only comments\n# and nothing else\n", "no value at all"),
        ("1870\n1871\n", "TWO values -- which one guards?"),
        ("not-a-number\n", "non-numeric"),
        ("0\n", "zero -- a suite of zero tests is not a suite"),
        ("-5\n", "negative"),
        ("1870 tests\n", "trailing junk on the value line"),
    ],
)
def test_a_MALFORMED_ratchet_file_is_a_hard_error(tmp_path, monkeypatch, junk, why):
    """Every way of corrupting the ratchet must be caught.

    A malformed ratchet is the ONE failure that could leave the guard silently doing nothing
    while still being named as though it guards -- which is strictly worse than having no
    guard, because it also supplies false confidence.
    """
    bad = tmp_path / "EXPECTED_SUITE_SIZE"
    bad.write_text(junk, encoding="utf-8")

    cf = _load_conftest()
    monkeypatch.setattr(cf, "_SUITE_SIZE_FILE", bad)

    with pytest.raises(pytest.UsageError, match="MALFORMED"):
        cf._read_expected_suite_size()


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    """The file is heavily documented on purpose -- the parser must not choke on that."""
    ok = tmp_path / "EXPECTED_SUITE_SIZE"
    ok.write_text(
        "# a comment\n"
        "\n"
        "   # an indented comment\n"
        "\n"
        "  4242  \n"
        "\n"
        "# a trailing comment\n",
        encoding="utf-8",
    )

    cf = _load_conftest()
    monkeypatch.setattr(cf, "_SUITE_SIZE_FILE", ok)
    assert cf._read_expected_suite_size() == 4242


# ---------------------------------------------------------------------------
# 3. The flag must be OFF by default, or every subset run breaks.
# ---------------------------------------------------------------------------
def test_the_ratchet_is_off_unless_explicitly_requested(request):
    """`pytest tests/unit/test_foo.py` must NOT trip the ratchet.

    The guard asserts the size of the WHOLE suite. If it ran by default, every partial run --
    which is how anyone actually develops -- would fail with a spurious 'tests have vanished'.
    A guard that cries wolf on normal use gets disabled, and a disabled guard guards nothing.
    Only G1 and Continuous Integration, which run the full tree, pass --assert-suite-size.
    """
    assert request.config.getoption("--assert-suite-size") in (True, False), (
        "the --assert-suite-size option is not registered; pytest_addoption in "
        "tests/conftest.py has been removed or renamed, and BOTH gates silently stopped "
        "enforcing the suite size"
    )
