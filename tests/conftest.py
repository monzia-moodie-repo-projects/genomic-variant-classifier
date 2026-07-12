"""Test bootstrap (tests/conftest.py).

Makes the repo-local `scripts/` directory importable during tests regardless of CWD or
environment, so tests can `import clean_cohort` (and other script-level modules) the same way
locally and in CI.

Root cause this fixes: CI runs `pytest tests/unit/ -x -q` from the repo root, where `scripts/` is
not a package and not on sys.path; `from clean_cohort import run_clean` raised ModuleNotFoundError
on the GitHub runner (CI run #233) even though it imported in the local dev environment. Walking
ancestors (rather than hardcoding depth) keeps this correct if the tests tree is ever moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_here = Path(__file__).resolve()
for _anc in (_here.parent, *_here.parents):
    _scripts = _anc / "scripts"
    if _scripts.is_dir() and (_scripts / "clean_cohort.py").is_file():
        for _p in (str(_scripts), str(_anc)):  # scripts dir + repo root
            if _p not in sys.path:
                sys.path.insert(0, _p)
        break


# ---------------------------------------------------------------------------
# sys.path leak guard (added 2026-07-11 -- cluster E)
# ---------------------------------------------------------------------------
# A test must not permanently mutate sys.path. Global-state pollution of this kind
# is invisible in isolation and only detonates later, in an unrelated test.
#
# It has already cost us once. tests/test_rekey_seq_windows_v2.py wrote a COUNTERFEIT
# `genomic_variant_classifier` package (real __init__.py; only a `data` subpackage)
# into a tmp dir and published it with a bare
#
#     sys.path.insert(0, str(tmp_path / "gvc_join"))     # never removed
#
# Nothing broke in-process -- the real package was already in sys.modules and was
# never re-resolved. But five later tests launch a CHILD interpreter with
# PYTHONPATH built from sys.path. The child starts with an empty sys.modules, so it
# resolved the counterfeit first: the top-level import SUCCEEDED and every subpackage
# import failed --
#
#     ModuleNotFoundError: No module named 'genomic_variant_classifier.evaluation'
#     ModuleNotFoundError: No module named 'genomic_variant_classifier.agent_layer'
#
# Green in isolation (38 passed), red in the full suite. Five tests, unexplained for
# three days. See docs/status/REMEDIATION_2026-07-11_test-suite-red.md, cluster E.
#
# This fixture makes that failure mode IMPOSSIBLE TO INTRODUCE SILENTLY: any test that
# leaves sys.path modified now fails loudly, in the test that did it, naming the
# entries it added or removed. Tests that legitimately need a path change must use
# `monkeypatch.syspath_prepend`, which pytest reverts at teardown.
# Design notes, so this guard is neither too loose nor too brittle:
#
#  * It compares SETS, not lists. Several existing tests legitimately call
#    sys.path.insert() inside the test body with a path that tests/conftest.py has
#    ALREADY added (e.g. the scripts dir). That merely creates a duplicate entry --
#    harmless, and a strict list comparison would fail those tests for no reason.
#    Only a genuinely NEW entry, or a VANISHED one, can change what an import resolves
#    to. Those are the two conditions worth failing on.
#
#  * It RESTORES sys.path unconditionally, before raising. A leaking test must not be
#    able to cascade into the rest of the suite -- the offender fails, and only the
#    offender. Without this, the guard would report the leak and then let it corrupt
#    everything downstream anyway, which is the exact behaviour we are eliminating.
@pytest.fixture(autouse=True)
def _no_sys_path_leaks():
    before = list(sys.path)
    try:
        yield
    finally:
        after = list(sys.path)
        added = [p for p in dict.fromkeys(after) if p not in set(before)]
        removed = [p for p in dict.fromkeys(before) if p not in set(after)]
        # Restore FIRST -- containment before reporting.
        sys.path[:] = before
        if added or removed:
            raise AssertionError(
                "This test leaked a permanent change to sys.path. That corrupts every "
                "later test, and every subprocess they launch with a PYTHONPATH built "
                "from sys.path (the child starts with an empty sys.modules and will "
                "resolve the leaked path FIRST).\n"
                f"  ADDED  : {added}\n"
                f"  REMOVED: {removed}\n"
                "Use `monkeypatch.syspath_prepend(...)` rather than `sys.path.insert(...)`"
                " -- pytest reverts it at teardown. Precedent: "
                "tests/test_rekey_seq_windows_v2.py. Full write-up: "
                "docs/status/REMEDIATION_2026-07-11_test-suite-red.md (cluster E)."
            )
