"""A warning filter may not outlive the thing it was written for.

TEST-MODULE-SUPPRESSES-ALL-WARNINGS-1. Created 2026-08-26.

WHAT THIS GUARDS
----------------
`warnings.filterwarnings` and `warnings.simplefilter` MUTATE A PROCESS-WIDE
list. Called at module level, they apply from the moment that module is
imported and for the remainder of the run -- so a filter written for one test
silently governs every test collected after it.

MEASURED 2026-08-26, five such calls existed:

    tests/unit/test_ablate_gnn.py:9         filterwarnings("ignore")
    tests/unit/test_gnn_gps.py:2            filterwarnings("ignore")
    tests/unit/test_gnn_tier2_denoise.py:2  filterwarnings("ignore")
    tests/unit/test_gnn_typed_output.py:39  filterwarnings("ignore")
    models/variant_ensemble.py:142          filterwarnings("ignore", UserWarning)

Four were BARE -- no category at all. The fifth was in the LIBRARY, so it
applied to every consumer including the inference interface, not merely the
suite.

AND THEY SUPPRESSED NOTHING. Measured by removing them inside a repository
transaction and running the four modules: 30 passed with ZERO warnings both
before and after. Zero local benefit, process-wide cost.

THE COST WAS OBSERVED. GATE-WARNING-COUNT-UNSTABLE-1: the same suite reported
33 warnings on three runs and 914 on one, with no source change between them.
`src/` contains ZERO `warnings.warn` call sites -- every warning in the gate
comes from a third-party library -- so whether one is reported at all depended
on whether a graph-neural-network module had been imported first. That is
collection order, and it should not decide what a run reports.

WHY A GUARD RATHER THAN JUST A DELETION
---------------------------------------
`pyproject.toml` already carries the correct pattern: two filters, each pinned
to an exact MESSAGE, each with a paragraph of measured justification, and an
explicit instruction beside one of them --

    DO NOT broaden this to `ignore::UserWarning`. It is pinned to this exact
    message so a DIFFERENT UserWarning still reaches us.

Deleting five calls fixes today. This test is what stops the sixth.

WHAT IS STILL ALLOWED
---------------------
A filter inside `warnings.catch_warnings()` is RESTORED on exit and is not
touched here -- ten such calls exist and all ten are legitimate. A filter in
`pyproject.toml` is declared once, reviewed in one place, and applies to the
whole run by design rather than by import accident.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
PYPROJECT = ROOT / "pyproject.toml"

#: Mutating calls. `catch_warnings` is a context manager and restores; these do
#: not.
MUTATORS = ("filterwarnings", "simplefilter")


def module_level_filters(source: str):
    """Filter mutations in `tree.body` -- module level BY CONSTRUCTION.

    Parsed, not searched. A call inside a function or a `with` block is not in
    `tree.body`, so scoped filters are excluded structurally rather than by a
    pattern that might or might not notice the indentation.
    """
    found = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in MUTATORS:
            found.append((node.lineno, ast.unparse(node.value)[:70]))
    return found


def _scan(root: Path, base: Path = None):
    """Every module-level filter under `root`, named relative to `base`.

    `base` defaults to `root`. An earlier version hardcoded `relative_to(ROOT)`
    and raised ValueError the moment the scan was pointed anywhere outside the
    repository -- which is exactly what `test_the_scan_finds_a_planted_offender`
    does, and it failed on its first run.

    That is the guard-the-guard test doing its job: a helper that cannot be
    exercised on a fixture cannot be shown to work at all.
    """
    anchor = base if base is not None else root
    offenders = []
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:                            # pragma: no cover
            continue
        try:
            hits = module_level_filters(source)
        except SyntaxError:                        # pragma: no cover
            continue
        for line, call in hits:
            try:
                shown = path.relative_to(anchor).as_posix()
            except ValueError:                     # pragma: no cover
                shown = path.as_posix()
            offenders.append("{}:{}  {}".format(shown, line, call))
    return offenders


def test_no_module_level_warning_filter_in_the_package():
    """A library that silences warnings silences them for its CONSUMERS.

    `variant_ensemble.py` did this with `category=UserWarning`, so every
    caller -- including the inference interface -- inherited it by importing
    the ensemble.
    """
    offenders = _scan(SRC, ROOT)
    assert not offenders, (
        "these modules install a process-wide warning filter at import:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse `with warnings.catch_warnings():` around the narrow scope "
          "that needs it, or pin a message in pyproject.toml.")


def test_no_module_level_warning_filter_in_the_suite():
    """A filter written for one test governs every test collected after it."""
    offenders = _scan(TESTS, ROOT)
    assert not offenders, (
        "these test modules install a process-wide warning filter at import:\n"
        "  " + "\n  ".join(offenders)
        + "\n\nMEASURED 2026-08-26: removing five such filters changed the "
          "affected modules' own results by nothing -- 30 passed, 0 warnings, "
          "before and after. The cost was borne entirely by other tests.")


def test_the_scan_finds_a_planted_offender(tmp_path):
    """Guards the guard.

    A structural search that matched nothing would pass over an empty result
    and report green forever -- the vacuous-check shape this repository has
    found repeatedly. Proven on a synthetic offender so the silence above
    means something.
    """
    planted = tmp_path / "offender.py"
    planted.write_text(
        'import warnings\n\nwarnings.filterwarnings("ignore")\n\n\n'
        'def test_x():\n    pass\n', encoding="utf-8")
    offenders = _scan(tmp_path)
    assert len(offenders) == 1, offenders
    assert "offender.py:3" in offenders[0]


def test_a_scoped_filter_is_not_an_offender(tmp_path):
    """The permissive direction, and it matters.

    A rule that also refused `catch_warnings` would push the next author to
    delete a legitimate scoped filter -- or to stop scoping at all. Ten such
    calls exist in this repository and every one is correct.
    """
    scoped = tmp_path / "scoped.py"
    scoped.write_text(
        'import warnings\n\n\ndef test_y():\n'
        '    with warnings.catch_warnings():\n'
        '        warnings.simplefilter("error")\n'
        '        pass\n', encoding="utf-8")
    assert _scan(tmp_path) == []


def test_pyproject_filters_are_pinned_to_messages():
    """The declared filters must not be bare categories.

    `pyproject.toml` says it itself: "DO NOT broaden this to
    `ignore::UserWarning`. It is pinned to this exact message so a DIFFERENT
    UserWarning still reaches us." A bare `ignore::SomeWarning` entry would
    hide the next real signal of that class.
    """
    assert PYPROJECT.is_file(), PYPROJECT
    text = PYPROJECT.read_text(encoding="utf-8")
    start = text.find("filterwarnings")
    assert start != -1, "pyproject.toml declares no filterwarnings list"
    block = text[start:text.find("]", start) + 1]
    entries = [line.strip().strip(",").strip('"')
               for line in block.split("\n")
               if line.strip().startswith('"')]
    assert entries, "the filterwarnings list is empty"
    # A pytest filter is `action:message:category:module:lineno`. The MESSAGE
    # is field 1, and `ignore::UserWarning` has an EMPTY one -- which is the
    # exact form pyproject.toml warns against.
    #
    # An earlier version of this check counted COLONS: `ignore::UserWarning`
    # has two, so it passed. Sabotage caught it, and it was the one case that
    # mattered most. Counting separators is not the same as reading a field.
    bare = []
    for entry in entries:
        fields = entry.split(":")
        action = fields[0].strip()
        if action in ("error", "always", "default", "module", "once"):
            continue          # an action with no target is a global POLICY
        message = fields[1].strip() if len(fields) > 1 else ""
        if not message:
            bare.append(entry)
    assert not bare, (
        "these entries suppress a whole CATEGORY with no message pinned: {}.\n"
        "pyproject.toml states the rule beside its own filter: \"DO NOT "
        "broaden this to `ignore::UserWarning`. It is pinned to this exact "
        "message so a DIFFERENT UserWarning still reaches us.\"".format(bare))


@pytest.mark.parametrize(
    "path",
    ["tests/unit/test_ablate_gnn.py", "tests/unit/test_gnn_gps.py",
     "tests/unit/test_gnn_tier2_denoise.py",
     "tests/unit/test_gnn_typed_output.py",
     "src/genomic_variant_classifier/models/variant_ensemble.py"],
    ids=["ablate_gnn", "gnn_gps", "gnn_tier2_denoise", "gnn_typed_output",
         "variant_ensemble"])
def test_the_five_repaired_modules_stay_clean(path):
    """Named, so a regression in one of them says WHICH.

    The scans above would catch a reintroduction anywhere; these five are
    called out because they are where it happened, and a failure that names
    the file is a failure someone can act on immediately.
    """
    target = ROOT / path
    assert target.is_file(), target
    hits = module_level_filters(target.read_text(encoding="utf-8"))
    assert not hits, "{} reinstalled a module-level filter: {}".format(path, hits)
