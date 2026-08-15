"""Ignore rules must distinguish source packages from generated artifacts.

REPORTS-DIR-IGNORED-1
=====================
`.gitignore:101` read `reports/`. A pattern with no leading slash matches at ANY
DEPTH, so it caught four directories, not the one it was written for:

    reports/                                     41 files   the intended target
    notebooks/genomic_variant_classifier/reports/ 1 file    notebook output
    src/genomic_variant_classifier/agent_layer/reports/ 3   stray agent output
    src/genomic_variant_classifier/reports/       2 files   A SOURCE PACKAGE

PROVEN BY PROBE, not by reading the pattern: an untracked `.py` written into the
source package was reported ignored by `git check-ignore` and did not appear in
`git status` at all. Its two existing files survive only because they were added
before the rule; the next module added there would vanish without a word.

That is the torch_geometric shape -- an absence that produces no signal. This
repository has one recorded instance of it lasting 508 continuous-integration
runs.

WHY THESE TESTS PROBE BEHAVIOUR AND NOT TEXT
A test asserting `"/reports/" in gitignore_text` would pass against a file whose
rules had been reordered into uselessness, and would fail against a correct
rewrite that expressed the same boundary differently. What matters is what git
DOES with a path, so each case asks git directly via

    git check-ignore --no-index <path>

`--no-index` evaluates the rules WITHOUT consulting the index, so a tracked
file's status cannot mask the answer -- which is exactly how the original
investigation was misled: `check-ignore` returned nothing for the two tracked
files while the directory was plainly ignored.

THE INVARIANT THIS ENCODES
    Generated artifacts are ignored.
    Source is never ignored.
    Stray generated output INSIDE the source tree is VISIBLE, not hidden.

The third clause matters because AGENT-ROOT-ANCHOR-1 anchored five agents to
PROJECT_ROOT so their reports land at the repository root. If output ever
appears under src/ again, that is a regression and `git status` must say so
rather than silently swallowing it.

Author: Monzia Moodie
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: (path, must_be_ignored, why)
#:
#: These paths are SENTINELS. They need not exist -- `--no-index` evaluates the
#: rules against the path itself -- so the test states the intended boundary
#: without creating files or depending on what happens to be on disk.
CASES = (
    ("reports/generated.html", True,
     "the root reports/ directory holds 41 generated artifacts"),
    ("reports/data_freshness/FRESHNESS_2026-06-30.md", True,
     "agent reports land here now that five agents anchor to PROJECT_ROOT"),
    ("reports/agent_ops/OPS_2026-06-20.md", True,
     "same, for the agent_ops series"),
    ("notebooks/genomic_variant_classifier/reports/stage2_summary.json", True,
     "notebook run output, generated 2026-03-15"),
    ("outputs/drift_reports/concept_drift/report.json", True,
     "drift agents write under outputs/, which is separately ignored"),
    (".gvc-state/literature_scout/state.json", True,
     "CANONICAL MUTABLE STATE. RuntimePaths puts the literature-scout "
     "store here, and version_monitor_agent creates it on first write. "
     "Measured 2026-08-15: before this rule, git check-ignore returned "
     "NOTHING for it -- so the first real agent run would have left an "
     "untracked directory that someone eventually commits. This is the "
     "state analogue of REPORTS-DIR-IGNORED-1, inverted: canonical state "
     "MUST be ignored, while state appearing under src/ must be VISIBLE"),

    ("src/genomic_variant_classifier/reports/__init__.py", False,
     "THE DEFECT: this is a source package, not an artifact directory"),
    ("src/genomic_variant_classifier/reports/report_generator.py", False,
     "23,530 bytes of production source"),
    ("src/genomic_variant_classifier/reports/a_new_module.py", False,
     "a file that does not exist yet -- the case that was silently swallowed"),
    ("src/genomic_variant_classifier/agent_layer/reports/agent_ops/OPS.md", False,
     "stray agent output under src/ must be VISIBLE, so a regression of "
     "AGENT-ROOT-ANCHOR-1 shows up in git status instead of being hidden"),
)


def _check_ignore(path: str) -> bool:
    """True if git would ignore `path`. Asks git, does not parse .gitignore."""
    out = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", path],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60)
    # 0 = ignored, 1 = not ignored, 128 = error.
    if out.returncode not in (0, 1):
        pytest.fail("git check-ignore failed ({}): {}".format(
            out.returncode, out.stderr.strip()[:200]))
    return out.returncode == 0


def test_git_is_available_and_check_ignore_discriminates():
    """A guard on the guard.

    If git were unavailable or check-ignore always returned the same code,
    every case below would pass or fail together and prove nothing. This asserts
    the instrument distinguishes two paths whose status is not in question.
    """
    assert _check_ignore(".venv312/lib/site-packages/x.py") is True
    assert _check_ignore("README.md") is False


@pytest.mark.parametrize(
    "path,ignored,why", CASES,
    ids=[c[0].replace("/", "|") for c in CASES])
def test_the_ignore_boundary_separates_artifacts_from_source(path, ignored, why):
    got = _check_ignore(path)
    assert got == ignored, (
        "{}\n  expected ignored={}, got {}\n  {}".format(path, ignored, got, why))


def test_a_bare_directory_rule_would_reach_into_src():
    """Why the leading slash is load-bearing, stated as a property.

    `reports/` and `/reports/` differ only in that slash, and the difference is
    whether a source package disappears. This asserts the two source paths and
    the root artifact path cannot BOTH be ignored -- which is precisely what a
    bare rule produced.
    """
    root_artifact = _check_ignore("reports/generated.html")
    source_module = _check_ignore("src/genomic_variant_classifier/reports/x.py")
    assert root_artifact is True, "the root artifact directory must stay ignored"
    assert source_module is False, "the source package must never be ignored"


def test_no_generated_report_directory_exists_under_src():
    """The stronger invariant: not "we know which artifact dirs to hide", but
    "no artifact directory exists beneath src/ at all".

    Three stray reports were moved to the repository root when
    AGENT-ROOT-ANCHOR-1 landed. If agents ever write under src/ again, this
    fails -- which is a louder and earlier signal than an ignore rule.
    """
    strays = sorted(
        str(p.relative_to(_REPO_ROOT)).replace("\\", "/")
        for p in (_REPO_ROOT / "src").rglob("reports")
        if p.is_dir() and any(p.iterdir())
        and p != _REPO_ROOT / "src" / "genomic_variant_classifier" / "reports"
    )
    assert not strays, (
        "generated report directory/directories under src/: {}. Agents anchor "
        "to PROJECT_ROOT; output belongs at the repository root.".format(strays))


def test_the_source_reports_package_is_tracked():
    """Both files must be in the index, or the boundary is theoretical."""
    out = subprocess.run(
        ["git", "ls-files", "src/genomic_variant_classifier/reports"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr.strip()[:200]
    tracked = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    for expected in ("src/genomic_variant_classifier/reports/__init__.py",
                     "src/genomic_variant_classifier/reports/report_generator.py"):
        assert expected in tracked, (expected, sorted(tracked))
