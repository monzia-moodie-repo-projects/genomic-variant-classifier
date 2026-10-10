"""PATH-BUDGET-1: every tracked path fits the Windows limits from any checkout or clone root of up to 108 characters.

MEASURED 2026-10-10 (owner upload 544a2b43): the lockfile-migration installer's apply stage failed twice on Windows. Its clone root
C:\\Users\\monzi\\AppData\\Local\\Temp\\gvc_lockfile_migration_candidate_<23-character stamp> is 90 characters and the longest path the
patch wrote was 169: 260 in all, one beyond the 259 a Windows path may hold. Every installer simulation ran on Linux, which has no
such limit, so the repository itself must hold the budget (path_budget.py) and these tests make it a property of the repository.

The budget is checked against what GIT tracks (`git ls-files -z`), the set a clone materialises -- not the working tree, which also
holds caches and virtual environments that are never cloned.

Author: Monzia Moodie
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.repository_records.identity import ArtifactInstance
from genomic_variant_classifier.repository_records.path_budget import (
    MAX_TRACKED_DIRECTORY_PATH_CHARS, MAX_TRACKED_FILE_PATH_CHARS, ROOT_ALLOWANCE_CHARS, WINDOWS_MAX_DIRECTORY_PATH_CHARS,
    WINDOWS_MAX_FILE_PATH_CHARS, PathBudgetError, budget_violations, require_within_budget, windows_length)
from genomic_variant_classifier.repository_records.roles import RecordsOntologyError

_REPO = Path(__file__).resolve().parents[2]
#: The clone root the failed run used, exactly as the installer composed it (owner upload 544a2b43, line 659).
_MEASURED_CLONE_ROOT = r"C:\Users\monzi\AppData\Local\Temp\gvc_lockfile_migration_candidate_20261010T0440467314849Z"
#: The path whose application failed (the per-part layout, retired 2026-10-10).
_INCIDENT_PATH = ("records/migrations/environment-qualification/lockfile/REC-b0b49d619cc44e76ac3cba537807c400/artifacts/"
                  "regenerated_proposal/lock_transition_proposal_regenerated_bound.json")


def _tracked() -> list:
    out = subprocess.run(["git", "-C", str(_REPO), "ls-files", "-z"], capture_output=True, timeout=120)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    paths = [p for p in out.stdout.decode("utf-8").split("\0") if p]
    assert len(paths) > 1000, len(paths)       # a non-discriminating listing (wrong directory, empty index) fails here
    return paths


_HAS_GIT = (_REPO / ".git").exists()


# ------------------------------------------------------------------ the arithmetic and the incident

def test_the_budget_is_derived_from_the_two_windows_limits_and_the_root_allowance():
    assert (WINDOWS_MAX_FILE_PATH_CHARS, WINDOWS_MAX_DIRECTORY_PATH_CHARS) == (260 - 1, 260 - 12 - 1)
    assert ROOT_ALLOWANCE_CHARS + 1 + MAX_TRACKED_FILE_PATH_CHARS == WINDOWS_MAX_FILE_PATH_CHARS
    assert ROOT_ALLOWANCE_CHARS + 1 + MAX_TRACKED_DIRECTORY_PATH_CHARS == WINDOWS_MAX_DIRECTORY_PATH_CHARS
    assert (MAX_TRACKED_FILE_PATH_CHARS, MAX_TRACKED_DIRECTORY_PATH_CHARS) == (150, 138)


def test_the_measured_clone_root_is_within_the_allowance_and_the_incident_reproduces():
    assert len(_MEASURED_CLONE_ROOT) == 90 <= ROOT_ALLOWANCE_CHARS
    full = _MEASURED_CLONE_ROOT + "\\" + _INCIDENT_PATH.replace("/", "\\")
    assert len(full) == 260 > WINDOWS_MAX_FILE_PATH_CHARS             # exactly the failure
    assert len(_INCIDENT_PATH) == 169
    assert budget_violations(_INCIDENT_PATH) and "169 characters" in budget_violations(_INCIDENT_PATH)[0]


@pytest.mark.parametrize("length, fits", [(149, True), (150, True), (151, False)])
def test_the_file_boundary_is_exact(length, fits):
    path = "d/" + "f" * (length - 2)
    assert len(path) == length and (budget_violations(path) == []) is fits


@pytest.mark.parametrize("length, fits", [(137, True), (138, True), (139, False)])
def test_the_directory_boundary_is_exact(length, fits):
    directory = "d" * length
    problems = budget_violations(directory + "/f")
    assert [p for p in problems if p.startswith("directory")] == ([] if fits else problems)


def test_length_is_counted_in_utf16_code_units():
    assert windows_length("a") == 1 and windows_length("\u00e9") == 1 and windows_length("\U0001F600") == 2
    with pytest.raises(PathBudgetError):          # measured directly too: a non-text value is refused, never measured
        windows_length(None)
    assert budget_violations("d/" + "\U0001F600" * 74) == []                       # 2 + 148 = 150 units
    assert budget_violations("d/" + "\U0001F600" * 74 + "x") != []                 # 151 units, although only 77 characters


@pytest.mark.parametrize("bad", ["", " a", "a ", "C:\\x", "/abs/x", "a\\b", None, 7])
def test_a_non_path_is_refused_not_measured(bad):
    with pytest.raises(PathBudgetError):
        budget_violations(bad)


def test_require_within_budget_raises_with_every_violation():
    with pytest.raises(PathBudgetError) as exc:
        require_within_budget("d" * 139 + "/" + "f" * 20)
    assert str(exc.value).startswith("file path") and "; directory" in str(exc.value)
    assert issubclass(PathBudgetError, RecordsOntologyError)


def test_every_record_artifact_instance_is_held_to_the_budget():
    ok = ArtifactInstance(content_sha256="a" * 64, canonical_path="records/x/" + "f" * 140, size_bytes=1)
    assert len(ok.canonical_path) == 150
    with pytest.raises(PathBudgetError):
        ArtifactInstance(content_sha256="a" * 64, canonical_path=_INCIDENT_PATH, size_bytes=1)


# ------------------------------------------------------------------ the repository property

@pytest.mark.skipif(not _HAS_GIT, reason="{} is not a git working tree; nothing tracked to measure".format(_REPO))
def test_every_tracked_path_is_within_the_budget():
    tracked = _tracked()
    offenders = {p: budget_violations(p) for p in tracked if budget_violations(p)}
    assert offenders == {}, offenders


@pytest.mark.skipif(not _HAS_GIT, reason="{} is not a git working tree; nothing tracked to measure".format(_REPO))
def test_the_instrument_discriminates():
    """The listing is THIS repository's, and the measure would catch the incident path if it were tracked."""
    tracked = _tracked()
    assert "tests/unit/test_gitattributes_contract.py" in tracked and "pyproject.toml" in tracked
    assert _INCIDENT_PATH not in tracked and budget_violations(_INCIDENT_PATH) != []
