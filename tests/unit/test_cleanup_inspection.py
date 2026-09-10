"""Cleanup inspection: A-F discovery, tri-state observation, disabled apply.

Author: Monzia Moodie

Every case below was REPRODUCED against the implementation before the
correction that refuses it. The originals are recorded beside each test.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from genomic_variant_classifier.repository_hygiene.cleanup_categories import (
    CATEGORY_INSPECTORS, Candidate, CategoryInspection, ConditionState,
    InspectionState, capture_head, inspect_category_a, inspect_category_b,
    inspect_category_c, inspect_category_d, inspect_category_e,
    inspect_category_f, inspect_cleanup_candidates, observed_size,
    require_category_coverage)
from genomic_variant_classifier.repository_hygiene.repository_inspection import (
    InspectionError, RepositoryInspection, RepositorySelectionError,
    TrackingState, select_repository)

ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)


def _git(repo, *args, check=True):
    done = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, env=ENV)
    if check and done.returncode != 0:
        raise RuntimeError(done.stderr.decode())
    return done


@pytest.fixture
def populated(tmp_path):
    """One genuine candidate for every category A-F."""
    repo = tmp_path / "repo"
    for part in ("scripts", "data/processed/seq_windows",
                 "data/external/dbnsfp", "sub"):
        (repo / part).mkdir(parents=True, exist_ok=True)
    (repo / "scripts/train.py").write_text("live\n")
    (repo / "scripts/train.py.w1bak").write_text("bak\n")
    (repo / ".gitignore").write_text("install_*.py\n")
    (repo / ".gitignore.prebakfix").write_text("old\n")
    (repo / "data/processed/seq_windows/seq_windows.parquet").write_text("m\n")
    (repo / "data/processed/seq_windows/part_0.parquet").write_text("p\n")
    (repo / "data/external/dbnsfp/dbnsfp_full_index.parquet").write_text("i\n")
    (repo / "data/external/dbnsfp/"
            "dbnsfp_full_index.parquet.OOMbak").write_text("o\n")
    (repo / "install_thing.py").write_text("i\n")
    (repo / "scripts/dump_x.py").write_text("d\n")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "scripts/train.py", ".gitignore")
    _git(repo, "commit", "-qm", "v1")
    return repo


@pytest.fixture
def committed_empty(tmp_path):
    """A valid repository with a commit and no cleanup candidates at all."""
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "v1")
    return repo


def test_every_category_discovers_its_candidate(populated):
    proposal = inspect_cleanup_candidates(select_repository(str(populated)))
    assert [row.category for row in proposal.categories] == \
        [name for name, _ in CATEGORY_INSPECTORS]
    assert all(row.state is InspectionState.COMPLETE
               for row in proposal.categories)
    assert all(len(row.candidates) == 1 for row in proposal.categories)
    assert all(c.conditions is ConditionState.SATISFIED
               for row in proposal.categories for c in row.candidates)


def test_counts_are_three_distinct_quantities(populated):
    proposal = inspect_cleanup_candidates(select_repository(str(populated)))
    assert proposal.category_findings == 6
    assert proposal.unique_candidate_paths == 6
    assert proposal.conditions_determined == 6
    assert proposal.inspection_failures == 0


def test_zero_is_a_measured_result_only_after_a_complete_search(
        committed_empty):
    proposal = inspect_cleanup_candidates(
        select_repository(str(committed_empty)))
    assert proposal.category_findings == 0
    assert proposal.inspection_status == "completed"


def test_a_missing_committed_counterpart_is_not_an_inspection_failure(
        committed_empty):
    """MEASURED: `git cat-file -e HEAD:<absent>` exits 128, not 1, so the
    previous implementation returned UNKNOWN and category A reported
    INCOMPLETE with zero candidates for an ordinary absence."""
    (committed_empty / "scripts").mkdir()
    (committed_empty / "scripts/train.py.w1bak").write_text("bak\n")
    result = inspect_category_a(committed_empty,
                                RepositoryInspection(root=committed_empty))
    assert result.state is InspectionState.COMPLETE
    assert len(result.candidates) == 1
    assert result.candidates[0].conditions is ConditionState.NOT_SATISFIED


def test_head_is_resolved_once_and_shared(populated, monkeypatch):
    """MEASURED: A called head_paths(), B called it AGAIN, and head_paths()
    re-runs `rev-parse HEAD` each time -- so A and B could inspect different
    commits. The docstring claimed otherwise; the claim was a comment."""
    inspection = select_repository(str(populated))
    calls = []
    original = type(inspection).head_paths

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(type(inspection), "head_paths", counting)
    inspect_cleanup_candidates(inspection)
    assert len(calls) == 1, "head_paths ran {} times".format(len(calls))


def test_a_failed_head_retains_candidates_with_unknown_conditions(tmp_path):
    """A carried failure must not drop a path already discovered."""
    outside = tmp_path / "outside"
    (outside / "scripts").mkdir(parents=True)
    (outside / "scripts/train.py.w1bak").write_text("b\n")
    head = capture_head(RepositoryInspection(root=outside))
    assert head.failure
    result = inspect_category_a(outside, RepositoryInspection(root=outside),
                                head)
    assert result.state is InspectionState.INCOMPLETE
    assert len(result.candidates) == 1
    assert result.candidates[0].conditions is ConditionState.UNKNOWN


def test_head_membership_comes_from_an_enumerated_tree(populated):
    inspection = RepositoryInspection(root=populated)
    commit, paths = inspection.head_paths()
    assert len(commit) == 40
    assert "scripts/train.py" in paths
    assert "scripts/dump_x.py" not in paths


def test_an_unresolvable_head_is_an_inspection_failure(tmp_path):
    outside = tmp_path / "outside"
    (outside / "scripts").mkdir(parents=True)
    (outside / "scripts/train.py.w1bak").write_text("b\n")
    result = inspect_category_a(outside, RepositoryInspection(root=outside))
    assert result.state is InspectionState.INCOMPLETE
    assert result.failures


def test_git_failure_cannot_become_eligibility(tmp_path):
    """The demonstrated category-F defect: outside a repository the original
    condition `ok = not tracked(p)` made every candidate eligible."""
    outside = tmp_path / "outside"
    (outside / "scripts").mkdir(parents=True)
    (outside / "scripts/dump_y.py").write_text("d\n")
    result = inspect_category_f(outside, RepositoryInspection(root=outside))
    assert result.state is InspectionState.INCOMPLETE
    assert len(result.candidates) == 1
    assert result.candidates[0].conditions is ConditionState.UNKNOWN
    assert result.failures


def test_a_discovered_candidate_is_never_dropped(tmp_path):
    """Paths discovered and paths whose conditions could be determined are
    two populations. The previous code `continue`d past the first."""
    outside = tmp_path / "outside"
    (outside / "scripts").mkdir(parents=True)
    (outside / "scripts/dump_a.py").write_text("a\n")
    (outside / "scripts/patch_b.py").write_text("b\n")
    result = inspect_category_f(outside, RepositoryInspection(root=outside))
    assert len(result.candidates) == 2
    assert all(c.conditions is ConditionState.UNKNOWN
               for c in result.candidates)


def test_observed_size_raises_rather_than_returning_zero():
    """MEASURED: the previous `_size` returned 0 for a PermissionError, so a
    category could report a failed nonempty condition while marked
    COMPLETE."""
    class Denied:
        def stat(self):
            raise PermissionError(13, "Permission denied")

    with pytest.raises(InspectionError):
        observed_size(Denied())


@pytest.mark.parametrize("state", ["complete", 1, None, True])
def test_a_non_enum_state_is_refused(state):
    """MEASURED: CategoryInspection('A', 'complete') constructed, and
    rendering would then fail at `.value`."""
    with pytest.raises(ValueError):
        CategoryInspection("A", state)


@pytest.mark.parametrize("conditions", [True, False, "satisfied", None])
def test_a_non_enum_condition_is_refused(conditions):
    with pytest.raises(ValueError):
        Candidate("x", conditions, "")


@pytest.mark.parametrize("kwargs", [
    {"state": InspectionState.COMPLETE, "failures": ("x",)},
    {"state": InspectionState.INCOMPLETE},
    {"state": InspectionState.NOT_RUN,
     "candidates": (Candidate("a", ConditionState.SATISFIED, ""),)},
])
def test_contradictory_inspection_shapes_are_unconstructible(kwargs):
    with pytest.raises(ValueError):
        CategoryInspection(category="A", **kwargs)


def test_category_coverage_is_validated(populated):
    proposal = inspect_cleanup_candidates(select_repository(str(populated)))
    expected = [name for name, _ in CATEGORY_INSPECTORS]
    with pytest.raises(InspectionError):
        require_category_coverage(expected, proposal.categories[:5])
    with pytest.raises(InspectionError):
        require_category_coverage(
            expected, proposal.categories + (proposal.categories[0],))


def test_tracking_is_tri_state(populated, tmp_path):
    inspection = RepositoryInspection(root=populated)
    assert inspection.tracking("scripts/train.py").state is \
        TrackingState.TRACKED
    assert inspection.tracking("scripts/dump_x.py").state is \
        TrackingState.UNTRACKED
    outside = tmp_path / "nowhere"
    outside.mkdir()
    assert RepositoryInspection(root=outside).tracking("x").state is \
        TrackingState.UNKNOWN


def test_subdirectory_selection_is_refused(populated):
    with pytest.raises(RepositorySelectionError):
        select_repository(str(populated / "sub"))


def test_root_selection_is_preserved(populated):
    assert select_repository(str(populated)).root == populated.resolve()


def test_a_non_repository_is_refused(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RepositorySelectionError):
        select_repository(str(plain))


def test_a_linked_worktree_is_accepted(populated, tmp_path):
    linked = tmp_path / "linked"
    done = subprocess.run(
        ["git", "-C", str(populated), "worktree", "add", "-q", str(linked),
         "-b", "wt"], capture_output=True, env=ENV)
    if done.returncode != 0:
        pytest.skip("this git build did not create a linked worktree")
    assert select_repository(str(linked)).root == linked.resolve()


def test_inspection_mutates_nothing(populated):
    def snapshot():
        return {p: p.read_bytes() for p in sorted(populated.rglob("*"))
                if p.is_file() and ".git" not in p.parts}

    before = snapshot()
    inspect_cleanup_candidates(select_repository(str(populated)))
    assert snapshot() == before
