"""An installer is one atomic repository state transition.

INSTALLER-TRANSACTION-1 step 4.

    Before the first byte of the repository changes, the system must possess an
    immutable description of the complete transition it intends to perform;
    after the last byte changes, it must prove that the observed repository
    equals that description plus no unintended transition.

WHAT THIS REPLACES
Every installer before this one wrote imperatively -- patch a file, write
another, edit the ratchet, edit the README -- with the ratchet and README
handled OUTSIDE any transaction by a PowerShell helper that created `.bak_`
siblings. Nothing declared the complete set in advance, so nothing could detect
an installer that touched one file more than it meant to.

    There are no transactional "payload files" and nontransactional
    "bookkeeping files".

MEASURED 2026-08-20, before this module existed:

    RepositoryTransaction exposed NO public accessor for its target set, so a
    runner comparing declared against actual would have reached into the
    private `_targets`.

    The journal resolves to <LOCALAPPDATA>/GenomicVariantClassifier/
    transactions -- OUTSIDE the repository -- so the hygiene invariant needs no
    journal exception.

THE RATCHET IS DERIVED, NEVER TRANSCRIBED
Hand-carried counts produced repeated stale-count incidents: on 2026-08-20
alone a file expected to collect 52 cases collected 54, and another expected 38
collected 44. A plan therefore carries a DELTA; the runner measures collection
and renders both counters from one number.

Author: Monzia Moodie
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.transactions.install_plan import (
    DerivedCount, InstallPlan, PlanError, PlannedTarget, TargetAction,
    WriteSetViolation, render_ratchet, render_readme,
)
from genomic_variant_classifier.transactions.repository_transaction import (
    RepositoryTransaction,
)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "src" / "mod.py").write_bytes(b"old = 1\n")
    (r / "tests" / "EXPECTED_SUITE_SIZE").write_bytes(b"# log\n# older\n100\n")
    (r / "README.md").write_bytes(b"x\nbadge/tests-100-brightgreen\ny\n")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=str(r), capture_output=True, timeout=120)
    return r


def _plan(rat: bytes, rdm: bytes) -> InstallPlan:
    return InstallPlan(
        unit="DEMO-1",
        targets=(
            PlannedTarget("src/mod.py", TargetAction.PATCH, b"new = 2\n"),
            PlannedTarget("src/added.py", TargetAction.CREATE, b"fresh = 3\n"),
            PlannedTarget("tests/EXPECTED_SUITE_SIZE", TargetAction.PATCH, rat),
            PlannedTarget("README.md", TargetAction.PATCH, rdm),
        ),
        expected_delta=3)


# ---- the write-set contract, which is the point ------------------------
def test_an_UNDECLARED_write_is_refused(repo, tmp_path):
    """WRITE-SET COMPLETENESS.

    An installer that unexpectedly touches one more file fails here EVEN IF
    every test passes. No installer before step 4 could detect that, because
    none declared its complete set in advance.
    """
    (repo / "src" / "extra.py").write_bytes(b"e = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "v2"], cwd=str(repo), capture_output=True)
    plan = InstallPlan(unit="D", targets=(
        PlannedTarget("src/mod.py", TargetAction.PATCH, b"z = 9\n"),))

    with pytest.raises(WriteSetViolation) as exc:
        with RepositoryTransaction(repo, tmp_path / "j") as tx:
            tx.patch("src/mod.py", b"z = 9\n")
            tx.patch("src/extra.py", b"UNDECLARED\n")
            plan.assert_write_set(tx)
            tx.commit()
    assert "src/extra.py" in str(exc.value)
    assert (repo / "src" / "extra.py").read_bytes() == b"e = 1\n", "not rolled back"
    assert (repo / "src" / "mod.py").read_bytes() == b"old = 1\n", "not rolled back"


def test_a_DECLARED_but_unwritten_target_is_refused(repo, tmp_path):
    """The other direction: a plan promising a write that never happened."""
    plan = InstallPlan(unit="D", targets=(
        PlannedTarget("src/mod.py", TargetAction.PATCH, b"a\n"),
        PlannedTarget("src/added.py", TargetAction.CREATE, b"b\n")))
    with pytest.raises(WriteSetViolation) as exc:
        with RepositoryTransaction(repo, tmp_path / "j") as tx:
            tx.patch("src/mod.py", b"a\n")
            plan.assert_write_set(tx)
            tx.commit()
    assert "src/added.py" in str(exc.value)


def test_a_target_CAPTURED_but_never_written_is_refused(repo, tmp_path):
    """The mutated-set check, which no other test exercises.

    MEASURED 2026-08-20: dropping `if mutated != declared` left all 121 tests
    passing. In every other test the declared set and the mutated set are
    identical by the time assert_write_set runs, so the check was load-bearing
    in production and untested.

    This reaches _write_ahead directly, which records a target BEFORE the
    repository is touched -- the write-ahead ordering step 3B introduced. The
    path is declared, the plan expects it, and the bytes never landed. Only the
    mutated-set comparison can see that.
    """
    from genomic_variant_classifier.transactions.repository_transaction import (
        TransactionState)
    plan = InstallPlan(unit="D", targets=(
        PlannedTarget("src/mod.py", TargetAction.PATCH, b"a\n"),))
    tx = RepositoryTransaction(repo, tmp_path / "j")
    tx._transition(TransactionState.APPLYING)
    tx._write_ahead("src/mod.py")

    assert tx.write_set == plan.declared_write_set, "the path IS declared"
    assert tx.mutated_set == frozenset(), "and nothing was written"
    with pytest.raises(WriteSetViolation) as exc:
        plan.assert_write_set(tx)
    assert "never mutated" in str(exc.value), str(exc.value)
    assert "src/mod.py" in str(exc.value)
    tx.rollback()


def test_the_exact_declared_set_is_accepted(repo, tmp_path):
    rat = render_ratchet((repo / "tests" / "EXPECTED_SUITE_SIZE").read_bytes(),
                         "# entry\n\n", 103)
    rdm = render_readme((repo / "README.md").read_bytes(), 100, 103)
    plan = _plan(rat, rdm)
    plan.validate_against(repo)
    with RepositoryTransaction(repo, tmp_path / "j") as tx:
        for t in plan.targets:
            (tx.patch if t.action is TargetAction.PATCH else tx.create)(
                t.relpath, t.payload)
        plan.assert_write_set(tx)
        tx.commit()
    assert (repo / "src" / "added.py").exists()
    bare = [l for l in (repo / "tests" / "EXPECTED_SUITE_SIZE")
            .read_text(encoding="utf-8").split("\n") if l.strip().isdigit()]
    assert bare == ["103"], bare


def test_the_ratchet_and_README_are_ordinary_TRANSACTION_TARGETS(repo, tmp_path):
    """THE ARCHITECTURAL POINT.

    Split-brain semantics -- a transactional payload plus a nontransactional
    ratchet plus a nontransactional README -- admits partially committed
    states:

        payload changed      ratchet old       README old
        payload changed      ratchet new       README old
        payload rolled back  ratchet new       README new

    A failure after either plain write would have no authoritative rollback.
    Here a forced failure restores all four.
    """
    before = {p: (repo / p).read_bytes()
              for p in ("src/mod.py", "tests/EXPECTED_SUITE_SIZE", "README.md")}
    rat = render_ratchet(before["tests/EXPECTED_SUITE_SIZE"], "# entry\n\n", 103)
    rdm = render_readme(before["README.md"], 100, 103)
    plan = _plan(rat, rdm)
    with pytest.raises(RuntimeError):
        with RepositoryTransaction(repo, tmp_path / "j") as tx:
            for t in plan.targets:
                (tx.patch if t.action is TargetAction.PATCH else tx.create)(
                    t.relpath, t.payload)
            raise RuntimeError("the gate failed")
    for rel, original in before.items():
        assert (repo / rel).read_bytes() == original, rel
    assert not (repo / "src" / "added.py").exists()


# ---- the counters come from ONE number ---------------------------------
def test_both_counters_are_rendered_from_a_single_count(repo):
    """MEASURED 2026-08-20: hand-carried counts produced repeated stale-count
    incidents -- 52 against an actual 54, and 38 against an actual 44."""
    count = DerivedCount(before=100, after=103)
    assert count.delta == 3
    rat = render_ratchet((repo / "tests" / "EXPECTED_SUITE_SIZE").read_bytes(),
                         "# entry\n\n", count.after)
    rdm = render_readme((repo / "README.md").read_bytes(), count.before, count.after)
    bare = [l for l in rat.decode("utf-8").split("\n") if l.strip().isdigit()]
    assert bare == [str(count.after)], bare
    assert "badge/tests-{}-".format(count.after) in rdm.decode("utf-8")
    assert "badge/tests-{}-".format(count.before) not in rdm.decode("utf-8")


def test_the_ratchet_keeps_exactly_one_bare_number(repo):
    rat = render_ratchet((repo / "tests" / "EXPECTED_SUITE_SIZE").read_bytes(),
                         "# entry\n\n", 103)
    bare = [l for l in rat.decode("utf-8").split("\n") if l.strip().isdigit()]
    assert len(bare) == 1, bare
    assert "# older" in rat.decode("utf-8"), "the append-only log was lost"


def test_rendering_a_ratchet_with_two_numbers_is_refused():
    with pytest.raises(PlanError):
        render_ratchet(b"# log\n100\n101\n", "# entry\n", 103)


def test_a_README_whose_badge_has_drifted_is_refused(repo):
    """Refuse rather than silently produce an unchanged file."""
    with pytest.raises(PlanError) as exc:
        render_readme((repo / "README.md").read_bytes(), 999, 1000)
    assert "refusing to guess" in str(exc.value)


# ---- validation happens before any write -------------------------------
@pytest.mark.parametrize("relpath,action,payload,fragment", [
    ("src/absent.py", TargetAction.PATCH, b"x", "does not exist"),
    ("src/mod.py", TargetAction.CREATE, b"x", "already exists"),
    ("../outside.py", TargetAction.PATCH, b"x", "outside the repository"),
    ("src/mod.py", TargetAction.PATCH, b"", "empty payload"),
])
def test_validation_refuses_before_the_repository_is_touched(
        repo, relpath, action, payload, fragment):
    plan = InstallPlan(unit="X", targets=(
        PlannedTarget(relpath, action, payload),))
    with pytest.raises(PlanError) as exc:
        plan.validate_against(repo)
    assert fragment in str(exc.value)
    assert (repo / "src" / "mod.py").read_bytes() == b"old = 1\n"


def test_a_duplicate_path_is_refused_at_construction():
    with pytest.raises(PlanError) as exc:
        InstallPlan(unit="X", targets=(
            PlannedTarget("src/mod.py", TargetAction.PATCH, b"a"),
            PlannedTarget("src/mod.py", TargetAction.CREATE, b"b")))
    assert "declared twice" in str(exc.value)


def test_a_non_target_in_the_plan_is_refused():
    with pytest.raises(PlanError):
        InstallPlan(unit="X", targets=("src/mod.py",))


# ---- the plan digest binds dry-run to apply ----------------------------
def test_the_plan_digest_changes_with_any_target_change():
    base = InstallPlan(unit="X", targets=(
        PlannedTarget("a.py", TargetAction.PATCH, b"one"),))
    same = InstallPlan(unit="X", targets=(
        PlannedTarget("a.py", TargetAction.PATCH, b"one"),))
    assert base.digest == same.digest, "identical plans must agree"
    for changed in (
        InstallPlan(unit="X", targets=(
            PlannedTarget("b.py", TargetAction.PATCH, b"one"),)),
        InstallPlan(unit="X", targets=(
            PlannedTarget("a.py", TargetAction.CREATE, b"one"),)),
        InstallPlan(unit="X", targets=(
            PlannedTarget("a.py", TargetAction.PATCH, b"two"),)),
        InstallPlan(unit="Y", targets=(
            PlannedTarget("a.py", TargetAction.PATCH, b"one"),)),
    ):
        assert changed.digest != base.digest, changed.describe()


def test_target_order_does_not_change_the_digest():
    a = InstallPlan(unit="X", targets=(
        PlannedTarget("a.py", TargetAction.PATCH, b"1"),
        PlannedTarget("b.py", TargetAction.PATCH, b"2")))
    b = InstallPlan(unit="X", targets=(
        PlannedTarget("b.py", TargetAction.PATCH, b"2"),
        PlannedTarget("a.py", TargetAction.PATCH, b"1")))
    assert a.digest == b.digest


# ---- the accessors the contract rests on -------------------------------
def test_write_set_is_immutable_and_reports_declared_paths(repo, tmp_path):
    tx = RepositoryTransaction(repo, tmp_path / "j")
    assert tx.write_set == frozenset()
    tx.patch("src/mod.py", b"a\n")
    assert tx.write_set == frozenset({"src/mod.py"})
    assert isinstance(tx.write_set, frozenset)
    with pytest.raises(AttributeError):
        tx.write_set.add("nope")
    tx.rollback()


def test_mutated_set_excludes_a_captured_but_unwritten_target(repo, tmp_path):
    """write_set records a target BEFORE the repository is touched -- that is
    the write-ahead ordering step 3B introduced. mutated_set is the narrower
    question a postcondition needs."""
    from genomic_variant_classifier.transactions.repository_transaction import (
        TransactionState)
    tx = RepositoryTransaction(repo, tmp_path / "j")
    tx._transition(TransactionState.APPLYING)
    tx._write_ahead("src/mod.py")
    assert tx.write_set == frozenset({"src/mod.py"})
    assert tx.mutated_set == frozenset(), "nothing has been written yet"
    tx.rollback()
