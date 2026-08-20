"""One authority for repository hygiene, and the drift it removes.

INSTALLER-TRANSACTION-1 step 5.

    A filter that reports zero is not evidence of zero. It is evidence about
    the filter.

WHY THIS MODULE EXISTS
MEASURED 2026-08-20, before it was written:

    SECRET_PATTERNS  defined TWICE   repository_transaction.py:94  (11 entries)
                                     retire_backup_artifacts.py:109 (11 entries)
    SECRET_CANARIES  defined TWICE   repository_transaction.py:101 (7)
                                     retire_backup_artifacts.py:133 (7)

Both pairs were verified IDENTICAL at runtime -- element for element, order
included. That is the best case for consolidation and exactly what makes the
duplication dangerous: nothing enforced the agreement, and a future edit to
either copy would have drifted in silence. I wrote the second copy myself, by
transcribing the first.

A third consumer -- the no-detritus test -- was about to be added.

THREE DISTINCT QUESTIONS, WHICH WERE BEING CONFLATED
    NOT_THIS_REPOSITORY   contents that are not this project's artefacts
    SCRATCH_ROOTS         declared working space where backups are permitted
    BACKUP_SHAPES         the filename shapes indicating rollback residue

`.gitignore` answers "should git normally show this path?", NOT "may this path
legitimately contain rollback detritus?". Deriving hygiene from ignore rules
would let anyone adding `some_directory/` silently confer scratch legitimacy.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.repository_hygiene import backup_artifacts as H

_REPO = Path(__file__).resolve().parents[2]


# ---- the single-authority property, which is the point -----------------
def test_the_transaction_primitive_imports_the_shared_secret_patterns():
    """THE DRIFT SURFACE, closed.

    Two hand-transcribed copies of an eleven-entry list agreed by luck and
    discipline. Identity of object is the only thing that keeps them agreeing.
    """
    from genomic_variant_classifier.transactions import repository_transaction as txn
    assert txn.SECRET_PATTERNS is H.SECRET_PATTERNS, (
        "the transaction primitive holds its own copy of SECRET_PATTERNS")
    assert txn.SECRET_CANARIES is H.SECRET_CANARIES


def test_no_module_defines_a_SECOND_backup_or_secret_pattern_list():
    """A census, not a spot check.

    Sabotage-proof in the way that matters: adding a fourth list anywhere under
    src/ or scripts/ fails here rather than drifting quietly.
    """
    import ast
    OWNED = {"BACKUP_SHAPES", "BACKUP_PATTERNS", "SECRET_PATTERNS",
             "SECRET_CANARIES", "SCRATCH_ROOTS"}
    here = (_REPO / "src" / "genomic_variant_classifier" / "repository_hygiene"
            / "backup_artifacts.py").resolve()
    offenders = []
    for root in ("src", "scripts"):
        for f in sorted((_REPO / root).rglob("*.py")):
            if f.resolve() == here:
                continue
            try:
                tree = ast.parse(io.open(f, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for n in tree.body:
                name = None
                if isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Name):
                    name = n.targets[0].id
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    name = n.target.id
                if name in OWNED and isinstance(n.value, (ast.Tuple, ast.List, ast.Set)):
                    offenders.append("{}:{} {}".format(
                        f.relative_to(_REPO).as_posix(), n.lineno, name))
    assert not offenders, (
        "these modules define their own copy instead of importing the "
        "authority: {}".format(offenders))


# ---- the three questions are distinct ----------------------------------
@pytest.mark.parametrize("relpath,backup,scratch,excluded", [
    (".af_fix_work/x.py.bak", True, True, False),
    (".venv312/lib/y.py.bak", True, False, True),
    ("scripts/z.py.bak", True, False, False),
    ("src/a.py", False, False, False),
])
def test_the_three_questions_are_answered_separately(relpath, backup, scratch, excluded):
    assert H.is_backup_shaped(relpath) is backup
    assert H.in_scratch_root(relpath) is scratch
    assert H.in_excluded_root(relpath) is excluded


def test_scratch_legitimacy_does_NOT_come_from_gitignore():
    """The authority question, asserted directly.

    A directory ignored by git is not thereby scratch space. If it were, any
    unrelated ignore entry would silently widen the hygiene exception -- the
    semantic coupling this project keeps removing.
    """
    assert "docs" not in H.SCRATCH_ROOTS
    assert "models" not in H.SCRATCH_ROOTS
    assert H.SCRATCH_ROOTS == (".af_fix_work",), H.SCRATCH_ROOTS
    # models/ and docs/ are git-ignored in this repository; neither is scratch.
    assert not H.in_scratch_root("models/x.py.bak")
    assert not H.in_scratch_root("docs/y.md.bak")


def test_the_declared_scratch_roots_are_also_git_ignored():
    """CORRESPONDENCE, not derivation.

    Hygiene does not read `.gitignore`, but a scratch root git insists on
    showing would be a contradiction worth knowing about.
    """
    for root in H.SCRATCH_ROOTS:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "check-ignore", "-v", "--no-index", "--", root],
            capture_output=True, text=True, timeout=120)
        assert out.stdout.strip(), (
            "{} is declared scratch but git does not ignore it".format(root))


# ---- backup shapes -----------------------------------------------------
@pytest.mark.parametrize("name", [
    "x.py.bak_2026-08-19_164056", "x.py.pre_cfgroot.bak", "x.py.precosmic.bak",
    "x.py.20260702_183508.bak", "x.py.bak", "x.py.orig", "x.py.rej",
])
def test_every_observed_backup_shape_is_recognised(name):
    """MEASURED 2026-08-19: these are the shapes actually present. The
    retirement tool scanned only the first and reported zero remaining while
    107 files matched the others."""
    assert H.is_backup_shaped(name), name


@pytest.mark.parametrize("name", ["x.py", "x.backup", "x.py.bakery", "README.md"])
def test_ordinary_names_are_not_backup_shaped(name):
    assert not H.is_backup_shaped(name)


@pytest.mark.parametrize("relpath,expected", [
    (".env.pre_token.bak", ".env"),
    (".env.bak_2026-08-15_205854", ".env"),
    ("src/config.py.pre_cfgroot.bak", "config.py"),
    ("a/b.py.20260702_183508.bak", "b.py"),
    ("server.pem.pre_rotate.bak", "server.pem"),
    ("x/y.py.orig", "y.py"),
    ("x/y.py.rej", "y.py"),
])
def test_the_original_name_is_recovered_from_the_suffix(relpath, expected):
    assert H.strip_backup_suffix(relpath) == expected


def test_a_non_backup_name_strips_to_nothing():
    assert H.strip_backup_suffix("src/mod.py") is None


# ---- secrets -----------------------------------------------------------
@pytest.mark.parametrize("name", list(H.SECRET_CANARIES))
def test_every_canary_is_recognised_as_secret(name):
    assert H.is_secret_path(name), name


def test_a_weakened_secret_classifier_is_REFUSED(monkeypatch):
    """Emptying SECRET_PATTERNS raises no error and leaks nothing in any single
    run -- it simply reclassifies a credential as ordinary. An
    absence-of-failure check cannot see that; this can."""
    monkeypatch.setattr(H, "SECRET_PATTERNS", ())
    with pytest.raises(H.HygieneError) as exc:
        H.assert_secret_detection_intact()
    assert "does not recognise" in str(exc.value)
    monkeypatch.undo()
    H.assert_secret_detection_intact()


def test_a_secret_shape_survives_its_original_being_deleted():
    """MEASURED 2026-08-19: an ordering that consulted the filesystem first let
    a `.env.pre_token.bak` whose live `.env` was gone fall through to
    unclassified, losing the shape metadata that is the point of the secret
    branch."""
    base = H.strip_backup_suffix(".env.pre_token.bak")
    assert base == ".env"
    assert H.is_secret_path(base)


# ---- relocation --------------------------------------------------------
def _mkrepo(base: Path) -> Path:
    repo = base / "r"
    (repo / "scripts" / "forensics").mkdir(parents=True)
    (repo / "scripts" / "forensics" / "tool.py").write_text("v = 2\n", encoding="utf-8")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=str(repo), capture_output=True, timeout=120)
    return repo


def test_relocation_is_proven_by_exact_blob_identity(tmp_path):
    repo = _mkrepo(tmp_path)
    (repo / "scripts" / "exact.py.bak").write_text("v = 2\n", encoding="utf-8")
    r = H.resolve_relocation(repo, "scripts/exact.py.bak")
    assert r is not None
    assert r.original_now_at == "scripts/forensics/tool.py"
    assert "blob identity" in r.evidence


def test_a_divergent_predecessor_resolves_by_basename_with_its_successor(tmp_path):
    """THE CASE THAT MOTIVATED THIS FUNCTION.

    `scripts/verify_written_cohorts.py.bak` was classified UNCLASSIFIED because
    no file existed at its derived path. Manual investigation then found
    `scripts/forensics/verify_written_cohorts.py` -- tracked, arriving at
    0b93d30, 171 lines against the backup's 171, differing in exactly two prose
    passages where the canonical file records that a defect was fixed.

    Basename search plus git history says that automatically.
    """
    repo = _mkrepo(tmp_path)
    (repo / "scripts" / "tool.py.bak").write_text("v = 1\n", encoding="utf-8")
    r = H.resolve_relocation(repo, "scripts/tool.py.bak")
    assert r is not None
    assert r.original_now_at == "scripts/forensics/tool.py"
    assert "basename" in r.evidence
    assert r.successor_commit and len(r.successor_commit) == 40


def test_relocation_returns_NOTHING_rather_than_guessing(tmp_path):
    """A classifier that guesses is worse than one that refuses."""
    repo = _mkrepo(tmp_path)
    (repo / "scripts" / "ghost.py.bak").write_text("g = 0\n", encoding="utf-8")
    assert H.resolve_relocation(repo, "scripts/ghost.py.bak") is None


def test_an_ambiguous_basename_does_not_resolve(tmp_path):
    """Two tracked candidates is not proof of anything."""
    repo = _mkrepo(tmp_path)
    (repo / "scripts" / "other").mkdir()
    (repo / "scripts" / "other" / "tool.py").write_text("v = 3\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "v2"], cwd=str(repo), capture_output=True)
    (repo / "scripts" / "tool.py.bak").write_text("v = 1\n", encoding="utf-8")
    assert H.resolve_relocation(repo, "scripts/tool.py.bak") is None


# ---- the retirement tool honours the declaration -----------------------
def _mk_scratch_repo(base: Path) -> Path:
    """A repository with a scratch backup whose original IS resolvable.

    That combination is the one the earlier sweeps never produced: the twelve
    real .af_fix_work files survived only because their originals happened to
    be untracked.
    """
    repo = base / "r"
    (repo / ".af_fix_work").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / ".gitignore").write_text("*.bak*\n.af_fix_work/\n", encoding="utf-8")
    (repo / ".af_fix_work" / "tool.py").write_text("v = 1\n", encoding="utf-8")
    (repo / "scripts" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"], ["git", "add", "-A", "-f"],
                ["git", "commit", "-qm", "v1"]):
        subprocess.run(cmd, cwd=str(repo), capture_output=True, timeout=120)
    (repo / ".af_fix_work" / "tool.py.bak").write_text("v = 1\n", encoding="utf-8")
    (repo / "scripts" / "mod.py.bak").write_text("x = 1\n", encoding="utf-8")
    (repo / ".af_fix_work" / "tool.py").write_text("v = 2\n", encoding="utf-8")
    (repo / "scripts" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    for cmd in (["git", "add", "-A", "-f"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-qm", "v2"]):
        subprocess.run(cmd, cwd=str(repo), capture_output=True, timeout=120)
    return repo


def test_the_retirement_tool_RETAINS_a_resolvable_scratch_backup(tmp_path):
    """THE DEFECT THIS BRANCH EXISTS FOR.

    MEASURED 2026-08-20: without a scratch branch the retirement tool deleted a
    backup inside .af_fix_work whose original was resolvable -- classifying it
    git_exact_preimage and removing it. The twelve real scratch files had
    survived every earlier sweep only because their originals happened to be
    untracked.

    An outcome one approves of, produced by a mechanism one has not checked, is
    not evidence the mechanism is right.
    """
    import importlib.util
    repo = _mk_scratch_repo(tmp_path)
    manifest = repo / "docs" / "incidents" / "M.json"
    spec = importlib.util.spec_from_file_location(
        "retire_probe", str(_REPO / "scripts" / "retire_backup_artifacts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--repo-root", str(repo), "--manifest",
                   "docs/incidents/M.json", "--apply"])
    assert rc == 0, rc
    assert (repo / ".af_fix_work" / "tool.py.bak").exists(), (
        "a backup inside a DECLARED scratch root was deleted")
    assert not (repo / "scripts" / "mod.py.bak").exists(), (
        "an ordinary redundant backup was NOT retired")


def test_the_manifest_records_scratch_as_its_own_classification(tmp_path):
    """Retention must be recorded with its reason, not merely happen."""
    import importlib.util
    import json
    repo = _mk_scratch_repo(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "retire_probe2", str(_REPO / "scripts" / "retire_backup_artifacts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--repo-root", str(repo), "--manifest", "docs/incidents/M.json"])
    values = json.loads((repo / "docs" / "incidents" / "M.json")
                        .read_text(encoding="utf-8"))
    entry = [f for f in values["files"] if ".af_fix_work" in f["backup"]]
    assert len(entry) == 1, values["classification"]
    assert entry[0]["classification"] == "scratch", entry[0]
    assert "DECLARED scratch root" in entry[0]["rationale"], entry[0]["rationale"]


def test_the_retirement_tool_imports_the_shared_vocabulary():
    """No fourth list. The census test covers src/ and scripts/ generally;
    this names the specific consumer the consolidation was built for."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retire_probe3", str(_REPO / "scripts" / "retire_backup_artifacts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SECRET_PATTERNS is H.SECRET_PATTERNS
    assert mod.SECRET_CANARIES is H.SECRET_CANARIES
    assert mod.BACKUP_PATTERNS is H.BACKUP_SHAPES
    assert mod.EXCLUDED_ROOTS is H.NOT_THIS_REPOSITORY


# ---- the live repository -----------------------------------------------
def test_iter_detritus_excludes_scratch_and_foreign_roots(tmp_path):
    repo = tmp_path / "r"
    for d in (".af_fix_work", ".venv312/lib", "scripts"):
        (repo / d).mkdir(parents=True)
    (repo / ".af_fix_work" / "a.py.bak").write_text("a\n", encoding="utf-8")
    (repo / ".venv312" / "lib" / "b.py.bak").write_text("b\n", encoding="utf-8")
    (repo / "scripts" / "c.py.bak").write_text("c\n", encoding="utf-8")
    found = sorted(H.iter_repository_detritus(repo))
    assert found == ["scripts/c.py.bak"], found
    with_scratch = sorted(H.iter_repository_detritus(repo, include_scratch=True))
    assert with_scratch == [".af_fix_work/a.py.bak", "scripts/c.py.bak"], with_scratch
