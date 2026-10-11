"""Branch equivalence: the derivation on constructed histories (every outcome), the strict loader, and the COMMITTED run9a-prep record.

MEASURED 2026-10-10: the GitHub branch run9a-prep counted "180 commits not in main", yet held no content main lacks -- the July 2026
history rewrite renamed every commit on main and the identifiers quoted in messages, and removed notebooks./gitkeep; the branch kept the
old names. The committed record states that commit by commit; it was derived identically from the GitHub branch and from the owner's
preserved bundle (sha256 1e83929f...), and the installer re-derives it from that bundle before this change can be committed.

The constructed histories here are built with git plumbing (fixed identities and dates), so every outcome is produced by real git
objects, not by fixtures of the record format. Their trees are written with `git mktree`, not through an index: an index on Windows
refuses the excised path `notebooks./gitkeep` (measured 2026-10-11; see History).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from genomic_variant_classifier.repository_records import branch_equivalence as be
from genomic_variant_classifier.repository_records.identity import ArtifactInstance, RecordId, RecordIdentity
from genomic_variant_classifier.repository_records.path_budget import budget_violations
from genomic_variant_classifier.repository_records.roles import RecordsOntologyError

ROOT = Path(__file__).resolve().parents[2]
RECORD_ID = "REC-9ccf9fa0863b447292230fe1130850b0"
RECORD = ROOT.joinpath(*be.family_root().parts, RECORD_ID + ".json")
EXCISED = ["data/alphafold_cif_cache_2026-07-03.tar.gz", "notebooks./gitkeep"]
COPY = {"store": "gvc-artifacts", "path": "history/x.bundle", "sha256": "a" * 64, "size_bytes": 1}   # + "heads", which names the tip


# ------------------------------------------------------------------------------------------------------------------ constructed histories

def _git(repo: Path, *args: str, env=None) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, (args, out.stderr)
    return out.stdout.strip()


def _git_input(repo: Path, args: list, data: bytes) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, timeout=120)
    assert out.returncode == 0, (args, out.stderr.decode(errors="replace"))
    return out.stdout.decode().strip()


class History:
    """A bare-bones repository with commits made from {path: text} snapshots, with fixed identities and dates.

    Trees are written with `git mktree`, one directory level at a time, never through an index. MEASURED 2026-10-11 on the owner's
    machine (Git 2.55.0.windows.5): an index refuses `notebooks./gitkeep` ("error: Invalid path"), because Git for Windows checks every
    index path against Windows naming (verify_path -> is_valid_win32_path: no component may end in a period or a space, and no
    reserved device names) while core.protectNTFS is on, its default. That check protects a CHECKOUT. The history reproduced here --
    like the real run9a-prep history -- holds that path as a tree entry, which `git mktree` writes on every platform (it refuses only
    a name containing a slash). The module under test reads objects only, as these tests do."""

    def __init__(self, path: Path):
        self.repo = path
        _git(path.parent, "init", "-q", "--bare", path.name)

    def _blob(self, text: str) -> str:
        return _git_input(self.repo, ["hash-object", "-w", "--stdin"], text.encode())

    def tree(self, files: dict) -> str:
        """{relative posix path: text} -> tree id. Each directory level is one `git mktree -z` call (mktree orders the entries)."""
        here: dict = {}
        below: dict = {}
        for path, text in files.items():
            head, sep, rest = path.partition("/")
            if sep:
                below.setdefault(head, {})[rest] = text
            else:
                here[head] = text
        assert not set(here) & set(below), "a name cannot be both a file and a directory"
        entries = ["100644 blob {}\t{}".format(self._blob(text), name) for name, text in here.items()]
        entries += ["040000 tree {}\t{}".format(self.tree(sub), name) for name, sub in below.items()]
        return _git_input(self.repo, ["mktree", "-z"], b"".join(e.encode() + b"\0" for e in entries))

    def commit(self, files: dict, parents: list, message: str, when: int) -> str:
        env = dict(os.environ, GIT_AUTHOR_NAME="A", GIT_AUTHOR_EMAIL="a@example.invalid",
                   GIT_COMMITTER_NAME="C", GIT_COMMITTER_EMAIL="c@example.invalid",
                   GIT_AUTHOR_DATE="{} +0000".format(1700000000 + when), GIT_COMMITTER_DATE="{} +0000".format(1700000000 + when))
        args = ["commit-tree", self.tree(files), "-m", message]
        for p in parents:
            args += ["-p", p]
        return _git(self.repo, *args, env=env)

    def ref(self, name: str, oid: str) -> None:
        _git(self.repo, "update-ref", name, oid)


def build(tmp_path, *, extra_blob=False, message_change=False, parents_mismatch=False, ambiguous=False, impure_excision=False,
          claimed_twice=False, excised_content="\n"):
    """old = the pre-rewrite branch; main = the rewritten reference. As in the July 2026 rewrite: the commit that only added the
    excised path is dropped, every later commit gets a new identifier, and identifiers quoted in later messages are renamed."""
    h = History(tmp_path / "h.git")
    base = {"README": "r\n"}
    root = h.commit(base, [], "root", 0)
    s1 = dict(base, **{"a.txt": "a\n"})
    s2 = dict(s1, **{"notebooks./gitkeep": excised_content})
    if impure_excision:
        s2["z.txt"] = "only on the old branch\n"
    s3 = dict(s1, **{"notebooks/gitkeep": "\n"})
    s4 = dict(s3, **{"a.txt": "a fixed\n"})
    s5 = dict(s4, **{"b.txt": "b\n"})
    a1 = h.commit(s1, [root], "add a", 1)                 # before the excised commit: unchanged, therefore SHARED
    a2 = h.commit(s2, [a1], "Create gitkeep", 2)
    a3 = h.commit(s3, [a2], "move gitkeep", 3)
    a4 = h.commit(s4, [a3], "fix a", 4)
    a5 = h.commit(dict(s5, **{"c.txt": "never on main\n"}) if extra_blob else s5, [a4], "plain (after {})".format(a3[:7]), 5)
    tip = h.commit(s5, [a5], "plain (after {})".format(a3[:7]), 5) if claimed_twice else a5
    n3 = h.commit(s3, [root if parents_mismatch else a1], "move gitkeep", 3)
    n4 = h.commit(s4, [n3], "fix a", 4)
    n5 = h.commit(s5, [n4], "plain! (after {})".format(n3[:7]) if message_change else "plain (after {})".format(n3[:7]), 5)
    main = n5
    if ambiguous:
        twin = h.commit(s4, [root], "fix a, again", 4)    # same tree and identity as n4
        main = h.commit(s5, [n5, twin], "merge", 6)
    h.ref("refs/heads/old", tip)
    h.ref("refs/heads/main", main)
    return h.repo, {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "n3": n3, "n4": n4, "n5": n5, "tip": tip}


def copy_of(repo) -> dict:
    """The synthetic preserved copy; like the real bundle, it names the subject tip among its heads (load() requires that)."""
    return dict(COPY, heads=[_git(repo, "rev-parse", "refs/heads/old") + " refs/heads/old"])


def derive(repo, excised=("notebooks./gitkeep",)):
    return be.derive(repo, record_id=RECORD_ID, subject_name="old", subject_ref="refs/heads/old", reference_name="main",
                     reference_ref="refs/heads/main", excised_paths=list(excised), preserved_copy=copy_of(repo))


def rows_of(rec):
    return {r["commit"]: r for r in rec["branch_only_commits"]}


def test_the_helper_writes_the_same_tree_an_index_writes_for_names_every_platform_accepts(tmp_path):
    # Tree objects are canonical, so building them without an index changes nothing the tests compare. The names exercise git's
    # entry order, in which a directory sorts as if its name ended in "/" ("a-b" < "a.b" < "a/" < "a0").
    h = History(tmp_path / "h.git")
    files = {"README": "r\n", "a.b": "1\n", "a/c": "2\n", "a/d/e.txt": "3\n", "a-b": "4\n", "a0": "5\n", "notebooks/gitkeep": "\n"}
    env = dict(os.environ, GIT_INDEX_FILE=str(tmp_path / "index"))
    for path, text in sorted(files.items()):
        blob = _git_input(h.repo, ["hash-object", "-w", "--stdin"], text.encode())
        _git(h.repo, "update-index", "--add", "--cacheinfo", "100644,{},{}".format(blob, path), env=env)
    assert h.tree(files) == _git(h.repo, "write-tree", env=env)


def test_the_constructed_history_holds_the_excised_path_as_a_tree_entry(tmp_path):
    repo, c = build(tmp_path)
    assert _git(repo, "ls-tree", "-r", "--name-only", c["a2"]).splitlines() == ["README", "a.txt", "notebooks./gitkeep"]
    # the excised subtree IS main's notebooks/ subtree (same single blob), as test_an_equivalent_branch relies on
    assert _git(repo, "rev-parse", c["a2"] + ":notebooks.") == _git(repo, "rev-parse", c["n3"] + ":notebooks")


def test_the_history_helper_never_builds_a_tree_through_an_index():
    # Every string the helper's CODE uses (docstrings excluded): none names an index operation, so no constructed history can depend
    # on the platform's index path rules again.
    import ast
    import inspect
    cls = ast.parse(inspect.getsource(History)).body[0]
    docstrings = {id(n.body[0].value) for n in ast.walk(cls) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    strings = [n.value for n in ast.walk(cls) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    assert "mktree" in strings and "commit-tree" in strings
    assert [s for s in strings if any(w in s for w in ("update-index", "read-tree", "write-tree", "GIT_INDEX_FILE"))] == []


def test_commits_before_the_excised_one_are_shared_so_only_the_rewritten_tail_is_branch_only(tmp_path):
    repo, c = build(tmp_path)
    assert [r["commit"] for r in derive(repo)["branch_only_commits"]] == [c["a2"], c["a3"], c["a4"], c["a5"]]


def test_an_equivalent_branch(tmp_path):
    repo, c = build(tmp_path)
    rec = derive(repo)
    rows = rows_of(rec)
    assert (rows[c["a2"]]["outcome"], rows[c["a2"]]["excised_paths"], rows[c["a2"]]["counterpart"]) == ("excised_paths_only", ["notebooks./gitkeep"], None)
    assert (rows[c["a3"]]["outcome"], rows[c["a3"]]["counterpart"]) == ("identical", c["n3"])
    assert (rows[c["a4"]]["outcome"], rows[c["a4"]]["counterpart"]) == ("identical", c["n4"])
    assert (rows[c["a5"]]["outcome"], rows[c["a5"]]["counterpart"]) == ("identifier_rewrite", c["n5"])
    assert rows[c["a5"]]["identifier_rewrites"] == [[c["a3"][:7], c["n3"][:7]]]
    assert rec["verdict"] == "equivalent" and rec["objects_not_in_reference"]["blobs"] == []
    # Exactly one tree is new: a2's root tree. Its notebooks./ subtree holds the same single blob as main's notebooks/ subtree, so
    # git stores ONE tree object for both -- an excised path whose content main still has adds no subtree of its own.
    assert rec["objects_not_in_reference"]["commits"] == 4
    assert rec["objects_not_in_reference"]["trees"] == [_git(repo, "rev-parse", c["a2"] + "^{tree}")]
    assert rec["summary"] == {"branch_only_commits": 4, "identical": 2, "identifier_rewrite": 1, "excised_paths_only": 1,
                              "differs": 0, "ambiguous": 0, "no_counterpart": 0}


def test_a_blob_the_reference_lacks_is_not_equivalent(tmp_path):
    repo, c = build(tmp_path, extra_blob=True)
    rec = derive(repo)
    rows = rows_of(rec)
    assert rows[c["a5"]]["outcome"] == "no_counterpart"
    assert len(rec["objects_not_in_reference"]["blobs"]) == 1 and rec["verdict"] == "not_equivalent"


def test_a_message_that_differs_beyond_identifiers(tmp_path):
    repo, c = build(tmp_path, message_change=True)
    rec = derive(repo)
    rows = rows_of(rec)
    assert rows[c["a5"]]["outcome"] == "differs" and rows[c["a5"]]["counterpart"] == c["n5"] and rec["verdict"] == "not_equivalent"


def test_parents_that_differ(tmp_path):
    repo, c = build(tmp_path, parents_mismatch=True)
    rec = derive(repo)
    rows = rows_of(rec)
    assert rows[c["a3"]]["outcome"] == "differs" and rec["verdict"] == "not_equivalent"


def test_two_reference_matches_are_ambiguous(tmp_path):
    repo, c = build(tmp_path, ambiguous=True)
    rec = derive(repo)
    rows = rows_of(rec)
    assert rows[c["a4"]]["outcome"] == "ambiguous" and rows[c["a4"]]["counterpart"] is None and rec["verdict"] == "not_equivalent"


def test_a_reference_commit_claimed_twice_matches_neither(tmp_path):
    repo, c = build(tmp_path, claimed_twice=True)
    rows = rows_of(derive(repo))
    assert rows[c["a5"]]["outcome"] == rows[c["tip"]]["outcome"] == "ambiguous"
    assert rows[c["a5"]]["counterpart"] is rows[c["tip"]]["counterpart"] is None


def test_an_undeclared_excision_is_no_counterpart(tmp_path):
    repo, c = build(tmp_path)
    rec = derive(repo, excised=())
    rows = rows_of(rec)
    assert rows[c["a2"]]["outcome"] == "no_counterpart" and rec["verdict"] == "not_equivalent"
    assert rec["objects_not_in_reference"]["blobs"] == []             # the content IS on main -- the commit is still unexplained


def test_a_declared_excision_the_reference_still_holds_explains_nothing(tmp_path):
    # a.txt is on the reference (the shared a1 holds it), so it was not excised there: a2 changes only declared paths, but its tree
    # without them is not its parent's counterpart's tree, and the commit stays unexplained.
    repo, c = build(tmp_path)
    rec = derive(repo, excised=("a.txt", "notebooks./gitkeep"))
    assert rows_of(rec)[c["a2"]]["outcome"] == "no_counterpart" and rec["verdict"] == "not_equivalent"


def test_an_excision_that_also_touches_another_path_is_not_excised_only(tmp_path):
    repo, c = build(tmp_path, impure_excision=True)
    rec = derive(repo)
    rows = rows_of(rec)
    assert rows[c["a2"]]["outcome"] == "no_counterpart" and len(rec["objects_not_in_reference"]["blobs"]) == 1


def test_excised_content_the_reference_lacks_is_still_content_the_branch_holds(tmp_path):
    # The commit is correctly explained (excised paths only), but its blob exists nowhere on the reference: deleting the branch would
    # lose it. The verdict answers the record's question -- content -- so it is not_equivalent although every row is "explained".
    repo, c = build(tmp_path, excised_content="only in the excised file\n")
    rec = derive(repo)
    assert all(be.Outcome(r["outcome"]) in be.EQUIVALENT_OUTCOMES for r in rec["branch_only_commits"])
    assert rows_of(rec)[c["a2"]]["outcome"] == "excised_paths_only"
    assert rec["objects_not_in_reference"]["blobs"] == [_git(repo, "rev-parse", c["a2"] + ":notebooks./gitkeep")]
    assert rec["verdict"] == "not_equivalent"


def test_derivation_is_deterministic_and_render_load_round_trips(tmp_path):
    repo, _ = build(tmp_path)
    a, b = be.render(derive(repo)), be.render(derive(repo))
    assert a == b and be.load(a) == json.loads(a)


def test_the_command_line_writes_the_same_bytes(tmp_path):
    repo, _ = build(tmp_path)
    out = tmp_path / "rec.json"
    copy = json.dumps(copy_of(repo))
    assert be.main(["derive", "--repo", str(repo), "--record-id", RECORD_ID, "--subject-name", "old", "--subject-ref", "refs/heads/old",
                    "--reference-name", "main", "--reference-ref", "refs/heads/main", "--excise", "notebooks./gitkeep",
                    "--preserved-copy", copy, "--out", str(out)]) == 0
    assert out.read_bytes() == be.render(derive(repo))
    assert be.main(["verify", str(out)]) == 0


def test_a_ref_that_names_no_commit_is_refused(tmp_path):
    repo, _ = build(tmp_path)
    with pytest.raises(be.BranchEquivalenceError):
        be.derive(repo, record_id=RECORD_ID, subject_name="old", subject_ref="refs/heads/absent", reference_name="main",
                  reference_ref="refs/heads/main", excised_paths=[], preserved_copy=copy_of(repo))


# ------------------------------------------------------------------------------------------------------------------ strict loading

def _committed() -> bytes:
    return RECORD.read_bytes()


def _mutated(edit) -> bytes:
    doc = json.loads(_committed())
    edit(doc)
    return be.render(doc)


@pytest.mark.parametrize("name, edit", [
    ("summary", lambda d: d["summary"].__setitem__("identical", 168)),
    ("verdict", lambda d: d.__setitem__("verdict", "not_equivalent")),
    ("outcome", lambda d: d["branch_only_commits"][0].__setitem__("outcome", "same")),
    ("counterpart twice", lambda d: d["branch_only_commits"][1].__setitem__("counterpart", d["branch_only_commits"][0]["counterpart"])),
    ("rewrite without pairs", lambda d: next(r for r in d["branch_only_commits"] if r["outcome"] == "identifier_rewrite").__setitem__("identifier_rewrites", [])),
    ("pair lengths", lambda d: next(r for r in d["branch_only_commits"] if r["outcome"] == "identifier_rewrite")["identifier_rewrites"][0].__setitem__(1, "abc1234ff")),
    ("excised path undeclared", lambda d: next(r for r in d["branch_only_commits"] if r["outcome"] == "excised_paths_only").__setitem__("excised_paths", ["other"])),
    ("a blob missing", lambda d: d["objects_not_in_reference"].__setitem__("blobs", ["0" * 40])),
    ("a malformed blob identifier", lambda d: (d["objects_not_in_reference"].__setitem__("blobs", ["not-an-object"]),
                                               d.__setitem__("verdict", "not_equivalent"))),
    ("a malformed tree identifier", lambda d: d["objects_not_in_reference"].__setitem__("trees", ["0" * 39])),
    ("commit count", lambda d: d["objects_not_in_reference"].__setitem__("commits", 179)),
    ("head is not the tip", lambda d: d["preserved_copy"].__setitem__("heads", ["0" * 40 + " refs/x"])),
    ("absolute copy path", lambda d: d["preserved_copy"].__setitem__("path", "/history/x.bundle")),
    ("copy digest", lambda d: d["preserved_copy"].__setitem__("sha256", "1e83929f")),
    ("schema", lambda d: d.__setitem__("schema_version", 2)),
    ("extra key", lambda d: d.__setitem__("extra", 1)),
])
def test_a_tampered_record_is_refused(name, edit):
    with pytest.raises(be.BranchEquivalenceError):
        be.load(_mutated(edit))


def test_a_malformed_record_id_is_refused_by_the_identity_owner():
    with pytest.raises(RecordsOntologyError, match="not a record identifier"):
        be.load(_mutated(lambda d: d.__setitem__("record_id", "REC-1")))


@pytest.mark.parametrize("raw", [
    b"", b"\xef\xbb\xbf{}\n", b"{}", b'{"schema": 1.5}\n',
])
def test_malformed_bytes_are_refused(raw):
    with pytest.raises(be.BranchEquivalenceError):
        be.load(raw)


def test_a_duplicate_key_and_a_non_canonical_rendering_are_refused():
    raw = _committed()
    with pytest.raises(be.BranchEquivalenceError, match="duplicate key"):
        be.load(raw.replace(b'"schema": "gvc.branch-equivalence",', b'"schema": "gvc.branch-equivalence", "schema": "x",', 1))
    with pytest.raises(be.BranchEquivalenceError, match="authored form"):
        be.load(raw.replace(b'  "schema"', b'   "schema"', 1))
    with pytest.raises(be.BranchEquivalenceError, match="keys"):
        doc = json.loads(raw)
        be.load((json.dumps({k: doc[k] for k in reversed(list(doc))}, indent=2) + "\n").encode())


# ------------------------------------------------------------------------------------------------------------------ the committed record

def test_the_family_holds_exactly_the_run9a_prep_record():
    family = ROOT.joinpath(*be.family_root().parts)
    assert sorted(p.name for p in family.iterdir()) == [RECORD_ID + ".json"]


def test_the_committed_record_states_the_measured_equivalence():
    doc = be.load(_committed())
    assert doc["record_id"] == RECORD_ID and doc["verdict"] == "equivalent"
    assert doc["subject"] == {"name": "run9a-prep", "tip": "c1c01920aa0b6fc21d0b3e926cbe1c66af016acc",
                              "tree": "b6478560bc43217ed3008013b42701e456ce5cd1", "commits": 183,
                              "roots": ["08e881b23bb545c85ebbc5b73b8dbbe78c04b9f6"]}
    assert doc["reference"] == {"name": "main", "commit": "ffc549f4d346feb906348a183f18e31613707d40",
                                "tree": "4dfc615f0f888d6290c71db431a1f49ffb6d6744"}
    assert doc["excised_paths"] == EXCISED
    assert doc["summary"] == {"branch_only_commits": 180, "identical": 169, "identifier_rewrite": 10, "excised_paths_only": 1,
                              "differs": 0, "ambiguous": 0, "no_counterpart": 0}
    assert doc["objects_not_in_reference"] == {"commits": 180, "trees": ["3a113fe20650a5c39c724e3b8777e01a3806cb2a",
                                                                         "793bc8da80220def928ad1f44ca57bd83d95c143"], "blobs": []}
    excised = [r for r in doc["branch_only_commits"] if r["outcome"] == "excised_paths_only"]
    assert [(r["commit"], r["excised_paths"]) for r in excised] == [("9bdd27091aa3b9227adca30d8a984473d526bd1e", ["notebooks./gitkeep"])]
    pairs = {tuple(p) for r in doc["branch_only_commits"] for p in r["identifier_rewrites"]}
    assert len(pairs) == 13 and ("cb9a4cb", "a9f5ff0") in pairs and ("72de60e", "5a326cb") in pairs


def test_the_record_is_bound_to_the_owners_preserved_bundle():
    pc = be.load(_committed())["preserved_copy"]
    assert pc == {"store": "gvc-artifacts", "path": "history/run9a-prep_c1c01920aa0b_20261010T221514Z.bundle",
                  "sha256": "1e83929fee92f430b1bd63745eb4be8d86a6a9663beb540cf017c3c431e1e4f4", "size_bytes": 3796302,
                  "heads": ["c1c01920aa0b6fc21d0b3e926cbe1c66af016acc refs/remotes/origin/run9a-prep"]}


def test_the_record_is_placed_by_role_within_the_path_budget():
    rel = RECORD.relative_to(ROOT).as_posix()
    raw = _committed()
    ident = RecordIdentity(RecordId(RECORD_ID), ArtifactInstance(hashlib.sha256(raw).hexdigest(), rel, len(raw)), be.ROLE)
    assert PurePosixPath(rel).parent == be.family_root() and budget_violations(rel) == [] and ident.basename == RECORD_ID + ".json"
