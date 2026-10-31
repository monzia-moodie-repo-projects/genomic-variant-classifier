"""repository_hygiene/workspace_snapshot.py: what a test session leaves behind in the working tree.

Each test builds a small tree under tmp_path, snapshots it, changes it, snapshots it again and checks what the comparison names.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from genomic_variant_classifier.repository_hygiene import backup_artifacts as H
from genomic_variant_classifier.repository_hygiene import workspace_snapshot as ws


def _tree(root: Path) -> Path:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    return root


def _bump(path: Path) -> None:
    """Move the modification time unambiguously (a coarse filesystem clock must not hide a rewrite of equal size)."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_the_exclusions_are_the_hygiene_authoritys_object_not_a_copy():
    assert ws.NOT_THIS_REPOSITORY is H.NOT_THIS_REPOSITORY


def test_an_unchanged_tree_compares_equal(tmp_path):
    root = _tree(tmp_path)
    before = ws.take_snapshot(root)
    after = ws.take_snapshot(root)
    assert not ws.compare(before, after)
    assert set(before.entries) == {"src", "src/pkg", "src/pkg/module.py", "README.md"}
    assert before.entries["src"] == ("dir",) and before.entries["README.md"][0] == "file"
    assert before.unobserved == () and before.seconds >= 0


def test_added_removed_and_rewritten_files_are_each_named(tmp_path):
    root = _tree(tmp_path)
    before = ws.take_snapshot(root)
    (root / "models").mkdir()
    (root / "models" / "best_gat.pt").write_bytes(b"weights")
    (root / "README.md").unlink()
    (root / "src" / "pkg" / "module.py").write_text("x = 2\n", encoding="utf-8")       # same size: only the time differs
    _bump(root / "src" / "pkg" / "module.py")
    change = ws.compare(before, ws.take_snapshot(root))
    assert change.added == ("models", "models/best_gat.pt")
    assert change.removed == ("README.md",)
    assert change.changed == ("src/pkg/module.py",)
    assert bool(change)


def test_a_file_made_and_removed_again_leaves_no_difference(tmp_path):
    """Only what REMAINS is a finding: a directory's own modification time is not recorded."""
    root = _tree(tmp_path)
    before = ws.take_snapshot(root)
    scratch = root / "src" / "pkg" / "scratch.txt"
    scratch.write_text("temporary", encoding="utf-8")
    scratch.unlink()
    _bump(root / "src" / "pkg")
    assert not ws.compare(before, ws.take_snapshot(root))


def test_an_empty_new_directory_is_a_finding(tmp_path):
    root = _tree(tmp_path)
    before = ws.take_snapshot(root)
    (root / "models").mkdir()
    assert ws.compare(before, ws.take_snapshot(root)).added == ("models",)


@pytest.mark.parametrize("declared", sorted(H.NOT_THIS_REPOSITORY))
def test_declared_directories_are_not_entered_at_any_depth(tmp_path, declared):
    root = _tree(tmp_path)
    (root / declared).mkdir()
    (root / "src" / "pkg" / declared).mkdir()
    before = ws.take_snapshot(root)
    (root / declared / "inside.bin").write_bytes(b"x")
    (root / "src" / "pkg" / declared / "module.cpython-312.pyc").write_bytes(b"x")
    after = ws.take_snapshot(root)
    assert not ws.compare(before, after)
    assert not [p for p in after.entries if declared in p.split("/")]


def test_a_symbolic_link_is_recorded_as_a_link_and_never_followed(tmp_path):
    root = _tree(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, root / "data", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links need a privilege this account lacks (the junction test covers the boundary)")
    before = ws.take_snapshot(root)
    assert before.entries["data"] == ("link", str(outside))
    (outside / "big.parquet").write_bytes(b"x")                       # behind the link: not this tree's content
    assert not ws.compare(before, ws.take_snapshot(root))
    (root / "data").unlink()                                           # the link itself going away IS a finding
    assert ws.compare(before, ws.take_snapshot(root)).removed == ("data",)


@pytest.mark.skipif(os.name != "nt", reason="directory junctions exist only on Windows")
def test_a_windows_junction_is_recorded_as_a_link_and_never_followed(tmp_path):
    import subprocess

    root = _tree(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(root / "data"), str(outside)], capture_output=True, text=True)
    assert made.returncode == 0, made.stdout + made.stderr
    before = ws.take_snapshot(root)
    assert before.entries["data"][0] == "link"
    (outside / "big.parquet").write_bytes(b"x")
    assert not ws.compare(before, ws.take_snapshot(root))


def test_a_directory_that_cannot_be_listed_is_reported_never_read_as_empty(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    locked = str(root / "src" / "pkg")
    real_scandir = os.scandir

    def scandir(path="."):
        if os.fspath(path) == locked:
            raise PermissionError(13, "Permission denied", locked)
        return real_scandir(path)

    monkeypatch.setattr(ws.os, "scandir", scandir)
    snap = ws.take_snapshot(root)
    assert snap.unobserved == ("src/pkg (PermissionError)",)
    assert "src/pkg" in snap.entries and "src/pkg/module.py" not in snap.entries


def test_snapshots_of_different_roots_are_not_compared(tmp_path):
    a = ws.take_snapshot(_tree(tmp_path / "a"))
    b = ws.take_snapshot(_tree(tmp_path / "b"))
    with pytest.raises(ValueError, match="different roots"):
        ws.compare(a, b)


def test_describe_names_every_kind_and_counts_beyond_the_limit():
    change = ws.WorkspaceChange(added=tuple("a{}".format(i) for i in range(5)), removed=("r",), changed=())
    assert ws.describe(change, limit=3) == ["ADDED (5):", "    a0", "    a1", "    a2", "    ... and 2 more", "REMOVED (1):", "    r"]
    assert ws.describe(ws.WorkspaceChange((), (), ())) == []
