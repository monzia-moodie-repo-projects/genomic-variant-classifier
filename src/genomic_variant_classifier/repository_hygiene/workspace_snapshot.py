"""What a test session leaves behind in the working tree.

Author: Monzia Moodie

WHY THIS EXISTS
===============
MEASURED 2026-10-11: two validation clones that had run the full suite each held
`models/best_gat.pt` (written by `GNNTrainer`, whose default checkpoint path is
relative to the current directory) and compiled `__pycache__` files, so the
owner's clean-up block refused to delete them as "untracked or ignored files".
`git status` showed nothing: `/models/` and `*.pt` are git-ignored. The
ROADMAP entry of 2026-09-23 ("the suite no longer writes into the repository")
was therefore not true, and nothing had measured it.

The `data/raw/cache` guard in `tests/conftest.py` watches one directory per
test. This module watches the WHOLE working tree once per session: a snapshot
when the session starts, another when it finishes, and the difference named
path by path.

WHAT IS OBSERVED
================
Every file (size and modification time), every directory (presence) and every
link (its target) below the root. A directory's own modification time is not
recorded, so a test that creates a file and removes it again leaves no
difference -- only what REMAINS is a finding.

Never followed: anything `artifact_inventory.is_redirected` reports -- symbolic
links, junctions and every other Windows reparse point (ruling 2026-10-08f).
They are recorded as links, so a link created or removed is still a finding.

Not entered: the directories `backup_artifacts.NOT_THIS_REPOSITORY` declares
(.git, virtual environments, __pycache__, .pytest_cache, ...). That
declaration is the hygiene authority; this module imports it and restates
nothing.

A directory that cannot be listed is not an absence: it is reported as
unobserved, by name, in every summary.
"""

from __future__ import annotations

import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from genomic_variant_classifier.repository_hygiene.backup_artifacts import NOT_THIS_REPOSITORY
from genomic_variant_classifier.repository_records.artifact_inventory import ArtifactInventoryError, is_redirected

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Every observed entry below `root`, keyed by its relative POSIX path.

    An entry is ("file", size, mtime_ns), ("dir",) or ("link", target).
    """

    root: Path
    entries: Mapping[str, tuple]
    unobserved: tuple[str, ...]
    seconds: float


@dataclass(frozen=True)
class WorkspaceChange:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def _relative(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _link_target(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError as exc:              # a reparse point readlink cannot interpret is still a link; its target is unknown
        return "unreadable target ({})".format(type(exc).__name__)


def take_snapshot(root) -> WorkspaceSnapshot:
    """Walk `root` without following any link; see the module docstring for what is recorded."""
    started = time.perf_counter()
    top = os.fspath(Path(root))
    entries: dict[str, tuple] = {}
    unobserved: list[str] = []
    pending = [top]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as listing:
                children = list(listing)
        except OSError as exc:
            unobserved.append("{} ({})".format(_relative(top, directory) if directory != top else ".", type(exc).__name__))
            continue
        for child in children:
            rel = _relative(top, child.path)
            try:
                observed = child.stat(follow_symlinks=False)
            except OSError as exc:
                unobserved.append("{} ({})".format(rel, type(exc).__name__))
                continue
            if stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                try:
                    redirected = is_redirected(child.path)
                except ArtifactInventoryError as exc:
                    unobserved.append("{} ({})".format(rel, exc))
                    continue
                if redirected:
                    entries[rel] = ("link", _link_target(child.path))
                    continue
                if child.name in NOT_THIS_REPOSITORY:
                    continue
                entries[rel] = ("dir",)
                pending.append(child.path)
            else:
                entries[rel] = ("file", observed.st_size, observed.st_mtime_ns)
    return WorkspaceSnapshot(root=Path(top), entries=entries, unobserved=tuple(sorted(unobserved)),
                             seconds=time.perf_counter() - started)


def compare(before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> WorkspaceChange:
    """Entries added, removed, or changed (another kind, size, modification time or link target)."""
    if before.root != after.root:
        raise ValueError("snapshots of different roots: {} and {}".format(before.root, after.root))
    old, new = before.entries, after.entries
    return WorkspaceChange(added=tuple(sorted(set(new) - set(old))),
                           removed=tuple(sorted(set(old) - set(new))),
                           changed=tuple(sorted(p for p in set(old) & set(new) if old[p] != new[p])))


def describe(change: WorkspaceChange, *, limit: int = 40) -> list[str]:
    """Human-readable lines naming every kind of difference (the first `limit` paths of each kind, then a count)."""
    lines: list[str] = []
    for label, paths in (("ADDED", change.added), ("REMOVED", change.removed), ("CHANGED", change.changed)):
        if not paths:
            continue
        lines.append("{} ({}):".format(label, len(paths)))
        lines.extend("    " + p for p in paths[:limit])
        if len(paths) > limit:
            lines.append("    ... and {} more".format(len(paths) - limit))
    return lines
