"""Read-only repository observation. Imports no executor, no configuration.

Author: Monzia Moodie

WHY A SEPARATE OWNER
====================
`cleanup_apply_adapter` and `cleanup_authorization` each carried a private
`git_checked`. Two nearly identical Git wrappers drift. This is the single
read-only contract both can consume without the reporting tool importing
deletion machinery.

Shared OBSERVATION does not mean shared ACTION policy: a reporting tool may
describe a symbolic link, while the executor refuses to act on one. This
module answers questions; it decides nothing.

WHAT IT KEEPS OUT
=================
The cleanup executor, production authorization, operation receipts and
foundation configuration. Read-only reporting must work without any of them.

WHAT "READ-ONLY" MEANS HERE
===========================
No intended mutation of candidates, supporting artifacts or repository state.
It is not a claim that reading has no possible filesystem side effect
whatsoever -- Git itself may touch its own caches.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class InspectionError(RuntimeError):
    """An observation could not be completed. NEVER an absence result."""


class RepositorySelectionError(InspectionError):
    """The intended working tree could not be established."""


class TrackingState(Enum):
    TRACKED = "tracked"
    UNTRACKED = "untracked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    """A tri-state answer. MEASURED 2026-09-08 in the original script:

        def tracked(p):
            return subprocess.run([...]).returncode == 0

    Outside a repository that is always False, so failed inspection became
    "untracked" and then deletion eligibility. UNKNOWN exists so operational
    failure has no path to an absence result.
    """

    state: TrackingState
    detail: str = ""


@dataclass(frozen=True)
class RepositoryInspection:
    """One validated working tree, answering questions about itself."""

    root: Path

    def _git(self, *arguments):
        environment = dict(os.environ)
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                     "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"):
            environment.pop(name, None)
        try:
            return subprocess.run(["git", "-C", str(self.root), *arguments],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=120,
                                  check=False, env=environment)
        except (OSError, subprocess.SubprocessError) as exc:
            raise InspectionError(
                "git {} could not be executed: {}: {}".format(
                    " ".join(arguments), type(exc).__name__, exc)) from exc

    def tracking(self, relative_path: str) -> Observation:
        """TRACKED, UNTRACKED, or UNKNOWN. Exit 1 means no match, not failure.

        `--literal-pathspecs` is a GLOBAL option and precedes the subcommand,
        so a filename containing pathspec magic is matched rather than
        interpreted. Any indexed output means tracked; no comparison is made
        against a caller's spelling.
        """
        result = self._git("--literal-pathspecs", "ls-files", "--cached",
                           "-z", "--", relative_path)
        if result.returncode != 0:
            return Observation(TrackingState.UNKNOWN,
                               result.stderr.decode("utf-8", "replace").strip()
                               [:200] or "git ls-files failed")
        return Observation(TrackingState.TRACKED
                           if result.stdout.strip(b"\x00")
                           else TrackingState.UNTRACKED)

    def head_paths(self):
        """(commit, every path in HEAD's tree). Membership from a SET.

        MEASURED 2026-09-08: `git cat-file -e HEAD:scripts/train.py` for a
        path absent from a valid commit exits **128**, not 1 --

            fatal: path 'scripts/train.py' does not exist in 'HEAD'

        The previous implementation assumed 1, so an ORDINARY ABSENCE became
        UNKNOWN and category A reported INCOMPLETE with zero candidates. The
        repair is not to match English diagnostics, which vary by version and
        locale: a successfully enumerated tree establishes the set against
        which absence can be tested, and a failure to enumerate remains an
        inspection failure.

        Resolved ONCE per inspection, so categories A and B cannot silently
        examine different commits. That does not make the working tree and
        index a simultaneous snapshot, and this does not claim it does.
        """
        head = self._git("rev-parse", "--verify", "HEAD^{commit}")
        if head.returncode != 0:
            raise InspectionError(
                "could not resolve HEAD: {}".format(
                    head.stderr.decode("utf-8", "replace").strip()[:200]))
        commit = head.stdout.decode("ascii", "replace").strip()
        listing = self._git("ls-tree", "-r", "-z", "--name-only",
                            "--full-tree", commit)
        if listing.returncode != 0:
            raise InspectionError(
                "could not enumerate committed paths: {}".format(
                    listing.stderr.decode("utf-8", "replace").strip()[:200]))
        paths = frozenset(os.fsdecode(item)
                          for item in listing.stdout.split(b"\x00") if item)
        return commit, paths

    def ignored(self, relative_path: str) -> Observation:
        """Whether the path is ignored. check-ignore exits 1 for "not ignored".

        Any other status -- notably 128 outside a repository -- is UNKNOWN.
        """
        result = self._git("check-ignore", "--quiet", "--", relative_path)
        if result.returncode == 0:
            return Observation(TrackingState.TRACKED)
        if result.returncode == 1:
            return Observation(TrackingState.UNTRACKED)
        return Observation(TrackingState.UNKNOWN,
                           result.stderr.decode("utf-8", "replace").strip()
                           [:200] or "git check-ignore exited {}".format(
                               result.returncode))


def select_repository(repo_root=None) -> RepositoryInspection:
    """Conservative selection. A subdirectory never silently widens scope.

    `rev-parse --show-toplevel` rather than a `.git` DIRECTORY test: a linked
    worktree's metadata is a gitfile, and a directory test would reject it.
    """
    supplied = Path(repo_root) if repo_root is not None else Path.cwd()
    if not supplied.is_dir():
        raise RepositorySelectionError("{} is not a directory".format(supplied))
    probe = RepositoryInspection(root=supplied)
    result = probe._git("rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise RepositorySelectionError(
            "no Git working tree at {}: {}".format(
                supplied, result.stderr.decode("utf-8", "replace").strip()))
    top = Path(result.stdout.decode("utf-8", "replace").strip()).resolve()
    if supplied.resolve() != top:
        raise RepositorySelectionError(
            "{} is a SUBDIRECTORY of the working tree {}. Refusing rather "
            "than silently widening discovery to the whole repository. Run "
            "from the root, or pass --repo-root {}.".format(
                supplied.resolve(), top, top))
    return RepositoryInspection(root=top)
