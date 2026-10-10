"""The repository's PATH BUDGET: every tracked path must fit the Windows limits from any checkout or clone root of up to
ROOT_ALLOWANCE_CHARS characters.

MEASURED 2026-10-10 (owner run, upload 544a2b43): the lockfile-migration installer's apply stage failed twice on the owner's
Windows machine. Its disposable clone root, %TEMP%\\gvc_lockfile_migration_candidate_<23-character stamp>, is 90 characters; the
longest path the patch wrote was 169 characters; together, with the separator, 260 -- one more than a Windows path may hold.
Nothing in the repository bounded path length, and the installer simulations ran on Linux, which has no such limit.

THE TWO WINDOWS LIMITS (Microsoft, "Maximum Path Length Limitation"):
    a file path       MAX_PATH is 260 characters INCLUDING the terminating null       -> at most 259 characters
    a directory path  "cannot exceed MAX_PATH minus 12" (room for an 8.3 file name)   -> at most 247 characters
Git for Windows refuses a longer path ("Filename too long") unless core.longpaths is set; Python refuses it unless the
LongPathsEnabled registry value is set. Both are per-machine settings, and the repository must work without them: in a
disposable clone, inside Windows Sandbox, and on any collaborator's machine.

A repository-relative path P under a root R occupies len(R) + 1 + len(P). With R at most ROOT_ALLOWANCE_CHARS (108):
    every tracked FILE path            <= 259 - 1 - 108 = 150 characters
    every DIRECTORY on a tracked path  <= 247 - 1 - 108 = 138 characters
Lengths are counted in UTF-16 code units, the unit Windows counts (a character outside the Basic Multilingual Plane is two).

The allowance is a declared contract, not a measurement of one machine: a checkout or clone root longer than 108 characters is
outside what this budget guarantees (an installer measures its own root against the Windows limits before it clones).

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath

from .roles import RecordsOntologyError

logger = logging.getLogger(__name__)

__all__ = ["WINDOWS_MAX_FILE_PATH_CHARS", "WINDOWS_MAX_DIRECTORY_PATH_CHARS", "ROOT_ALLOWANCE_CHARS",
           "MAX_TRACKED_FILE_PATH_CHARS", "MAX_TRACKED_DIRECTORY_PATH_CHARS", "PathBudgetError", "windows_length",
           "budget_violations", "require_within_budget"]

WINDOWS_MAX_FILE_PATH_CHARS = 259          # MAX_PATH (260) less the terminating null
WINDOWS_MAX_DIRECTORY_PATH_CHARS = 247     # MAX_PATH - 12 (248) less the terminating null
ROOT_ALLOWANCE_CHARS = 108
MAX_TRACKED_FILE_PATH_CHARS = WINDOWS_MAX_FILE_PATH_CHARS - 1 - ROOT_ALLOWANCE_CHARS                 # 150
MAX_TRACKED_DIRECTORY_PATH_CHARS = WINDOWS_MAX_DIRECTORY_PATH_CHARS - 1 - ROOT_ALLOWANCE_CHARS       # 138


class PathBudgetError(RecordsOntologyError):
    """A repository path that does not fit the budget."""


def windows_length(text: str) -> int:
    """The length Windows counts: UTF-16 code units."""
    if type(text) is not str:
        raise PathBudgetError("a path must be text, not {}".format(type(text).__name__))
    return len(text.encode("utf-16-le")) // 2


def budget_violations(path: str) -> list:
    """Every way a repository-relative POSIX path exceeds the budget; empty when it fits.

    Only the file path and its PARENT directory are measured: every other directory on the path is a prefix of the parent, so it
    is shorter.
    """
    if type(path) is not str or path == "" or path != path.strip():
        raise PathBudgetError("{!r} is not a non-empty, trimmed path".format(path))
    if "\\" in path or PurePosixPath(path).is_absolute():
        raise PathBudgetError("{!r} is not a repository-relative POSIX path".format(path))
    problems = []
    n = windows_length(path)
    if n > MAX_TRACKED_FILE_PATH_CHARS:
        problems.append("file path {!r} is {} characters, over the budget of {} (Windows allows {} from a root of up to {})".format(
            path, n, MAX_TRACKED_FILE_PATH_CHARS, WINDOWS_MAX_FILE_PATH_CHARS, ROOT_ALLOWANCE_CHARS))
    parent = PurePosixPath(path).parent.as_posix()
    if parent != ".":
        m = windows_length(parent)
        if m > MAX_TRACKED_DIRECTORY_PATH_CHARS:
            problems.append("directory {!r} is {} characters, over the budget of {} (Windows allows {} from a root of up to {})".format(
                parent, m, MAX_TRACKED_DIRECTORY_PATH_CHARS, WINDOWS_MAX_DIRECTORY_PATH_CHARS, ROOT_ALLOWANCE_CHARS))
    return problems


def require_within_budget(path: str) -> None:
    problems = budget_violations(path)
    if problems:
        raise PathBudgetError("; ".join(problems))
