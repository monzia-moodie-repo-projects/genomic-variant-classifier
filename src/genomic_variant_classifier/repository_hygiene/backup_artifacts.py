"""One authority for what belongs in a repository, and what is residue.

INSTALLER-TRANSACTION-1, step 5.

    A filter that reports zero is not evidence of zero. It is evidence about
    the filter.

WHY THIS MODULE EXISTS
The retirement tool scanned `*.bak_*` alone, retired 148 artefacts, and
reported "remaining .bak_* artefact(s): 0" while 107 more sat beside them in a
shape it never looked for. Extending that one list fixed that one tool.

MEASURED 2026-08-20, before this module was written:

    SECRET_PATTERNS  defined TWICE   repository_transaction.py:94  (11 entries)
                                     retire_backup_artifacts.py:109 (11 entries)
    SECRET_CANARIES  defined TWICE   repository_transaction.py:101 (7)
                                     retire_backup_artifacts.py:133 (7)
    BACKUP_PATTERNS  defined ONCE    retire_backup_artifacts.py:105
    EXCLUDED_ROOTS   defined ONCE    retire_backup_artifacts.py:84

The duplicated pairs were verified IDENTICAL at runtime -- element for element,
order included. That is the best case for consolidation and precisely what
makes the duplication dangerous: nothing enforced the agreement, and a future
edit to either copy would have drifted in silence. I wrote the second copy
myself, by transcribing the first.

A third consumer was about to be added. Three independently maintained lists is
the drift surface this module removes.

THREE DISTINCT QUESTIONS, THREE DISTINCT NAMES
They were being conflated, and conflating them is how `.gitignore` nearly
became the hygiene authority.

    NOT_THIS_REPOSITORY   .venv312, .git, node_modules -- contents that are not
                          this project's artefacts at all. A vendored library's
                          backup file is not our detritus.

    SCRATCH_ROOTS         .af_fix_work -- a directory EXPLICITLY classified as
                          working space where rollback artefacts are permitted.

    BACKUP_SHAPES         the filename shapes that indicate rollback residue.

`.gitignore` answers "should git normally show this path?", NOT "may this path
legitimately contain rollback detritus?". Deriving hygiene from ignore rules
would mean anyone adding `some_directory/` to `.gitignore` silently confers
scratch legitimacy. Scratch roots are declared HERE, and a test asserts the two
correspond without either deriving meaning from the other.

RELOCATION RESOLUTION
`scripts/verify_written_cohorts.py.bak` was classified "unclassified orphan"
because no file existed at its derived path. Manual investigation then found
`scripts/forensics/verify_written_cohorts.py` -- tracked, 171 lines against the
backup's 171, differing in exactly two prose passages where the canonical file
records that a defect was fixed.

It was a superseded relocated preimage, and basename search plus git history
would have said so automatically. `resolve_relocation()` performs that search,
and it never guesses: exact blob identity or a tracked basename match with a
recorded successor commit, or nothing.

Author: Monzia Moodie
"""
from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

#: Filename shapes that indicate installer rollback residue.
#:
#: MEASURED 2026-08-19, the shapes actually present in this repository:
#:     foo.py.bak_2026-08-19_164056   the PowerShell installers
#:     foo.py.pre_cfgroot.bak         the Python appliers
#:     foo.py.precosmic.bak           older appliers, no underscore
#:     foo.py.20260702_183508.bak     a dated convention in .af_fix_work
#:     foo.py.bak                     bare
BACKUP_SHAPES = ("*.bak_*", "*.bak", "*.orig", "*.rej")

#: Directories whose contents are NOT this project's artefacts. A vendored
#: library's backup file is not our detritus, and scanning them wastes time
#: and produces noise.
NOT_THIS_REPOSITORY = (".venv312", "renv", ".git", "node_modules",
                       "__pycache__", ".pytest_cache", ".mypy_cache")

#: Directories EXPLICITLY declared as working space, where rollback artefacts
#: are permitted. This is the hygiene authority -- NOT `.gitignore`.
#:
#: .af_fix_work holds 12 dated backups from the 2026-07 AlphaFold repair. It is
#: also git-ignored at .gitignore:198, and a test asserts that correspondence,
#: but the legitimacy comes from THIS declaration.
SCRATCH_ROOTS = (".af_fix_work",)

#: Path shapes that may carry credential material. The single source of truth;
#: repository_transaction.py and retire_backup_artifacts.py both import it.
SECRET_PATTERNS = (
    ".env", "*.env", "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials*", "token*", "secrets*", "*_rsa", "*_ed25519",
)

#: Shapes the secret classifier MUST recognise. Emptying SECRET_PATTERNS raises
#: no error and leaks nothing in any single run -- it simply reclassifies a
#: credential file as ordinary. An absence-of-failure check cannot see that.
SECRET_CANARIES = (".env", "id_rsa", "server.pem", "api.key",
                   "credentials.json", "token.txt", "secrets.yaml")

#: Roots whose artefacts a deployment excludes, so a sentinel or a required
#: file must never live under one.
ARTEFACT_ROOTS = ("tests", "test", "docs", "doc", "build", "dist",
                  "notebooks", "htmlcov", ".github", "logs", "outputs")


class HygieneError(RuntimeError):
    """A refusal by the hygiene classifier."""


class ArtifactClass(str, Enum):
    """What a backup-shaped path IS, decided by evidence rather than name."""

    #: git holds these exact bytes for this path. Redundant.
    GIT_EXACT_PREIMAGE = "git_exact_preimage"
    #: A working-tree state captured mid-edit and superseded before any commit.
    SUPERSEDED_PREIMAGE = "superseded_uncommitted_preimage"
    #: The original moved; this is a predecessor of a file now living elsewhere.
    RELOCATED_PREIMAGE = "relocated_preimage"
    #: Credential-shaped. Recorded by digest and structure, never by content.
    SECRET_BEARING = "secret_bearing"
    #: Inside a declared scratch root. Legitimate, not residue.
    SCRATCH = "scratch"
    #: Evidence is insufficient. NEVER deleted.
    UNCLASSIFIED = "unclassified"


def assert_secret_detection_intact() -> None:
    """Refuse to operate if the secret classifier has been weakened."""
    missed = [c for c in SECRET_CANARIES if not is_secret_path(c)]
    if missed:
        raise HygieneError(
            "the secret-path classifier does not recognise {}. A weakened "
            "classifier lets credential material be treated as ordinary."
            .format(missed))


def is_secret_path(path) -> bool:
    return any(fnmatch.fnmatch(Path(path).name, pat) for pat in SECRET_PATTERNS)


def is_backup_shaped(path) -> bool:
    name = Path(path).name
    return any(fnmatch.fnmatch(name, pat) for pat in BACKUP_SHAPES)


def in_scratch_root(relpath) -> bool:
    """Whether a path lies inside an EXPLICITLY declared scratch root.

    Deliberately not `.gitignore`-derived: an ignore rule answers a different
    question, and letting it confer scratch legitimacy would mean any unrelated
    ignore entry silently widens the hygiene exception.
    """
    parts = Path(str(relpath).replace("\\", "/")).parts
    return bool(parts) and parts[0] in SCRATCH_ROOTS


def in_excluded_root(relpath) -> bool:
    parts = Path(str(relpath).replace("\\", "/")).parts
    return any(p in NOT_THIS_REPOSITORY for p in parts)


def under_artefact_root(relpath) -> bool:
    parts = Path(str(relpath).replace("\\", "/")).parts
    return bool(parts) and parts[0] in ARTEFACT_ROOTS


def strip_backup_suffix(relpath):
    """The name a backup was taken FROM, by suffix alone -- no filesystem.

    Used to decide whether a path SHAPE is credential-bearing, which must be
    answerable even when the original no longer exists. MEASURED 2026-08-19: an
    ordering that consulted the filesystem first let a `.env.pre_token.bak`
    whose live `.env` was gone fall through to unclassified, losing the shape
    metadata that is the whole point of the secret branch.
    """
    name = Path(str(relpath).replace("\\", "/")).name
    if ".bak_" in name:
        base = name.rsplit(".bak_", 1)[0]
    elif name.endswith(".bak"):
        base = name[: -len(".bak")]
    elif name.endswith(".orig"):
        base = name[: -len(".orig")]
    elif name.endswith(".rej"):
        base = name[: -len(".rej")]
    else:
        return None
    if "." in base and not base.startswith("."):
        head, _, tail = base.rpartition(".")
        if tail.startswith("pre") or tail.replace("_", "").isdigit():
            return head
        return base
    if base.count(".") > 1:
        return "." + base.lstrip(".").split(".")[0]
    return base


@dataclass(frozen=True)
class Relocation:
    """Proof that a backup's original moved, never a guess."""

    original_now_at: str
    evidence: str
    successor_commit: str | None


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(("git", "-C", str(repo)) + args,
                              capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def resolve_relocation(repo: Path, backup_relpath: str):
    """Find where a backup's original went, by EVIDENCE or not at all.

    MEASURED 2026-08-20: `scripts/verify_written_cohorts.py.bak` had no file at
    its derived path and was classified UNCLASSIFIED. Manual investigation
    found `scripts/forensics/verify_written_cohorts.py` -- tracked, arriving at
    0b93d30 ("archive 62 spent forensic scripts"), 171 lines against the
    backup's 171, differing in exactly two prose passages where the canonical
    file records that a defect was fixed. A superseded relocated preimage.

    Two admissible proofs, in order of strength:

        1. EXACT BLOB IDENTITY. The backup's bytes appear in git history under
           some path. Then the relocation is certain.

        2. A TRACKED FILE WITH THE SAME BASENAME, and exactly one such file.
           Weaker, so the evidence string says so and the caller decides.

    Anything else returns None. A classifier that guesses is worse than one
    that refuses.
    """
    repo = Path(repo)
    base = strip_backup_suffix(backup_relpath)
    if not base:
        return None
    basename = Path(base).name

    blob = _git(repo, "hash-object", "--", str(backup_relpath)).strip()
    if blob:
        found = _git(repo, "rev-list", "--objects", "--all")
        for line in found.splitlines():
            if line.startswith(blob):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    return Relocation(
                        original_now_at=parts[1],
                        evidence="exact blob identity in git history",
                        successor_commit=None)

    tracked = [p for p in _git(repo, "ls-files").splitlines()
               if Path(p).name == basename]
    if len(tracked) == 1:
        log = _git(repo, "log", "-1", "--format=%H", "--", tracked[0]).strip()
        return Relocation(
            original_now_at=tracked[0],
            evidence="exactly one tracked file shares the basename",
            successor_commit=log or None)
    return None


def iter_repository_detritus(repo, *, include_scratch: bool = False):
    """Every backup-shaped path that is repository detritus.

    Scratch roots are excluded by default -- they are declared working space --
    and directories that are not this project's artefacts always are.
    """
    repo = Path(repo)
    seen = set()
    for shape in BACKUP_SHAPES:
        for p in repo.rglob(shape):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            rel = p.relative_to(repo).as_posix()
            if in_excluded_root(rel):
                continue
            if in_scratch_root(rel) and not include_scratch:
                continue
            yield rel
