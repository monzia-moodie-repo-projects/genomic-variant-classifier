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
import os
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

    # NOTHING MOVED IF THE ORIGINAL IS STILL THERE.
    #
    # RELOCATION-FALSE-POSITIVE-1 (2026-08-20). This function searched for a
    # tracked file sharing the basename WITHOUT first checking whether the
    # original still sat at its derived path. So an ordinary backup resolved to
    # its own unmoved original:
    #
    #     README.md.bak_2026-08-20_065912
    #         derived original : README.md         EXISTS=True
    #         resolve_relocation -> README.md      <- claimed a relocation
    #
    #     scripts/verify_written_cohorts.py.bak
    #         derived original : scripts/verify_written_cohorts.py  EXISTS=False
    #         resolve_relocation -> scripts/forensics/...           <- genuine
    #
    # The retirement tool never saw this because it asks only inside its
    # `original is None` branch. iter_repository_detritus asked
    # unconditionally, excluded EIGHT ordinary artefacts as "relocated", and
    # reported ZERO detritus -- a vacuous invariant.
    #
    # The guard belongs HERE rather than at each call site: a function that
    # answers "where did the original move to" must not answer at all when the
    # original did not move.
    derived = (Path(backup_relpath).parent / base).as_posix().lstrip("./")
    if (repo / derived).exists():
        return None

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


def iter_repository_detritus(repo, *, include_scratch: bool = False,
                             include_relocated: bool = False):
    """Every backup-shaped path that is repository detritus.

    Scratch roots are excluded by default -- they are declared working space.
    Relocated preimages are excluded by default because the retirement tool
    RETAINS them deliberately, and a repository-wide invariant that reported a
    file its own classifier had decided to keep would set the two authorities
    against each other.

    ONE PRUNED WALK, NOT FOUR FULL ONES.
    DETRITUS-WALK-COST-1 (2026-08-20). This called repo.rglob() once PER
    PATTERN, and rglob cannot prune: it descends into every directory and the
    caller discards the results afterwards. MEASURED on this repository:

        rglob("*.bak_*")   1.931s   4 matches
        rglob("*.bak")     1.889s  13 matches
        rglob("*.orig")    1.900s   0 matches
        rglob("*.rej")     1.901s   0 matches
        -------------------------------------
        total              7.572s   5 reported files

        rglob("*") whole tree      2.093s   135,832 entries
        os.walk with pruning       0.478s    43,070 files

    92,762 entries -- almost all of .venv312 -- were being enumerated FOUR
    TIMES and thrown away. os.walk prunes by mutating `dirs` in place, so the
    excluded roots are never descended into at all. Roughly sixteen times
    faster, and the cost now scales with the project rather than with its
    virtual environment.

    That matters because this is a repository-wide invariant: it runs in every
    suite execution, locally and twice per continuous-integration run. A check
    people avoid running is a check that has stopped working.
    """
    repo = Path(repo)
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in NOT_THIS_REPOSITORY]
        here = Path(root)
        for name in files:
            if not is_backup_shaped(name):
                continue
            rel = (here / name).relative_to(repo).as_posix()
            if in_scratch_root(rel) and not include_scratch:
                continue
            if not include_relocated and resolve_relocation(repo, rel) is not None:
                # A predecessor of a file that MOVED, which the retirement tool
                # retains with its successor recorded. Not detritus.
                #
                # resolve_relocation returns None when the original is still at
                # its derived path, so an ordinary backup is never excluded
                # here -- see RELOCATION-FALSE-POSITIVE-1.
                continue
            yield rel
