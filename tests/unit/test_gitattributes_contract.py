"""The .gitattributes rules are a contract, not a convention.

GITATTRIBUTES-UNGATED-1
=======================
MEASURED 2026-08-19: .gitattributes carries 37 rule lines and a documented
near-corruption of a test fixture, and NO test asserted any of them. Delete
`*.py text eol=lf` and nothing failed.

WHY THIS MATTERS MORE FOR THE BINARY RULES
The file records the incident that produced them, on 2026-07-12:

    tests/fixtures/alphafold/AF-E7ENB7-F1-model_v4.cif was committed as a
    99,647-byte blob while the working copy was 101,171 bytes -- exactly 1,524
    carriage returns stripped, one per line.

Benign there, because the mmCIF parser is line-ending-agnostic. But the file
states the consequence plainly:

    had this fixture been genuinely binary (a parquet, an .npy),
    normalization would have SILENTLY CORRUPTED it rather than merely
    shortening it.

A .npy whose bytes git has "helpfully" rewritten does not fail loudly. It
loads, and the numbers are wrong.

WHY git check-attr AND NOT PARSING THE FILE
Parsing .gitattributes here would reproduce git's pattern semantics --
precedence, `**` matching, later rules overriding earlier -- which is a second
implementation of someone else's parser, and the parallel-vocabulary defect
this project keeps removing. `git check-attr` is GIT ANSWERING FOR ITSELF, so
these tests assert what git will actually do.

It also works on paths that DO NOT EXIST, which is the point: the guard
protects the next .npy someone adds, not only the fixtures present today.
MEASURED: 0 tracked .npy, .gz, .sqlite, .joblib, .pkl or .png files, and every
one of those rules still resolves.

THE INVARIANT IS ABOUT THE INDEX, NOT THE WORKING TREE
124 of 981 tracked Python files are CRLF in this Windows working tree, and that
is CORRECT -- `core.autocrlf=true` with `eol=lf` means LF in the repository and
native endings on checkout. WORKTREE-EOL-DRIFT-1 recorded this on 2026-08-11 as
"benign for commits; load-bearing for byte-exact tooling."

So the assertion is that nothing ENTERS the repository with carriage returns.
MEASURED 2026-08-19: 981 tracked .py files, 0 with CRLF in the committed blob.

Author: Monzia Moodie
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ATTRIBUTES = _REPO / ".gitattributes"


def _git(*args) -> str:
    """Run git IN THE REPOSITORY THIS FILE BELONGS TO, never the caller's cwd.

    Measured 2026-08-19: run from a temporary directory, an earlier draft had
    37 of 38 cases fail loudly on the wrong _REPO -- which is correct -- while
    ONE passed, because it read `git ls-files` output that had inherited the
    shell's working directory. A test that passes wherever a clean repository
    happens to be current is not testing this repository.

    -C forces the directory, and check=False keeps a non-repository answerable
    rather than raising something unrelated to what is being asserted.
    """
    if not (_REPO / ".git").exists():
        return ""
    return subprocess.run(("git", "-C", str(_REPO)) + args, capture_output=True,
                          text=True, timeout=120).stdout


def _attrs(path: str) -> dict:
    """What GIT says the attributes are. The path need not exist."""
    out = _git("check-attr", "text", "eol", "binary", "--", path)
    found = {}
    for line in out.strip().splitlines():
        # Format: <path>: <attr>: <value>; the path may itself contain colons.
        parts = line.rsplit(": ", 2)
        if len(parts) == 3:
            found[parts[1]] = parts[2]
    return found


#: Every assertion here asks GIT. Without a working tree there is nothing to
#: ask, and a silent pass would be worse than a skip.
_HAS_GIT = (_REPO / ".git").exists() and bool(_git("rev-parse", "--git-dir").strip())
pytestmark = pytest.mark.skipif(
    not _HAS_GIT,
    reason="{} is not a git working tree; check-attr cannot answer".format(_REPO))


def test_the_repository_root_is_THIS_repository():
    """The anchor itself, asserted. If parents[2] ever stops naming the
    repository root, every other test in this file becomes a statement about
    some other directory -- and one of them would PASS."""
    assert (_REPO / ".git").exists(), _REPO
    assert _ATTRIBUTES.is_file(), _ATTRIBUTES
    assert (_REPO / "tests" / "unit").is_dir(), _REPO


# ---- text extensions resolve to LF --------------------------------------
@pytest.mark.parametrize("suffix", [
    ".py", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".lock", ".sh",
])
def test_text_extensions_resolve_to_LF(suffix):
    """Source, configuration, documentation and shell scripts are LF in the
    repository regardless of the platform that wrote them."""
    a = _attrs("probe_gitattributes_contract" + suffix)
    assert a.get("text") == "set", (suffix, a)
    assert a.get("eol") == "lf", (suffix, a)


@pytest.mark.parametrize("suffix", [".ps1", ".bat", ".cmd"])
def test_windows_script_extensions_resolve_to_CRLF(suffix):
    """Windows-native scripts are checked out CRLF deliberately. cmd.exe and
    older PowerShell hosts are not reliably LF-tolerant."""
    a = _attrs("probe_gitattributes_contract" + suffix)
    assert a.get("text") == "set", (suffix, a)
    assert a.get("eol") == "crlf", (suffix, a)


# ---- binary extensions are NEVER normalised -----------------------------
@pytest.mark.parametrize("suffix", [
    ".gz", ".zip", ".sqlite", ".db", ".pkl", ".joblib", ".parquet", ".feather",
    ".npy", ".npz", ".png", ".jpg", ".jpeg", ".pdf",
])
def test_binary_extensions_are_never_normalised(suffix):
    """THE ONE THAT MATTERS.

    A text file git normalises is merely shortened. A BINARY file git
    normalises is silently corrupted -- it still loads, and the numbers are
    wrong. `binary` sets text=unset, which is what forbids the rewrite.
    """
    a = _attrs("probe_gitattributes_contract" + suffix)
    assert a.get("binary") == "set", (suffix, a)
    assert a.get("text") == "unset", (
        "{}: text is {!r}; git would normalise a binary payload"
        .format(suffix, a.get("text")))


# ---- the fixture overrides protect files not yet added ------------------
@pytest.mark.parametrize("suffix", [".parquet", ".npy", ".npz", ".gz", ".sqlite"])
def test_fixture_binaries_are_protected_on_NONEXISTENT_paths(suffix):
    """MEASURED 2026-08-19: 0 tracked .npy, .gz or .sqlite fixtures exist. The
    protection must therefore cover the NEXT one added, which is exactly what
    check-attr on a nonexistent path verifies.

    A NOTE ON WHAT THIS DOES NOT PROVE. The `tests/fixtures/**` overrides are
    REDUNDANT while the general `*.parquet binary` rules exist -- measured by
    building two repositories differing only in those lines and comparing
    git's answers, which were byte-identical for both a fixture path and a
    root-level one. Sabotage confirmed it: deleting the override changes
    nothing and no test can detect it, because there is nothing to detect.

    They are defence-in-depth, kept so that narrowing a general rule later does
    not silently unprotect the fixtures. This test asserts the PROTECTION,
    which is what matters; it deliberately does not claim the override line is
    load-bearing, because it is not.
    """
    probe = "tests/fixtures/not_yet_added/probe" + suffix
    assert not (_REPO / probe).exists(), "the probe path must NOT exist"
    a = _attrs(probe)
    assert a.get("binary") == "set", (probe, a)
    assert a.get("text") == "unset", (probe, a)


def test_the_alphafold_fixture_rule_survives():
    """The rule the 2026-07-12 incident produced. A .cif is text, but its
    line endings must be the canonical LF the European Bioinformatics
    Institute serves -- not whatever the download cache wrote on Windows."""
    a = _attrs("tests/fixtures/alphafold/probe.cif")
    assert a.get("text") == "set", a
    assert a.get("eol") == "lf", a


def test_the_extensionless_ratchet_file_is_pinned():
    """EXPECTED_SUITE_SIZE has no extension, so only `* text=auto` matched it
    -- and that means CRLF in a Windows working tree while every other
    governed text file is LF. Pinned 2026-07-18."""
    a = _attrs("tests/EXPECTED_SUITE_SIZE")
    assert a.get("text") == "set", a
    assert a.get("eol") == "lf", a


@pytest.mark.parametrize("path", [".gitattributes", ".gitignore"])
def test_repository_metadata_is_pinned_to_LF(path):
    a = _attrs(path)
    assert a.get("text") == "set", (path, a)
    assert a.get("eol") == "lf", (path, a)


# ---- and the invariant the rules exist to produce -----------------------
def test_no_tracked_python_file_has_CRLF_in_the_INDEX():
    """THE INVARIANT. Not that the working tree is LF -- MEASURED 2026-08-19,
    124 of 981 tracked .py files are CRLF here, and that is correct under
    core.autocrlf=true. The assertion is that nothing ENTERS the repository
    with carriage returns.

    Reads `git ls-files --eol`, which reports the index and working-tree
    endings side by side, rather than materialising 981 blobs.

    THE GUARDED STATE IS REACHABLE, not hypothetical. Measured 2026-08-19 in an
    isolated repository with `* -text`: a file written with carriage returns
    committed as `i/crlf`, and its blob carried them. Normalisation is what
    prevents that here, and normalisation is exactly what the rules configure.
    """
    out = _git("ls-files", "--eol", "--", "*.py")
    bad = []
    for line in out.strip().splitlines():
        # Format: i/<index-eol> w/<worktree-eol> attr/<attrs>\t<path>
        fields, _, path = line.partition("\t")
        index_eol = next((f[2:] for f in fields.split() if f.startswith("i/")), "")
        if index_eol not in ("lf", "none", ""):
            bad.append((path.strip(), index_eol))
    assert not bad, (
        "{} tracked Python file(s) are not LF in the index: {}"
        .format(len(bad), bad[:10]))


def test_the_attributes_file_itself_is_LF_and_ASCII():
    """A rules file that is itself inconsistent with its rules would be a poor
    advertisement for them."""
    raw = _ATTRIBUTES.read_bytes()
    assert raw.count(b"\r\n") == 0, "CRLF in .gitattributes"
    assert not any(b > 0x7F for b in raw), "non-ASCII in .gitattributes"


def test_every_rule_line_is_reachable_by_check_attr():
    """A rule that git does not honour is a comment with extra steps.

    Parses only the PATTERN column -- which is unambiguous -- and asks git
    whether any attribute resolves for a path matching it. Deliberately does
    NOT reimplement git's matching to predict WHICH attribute.
    """
    lines = _ATTRIBUTES.read_text(encoding="utf-8").splitlines()
    patterns = [l.split()[0] for l in lines
                if l.strip() and not l.strip().startswith("#")]
    assert len(patterns) >= 30, len(patterns)
    unreachable = []
    for pat in patterns:
        if pat == "*":
            continue                      # the text=auto default; always applies
        probe = pat.replace("**/", "not_yet_added/").replace("*", "probe")
        a = _attrs(probe)
        if not any(v not in ("unspecified", "") for v in a.values()):
            unreachable.append(pat)
    assert not unreachable, (
        "rule(s) for which git resolves NO attribute: {}".format(unreachable))
