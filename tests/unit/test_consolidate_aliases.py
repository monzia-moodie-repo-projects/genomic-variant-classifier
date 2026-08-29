"""Folding an alias directory must never destroy an alias file.

Created 2026-08-29 after `ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1`.

THE DEFECT THESE TESTS GUARD
----------------------------
`scripts/maintenance/consolidate_aliases.py` folds an alias directory into its
canonical name and then REMOVES the alias directory. Until 2026-08-29 the
collision check and the post-merge verification both compared `st_size`.

REPRODUCED on 2026-08-29 with two files named `scores.csv`, both exactly
612,501 bytes and different content:

    exit 0   "merged + verified; removed .../spliceai_scores"
    the alias file was DESTROYED

The script never overwrites -- `shutil.copy2` is skipped when the target
exists -- so it discarded the SOURCE instead, which is worse: the loss is
silent and the report says "verified".

612,501 bytes is not an arbitrary figure. The artifact lineage census of
2026-08-28 measured exactly that size for two EVE score files with different
digests:

    data/external/eve/EVE_all_data/variant_files/TPIS_HUMAN.csv  465d9fd2...
    data/external/eve/EVE_all_data/variant_files/TSHB_HUMAN.csv  2ef2b73a...

WHY THIS FILE EXISTS AT ALL
---------------------------
`consolidate_aliases.py` had NO test. A size comparison guarding an
irreversible deletion survived because nothing exercised it. Of the five
scripts that read `configs/data_manifest.yaml`, only the storage guard was
bound by a test.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; EVE = Evolutionary model of
Variant Effect; CSV = comma-separated values.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "maintenance" / "consolidate_aliases.py"

#: The measured size of the two EVE files that are equal in size and different
#: in content. Using the real figure keeps the test tied to the observation.
_COLLIDING_SIZE = 612_501


def _manifest(tmp_path: Path) -> Path:
    p = tmp_path / "configs"
    p.mkdir(parents=True, exist_ok=True)
    m = p / "data_manifest.yaml"
    m.write_text(
        "version: 1\n"
        "sources:\n"
        "  spliceai:\n"
        "    location: external\n"
        "    tier: public\n"
        "    class: public_redownloadable\n"
        "    aliases: [spliceai_scores]\n",
        encoding="utf-8")
    return m


def _run(tmp_path: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "-B", str(_SCRIPT),
         "--data-dir", str(tmp_path / "data"),
         "--manifest", str(_manifest(tmp_path)), *extra],
        capture_output=True, text=True, timeout=300)


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dirs(tmp_path: Path):
    alias = tmp_path / "data" / "external" / "spliceai_scores"
    canon = tmp_path / "data" / "external" / "spliceai"
    alias.mkdir(parents=True, exist_ok=True)
    canon.mkdir(parents=True, exist_ok=True)
    return alias, canon


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_equal_size_different_content_aborts_and_preserves_the_alias(tmp_path):
    """THE DEFECT. Two files of equal size and different content.

    Under the size comparison this reported "merged + verified", removed the
    alias directory, and lost the alias file.
    """
    alias, canon = _dirs(tmp_path)
    (alias / "scores.csv").write_bytes(b"A" * _COLLIDING_SIZE)
    (canon / "scores.csv").write_bytes(b"B" * _COLLIDING_SIZE)
    kept = _digest(alias / "scores.csv")

    r = _run(tmp_path, "--execute")

    assert r.returncode == 1, r.stdout + r.stderr
    assert "ABORT" in r.stdout
    assert alias.is_dir(), "the alias directory was removed despite a collision"
    assert (alias / "scores.csv").is_file()
    assert _digest(alias / "scores.csv") == kept, "the alias file was altered"


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_the_sizes_really_are_equal(tmp_path):
    """The fixture must exercise the defect, not merely differ.

    If the two files differed in SIZE, the old code would also have aborted and
    the test above would pass against the defective script.
    """
    alias, canon = _dirs(tmp_path)
    (alias / "scores.csv").write_bytes(b"A" * _COLLIDING_SIZE)
    (canon / "scores.csv").write_bytes(b"B" * _COLLIDING_SIZE)
    a, b = alias / "scores.csv", canon / "scores.csv"
    assert a.stat().st_size == b.stat().st_size == _COLLIDING_SIZE
    assert _digest(a) != _digest(b)


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_identical_content_merges_and_removes_the_alias(tmp_path):
    """The PERMISSIVE direction. A digest check must not refuse a real merge."""
    alias, canon = _dirs(tmp_path)
    (alias / "same.csv").write_bytes(b"IDENTICAL BYTES")
    (canon / "same.csv").write_bytes(b"IDENTICAL BYTES")
    (alias / "only_in_alias.csv").write_bytes(b"unique to the alias")

    r = _run(tmp_path, "--execute")

    assert r.returncode == 0, r.stdout + r.stderr
    assert not alias.exists(), "a clean merge must remove the alias directory"
    assert (canon / "only_in_alias.csv").read_bytes() == b"unique to the alias"


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_dry_run_changes_nothing(tmp_path):
    """The default must be inert on a MERGEABLE alias.

    A colliding fixture would abort before reaching the execute branch, so the
    test could not see whether `--execute` was honoured. MEASURED: with a
    colliding fixture, deleting `if not args.execute: continue` changed
    nothing and every test still passed.
    """
    alias, canon = _dirs(tmp_path)
    (alias / "payload.csv").write_bytes(b"payload")
    (canon / "other.csv").write_bytes(b"other")

    r = _run(tmp_path)

    assert "DRY-RUN" in r.stdout
    assert alias.is_dir(), "dry-run removed the alias directory"
    assert (alias / "payload.csv").is_file(), "dry-run consumed the alias file"
    assert not (canon / "payload.csv").exists(), "dry-run WROTE into canonical"


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_the_two_defensive_guards_are_present(tmp_path):
    """Two guards are UNREACHABLE from a single-process test, and both stay.

    MEASURED by sabotage on 2026-08-29: deleting either changed no test.

        `if _files(alias_dir):` before removing an "empty" alias guards a
        RACE -- the directory was empty when planned and is not when executed.
        No test can put a file there in between.

        `if bad:` after the merge fires only when a copy SUCCEEDED and the
        target then failed the content comparison. A correct copy cannot
        produce it.

    This repository has ruled on unreachable defences: `suite_transition.py`
    DELETED three, and `publish()`'s re-parse survived only once a reachable
    case was found. These two are different -- they guard against a filesystem
    changing underfoot and against a copy that silently did not happen, which
    are real conditions on a real machine, not impossible states.

    So they are kept and asserted STRUCTURALLY, with the reason recorded here
    rather than left implicit in an unexercised branch.
    """
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "if _files(alias_dir):" in text, (
        "the re-confirm-empty guard was removed; an alias that gained a file "
        "between planning and execution would be deleted with its contents")
    assert "if bad:" in text, (
        "the post-merge verification guard was removed; the alias directory "
        "would be deleted without confirming its files reached canonical")


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_an_empty_alias_directory_is_removed(tmp_path):
    alias, _canon = _dirs(tmp_path)
    r = _run(tmp_path, "--execute")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not alias.exists()


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_a_populated_alias_moves_into_an_empty_canonical(tmp_path):
    alias, canon = _dirs(tmp_path)
    (alias / "payload.csv").write_bytes(b"payload")
    r = _run(tmp_path, "--execute")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not alias.exists()
    assert (canon / "payload.csv").read_bytes() == b"payload"


@pytest.mark.skipif(not _SCRIPT.is_file(), reason="consolidate_aliases.py absent")
def test_the_script_compares_content_not_size(tmp_path):
    """STRUCTURAL. The two decisions that guard a deletion must not read
    `st_size` as their verdict.

    A behavioural test can be satisfied by a script that happens to abort; this
    asserts that the comparison itself is a digest, so a future edit cannot
    reintroduce the defect while the behavioural tests still pass.
    """
    import ast

    text = _SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # THE IMPORT, not the substring. MEASURED: asserting `"hashlib" in text`
    # passed after the import was deleted, because `hashlib.sha256` remained
    # inside the function body. The script raised NameError at runtime and
    # three OTHER tests caught it -- this one did not.
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    assert "hashlib" in imported, (
        "hashlib is not IMPORTED; a mention inside a function body is not an "
        "import, and the script would raise NameError")

    assert "st_size !=" not in text.replace(
        "if a.stat().st_size != b.stat().st_size:", ""), (
        "a size INEQUALITY still decides an outcome; the only admissible use "
        "is the cheap pre-check inside the content comparison")
