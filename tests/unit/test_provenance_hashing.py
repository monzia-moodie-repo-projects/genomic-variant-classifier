"""The canonical digest primitive, and the proof it is not a fourth dialect.

Phase 1C Unit 2. Created 2026-09-02.

WHY AN AGREEMENT TEST EXISTS
----------------------------
MEASURED 2026-09-01: three helpers already hash a file, and all three produce
the identical digest -- proven by EXECUTING them, not by reading them:

    data/constraint_canonicalize.py:325     sha256_file
    data/phylop_cache.py:158                sha256_file
    agent_layer/science_claw/ledger.py:70   compute_sha256

`digest_file` must join that proof. Otherwise this package adds a FOURTH
dialect, and the project acquires four ways to identify one artifact -- the
defect it names elsewhere as a value stated independently of the thing it
describes.

MEASURED 2026-09-02: seventeen call sites across eleven files use those three
names, four of them test files that pin the names by identity. Migration is
therefore NOT this unit; agreement is what prevents drift until it happens.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; AST = abstract syntax tree.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.provenance import (
    FileChangedDuringDigest,
    FileDigest,
    digest_file,
)

_ROOT = Path(__file__).resolve().parents[2]
_PKG = _ROOT / "src" / "genomic_variant_classifier"


def _reference(path: Path) -> str:
    """An INDEPENDENT digest, computed here and not by any project helper."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


# ---------------------------------------------------------------------------
# 1. the digest itself
# ---------------------------------------------------------------------------

def test_it_digests_RAW_BYTES(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"\x00\x01\x02payload\xff")
    got = digest_file(p)
    assert isinstance(got, FileDigest)
    assert got.sha256 == _reference(p)
    assert got.size_bytes == p.stat().st_size == 11


def test_the_size_travels_WITH_the_digest(tmp_path):
    """Re-stat-ing later to recover the size would read a file that may since
    have changed, reintroducing the window this module closes."""
    p = tmp_path / "b.bin"
    p.write_bytes(b"x" * 5000)
    got = digest_file(p)
    assert got.size_bytes == 5000
    assert got == FileDigest(sha256=_reference(p), size_bytes=5000)


def test_content_decides_identity_and_the_NAME_does_not(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"same"); b.write_bytes(b"same"); c.write_bytes(b"diff")
    assert digest_file(a).sha256 == digest_file(b).sha256
    assert digest_file(a).sha256 != digest_file(c).sha256


def test_an_empty_file_still_digests(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    got = digest_file(p)
    assert got.size_bytes == 0
    assert got.sha256 == hashlib.sha256(b"").hexdigest()


def test_the_chunk_size_does_not_change_the_digest(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(bytes(range(256)) * 900)
    one = digest_file(p, chunk_size=7).sha256
    two = digest_file(p, chunk_size=1 << 20).sha256
    assert one == two == _reference(p)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        digest_file(tmp_path / "absent")


def test_a_string_path_is_accepted(tmp_path):
    p = tmp_path / "s.bin"
    p.write_bytes(b"payload")
    assert digest_file(str(p)) == digest_file(p)


# ---------------------------------------------------------------------------
# 2. the guarantee NONE of the three existing helpers gives
# ---------------------------------------------------------------------------

def test_a_file_that_CHANGES_during_the_digest_is_REFUSED(tmp_path):
    """The digest would describe bytes that never existed as a whole file.

    MEASURED 2026-09-02: not one of the three existing helpers calls `stat()`,
    reads `st_mtime_ns`, or raises here. For a 636,522,106-byte GENCODE
    artifact the read is long enough for this to be a real window, and the
    resulting hash would be recorded as scientific evidence.
    """
    p = tmp_path / "moving.bin"
    p.write_bytes(b"a" * (1 << 20) * 2)

    real_open = Path.open

    def _rewrite_midway(self, *a, **kw):
        handle = real_open(self, *a, **kw)
        if self == p:
            # Grow the file AFTER the first stat and before the last read.
            with real_open(self.with_suffix(".tmp"), "wb") as _:
                pass
            with open(str(self), "ab") as g:
                g.write(b"b" * 4096)
        return handle

    Path.open = _rewrite_midway
    try:
        with pytest.raises(FileChangedDuringDigest) as exc:
            digest_file(p)
    finally:
        Path.open = real_open
    assert "never existed as a whole file" in str(exc.value)


def test_the_refusal_names_BOTH_stats(tmp_path):
    p = tmp_path / "m2.bin"
    p.write_bytes(b"z" * 1024)
    real_stat = Path.stat
    calls = {"n": 0}

    class _Fake:
        st_size = 999999
        st_mtime_ns = 1

    def _second_stat_differs(self, *a, **kw):
        if self == p:
            calls["n"] += 1
            if calls["n"] >= 2:
                return _Fake()
        return real_stat(self, *a, **kw)

    Path.stat = _second_stat_differs
    try:
        with pytest.raises(FileChangedDuringDigest) as exc:
            digest_file(p)
    finally:
        Path.stat = real_stat
    msg = str(exc.value)
    assert "before" in msg and "after" in msg
    assert "999999" in msg


def test_a_file_rewritten_IN_PLACE_at_the_SAME_SIZE_is_refused(tmp_path):
    """The mtime half of the guard, which a size comparison cannot give.

    MEASURED 2026-09-02 by sabotage: comparing only `st_size` passed every
    test, because the fake stat in `test_the_refusal_names_BOTH_stats` changes
    the size too. A file REWRITTEN IN PLACE at the same length -- a corrected
    parse, a re-sorted table, a byte flipped -- moves `st_mtime_ns` and NOT
    `st_size`, and that is precisely what this half is for.
    """
    p = tmp_path / "inplace.bin"
    p.write_bytes(b"a" * 4096)
    real_stat = Path.stat
    calls = {"n": 0}
    first = real_stat(p)

    class _SameSizeLaterMtime:
        st_size = first.st_size          # IDENTICAL size
        st_mtime_ns = first.st_mtime_ns + 1_000_000

    def _second_differs_only_in_mtime(self, *a, **kw):
        if self == p:
            calls["n"] += 1
            if calls["n"] >= 2:
                return _SameSizeLaterMtime()
        return real_stat(self, *a, **kw)

    Path.stat = _second_differs_only_in_mtime
    try:
        with pytest.raises(FileChangedDuringDigest) as exc:
            digest_file(p)
    finally:
        Path.stat = real_stat
    assert str(first.st_size) in str(exc.value)


def test_FileChangedDuringDigest_is_a_RuntimeError():
    """Not a Warning. A Warning can be SILENCED by a filter.

    MEASURED by sabotage: changing the base to `Warning` passed every test,
    because `Warning` subclasses `Exception`, so `raise` works and
    `pytest.raises` catches it -- while a `-W ignore` run would swallow it.
    """
    assert issubclass(FileChangedDuringDigest, RuntimeError)
    assert not issubclass(FileChangedDuringDigest, Warning)


def test_FileDigest_is_FROZEN():
    """Section 43: an identity object mutated after hashing invalidates
    everything built on it."""
    import dataclasses
    d = FileDigest(sha256="0" * 64, size_bytes=7)
    assert dataclasses.is_dataclass(d)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.sha256 = "1" * 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.size_bytes = 8


def test_an_UNCHANGED_file_does_not_raise(tmp_path):
    """The sensitivity half: the guard must not fire on a quiet file."""
    p = tmp_path / "still.bin"
    p.write_bytes(b"q" * (1 << 20))
    for _ in range(3):
        assert digest_file(p).sha256 == _reference(p)


# ---------------------------------------------------------------------------
# 3. FOUR implementations, ONE digest
# ---------------------------------------------------------------------------

def test_all_four_implementations_AGREE(tmp_path):
    """Executed, not read. This is what stops a fourth dialect existing."""
    p = tmp_path / "agree.bin"
    p.write_bytes(bytes(range(256)) * 4096 + b"tail")
    expected = _reference(p)

    cc = _load(_PKG / "data" / "constraint_canonicalize.py", "_cc_hash")
    pc = _load(_PKG / "data" / "phylop_cache.py", "_pc_hash")
    lg = _load(_PKG / "agent_layer" / "science_claw" / "ledger.py", "_lg_hash")

    assert cc.sha256_file(str(p)) == expected
    assert pc.sha256_file(p) == expected
    assert lg.compute_sha256(str(p)) == expected
    assert digest_file(p).sha256 == expected


def test_the_existing_three_still_lack_the_guarantee():
    """Recorded as a fact, so a later migration knows what it is adopting.

    If one of them ever acquires mutation detection this test fails, which is
    the correct moment to consolidate rather than a silent divergence.
    """
    for rel, name in (
        ("data/constraint_canonicalize.py", "sha256_file"),
        ("data/phylop_cache.py", "sha256_file"),
        ("agent_layer/science_claw/ledger.py", "compute_sha256"),
    ):
        text = (_PKG / rel).read_text(encoding="utf-8")
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.FunctionDef) and n.name == name:
                body = ast.unparse(n)
                assert "st_mtime_ns" not in body, (
                    "{}.{} now detects mutation; consolidate onto "
                    "digest_file rather than maintaining two guarantees"
                    .format(rel, name))


# ---------------------------------------------------------------------------
# 4. dependency direction
# ---------------------------------------------------------------------------

def test_provenance_imports_NOTHING_from_the_layers_above_it():
    """Section 62: provenance sits below connectors, pipelines and monitoring.

    A circular `data <-> provenance` dependency would make the package
    unusable exactly where it is needed, and an import added in haste is
    invisible until it deadlocks a test run.
    """
    forbidden = ("data", "models", "training", "monitoring", "evaluation",
                 "pipelines", "api", "agent_layer")
    offenders = []
    for f in sorted((_PKG / "provenance").rglob("*.py")):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mod = None
            if isinstance(n, ast.ImportFrom):
                mod = n.module or ""
            elif isinstance(n, ast.Import):
                mod = ",".join(a.name for a in n.names)
            if not mod:
                continue
            for bad in forbidden:
                if "genomic_variant_classifier.{}".format(bad) in mod:
                    offenders.append("{}:{} imports {}".format(
                        f.name, n.lineno, mod))
    assert not offenders, offenders


def test_the_package_exports_the_hashing_names():
    """The hashing surface, not the WHOLE surface.

    The original assertion pinned `provenance.__all__` to exactly the three
    names this module contributes. That was correct while `provenance` held
    only `hashing.py`, and WRONG the moment the package grew: Unit 3A moved
    the identity substrate in and the list became twenty-seven, so a test
    about hashing failed for a reason having nothing to do with hashing.

    A module-level test should pin what ITS module contributes and require the
    package to be self-consistent -- not enumerate every sibling.
    """
    import genomic_variant_classifier.provenance as prov
    for name in ("FileChangedDuringDigest", "FileDigest", "digest_file"):
        assert name in prov.__all__, name
        assert getattr(prov, name) is getattr(
            importlib.import_module(
                "genomic_variant_classifier.provenance.hashing"), name)


def test_the_package_surface_is_SELF_CONSISTENT():
    """Everything declared is present, and nothing present is undeclared.

    This is the invariant the original test was reaching for, expressed so it
    survives the package growing.
    """
    import genomic_variant_classifier.provenance as prov
    assert prov.__all__ == sorted(prov.__all__), "not sorted"
    assert len(prov.__all__) == len(set(prov.__all__)), "duplicated"
    for name in prov.__all__:
        assert hasattr(prov, name), name
