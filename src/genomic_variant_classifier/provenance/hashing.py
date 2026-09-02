"""One raw-byte digest primitive, and a guarantee the others do not give.

Phase 1C Unit 2. Created 2026-09-02.

WHY A FOURTH IMPLEMENTATION IS NOT WHAT THIS IS
-----------------------------------------------
MEASURED 2026-09-01: three helpers hash a file, and all three produce the
identical digest, proven by EXECUTING them against
`configs/data_manifest.yaml` and comparing to an independently computed
reference:

    data/constraint_canonicalize.py:325     sha256_file(path)
    data/phylop_cache.py:158                sha256_file(path, *, chunk=1<<20)
    agent_layer/science_claw/ledger.py:70   compute_sha256(path)

Duplication, not disagreement. Adding a fourth that merely repeats them would
be the defect this project names elsewhere: a value stated independently of the
thing it describes.

WHAT THIS ADDS THAT NONE OF THE THREE HAS
-----------------------------------------
MEASURED 2026-09-02: not one of them calls `stat()`, reads `st_mtime_ns`, or
raises when the file changes underneath the read.

A digest computed over a file being rewritten describes BYTES THAT NEVER
EXISTED AS A WHOLE FILE. It is not merely stale -- it is an identity for
something that was never on disk. For a 636,522,106-byte GENCODE artifact the
read takes long enough for that to be a real window, and the resulting hash
would be recorded as scientific evidence.

`digest_file` stats before and after and refuses when either the size or the
modification time in nanoseconds has moved.

WHY NOT MIGRATE THE THREE HERE
------------------------------
MEASURED 2026-09-02: `sha256_file` has nine call sites across six files and
`compute_sha256` has eight across five -- SEVENTEEN in ELEVEN files, four of
them test files that pin the current names by identity. Rewriting those would
move test identities in modules this unit has no business touching.

The canonical helper exists here. Migration belongs to the units that own
those callers.

WHAT KEEPS THIS FROM DIVERGING
------------------------------
`tests/unit/test_provenance_hashing.py` executes all four implementations
against one file and asserts they agree. The three were proven identical by
running them; this one joins that proof, so divergence becomes a gate failure
rather than a later discovery.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

#: One mebibyte. The same block size all three existing helpers use, so the
#: read pattern is unchanged and only the guarantee is new.
_CHUNK = 1 << 20


class FileChangedDuringDigest(RuntimeError):
    """The file was modified while its digest was being computed.

    The digest that would have been returned describes bytes that never
    existed as a whole file, so it is refused rather than returned with a
    warning: a caller that receives a string has no way to tell it apart from
    a sound one.
    """


@dataclass(frozen=True)
class FileDigest:
    """A digest AND the size it was computed over.

    The size travels with the digest because `MaterializationIdentity` needs
    both, and re-stat-ing later to recover it would read a file that may since
    have changed -- reintroducing exactly the window this module closes.
    """

    sha256: str
    size_bytes: int


def digest_file(path: Path | str, *, chunk_size: int = _CHUNK) -> FileDigest:
    """Digest RAW BYTES, and refuse if the file moves underneath the read.

    Raw bytes, never a parsed-and-reserialised object: normalisation would
    silently change the identity of the artifact being recorded. That rule is
    `constraint_canonicalize.sha256_file`'s, stated in its own docstring, and
    it is carried here unchanged.

    Raises `FileChangedDuringDigest` when `st_size` or `st_mtime_ns` differs
    between the stat taken before the read and the stat taken after.
    """
    path = Path(path)
    before = path.stat()

    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)

    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise FileChangedDuringDigest(
            "{} changed while its SHA-256 was being computed: "
            "{} bytes at mtime {} before, {} bytes at mtime {} after. The "
            "digest would describe bytes that never existed as a whole file."
            .format(path, before.st_size, before.st_mtime_ns,
                    after.st_size, after.st_mtime_ns))

    return FileDigest(sha256=h.hexdigest(), size_bytes=after.st_size)
