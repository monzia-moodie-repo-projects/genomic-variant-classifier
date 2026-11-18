"""Stage 1D-1c.1C set "av" revision c: the two integrity primitives the qualified preparation and the reference-status
authority share (owner ruling 6fa240c0; audit AUDIT_OF_RULING_6fa240c0_2026-11-17.md).

* SOURCE BYTES AT CONSUMPTION. ``materialization.Parsed`` is a public dataclass: its ``sha256`` field is an assertion
  until the bytes it carries are hashed again. ``require_verified_bytes`` re-hashes ``Parsed.value`` where it is consumed
  and refuses unless the measured digest and size equal both the object's own fields and the independently declared digest.
  It does not read files (the reader is ``materialization.read_verified_bytes``, unchanged).
* A TWO-LEVEL CONTENT-ADDRESSED STORE. A LOGICAL identity (what the artifact means: its receipt) can have more than one
  valid PHYSICAL payload -- the canonical matrix's af_log10 bits depend on the numerical environment
  (CANONICAL-IDENTITY-ENVIRONMENT-1), and any payload's bytes depend on its serialiser's version. So a payload is stored
  under ``<directory>/<logical>/<sha256 of its bytes>.parquet`` with its record ``<sha256>.receipt.json`` beside it: two
  valid payloads of one logical identity never collide, and a name is never reused for other bytes.

  Publication is EXCLUSIVE and ATOMIC: the bytes go to a uniquely named temporary file in the same folder (``mkstemp``),
  are flushed to disk, and are published by ``os.link`` -- which creates the name only if it does not exist, on POSIX and
  on Windows (NTFS) alike, so a concurrent writer can neither replace a published file nor see a partial one. An existing
  file is accepted only when its bytes are identical. The payload is published first and the record last, so a record's
  presence means its payload was complete; a record whose payload is missing or differs is REFUSED (an incomplete or
  corrupt store is reported, never repaired by a writer). A file system without hard links is refused loudly, never
  served by a non-exclusive fallback.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from genomic_variant_classifier.information_sources.census import HEX64
from genomic_variant_classifier.information_sources.materialization import Parsed

logger = logging.getLogger(__name__)

__all__ = ["PAYLOAD_SUFFIX", "RECEIPT_SUFFIX", "payload_entry", "read_stored", "require_verified_bytes", "store",
           "stored_payloads"]

PAYLOAD_SUFFIX = ".parquet"
RECEIPT_SUFFIX = ".receipt.json"


def require_verified_bytes(parsed: Any, *, what: str, declared_sha256: str | None = None) -> str:
    """The SHA-256 of ``parsed.value`` measured now; refused unless it and the byte count equal ``parsed``'s own fields
    and, when given, ``declared_sha256`` (the independently declared digest)."""
    if type(parsed) is not Parsed or type(parsed.value) is not bytes:
        raise ValueError(f"{what}: a source must be the Parsed result of read_verified_bytes (bytes)")
    measured = hashlib.sha256(parsed.value).hexdigest()
    if measured != parsed.sha256 or len(parsed.value) != parsed.n_bytes:
        raise ValueError(f"{what}: the bytes supplied have SHA-256 {measured} and {len(parsed.value)} bytes, not the "
                         f"{parsed.sha256} and {parsed.n_bytes} their wrapper states; refused")
    if declared_sha256 is not None and measured != declared_sha256:
        raise ValueError(f"{what}: the bytes supplied have SHA-256 {measured}, not the declared {declared_sha256}; "
                         "an undeclared source is refused, never substituted")
    return measured


def _check_id(value: str, what: str) -> None:
    if type(value) is not str or not HEX64.match(value):
        raise ValueError(f"{what} is a lower-case SHA-256")


def _publish(path: Path, data: bytes) -> bool:
    """Create ``path`` holding exactly ``data``, exclusively and atomically; True when this call created it, False when
    it already held exactly these bytes. Refused when it holds other bytes."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".partial")       # short: Windows path budget
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            pass
        except OSError as exc:
            raise OSError(f"{path.parent}: exclusive publication needs hard links ({exc}); a non-exclusive fallback is "
                          "refused") from exc
    finally:
        os.unlink(tmp)
    if path.read_bytes() != data:
        raise FileExistsError(f"{path} exists with different content; a content-addressed file is never replaced")
    return False


def payload_entry(payload: bytes) -> dict[str, Any]:
    """What a record states about its payload: the file name, its SHA-256 and its size."""
    sha = hashlib.sha256(payload).hexdigest()
    return {"file": f"{sha}{PAYLOAD_SUFFIX}", "sha256": sha, "bytes": len(payload)}


def _verify_payload(folder: Path, entry: Mapping[str, Any], what: str) -> bytes:
    path = folder / f"{entry['sha256']}{PAYLOAD_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"{what}: the record exists but its payload {path.name} does not; the store is "
                                "incomplete and is refused, never repaired by a writer")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"{what}: the payload {path.name} is not the one its record names; the store is corrupt and "
                         "is refused")
    return raw


def store(directory: Path, logical: str, payload: bytes, record: bytes, *, what: str) -> str:
    """Store ``payload`` and its ``record`` under ``logical``; returns the payload's SHA-256. ``record`` must state
    ``payload_entry(payload)`` under its key "payload" (the caller builds it). Idempotent for identical bytes; an existing
    record is accepted only with its payload present and intact."""
    _check_id(logical, f"{what}: the logical identity")
    entry = payload_entry(payload)
    if json.loads(record.decode("ascii")).get("payload") != entry:
        raise ValueError(f"{what}: the record does not state its payload's name, SHA-256 and size")
    folder = Path(directory) / logical
    folder.mkdir(parents=True, exist_ok=True)
    rpath = folder / f"{entry['sha256']}{RECEIPT_SUFFIX}"
    if rpath.exists():
        if rpath.read_bytes() != record:
            raise FileExistsError(f"{rpath} exists with different content; a content-addressed record is never replaced")
        _verify_payload(folder, entry, what)
        logger.info("%s %s/%s already stored in %s", what, logical, entry["sha256"], directory)
        return entry["sha256"]
    _publish(folder / entry["file"], payload)            # the payload first ...
    _publish(rpath, record)                              # ... the record last: its presence means a complete payload
    return entry["sha256"]


def read_stored(directory: Path, logical: str, payload_sha256: str, *, what: str) -> tuple[bytes, bytes]:
    """(record bytes, payload bytes) of one stored payload, the payload verified against the record's entry and its
    own name. The caller verifies everything the record states about meaning."""
    _check_id(logical, f"{what}: the logical identity")
    _check_id(payload_sha256, f"{what}: the payload identity")
    folder = Path(directory) / logical
    record = (folder / f"{payload_sha256}{RECEIPT_SUFFIX}").read_bytes()
    entry = json.loads(record.decode("ascii")).get("payload")
    if not isinstance(entry, Mapping) or entry.get("sha256") != payload_sha256 or \
            entry.get("file") != f"{payload_sha256}{PAYLOAD_SUFFIX}" or type(entry.get("bytes")) is not int:
        raise ValueError(f"{what}: the record does not name the payload it is stored beside")
    return record, _verify_payload(folder, entry, what)


def stored_payloads(directory: Path, logical: str) -> tuple[str, ...]:
    """The payload identities with a record under ``logical``, sorted (temporary files and other names are ignored)."""
    _check_id(logical, "the logical identity")
    folder = Path(directory) / logical
    if not folder.is_dir():
        return ()
    names = (p.name[:-len(RECEIPT_SUFFIX)] for p in folder.iterdir() if p.name.endswith(RECEIPT_SUFFIX))
    return tuple(sorted(n for n in names if HEX64.match(n)))
