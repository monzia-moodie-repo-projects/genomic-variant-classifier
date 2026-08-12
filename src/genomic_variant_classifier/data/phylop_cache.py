"""PhyloP index cache identity. A3: PHYLOP-CACHE-INTEGRITY-1.

WHAT THE CACHE COULD NOT SAY
============================
The parquet cache carried a filename and nothing else:

    phylop100way_index.parquet

No schema version. No source digest. No assembly, track or coordinate
convention. So a cache built from one source could be served for another and
nothing in the artefact would contradict it -- the same failure as
CACHEIDENTITY-1 in the gnomAD constraint connector, where a sidecar built by a
defective parser was preferred to a repaired one because identity was derived
from the source FILENAME alone.

A cache is a claim about a source. A claim with no evidence attached is not a
cache, it is a coincidence of file paths.

THREE STATES, ONE `except` BLOCK
=================================
The cache read at phylop.py:427 sat inside a bare try/except that absorbed
every failure and rebuilt silently. That collapses three different situations:

    ABSENT        no cache exists. Normal. Build one.
    STALE         a cache exists and describes a DIFFERENT source. Normal for a
                  source change; must be RECORDED, because a run that silently
                  rebuilt may be slow for a reason nobody can see.
    CORRUPT       a cache exists, claims to describe THIS source, and cannot be
                  read. That is a FAULT. Rebuilding hides it, and the same
                  corruption will recur every run while presenting only as
                  unexplained slowness.

The first two are outcomes. The third is an error, and this module refuses it.

THE COLUMN NAME, RECONCILED HERE AND NOT BEFORE
===============================================
_save_cache wrote `phylop_score`; the A2 ingest contract is `chrom`/`pos`/
`score`. Those had to agree, and the decision belongs with identity rather than
ahead of it: a schema VERSION is what lets an old layout be recognised and
refused instead of silently misread. CACHE_SCHEMA_VERSION is therefore pinned,
the cache stores the ingest contract's names, and a sidecar recording a
different version causes a rebuild rather than a wrong answer.

An earlier register entry, PHYLOPCACHE-SCHEMA-1, claimed _save_cache and
_parquet_to_index disagreed about this name. That was WITHDRAWN on 2026-08-12:
reading _parquet_to_index from its parse tree showed both used `phylop_score`
and the round-trip was consistent. The reconciliation here is a deliberate
change, not a repair of a mismatch that never existed.

Author: Monzia Moodie
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Bump when the cache LAYOUT changes. A sidecar recording a different version
#: causes a rebuild, never a reinterpretation: the point of a version is that an
#: old layout is RECOGNISED and refused rather than silently misread.
#:
#: v2 (2026-08-12) stores the A2 ingest contract's column names --
#: chrom / pos / score. v1 stored `phylop_score` and carried no sidecar at all,
#: so a v1 cache is indistinguishable from a v2 one by its data; the ABSENCE of
#: a sidecar is what identifies it.
CACHE_SCHEMA_VERSION: str = "phylop_index_v2"

#: The columns a v2 cache holds, in order. Identical to the ingest contract, so
#: a cache is a materialised ingest result and not a second format.
CACHE_COLUMNS: tuple = ("chrom", "pos", "score")

_SIDECAR_SUFFIX = ".identity.json"


class PhyloPCacheError(RuntimeError):
    """A cache claiming to describe this source could not be read.

    Distinct from a stale or absent cache, which are OUTCOMES rather than
    errors. This is raised only when a cache asserts it matches and then fails,
    because rebuilding would hide a fault that recurs on every run.
    """


class CacheState(str, Enum):
    """Why a cache was or was not used. Recorded, never inferred."""

    #: No cache file. Build one. Not a problem.
    ABSENT = "absent"
    #: A cache exists but describes a different source, or a different schema
    #: version. Rebuild -- and SAY SO, because an unexplained rebuild is a run
    #: that is slow for a reason nobody can see.
    STALE = "stale"
    #: Identity matches and the data loaded.
    USABLE = "usable"


@dataclass(frozen=True)
class CacheIdentity:
    """What a cache CLAIMS about the source it was built from.

    `source_sha256` is the digest of the SOURCE, not of the cache. A cache is a
    claim about a source; verifying it against the cache's own bytes would be
    circular.
    """
    schema_version: str = CACHE_SCHEMA_VERSION
    source_path: str = ""
    source_sha256: str = ""
    has_header: bool = False
    n_loci: int = 0
    rows_accepted: int = 0
    built_at: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheIdentity":
        known = {k: d[k] for k in cls().__dict__ if k in d}
        return cls(**known)

    def matches(self, other: "CacheIdentity") -> bool:
        """Identity is the schema version AND the source digest.

        The PATH is deliberately excluded: the same source moved to a different
        directory is the same source, and a different source at the same path is
        NOT. Comparing paths would get both cases backwards -- which is exactly
        how CACHEIDENTITY-1 arose in the gnomAD connector, where identity came
        from the filename.
        """
        return (self.schema_version == other.schema_version
                and bool(self.source_sha256)
                and self.source_sha256 == other.source_sha256)

    def why_not(self, other: "CacheIdentity") -> str:
        if self.schema_version != other.schema_version:
            return ("schema version {!r} != {!r}".format(
                self.schema_version, other.schema_version))
        if not self.source_sha256 or not other.source_sha256:
            return "one or both identities carry no source digest"
        if self.source_sha256 != other.source_sha256:
            return ("source digest {}... != {}...".format(
                self.source_sha256[:16], other.source_sha256[:16]))
        return "identities match"


def sidecar_path(cache_path) -> Path:
    return Path(str(cache_path) + _SIDECAR_SUFFIX)


def sha256_file(path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_cache(index_frame: pd.DataFrame, cache_path, identity: CacheIdentity) -> None:
    """Publish a cache and its identity ATOMICALLY-ish: data first, sidecar last.

    Order matters. If the process dies between the two, the result is a cache
    with NO sidecar -- which `read_cache` treats as STALE and rebuilds. The
    reverse order would leave a sidecar vouching for data that was never
    written, which is a claim without evidence.

    This mirrors the publication order the gnomAD constraint connector adopted
    after CACHEIDENTITY-2.
    """
    missing = [c for c in CACHE_COLUMNS if c not in index_frame.columns]
    if missing:
        raise PhyloPCacheError(
            "refusing to write a cache missing column(s) {}; a cache must hold "
            "the ingest contract {}".format(missing, list(CACHE_COLUMNS)))
    if identity.schema_version != CACHE_SCHEMA_VERSION:
        raise PhyloPCacheError(
            "refusing to write a cache whose identity claims schema {!r} while "
            "this code writes {!r}".format(
                identity.schema_version, CACHE_SCHEMA_VERSION))
    if not identity.source_sha256:
        raise PhyloPCacheError(
            "refusing to write a cache with no source digest; a cache is a "
            "CLAIM about a source, and a claim with no evidence is not a cache")

    cache_path = Path(cache_path)
    index_frame.loc[:, list(CACHE_COLUMNS)].to_parquet(cache_path, index=False)
    sidecar_path(cache_path).write_text(
        json.dumps(identity.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8")
    logger.info(
        "PhyloP cache written: %s (%d locus/loci, schema %s, source %s...)",
        cache_path, identity.n_loci, identity.schema_version,
        identity.source_sha256[:16])


def read_cache(cache_path, expected: CacheIdentity):
    """Return (frame, state). RAISE only when a MATCHING cache cannot be read.

    ABSENT and STALE are outcomes and return (None, state). A cache that claims
    to match and then fails to load is a FAULT and raises, because rebuilding
    would hide corruption that recurs every run and presents only as slowness.
    """
    cache_path = Path(cache_path)
    side = sidecar_path(cache_path)

    if not cache_path.exists():
        return None, CacheState.ABSENT
    if not side.exists():
        logger.warning(
            "PhyloP cache %s has NO identity sidecar; treating as stale and "
            "rebuilding. A cache that cannot say what source it describes "
            "cannot be trusted to describe this one.", cache_path)
        return None, CacheState.STALE

    try:
        recorded = CacheIdentity.from_dict(json.loads(side.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            "PhyloP cache identity %s is unreadable (%s); treating as stale "
            "and rebuilding.", side, exc)
        return None, CacheState.STALE

    if not recorded.matches(expected):
        logger.warning(
            "PhyloP cache %s describes a DIFFERENT source and will be rebuilt: "
            "%s. This is not an error, but an unexplained rebuild is a run "
            "that is slow for a reason nobody can see.",
            cache_path, recorded.why_not(expected))
        return None, CacheState.STALE

    # From here the cache CLAIMS to match. A failure now is a fault, not a miss.
    try:
        frame = pd.read_parquet(cache_path, columns=list(CACHE_COLUMNS))
    except Exception as exc:                                  # noqa: BLE001
        raise PhyloPCacheError(
            "cache {} has a MATCHING identity (schema {}, source {}...) but "
            "could not be read: {}: {}. This is corruption, not a miss -- "
            "rebuilding would hide a fault that recurs on every run and "
            "presents only as unexplained slowness. Delete the cache and its "
            "sidecar deliberately if that is what you intend.".format(
                cache_path, recorded.schema_version,
                recorded.source_sha256[:16], type(exc).__name__, exc)) from exc

    if len(frame) != recorded.rows_accepted:
        raise PhyloPCacheError(
            "cache {} holds {} row(s) but its identity claims {}. The data and "
            "the claim disagree; the cache is not what it says it is.".format(
                cache_path, len(frame), recorded.rows_accepted))

    logger.info(
        "PhyloP cache HIT: %s (%d row(s), schema %s, source %s...)",
        cache_path, len(frame), recorded.schema_version,
        recorded.source_sha256[:16])
    return frame, CacheState.USABLE
