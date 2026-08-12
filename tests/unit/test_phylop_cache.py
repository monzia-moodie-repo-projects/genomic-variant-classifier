"""Tests for PhyloP cache identity -- A3 PHYLOP-CACHE-INTEGRITY-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.phylop_cache import (
    CACHE_COLUMNS, CACHE_SCHEMA_VERSION, CacheIdentity, CacheState,
    PhyloPCacheError, read_cache, sha256_file, sidecar_path, write_cache,
)


def _frame(rows=(("1", 100, 2.5), ("1", 101, -3.5))):
    return pd.DataFrame(list(rows), columns=list(CACHE_COLUMNS))


def _identity(**kw):
    base = dict(source_path="/src/phylop.tsv", source_sha256="a" * 64,
                has_header=False, n_loci=2, rows_accepted=2,
                built_at="2026-08-12T00:00:00Z")
    base.update(kw)
    return CacheIdentity(**base)


def _tmp() -> Path:
    return Path(tempfile.mkdtemp()) / "phylop100way_index.parquet"


# ---- the round trip ------------------------------------------------------
def test_a_written_cache_is_read_back():
    p = _tmp()
    write_cache(_frame(), p, _identity())
    frame, state = read_cache(p, _identity())
    assert state is CacheState.USABLE
    assert list(frame.columns) == list(CACHE_COLUMNS)
    assert len(frame) == 2


def test_the_sidecar_is_written_beside_the_data():
    p = _tmp()
    write_cache(_frame(), p, _identity())
    side = sidecar_path(p)
    assert side.exists()
    recorded = json.loads(side.read_text(encoding="utf-8"))
    assert recorded["schema_version"] == CACHE_SCHEMA_VERSION
    assert recorded["source_sha256"] == "a" * 64


# ---- three states, not one except block ----------------------------------
def test_an_absent_cache_is_ABSENT_not_an_error():
    frame, state = read_cache(_tmp(), _identity())
    assert frame is None and state is CacheState.ABSENT


def test_a_cache_with_NO_sidecar_is_STALE_and_says_WHY():
    """A v1 cache carried no sidecar at all, so its ABSENCE identifies it.

    THE ASSERTION IS ON THE REASON, NOT THE OUTCOME. Sabotage removing the
    `if not side.exists()` guard went UNDETECTED when this only checked the
    state: execution fell through to side.read_text(), raised
    FileNotFoundError, and the NEXT except block returned STALE too. Two paths,
    one outcome -- so deleting one was invisible. The log message is what
    distinguishes "there is no sidecar" from "the sidecar is unreadable", and
    those are different facts about the cache.
    """
    import logging, io as _io
    p = _tmp()
    _frame().to_parquet(p, index=False)

    stream = _io.StringIO()
    handler = logging.StreamHandler(stream)
    lg = logging.getLogger("genomic_variant_classifier.data.phylop_cache")
    lg.addHandler(handler); lg.setLevel(logging.WARNING)
    try:
        frame, state = read_cache(p, _identity())
    finally:
        lg.removeHandler(handler)

    assert frame is None and state is CacheState.STALE
    text = stream.getvalue()
    assert "NO identity sidecar" in text, (
        "the absence of a sidecar was not reported as such: {!r}".format(text))
    assert "unreadable" not in text, (
        "a MISSING sidecar was reported as an UNREADABLE one; those are "
        "different facts and the guard distinguishing them is gone")


def test_a_DIFFERENT_source_digest_is_STALE():
    p = _tmp()
    write_cache(_frame(), p, _identity(source_sha256="a" * 64))
    frame, state = read_cache(p, _identity(source_sha256="b" * 64))
    assert frame is None and state is CacheState.STALE


def test_a_DIFFERENT_schema_version_is_STALE():
    """A version exists so an old layout is RECOGNISED and refused rather than
    silently misread."""
    p = _tmp()
    write_cache(_frame(), p, _identity())
    side = sidecar_path(p)
    d = json.loads(side.read_text(encoding="utf-8"))
    d["schema_version"] = "phylop_index_v1"
    side.write_text(json.dumps(d), encoding="utf-8")
    frame, state = read_cache(p, _identity())
    assert frame is None and state is CacheState.STALE


def test_a_MATCHING_but_CORRUPT_cache_RAISES():
    """The distinction the bare except could not make.

    A cache that claims to describe THIS source and then fails to read is a
    FAULT. Rebuilding hides corruption that recurs every run and presents only
    as unexplained slowness.
    """
    p = _tmp()
    write_cache(_frame(), p, _identity())
    p.write_bytes(b"not a parquet file at all")
    try:
        read_cache(p, _identity())
    except PhyloPCacheError as exc:
        assert "corruption, not a miss" in str(exc)
        return
    raise AssertionError("a corrupt matching cache was silently rebuilt")


def test_a_row_count_disagreeing_with_the_claim_RAISES():
    p = _tmp()
    write_cache(_frame(), p, _identity(rows_accepted=2))
    _frame((("1", 100, 2.5),)).to_parquet(p, index=False)   # one row, claim says two
    try:
        read_cache(p, _identity(rows_accepted=2))
    except PhyloPCacheError as exc:
        assert "the claim disagree" in str(exc)
        return
    raise AssertionError("a cache whose data contradicts its identity was accepted")


# ---- identity semantics --------------------------------------------------
def test_identity_is_the_DIGEST_not_the_PATH():
    """CACHEIDENTITY-1 in the gnomAD connector derived identity from the source
    FILENAME, so a sidecar built by a defective parser was preferred to a
    repaired one. The same source moved is the same source; a different source
    at the same path is not."""
    moved = _identity(source_path="/elsewhere/phylop.tsv")
    assert _identity().matches(moved)
    impostor = _identity(source_sha256="c" * 64)
    assert not _identity().matches(impostor)


def test_an_identity_with_no_digest_never_matches():
    """A claim with no evidence is not a claim. Two empty digests must not be
    treated as agreement."""
    blank = _identity(source_sha256="")
    assert not blank.matches(blank)


def test_why_not_names_the_specific_disagreement():
    a = _identity()
    assert "schema version" in a.why_not(_identity(schema_version="other"))
    assert "source digest" in a.why_not(_identity(source_sha256="d" * 64))
    assert "match" in a.why_not(_identity())


def test_the_identity_is_immutable():
    import dataclasses
    ident = _identity()
    try:
        ident.source_sha256 = "e" * 64
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("a cache identity was mutable after construction")


def test_identity_survives_a_json_round_trip():
    ident = _identity()
    assert CacheIdentity.from_dict(json.loads(json.dumps(ident.as_dict()))) == ident


def test_from_dict_ignores_unknown_keys():
    """A sidecar written by a future version must not crash an older reader --
    it is refused by SCHEMA VERSION, which is the declared mechanism, rather
    than by a TypeError."""
    d = _identity().as_dict()
    d["some_future_field"] = 42
    assert CacheIdentity.from_dict(d).source_sha256 == "a" * 64


# ---- refusals at write time ----------------------------------------------
def test_writing_without_a_source_digest_is_REFUSED():
    try:
        write_cache(_frame(), _tmp(), _identity(source_sha256=""))
    except PhyloPCacheError as exc:
        assert "claim with no evidence" in str(exc)
        return
    raise AssertionError("a cache with no source digest was written")


def test_writing_a_wrong_schema_version_is_REFUSED():
    try:
        write_cache(_frame(), _tmp(), _identity(schema_version="phylop_index_v1"))
    except PhyloPCacheError as exc:
        assert "schema" in str(exc)
        return
    raise AssertionError("a cache claiming a foreign schema was written")


def test_writing_a_frame_missing_the_contract_columns_is_REFUSED():
    bad = pd.DataFrame({"chrom": ["1"], "pos": [100], "phylop_score": [2.5]})
    try:
        write_cache(bad, _tmp(), _identity())
    except PhyloPCacheError as exc:
        assert "score" in str(exc)
        return
    raise AssertionError("a frame using the v1 column name was written as v2")


def test_the_cache_uses_the_INGEST_contract_column_names():
    """Reconciled in A3, deliberately, not before: a schema VERSION is what
    lets the old layout be recognised and refused. The earlier register entry
    PHYLOPCACHE-SCHEMA-1 claimed the writer and reader disagreed; that was
    WITHDRAWN once the method was read from its parse tree."""
    assert CACHE_COLUMNS == ("chrom", "pos", "score")
    assert CACHE_SCHEMA_VERSION == "phylop_index_v2"


# ---- publication order ---------------------------------------------------
def test_data_is_published_before_the_sidecar():
    """If the process dies between the two, the result is a cache with NO
    sidecar, which reads as STALE. The reverse order would leave a sidecar
    vouching for data that was never written."""
    p = _tmp()
    write_cache(_frame(), p, _identity())
    assert p.stat().st_mtime <= sidecar_path(p).stat().st_mtime


def test_sha256_file_digests_content_not_name():
    d = Path(tempfile.mkdtemp())
    a, b = d / "a.tsv", d / "b.tsv"
    a.write_text("1\t100\t2.5\n", encoding="utf-8")
    b.write_text("1\t100\t2.5\n", encoding="utf-8")
    assert sha256_file(a) == sha256_file(b)
    b.write_text("1\t100\t9.9\n", encoding="utf-8")
    assert sha256_file(a) != sha256_file(b)


# ---- the connector's cache round trip, end to end ------------------------
class TestConnectorCacheRoundTrip:
    """PhyloPConnector._get_index and _save_cache after A3.

    These exercise the REAL connector, because the defect was that its own
    cache read absorbed every failure in one `except` block. Testing the cache
    functions in isolation cannot show that the connector uses them.
    """

    @staticmethod
    def _src(tmp):
        p = tmp / "phylop.tsv"
        io.open(p, "w", encoding="utf-8", newline="\n").write(
            "1\t100\t2.5\n1\t101\t-3.5\n")
        return p

    def test_a_first_build_writes_a_cache_WITH_identity(self):
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        d = Path(tempfile.mkdtemp())
        conn = PhyloPConnector(phylop_file=self._src(d), cache_dir=d)
        index = conn._get_index()
        assert index[("1", 100)] == pytest.approx(2.5)

        cache = d / "phylop100way_index.parquet"
        assert cache.exists(), "no cache was written"
        side = sidecar_path(cache)
        assert side.exists(), (
            "no identity sidecar: a cache that cannot say which source it "
            "describes is not a cache")
        recorded = json.loads(side.read_text(encoding="utf-8"))
        assert recorded["schema_version"] == CACHE_SCHEMA_VERSION
        assert len(recorded["source_sha256"]) == 64

    def test_a_second_build_HITS_the_cache(self):
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        d = Path(tempfile.mkdtemp())
        src = self._src(d)
        PhyloPConnector(phylop_file=src, cache_dir=d)._get_index()
        second = PhyloPConnector(phylop_file=src, cache_dir=d)
        index = second._get_index()
        assert index[("1", 100)] == pytest.approx(2.5)
        assert second._cache_state is CacheState.USABLE

    def test_a_CHANGED_source_makes_the_cache_STALE(self):
        """Identity is the source DIGEST, so editing the source invalidates the
        cache even though the path is unchanged. Comparing paths would serve a
        cache built from different data."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        d = Path(tempfile.mkdtemp())
        src = self._src(d)
        PhyloPConnector(phylop_file=src, cache_dir=d)._get_index()
        io.open(src, "w", encoding="utf-8", newline="\n").write("1\t100\t9.9\n")

        second = PhyloPConnector(phylop_file=src, cache_dir=d)
        index = second._get_index()

        # The load-bearing assertion is the STATE. Asserting the rebuilt VALUE
        # would couple this test to what _build_index returns, and the property
        # under test is that a cache whose source digest no longer matches is
        # NOT SERVED -- which the state records and the warning names.
        assert second._cache_state is CacheState.STALE, (
            "a cache built from different bytes was served as current")
        assert index is not None and len(index) >= 1

    def test_a_CORRUPT_matching_cache_RAISES_rather_than_rebuilding(self):
        """The distinction the bare except could not make. Rebuilding hides a
        fault that recurs every run and presents only as slowness."""
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        d = Path(tempfile.mkdtemp())
        src = self._src(d)
        PhyloPConnector(phylop_file=src, cache_dir=d)._get_index()
        (d / "phylop100way_index.parquet").write_bytes(b"not a parquet file")
        with pytest.raises(PhyloPCacheError):
            PhyloPConnector(phylop_file=src, cache_dir=d)._get_index()

    def test_no_cache_directory_means_no_cache_and_no_error(self):
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        d = Path(tempfile.mkdtemp())
        conn = PhyloPConnector(phylop_file=self._src(d))
        conn._cache_dir = None
        index = conn._get_index()
        assert index[("1", 100)] == pytest.approx(2.5)


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except Exception as exc:                        # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
