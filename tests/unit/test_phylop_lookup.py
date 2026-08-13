"""Tests for the columnar phyloP lookup -- A4 PHYLOPPERF-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.phylop_lookup import (
    PHYLOP_LOOKUP_SUBSTRATE_V2, DuplicateLocusInIndexError, FramePhyloPBackend,
    PhyloPLookupError, verify_vectorised_normalisation,
)


def _norm(chrom: str) -> str:
    c = str(chrom).strip()
    if c.upper().startswith("CHR"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c


def _backend(pairs=(("1", 100, 2.5), ("1", 101, -3.5), ("X", 500, 0.0))):
    return FramePhyloPBackend({(c, p): s for c, p, s in pairs}, normalise=_norm)


def _loci(rows, index=None):
    return pd.DataFrame(rows, columns=["chrom", "pos"], index=index)


# ---- the protocol A1 declared -------------------------------------------
def test_one_score_per_row_in_the_callers_index():
    b = _backend()
    loci = _loci([("1", 100), ("1", 101)], index=[7, 9])
    out = b.lookup_many(loci)
    assert isinstance(out, pd.Series)
    assert out.index.tolist() == [7, 9]
    assert out.iloc[0] == pytest.approx(2.5)
    assert out.iloc[1] == pytest.approx(-3.5)


def test_row_identity_is_preserved_BY_CONSTRUCTION():
    """A merge would return a new frame with a reset index and need a
    restoration step. reindex preserves the caller's index inherently, which is
    why A1's test_a_backend_that_loses_row_identity_is_refused exists."""
    b = _backend()
    loci = _loci([("1", 101), ("1", 100)], index=["a", "b"])
    out = b.lookup_many(loci)
    assert out.index.tolist() == ["a", "b"]
    assert out.loc["a"] == pytest.approx(-3.5)
    assert out.loc["b"] == pytest.approx(2.5)


def test_an_absent_locus_is_NaN_not_a_sentinel():
    b = _backend()
    out = b.lookup_many(_loci([("1", 999999)]))
    assert pd.isna(out.iloc[0])


def test_a_genuine_zero_survives_the_lookup():
    """phyloP is signed: 0.0 means neutral evolution, a real observation. A
    backend that conflated it with absence would recreate
    PHYLOP-SOURCE-OWNERSHIP-1 inside the lookup."""
    b = _backend()
    out = b.lookup_many(_loci([("X", 500)]))
    assert out.iloc[0] == 0.0
    assert not pd.isna(out.iloc[0])


def test_mixed_hits_and_misses_in_one_call():
    b = _backend()
    out = b.lookup_many(_loci([("1", 100), ("2", 100), ("1", 101)]))
    assert out.iloc[0] == pytest.approx(2.5)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(-3.5)


# ---- duplicate refusal, at CONSTRUCTION ----------------------------------
def test_a_duplicated_locus_cannot_ENTER_the_structure():
    """Stronger than detecting one. A dictionary resolved this by LAST ROW
    WINS; a MultiIndex refuses it when the structure is built, so an index
    carrying a duplicate cannot exist."""
    mi_source = [("1", 100, 2.5), ("1", 100, 9.9)]
    # A dict CANNOT hold the duplicate -- which is the point: the loss already
    # happened before the backend saw it. Build the MultiIndex directly to show
    # the structure refuses what a dict silently collapsed.
    idx = pd.MultiIndex.from_arrays(
        [np.array(["1", "1"], dtype=object), np.array([100, 100], dtype="int64")],
        names=["chrom", "pos"])
    assert idx.has_duplicates, "the fixture does not contain a duplicate"

    class _Leaky(dict):
        def keys(self): return [("1", 100), ("1", 100)]
        def values(self): return [2.5, 9.9]
        def __len__(self): return 2
        def __bool__(self): return True

    try:
        FramePhyloPBackend(_Leaky(), normalise=_norm)
    except DuplicateLocusInIndexError as exc:
        assert "1:100" in str(exc)
        return
    raise AssertionError("a duplicated locus entered the lookup structure")


def test_distinct_loci_build_normally():
    b = _backend()
    assert len(b) == 3


def test_the_same_position_on_different_chromosomes_is_not_a_duplicate():
    b = FramePhyloPBackend({("1", 100): 2.5, ("2", 100): 3.5}, normalise=_norm)
    assert len(b) == 2


# ---- normalisation, applied once and vectorised --------------------------
def test_chromosome_normalisation_is_applied_to_the_QUERY():
    b = _backend()
    out = b.lookup_many(_loci([("chr1", 100)]))
    assert out.iloc[0] == pytest.approx(2.5)


def test_normalisation_handles_the_sex_and_mitochondrial_forms():
    b = FramePhyloPBackend({("X", 1): 1.0, ("MT", 2): 2.0}, normalise=_norm)
    assert b.lookup_many(_loci([("chrX", 1)])).iloc[0] == pytest.approx(1.0)
    assert b.lookup_many(_loci([("chrM", 2)])).iloc[0] == pytest.approx(2.0)


# ---- refusals ------------------------------------------------------------
def test_missing_required_columns_RAISE():
    b = _backend()
    try:
        b.lookup_many(pd.DataFrame({"chrom": ["1"]}))
    except PhyloPLookupError as exc:
        assert "pos" in str(exc)
        return
    raise AssertionError("a locus frame without 'pos' was accepted")


def test_a_non_integer_position_RAISES_rather_than_being_skipped():
    """A lookup that silently skipped unparseable positions would return a
    shorter answer than the question, and the caller could not tell."""
    b = _backend()
    try:
        b.lookup_many(pd.DataFrame({"chrom": ["1"], "pos": ["not_a_number"]}))
    except PhyloPLookupError as exc:
        assert "not an integer" in str(exc)
        return
    raise AssertionError("an unparseable position was silently accepted")


def test_an_empty_index_yields_all_NaN_without_raising():
    """An empty source is a legitimate state -- stub mode, or a source with no
    overlap. Every answer is absent, and absent is NaN."""
    b = FramePhyloPBackend({}, normalise=_norm)
    assert len(b) == 0
    out = b.lookup_many(_loci([("1", 100), ("2", 200)]))
    assert bool(out.isna().all())
    assert len(out) == 2


def test_an_empty_locus_frame_returns_an_empty_result():
    b = _backend()
    out = b.lookup_many(pd.DataFrame({"chrom": [], "pos": []}))
    assert len(out) == 0


# ---- the substrate is DECLARED -------------------------------------------
def test_the_substrate_marker_records_the_engine():
    """A1 recorded 'legacy_dict_v1' so a transitional substrate stayed visible.
    This is its replacement, and it is equally visible."""
    assert PHYLOP_LOOKUP_SUBSTRATE_V2 == "columnar_series_v2"
    assert FramePhyloPBackend({}, normalise=_norm).substrate == "columnar_series_v2"


def test_the_backend_satisfies_the_A1_protocol_shape():
    """lookup_many(frame) -> Series is the contract A1 declared. A structural
    check, so a signature change is caught rather than discovered at a call
    site."""
    import inspect
    params = list(inspect.signature(FramePhyloPBackend.lookup_many).parameters)
    assert params == ["self", "loci"]


def test_results_agree_with_a_plain_dictionary_lookup():
    """THE EQUIVALENCE. A4 changes the ENGINE, not the semantics: for every
    query the columnar backend must return exactly what the dictionary
    returned, or this is a science change wearing a performance change's
    clothes."""
    pairs = {("1", 100): 2.5, ("1", 101): -3.5, ("X", 500): 0.0}
    b = FramePhyloPBackend(pairs, normalise=_norm)
    queries = [("1", 100), ("chr1", 101), ("X", 500), ("2", 7), ("chrX", 500)]
    out = b.lookup_many(_loci(queries))
    for i, (chrom, pos) in enumerate(queries):
        expected = pairs.get((_norm(str(chrom)), int(pos)), float("nan"))
        got = out.iloc[i]
        if pd.isna(expected):
            assert pd.isna(got), "{}:{} -> {} but the dict says absent".format(
                chrom, pos, got)
        else:
            assert got == pytest.approx(expected)


def test_the_vectorised_normaliser_agrees_with_the_per_row_authority():
    """THE EQUIVALENCE THAT MAKES THE OPTIMISATION ADMISSIBLE.

    MEASURED 2026-08-12: `.map(per_row)` was 53.5% of lookup_many's cost, so
    "vectorising the lookup" while leaving normalisation per-row moved the
    bottleneck rather than removing it. The string-accessor form is 3.8x
    faster -- and a fast path that quietly disagreed about which chromosome a
    variant is on would be a science defect wearing a performance change's
    clothes.

    Three chromosome-normalisation rules already exist in this repository. A
    fourth that could silently differ is worse than the cost it saves.
    """
    cases = pd.Series(["chr1", "CHR1", "Chr1", "1", "chrX", "X", "chrM", "M",
                       "MT", "chrY", "y", "22", "GL000009.2", " chr7 "])
    verify_vectorised_normalisation(cases, _norm)


def test_the_normaliser_equivalence_is_CHECKED_not_assumed():
    """The verifier must actually fail on a disagreement, or it proves nothing."""
    def _wrong(chrom):
        return "ALWAYS_WRONG"
    try:
        verify_vectorised_normalisation(pd.Series(["chr1"]), _wrong)
    except PhyloPLookupError as exc:
        assert "disagrees with the per-row authority" in str(exc)
        return
    raise AssertionError("the verifier accepted a normaliser that disagrees")


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
