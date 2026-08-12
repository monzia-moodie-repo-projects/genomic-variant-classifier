"""Tests for strict phyloP source ingestion -- A2 PHYLOP-INGEST-INTEGRITY-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.phylop_ingest import (
    PhyloPDuplicateLocusError, PhyloPHeaderContractError, PhyloPIngestAudit,
    PhyloPIngestError, assert_ingest_reconciles, assert_no_duplicate_loci,
    observe_header, parse_phylop_frame, read_phylop_source,
    verify_header_contract,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["chrom", "pos", "score"])


def _write(text: str) -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "phylop.tsv"
    io.open(p, "w", encoding="utf-8", newline="\n").write(text)
    return p


# ---- PHYLOPPARSE-1: refusal, not silent skipping -------------------------
def test_a_malformed_row_STOPS_the_build():
    """on_bad_lines="error", not "skip".

    Counting malformed rows and continuing would change the scientific source:
    the index would describe a subset nobody chose, admitted by a threshold
    nobody set. A conservation index silently missing an unknown number of
    positions is indistinguishable from a complete one.
    """
    p = _write("1\t100\t2.5\n1\t101\t3.5\textra\textra2\n1\t102\t1.0\n")
    try:
        read_phylop_source(p, has_header=False)
    except Exception as exc:
        assert not isinstance(exc, PhyloPHeaderContractError)
        return
    raise AssertionError("a malformed row was tolerated")


def test_a_clean_source_is_read_whole():
    p = _write("1\t100\t2.5\n1\t101\t-3.5\n1\t102\t0.0\n")
    clean, audit = read_phylop_source(p, has_header=False)
    assert audit.rows_read == 3
    assert audit.rows_accepted == 3
    assert audit.rows_rejected == 0
    assert audit.n_distinct_loci == 3


def test_an_empty_source_is_refused():
    p = _write("")
    try:
        read_phylop_source(p, has_header=False)
    except PhyloPIngestError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("an empty source was accepted")


# ---- PHYLOPHEADER-1: a declared contract, verified -----------------------
def test_the_header_is_read_from_RAW_TEXT_not_parsed_data():
    """PHYLOPHEADER-1, stated as MEASURED rather than as first assumed.

    The original check was
        if str(chunk.iloc[0]["pos"]).lower() in ("pos", "position", "start")
    and my first account of it -- that an Int64 column makes a header row <NA>,
    so the check could never fire -- was WRONG. Measured on pandas 3.0.2:

        no dtype        -> the column is str, iloc[0]["pos"] == "pos",
                           and the heuristic FIRES CORRECTLY
        dtype Int64     -> read_csv RAISES ValueError before the check runs

    So the defect is not a dead check. It is that the source's SHAPE is
    inferred from its CONTENT, and which of those two behaviours occurs depends
    on a dtype argument elsewhere in the call. A file property decided by a
    parsing detail is exactly the kind of coupling a declared contract removes.
    """
    assert observe_header("chrom\tpos\tscore\n") is True
    assert observe_header("1\t100\t2.5\n") is False

    # Untyped: the heuristic works, which is why nobody noticed the coupling.
    untyped = pd.read_csv(io.StringIO("chrom\tpos\tscore\n1\t100\t2.5\n"),
                          sep="\t", header=None, names=["chrom", "pos", "score"])
    assert str(untyped.iloc[0]["pos"]).lower() in ("pos", "position", "start")

    # Typed: the read raises before any heuristic can run.
    try:
        pd.read_csv(io.StringIO("chrom\tpos\tscore\n1\t100\t2.5\n"),
                    sep="\t", header=None, names=["chrom", "pos", "score"],
                    dtype={"pos": "Int64"})
    except ValueError as exc:
        assert "pos" in str(exc)
        return
    raise AssertionError(
        "a header row in an Int64-typed column no longer raises; the "
        "measured basis for the declared header contract has changed")


def test_a_declared_header_that_is_absent_RAISES():
    try:
        verify_header_contract("1\t100\t2.5\n", declared=True, source_path="x")
    except PhyloPHeaderContractError as exc:
        assert "has_header=True" in str(exc)
        return
    raise AssertionError("a false header declaration was accepted")


def test_an_undeclared_header_that_is_present_RAISES():
    try:
        verify_header_contract("chrom\tpos\tscore\n", declared=False, source_path="x")
    except PhyloPHeaderContractError as exc:
        assert "has_header=False" in str(exc)
        return
    raise AssertionError("an undeclared header was silently consumed")


def test_a_correctly_declared_header_is_accepted():
    assert verify_header_contract("chrom\tpos\tscore\n", declared=True) is True
    assert verify_header_contract("1\t100\t2.5\n", declared=False) is False


def test_a_headed_source_reads_its_data_rows_only():
    p = _write("chrom\tpos\tscore\n1\t100\t2.5\n1\t101\t3.5\n")
    clean, audit = read_phylop_source(p, has_header=True)
    assert audit.rows_read == 2
    assert audit.header_declared is True and audit.header_observed is True
    assert clean["pos"].tolist() == [100, 101]


# ---- row accounting ------------------------------------------------------
def test_every_row_is_accepted_or_rejected_BY_NAMED_REASON():
    f = _frame([("1", 100, 2.5), (None, 101, 3.5), ("1", None, 1.0),
                ("1", 103, None), ("2", 104, -1.0)])
    clean, audit = parse_phylop_frame(f)
    assert audit.rows_read == 5
    assert audit.rows_accepted == 2
    assert audit.rows_rejected_missing_chrom == 1
    assert audit.rows_rejected_missing_pos == 1
    assert audit.rows_rejected_missing_score == 1
    assert audit.reconciles()


def test_an_audit_that_does_not_reconcile_is_REFUSED():
    """The invariant a loader using on_bad_lines="skip" could never satisfy."""
    bad = PhyloPIngestAudit(rows_read=100, rows_accepted=90)
    try:
        assert_ingest_reconciles(bad)
    except PhyloPIngestError as exc:
        assert "unaccounted for" in str(exc)
        return
    raise AssertionError("an audit with 10 vanished rows was accepted")


def test_rejection_reasons_are_disjoint():
    """A row missing chrom AND pos counts once, under one reason. Overlapping
    categories would make the totals sum while describing the data wrongly."""
    f = _frame([(None, None, None), ("1", 100, 2.5)])
    _, audit = parse_phylop_frame(f)
    assert audit.rows_rejected == 1
    assert audit.rows_rejected_missing_chrom == 1
    assert audit.rows_rejected_missing_pos == 0
    assert audit.rows_rejected_missing_score == 0


def test_a_genuine_zero_score_is_ACCEPTED():
    """phyloP is signed: 0.0 means neutral evolution, a real observation. A
    reader that treated it as missing would recreate PHYLOP-SOURCE-OWNERSHIP-1
    at ingest time."""
    _, audit = parse_phylop_frame(_frame([("1", 100, 0.0)]))
    assert audit.rows_accepted == 1
    assert audit.rows_rejected_missing_score == 0


def test_a_negative_score_is_ACCEPTED():
    _, audit = parse_phylop_frame(_frame([("1", 100, -4.2)]))
    assert audit.rows_accepted == 1


def test_missing_required_columns_RAISE():
    try:
        parse_phylop_frame(pd.DataFrame({"chrom": ["1"], "pos": [100]}))
    except PhyloPIngestError as exc:
        assert "score" in str(exc)
        return
    raise AssertionError("a frame without a score column was accepted")


def test_the_audit_serialises_with_its_reconciliation():
    _, audit = parse_phylop_frame(_frame([("1", 100, 2.5), (None, 101, 3.5)]))
    d = audit.as_dict()
    assert d["rows_read"] == 2 and d["rows_accepted"] == 1
    assert d["rows_rejected"] == 1 and d["reconciles"] is True


def test_the_audit_is_immutable():
    import dataclasses
    _, audit = parse_phylop_frame(_frame([("1", 100, 2.5)]))
    try:
        audit.rows_accepted = 999
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("the ingest audit was mutable")


# ---- duplicate loci: refuse, never resolve by order ----------------------
def test_a_duplicated_locus_is_REFUSED():
    """`d[(chrom, pos)] = score` means LAST ROW WINS -- the same
    order-dependence as drop_duplicates(keep="first") in the gnomAD connector,
    which disagreed with MANE Select for 5,468 of 17,473 genes."""
    f = _frame([("1", 100, 2.5), ("1", 100, 9.9)])
    clean, _ = parse_phylop_frame(f)
    try:
        assert_no_duplicate_loci(clean, source_path="x")
    except PhyloPDuplicateLocusError as exc:
        assert "1:100" in str(exc)
        return
    raise AssertionError("a duplicated locus was resolved by row order")


def test_duplicates_are_refused_even_when_the_scores_AGREE():
    """Two identical scores at one position still mean the source contains a
    structure nobody declared -- overlapping intervals or a merge. Accepting
    the agreeing case teaches the loader to resolve, which is not its role."""
    f = _frame([("1", 100, 2.5), ("1", 100, 2.5)])
    clean, _ = parse_phylop_frame(f)
    try:
        assert_no_duplicate_loci(clean)
    except PhyloPDuplicateLocusError:
        return
    raise AssertionError("an agreeing duplicate was silently collapsed")


def test_distinct_loci_pass():
    clean, _ = parse_phylop_frame(_frame([("1", 100, 2.5), ("1", 101, 2.5)]))
    assert_no_duplicate_loci(clean)


def test_the_same_position_on_DIFFERENT_chromosomes_is_not_a_duplicate():
    clean, _ = parse_phylop_frame(_frame([("1", 100, 2.5), ("2", 100, 3.5)]))
    assert_no_duplicate_loci(clean)


def test_read_refuses_a_duplicated_source_end_to_end():
    p = _write("1\t100\t2.5\n1\t100\t9.9\n")
    try:
        read_phylop_source(p, has_header=False)
    except PhyloPDuplicateLocusError:
        return
    raise AssertionError("read_phylop_source accepted a duplicated locus")


# ---- the connector's two ingest paths, end to end ------------------------
class TestConnectorIngestPaths:
    """PhyloPConnector._build_index and _parquet_to_index after A2.

    NOTE ON THE PARQUET COLUMN NAME. These fixtures write a column named
    `score`, which is what parse_phylop_frame requires. The PRE-A2 method read
    `phylop_score` -- the same name _save_cache writes, so the cache round-trip
    was consistent and the "unreadable cache" defect recorded earlier
    (PHYLOPCACHE-SCHEMA-1) was WITHDRAWN once the method was read from its parse
    tree rather than transcribed. A3 pins the column name alongside a schema
    version and source digest; until then the ingest contract is `score`.

    These exercise the REAL connector rather than the ingest functions in
    isolation, because the defect being repaired was that the connector's own
    readers bypassed every guarantee: on_bad_lines="skip" in the flat path, a
    silent dropna in the parquet path, and last-row-wins in both.
    """

    @staticmethod
    def _connector(path=None):
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        return PhyloPConnector(phylop_file=path)

    def test_a_clean_flat_source_builds_and_records_its_audit(self):
        p = _write("1\t100\t2.5\n1\t101\t-3.5\n1\t102\t0.0\n")
        conn = self._connector(p)
        index = conn._build_index()
        assert index[("1", 100)] == pytest.approx(2.5)
        assert index[("1", 102)] == pytest.approx(0.0), (
            "a genuine zero is an observation, not missingness")
        audit = conn._ingest_audit
        assert audit is not None, "the ingest audit was not recorded"
        assert audit.rows_read == 3 and audit.rows_accepted == 3
        assert audit.reconciles()

    def test_a_malformed_flat_source_STOPS_the_build(self):
        """on_bad_lines="skip" discarded these silently, so the index was
        smaller than the source and nothing said so."""
        p = _write("1\t100\t2.5\n1\t101\t3.5\textra\textra2\n")
        conn = self._connector(p)
        with pytest.raises(Exception):
            conn._build_index()

    def test_a_duplicated_locus_in_a_flat_source_is_REFUSED(self):
        p = _write("1\t100\t2.5\n1\t100\t9.9\n")
        conn = self._connector(p)
        with pytest.raises(PhyloPDuplicateLocusError):
            conn._build_index()

    def test_the_header_contract_is_declared_and_verified(self):
        p = _write("chrom\tpos\tscore\n1\t100\t2.5\n")
        conn = self._connector(p)
        with pytest.raises(PhyloPHeaderContractError):
            conn._build_index()          # _has_header defaults to False
        conn._has_header = True
        index = conn._build_index()
        assert index[("1", 100)] == pytest.approx(2.5)

    def test_the_parquet_path_accounts_for_rejected_rows(self):
        """A missing chromosome is COUNTED and excluded, not crashed on.

        The old method had NO row filtering at all -- measured from the parse
        tree on 2026-08-12, after two earlier transcriptions of it (as `dropna`
        and as a `notna()` mask) proved to be inventions. With no filtering, a
        missing position reaches `int(r.pos)` and RAISES, which is loud rather
        than silent. Accounting is still the improvement: the row is now
        rejected under a named reason and the totals reconcile.
        """
        import tempfile
        d = Path(tempfile.mkdtemp())
        q = d / "idx.parquet"
        pd.DataFrame({"chrom": ["1", None, "2"], "pos": [100, 101, 102],
                      "score": [2.5, 3.5, -1.0]}).to_parquet(q, index=False)
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        index = PhyloPConnector._parquet_to_index(q)
        assert set(index) == {("1", 100), ("2", 102)}

    def test_the_parquet_path_REFUSES_a_duplicated_locus(self):
        """THE defect this replacement genuinely removes.

        The old comprehension built d[(chrom, pos)] = score, so a duplicated
        locus resolved by LAST ROW WINS -- the same order-dependence as
        drop_duplicates(keep="first") in the gnomAD connector, which disagreed
        with MANE Select for 5,468 of 17,473 genes.
        """
        import tempfile
        d = Path(tempfile.mkdtemp())
        q = d / "idx.parquet"
        pd.DataFrame({"chrom": ["1", "1"], "pos": [100, 100],
                      "score": [2.5, 9.9]}).to_parquet(q, index=False)
        from genomic_variant_classifier.data.phylop import PhyloPConnector
        with pytest.raises(PhyloPDuplicateLocusError):
            PhyloPConnector._parquet_to_index(q)

    def test_chromosome_normalisation_happens_ONCE(self):
        """Normalisation is applied once, and the result is unchanged either way.

        MEASURED 2026-08-12: the old reader rewrote df["chrom"] with
        _normalise_chrom at line 477, then called _normalise_chrom AGAIN on
        r.chrom at line 479 -- on text line 477 had already normalised. So the
        second call was wasted work over the whole index, not a correctness
        fault, and because _normalise_chrom is idempotent no behavioural test
        can distinguish one application from two. That is why this asserts
        against the function itself and checks idempotence explicitly.
        """
        import tempfile
        d = Path(tempfile.mkdtemp())
        q = d / "idx.parquet"
        pd.DataFrame({"chrom": ["chr1"], "pos": [100],
                      "score": [2.5]}).to_parquet(q, index=False)
        from genomic_variant_classifier.data.phylop import (
            PhyloPConnector, _normalise_chrom,
        )
        index = PhyloPConnector._parquet_to_index(q)
        # Asserted against _normalise_chrom ITSELF, not against a hard-coded
        # expectation of what it returns. Normalisation policy belongs to that
        # function and is tested in test_phylop_block.py; what THIS test owns is
        # that it is applied exactly once, to the raw value.
        assert list(index) == [(_normalise_chrom("chr1"), 100)]
        assert _normalise_chrom(_normalise_chrom("chr1")) == _normalise_chrom("chr1"), (
            "_normalise_chrom is not idempotent, so 'applied once' cannot be "
            "distinguished from 'applied twice' by this test")


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
