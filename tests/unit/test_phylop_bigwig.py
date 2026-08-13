"""Tests for bigWig query integrity -- A5 PHYLOPBACKEND-1, PHYLOPSWALLOW-1.

TWO LAYERS, DELIBERATELY.

FAKE HANDLES reproduce each library's return contract and run everywhere,
including where neither library is installed. They exercise the adapter logic,
the refusal paths and the parity property.

A REAL BIGWIG, written with pyBigWig and queried with BOTH libraries, verifies
that those contracts are what the libraries actually do. Those tests skip when
either library is absent, so the suite still runs in a minimal environment --
but where the libraries exist, the claim is MEASURED rather than described.

An earlier draft of this file stated that only a Run 17 preflight against the
9.19-gibibyte asset could close that gap. That was an assumption, not a
measurement: both libraries install from PyPI, pyBigWig can WRITE a bigWig, and
a 659-byte synthetic asset with known values and known gaps settles it in
milliseconds. Measured 2026-08-12 -- pybigtools 0.3.0, pyBigWig 0.3.25:

    position        pybigtools      pybigtools      pyBigWig
                    (no fillna)     fillna=0.0
    1:100  (2.5)        2.5             2.5            2.5
    1:500  (0.0)        0.0             0.0            0.0     <- a GENUINE zero
    1:900  (gap)        nan             0.0            nan     <- THE DEFECT

Under fillna=0.0 a gap and a measured zero are INDISTINGUISHABLE, and pyBigWig
never had that behaviour -- so the two backends disagreed by import order.

pybigtools 0.3.0 also emits:

    DeprecationWarning: The default behavior of values() has changed: empty
    bins / uncovered positions are now returned as NaN instead of filled with
    0. Pass fillna=0 to keep the previous behavior.

The library CORRECTED this default. The repository's code passed fillna=0.0
explicitly, restoring the defective behaviour against the library's own repair.
Recorded as PHYLOPFILLNA-DEPRECATED-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import math
import sys
from enum import Enum

import pytest

from genomic_variant_classifier.data.phylop_bigwig import (
    BACKEND_QUERY, BigWigBackend, BigWigUnreadableError, assert_backend_parity,
    backend_of, query_bigwig, query_pybigtools, query_pybigwig,
)


class _PyBigTools:
    """pybigtools 0.3.0: `values(chrom, start, end, fillna=...)`.

    THE SIGNATURE INCLUDES fillna, AND THAT MATTERS. An earlier version of this
    stub omitted it, so it accepted a call the real library would reject -- and
    ten tests passed against a fake MORE PERMISSIVE than the thing it stands
    for. A stub that tolerates what the library refuses is worse than no stub,
    because it converts a real incompatibility into a green suite.

    Measured: pybigtools 0.3.0 returns NaN for an uncovered position when
    fillna is None, and 0.0 when fillna=0.0 -- which is PHYLOPBACKEND-1.
    """

    def __init__(self, table=None, raises=None):
        self._table = table or {}
        self._raises = raises

    def chroms(self):
        return {c: 100000 for c, _ in self._table}

    def values(self, chrom, start, end, fillna="__unset__"):
        if self._raises is not None:
            raise self._raises
        if chrom not in self.chroms():
            # MEASURED: pybigtools raises KeyError for an absent chromosome.
            # An earlier stub returned NaN, so sabotage S13 -- disabling the
            # chrom_in_asset guard -- went undetected: the fake was gentler
            # than the library it stands for.
            raise KeyError(
                "No chromomsome with name `{}` found.".format(chrom))
        if fillna == "__unset__":
            raise AssertionError(
                "the adapter called values() without fillna; the real library "
                "emits a DeprecationWarning per call on that path")
        key = (chrom, start + 1)
        if key not in self._table:
            return iter([float("nan") if fillna is None else fillna])
        return iter([self._table[key]])


class _PyBigWig:
    """pyBigWig: `values(chrom, start, end)` returns a list, or None."""

    def __init__(self, table=None, raises=None, none_for_missing=False):
        self._table = table or {}
        self._raises = raises
        self._none = none_for_missing

    def chroms(self):
        return {c: 100000 for c, _ in self._table}

    def values(self, chrom, start, end):
        if self._raises is not None:
            raise self._raises
        key = (chrom, start + 1)
        if key not in self._table:
            return None if self._none else [float("nan")]
        return [self._table[key]]


TABLE = {("1", 100): 2.5, ("1", 101): -3.5, ("X", 500): 0.0}


# ---- PHYLOPBACKEND-1: the two backends must AGREE --------------------------
def test_an_unmeasured_position_is_None_on_BOTH_backends():
    """THE DEFECT. `fillna=0.0` made pybigtools return 0.0 for an absent
    position while pyBigWig returned NaN -- so the same query gave a different
    scientific answer depending on which library imported first."""
    a = query_pybigtools(_PyBigTools(TABLE), "1", 999999)
    b = query_pybigwig(_PyBigWig(TABLE), "1", 999999)
    assert a is None and b is None


def test_a_measured_position_agrees_on_BOTH_backends():
    assert query_pybigtools(_PyBigTools(TABLE), "1", 100) == pytest.approx(2.5)
    assert query_pybigwig(_PyBigWig(TABLE), "1", 100) == pytest.approx(2.5)


def test_a_GENUINE_ZERO_is_returned_not_treated_as_absence():
    """phyloP is signed: 0.0 means NEUTRAL EVOLUTION, a real observation. The
    old code could not tell it from an absent position, which is
    PHYLOP-SOURCE-OWNERSHIP-1 injected one layer below where A1 repaired it."""
    for got in (query_pybigtools(_PyBigTools(TABLE), "X", 500),
                query_pybigwig(_PyBigWig(TABLE), "X", 500)):
        assert got == 0.0
        assert got is not None


def test_a_negative_score_survives_both_backends():
    assert query_pybigtools(_PyBigTools(TABLE), "1", 101) == pytest.approx(-3.5)
    assert query_pybigwig(_PyBigWig(TABLE), "1", 101) == pytest.approx(-3.5)


def test_an_ABSENT_chromosome_is_absence_on_both_backends():
    """MEASURED 2026-08-12: an absent chromosome makes BOTH libraries raise,
    and differently -- pybigtools KeyError, pyBigWig RuntimeError. An earlier
    version of this test asserted pyBigWig returns None, which no real version
    does; the stub was wrong and the test passed against it.

    "This asset does not carry that chromosome" is an ANSWER. It is what the
    chr-prefix retry exists to discover, and treating it as a fault made a
    naming convention look like a corrupt file."""
    assert query_pybigtools(_PyBigTools(TABLE), "ZZ", 1) is None
    assert query_pybigwig(_PyBigWig(TABLE), "ZZ", 1) is None


def test_the_backends_agree_across_a_mixed_battery():
    loci = [("1", 100), ("1", 101), ("X", 500), ("1", 999999), ("2", 5)]
    assert_backend_parity(_PyBigTools(TABLE), _PyBigWig(TABLE), loci)


def test_parity_FAILS_when_one_backend_fabricates_a_zero():
    """The exact shape of PHYLOPBACKEND-1, as a test: one backend substituting
    0.0 at the library boundary while the other reports absence."""

    class _Fabricating(_PyBigTools):
        def values(self, chrom, start, end, fillna=None):
            key = (chrom, start + 1)
            return iter([self._table.get(key, 0.0)])      # the fillna=0.0 defect

    try:
        assert_backend_parity(_Fabricating(TABLE), _PyBigWig(TABLE),
                              [("1", 999999)])
    except BigWigUnreadableError as exc:
        assert "disagree" in str(exc)
        assert "imported" in str(exc)
        return
    raise AssertionError("a fabricated zero passed the parity check")


# ---- PHYLOPSWALLOW-1: a read failure is a FAULT ---------------------------
def test_a_read_failure_RAISES_on_pybigtools():
    """`except Exception: ... return missing_value`, logged at DEBUG -- below
    the default threshold. Four million silent failures would have produced a
    uniformly zero column with no visible evidence.

    THE STUB CARRIES THE CHROMOSOME DELIBERATELY. chrom_in_asset runs BEFORE the
    read, so a stub with an empty table would return None at the guard and this
    test would prove only that an absent chromosome is absence -- not that a
    read failure raises. Found by reading the failure rather than assuming the
    stubs were simply missing a method."""
    try:
        query_pybigtools(
            _PyBigTools(TABLE, raises=OSError("corrupt block")), "1", 100)
    except BigWigUnreadableError as exc:
        assert "SOURCE FAULT" in str(exc)
        assert "corrupt block" in str(exc)
        return
    raise AssertionError("a read failure was absorbed into a sentinel")


def test_a_read_failure_RAISES_on_pyBigWig():
    try:
        query_pybigwig(
            _PyBigWig(TABLE, raises=RuntimeError("bad header")), "1", 100)
    except BigWigUnreadableError as exc:
        assert "SOURCE FAULT" in str(exc)
        return
    raise AssertionError("a read failure was absorbed into a sentinel")


def test_the_original_exception_is_CHAINED_not_discarded():
    """The cause must survive. Reporting that a read failed without saying why
    replaces one silent failure with a louder uninformative one."""
    cause = OSError("truncated download")
    try:
        query_pybigtools(_PyBigTools(TABLE, raises=cause), "1", 100)
    except BigWigUnreadableError as exc:
        assert exc.__cause__ is cause
        return
    raise AssertionError("no exception was raised")


def test_a_missing_handle_RAISES_rather_than_defaulting():
    try:
        query_bigwig(None, "1", 100, BigWigBackend.PYBIGTOOLS)
    except BigWigUnreadableError as exc:
        assert "must not be silently defaulted" in str(exc)
        return
    raise AssertionError("a query with no handle returned a default")


def test_backend_NONE_RAISES():
    try:
        query_bigwig(_PyBigTools(TABLE), "1", 100, BigWigBackend.NONE)
    except BigWigUnreadableError:
        return
    raise AssertionError("a query with no backend was answered")


# ---- the public dispatch ---------------------------------------------------
def test_dispatch_routes_to_the_declared_backend():
    assert query_bigwig(_PyBigTools(TABLE), "1", 100,
                        BigWigBackend.PYBIGTOOLS) == pytest.approx(2.5)
    assert query_bigwig(_PyBigWig(TABLE), "1", 100,
                        BigWigBackend.PYBIGWIG) == pytest.approx(2.5)


def test_there_is_no_caller_supplied_sentinel():
    """A caller could previously pass missing_value=0.0 and thereby decide that
    an unobserved conservation score means a specific biological value -- the
    semantic hole CONSTRAINTFILL-1 closed for gnomAD constraint."""
    import inspect
    for fn in (query_bigwig, query_pybigtools, query_pybigwig):
        params = list(inspect.signature(fn).parameters)
        assert "missing_value" not in params, (
            "{} still accepts a caller sentinel: {}".format(fn.__name__, params))


def test_every_declared_backend_has_an_adapter():
    declared = {b for b in BigWigBackend if b is not BigWigBackend.NONE}
    assert declared == set(BACKEND_QUERY), (
        "backend enum and adapter table disagree: {}".format(
            declared.symmetric_difference(set(BACKEND_QUERY))))


def test_an_OLDER_pybigtools_without_fillna_is_REFUSED():
    """The branch a downgrade would take, tested through the door it enters by.

    pybigtools gained `fillna` in 0.3.0. BEFORE that release its default was
    FILL-WITH-ZERO -- which is PHYLOPBACKEND-1 itself, not a variation of it. On
    such a release an uncovered position is indistinguishable from a measured
    zero and the defect is UNREPAIRABLE, so the adapter must refuse rather than
    quietly produce conservation scores that conflate the two.

    Sabotage W5 -- replacing that refusal with `return None` -- went UNDETECTED,
    because the installed library accepts fillna and no test drove the older
    path. An unreachable guard is still a guard.
    """
    class _OldPyBigTools:
        """pybigtools < 0.3.0: values() has no fillna parameter."""

        def chroms(self):
            return {"1": 100000}

        def values(self, chrom, start, end):
            return iter([0.0])

    try:
        query_pybigtools(_OldPyBigTools(), "1", 100)
    except BigWigUnreadableError as exc:
        assert "does not accept fillna=None" in str(exc)
        assert "unrepairable" in str(exc).lower()
        return
    raise AssertionError(
        "an older pybigtools, whose default conflates a gap with a measured "
        "zero, was used to answer a query")


def test_a_backend_with_NO_adapter_RAISES_rather_than_defaulting():
    """The unreachable branch, reached.

    Sabotage W10 -- replacing the KeyError refusal with `return None` -- went
    UNDETECTED, because every declared backend currently has an adapter and no
    test drove the fallback. That is precisely the branch a FUTURE backend would
    take: added to the enum, adapter forgotten, and every query silently
    answering "no observation" instead of refusing.

    The registry pattern this repository already uses says a missing entry must
    refuse rather than default -- policy_for in model_input_view.py, and
    ESTIMATOR_INPUT_POLICIES. An unreachable guard is still a guard, and it must
    be tested through the door it will actually be entered by.
    """
    class _FutureBackend(str, Enum):
        DEEPTOOLS = "deeptools"

    try:
        query_bigwig(_PyBigTools(TABLE), "1", 100, _FutureBackend.DEEPTOOLS)
    except BigWigUnreadableError as exc:
        assert "no adapter for backend" in str(exc)
        assert "rather than falling back" in str(exc)
        return
    raise AssertionError(
        "a backend with no adapter was answered with a default instead of "
        "being refused")


def test_both_adapters_share_one_signature():
    """Parity is only meaningful if both answer the same question the same way."""
    import inspect
    sigs = {fn.__name__: list(inspect.signature(fn).parameters)
            for fn in BACKEND_QUERY.values()}
    assert len(set(map(tuple, sigs.values()))) == 1, sigs


def test_coordinates_are_zero_based_half_open():
    """A one-based position `pos` is the interval [pos - 1, pos). An off-by-one
    here would shift every conservation score by one base -- silently, since
    neighbouring positions carry plausible values."""
    seen = {}

    class _Recording(_PyBigTools):
        def chroms(self):
            return {"1": 100000}

        def values(self, chrom, start, end, fillna=None):
            seen["span"] = (start, end)
            seen["fillna"] = fillna
            return iter([1.0])

    query_pybigtools(_Recording(), "1", 100)
    assert seen["span"] == (99, 100)
    assert seen["fillna"] is None, (
        "the adapter must pass fillna=None -- omitting it warns per call, and "
        "fillna=0.0 is PHYLOPBACKEND-1 itself")


# ---- the same properties, against a REAL bigWig ---------------------------
_HAVE_BOTH = True
try:
    import pybigtools as _pbt
    import pyBigWig as _pbw
except ImportError:                                            # pragma: no cover
    _HAVE_BOTH = False

_needs_libs = pytest.mark.skipif(
    not _HAVE_BOTH,
    reason="pybigtools and pyBigWig are both required for real-asset parity")


def _real_bigwig(tmp):
    """A 659-byte asset with known values AND known gaps.

    1:100 = 2.5, 1:101 = -3.5, 1:500 = 0.0 (a GENUINE zero), everything else
    UNCOVERED. The gap is the whole point: it is what fillna=0.0 turned into a
    measurement.
    """
    import pyBigWig
    path = str(tmp / "real.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("1", 1000)])
    bw.addEntries(["1", "1", "1"], [99, 100, 499], ends=[100, 101, 500],
                  values=[2.5, -3.5, 0.0])
    bw.close()
    return path


class TestAgainstARealBigWig:
    """Everything above, measured rather than described."""

    @_needs_libs
    def test_both_libraries_agree_on_a_real_asset(self, tmp_path):
        import pybigtools, pyBigWig
        path = _real_bigwig(tmp_path)
        loci = [("1", 100), ("1", 101), ("1", 500), ("1", 900), ("1", 1)]
        assert_backend_parity(pybigtools.open(path), pyBigWig.open(path), loci)

    @_needs_libs
    def test_a_real_GAP_is_absence_on_both_libraries(self, tmp_path):
        import pybigtools, pyBigWig
        path = _real_bigwig(tmp_path)
        assert query_pybigtools(pybigtools.open(path), "1", 900) is None
        assert query_pybigwig(pyBigWig.open(path), "1", 900) is None

    @_needs_libs
    def test_a_real_GENUINE_ZERO_is_an_observation_on_both(self, tmp_path):
        """1:500 carries a measured 0.0. If either adapter returned None here,
        it would be discarding a real phyloP observation of neutral evolution."""
        import pybigtools, pyBigWig
        path = _real_bigwig(tmp_path)
        assert query_pybigtools(pybigtools.open(path), "1", 500) == 0.0
        assert query_pybigwig(pyBigWig.open(path), "1", 500) == 0.0

    @_needs_libs
    def test_fillna_zero_MEASURABLY_conflates_a_gap_with_a_zero(self, tmp_path):
        """THE DEFECT, demonstrated against the real library.

        This is what the repository's code did. It is not a reconstruction.
        """
        import warnings

        import pybigtools
        path = _real_bigwig(tmp_path)
        # The DeprecationWarning is the LIBRARY telling us this default was
        # wrong -- PHYLOPFILLNA-DEPRECATED-1. This test provokes it on purpose,
        # so it is suppressed HERE rather than left to add ten warnings to a
        # suite the project keeps at 33. Suppressing it anywhere else would
        # hide the library's own correction.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gap_plain = list(pybigtools.open(path).values("1", 899, 900))[0]
            gap_fill = list(
                pybigtools.open(path).values("1", 899, 900, fillna=0.0))[0]
            real_zero = list(pybigtools.open(path).values("1", 499, 500))[0]

        assert math.isnan(gap_plain), "the library no longer reports a gap as NaN"
        assert gap_fill == 0.0
        assert real_zero == 0.0
        assert gap_fill == real_zero, (
            "under fillna=0.0 a gap and a measured zero must be shown to be "
            "indistinguishable -- that is the defect being recorded")

    @_needs_libs
    def test_the_two_libraries_return_different_TYPES_and_both_normalise(self, tmp_path):
        """pybigtools yields numpy.float64, pyBigWig a Python float. Both
        adapters call float(), so they normalise -- verified rather than
        assumed."""
        import pybigtools, pyBigWig
        path = _real_bigwig(tmp_path)
        a = query_pybigtools(pybigtools.open(path), "1", 100)
        b = query_pybigwig(pyBigWig.open(path), "1", 100)
        assert type(a) is float and type(b) is float
        assert a == b == pytest.approx(2.5)

    @_needs_libs
    def test_a_TRUNCATED_asset_RAISES_rather_than_returning_a_sentinel(self, tmp_path):
        """PHYLOPSWALLOW-1 against a really corrupt file, not a mock."""
        import pybigtools
        path = _real_bigwig(tmp_path)
        data = open(path, "rb").read()
        broken = str(tmp_path / "broken.bw")
        open(broken, "wb").write(data[: len(data) // 2])
        try:
            handle = pybigtools.open(broken)
            query_pybigtools(handle, "1", 100)
        except BigWigUnreadableError:
            return
        except Exception:
            # The library refused to OPEN it, which is equally loud and equally
            # not a silent sentinel. Either outcome satisfies the property.
            return
        raise AssertionError("a truncated bigWig produced a value")


def test_a_chroms_failure_is_a_SOURCE_FAULT():
    """A source that cannot say what it contains cannot be queried.

    Sabotage S14 -- returning True when chroms() fails -- went undetected
    because no test drove that path. Swallowing it would mean querying an asset
    whose contents are unknown and treating every miss as absence.
    """
    class _NoChroms:
        def chroms(self):
            raise OSError("index block unreadable")

        def values(self, chrom, start, end, fillna=None):
            return iter([1.0])

    try:
        query_pybigtools(_NoChroms(), "1", 100)
    except BigWigUnreadableError as exc:
        assert "could not list the chromosomes" in str(exc)
        return
    raise AssertionError("a bigWig with an unreadable index answered a query")


def test_an_unrecognised_handle_is_REFUSED_not_guessed():
    """A third library may report an uncovered position a third way -- which is
    PHYLOPBACKEND-1 itself. Refuse rather than apply an adapter's semantics to
    a handle it was not written for."""
    class _SomeOtherLibrary:
        def values(self, chrom, start, end, fillna=None):
            return iter([0.0])

    try:
        backend_of(_SomeOtherLibrary())
    except BigWigUnreadableError as exc:
        assert "cannot identify the bigWig backend" in str(exc)
        return
    raise AssertionError("an unrecognised handle was assigned a backend")


class TestBackendIdentification:
    """backend_of, against REAL handles."""

    @_needs_libs
    def test_a_real_pybigtools_handle_is_identified(self, tmp_path):
        import pybigtools
        assert backend_of(pybigtools.open(_real_bigwig(tmp_path))) is (
            BigWigBackend.PYBIGTOOLS)

    @_needs_libs
    def test_a_real_pyBigWig_handle_is_identified(self, tmp_path):
        import pyBigWig
        assert backend_of(pyBigWig.open(_real_bigwig(tmp_path))) is (
            BigWigBackend.PYBIGWIG)

    @_needs_libs
    def test_dispatch_via_backend_of_agrees_with_the_direct_adapters(self, tmp_path):
        """Identification must not change the answer -- only remove the chance
        of a caller declaring one backend and passing the other's handle."""
        import pybigtools, pyBigWig
        path = _real_bigwig(tmp_path)
        for opener, direct in ((pybigtools.open, query_pybigtools),
                               (pyBigWig.open, query_pybigwig)):
            for pos in (100, 500, 900):
                h1, h2 = opener(path), opener(path)
                via = query_bigwig(h1, "1", pos, backend_of(h1))
                straight = direct(h2, "1", pos)
                assert (via is None and straight is None) or via == straight


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
