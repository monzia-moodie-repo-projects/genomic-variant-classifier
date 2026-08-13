"""BigWig query integrity. A5: PHYLOPBACKEND-1 and PHYLOPSWALLOW-1.

TWO DEFECTS IN ONE METHOD, BOTH MEASURED 2026-08-12
===================================================

PHYLOPBACKEND-1 -- the two backends disagree, by import order
    vals = list(bw.values(chrom_bw, pos - 1, pos, fillna=0.0))
    if vals and vals[0] is not None and not math.isnan(vals[0]):
        return float(vals[0])

`fillna=0.0` asks the library to substitute 0.0 for NaN AT THE LIBRARY
BOUNDARY, so the `isnan` guard on the very next line can never fire on that
path. A position genuinely absent from the bigWig returns 0.0 and is
indistinguishable from a position measured as neutral.

phyloP is SIGNED. Zero means NEUTRAL EVOLUTION -- a real observation -- not
"unmeasured". This is PHYLOP-SOURCE-OWNERSHIP-1's defect injected one layer
below where A1 repaired it.

And the `pyBigWig` branch carries no `fillna` at all. So THE SAME QUERY RETURNS
0.0 OR NaN FOR AN UNMEASURED POSITION DEPENDING ON WHICH LIBRARY IMPORTED
FIRST -- a scientific difference decided by an import order nobody selected.

PHYLOPSWALLOW-1 -- every failure becomes a sentinel
    except Exception as exc:
        logger.debug("PhyloP BigWig query failed ...")
    return missing_value

A malformed file, a chromosome-naming mismatch, a corrupt block, a truncated
download -- all become the sentinel, logged at DEBUG, which is BELOW the
default threshold. Four million silent failures would produce a uniformly zero
column and no visible evidence anywhere.

It also makes `SOURCE_UNREADABLE` unrepresentable: the state exists in reality
and no code path can report it, because the exception never escapes.

THE DISTINCTION THIS MODULE DRAWS
=================================
    NOT COVERED   the position is outside the bigWig's data. NORMAL. -> None
    UNMEASURED    the interval exists but carries no value.      NORMAL. -> None
    UNREADABLE    the file, chromosome or block could not be read. FAULT. -> raise

The first two are answers. The third is an error, and conflating it with an
answer is what let a broken source look like a neutral genome.

WHAT IS AND IS NOT TESTED HERE
==============================
The libraries themselves are NOT installed in the environment this module was
written in, so the backend adapters are exercised against FAKE HANDLES that
reproduce each library's documented return contract -- pybigtools yielding an
iterable of float-or-None, pyBigWig returning a list or None. That verifies the
ADAPTER LOGIC and the parity property; it does not verify my reading of either
library's behaviour on a real asset.

That limit is stated rather than glossed: A4's regex defect passed nineteen
tests and ten sabotage mutations in an environment whose pandas differed from
the repository's, and was fatal in the one that matters. The equivalent risk
here is a library contract I have described but not exercised. The Run 17
preflight against the real 9.19-gibibyte asset is what closes it.

Author: Monzia Moodie
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class BigWigBackend(str, Enum):
    """Which library served a query. Recorded so parity is auditable.

    PHYLOPBACKEND-1 existed because nothing said which backend was in use, and
    the two behaved differently. A run artefact can now state it.
    """
    PYBIGTOOLS = "pybigtools"
    PYBIGWIG = "pyBigWig"
    NONE = "none"


class BigWigUnreadableError(RuntimeError):
    """The source could not be read. NOT the same as "no value here".

    Raised so `ConnectorAvailability.SOURCE_UNREADABLE` becomes reachable. It
    was previously unrepresentable: the exception was absorbed into a sentinel
    and logged below the default threshold, so a corrupt asset and a neutral
    genome produced identical output.
    """


def chrom_in_asset(handle, chrom: str) -> bool:
    """Does the asset carry this chromosome? Asked, not inferred.

    MEASURED 2026-08-12 -- an ABSENT chromosome makes both libraries RAISE, and
    they raise DIFFERENTLY:

        pybigtools : KeyError    'No chromomsome with name `chr1` found.'
        pyBigWig   : RuntimeError 'Invalid interval bounds!'

    Classifying either as a source fault is wrong: "this asset does not carry
    that chromosome" is an ANSWER. It is exactly what the chr-prefix retry
    exists to discover, and turning it into an exception made a naming
    convention look like a corrupt file.

    Both expose .chroms(), so the question can be ASKED. That also removes the
    need to parse exception TEXT, which differs by library and by version --
    the same reason identity comes from a digest and not a filename.
    """
    try:
        names = handle.chroms()
    except Exception as exc:                                   # noqa: BLE001
        raise BigWigUnreadableError(
            "could not list the chromosomes of this bigWig: {}: {}. A source "
            "that cannot say what it contains cannot be queried.".format(
                type(exc).__name__, exc)) from exc
    return chrom in names


def query_pybigtools(handle, chrom: str, pos: int) -> Optional[float]:
    """One position from a pybigtools handle. None means NO OBSERVATION.

    NO `fillna`. The previous code passed `fillna=0.0`, which substitutes at the
    LIBRARY BOUNDARY -- so the `isnan` guard on the following line could never
    fire, and an absent position became a measured zero. phyloP is signed; zero
    means neutral evolution.

    BigWig coordinates are zero-based half-open, so a one-based position `pos`
    is the interval [pos - 1, pos).
    """
    # "this asset does not carry that chromosome" is an ANSWER, not a
    # fault. Both libraries RAISE on it -- differently -- so it is ASKED
    # via .chroms() before the query rather than inferred from an
    # exception whose text differs by library and version.
    if not chrom_in_asset(handle, chrom):
        return None

    try:
        # fillna=None is pybigtools' way of saying "NaN for an uncovered
        # position, DELIBERATELY". Omitting the argument means "I have not
        # chosen", and the library emits a DeprecationWarning per call --
        # roughly 4.4 million on a full cohort, and under -W error it becomes
        # an exception this very handler would misreport as a SOURCE FAULT.
        # Measured 2026-08-12 with pybigtools 0.3.0.
        #
        # The same shape as the "Degrees of freedom <= 0" RuntimeWarning removed
        # from feature_selection.py: noise inside a defect-detection instrument
        # teaches a reader to ignore it.
        values = list(handle.values(chrom, pos - 1, pos, fillna=None))
    except TypeError:
        # An older pybigtools without the fillna parameter. Its DEFAULT was
        # fill-with-zero, which is the defect -- so this path must not be
        # silently taken.
        raise BigWigUnreadableError(
            "this pybigtools does not accept fillna=None, so its uncovered "
            "positions default to 0.0 and cannot be distinguished from a "
            "measured zero. Upgrade to 0.3.0 or later; PHYLOPBACKEND-1 is "
            "unrepairable on an older release.") from None
    except Exception as exc:                                   # noqa: BLE001
        raise BigWigUnreadableError(
            "pybigtools could not read {}:{}: {}: {}. This is a SOURCE FAULT, "
            "not an absent value -- absorbing it into a sentinel is what made a "
            "corrupt asset indistinguishable from a neutral genome.".format(
                chrom, pos, type(exc).__name__, exc)) from exc

    if not values:
        return None
    value = values[0]
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def query_pybigwig(handle, chrom: str, pos: int) -> Optional[float]:
    """One position from a pyBigWig handle. None means NO OBSERVATION.

    pyBigWig returns None for an out-of-range chromosome and a list otherwise.
    Its NaN is left as NaN and converted here, so both backends answer the same
    question the same way -- which is the whole of PHYLOPBACKEND-1.
    """
    # "this asset does not carry that chromosome" is an ANSWER, not a
    # fault. Both libraries RAISE on it -- differently -- so it is ASKED
    # via .chroms() before the query rather than inferred from an
    # exception whose text differs by library and version.
    if not chrom_in_asset(handle, chrom):
        return None

    try:
        values = handle.values(chrom, pos - 1, pos)
    except Exception as exc:                                   # noqa: BLE001
        raise BigWigUnreadableError(
            "pyBigWig could not read {}:{}: {}: {}. This is a SOURCE FAULT, "
            "not an absent value.".format(
                chrom, pos, type(exc).__name__, exc)) from exc

    if not values:
        return None
    value = values[0]
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


#: The adapters, by backend. Both must satisfy the SAME contract:
#: (handle, chrom, pos) -> float | None, raising only on a source fault.
BACKEND_QUERY = {
    BigWigBackend.PYBIGTOOLS: query_pybigtools,
    BigWigBackend.PYBIGWIG: query_pybigwig,
}


#: The module a real handle's type reports. MEASURED 2026-08-12 rather than
#: assumed: pybigtools 0.3.0 yields type `BBIReader` with __module__
#: 'pybigtools'; pyBigWig 0.3.25 yields `bigWigFile` with __module__ 'pyBigWig'.
_HANDLE_MODULE_TO_BACKEND = {
    "pybigtools": BigWigBackend.PYBIGTOOLS,
    "pyBigWig": BigWigBackend.PYBIGWIG,
}


def backend_of(handle) -> BigWigBackend:
    """Identify which library produced a handle, from the handle itself.

    PHYLOPBACKEND-1 existed partly because nothing recorded which backend was
    serving a query. Deriving it from the object removes the possibility of a
    caller declaring one backend and passing the other's handle -- a mismatch
    that would silently apply the wrong adapter's semantics.

    It refuses rather than guessing. A handle whose module is unrecognised may
    be a third library with a third convention for an uncovered position, which
    is exactly the class of difference this unit exists to eliminate.
    """
    module = type(handle).__module__.split(".")[0]
    try:
        return _HANDLE_MODULE_TO_BACKEND[module]
    except KeyError:
        raise BigWigUnreadableError(
            "cannot identify the bigWig backend for a handle of type {!r} from "
            "module {!r}. Declare it in _HANDLE_MODULE_TO_BACKEND with a "
            "measured adapter; an unrecognised library may report an uncovered "
            "position differently, which is PHYLOPBACKEND-1 itself.".format(
                type(handle).__name__, module)) from None


def query_bigwig(handle, chrom: str, pos: int,
                 backend: BigWigBackend) -> Optional[float]:
    """Dispatch to a backend adapter. No caller-supplied sentinel.

    The previous signature took `missing_value`, so a caller could decide that
    an unobserved conservation score means a specific biological value. That is
    the semantic hole CONSTRAINTFILL-1 closed for gnomAD constraint, where a
    missing loss-of-function observed/expected ratio was recorded as 1.0 --
    "completely tolerant". Absence is None here, and becomes NaN only at the
    tabular boundary where it cannot enter arithmetic as data.
    """
    if backend is BigWigBackend.NONE or handle is None:
        raise BigWigUnreadableError(
            "no bigWig backend is available; a query cannot be answered and "
            "must not be silently defaulted")
    try:
        adapter = BACKEND_QUERY[backend]
    except KeyError:
        raise BigWigUnreadableError(
            "no adapter for backend {!r}; declare one rather than falling back "
            "to a default".format(backend)) from None
    return adapter(handle, chrom, pos)


def assert_backend_parity(handle_a, handle_b, loci) -> None:
    """Refuse if the two backends answer the same question differently.

    PHYLOPBACKEND-1 was exactly this: `fillna=0.0` on one path and nothing on
    the other, so an unmeasured position returned 0.0 or NaN depending on which
    library imported first. A scientific difference decided by import order.

    This is a DIAGNOSTIC, run against a real asset when both libraries are
    installed. It is separate from production dispatch for the same reason
    measuring source agreement is separate from admitting it: a function that
    can both measure and resolve a disagreement will eventually resolve one it
    should have reported.
    """
    disagreements = []
    for chrom, pos in loci:
        a = query_pybigtools(handle_a, chrom, pos)
        b = query_pybigwig(handle_b, chrom, pos)
        same = (a is None and b is None) or (
            a is not None and b is not None and a == b)
        if not same:
            disagreements.append((chrom, pos, a, b))
    if disagreements:
        raise BigWigUnreadableError(
            "the two bigWig backends disagree at {} of {} locus/loci, e.g. {}. "
            "The same query must not depend on which library imported "
            "first.".format(len(disagreements), len(loci), disagreements[:5]))
    logger.info(
        "bigWig backend parity verified at %d locus/loci: pybigtools and "
        "pyBigWig agree, including on which positions are UNOBSERVED.",
        len(loci))
