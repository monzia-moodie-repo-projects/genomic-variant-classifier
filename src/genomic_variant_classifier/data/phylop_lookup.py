"""Columnar phyloP lookup. A4: PHYLOPPERF-1.

WHAT A1 DEFERRED, AND WHY IT COULD BE DEFERRED
==============================================
A1 introduced `PhyloPLookupBackend` and declared `DictPhyloPBackend`
transitional, with the substrate recorded in a constant so transitional
architecture could not quietly become permanent:

    PHYLOP_LOOKUP_SUBSTRATE = "legacy_dict_v1"

Its `lookup_many` is a Python-level loop:

    values = [
        self._index.get((self._normalise(str(chrom)), int(pos)), float("nan"))
        for chrom, pos in zip(loci["chrom"], loci["pos"])
    ]

That is one interpreter dispatch per row -- a `str()`, an `int()`, a function
call and a dictionary probe, 4.4 million times per annotation pass. The gnomAD
constraint connector carried the same shape in four `.map(lambda ...)` passes,
removed in commit 7161132, and the measurement there was roughly 17.6 million
interpreter calls on this cohort.

WHY A SERIES WITH A MULTIINDEX AND NOT A MERGE
==============================================
A relational merge would also vectorise, and `validate="many_to_one"` would make
LAST-ROW-WINS structurally impossible rather than merely refused -- which is the
stronger property, and A2 currently achieves the weaker one by checking.

But a merge returns a NEW FRAME with a reset index, and the connector's contract
is that a lookup preserves row identity: A1's
`test_a_backend_that_loses_row_identity_is_refused` exists precisely because a
fast backend that reorders rows would be fast and wrong.

`Series.reindex` on a MultiIndex vectorises the lookup, preserves the caller's
index by construction, and yields NaN for a missing locus -- which is the
declared absence semantics rather than a sentinel. Uniqueness is enforced ONCE
at construction via `MultiIndex.has_duplicates`, so last-row-wins is refused at
the point the structure is built rather than at every query.

WHAT THIS DOES NOT CHANGE
=========================
The connector's `_index` remains a dictionary. That is deliberate: A1's, A2's
and A3's tests assert on it as a mapping -- `index[("1", 100)]`,
`set(index) == {...}` -- and a representation change would reopen scientific
contracts that have nothing to do with lookup speed. The SEMANTICS are frozen;
only the engine moves, which is what A1's abstraction was for.

Construction still loops. `_build_index` and `_parquet_to_index` materialise
their dictionaries row by row, and vectorising those is a separate measurable
change against a different code path. Doing both here would make a performance
regression unattributable to either.

Author: Monzia Moodie
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Recorded on the backend so a run artefact can state which engine served it.
#: A1 used "legacy_dict_v1"; the marker is how a transitional substrate is kept
#: visible rather than becoming permanent by inertia.
PHYLOP_LOOKUP_SUBSTRATE_V2: str = "columnar_series_v2"


class PhyloPLookupError(RuntimeError):
    """The lookup structure could not be built, or a query violated its contract."""


def _normalise_chrom_vectorised(chrom_series, per_row):
    """Vectorised equivalent of the per-row chromosome normaliser.

    MEASURED: 3.8x faster than `.map(per_row)` on 200,000 rows, which was the
    single largest cost in `lookup_many`.

    It is an OPTIMISATION of `per_row`, never a second definition of chromosome
    identity. Three normalisation rules already exist in this repository
    (phylop.py, constraint_canonicalize.py, the gnomAD connector) and a fourth
    that could silently disagree would be worse than the cost it saves --
    `verify_vectorised_normalisation` is what keeps them one rule.
    """
    s = chrom_series.astype("string").str.strip()
    # `case=False`, NOT an inline `(?i)` flag. MEASURED 2026-08-12:
    #     re.compile(r"^(?i)chr")
    #     re.error: global flags not at the start of the expression at position 1
    # on Python 3.11+, where inline global flags away from position 0 became an
    # error. pandas 3.0.2 SWALLOWS that and returns the right answer; pandas
    # 2.3.3 -- which this repository uses -- propagates it. So the defect was
    # invisible in the environment this module was built in and fatal in the one
    # it runs in: 13 tests here and 11 connector tests, one cause.
    #
    # case=False is pandas' own parameter, so the flag never enters the pattern.
    s = s.str.replace(r"^chr", "", regex=True, case=False)

    # `if c.upper() == "M": c = "MT"` -- case-INSENSITIVE, matching the
    # authority.
    s = s.where(~s.str.upper().eq("M"), "MT")

    # `return c.upper() if c in ("X", "Y", "MT") else c` -- case-SENSITIVE on
    # the already-stripped value. Lowercase "y" fails that membership test and
    # is returned UNCHANGED.
    #
    # My first vectorised form compared the UPPERCASED value against the same
    # list, so "y" matched and became "Y". verify_vectorised_normalisation
    # caught it on its first run: ('y', 'Y', 'y'). That is a one-character
    # difference in chromosome identity, and it is exactly the class of silent
    # divergence a fourth normalisation rule would introduce.
    s = s.where(~s.isin(["X", "Y", "MT"]), s.str.upper())
    return s.astype(object)


def verify_vectorised_normalisation(chrom_series, per_row) -> None:
    """Refuse if the vectorised form disagrees with the authority.

    Called by the tests on every fixture. A fast path that quietly differs
    about which chromosome a variant is on would be a science defect wearing a
    performance change's clothes.
    """
    fast = _normalise_chrom_vectorised(chrom_series, per_row)
    slow = chrom_series.astype(str).map(per_row)
    mismatched = fast.astype(object) != slow.astype(object)
    if bool(mismatched.any()):
        n = int(mismatched.sum())
        sample = list(zip(chrom_series[mismatched].head(5),
                          fast[mismatched].head(5), slow[mismatched].head(5)))
        raise PhyloPLookupError(
            "the vectorised chromosome normaliser disagrees with the per-row "
            "authority on {} value(s): {}".format(n, sample))


class DuplicateLocusInIndexError(PhyloPLookupError):
    """The index contains one genomic position more than once.

    Separate because it is a SCIENTIFIC question, not a performance fault. A
    dictionary silently resolved this by LAST ROW WINS, making the index depend
    on insertion order -- the same order-dependence as
    `drop_duplicates(keep="first")` in the gnomAD constraint connector, which
    disagreed with MANE Select for 5,468 of 17,473 genes.

    A2 refuses duplicates by CHECKING at ingest. Here the condition is refused
    at CONSTRUCTION, so an index carrying one cannot exist -- which is stronger
    than detecting one that does.
    """


class FramePhyloPBackend:
    """Vectorised lookup over a Series with a (chrom, pos) MultiIndex.

    Satisfies the PhyloPLookupBackend protocol A1 declared: one score per input
    row, index preserved, NaN where there is no observation, and a raise on any
    failure to query rather than a sentinel.
    """

    substrate = PHYLOP_LOOKUP_SUBSTRATE_V2

    def __init__(self, index, *, normalise) -> None:
        """Build the lookup structure ONCE, and refuse a duplicated locus.

        `index` is the connector's mapping of (chrom, pos) -> score. It is
        consumed here and converted; the connector keeps its own dictionary,
        because A1's, A2's and A3's tests assert on that mapping and this unit
        changes the ENGINE, not the contract.
        """
        self._normalise = normalise
        if not index:
            self._series = pd.Series(
                dtype="float64",
                index=pd.MultiIndex.from_arrays(
                    [np.array([], dtype=object), np.array([], dtype="int64")],
                    names=["chrom", "pos"]),
            )
            self._n = 0
            return

        keys = list(index.keys())
        chroms = np.array([k[0] for k in keys], dtype=object)
        positions = np.array([k[1] for k in keys], dtype="int64")
        scores = np.fromiter(index.values(), dtype="float64", count=len(keys))

        mi = pd.MultiIndex.from_arrays([chroms, positions], names=["chrom", "pos"])
        if mi.has_duplicates:
            dup = mi[mi.duplicated(keep=False)].unique()
            raise DuplicateLocusInIndexError(
                "the phyloP index contains {} duplicated locus/loci, e.g. {}. A "
                "dictionary resolved this by LAST ROW WINS, making the index "
                "depend on insertion order; this structure refuses it at "
                "construction so such an index cannot exist.".format(
                    len(dup), [":".join(map(str, t)) for t in dup[:5]]))

        self._series = pd.Series(scores, index=mi, name="phylop")
        self._n = len(self._series)

    def __len__(self) -> int:
        return self._n

    def lookup_many(self, loci: pd.DataFrame) -> pd.Series:
        """One score per row of `loci`, in the caller's own index.

        `reindex` is the whole point: it is vectorised, it preserves the
        caller's index BY CONSTRUCTION rather than by a subsequent restoration
        step, and it yields NaN for an absent key -- the declared absence
        semantics rather than a sentinel a caller could mistake for a
        measurement.
        """
        absent = [c for c in ("chrom", "pos") if c not in loci.columns]
        if absent:
            raise PhyloPLookupError(
                "locus frame is missing required column(s): {}".format(absent))

        # MEASURED 2026-08-12, 200,000 rows: `.map(self._normalise)` was 53.5%
        # of this method's cost -- more than the reindex it was supposed to be
        # supporting. `.map` is a Python call per row, so "vectorising the
        # lookup" while leaving normalisation per-row moved the bottleneck
        # rather than removing it. The string-accessor form measured 3.8x
        # faster and agreed exactly with the per-row function.
        #
        # The per-row function remains the AUTHORITY: this is an optimisation of
        # it, not a reimplementation, and `verify_vectorised_normalisation`
        # asserts they agree so a divergence is a test failure rather than a
        # silent difference in chromosome identity.
        chroms = _normalise_chrom_vectorised(loci["chrom"], self._normalise)
        positions = pd.to_numeric(loci["pos"], errors="coerce")
        if bool(positions.isna().any()):
            n_bad = int(positions.isna().sum())
            raise PhyloPLookupError(
                "{} locus/loci carry a position that is not an integer; a "
                "lookup cannot silently skip them".format(n_bad))

        query = pd.MultiIndex.from_arrays(
            [chroms.to_numpy(dtype=object),
             positions.to_numpy(dtype="int64")],
            names=["chrom", "pos"])
        values = self._series.reindex(query).to_numpy(dtype="float64")
        return pd.Series(values, index=loci.index, dtype="float64")
