"""
src/genomic_variant_classifier/data/phylop.py
==================
PhyloP evolutionary conservation score connector.
Phase 2, Pillar 1, Connector 5.

PhyloP (Phylogenetic P-values) measures conservation at individual genomic
positions by comparing observed versus expected substitution rates under a
neutral model across 100 vertebrate genomes (phyloP100way, GRCh38).

    Score > 0   conserved      (slower evolution than expected)
    Score = 0   neutral
    Score < 0   accelerated    (faster evolution than expected)
    Range:       approximately −30 to +30

Data source — manual download required (~9 GB BigWig, or pre-extracted TSV):
    BigWig:  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/
             hg38.phyloP100way.bw
    Pre-extracted TSV (recommended for this pipeline):
             Produce with bigWigToWig or extract via pyBigWig/pybigtools.

Two lookup modes
----------------
BigWig mode (preferred):
    Requires pyBigWig (pip install pyBigWig) or pybigtools.
    Set phylop_file to the .bw path.  Scores are fetched per-position at
    query time with no pre-loading — low memory, slower for large batches.

TSV / Parquet mode (fast bulk):
    Set phylop_file to a tab-delimited file with columns:
        chrom  pos  phylop_score
    (no header or with header — auto-detected by checking first token).
    On first use the connector builds an in-memory dict index and writes a
    parquet cache for fast subsequent loads.

Stub mode:
    phylop_file=None → every lookup returns DEFAULT_SCORE (0.0) without error.
    Useful during development when the large data file is not yet downloaded.

Public interface
----------------
    connector    = PhyloPConnector(phylop_file="path/to/file")
    annotated_df = connector.annotate_dataframe(canonical_df)
    score        = connector.get_score("17", 43071077)

Phase 2 feature delivered:
    phylop_score  — PhyloP100way conservation score (float, −30 … +30)

CHANGES:
    Initial implementation for Phase 2, Connector 5.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCORE: float = 0.0      # neutral — used for missing positions / stub mode
CHUNK_SIZE: int = 1_000_000     # rows per chunk when parsing flat TSV files

# ---------------------------------------------------------------------------
# PHYLOP-SOURCE-OWNERSHIP-1 (2026-08-12)
# ---------------------------------------------------------------------------
#
# THE DEFECT. Two connectors owned and mutated one semantic field:
#
#     real_data_prep.py:823   # 1. dbNSFP  -> writes phylop_score
#     real_data_prep.py:833   # 2. PhyloP  -> OVERWROTE it
#     phylop.py:162           out["phylop_score"] = scores   (unconditional)
#
# dbNSFP supplies phylop_score with 17,706 distinct values across the ClinVar
# index. PhyloPConnector replaced the WHOLE COLUMN -- and in STUB MODE, a
# documented supported configuration in which it has no source at all, every
# value became 0.0.
#
# Zero is not a sentinel here. phyloP is SIGNED: positive means conservation,
# negative means faster-than-neutral evolution, zero means NEUTRAL. An absent
# source therefore asserted "this position evolves neutrally" for every variant
# in the cohort, and destroyed the measurements already present.
#
# THE COLLISION WAS TESTED AS INTENDED. test_phylop_block.py carried
# `test_annotate_replaces_existing_phylop_score`, whose docstring read "If the
# DataFrame already has 'phylop_score' it is overwritten." That was a contract,
# and this commit deliberately supersedes it.
#
# THE REPAIR IS OWNERSHIP, NOT POLITER OVERWRITING. Even gap-filling would
# silently assert PhyloP_dbNSFP == PhyloP_bigWig, which nobody has established
# for the installed dbNSFP release, the BigWig asset, the assembly, the track,
# or the coordinate convention.
#
#     PhyloPConnector observes BigWig evidence.
#     It does not define canonical PhyloP evidence.
#
# STAGED MIGRATION: dbNSFP remains the temporary canonical producer of
# phylop_score until PHYLOP-RECONCILE-1 renames it to phylop_dbnsfp and a
# resolver becomes the sole canonical owner. Downstream feature engineering is
# unchanged by this commit.
# ---------------------------------------------------------------------------

#: The only column this connector may publish.
OUTPUT_COLUMN: str = "phylop_bigwig"

#: The canonical feature. This connector must NEVER write it. Named so the
#: prohibition is greppable and the ownership test imports one authority.
CANONICAL_COLUMN: str = "phylop_score"

#: TRANSITIONAL MARKERS -- the current substrate is not the approved endpoint.
#: PHYLOPPERF-1 changes the first; the assembly-registry platform commit
#: changes the second. Recorded so transitional state is visible rather than
#: quietly permanent.
PHYLOP_LOOKUP_SUBSTRATE: str = "legacy_dict_v1"
PHYLOP_CHROMOSOME_RESOLUTION: str = "legacy_normalise_chrom_v1"

#: CANONICAL OWNERSHIP IS TRANSITIONAL, AND THIS RECORDS IT STRUCTURALLY.
#:
#: A1 stops PhyloPConnector from writing phylop_score. It does NOT establish
#: dbNSFP as the permanent owner of that feature. dbNSFP INHERITS it for now
#: because it is the incumbent producer and removing the collision is urgent;
#: the endpoint is that NEITHER source connector owns the canonical name.
#:
#:     dbNSFP  -> phylop_dbnsfp  ---+
#:                                  +--> reconciler -> phylop_score
#:     BigWig  -> phylop_bigwig  ---+
#:
#: PHYLOP-RECONCILE-1 renames dbNSFP's observation, measures the agreement
#: distribution before choosing any tolerance, preserves BOTH_AGREE /
#: BIGWIG_ONLY / DBNSFP_ONLY / BOTH_CONFLICT / UNOBSERVED as evidence rather
#: than collapsing them with fillna, and sets this constant to
#: "explicit_reconciliation_v1".
#:
#: It is a constant rather than a comment because a surgical repair has a habit
#: of becoming permanent architecture through inertia, and a comment cannot be
#: asserted on.
PHYLOP_CANONICALIZATION_STATE: str = "transitional_dbnsfp_inherited_v1"


class PhyloPContractError(RuntimeError):
    """A backend or caller violated the connector's declared contract."""


class PhyloPLookupBackend(Protocol):
    """One score per input row, index preserved.

    No observation at a locus -> NaN.
    Failure to READ or QUERY the source -> raise. PHYLOP-QUERY-INTEGRITY-1
    makes the current swallowing explicit; this protocol already forbids it.
    """

    def lookup_many(self, loci: "pd.DataFrame") -> "pd.Series":  # pragma: no cover
        ...


class DictPhyloPBackend:
    """TRANSITIONAL backend over the existing position dictionary.

    Deliberately not the endpoint. A Python-level loop over 4.4 million rows is
    what PHYLOPPERF-1 removes, and the dictionary carries a second defect it
    also removes: `d[(chrom, pos)] = score` means LAST ROW WINS, so a duplicated
    locus resolves by source row order -- the identical failure as
    `drop_duplicates(keep="first")` in the gnomAD connector, which disagreed
    with MANE Select for 5,468 of 17,473 genes. A relational join with
    validate="many_to_one" makes that an integrity error instead.

    It exists because SEMANTICS are what this commit fixes. Putting them behind
    a stable interface lets PHYLOPPERF-1 replace the engine without reopening
    the ownership contract: temporary implementation is acceptable, temporary
    semantics are not.
    """

    substrate = PHYLOP_LOOKUP_SUBSTRATE

    def __init__(self, index: dict) -> None:
        self._index = index

    def lookup_many(self, loci: "pd.DataFrame") -> "pd.Series":
        absent = [c for c in ("chrom", "pos") if c not in loci.columns]
        if absent:
            raise PhyloPContractError(
                "locus frame is missing required column(s): {}".format(absent))
        values = [
            # float("nan"), not np.nan: this module does NOT import numpy.
            # The reconstruction A1 was tested against did, so np.nan
            # resolved there and raised NameError against the repository --
            # ten failures from one symbol. Identical semantics, no new
            # import, no blast radius.
            self._index.get(
                (_normalise_chrom(str(chrom)), int(pos)), float("nan"))
            for chrom, pos in zip(loci["chrom"], loci["pos"])
        ]
        return pd.Series(values, index=loci.index, dtype="float64")

# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise_chrom(chrom: str) -> str:
    """Strip 'chr' prefix; 'chrM' / 'M' → 'MT'; upper-case sex chromosomes."""
    c = str(chrom).strip()
    if c.upper().startswith("CHR"):
        c = c[3:]
    if c.upper() == "M":
        c = "MT"
    return c.upper() if c in ("X", "Y", "MT") else c


# ---------------------------------------------------------------------------
# PhyloPConnector
# ---------------------------------------------------------------------------

class PhyloPConnector:
    """
    Annotates variants with PhyloP100way conservation scores.

    Parameters
    ----------
    phylop_file:
        Path to the PhyloP data file.  Accepted formats:

        * ``*.bw`` / ``*.bigWig`` — UCSC BigWig (requires pyBigWig).
        * ``*.tsv`` / ``*.tsv.gz`` / ``*.txt`` — flat tab-delimited file with
          columns ``chrom``, ``pos``, ``phylop_score``.
        * ``*.parquet`` — pre-built index parquet (fastest warm start).

        Pass *None* to operate in stub mode (returns DEFAULT_SCORE for every
        variant without raising an error).

    cache_dir:
        Where to write the parquet index cache when parsing a flat file.
        Defaults to the directory of *phylop_file*.
    """

    source_name = "phylop"

    def __init__(
        self,
        phylop_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._path: Optional[Path] = Path(phylop_file) if phylop_file else None
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir
            else (self._path.parent if self._path else None)
        )
        self._index: Optional[dict[tuple[str, int], float]] = None
        self._bw = None   # pyBigWig handle, opened lazily

    @property
    def available(self) -> bool:
        """Whether this connector has any source of observations.

        An INJECTED index counts. Tests construct PhyloPConnector(None) and set
        `_index` directly to avoid BigWig file input/output; if availability
        were path-only, every one of those would silently become a stub no-op
        and the tests would pass while measuring nothing.
        """
        return self._index is not None or self._path is not None

    def _lookup_backend(self) -> "PhyloPLookupBackend":
        """The transitional dictionary-backed lookup. Replaced by PHYLOPPERF-1."""
        return DictPhyloPBackend(self._get_index())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add (or replace) a ``phylop_score`` column on a copy of *df*.

        The input DataFrame must have columns ``chrom`` and ``pos``
        (canonical schema).

        Parameters
        ----------
        df:
            Canonical-schema DataFrame.
        missing_value:
            Score for positions absent from the index (default 0.0).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``phylop_score`` column appended / replaced.
        """
        if not self.available:
            logger.warning(
                "PhyloP BigWig source is not configured; annotation is a "
                "STRICT NO-OP. No %s column is created and %s is left "
                "untouched. Before 2026-08-12 this path overwrote %s with 0.0 "
                "for every row in the cohort.",
                OUTPUT_COLUMN, CANONICAL_COLUMN, CANONICAL_COLUMN,
            )
            return df.copy()

        absent = [c for c in ("chrom", "pos") if c not in df.columns]
        if absent:
            raise PhyloPContractError(
                "PhyloP annotation requires column(s) {}; the frame has "
                "{}".format(absent, sorted(df.columns)[:12]))

        # Only the columns the lookup needs. A connector that cannot see a
        # column cannot corrupt it.
        loci = df.loc[:, ["chrom", "pos"]]
        scores = self._lookup_backend().lookup_many(loci)

        if not isinstance(scores, pd.Series):
            raise PhyloPContractError(
                "lookup backend returned {}, expected a pandas Series".format(
                    type(scores).__name__))
        if not scores.index.equals(df.index):
            raise PhyloPContractError(
                "lookup backend did not preserve row identity: {} value(s) "
                "for {} row(s)".format(len(scores), len(df)))

        out = df.copy()
        out[OUTPUT_COLUMN] = scores.astype("float64")
        return out

    def get_score(
        self,
        chrom: str,
        pos: int,
    ) -> Optional[float]:
        """
        Return the PhyloP score for a single genomic position.

        Parameters
        ----------
        chrom:
            Chromosome (with or without 'chr' prefix).
        pos:
            1-based genomic position (GRCh38).
        missing_value:
            Returned when the position is not in the index.
        """
        chrom_norm = _normalise_chrom(chrom)

        # None, not a numeric sentinel. The caller could previously pass
        # missing_value=0.0 (or 42) and thereby decide that an unobserved
        # conservation score means a specific biological value. That is the
        # semantic hole CONSTRAINTFILL-1 closed for gnomAD constraint, and it
        # is closed here. Conversion to NaN happens at the tabular boundary,
        # so a sentinel can never enter arithmetic as if it were data.
        if self._index is not None:
            return self._index.get((chrom_norm, int(pos)))

        if self._path is None:
            return None

        # BigWig path
        if self._path.suffix.lower() in (".bw", ".bigwig"):
            return self._query_bigwig(chrom_norm, pos, float("nan"))

        # Flat-file / parquet path — build and cache index
        return self._get_index().get((chrom_norm, int(pos)))

    # ------------------------------------------------------------------
    # BigWig lookup
    # ------------------------------------------------------------------

    def _open_bigwig(self):
        """Open the BigWig file, trying pyBigWig then pybigtools."""
        if self._bw is not None:
            return self._bw
        try:
            import pyBigWig
            self._bw = pyBigWig.open(str(self._path))
            self._bw_type = "pybigwig"
            return self._bw
        except ImportError:
            pass
        try:
            import pybigtools
            self._bw = pybigtools.open(str(self._path))
            self._bw_type = "pybigtools"
            return self._bw
        except ImportError:
            pass
        raise ImportError(
            "A BigWig reader is required for .bw files. "
            "Install one with: pip install pyBigWig"
        )

    def _query_bigwig(self, chrom: str, pos: int, missing_value: float) -> float:
        """Fetch a single position from the BigWig file (1-based pos → 0-based interval)."""
        try:
            bw = self._open_bigwig()
            # pyBigWig / pybigtools use 0-based half-open intervals
            chrom_bw = f"chr{chrom}" if not chrom.startswith("chr") else chrom
            if self._bw_type == "pybigwig":
                vals = bw.values(chrom_bw, pos - 1, pos)
                if vals and vals[0] is not None and not math.isnan(vals[0]):
                    return float(vals[0])
            else:
                vals = list(bw.values(chrom_bw, pos - 1, pos, fillna=0.0))
                if vals and vals[0] is not None and not math.isnan(vals[0]):
                    return float(vals[0])
        except Exception as exc:
            logger.debug("PhyloP BigWig query failed for %s:%d -- %s", chrom, pos, exc)
        return missing_value

    # ------------------------------------------------------------------
    # Flat-file index (TSV / parquet)
    # ------------------------------------------------------------------

    def _get_index(self) -> dict[tuple[str, int], float]:
        """Return (building if necessary) the in-memory position → score index."""
        if self._index is not None:
            return self._index

        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            logger.info("PhyloP: loading index from parquet cache %s", cache_path)
            self._index = self._parquet_to_index(cache_path)
            return self._index

        logger.info("PhyloP: building index from %s (this may take a minute)...", self._path)
        self._index = self._build_index()

        if cache_path:
            self._save_cache(cache_path)

        return self._index

    def _build_index(self) -> dict[tuple[str, int], float]:
        """Parse the flat TSV / parquet file and return the index dict."""
        if self._path is None:
            return {}

        suffix = self._path.suffix.lower()
        if suffix == ".parquet":
            return self._parquet_to_index(self._path)

        # TSV / TSV.GZ — chunked read
        index: dict[tuple[str, int], float] = {}
        compression = "gzip" if suffix == ".gz" else "infer"
        first_chunk = True
        for chunk in pd.read_csv(
            self._path,
            sep="\t",
            header=None,
            names=["chrom", "pos", "phylop_score"],
            compression=compression,
            chunksize=CHUNK_SIZE,
            dtype={"chrom": str, "pos": "Int64", "phylop_score": float},
            on_bad_lines="skip",
        ):
            if first_chunk:
                # Drop header row if the file has one
                if str(chunk.iloc[0]["pos"]).lower() in ("pos", "position", "start"):
                    chunk = chunk.iloc[1:]
                first_chunk = False
            chunk = chunk.dropna(subset=["pos", "phylop_score"])
            for row in chunk.itertuples(index=False):
                chrom = _normalise_chrom(str(row.chrom))
                index[(chrom, int(row.pos))] = float(row.phylop_score)

        logger.info("PhyloP: index built with %d positions.", len(index))
        return index

    @staticmethod
    def _parquet_to_index(path: Path) -> dict[tuple[str, int], float]:
        df = pd.read_parquet(path, columns=["chrom", "pos", "phylop_score"])
        df["chrom"] = df["chrom"].apply(_normalise_chrom)
        return {
            (_normalise_chrom(str(r.chrom)), int(r.pos)): float(r.phylop_score)
            for r in df.itertuples(index=False)
        }

    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "phylop100way_index.parquet"

    def _save_cache(self, cache_path: Path) -> None:
        if not self._index:
            return
        try:
            rows = [
                {"chrom": chrom, "pos": pos, "phylop_score": score}
                for (chrom, pos), score in self._index.items()
            ]
            pd.DataFrame(rows).to_parquet(cache_path, index=False)
            logger.info("PhyloP: parquet cache written to %s", cache_path)
        except Exception as exc:
            logger.warning("PhyloP: could not write parquet cache: %s", exc)
