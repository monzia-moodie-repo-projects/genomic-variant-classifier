"""
src/genomic_variant_classifier/data/dbnsfp.py
==================
Connector 7: dbNSFP full-width score annotator.

Delivers six pre-computed pathogenicity and conservation scores for all
possible missense SNVs in the human exome (GRCh38) in a single file pass:

    sift_score      — SIFT [0, 1];           ≤ 0.05 = deleterious (ACMG PP3)
    polyphen2_score — PolyPhen-2 HDIV [0, 1]; ≥ 0.908 = probably damaging
    revel_score     — REVEL ensemble [0, 1];  ≥ 0.5 suggestive of pathogenicity
    cadd_phred      — CADD PHRED (unbounded); ≥ 20 = likely deleterious
    phylop_score    — phyloP100way (~ −30…+30); > 0 = conserved
    gerp_score      — GERP++ RS (~ −12…+6);   > 0 = constrained

All six columns are already in TABULAR_FEATURES (gerp_score added in
Connector 7).  For variants absent from dbNSFP — indels, synonymous SNVs,
intronic variants, novel variants not yet in the database — each score
receives its population-median default, identical to the fill-ins used by
engineer_features() so the two code paths stay consistent.

Relationship to earlier connectors
-----------------------------------
Connector 4 (REVELConnector)         → superseded for missense variants
Connector 5 (PhyloPConnector)        → superseded for missense variants
                                       (PhyloP is per-position, not per-allele,
                                        so PhyloPConnector remains the authoritative
                                        source for non-missense variants)
Connector 6 (SIFTPolyPhenConnector)  → superseded for missense variants

Running DbNSFPConnector and the earlier connectors together is safe but
redundant; DbNSFPConnector.annotate_dataframe() always overwrites its six
columns.  The recommended production sequence is:

    1. DbNSFPConnector.annotate_dataframe(df)   — covers all missense SNVs
    2. PhyloPConnector.annotate_dataframe(df)   — fills phylop_score for
                                                   non-missense variants
    3. CADDConnector.fetch(df)                  — fills cadd_phred for
                                                   non-missense variants

Data source — manual download required:
    https://sites.google.com/site/jpopgen/dbNSFP
    Recommended file: dbNSFP4.x_variant.chr*.gz files, concatenated.
    Academic / non-commercial use only (see licence on download page).

Raw file format (tab-delimited, has header row starting with '#chr'):
    #chr  pos(1-based)  ref  alt  ...  SIFT_score  ...  Polyphen2_HDIV_score
    ...   REVEL_score   ...  CADD_phred  ...  phyloP100way_vertebrate
    ...   GERP++_RS  ...

    • Chromosome values have no 'chr' prefix (e.g. '1', 'X')
    • Scores are semicolon-delimited when multiple transcripts are present
    • Dots represent missing transcript values

Multi-transcript aggregation:
    SIFT        → min  (most deleterious transcript)
    PolyPhen-2  → max  (most damaging transcript)
    REVEL       → max  (highest ensemble pathogenicity score)
    CADD_phred  → max  (most deleterious prediction)
    phyloP      → max  (most conserved position across transcripts)
    GERP++      → max  (strongest evolutionary constraint signal)

Lookup strategy (identical to REVELConnector and SIFTPolyPhenConnector):
    Pass an explicit historical parquet for indexed replay, or a raw table for
    an in-memory legacy projection. Repeated raw allele keys are refused.

Stub mode:
    Pass dbnsfp_file=None.  Every lookup returns default scores and a
    WARNING is emitted.  All connectors in the pipeline continue normally.

Public interface:
    connector = DbNSFPConnector(dbnsfp_file="path/to/dbNSFP4.x.gz")
    df_out    = connector.annotate_dataframe(canonical_df)
    scores    = connector.get_scores("17", 43071077, "G", "T")  # → DbNSFPScores
    sift      = connector.get_scores("17", 43071077, "G", "T").sift_score

PHASE_2_PLACEHOLDER:
    CADD REST API fallback for variants absent from dbNSFP is not yet
    implemented.  Currently CADDConnector handles those variants; wire
    DbNSFPConnector + CADDConnector in pipeline.py as complementary
    sources once both are confirmed working end-to-end.

CHANGES:
    Initial implementation for Phase 2, Connector 7.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# Re-use normalisation helpers from Connector 6 — no duplication
from genomic_variant_classifier.data.sift_polyphen import _normalise_chrom, _parse_multival

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score defaults — must match engineer_features() score_defaults exactly
# ---------------------------------------------------------------------------

DEFAULT_SIFT:   float = 0.5
DEFAULT_PP2:    float = 0.5
DEFAULT_REVEL:  float = 0.5
DEFAULT_CADD:   float = 15.0
DEFAULT_PHYLOP: float = 0.0
DEFAULT_GERP:   float = 0.0

CHUNK_SIZE: int = 500_000

# dbNSFP v4.x column names (verify against file header for other versions)
_COL_CHROM  = "#chr"
_COL_POS    = "pos(1-based)"
_COL_REF    = "ref"
_COL_ALT    = "alt"
_COL_SIFT   = "SIFT_score"
_COL_PP2    = "Polyphen2_HDIV_score"
_COL_REVEL  = "REVEL_score"
_COL_CADD   = "CADD_phred"
_COL_PHYLOP = "phyloP100way_vertebrate"
_COL_GERP   = "GERP++_RS"

_ALL_SCORE_COLS = [
    _COL_SIFT, _COL_PP2, _COL_REVEL, _COL_CADD, _COL_PHYLOP, _COL_GERP,
]

# Aggregation strategy per score column
_AGG: dict[str, str] = {
    _COL_SIFT:   "min",
    _COL_PP2:    "max",
    _COL_REVEL:  "max",
    _COL_CADD:   "max",
    _COL_PHYLOP: "max",
    _COL_GERP:   "max",
}

# Mapping: dbNSFP column → output DataFrame column name
_OUTPUT_COLS: dict[str, str] = {
    _COL_SIFT:   "sift_score",
    _COL_PP2:    "polyphen2_score",
    _COL_REVEL:  "revel_score",
    _COL_CADD:   "cadd_phred",
    _COL_PHYLOP: "phylop_score",
    _COL_GERP:   "gerp_score",
}

# Default value per output column (must stay in sync with engineer_features)
_DEFAULTS: dict[str, float] = {
    "sift_score":       DEFAULT_SIFT,
    "polyphen2_score":  DEFAULT_PP2,
    "revel_score":      DEFAULT_REVEL,
    "cadd_phred":       DEFAULT_CADD,
    "phylop_score":     DEFAULT_PHYLOP,
    "gerp_score":       DEFAULT_GERP,
}


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DbNSFPScores:
    """
    All six dbNSFP scores for a single variant.

    Frozen so instances are hashable and can be used as dict values safely.
    All fields are plain Python floats (never NaN — missing scores receive
    the corresponding DEFAULT_* constant).
    """
    sift_score:       float = DEFAULT_SIFT
    polyphen2_score:  float = DEFAULT_PP2
    revel_score:      float = DEFAULT_REVEL
    cadd_phred:       float = DEFAULT_CADD
    phylop_score:     float = DEFAULT_PHYLOP
    gerp_score:       float = DEFAULT_GERP

    def to_dict(self) -> dict[str, float]:
        """Return scores as a plain dict keyed by output column name."""
        return {
            "sift_score":       self.sift_score,
            "polyphen2_score":  self.polyphen2_score,
            "revel_score":      self.revel_score,
            "cadd_phred":       self.cadd_phred,
            "phylop_score":     self.phylop_score,
            "gerp_score":       self.gerp_score,
        }


# Default scores object used when a variant is absent from the index
_DEFAULT_SCORES = DbNSFPScores()


# ---------------------------------------------------------------------------
# DbNSFPConnector
# ---------------------------------------------------------------------------

class DbNSFPSourceError(ValueError):
    """A configured source cannot support the requested legacy lookup."""


def _checked_score(raw: object, aggregation: str) -> Optional[float]:
    """Preserve declared missing slots while refusing unparseable or infinite observations."""
    if pd.isna(raw):
        return None
    for token in str(raw).split(";"):
        if token in {".", "", "NA"}:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise DbNSFPSourceError(f"Malformed dbNSFP score token {token!r}") from exc
        if not math.isfinite(value):
            raise DbNSFPSourceError(f"Nonfinite dbNSFP score token {token!r}")
    return _parse_multival(raw, aggregation)


class DbNSFPConnector:
    """
    Full-width dbNSFP connector delivering six pre-computed scores per variant.

    Reads the selected raw table or an explicitly requested historical index.

    Parameters
    ----------
    dbnsfp_file:
        Path to the dbNSFP flat file (plain text, gzip, or zip).  Accepts
        a single concatenated file (all chromosomes) or a single-chromosome
        file for testing.  Pass *None* to operate in stub mode.
    cache_dir:
        Legacy index directory when supplied alone. An explicit source takes precedence.
    """

    source_name = "dbnsfp"

    def __init__(
        self,
        dbnsfp_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._path = Path(dbnsfp_file) if dbnsfp_file is not None else None
        if self._path is None and cache_dir is not None:
            self._path = Path(cache_dir) / "dbnsfp_clinvar_index.parquet"
        self._index: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        **missing_overrides: float,
    ) -> pd.DataFrame:
        """
        Add (or replace) all six score columns on a copy of *df*.

        Columns written: ``sift_score``, ``polyphen2_score``, ``revel_score``,
        ``cadd_phred``, ``phylop_score``, ``gerp_score``.

        Variants absent from the dbNSFP index receive the corresponding
        DEFAULT_* constant, matching engineer_features() fill-ins exactly.

        Parameters
        ----------
        df:
            Canonical-schema DataFrame (must have chrom, pos, ref, alt).
        **missing_overrides:
            Optional per-column default overrides, e.g.
            ``cadd_phred=20.0`` to use a different fill for missing CADD.
            Unspecified columns use their module-level DEFAULT_* values.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with all six score columns added / replaced.
        """
        out = df.copy()
        # Extract unique normalised chromosomes to filter parquet cache
        _chroms: set[str] = set(
            df["chrom"].astype(str).apply(_normalise_chrom).unique()
        ) if "chrom" in df.columns else set()
        index_df = self._get_index(filter_chroms=_chroms if _chroms else None)

        # Resolve effective defaults (module defaults + any overrides)
        effective_defaults = dict(_DEFAULTS)
        effective_defaults.update(missing_overrides)

        if index_df.empty:
            for col_name, default in effective_defaults.items():
                out[col_name] = default
            return out

        # Normalise join keys on left side
        left = out[["chrom", "pos", "ref", "alt"]].copy()
        left["_chrom"] = left["chrom"].astype(str).apply(_normalise_chrom)
        left["_pos"]   = pd.to_numeric(left["pos"], errors="coerce").astype("Int64")
        left["_ref"]   = left["ref"].astype(str).str.upper()
        left["_alt"]   = left["alt"].astype(str).str.upper()

        # Memory-safe pre-filter: shrink the index to rows at cohort positions
        # BEFORE materialising the six score columns below. Without this, a full
        # dbNSFP index (85.3M rows) would copy 6 x 85.3M float64 (~3.8 GiB) into
        # `right` and OOM. Position is a necessary condition for any (chrom,pos,
        # ref,alt) match, so this pre-filter never drops a true match; the exact
        # 4-key merge below still does the real matching. Vectorised int isin.
        _cohort_pos = set(
            pd.to_numeric(out["pos"], errors="coerce").dropna().astype("int64")
        )
        _idx_pos = pd.to_numeric(index_df["pos"], errors="coerce")
        index_df = index_df.loc[_idx_pos.isin(_cohort_pos)]

        # Normalise join keys on right side
        right = index_df[["chrom", "pos", "ref", "alt"] + list(_OUTPUT_COLS.values())].copy()
        right = right.rename(columns={
            "chrom": "_chrom", "pos": "_pos", "ref": "_ref", "alt": "_alt"
        })
        right["_chrom"] = right["_chrom"].astype(str).apply(_normalise_chrom)
        right["_pos"]   = pd.to_numeric(right["_pos"], errors="coerce").astype("Int64")
        right["_ref"]   = right["_ref"].astype(str).str.upper()
        right["_alt"]   = right["_alt"].astype(str).str.upper()

        # Vectorised merge
        left = left.reset_index(drop=True)
        merged = left.merge(
            right,
            on=["_chrom", "_pos", "_ref", "_alt"],
            how="left",
            validate="many_to_one",
        )

        score_cols = list(_OUTPUT_COLS.values())
        for col_name in score_cols:
            default = effective_defaults.get(col_name, 0.0)
            out[col_name] = merged[col_name].fillna(default).values

        n_hit = int(merged[score_cols[0]].notna().sum())
        logger.debug(
            "DbNSFP: annotated %d/%d variants with real scores.",
            n_hit, len(df),
        )
        return out

    def get_scores(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
    ) -> DbNSFPScores:
        """
        Return all six scores for a single variant.

        Parameters
        ----------
        chrom:
            Chromosome (with or without 'chr' prefix).
        pos:
            GRCh38 position (1-based).
        ref:
            Reference allele (case-insensitive).
        alt:
            Alternate allele (case-insensitive).

        Returns
        -------
        DbNSFPScores
            All six scores.  Fields equal their DEFAULT_* constant for
            variants absent from the index.
        """
        index_df = self._get_index(filter_chroms={_normalise_chrom(chrom)})
        if index_df.empty:
            return _DEFAULT_SCORES
        chrom_n = _normalise_chrom(chrom)
        mask = (
            (index_df["chrom"].apply(_normalise_chrom) == chrom_n) &
            (index_df["pos"] == int(pos)) &
            (index_df["ref"].str.upper() == ref.upper()) &
            (index_df["alt"].str.upper() == alt.upper())
        )
        rows = index_df[mask]
        if rows.empty:
            return _DEFAULT_SCORES
        row = rows.iloc[0]
        return DbNSFPScores(
            sift_score=float(row.get("sift_score",      DEFAULT_SIFT)),
            polyphen2_score=float(row.get("polyphen2_score", DEFAULT_PP2)),
            revel_score=float(row.get("revel_score",    DEFAULT_REVEL)),
            cadd_phred=float(row.get("cadd_phred",      DEFAULT_CADD)),
            phylop_score=float(row.get("phylop_score",  DEFAULT_PHYLOP)),
            gerp_score=float(row.get("gerp_score",      DEFAULT_GERP)),
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _get_index(
        self,
        filter_chroms: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        if self._index is not None:
            return self._index
        if self._path is not None and self._path.suffix.lower() in {".parquet", ".pq"}:
            return self._load_index(filter_chroms=filter_chroms)
        self._index = self._load_index()
        return self._index

    def _load_index(
        self,
        filter_chroms: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Read only the specified input. An unconfigured connector is the legacy stub."""
        keys = ["chrom", "pos", "ref", "alt"]
        columns = keys + list(_OUTPUT_COLS.values())
        if self._path is None:
            logger.warning("DbNSFP: explicit stub, using legacy defaults")
            return pd.DataFrame(columns=columns)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        suffix = self._path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            with pq.ParquetFile(self._path) as file:
                if file.metadata.num_rows == 0 or not set(columns) <= set(file.schema_arrow.names):
                    raise DbNSFPSourceError("empty or incomplete historical parquet index")
            filters = None
            if filter_chroms:
                names = set(filter_chroms)
                if "MT" in names:
                    names.add("M")
                aliases = {p + c for c in names for p in ("", "chr", "CHR")}
                filters = [("chrom", "in", sorted(aliases))]
            frame = pd.read_parquet(self._path, columns=columns, filters=filters)
            if frame[keys].isna().any().any():
                raise DbNSFPSourceError("missing allele key in historical parquet")
            frame["chrom"] = frame["chrom"].map(_normalise_chrom)
            frame["ref"] = frame["ref"].str.upper()
            frame["alt"] = frame["alt"].str.upper()
            self._require_unique(frame, keys)
            return frame

        compression = {".gz": "gzip", ".bgz": "gzip", ".bgzf": "gzip", ".zip": "zip"}.get(suffix)
        # Guard: some dbNSFP columns may be absent in older versions;
        # we discover which are actually present from the file header.
        chunks: list[pd.DataFrame] = []
        try:
            # Peek at header to determine which score columns are actually present
            header_df = pd.read_csv(
                self._path,
                sep="\t",
                compression=compression,
                nrows=0,
                dtype=str,
            )
            available_score_cols = [
                c for c in _ALL_SCORE_COLS if c in header_df.columns
            ]
            if not available_score_cols:
                raise DbNSFPSourceError("Configured table has no supported score columns")
            missing_from_file = set(_ALL_SCORE_COLS) - set(available_score_cols)
            if missing_from_file:
                logger.warning(
                    "DbNSFP: columns absent from file (will use defaults): %s",
                    missing_from_file,
                )
            usecols = [_COL_CHROM, _COL_POS, _COL_REF, _COL_ALT] + available_score_cols

            reader = pd.read_csv(
                self._path,
                sep="\t",
                compression=compression,
                usecols=usecols,
                dtype=str,
                keep_default_na=False,
                na_values=[".", "", "NA"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            with reader:
                for chunk in reader:
                    chunk = chunk.rename(columns={_COL_CHROM: "chrom", _COL_POS: "pos"})
                    if chunk[keys].isna().any().any() or not chunk["pos"].str.fullmatch(r"[1-9][0-9]*").all():
                        raise DbNSFPSourceError("missing allele key or invalid 1-based position")
                    chunk["pos"] = chunk["pos"].astype("int64")
                    chunk["chrom"] = chunk["chrom"].apply(_normalise_chrom)
                    chunk["ref"]   = chunk["ref"].str.upper()
                    chunk["alt"]   = chunk["alt"].str.upper()

                    # SNVs only
                    snv = (
                        chunk["ref"].str.len() == 1
                    ) & (
                        chunk["alt"].str.len() == 1
                    )
                    chunk = chunk.loc[snv].copy()

                    self._require_unique(chunk, keys)

                    # Parse and aggregate multi-transcript score fields
                    for raw_col in available_score_cols:
                        out_col = _OUTPUT_COLS[raw_col]
                        agg     = _AGG[raw_col]
                        default = _DEFAULTS[out_col]
                        chunk[out_col] = chunk[raw_col].apply(
                            lambda v, a=agg: _checked_score(v, a)
                        )
                        chunk[out_col] = chunk[out_col].fillna(default)

                    # Add default columns for any score columns absent from this file
                    for raw_col in _ALL_SCORE_COLS:
                        out_col = _OUTPUT_COLS[raw_col]
                        if out_col not in chunk.columns:
                            chunk[out_col] = _DEFAULTS[out_col]

                    keep = ["chrom", "pos", "ref", "alt"] + list(_OUTPUT_COLS.values())
                    chunks.append(chunk[keep].copy())

        except DbNSFPSourceError:
            raise
        except (OSError, EOFError, ValueError, KeyError) as exc:
            raise DbNSFPSourceError(f"Cannot parse configured dbNSFP source {self._path}: {exc}") from exc

        if not chunks or not any(len(chunk) for chunk in chunks):
            raise DbNSFPSourceError(f"No usable rows in configured source {self._path}")
        full = pd.concat(chunks, ignore_index=True)
        self._require_unique(full, keys)
        return full

    @staticmethod
    def _require_unique(frame: pd.DataFrame, keys: list[str]) -> None:
        extra = int(frame.duplicated(keys).sum())
        if extra:
            raise DbNSFPSourceError(
                f"{extra} excess allele rows in checked batch. Use the transcript-aware projection."
            )

    @staticmethod
    def _df_to_index(
        df: pd.DataFrame,
    ) -> dict[tuple[str, int, str, str], DbNSFPScores]:
        """
        Convert a processed (chrom, pos, ref, alt, score...) DataFrame
        to the lookup dict.

        Keys: (chrom_normalised, pos_int, ref_upper, alt_upper)
        Values: DbNSFPScores dataclass instances
        """
        index: dict[tuple[str, int, str, str], DbNSFPScores] = {}
        score_cols = list(_OUTPUT_COLS.values())
        for row in df.itertuples(index=False):
            key = (
                _normalise_chrom(row.chrom),
                int(row.pos),
                row.ref.upper(),
                row.alt.upper(),
            )
            index[key] = DbNSFPScores(
                sift_score=float(getattr(row, "sift_score",      DEFAULT_SIFT)),
                polyphen2_score=float(getattr(row, "polyphen2_score", DEFAULT_PP2)),
                revel_score=float(getattr(row, "revel_score",    DEFAULT_REVEL)),
                cadd_phred=float(getattr(row, "cadd_phred",     DEFAULT_CADD)),
                phylop_score=float(getattr(row, "phylop_score",   DEFAULT_PHYLOP)),
                gerp_score=float(getattr(row, "gerp_score",     DEFAULT_GERP)),
            )
        return index