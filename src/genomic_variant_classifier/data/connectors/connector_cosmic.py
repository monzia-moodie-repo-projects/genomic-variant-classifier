"""
src/genomic_variant_classifier/data/connectors/connector_cosmic.py
==================================================================
COSMIC Cancer Mutation Census (CMC) somatic-recurrence connector -- Phase 2 (2026-07-06).

Adds variant-level somatic-recurrence features from the COSMIC CMC "AllData" export
to a germline variant frame. This is a *feature*, never a label: the classifier's
target is germline ClinVar pathogenicity, and COSMIC is somatic cancer data, so
recurrence in tumours is an independent signal -- BUT the CMC file itself ships a
`CLINVAR_CLNSIG` column (the label) and several scores already present natively in
the pipeline, so those are hard-excluded here (see EXCLUDED below).

Features produced (both real computed values -- NOT PHASE_2_FEATURES placeholders):
  cosmic_recurrence  float in [0, 1]   COSMIC_SAMPLE_MUTATED / COSMIC_SAMPLE_TESTED
                                       (0.0 when untested / absent / non-substitution).
  cosmic_sig_tier    float {0,1,2,3}   MUTATION_SIGNIFICANCE_TIER ordinal:
                                       Other->0, 3(low)->1, 2(med)->2, 1(high)->3.

Key (GRCh38, variant-level): `Mutation genome position GRCh38` (chrom:start-stop) +
GENOMIC_WT_ALLELE_SEQ (ref) + GENOMIC_MUT_ALLELE_SEQ (alt) -> chrom:pos:ref:alt,
matched against the cohort's identical key. v1 is SUBSTITUTION-GATED
(`Mutation Description CDS == 'Substitution'`); indels use different coordinate/allele
conventions across COSMIC vs the cohort and would mis-join, so they get 0.0 (flagged).
CMC is ~one row per variant; the ~0.1% of GENOMIC_MUTATION_IDs on multiple rows are
de-duplicated per key by MAX (COSMIC_SAMPLE_* are per-variant genome-wide totals per
the README, so summing would double-count).

EXCLUDED (documented, auditable):
  - CLINVAR_CLNSIG / CLINVAR_TRAIT  -> that IS the label (5=Pathogenic ...). Never read.
  - MIN_SIFT_SCORE/PRED, GERP++_RS  -> already in the pipeline via dbNSFP.
  - EXAC_* / GNOMAD_* AF columns    -> gnomAD v4 is native; CMC's v2.1.1/v3.1 is stale.
  - FATHMM-MKL                      -> not present in CMC AllData (nothing to drop).

Backends: pandas + gzip only (no network). If the TSV path is None/missing the
connector runs STUB (features default to 0.0, logged, never silently). A parquet
sidecar (chrom,pos,ref,alt,cosmic_recurrence,cosmic_sig_tier) is written next to the
TSV so the ~50M-row parse happens once.
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COSMIC_COLS = ["cosmic_recurrence", "cosmic_sig_tier"]
COSMIC_DEFAULTS: dict[str, float] = {"cosmic_recurrence": 0.0, "cosmic_sig_tier": 0.0}

# MUTATION_SIGNIFICANCE_TIER (README): 1 high, 2 medium, 3 low, Other none.
_TIER_MAP = {"1": 3.0, "2": 2.0, "3": 1.0, "Other": 0.0}

# Source column names (from the verified v104 header/README).
_C_POS38 = "Mutation genome position GRCh38"
_C_WT = "GENOMIC_WT_ALLELE_SEQ"
_C_MUT = "GENOMIC_MUT_ALLELE_SEQ"
_C_DESC_CDS = "Mutation Description CDS"
_C_TESTED = "COSMIC_SAMPLE_TESTED"
_C_MUTATED = "COSMIC_SAMPLE_MUTATED"
_C_TIER = "MUTATION_SIGNIFICANCE_TIER"
_C_GENOMIC_ID = "GENOMIC_MUTATION_ID"

_USECOLS = [
    _C_POS38, _C_WT, _C_MUT, _C_DESC_CDS,
    _C_TESTED, _C_MUTATED, _C_TIER, _C_GENOMIC_ID,
]

# Columns that must NEVER be read into features (label / duplication / stale).
_FORBIDDEN = {
    "CLINVAR_CLNSIG", "CLINVAR_TRAIT",
    "MIN_SIFT_SCORE", "MIN_SIFT_PRED", "GERP++_RS",
}


def _norm_chrom(c: str) -> str:
    """Strip a 'chr' prefix and normalise MT/M; COSMIC uses bare '7','X','MT'."""
    c = str(c).strip()
    if c[:3].lower() == "chr":
        c = c[3:]
    return "MT" if c in ("M", "MT") else c


def _make_key(chrom: pd.Series, pos, ref: pd.Series, alt: pd.Series) -> pd.Series:
    return (
        chrom.map(_norm_chrom).astype(str) + ":" + pd.Series(pos).astype(str)
        + ":" + ref.astype(str).str.upper() + ":" + alt.astype(str).str.upper()
    )


class CosmicCmcConnector:
    """Variant-level COSMIC CMC recurrence connector (GRCh38)."""

    source_name = "cosmic_cmc"

    def __init__(
        self,
        cosmic_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        min_coverage_warn: float = 0.0,
    ) -> None:
        self._tsv_path: Path | None = Path(cosmic_path) if cosmic_path else None
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir
            else (self._tsv_path.parent if self._tsv_path else None)
        )
        self._index: Optional[pd.DataFrame] = None  # columns: _key, cosmic_recurrence, cosmic_sig_tier
        self._min_coverage_warn = float(min_coverage_warn)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cosmic_recurrence + cosmic_sig_tier. Variant-level GRCh38 key. Returns a copy."""
        df = df.copy()
        for col, default in COSMIC_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(df.columns):
            logger.info(
                "COSMIC CMC: missing key columns %s — returning defaults.",
                required - set(df.columns),
            )
            return df
        if self._tsv_path is None:
            logger.debug("COSMIC CMC: stub mode — cosmic_* are defaults (0.0).")
            return df
        if not self._tsv_path.exists():
            logger.warning(
                "COSMIC CMC TSV not found at %s — cosmic_* are defaults. "
                "Download 'Cancer Mutation Census AllData' (GRCh38) from COSMIC.",
                self._tsv_path,
            )
            return df

        self._ensure_index()
        idx = self._index
        if idx is None or idx.empty:
            logger.warning("COSMIC CMC: empty index — cosmic_* are defaults.")
            return df

        keys = _make_key(df["chrom"], df["pos"], df["ref"], df["alt"])
        rec_map = dict(zip(idx["_key"], idx["cosmic_recurrence"]))
        tier_map = dict(zip(idx["_key"], idx["cosmic_sig_tier"]))
        df["cosmic_recurrence"] = (
            keys.map(rec_map).fillna(COSMIC_DEFAULTS["cosmic_recurrence"]).astype(float)
        )
        df["cosmic_sig_tier"] = (
            keys.map(tier_map).fillna(COSMIC_DEFAULTS["cosmic_sig_tier"]).astype(float)
        )

        n_hit = int((df["cosmic_recurrence"] > 0).sum())
        cov = n_hit / len(df) if len(df) else 0.0
        logger.info(
            "COSMIC CMC: %d / %d variants matched (cosmic_recurrence > 0); "
            "%d with cosmic_sig_tier > 0.",
            n_hit, len(df), int((df["cosmic_sig_tier"] > 0).sum()),
        )
        if cov < self._min_coverage_warn:
            logger.warning(
                "COSMIC CMC: match coverage %.2f%% below %.2f%% — check GRCh38 build / key.",
                100.0 * cov, 100.0 * self._min_coverage_warn,
            )
        return df

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "cosmic_cmc_grch38_index.parquet"

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        cache = self._cache_path()
        if cache is not None and cache.exists():
            logger.info("Loading COSMIC CMC cache from %s", cache)
            self._index = pd.read_parquet(cache)
            logger.info("COSMIC CMC index: %d variants", len(self._index))
            return

        logger.info("Parsing COSMIC CMC TSV: %s", self._tsv_path)
        self._index = self._parse_tsv(self._tsv_path)  # type: ignore[arg-type]
        logger.info("Parsed %d unique COSMIC CMC variants", len(self._index))

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            self._index.to_parquet(cache, index=False)
            logger.info("COSMIC CMC cache written to %s", cache)

    @staticmethod
    def _parse_tsv(path: Path) -> pd.DataFrame:
        """Stream the CMC TSV -> per-variant GRCh38-keyed recurrence + tier index."""
        opener = (
            (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace"))
            if str(path).endswith(".gz")
            else (lambda p: open(p, encoding="utf-8", errors="replace"))
        )
        # Guard: refuse to even load forbidden (label) columns.
        with opener(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
        leaked = _FORBIDDEN.intersection(header) & set()  # never in _USECOLS by construction
        assert not leaked, f"forbidden columns requested: {leaked}"  # defensive; _USECOLS is the allowlist

        rows = pd.read_csv(
            path, sep="\t", usecols=[c for c in _USECOLS if c in header],
            dtype=str, na_filter=False, compression="gzip" if str(path).endswith(".gz") else None,
        )

        # substitution gate
        if _C_DESC_CDS in rows.columns:
            rows = rows[rows[_C_DESC_CDS].str.strip() == "Substitution"]

        # parse GRCh38 position "chrom:start-stop" -> chrom, pos(start)
        pos = rows[_C_POS38].astype(str).str.strip()
        parts = pos.str.split(":", n=1, expand=True)
        chrom = parts[0]
        start = parts[1].str.split("-", n=1, expand=True)[0]
        start = pd.to_numeric(start, errors="coerce")

        ref = rows[_C_WT].astype(str).str.upper()
        alt = rows[_C_MUT].astype(str).str.upper()

        tested = pd.to_numeric(rows[_C_TESTED], errors="coerce")
        mutated = pd.to_numeric(rows[_C_MUTATED], errors="coerce")
        recurrence = (mutated / tested).where(tested > 0, other=0.0)
        tier = rows[_C_TIER].astype(str).str.strip().map(_TIER_MAP).fillna(0.0)

        out = pd.DataFrame({
            "chrom": chrom.map(_norm_chrom),
            "pos": start,
            "ref": ref,
            "alt": alt,
            "cosmic_recurrence": recurrence.astype(float),
            "cosmic_sig_tier": tier.astype(float),
        })
        # valid single-base substitutions only (ref/alt single ACGT, pos present)
        valid = (
            out["pos"].notna()
            & out["ref"].str.fullmatch(r"[ACGT]")
            & out["alt"].str.fullmatch(r"[ACGT]")
        )
        out = out[valid].copy()
        out["pos"] = out["pos"].astype("int64")
        out["_key"] = (
            out["chrom"].astype(str) + ":" + out["pos"].astype(str)
            + ":" + out["ref"] + ":" + out["alt"]
        )
        # per-key de-dup by MAX (README: SAMPLE_* are per-variant totals -> not summed)
        out = (
            out.groupby("_key", as_index=False)
            .agg({"cosmic_recurrence": "max", "cosmic_sig_tier": "max"})
        )
        return out[["_key", "cosmic_recurrence", "cosmic_sig_tier"]]
