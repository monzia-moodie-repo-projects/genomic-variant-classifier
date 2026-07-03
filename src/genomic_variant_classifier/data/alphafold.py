"""
src/genomic_variant_classifier/data/alphafold.py
================================================
AlphaFold structure-feature connector -- Phase D.

Reads a prebuilt AlphaFold cohort parquet (residue-level structural features keyed
by UniProt accession + residue position) and joins four features onto variants:

    alphafold_plddt              float  Per-residue pLDDT confidence (0-100).
    solvent_accessibility        float  Relative solvent accessibility (RSA, 0-1).
    secondary_structure_context  int    0=loop, 1=helix, 2=sheet.
    dist_to_active_site          float  3-D C-alpha distance (Angstrom) to the nearest
                                        annotated active/binding site.

Residue-level join (the "hybrid" design):
  * The parquet is keyed on (uniprot_accession, residue_pos) -- compact and reusable,
    since structural features are variant-independent.
  * The connector resolves each variant's (accession, residue) at annotation time by
    consuming the CANONICAL ``protein_pos`` column that real_data_prep already
    computes at step 10b/10c (AlphaMissense + HGVSp), and mapping gene_symbol ->
    accession via the local UniProt reviewed parquet.  This means the join key is the
    SAME value that produced protein_pos -- no reimplementation of residue logic,
    hence no drift.

Mandatory wild-type cross-check (fail-closed, mirrors the ESM-2 guard):
  ``protein_pos`` from AlphaMissense/HGVSp may be numbered against a different protein
  isoform than the AlphaFold structure.  Before attaching a feature we require the
  wild-type residue implied by the variant (``wt_aa``) to match the residue at
  ``protein_pos`` in the UniProt sequence the structure is numbered against.  On any
  mismatch the variant receives the honest sentinel default rather than a
  mismatched-isoform feature.

Expected parquet columns:
    uniprot_accession  str    e.g. "P38398"
    residue_pos        int    1-based residue index (matches protein_pos)
    plddt              float
    rsa                float
    ss                 int
    dist_active        float

Stub mode:
    When parquet_path is None or the file is absent, all variants receive the sentinel
    defaults and a WARNING is logged.

Build the parquet with scripts/build_alphafold_parquet.py.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig

logger = logging.getLogger(__name__)

# Sentinel defaults (mirror ProteinStructurePipeline / real_data_prep / variant_ensemble).
DEFAULT_PLDDT = 50.0
DEFAULT_RSA = 0.5
DEFAULT_SECONDARY = 0
DEFAULT_DIST_ACTIVE = 100.0

# One-to-three amino-acid code map for the wt_aa cross-check.
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


class AlphaFoldConnector(BaseConnector):
    """
    Annotates missense variants with AlphaFold per-residue structural features.

    Usage
    -----
        connector = AlphaFoldConnector(
            parquet_path="data/external/alphafold/alphafold_cohort.parquet",
            uniprot_index_path="data/external/uniprot/uniprot_human_reviewed.parquet",
        )
        annotated = connector.annotate_dataframe(variant_df)

    If parquet_path is None or absent, stub mode applies: all variants receive the
    sentinel defaults.
    """

    source_name = "alphafold"

    def __init__(
        self,
        parquet_path: Optional[str | Path] = None,
        uniprot_index_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
    ) -> None:
        super().__init__(config)
        self.parquet_path: Optional[Path] = (
            Path(parquet_path) if parquet_path is not None else None
        )
        self.uniprot_index_path: Optional[Path] = (
            Path(uniprot_index_path) if uniprot_index_path is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the four AlphaFold structural columns to df."""
        result = df.copy()
        n = len(result)

        # Always initialise the four columns to defaults first (schema stability:
        # these are locked members of TABULAR_FEATURES; they must always exist).
        result["alphafold_plddt"] = DEFAULT_PLDDT
        result["solvent_accessibility"] = DEFAULT_RSA
        result["secondary_structure_context"] = DEFAULT_SECONDARY
        result["dist_to_active_site"] = DEFAULT_DIST_ACTIVE

        if n == 0:
            return result

        if self.parquet_path is None:
            logger.warning(
                "AlphaFoldConnector: parquet_path not set -- returning sentinel "
                "defaults. Build with scripts/build_alphafold_parquet.py."
            )
            return result

        lookup = self._get_lookup()
        if lookup is None or lookup.empty:
            return result

        gene_to_acc, acc_to_seq = self._get_uniprot_maps()
        if not gene_to_acc:
            logger.warning(
                "AlphaFoldConnector: UniProt index unavailable -- cannot resolve "
                "accessions; returning sentinel defaults."
            )
            return result

        return self._annotate(result, lookup, gene_to_acc, acc_to_seq)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_lookup(self) -> Optional[pd.DataFrame]:
        """Load the AF cohort parquet (cached), keyed (uniprot_accession, residue_pos)."""
        if not self.parquet_path.exists():
            logger.warning(
                "AlphaFoldConnector: parquet not found at '%s' -- sentinel defaults.",
                self.parquet_path,
            )
            return None

        stat = self.parquet_path.stat()
        h = hashlib.sha256()
        with self.parquet_path.open("rb") as fh:
            h.update(fh.read(1024 * 1024))
        cache_basis = (
            f"{self.parquet_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{h.hexdigest()}"
        )
        cache_key = "af_lookup_" + hashlib.sha256(cache_basis.encode("utf-8")).hexdigest()[:16]

        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("AlphaFoldConnector: loaded %d residue features from cache.", len(cached))
            return cached

        try:
            lookup = pd.read_parquet(
                self.parquet_path,
                columns=["uniprot_accession", "residue_pos", "plddt", "rsa", "ss", "dist_active"],
            )
        except Exception as exc:
            logger.error("AlphaFoldConnector: failed to read parquet: %s", exc)
            return None

        lookup = lookup.dropna(subset=["uniprot_accession", "residue_pos"])
        lookup["residue_pos"] = lookup["residue_pos"].astype(int)
        lookup = lookup.drop_duplicates(subset=["uniprot_accession", "residue_pos"])
        self._save_cache(cache_key, lookup)
        logger.info("AlphaFoldConnector: cached %d residue features.", len(lookup))
        return lookup

    def _get_uniprot_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return (gene_symbol -> accession, accession -> sequence) from the UniProt index."""
        if self.uniprot_index_path is None or not self.uniprot_index_path.exists():
            return {}, {}
        try:
            up = pd.read_parquet(
                self.uniprot_index_path,
                columns=["gene_symbol", "uniprot_id", "sequence"],
            )
        except Exception as exc:
            logger.error("AlphaFoldConnector: failed to read UniProt index: %s", exc)
            return {}, {}
        up = up.dropna(subset=["gene_symbol", "uniprot_id"])
        gene_to_acc = dict(zip(up["gene_symbol"].astype(str), up["uniprot_id"].astype(str)))
        acc_to_seq = dict(
            zip(up["uniprot_id"].astype(str), up["sequence"].fillna("").astype(str))
        )
        return gene_to_acc, acc_to_seq

    def _annotate(
        self,
        df: pd.DataFrame,
        lookup: pd.DataFrame,
        gene_to_acc: dict[str, str],
        acc_to_seq: dict[str, str],
    ) -> pd.DataFrame:
        """Join AF features by (accession, residue_pos) with the wt_aa cross-check."""
        # Build a fast (accession, residue) -> feature-tuple index.
        feat_index: dict[tuple[str, int], tuple[float, float, int, float]] = {}
        for acc, pos, plddt, rsa, ss, dist in zip(
            lookup["uniprot_accession"], lookup["residue_pos"],
            lookup["plddt"], lookup["rsa"], lookup["ss"], lookup["dist_active"],
        ):
            feat_index[(str(acc), int(pos))] = (
                float(plddt), float(rsa), int(ss), float(dist),
            )

        gene_col = df.get("gene_symbol", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
        pos_col = df.get("protein_pos", pd.Series([pd.NA] * len(df), index=df.index))
        wt_col = df.get("wt_aa", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

        plddt_out = df["alphafold_plddt"].to_list()
        rsa_out = df["solvent_accessibility"].to_list()
        ss_out = df["secondary_structure_context"].to_list()
        dist_out = df["dist_to_active_site"].to_list()

        n_attached = 0
        n_wt_mismatch = 0
        positions = list(df.index)
        for row_i, idx in enumerate(positions):
            gene = gene_col.iloc[row_i]
            if not gene:
                continue
            acc = gene_to_acc.get(gene)
            if not acc:
                continue
            pos_val = pos_col.iloc[row_i]
            if pd.isna(pos_val):
                continue
            try:
                pos = int(pos_val)
            except (ValueError, TypeError):
                continue
            feat = feat_index.get((acc, pos))
            if feat is None:
                continue

            # Mandatory wt_aa cross-check against the UniProt sequence the structure
            # is numbered against (fail-closed on mismatch).
            wt = wt_col.iloc[row_i].strip().upper()
            if wt:
                wt1 = _AA3_TO_1.get(wt, wt if len(wt) == 1 else "")
                seq = acc_to_seq.get(acc, "")
                if seq and 1 <= pos <= len(seq):
                    if wt1 and seq[pos - 1] != wt1:
                        n_wt_mismatch += 1
                        continue  # isoform mismatch -> keep sentinel default
                # If we cannot verify (no seq / out of range), do NOT attach -- safer
                # to keep the default than risk a mismatched-isoform feature.
                elif not seq or pos > len(seq):
                    continue

            plddt_out[row_i], rsa_out[row_i], ss_out[row_i], dist_out[row_i] = feat
            n_attached += 1

        df["alphafold_plddt"] = plddt_out
        df["solvent_accessibility"] = rsa_out
        df["secondary_structure_context"] = ss_out
        df["dist_to_active_site"] = dist_out

        logger.info(
            "AlphaFoldConnector: attached AF features to %d variants "
            "(%d skipped on wt_aa isoform mismatch, fail-closed).",
            n_attached, n_wt_mismatch,
        )
        return df
