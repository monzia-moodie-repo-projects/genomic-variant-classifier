"""
src/genomic_variant_classifier/data/eve.py
===============
EVE evolutionary model connector — Phase 4, Connector 5.

Reads EVE per-protein CSV files (or a merged parquet) and adds one
variant-level feature:

    eve_score   float (0-1)
        Higher = more pathogenic (EVE pathogenicity score).
        0.5 = not covered / ambiguous (EVE uses 0.5 as the uncertain midpoint).

EVE lookup:
    Variants are matched by gene_symbol + one-letter amino acid change
    (e.g. protein_change "p.Arg175His" → "R175H").

Constructor:
    eve_path can be:
    - A directory containing per-protein CSV files named <GENE>_HUMAN_*.csv
    - A merged parquet file with all proteins combined
    - None → stub mode (all variants receive eve_score=0.5)

Per-protein CSV columns (EVE format):
    mutations_protein_name   str   Protein name (e.g. "TP53_HUMAN")
    position                 int   Amino acid position (1-indexed)
    wt_aa                    str   Wild-type amino acid (one-letter)
    mt_aa                    str   Mutant amino acid (one-letter)
    EVE_scores_ASM           float EVE pathogenicity score
    EVE_classes_25_pct_retained  str  Classification

Default: eve_score = 0.5 (ambiguous / not covered).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.database_connectors import BaseConnector, FetchConfig
from genomic_variant_classifier.data.gene_symbols import (
    gene_symbol_candidates,
    normalize_gene_symbol,
)

logger = logging.getLogger(__name__)

DEFAULT_SCORE = 0.5

# EVE per-protein files are named by UniProt ENTRY NAME (e.g. "1433G_HUMAN"), not by
# HGNC symbol. The cohort keys on HGNC ("YWHAG"), so the filename stem must be resolved
# entry-name -> HGNC before keying the lookup, else the join silently misses (eve_score
# stays 0.5 everywhere). The map comes from the UniProt index parquet's entry_name column
# (built by scripts/build_uniprot_index.py). Below this fraction of files resolving via
# the map, the connector logs a LOUD warning rather than silently keying on entry names.
_EVE_MIN_RESOLVED_FRACTION = 0.80


def _eve_stem_to_entry_name(stem: object) -> str:
    """EVE filename stem -> UniProt entry name.
    "1433G_HUMAN" -> "1433G_HUMAN"; "TP53_HUMAN_singles_scores" -> "TP53_HUMAN".
    Returns "" for null-ish input."""
    s = "" if stem is None else str(stem).strip().upper()
    if not s:
        return ""
    i = s.find("_HUMAN")
    return s[: i + len("_HUMAN")] if i != -1 else s


def load_eve_entry_map(parquet_path: object) -> dict[str, str]:
    """Build {ENTRY_NAME_UPPER: HGNC_UPPER} from the UniProt index parquet.
    Returns {} if the path is missing or the parquet lacks an entry_name column
    (e.g. an index built before entry_name was added); the caller logs that loudly."""
    if parquet_path is None:
        return {}
    p = Path(parquet_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception as exc:  # pragma: no cover - I/O guard
        logger.warning("EVEConnector: failed to read entry-name map %s: %s", p, exc)
        return {}
    if "entry_name" not in df.columns or "gene_symbol" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for en, g in zip(df["entry_name"].astype(str), df["gene_symbol"].astype(str)):
        en = en.strip().upper()
        g = g.strip().upper()
        if en and g and en not in out:
            out[en] = g
    return out


def resolve_eve_gene(stem: object, entry_map: dict[str, str]) -> tuple[str, bool]:
    """Resolve an EVE filename stem to (HGNC_symbol, resolved_via_map).
    Falls back to the legacy prefix-before-"_" (so HGNC-named files still work),
    with resolved_via_map=False so callers can count + fail loud on mass misses."""
    entry = _eve_stem_to_entry_name(stem)
    hgnc = entry_map.get(entry) if entry else None
    if hgnc:
        return normalize_gene_symbol(hgnc), True
    return normalize_gene_symbol(str(stem).split("_")[0]), False


# Standard one-letter amino acid codes for three-letter → one-letter mapping
_THREE_TO_ONE: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}


def _hgvsp_to_eve_key(protein_change: str) -> Optional[str]:
    """
    Convert HGVSp string to EVE lookup key "<WT><pos><MT>".

    Examples:
        "p.Arg175His" → "R175H"
        "p.Gly12Val"  → "G12V"
        "p.Arg175*"   → None   (stop gained, not a missense)
        ""            → None
    """
    if not protein_change or not isinstance(protein_change, str):
        return None

    # Match pattern like p.Arg175His
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)", protein_change)
    if m:
        wt3  = m.group(1)
        pos  = m.group(2)
        mt3  = m.group(3)
        wt1  = _THREE_TO_ONE.get(wt3)
        mt1  = _THREE_TO_ONE.get(mt3, mt3 if mt3 == "*" else None)
        if wt1 and mt1 and mt1 != "*":
            return f"{wt1}{pos}{mt1}"
        return None

    # Match single-letter format like p.R175H
    m = re.match(r"p\.([A-Z])(\d+)([A-Z])", protein_change)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    return None


class EVEConnector(BaseConnector):
    """
    Annotates variants with EVE evolutionary model pathogenicity scores.

    Usage
    -----
        connector = EVEConnector(eve_path="data/external/eve/")
        annotated_df = connector.annotate_dataframe(variant_df)
        # annotated_df now has an eve_score column (default 0.5)

    eve_path may be a directory of per-protein CSV files or a merged parquet.
    If eve_path is None or absent, stub mode applies: all variants get 0.5.
    """

    source_name = "eve"

    def __init__(
        self,
        eve_path: Optional[str | Path] = None,
        config: Optional[FetchConfig] = None,
        entry_map_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        self.eve_path: Optional[Path] = (
            Path(eve_path) if eve_path is not None else None
        )
        # UniProt index parquet (entry_name column) for entry-name -> HGNC resolution.
        self.entry_map_path: Optional[Path] = (
            Path(entry_map_path) if entry_map_path is not None else None
        )
        self._last_csv_resolved: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add eve_score column to df.

        Parameters
        ----------
        df : pd.DataFrame
            Variant DataFrame; must contain 'gene_symbol' and 'protein_change'.

        Returns
        -------
        pd.DataFrame with eve_score column added (default 0.5).
        """
        if df.empty:
            result = df.copy()
            result["eve_score"] = pd.Series(dtype=float)
            return result

        if self.eve_path is None:
            logger.warning(
                "EVEConnector: eve_path not set — returning eve_score=0.5 (not covered).  "
                "Download EVE scores from https://evemodel.org."
            )
            result = df.copy()
            result["eve_score"] = DEFAULT_SCORE
            return result

        lookup = self._get_lookup()
        if lookup.empty:
            result = df.copy()
            result["eve_score"] = DEFAULT_SCORE
            return result

        return self._annotate(df, lookup)

    def fetch(self, variant_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Wraps annotate_dataframe for BaseConnector compatibility."""
        return self.annotate_dataframe(variant_df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lookup(self) -> pd.DataFrame:
        """Return EVE lookup DataFrame (gene_symbol + aa_change → eve_score)."""
        stat = self.eve_path.stat() if self.eve_path and self.eve_path.exists() else None
        cache_basis = f"{self.eve_path.resolve()}|{stat.st_size if stat else 'missing'}|{stat.st_mtime_ns if stat else 'missing'}"
        cache_key = "eve_lookup_" + hashlib.sha256(cache_basis.encode("utf-8")).hexdigest()[:16]
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("EVEConnector: loaded %d EVE scores from cache.", len(cached))
            return cached

        if not self.eve_path.exists():
            logger.warning(
                "EVEConnector: eve_path '%s' does not exist — returning eve_score=0.5.",
                self.eve_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        # Determine whether eve_path is a directory of CSVs or a merged parquet
        if self.eve_path.is_dir():
            lookup = self._parse_csv_directory(self.eve_path)
        elif self.eve_path.suffix in (".parquet", ".pq"):
            lookup = self._parse_merged_parquet(self.eve_path)
        else:
            logger.warning(
                "EVEConnector: eve_path '%s' is not a directory or parquet — "
                "returning eve_score=0.5.",
                self.eve_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        if not lookup.empty:
            self._save_cache(cache_key, lookup)
            logger.info("EVEConnector: cached %d EVE scores.", len(lookup))
        return lookup

    def _parse_csv_directory(self, directory: Path) -> pd.DataFrame:
        """Parse a directory of per-protein EVE CSV files."""
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            logger.warning(
                "EVEConnector: no CSV files found in directory '%s'.", directory
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        # Build the entry-name -> HGNC map once; pass it to every per-file parse so
        # filenames (UniProt entry names) resolve to the cohort's HGNC symbols.
        entry_map = load_eve_entry_map(self.entry_map_path)
        n_resolved = 0
        n_fallback = 0
        unresolved_sample: list[str] = []
        parts = []
        for csv_file in csv_files:
            try:
                self._last_csv_resolved = False
                part = self._parse_single_csv(csv_file, entry_map)
                if self._last_csv_resolved:
                    n_resolved += 1
                else:
                    n_fallback += 1
                    if len(unresolved_sample) < 5:
                        unresolved_sample.append(csv_file.stem)
                if not part.empty:
                    parts.append(part)
            except Exception as exc:
                logger.warning(
                    "EVEConnector: failed to parse %s: %s", csv_file.name, exc
                )
        # Fail loud if too few files resolved entry-name -> HGNC (stale/missing index).
        _total = n_resolved + n_fallback
        _frac = (n_resolved / _total) if _total else 0.0
        if not entry_map:
            logger.warning(
                "EVEConnector: entry-name map empty (entry_map_path=%s); keying EVE by "
                "filename prefix -> near-zero HGNC coverage. Rebuild the UniProt index "
                "with the entry_name column (scripts/build_uniprot_index.py).",
                self.entry_map_path,
            )
        elif _frac < _EVE_MIN_RESOLVED_FRACTION:
            logger.warning(
                "EVEConnector: only %d/%d (%.1f%%) EVE files resolved entry-name -> HGNC "
                "(min %.0f%%). Sample unresolved: %s. Check the UniProt index entry_name column.",
                n_resolved, _total, 100 * _frac, 100 * _EVE_MIN_RESOLVED_FRACTION, unresolved_sample,
            )
        else:
            logger.info(
                "EVEConnector: resolved %d/%d (%.1f%%) EVE files entry-name -> HGNC.",
                n_resolved, _total, 100 * _frac,
            )

        if not parts:
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        combined = pd.concat(parts, ignore_index=True)
        logger.info(
            "EVEConnector: parsed %d CSVs → %d EVE scores.", len(csv_files), len(combined)
        )
        return combined

    def _parse_single_csv(self, csv_file: Path, entry_map: Optional[dict] = None) -> pd.DataFrame:
        """Parse a single per-protein EVE CSV file."""
        raw = pd.read_csv(csv_file, dtype=str)
        raw.columns = [c.strip() for c in raw.columns]

        required = {"position", "wt_aa", "mt_aa", "EVE_scores_ASM"}
        if not required.issubset(raw.columns):
            logger.warning(
                "EVEConnector: %s missing required columns %s (found: %s).",
                csv_file.name, required - set(raw.columns), list(raw.columns),
            )
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        raw["position"]       = pd.to_numeric(raw["position"], errors="coerce")
        raw["EVE_scores_ASM"] = pd.to_numeric(raw["EVE_scores_ASM"], errors="coerce")
        raw = raw.dropna(subset=["position", "EVE_scores_ASM"])

        # Extract gene symbol from protein name when present, otherwise resolve the
        # filename entry-name (e.g. 1433G_HUMAN) to its HGNC symbol via the UniProt map.
        if "mutations_protein_name" in raw.columns:
            raw["gene_symbol"] = raw["mutations_protein_name"].astype(str).str.split("_").str[0]
            self._last_csv_resolved = True
        else:
            _hgnc, _resolved = resolve_eve_gene(csv_file.stem, entry_map or {})
            raw["gene_symbol"] = _hgnc
            self._last_csv_resolved = _resolved

        raw["aa_change"] = (
            raw["wt_aa"].str.strip() +
            raw["position"].astype(int).astype(str) +
            raw["mt_aa"].str.strip()
        )
        raw["eve_score"] = raw["EVE_scores_ASM"].clip(0.0, 1.0)

        return raw[["gene_symbol", "aa_change", "eve_score"]].copy()

    def _parse_merged_parquet(self, parquet_path: Path) -> pd.DataFrame:
        """Parse a merged EVE parquet file."""
        try:
            raw = pd.read_parquet(parquet_path)
        except Exception as exc:
            logger.error("EVEConnector: failed to read parquet %s: %s", parquet_path, exc)
            return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

        # Support both per-protein CSV column layout and pre-processed layout
        if "gene_symbol" in raw.columns and "aa_change" in raw.columns and "eve_score" in raw.columns:
            return raw[["gene_symbol", "aa_change", "eve_score"]].dropna().copy()

        if "mutations_protein_name" in raw.columns:
            # Treat as merged per-protein CSV
            tmp_path = parquet_path  # reuse the path reference
            raw["gene_symbol"] = raw["mutations_protein_name"].str.split("_").str[0]
            raw["position"]       = pd.to_numeric(raw.get("position", pd.Series()), errors="coerce")
            raw["EVE_scores_ASM"] = pd.to_numeric(raw.get("EVE_scores_ASM", pd.Series()), errors="coerce")
            raw = raw.dropna(subset=["position", "EVE_scores_ASM"])
            raw["aa_change"] = (
                raw["wt_aa"].str.strip() +
                raw["position"].astype(int).astype(str) +
                raw["mt_aa"].str.strip()
            )
            raw["eve_score"] = raw["EVE_scores_ASM"].clip(0.0, 1.0)
            return raw[["gene_symbol", "aa_change", "eve_score"]].copy()

        logger.error(
            "EVEConnector: parquet %s does not have expected columns.", parquet_path
        )
        return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"])

    def _annotate(self, variant_df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
        """Left-join EVE scores onto variant_df by gene_symbol + aa_change."""
        result = variant_df.copy()

        # Derive aa_change for the EVE join. The lookup side builds its key as
        # wt_aa + position + mt_aa (see _parse_single_csv); build the SAME key on
        # the variant side from the populated coordinate triple
        # (wt_aa / protein_pos / mut_aa, filled by AlphaMissense step 10b) FIRST,
        # then fall back to parsing protein_change (HGVSp) for cohorts that carry
        # it instead. Coordinate-first is load-bearing: the whole-genome ClinVar
        # cohort has protein_change 100% null, so a protein_change-only key left
        # eve_score at 0.5 for every variant despite coords being present.
        def _eve_key_from_triple(_wt: object, _pos: object, _mut: object) -> Optional[str]:
            if _wt is None or _mut is None or pd.isna(_pos):
                return None
            _wt_s = str(_wt).strip()
            _mut_s = str(_mut).strip()
            if not _wt_s or not _mut_s:
                return None
            try:
                return f"{_wt_s}{int(_pos)}{_mut_s}"
            except (TypeError, ValueError):
                return None

        if {"wt_aa", "protein_pos", "mut_aa"}.issubset(result.columns):
            _triple_key = [
                _eve_key_from_triple(_w, _p, _m)
                for _w, _p, _m in zip(
                    result["wt_aa"], result["protein_pos"], result["mut_aa"]
                )
            ]
        else:
            _triple_key = [None] * len(result)

        protein_change = result.get(
            "protein_change",
            pd.Series([""] * len(result), index=result.index),
        ).fillna("")
        _hgvsp_key = protein_change.map(_hgvsp_to_eve_key)

        # Coordinate triple wins where present; HGVSp fills the remaining rows.
        result["_aa_change"] = [
            _t if _t is not None else _h
            for _t, _h in zip(_triple_key, _hgvsp_key)
        ]

        gene_symbol = result.get(
            "gene_symbol",
            pd.Series([""] * len(result), index=result.index),
        ).fillna("")

        # Only attempt to join for rows with a valid aa_change
        has_key = result["_aa_change"].notna()

        score_table = lookup.rename(
            columns={"gene_symbol": "_gene_symbol", "aa_change": "_aa_change"}
        )
        # Normalize the lookup gene key so case/whitespace never blocks a
        # match, and drop unusable empty-gene rows so they cannot spuriously
        # match an empty variant gene_symbol.
        score_table["_gene_symbol"] = score_table["_gene_symbol"].map(
            normalize_gene_symbol
        )
        score_table = score_table[score_table["_gene_symbol"] != ""]
        _lookup_genes = set(score_table["_gene_symbol"])

        def _resolve_gene(_raw: object) -> str:
            # First candidate present in the EVE lookup wins: recovers
            # semicolon-joined multi-gene symbols and fixes case drift.
            # Never splits on "-".
            for _cand in gene_symbol_candidates(_raw):
                if _cand in _lookup_genes:
                    return _cand
            return normalize_gene_symbol(_raw)

        result["_gene_symbol"] = gene_symbol.map(_resolve_gene)

        result = result.merge(
            score_table,
            on=["_gene_symbol", "_aa_change"],
            how="left",
        )

        # Rows without a valid key get the default
        result["eve_score"] = result["eve_score"].fillna(DEFAULT_SCORE).clip(0.0, 1.0)
        result = result.drop(columns=["_aa_change", "_gene_symbol"])

        n_covered = (result["eve_score"] != DEFAULT_SCORE).sum()
        logger.info(
            "EVEConnector: %d / %d variants covered by EVE (score != default).",
            n_covered, len(result),
        )
        return result
