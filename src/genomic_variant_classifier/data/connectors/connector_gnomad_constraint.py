"""
src/genomic_variant_classifier/data/connectors/connector_gnomad_constraint.py
============================================================================
gnomAD v4.1 gene constraint connector.

Adds five gene-level constraint metrics to the variant feature matrix:

    pli_score           Probability of loss-of-function intolerance (0-1).
    loeuf               lof.oe_ci.upper -- the UPPER BOUND of the 90 per cent
                        confidence interval around the loss-of-function
                        observed/expected ratio.
    gene_constraint_oe  lof.oe -- the observed/expected ratio ITSELF.
    syn_z               Synonymous variant Z-score.
    mis_z               Missense variant Z-score.

WHAT CHANGED, AND WHY -- DUPLICATE-1A / CONSTRAINTTRANSCRIPT-1 / CACHEIDENTITY-1
================================================================================
Three defects were measured in the previous version of this file on 2026-08-09.

1. `gene_constraint_oe` DID NOT EXIST HERE.
   The connector emitted `loeuf` and nothing else resembling an
   observed/expected ratio, so `variant_ensemble.engineer_features` fell back
   to `df.get("gene_constraint_oe", df.get("loeuf", ...))` and the two model
   features became BIT-IDENTICAL:

       identical = True   max abs diff = 0.0   correlation = 1.0

   LOEUF is the upper bound of a confidence interval around the ratio; it is
   not the ratio. The source publishes both, two columns apart. `lof.oe` is
   now extracted, and its arithmetic is asserted against `lof.obs / lof.exp`.

2. TRANSCRIPT SELECTION WAS ARBITRARY.
   The old parser read all 211,523 source rows -- a median of 8 per gene, up
   to 201 -- and kept `drop_duplicates(subset=["gene"], keep="first")`, with a
   comment asserting "first = MANE transcript in gnomAD ordering". No
   `mane_select` filter existed anywhere in the file. Measured: first-row
   selection disagrees with MANE Select for 5,468 of 17,473 genes (31.3%),
   median absolute LOEUF difference 0.039, maximum 1.689, and 132 genes cross
   the 0.35 constrained boundary. Selection now runs through
   `constraint_canonicalize`, which applies a declared tier ladder
   (MANE Select -> canonical -> missing) and refuses to resolve ambiguity by
   source row order.

3. MISSINGNESS WAS FABRICATED AT THREE LEVELS.
   `_safe_float(..., 1.0)` at parse time, `ConstraintScores`'s class defaults,
   and `.fillna(CONSTRAINT_DEFAULTS[col])` in `annotate_dataframe` each
   replaced an absent measurement with a biological assertion. A LOEUF of 1.0
   means "completely tolerant of loss of function"; that is a claim, not a
   placeholder. All three now yield NaN. Downstream model adaptation owns
   imputation under a declared policy -- see CONSTRAINTFILL-1.

Two silent coercions were also removed. `df["loeuf"].clip(0.0, 5.0)` had never
altered a value: the observed maximum across 18,204 cached genes is 1.9970.
Ranges are now VALIDATED and never silently coerced.

CACHE IDENTITY
--------------
The old `_cache_path()` derived only from the source filename, so a sidecar
built by the defective parser would be loaded in preference to a repaired one
-- perfectly correct source code, still serving the old semantics. Cache
identity is now (source SHA-256, schema version, canonicalisation policy),
persisted beside the parquet and verified on load. Any mismatch rebuilds.

Stub mode
---------
When `tsv_path` is None or absent, every feature is NaN. Not zero, not 1.0 --
NaN, because "the connector was not given data" and "this gene is unconstrained"
are different statements and the matrix must be able to tell them apart.

Author: Monzia Moodie
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from genomic_variant_classifier.data.constraint_canonicalize import (
    COL_GENE,
    COL_LOEUF,
    COL_LOF_OE,
    ConstraintSourceError,
    canonicalize_mane_constraint,
    sha256_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Source column names, exactly as gnomAD publishes them.
_COL_PLI = "lof.pLI"
_COL_SYN_Z = "syn.z_score"
_COL_MIS_Z = "mis.z_score"

# Feature columns exposed to the rest of the pipeline. gene_constraint_oe is
# NEW: it is the observed/expected point estimate, distinct from loeuf.
CONSTRAINT_COLS: list[str] = [
    "pli_score", "loeuf", "gene_constraint_oe", "syn_z", "mis_z",
]

# NaN, deliberately. A default that is a plausible biological value cannot be
# distinguished from a measurement, and that indistinguishability is the whole
# of CONSTRAINTFILL-1.
CONSTRAINT_DEFAULTS: dict[str, float] = {c: float("nan") for c in CONSTRAINT_COLS}

# Columns ingested for SOURCE VALIDATION only. They are never model features:
# validation information does not automatically become predictive information.
AUDIT_COLS: list[str] = ["lof.obs", "lof.exp", "_tier", "_n_source_representations"]

# Published ranges. VALIDATED, never coerced. pLI is a probability, so its
# range is definitional. No upper bound is invented for LOEUF or the ratio:
# twelve well-powered genes carry lof.oe above 2.0, and clipping them would
# manufacture a statistic gnomAD never published.
_RANGE_CHECKS = {"pli_score": (0.0, 1.0)}

DEFAULT_TSV_PATH = Path("data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv")
_CACHE_SUFFIX = ".constraint_index.parquet"
_CACHE_META_SUFFIX = ".constraint_index.meta.json"

# Bump when the parser's output SEMANTICS change, so old sidecars are refused.
CONSTRAINT_INDEX_SCHEMA_VERSION = 3
CANONICALIZATION_POLICY = "mane_then_canonical_namespace_collapse_v1"


@dataclass(frozen=True)
class ConstraintCacheIdentity:
    """WHAT this cache is, as opposed to whether its bytes survived.

    `allow_canonical_fallback` is part of the IDENTITY, not a runtime option.
    Measured 2026-08-09: without it, a connector constructed with
    allow_canonical_fallback=False accepted a cache built with True and
    silently received the MANE-plus-canonical index -- returning 0.9 for a
    canonical-tier gene where the requested policy demands NaN. Same source
    digest, same schema version, same policy string, different science.

    That is CACHEIDENTITY-1's own principle violated inside the fix for
    CACHEIDENTITY-1: a cache key that omits a parameter which changes the
    output is not an identity.
    """
    source_sha256: str
    schema_version: int
    canonicalization_policy: str
    allow_canonical_fallback: bool

    def as_dict(self) -> dict:
        return {"source_sha256": self.source_sha256,
                "schema_version": self.schema_version,
                "canonicalization_policy": self.canonicalization_policy,
                "allow_canonical_fallback": self.allow_canonical_fallback}

    def matches(self, other: dict) -> tuple[bool, str]:
        if not isinstance(other, dict):
            return False, "sidecar metadata is not a mapping"
        for key, mine in self.as_dict().items():
            theirs = other.get(key)
            if theirs != mine:
                return False, "{} differs: cached {!r}, current {!r}".format(
                    key, theirs, mine)
        return True, ""


@dataclass(frozen=True)
class ConstraintScores:
    """gnomAD constraint metrics for a single gene. Absent means NaN."""
    pli_score: float = float("nan")
    loeuf: float = float("nan")
    gene_constraint_oe: float = float("nan")
    syn_z: float = float("nan")
    mis_z: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {c: getattr(self, c) for c in CONSTRAINT_COLS}


class GnomADConstraintConnector:
    """Gene-level gnomAD v4.1 constraint connector, keyed on HGNC gene symbol."""

    source_name = "gnomad_constraint"

    def __init__(self, tsv_path: str | Path | None = None,
                 cache_dir: str | Path | None = None,
                 allow_canonical_fallback: bool = True) -> None:
        self._tsv_path: Path | None = Path(tsv_path) if tsv_path else None
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir
            else (self._tsv_path.parent if self._tsv_path else None))
        self._allow_canonical_fallback = bool(allow_canonical_fallback)
        self._index: pd.DataFrame | None = None
        self._audit: dict | None = None
        self._coverage: dict | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def coverage(self) -> dict | None:
        """Match counts from the most recent annotate_dataframe call."""
        return self._coverage

    @property
    def audit(self) -> dict | None:
        """Canonicalisation evidence from the most recent index build."""
        return self._audit

    def get_scores(self, gene_symbol: str) -> ConstraintScores:
        if not self._usable():
            return ConstraintScores()
        self._ensure_index()
        key = str(gene_symbol).strip()
        hit = self._index[self._index[COL_GENE] == key]      # type: ignore[index]
        if hit.empty:
            return ConstraintScores()
        row = hit.iloc[0]
        return ConstraintScores(**{c: float(row[c]) for c in CONSTRAINT_COLS})

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the five constraint columns to df. Unmatched genes get NaN.

        One MERGE, not four per-row lambda passes. The previous form invoked a
        Python callable once per row per column -- roughly 17.6 million calls
        on a 4.4-million-row cohort.
        """
        df = df.copy()
        for col in CONSTRAINT_COLS:
            if col not in df.columns:
                df[col] = np.nan

        if COL_GENE + "_symbol" not in df.columns and "gene_symbol" not in df.columns:
            logger.info("gnomAD constraint: gene_symbol absent -- all values NaN.")
            return df
        if not self._usable():
            logger.warning(
                "gnomAD constraint: no source at %s -- all values NaN. This is "
                "MISSING data, not unconstrained genes.", self._tsv_path)
            return df

        self._ensure_index()
        idx = self._index[[COL_GENE] + CONSTRAINT_COLS]      # type: ignore[index]
        keys = df["gene_symbol"].astype(str).str.strip()
        merged = pd.DataFrame({COL_GENE: keys}).merge(
            idx, on=COL_GENE, how="left", validate="many_to_one")
        for col in CONSTRAINT_COLS:
            df[col] = merged[col].to_numpy()

        # Coverage by INDEX MEMBERSHIP. The previous form counted
        # `pli_score != 0.0`, so a gene with a genuine pLI of exactly zero was
        # logged as a miss and every coverage figure was understated.
        n_hit = int(keys.isin(set(idx[COL_GENE])).sum())
        # RECORDED, not merely logged. A figure that exists only in a log line
        # cannot be asserted, and a sabotage run on 2026-08-09 reverted this to
        # `(pli_score != 0.0).sum()` -- the old, understating form -- with no
        # test noticing. Coverage is evidence; evidence must be observable.
        self._coverage = {
            "n_rows": int(len(df)),
            "n_matched": n_hit,
            "n_unmatched": int(len(df)) - n_hit,
            "n_genes_in_index": int(len(idx)),
        }
        logger.info("gnomAD constraint: %d / %d variants matched a gene entry "
                    "(%d gene(s) in the index).", n_hit, len(df), len(idx))
        return df

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _usable(self) -> bool:
        return self._tsv_path is not None and self._tsv_path.exists()

    def _identity(self) -> ConstraintCacheIdentity:
        return ConstraintCacheIdentity(
            source_sha256=sha256_file(str(self._tsv_path)),
            schema_version=CONSTRAINT_INDEX_SCHEMA_VERSION,
            canonicalization_policy=CANONICALIZATION_POLICY,
            allow_canonical_fallback=self._allow_canonical_fallback)

    def _cache_paths(self) -> tuple[Path, Path] | tuple[None, None]:
        if self._tsv_path is None or self._cache_dir is None:
            return None, None
        stem = self._tsv_path.name.split(".tsv")[0]
        return (self._cache_dir / f"{stem}{_CACHE_SUFFIX}",
                self._cache_dir / f"{stem}{_CACHE_META_SUFFIX}")

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        cache, meta_path = self._cache_paths()
        identity = self._identity()

        if cache is not None and cache.exists():
            ok, why = False, "no sidecar metadata"
            if meta_path is not None and meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        ok, why = identity.matches(json.load(fh))
                except (OSError, ValueError) as exc:
                    ok, why = False, "sidecar unreadable: {}".format(exc)
            if ok:
                # IDENTITY and INTEGRITY are different contracts.
                #   identity  -- was this produced under the right source/policy?
                #   integrity -- are these still the bytes that were produced?
                # A cache with valid metadata and edited bytes passed the old
                # check. Both are now required.
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    want = meta.get("cache_sha256")
                    got = sha256_file(str(cache))
                    if want is None:
                        ok, why = False, "sidecar carries no cache_sha256"
                    elif want != got:
                        ok, why = False, ("cached parquet digest mismatch: "
                                          "sidecar {} on disk {}".format(want[:16], got[:16]))
                except (OSError, ValueError) as exc:
                    ok, why = False, "sidecar unreadable during integrity check: {}".format(exc)
            if ok:
                logger.info("gnomAD constraint: loading verified cache %s", cache)
                self._index = pd.read_parquet(cache)
                return
            # A cache built by a different parser must NEVER be preferred to a
            # repaired one. CACHEIDENTITY-1: the previous key was the source
            # FILENAME, so a stale sidecar bypassed the parser entirely.
            logger.warning(
                "gnomAD constraint: REBUILDING -- cached index rejected (%s). "
                "A cache keyed only on filename would have served the old "
                "semantics from correct source code.", why)

        raw = pd.read_csv(self._tsv_path, sep="\t", low_memory=False)
        logger.info("gnomAD constraint: parsed %d source row(s) from %s",
                    len(raw), self._tsv_path)
        index, audit = canonicalize_mane_constraint(
            raw, allow_canonical_fallback=self._allow_canonical_fallback,
            source_path=str(self._tsv_path), source_sha256=identity.source_sha256)
        index = _to_feature_frame(index)
        _validate_ranges(index)
        self._index = index
        self._audit = audit.as_dict()
        logger.info("gnomAD constraint index: %d gene(s); tiers %s",
                    len(index), self._audit.get("tier_counts"))

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # CRASH-SAFE PUBLICATION. Write both to temporaries, then move the
            # DATA into place first and the MANIFEST last. An interruption can
            # then leave a cache with no manifest -- which is rejected -- but
            # never a manifest vouching for bytes that were never written. This
            # is the cache equivalent of the rollback discipline the installers
            # apply to source edits.
            tmp_cache = cache.with_suffix(cache.suffix + ".tmp")
            tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
            index.to_parquet(tmp_cache, index=False)
            metadata = {
                **identity.as_dict(),
                "cache_sha256": sha256_file(str(tmp_cache)),
                "n_rows": int(len(index)),
                "columns": list(index.columns),
                "audit": self._audit,
            }
            with open(tmp_meta, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(metadata, fh, indent=2, default=str)
            os.replace(tmp_cache, cache)
            os.replace(tmp_meta, meta_path)
            logger.info("gnomAD constraint: cache published atomically "
                        "(%d rows, digest %s)",
                        metadata["n_rows"], metadata["cache_sha256"][:16])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_feature_frame(index: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to feature names. No fills, no clips."""
    rename = {_COL_PLI: "pli_score", COL_LOEUF: "loeuf",
              COL_LOF_OE: "gene_constraint_oe", _COL_SYN_Z: "syn_z",
              _COL_MIS_Z: "mis_z"}
    out = pd.DataFrame({COL_GENE: index[COL_GENE].astype(str).str.strip()})
    for src, dst in rename.items():
        out[dst] = (pd.to_numeric(index[src], errors="coerce")
                    if src in index.columns else np.nan)
    for extra in AUDIT_COLS:
        if extra in index.columns:
            out[extra] = index[extra].to_numpy()
    return out.reset_index(drop=True)


def _validate_ranges(index: pd.DataFrame) -> None:
    """Assert published ranges. NEVER coerce.

    The previous parser clipped pli_score to [0, 1] and loeuf to [0, 5]. The
    loeuf clip had never altered a value -- the observed maximum is 1.9970 --
    and a clip that has never fired is an undocumented coercion waiting to
    silently rewrite a future release's data.
    """
    for col, (lo, hi) in _RANGE_CHECKS.items():
        if col not in index.columns:
            continue
        s = pd.to_numeric(index[col], errors="coerce")
        bad = s.notna() & ((s < lo) | (s > hi))
        if bool(bad.any()):
            offenders = index.loc[bad, [COL_GENE, col]].head(10)
            raise ConstraintSourceError(
                "gnomAD constraint: {} value(s) of {!r} outside the published "
                "range [{}, {}]. Validated, not coerced.\n{}".format(
                    int(bad.sum()), col, lo, hi, offenders.to_string(index=False)))
