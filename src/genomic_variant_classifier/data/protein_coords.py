"""Protein-coordinate connector (AlphaMissense -> protein_pos / wt_aa / mut_aa).

ESM-2 (and EVE) need a 1-based ``protein_pos`` plus single-letter ``wt_aa`` /
``mut_aa`` per missense variant. None of the existing connectors surface these,
which is why ``esm2_delta_norm`` has been a silent zero across runs. AlphaMissense's
``protein_variant`` field (e.g. ``V123M``) carries exactly that triple, and because
AlphaMissense is defined against the UniProt canonical sequence, those positions
match the sequence ESM-2 fetches -- satisfying the ``esm2.py`` wt_aa cross-check at
a high rate (RefSeq/HGVSp coordinates would not).

This connector streams ``AlphaMissense_hg38.tsv.gz`` once, keeps only the rows whose
``(chrom, pos, ref, alt)`` appear in the cohort, parses ``protein_variant`` into the
triple, and caches a small index parquet. ``annotate_dataframe`` left-joins the three
columns on a normalised key (AlphaMissense uses ``chr1``; the cohort uses ``1``).

Degradation contract (matches the sibling connectors -- dbnsfp/spliceai/omim all
warn-and-stub on a missing path): when there is no data source at all (no cached
index and no AlphaMissense file), ``annotate_dataframe`` warns and returns the frame
unchanged, so ESM-2/EVE fall back to their stub paths and the annotation pipeline
never crashes. Fail-loud is reserved for the one case that is a genuine silent-zero
risk: a *present* file whose header is malformed, or whose matched ``protein_variant``
parse rate drops below ``_MIN_PARSE_RATE`` (format drift) -- the build raises rather
than caching a near-empty index.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from genomic_variant_classifier.data.hgvsp_parser import parse_am_protein_variant

logger = logging.getLogger(__name__)

_CACHE_NAME = "alphamissense_protein_index.parquet"
_CHUNK = 2_000_000
_MIN_PARSE_RATE = 0.90
_INDEX_COLS = ["_c", "_p", "_r", "_a", "protein_pos", "wt_aa", "mut_aa"]


def _norm_chrom(s: pd.Series) -> pd.Series:
    """Strip a leading 'chr' (any case) and upper-case; '15'->'15', 'chrX'->'X'."""
    return s.astype(str).str.replace(r"(?i)^chr", "", regex=True).str.upper()


def _norm_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    c = _norm_chrom(df["chrom"])
    p = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    r = df["ref"].astype(str).str.upper()
    a = df["alt"].astype(str).str.upper()
    return c, p, r, a


def _detect_header_skip_and_cols(path: Path) -> tuple[int, dict[str, str]]:
    """Find the header row (after the '#' comment lines) and locate the columns
    we need by normalised name. Returns (rows_to_skip, {logical: actual_name})."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:  # type: ignore[operator]
        for i, line in enumerate(f):
            cells = line.rstrip("\n").split("\t")
            norm = [c.lstrip("#").strip().upper() for c in cells]
            if "CHROM" in norm and "POS" in norm and "PROTEIN_VARIANT" in norm:
                want = {"CHROM": "chrom", "POS": "pos", "REF": "ref",
                        "ALT": "alt", "PROTEIN_VARIANT": "protein_variant"}
                mapping: dict[str, str] = {}
                for actual, n in zip(cells, norm):
                    if n in want:
                        mapping[want[n]] = actual
                missing = set(want.values()) - set(mapping)
                if missing:
                    raise ValueError(
                        f"AlphaMissense header missing {missing}; saw {cells}"
                    )
                return i, mapping
    raise ValueError(f"AlphaMissense header row not found in {path}")


class ProteinCoordConnector:
    def __init__(
        self,
        alphamissense_file: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self._am = Path(alphamissense_file) if alphamissense_file else None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        elif self._am is not None:
            self._cache_dir = self._am.parent
        else:
            self._cache_dir = Path(".")
        self._index: Optional[pd.DataFrame] = None

    @property
    def cache_path(self) -> Path:
        return self._cache_dir / _CACHE_NAME

    # -- index build / load --------------------------------------------------
    def _build_index(self, cohort: set[tuple]) -> pd.DataFrame:
        if self._am is None or not self._am.exists():
            raise FileNotFoundError(
                f"AlphaMissense file not found: {self._am}. "
                "Pass alphamissense_file=path/to/AlphaMissense_hg38.tsv.gz"
            )
        skip, cols = _detect_header_skip_and_cols(self._am)
        usecols = [cols["chrom"], cols["pos"], cols["ref"], cols["alt"], cols["protein_variant"]]
        rename = {v: k for k, v in cols.items()}
        matched: list[pd.DataFrame] = []
        reader = pd.read_csv(
            self._am, sep="\t", skiprows=skip, usecols=usecols,
            dtype=str, chunksize=_CHUNK,
        )
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            c, p, r, a = _norm_keys(chunk)
            keys = list(zip(c, p, r, a))
            mask = [k in cohort for k in keys]
            if not any(mask):
                continue
            sub = chunk.loc[mask].copy()
            sub["_c"], sub["_p"], sub["_r"], sub["_a"] = (
                c[mask].values, p[mask].values, r[mask].values, a[mask].values
            )
            matched.append(sub[["_c", "_p", "_r", "_a", "protein_variant"]])
        if not matched:
            logger.warning("ProteinCoord: zero cohort variants matched AlphaMissense.")
            return pd.DataFrame(columns=_INDEX_COLS)
        m = pd.concat(matched, ignore_index=True)
        nonnull = m["protein_variant"].notna().sum()
        parsed = [parse_am_protein_variant(v) for v in m["protein_variant"].to_numpy()]
        ok = sum(1 for t in parsed if t[0] is not None)
        rate = ok / max(int(nonnull), 1)
        if rate < _MIN_PARSE_RATE:
            raise ValueError(
                f"AlphaMissense protein_variant parse rate {rate:.3f} < {_MIN_PARSE_RATE}; "
                "format may have drifted -- refusing to build a near-empty index."
            )
        m["protein_pos"] = pd.array([t[0] if t[0] is not None else pd.NA for t in parsed], dtype="Int64")
        m["wt_aa"] = pd.array([t[1] for t in parsed], dtype="object")
        m["mut_aa"] = pd.array([t[2] for t in parsed], dtype="object")
        idx = (
            m[_INDEX_COLS]
            .dropna(subset=["protein_pos"])
            .drop_duplicates(subset=["_c", "_p", "_r", "_a"], keep="first")
            .reset_index(drop=True)
        )
        logger.info(
            "ProteinCoord: indexed %d/%d matched cohort variants (parse rate %.3f).",
            len(idx), len(m), rate,
        )
        return idx

    def _load_or_build_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._index is not None:
            return self._index
        if self.cache_path.exists():
            logger.info("ProteinCoord: loading index cache: %s", self.cache_path)
            self._index = pd.read_parquet(self.cache_path)
            return self._index
        c, p, r, a = _norm_keys(df)
        cohort = set(zip(c, p, r, a))
        idx = self._build_index(cohort)
        try:
            idx.to_parquet(self.cache_path, index=False)
            logger.info("ProteinCoord: wrote index cache -> %s", self.cache_path)
        except Exception as exc:  # cache is an optimisation; never fatal
            logger.warning("ProteinCoord: could not write cache (%s).", exc)
        self._index = idx
        return idx

    # -- annotate ------------------------------------------------------------
    @staticmethod
    def _stub_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["protein_pos"] = pd.array([pd.NA] * len(df), dtype="Int64")
        df["wt_aa"] = pd.array([None] * len(df), dtype="object")
        df["mut_aa"] = pd.array([None] * len(df), dtype="object")
        return df

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Graceful degradation, matching the other connectors (dbnsfp/spliceai/omim
        # all warn-and-stub on a missing path rather than raising): if there is
        # neither a cached index nor a usable AlphaMissense file, there is no data
        # source at all -- return the frame UNCHANGED (no new columns, pre-10b shape).
        # ESM-2/EVE handle absent protein columns via their documented stub path.
        # Raising here would crash the whole annotation pipeline whenever the file is
        # absent (unit tests, boxes without the 613 MB TSV). Fail-loud is reserved for
        # a *present* file whose protein_variant column has drifted (parse-rate guard
        # in _build_index), which is the only case that is a genuine silent-zero risk.
        if not self.cache_path.exists() and (self._am is None or not self._am.exists()):
            logger.warning(
                "ProteinCoord: no cached index and no AlphaMissense file (%s); "
                "leaving protein_pos/wt_aa/mut_aa unset (ESM-2/EVE will stub).",
                self._am,
            )
            return df
        idx = self._load_or_build_index(df)
        if idx is None or idx.empty:
            # Source was present but no cohort variant matched -> add all-NA columns
            # (consistent with the left-merge semantics of the populate path below).
            return self._stub_columns(df)
        c, p, r, a = _norm_keys(df)
        left = pd.DataFrame({"_c": c.values, "_p": p.values, "_r": r.values, "_a": a.values})
        merged = left.merge(idx, on=["_c", "_p", "_r", "_a"], how="left")
        df["protein_pos"] = pd.array(merged["protein_pos"].to_numpy(), dtype="Int64")
        df["wt_aa"] = merged["wt_aa"].to_numpy()
        df["mut_aa"] = merged["mut_aa"].to_numpy()
        return df


# ---------------------------------------------------------------------------
# Standalone build + coverage report (run before any wiring/regen)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build AlphaMissense protein-coord index + report coverage.")
    ap.add_argument("--alphamissense", required=True, help="AlphaMissense_hg38.tsv.gz")
    ap.add_argument("--clinvar", required=True, help="processed cohort parquet")
    ap.add_argument("--cache-dir", default=None, help="where to write the index parquet")
    args = ap.parse_args()

    cohort_df = pd.read_parquet(args.clinvar, columns=["chrom", "pos", "ref", "alt", "consequence"])
    pc = ProteinCoordConnector(alphamissense_file=args.alphamissense, cache_dir=args.cache_dir)
    out = pc.annotate_dataframe(cohort_df)
    mm = out["consequence"].fillna("").str.contains("missense", case=False)
    have = out["protein_pos"].notna()
    n_mm = int(mm.sum())
    print(f"cohort rows           : {len(out):,}")
    print(f"missense rows         : {n_mm:,}")
    print(f"rows with protein_pos : {int(have.sum()):,}")
    print(f"missense WITH coords  : {int((mm & have).sum()):,} "
          f"({100 * (mm & have).sum() / max(n_mm, 1):.1f}% of missense)")
    print(f"index cache           : {pc.cache_path}")
