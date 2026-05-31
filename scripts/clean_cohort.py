"""
clean_cohort.py - Phase 0 cohort de-leak for Run 15
===================================================
Resolves the null-key leak (INCIDENT_<date>_null-key-leak) and the duplicate /
label-conflict integrity problem in the source ClinVar cohort, BEFORE splits are
regenerated for Run 15. Operates at the cohort source so Run 15 trains on clean data.

INPUT  : data/processed/clinvar_grch38.parquet
OUTPUTS: data/processed/clinvar_grch38_clean.parquet        (0 null-key, 0 dup variant_id)
         data/processed/clinvar_grch38_structural.parquet   (null/bad ref or alt)
         data/processed/clinvar_grch38_conflicts.parquet    (irreducible label conflicts)
         data/processed/clean_cohort_reconciliation.json     (full audit; rows reconcile)

DESIGN PRINCIPLES (no silent failures, no guessing):
  * Introspects schema; auto-detects label / review-status columns from candidate sets
    and FAILS LOUD (lists actual columns) if it cannot identify the label column.
  * --audit (default) is a dry-run: prints schema, distributions, and the full
    reconciliation plan WITHOUT writing anything.
  * --apply writes outputs only after the reconciliation identity holds exactly.
  * Every source row is accounted for; the script raises if the arithmetic does not
    reconcile to the exact source row count.

USAGE (from project root, .venv312 active):
  python scripts/clean_cohort.py --audit
  python scripts/clean_cohort.py --apply
  python scripts/clean_cohort.py --apply --label-col label --review-col review_status

This module is import-safe: run_clean() is a pure function used by the unit test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
REQUIRED_KEY_COLS = ("variant_id", "ref", "alt")

LABEL_CANDIDATES = (
    "label", "is_pathogenic", "y", "target",
    "pathogenicity", "clinical_significance", "clinical_sig", "clnsig",
)
REVIEW_CANDIDATES = (
    "review_status", "review_status_tier", "clnrevstat", "gold_stars", "stars",
)

# Treat these string tokens (and true NaN) as an absent ref/alt -> structural.
BAD_ALLELE_TOKENS = {"", "nan", "none", "na", ".", "null", "-"}

# ClinVar review status -> tier (lower is better / more authoritative).
REVIEW_STATUS_TIER = {
    "practice guideline": 1,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no assertion criteria provided": 5,
    "no classification provided": 6,
    "no classification for the individual variant": 6,
}

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}


@dataclass
class Reconciliation:
    n_source: int = 0
    n_structural: int = 0
    n_exact_dup_dropped: int = 0
    n_conflict_resolved_dropped: int = 0
    n_conflict_rows: int = 0
    n_clean: int = 0
    label_col: str = ""
    review_col: str = ""
    notes: list[str] = field(default_factory=list)

    def identity_holds(self) -> bool:
        return self.n_source == (
            self.n_structural
            + self.n_exact_dup_dropped
            + self.n_conflict_resolved_dropped
            + self.n_conflict_rows
            + self.n_clean
        )

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["identity_holds"] = self.identity_holds()
        return d


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _normalize_label(series: pd.Series) -> pd.Series:
    """Map the label column to {1 (path), 0 (benign), -1 (uncertain/other)}.

    Numeric columns are interpreted as already-binary (>0 -> 1, ==0 -> 0).
    String columns are mapped via ClinVar term sets.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(-1).apply(lambda v: 1 if v >= 1 else (0 if v == 0 else -1)).astype(int)

    def _m(v: object) -> int:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return -1
        s = str(v).strip().lower()
        if s in PATHOGENIC_TERMS:
            return 1
        if s in BENIGN_TERMS:
            return 0
        return -1

    return series.apply(_m).astype(int)


def _review_tier(series: pd.Series | None, n: int) -> pd.Series:
    """Return a per-row review tier (lower = better). Absent -> all equal (tier 5)."""
    if series is None:
        return pd.Series([5] * n)
    if pd.api.types.is_numeric_dtype(series):
        # Numeric stars: more stars = better = lower tier. Invert.
        return (-series.fillna(0)).astype(float)
    return series.astype(str).str.strip().str.lower().map(REVIEW_STATUS_TIER).fillna(6).astype(int)


def _is_bad_allele(series: pd.Series) -> pd.Series:
    isna = series.isna()
    astxt = series.astype(str).str.strip().str.lower()
    return isna | astxt.isin(BAD_ALLELE_TOKENS)


def run_clean(
    df: pd.DataFrame,
    label_col: str | None = None,
    review_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Reconciliation]:
    """Pure de-leak function. Returns (clean, structural, conflicts, reconciliation).

    Raises ValueError on any unrecoverable schema problem or reconciliation failure.
    """
    recon = Reconciliation(n_source=len(df))

    missing = [c for c in REQUIRED_KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required key columns missing: {missing}. Present columns: {list(df.columns)}"
        )

    label_col = label_col or _detect_column(df, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(
            "Could not auto-detect a label column from candidates "
            f"{LABEL_CANDIDATES}. Pass --label-col explicitly. Present columns: {list(df.columns)}"
        )
    review_col = review_col or _detect_column(df, REVIEW_CANDIDATES)
    recon.label_col = label_col
    recon.review_col = review_col or "(none - conflicts treated as irreducible)"

    # 1. Quarantine structural / bad-key rows.
    bad_mask = _is_bad_allele(df["ref"]) | _is_bad_allele(df["alt"])
    structural = df[bad_mask].copy()
    work = df[~bad_mask].copy()
    recon.n_structural = len(structural)

    # 2. Annotate normalized label + review tier on the working set.
    work = work.assign(
        _norm_label=_normalize_label(work[label_col]).values,
        _tier=_review_tier(work[review_col] if review_col else None, len(work)).values,
    )

    # 3. Split singletons from duplicate variant_id groups.
    vc = work["variant_id"].value_counts()
    dup_ids = set(vc[vc > 1].index)
    singletons = work[~work["variant_id"].isin(dup_ids)]
    dups = work[work["variant_id"].isin(dup_ids)]

    kept_rows: list[pd.DataFrame] = [singletons]
    conflict_rows: list[pd.DataFrame] = []

    for _vid, grp in dups.groupby("variant_id", sort=False):
        distinct = set(grp["_norm_label"].unique())
        # Conflict only if the group contains BOTH a pathogenic(1) and a benign(0).
        is_conflict = (1 in distinct) and (0 in distinct)
        if not is_conflict:
            # Agreeing (or only uncertain) duplicate: keep best-review single row.
            best = grp.sort_values("_tier", kind="stable").iloc[[0]]
            kept_rows.append(best)
            recon.n_exact_dup_dropped += len(grp) - 1
        else:
            best_tier = grp["_tier"].min()
            at_best = grp[grp["_tier"] == best_tier]
            if set(at_best["_norm_label"].unique()) <= {l for l in (1, 0) if l in distinct} and \
               len(set(at_best["_norm_label"].unique())) == 1:
                # Single class wins at the best review tier -> resolvable.
                kept_rows.append(at_best.iloc[[0]])
                recon.n_conflict_resolved_dropped += len(grp) - 1
            else:
                # Irreducible: pathogenic and benign tie at the best tier.
                conflict_rows.append(grp)
                recon.n_conflict_rows += len(grp)

    clean = pd.concat(kept_rows, ignore_index=False) if kept_rows else work.iloc[0:0]
    conflicts = pd.concat(conflict_rows, ignore_index=False) if conflict_rows else work.iloc[0:0]

    clean = clean.drop(columns=["_norm_label", "_tier"], errors="ignore")
    conflicts = conflicts.drop(columns=["_norm_label", "_tier"], errors="ignore")
    recon.n_clean = len(clean)

    # 4. Post-conditions (fail loud).
    if clean["variant_id"].duplicated().any():
        raise ValueError("POST-CONDITION FAILED: clean cohort still has duplicate variant_id.")
    if (_is_bad_allele(clean["ref"]) | _is_bad_allele(clean["alt"])).any():
        raise ValueError("POST-CONDITION FAILED: clean cohort still has null/bad ref or alt.")
    if not recon.identity_holds():
        raise ValueError(
            "RECONCILIATION FAILED (rows lost or double-counted): " + json.dumps(recon.as_dict())
        )

    return clean, structural, conflicts, recon


def _print_report(recon: Reconciliation, df_head: pd.DataFrame) -> None:
    print("=" * 70)
    print("CLEAN_COHORT RECONCILIATION")
    print("=" * 70)
    print(f"label column detected : {recon.label_col}")
    print(f"review column detected: {recon.review_col}")
    print(f"source rows           : {recon.n_source:,}")
    print(f"  -> structural (null/bad key) : {recon.n_structural:,}")
    print(f"  -> agreeing-dup dropped      : {recon.n_exact_dup_dropped:,}")
    print(f"  -> conflict resolved dropped : {recon.n_conflict_resolved_dropped:,}")
    print(f"  -> irreducible conflict rows : {recon.n_conflict_rows:,}")
    print(f"  -> CLEAN rows                : {recon.n_clean:,}")
    print(f"reconciliation identity holds : {recon.identity_holds()}")
    print("=" * 70)
    print("Schema (first rows):")
    print(df_head.to_string(max_cols=12))
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-0 cohort de-leak.")
    p.add_argument("--input", default="data/processed/clinvar_grch38.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--label-col", default=None)
    p.add_argument("--review-col", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true", help="Dry-run: report only, write nothing (default).")
    g.add_argument("--apply", action="store_true", help="Write clean/structural/conflicts outputs.")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df):,} rows / {len(df.columns)} cols from {in_path}")
    print(f"Columns: {list(df.columns)}")

    clean, structural, conflicts, recon = run_clean(df, args.label_col, args.review_col)
    _print_report(recon, df.head(3))

    if not args.apply:
        print("\nAUDIT (dry-run) complete. No files written. Re-run with --apply to write.")
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(outdir / "clinvar_grch38_clean.parquet", index=False)
    structural.to_parquet(outdir / "clinvar_grch38_structural.parquet", index=False)
    conflicts.to_parquet(outdir / "clinvar_grch38_conflicts.parquet", index=False)
    (outdir / "clean_cohort_reconciliation.json").write_text(
        json.dumps(recon.as_dict(), indent=2), encoding="utf-8"
    )
    print(f"\nWROTE: clinvar_grch38_clean.parquet ({recon.n_clean:,} rows)")
    print(f"WROTE: clinvar_grch38_structural.parquet ({recon.n_structural:,} rows)")
    print(f"WROTE: clinvar_grch38_conflicts.parquet ({recon.n_conflict_rows:,} rows)")
    print("WROTE: clean_cohort_reconciliation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
