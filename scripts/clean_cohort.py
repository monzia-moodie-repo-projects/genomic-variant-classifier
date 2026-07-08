"""
clean_cohort.py - Phase 0 cohort de-leak
========================================
Resolves the null-key leak (INCIDENT_<date>_null-key-leak) and the duplicate /
label-conflict integrity problem in the source ClinVar cohort, BEFORE splits are
regenerated. Operates at the cohort source so training runs on clean data.

INPUT  : data/processed/clinvar_grch38.parquet
OUTPUTS: data/processed/clinvar_grch38_clean.parquet        (0 null-key, 0 dup variant_id)
         data/processed/clinvar_grch38_structural.parquet   (null/bad ref or alt)
         data/processed/clinvar_grch38_conflicts.parquet    (irreducible label conflicts)
         data/processed/clean_cohort_reconciliation.json     (full audit; rows reconcile)

NOTE ON PIPELINE ORDER. This script does NOT create the top-level `ReviewStatus`
column; scripts/augment_reviewstatus.py attaches it afterwards. The pipeline is:

    clinvar_grch38.parquet --clean_cohort--> clean.parquet (no ReviewStatus)
                           --augment_reviewstatus--> clean.parquet (+ ReviewStatus)

Re-running clean_cohort therefore REVERTS the augmentation. See §"schema-regression
guard" below: this script now refuses to silently drop a column the existing output
already carries.

=============================================================================
REVISION 2026-07-08 -- hardening after two incidents. Rows and columns written
are UNCHANGED by design; only the guards, the audit record, and two latent
correctness bugs are addressed.
=============================================================================

WHAT CHANGED AND WHY

  1. NESTED-AWARE REVIEW-COLUMN RESOLUTION, FAILING LOUD.
     `_detect_column` scanned only top-level `df.columns`. The source cohort keeps
     review status nested at `metadata.review_status` (struct<review_status, rs_id>),
     so detection returned None and `_review_tier(None, n)` silently assigned EVERY
     row tier 5. Conflict resolution then degraded to "conflicts treated as
     irreducible" -- a policy change announced only as an informational print.
     Now: struct fields are scanned too; an unresolvable review column RAISES unless
     --allow-no-review is passed explicitly.

  2. WRITE-TIME SCHEMA-REGRESSION GUARD.
     On 2026-07-08 a re-run of `--apply` overwrote the augmented 18-leaf cohort with
     an un-augmented 17-leaf one, silently dropping `ReviewStatus`. Every row-level
     post-condition passed (0 dups, 0 bad alleles, exact reconciliation) because none
     of them concerned the SCHEMA. Now: writing aborts if it would drop a column the
     existing output has, unless --allow-schema-regression is passed.

  3. POST-WRITE SCHEMA VERIFICATION + COMPOSITION IN THE AUDIT RECORD.
     The reconciliation JSON now records a schema fingerprint and per-variant-class
     counts (SNV / deletion / insertion / MNV). Guards on rows are not guards on
     populations: see docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md,
     where a cohort satisfying every row invariant had lost 99% of its deletions.

  4. UNDERSCORE-AWARE TERM MATCHING (latent bug, previously inert).
     PATHOGENIC_TERMS/BENIGN_TERMS were written with spaces ("likely pathogenic") but
     the data uses underscores ("likely_pathogenic"), so `_normalize_label` silently
     mapped both `likely_pathogenic` and `likely_benign` to -1 (uncertain). Inert only
     because the source has zero duplicate variant_id, so the conflict machinery never
     ran. Normalisation now converts underscores before matching. Same fix applied to
     REVIEW_STATUS_TIER lookups.

  5. DEAD TIER KEY REMOVED.
     "no classification for the individual variant" never matched; ClinVar says
     "...for the single variant". Both spellings are now present.

DESIGN PRINCIPLES (no silent failures, no guessing):
  * Introspects schema (top-level AND struct fields); FAILS LOUD, listing what it
    actually saw, if it cannot identify the label or review column.
  * --audit (default) is a dry-run: prints schema, distributions, composition, and
    the full reconciliation plan WITHOUT writing anything.
  * --apply writes outputs only after the reconciliation identity holds exactly AND
    the schema-regression guard passes.
  * Every source row is accounted for; the script raises if the arithmetic does not
    reconcile to the exact source row count.
  * A missing value and a bad value never share a representation.

USAGE (from project root, .venv312 active):
  python scripts/clean_cohort.py --audit
  python scripts/clean_cohort.py --apply
  python scripts/clean_cohort.py --apply --review-col metadata.review_status
  python scripts/clean_cohort.py --apply --allow-schema-regression   # deliberate only

EXIT CODES
  0  success (audit reported, or apply wrote and verified)
  2  input file not found
  3  schema-regression guard blocked the write; nothing written

This module is import-safe: run_clean() is a pure function used by the unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Tokens that mean "absent", however a given writer spelled it. A missing value and
# a bad value must never share a representation -- these normalise to NaN, and the
# tier lookup then falls through to its documented default rather than pretending.
MISSING_TOKENS = {"", "-", ".", "na", "nan", "none", "null", "<na>"}

# ClinVar review status -> tier (lower is better / more authoritative).
# Keys are SPACE-form; all lookups normalise underscores to spaces first.
REVIEW_STATUS_TIER = {
    "practice guideline": 1,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no assertion criteria provided": 5,
    "no classification provided": 6,
    "no classification for the single variant": 6,      # the spelling ClinVar uses
    "no classification for the individual variant": 6,  # retained: older releases
}
TIER_UNMATCHED = 6  # documented default for this module (real_data_prep uses 5)

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}


def _norm_term(v: object) -> str:
    """Canonicalise a ClinVar term: lowercase, underscores -> spaces, collapse ws."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return " ".join(str(v).strip().lower().replace("_", " ").split())


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
    schema_fingerprint: str = ""
    clean_columns: list[str] = field(default_factory=list)
    composition: dict[str, int] = field(default_factory=dict)
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


# ---------------------------------------------------------------------------
# Column resolution -- top-level AND nested struct fields
# ---------------------------------------------------------------------------
def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _struct_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map object-dtype columns holding dicts -> their field names."""
    out: dict[str, list[str]] = {}
    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue
        first = next((v for v in s.head(200) if isinstance(v, dict)), None)
        if first is not None:
            out[col] = list(first.keys())
    return out


def _detect_struct_field(
    df: pd.DataFrame, candidates: tuple[str, ...]
) -> tuple[str | None, str | None]:
    for col, fields in _struct_columns(df).items():
        lower = {f.lower(): f for f in fields}
        for cand in candidates:
            if cand in lower:
                return col, lower[cand]
    return None, None


def _extract_field(df: pd.DataFrame, col: str, fieldname: str) -> pd.Series:
    return pd.Series(
        [v.get(fieldname) if isinstance(v, dict) else None for v in df[col]],
        index=df.index, dtype="object",
    )


def resolve_review_series(
    df: pd.DataFrame, review_col: str | None
) -> tuple[pd.Series | None, str | None]:
    """Return (series, resolved_name). Supports dotted 'metadata.review_status'."""
    if review_col:
        if "." in review_col:
            col, fld = review_col.split(".", 1)
            if col not in df.columns:
                raise ValueError(f"--review-col '{review_col}': no column '{col}'. "
                                 f"Present: {list(df.columns)}")
            return _extract_field(df, col, fld), review_col
        if review_col not in df.columns:
            raise ValueError(f"--review-col '{review_col}' not present. "
                             f"Columns: {list(df.columns)}")
        return df[review_col], review_col

    top = _detect_column(df, REVIEW_CANDIDATES)
    if top:
        return df[top], top

    col, fld = _detect_struct_field(df, REVIEW_CANDIDATES)
    if col:
        return _extract_field(df, col, fld), f"{col}.{fld}"

    return None, None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _normalize_label(series: pd.Series) -> pd.Series:
    """Map the label column to {1 (path), 0 (benign), -1 (uncertain/other)}.

    Numeric columns are interpreted as already-binary (>0 -> 1, ==0 -> 0).
    String columns are mapped via ClinVar term sets, underscore-aware.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(-1).apply(lambda v: 1 if v >= 1 else (0 if v == 0 else -1)).astype(int)

    def _m(v: object) -> int:
        s = _norm_term(v)
        if s in PATHOGENIC_TERMS:
            return 1
        if s in BENIGN_TERMS:
            return 0
        return -1

    return series.apply(_m).astype(int)


def _review_tier(series: pd.Series | None, n: int) -> pd.Series:
    """Per-row review tier (lower = better). Unmatched/missing -> TIER_UNMATCHED."""
    if series is None:
        # Reachable only under --allow-no-review; the caller has accepted the
        # consequence (all rows equal tier => conflicts become irreducible).
        return pd.Series([TIER_UNMATCHED] * n)
    if pd.api.types.is_numeric_dtype(series):
        return (-series.fillna(0)).astype(float)  # more stars = better = lower tier
    norm = series.apply(_norm_term)
    norm = norm.where(~norm.isin(MISSING_TOKENS), other=pd.NA)
    return norm.map(REVIEW_STATUS_TIER).fillna(TIER_UNMATCHED).astype(int)


def _is_bad_allele(series: pd.Series) -> pd.Series:
    isna = series.isna()
    astxt = series.astype(str).str.strip().str.lower()
    return isna | astxt.isin(BAD_ALLELE_TOKENS)


def variant_class(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    lr, la = r.str.len(), a.str.len()
    out = pd.Series("MNV/other", index=r.index, dtype="object")
    out[(lr == 1) & (la == 1)] = "SNV"
    out[(lr > 1) & (la == 1)] = "deletion"
    out[(lr == 1) & (la > 1)] = "insertion"
    return out


def schema_fingerprint(columns) -> str:
    """Stable fingerprint of a column set. Order-independent by construction."""
    joined = ",".join(sorted(map(str, columns)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run_clean(
    df: pd.DataFrame,
    label_col: str | None = None,
    review_col: str | None = None,
    allow_no_review: bool = False,
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
    recon.label_col = label_col

    # PRE-CONDITION: a review column must be resolvable, or the caller must have
    # explicitly accepted the consequence. Never silently assign every row one tier.
    review_series, resolved = resolve_review_series(df, review_col)
    if review_series is None:
        if not allow_no_review:
            raise ValueError(
                "PRE-CONDITION FAILED: no review column resolvable from candidates "
                f"{REVIEW_CANDIDATES}.\n"
                f"  top-level columns : {list(df.columns)}\n"
                f"  struct fields     : {_struct_columns(df)}\n"
                "Without a review column every row receives the same tier, so duplicate "
                "label conflicts become 'irreducible' and resolution silently degrades. "
                "Pass --review-col explicitly (dotted paths supported, e.g. "
                "metadata.review_status), or pass --allow-no-review to accept this."
            )
        recon.notes.append("allow_no_review=True: all rows assigned the same review tier")
    recon.review_col = resolved or "(none - explicitly allowed; conflicts irreducible)"

    # 1. Quarantine structural / bad-key rows.
    bad_mask = _is_bad_allele(df["ref"]) | _is_bad_allele(df["alt"])
    structural = df[bad_mask].copy()
    work = df[~bad_mask].copy()
    recon.n_structural = len(structural)

    # 2. Annotate normalized label + review tier on the working set.
    rev_work = review_series[~bad_mask] if review_series is not None else None
    work = work.assign(
        _norm_label=_normalize_label(work[label_col]).values,
        _tier=_review_tier(rev_work, len(work)).values,
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
        is_conflict = (1 in distinct) and (0 in distinct)
        if not is_conflict:
            best = grp.sort_values("_tier", kind="stable").iloc[[0]]
            kept_rows.append(best)
            recon.n_exact_dup_dropped += len(grp) - 1
        else:
            best_tier = grp["_tier"].min()
            at_best = grp[grp["_tier"] == best_tier]
            classes_at_best = set(at_best["_norm_label"].unique())
            if len(classes_at_best) == 1 and classes_at_best <= {0, 1}:
                kept_rows.append(at_best.iloc[[0]])
                recon.n_conflict_resolved_dropped += len(grp) - 1
            else:
                conflict_rows.append(grp)
                recon.n_conflict_rows += len(grp)

    clean = pd.concat(kept_rows, ignore_index=False) if kept_rows else work.iloc[0:0]
    conflicts = pd.concat(conflict_rows, ignore_index=False) if conflict_rows else work.iloc[0:0]

    clean = clean.drop(columns=["_norm_label", "_tier"], errors="ignore")
    conflicts = conflicts.drop(columns=["_norm_label", "_tier"], errors="ignore")
    recon.n_clean = len(clean)

    # 4. Post-conditions on ROWS (fail loud).
    if clean["variant_id"].duplicated().any():
        raise ValueError("POST-CONDITION FAILED: clean cohort still has duplicate variant_id.")
    if (_is_bad_allele(clean["ref"]) | _is_bad_allele(clean["alt"])).any():
        raise ValueError("POST-CONDITION FAILED: clean cohort still has null/bad ref or alt.")
    if not recon.identity_holds():
        raise ValueError(
            "RECONCILIATION FAILED (rows lost or double-counted): " + json.dumps(recon.as_dict())
        )

    # 5. Post-conditions on SCHEMA and COMPOSITION (record; guards on rows are not
    #    guards on populations -- see INCIDENT_2026-07-08).
    if list(clean.columns) != list(df.columns):
        raise ValueError(
            "POST-CONDITION FAILED: clean columns differ from source columns.\n"
            f"  source: {list(df.columns)}\n  clean : {list(clean.columns)}"
        )
    recon.clean_columns = list(clean.columns)
    recon.schema_fingerprint = schema_fingerprint(clean.columns)
    recon.composition = {
        k: int(v) for k, v in variant_class(clean["ref"], clean["alt"]).value_counts().items()
    }

    return clean, structural, conflicts, recon


# ---------------------------------------------------------------------------
# Write-time guards
# ---------------------------------------------------------------------------
def assert_no_schema_regression(
    new_columns, out_path: Path, allow: bool = False
) -> list[str]:
    """Refuse to overwrite an existing output by DROPPING columns it already has.

    This is the guard that would have prevented INCIDENT_2026-07-08, in which a
    re-run silently replaced an 18-leaf augmented cohort with a 17-leaf one,
    discarding `ReviewStatus` -- while every row-level post-condition passed.
    """
    if not out_path.exists():
        return []
    import pyarrow.parquet as pq  # local import keeps run_clean pyarrow-free

    existing = list(pq.ParquetFile(out_path).schema_arrow.names)
    dropped = [c for c in existing if c not in set(new_columns)]
    if dropped and not allow:
        raise ValueError(
            f"SCHEMA-REGRESSION GUARD: writing {out_path.name} would DROP column(s) "
            f"{dropped} that the existing file carries.\n"
            f"  existing : {existing}\n"
            f"  incoming : {list(new_columns)}\n"
            "This is how the augmented cohort was silently reverted on 2026-07-08. "
            "If the drop is intended, re-run with --allow-schema-regression and "
            "re-apply the downstream augmentation (scripts/augment_reviewstatus.py)."
        )
    return dropped


def verify_written_schema(out_path: Path, expected_columns) -> None:
    """Read back what was actually written; trust the file, not the intent."""
    import pyarrow.parquet as pq

    got = list(pq.ParquetFile(out_path).schema_arrow.names)
    if got != list(expected_columns):
        raise ValueError(
            f"POST-WRITE VERIFY FAILED for {out_path.name}:\n"
            f"  expected: {list(expected_columns)}\n  written : {got}"
        )


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------
def _print_report(recon: Reconciliation, df_head: pd.DataFrame) -> None:
    print("=" * 70)
    print("CLEAN_COHORT RECONCILIATION")
    print("=" * 70)
    print(f"label column detected : {recon.label_col}")
    print(f"review column resolved: {recon.review_col}")
    print(f"source rows           : {recon.n_source:,}")
    print(f"  -> structural (null/bad key) : {recon.n_structural:,}")
    print(f"  -> agreeing-dup dropped      : {recon.n_exact_dup_dropped:,}")
    print(f"  -> conflict resolved dropped : {recon.n_conflict_resolved_dropped:,}")
    print(f"  -> irreducible conflict rows : {recon.n_conflict_rows:,}")
    print(f"  -> CLEAN rows                : {recon.n_clean:,}")
    print(f"reconciliation identity holds : {recon.identity_holds()}")
    print("-" * 70)
    print(f"schema fingerprint    : {recon.schema_fingerprint}")
    print(f"clean columns ({len(recon.clean_columns)})     : {recon.clean_columns}")
    print("composition (variant class):")
    for k in ("SNV", "deletion", "insertion", "MNV/other"):
        v = recon.composition.get(k, 0)
        pct = 100 * v / recon.n_clean if recon.n_clean else 0.0
        print(f"    {k:12s} {v:>10,}  ({pct:6.3f}%)")
    if recon.notes:
        print("notes:")
        for nte in recon.notes:
            print(f"    - {nte}")
    print("=" * 70)
    print("Schema (first rows):")
    print(df_head.to_string(max_cols=12))
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-0 cohort de-leak.")
    p.add_argument("--input", default="data/processed/clinvar_grch38.parquet")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--label-col", default=None)
    p.add_argument("--review-col", default=None,
                   help="Column name, or dotted struct path e.g. metadata.review_status")
    p.add_argument("--allow-no-review", action="store_true",
                   help="Accept that no review column exists; all rows get one tier and "
                        "duplicate conflicts become irreducible. Explicit opt-in only.")
    p.add_argument("--allow-schema-regression", action="store_true",
                   help="Permit overwriting an existing output while DROPPING columns "
                        "it carries (e.g. ReviewStatus). Deliberate use only.")
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
    structs = _struct_columns(df)
    if structs:
        print(f"Struct fields: {structs}")

    clean, structural, conflicts, recon = run_clean(
        df, args.label_col, args.review_col, allow_no_review=args.allow_no_review
    )
    _print_report(recon, df.head(3))

    outdir = Path(args.outdir)
    clean_path = outdir / "clinvar_grch38_clean.parquet"

    # Guard runs in BOTH modes so --audit surfaces the regression BEFORE --apply
    # can act on it. A dry run must report, not crash: the abort becomes exit 3.
    try:
        dropped = assert_no_schema_regression(
            clean.columns, clean_path, allow=args.allow_schema_regression
        )
    except ValueError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print("\nABORTED: nothing written. (exit 3 = schema regression blocked)", file=sys.stderr)
        return 3
    if dropped:
        print(f"\n!! SCHEMA REGRESSION PERMITTED by --allow-schema-regression: dropping {dropped}")

    if not args.apply:
        print("\nAUDIT (dry-run) complete. No files written. Re-run with --apply to write.")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(clean_path, index=False)
    structural.to_parquet(outdir / "clinvar_grch38_structural.parquet", index=False)
    conflicts.to_parquet(outdir / "clinvar_grch38_conflicts.parquet", index=False)
    (outdir / "clean_cohort_reconciliation.json").write_text(
        json.dumps(recon.as_dict(), indent=2), encoding="utf-8"
    )

    verify_written_schema(clean_path, clean.columns)

    print(f"\nWROTE: clinvar_grch38_clean.parquet ({recon.n_clean:,} rows, "
          f"{len(recon.clean_columns)} cols, fingerprint {recon.schema_fingerprint})")
    print(f"WROTE: clinvar_grch38_structural.parquet ({recon.n_structural:,} rows)")
    print(f"WROTE: clinvar_grch38_conflicts.parquet ({recon.n_conflict_rows:,} rows)")
    print("WROTE: clean_cohort_reconciliation.json (schema fingerprint + composition)")
    print("POST-WRITE SCHEMA VERIFY: PASS")
    if recon.review_col.startswith("metadata."):
        print("\nNOTE: review status resolved from the nested struct. The top-level "
              "`ReviewStatus` column is attached separately by "
              "scripts/augment_reviewstatus.py -- re-run it if downstream needs it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
