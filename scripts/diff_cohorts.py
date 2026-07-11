#!/usr/bin/env python
"""diff_cohorts.py (2026-07-10) -- two-mode parquet diff for GenAssoc.

MODES
  snapshot : compare two BUILT cohorts (e.g. cohort_stale.parquet vs cohort_fresh.parquet).
             Differences between snapshots are EXPECTED; everything is REPORTED, not asserted.
  labelfix : compare the raw processed source (clinvar_grch38.parquet) vs the label-corrected
             re-derivation (clinvar_grch38_pathfix.parquet). This is a PURE label-transition on
             a FIXED row set. STRICT integrity assertions apply:
               * identical variant_id sets (added == 0 and removed == 0)
               * EVERY shared non-pathogenicity column byte-identical (0 value diffs)
               * the ONLY pathogenicity transitions are pathogenic -> uncertain
             Any violation is a RED FLAG that the fix touched something it should not have.

Both modes share one engine: column-set reconciliation, variant_id set-diff, shared-row value
diff (normalized alleles; dict/large columns via element-wise equality, never hashing),
pathogenicity 5x5 transition matrix, full reconciliation guards, and a structured report.

Outputs (to --outdir, default 'outputs'):
  <prefix>_report.md            structured prose + tables (dates, MD5s, all counts)
  <prefix>_added.parquet        variant_ids in B not in A (+ key annotations)   [snapshot]
  <prefix>_removed.parquet      variant_ids in A not in B (+ key annotations)   [snapshot]
  <prefix>_reclassified.parquet shared variant_ids whose pathogenicity changed
  <prefix>_reclass_matrix.tsv   the 5x5 pathogenicity transition matrix
  <prefix>_summary.json         machine-readable counts + input checksums (provenance)

Read-only on inputs. Never mutates A or B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffcore import column_equal_series, transition_matrix, normalize_allele  # noqa: E402

PATH_CLASSES = ["pathogenic", "likely_pathogenic", "uncertain", "likely_benign", "benign"]
ALLELE_COLS = {"ref", "alt"}
EXCLUDE_VALUE_DIFF = {"protein_change"}          # rule R3: known all-null in fresh
KEY = "variant_id"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if KEY not in df.columns:
        raise ValueError(f"{path.name}: required key column {KEY!r} missing")
    return df



def _is_empty_allele(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    return str(v).strip().lower() in {"", "none", "nan", ".", "-", "na", "null"}


def _nana_mask(df: pd.DataFrame) -> np.ndarray:
    """True where BOTH ref and alt are empty (the builder's na:na quarantine condition)."""
    if "ref" not in df.columns or "alt" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    r = df["ref"].apply(_is_empty_allele).values
    a = df["alt"].apply(_is_empty_allele).values
    return r & a


def _composite_key(df: pd.DataFrame) -> pd.Series:
    """variant_id + '::' + source_id, with a within-group occurrence index appended so the key is
    ALWAYS unique even if (variant_id, source_id) still repeats. The occurrence index preserves
    row identity for exact alignment; it is deterministic given row order."""
    vid = df[KEY].astype(str)
    sid = df["source_id"].astype(str) if "source_id" in df.columns else pd.Series([""] * len(df), index=df.index)
    base = vid + "::" + sid
    occ = base.groupby(base).cumcount().astype(str)
    return base + "::" + occ


def run_diff(path_a: Path, path_b: Path, mode: str, outdir: Path, prefix: str,
             dup_policy: str | None = None) -> dict:
    assert mode in ("snapshot", "labelfix")
    if dup_policy is None:
        dup_policy = "report" if mode == "labelfix" else "strict"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    md5_a, md5_b = _md5(path_a), _md5(path_b)
    A, B = _load(path_a), _load(path_b)

    problems: list[str] = []
    report: list[str] = []
    rp = report.append

    rp(f"# Cohort/Source Diff Report ({mode})")
    rp("")
    rp(f"Generated: {stamp}")
    rp("")
    rp(f"- A: `{path_a.name}`  MD5 `{md5_a}`  rows {len(A):,}  cols {len(A.columns)}")
    rp(f"- B: `{path_b.name}`  MD5 `{md5_b}`  rows {len(B):,}  cols {len(B.columns)}")
    rp("")

    # ---- column-set reconciliation (rule R1) ----
    cols_a, cols_b = set(A.columns), set(B.columns)
    only_a, only_b = sorted(cols_a - cols_b), sorted(cols_b - cols_a)
    shared_cols = [c for c in A.columns if c in cols_b]     # preserve A's order
    rp("## Columns")
    rp(f"- shared: {len(shared_cols)}")
    if only_a:
        rp(f"- only in A: {only_a}")
    if only_b:
        rp(f"- only in B: {only_b}")
    excluded = sorted((set(shared_cols) & EXCLUDE_VALUE_DIFF))
    if excluded:
        rp(f"- excluded from value-diff (rule R3): {excluded}")
    rp("")

    # ---- duplicate-key handling (dimension A) ----
    ids_a = pd.Index(A[KEY])
    ids_b = pd.Index(B[KEY])
    dup_a = int(ids_a.duplicated().sum())
    dup_b = int(ids_b.duplicated().sum())
    rp("## Duplicate keys")
    rp(f"- duplicate {KEY} in A: {dup_a:,}   in B: {dup_b:,}   (policy: {dup_policy})")

    # na:na anomaly guard: every duplicate-involved row should be na:na (quarantinable). A
    # clean-allele duplicate is a genuine anomaly (a within-cohort dup that would NOT be dropped).
    if dup_a or dup_b:
        dupset_a = set(ids_a[ids_a.duplicated(keep=False)])
        dup_rows_a = A[A[KEY].isin(dupset_a)]
        clean_dup = int((~_nana_mask(dup_rows_a)).sum())
        rp(f"- duplicate-involved rows in A that are na:na: "
           f"{int(_nana_mask(dup_rows_a).sum()):,} / {len(dup_rows_a):,}")
        if clean_dup:
            problems.append(f"ANOMALY: {clean_dup} duplicate-involved row(s) in A have CLEAN "
                            f"alleles (not na:na) -- a genuine duplicate that would not be quarantined")
        else:
            rp("- all duplicate-involved rows are na:na (would be quarantined by the builder) -- OK")

    # decide whether duplicates are a failure, per policy, using COMPOSITE-KEY uniqueness.
    if dup_a or dup_b:
        # base composite (variant_id + source_id) WITHOUT the occurrence index -- this is what
        # strict mode evaluates: does source_id fully disambiguate the duplicated variant_ids?
        def _base_ck(df):
            vid = df[KEY].astype(str)
            sid = df["source_id"].astype(str) if "source_id" in df.columns else pd.Series([""] * len(df), index=df.index)
            return vid + "::" + sid
        bck_a, bck_b = _base_ck(A), _base_ck(B)
        base_unique = (not bck_a.duplicated().any()) and (not bck_b.duplicated().any())
        rp(f"- (variant_id, source_id) fully disambiguates duplicates: {base_unique}")
        if not base_unique:
            # occurrence index will still make alignment exact, but record the insufficiency
            rp(f"  NOTE: source_id does not fully disambiguate "
               f"(A residual {int(bck_a.duplicated().sum())}, B residual {int(bck_b.duplicated().sum())}); "
               f"falling back to occurrence-index for exact alignment.")
            if dup_policy == "strict":
                problems.append(f"strict dup-policy: (variant_id, source_id) still non-unique "
                                f"(A {int(bck_a.duplicated().sum())}, B {int(bck_b.duplicated().sum())}) "
                                f"-- occurrence-index fallback used")
    rp("")

    # ---- choose alignment key: composite for duplicated frames, plain variant_id otherwise ----
    use_composite = bool(dup_a or dup_b)
    if use_composite:
        A = A.assign(_ck=_composite_key(A))
        B = B.assign(_ck=_composite_key(B))
        ALIGN = "_ck"
    else:
        ALIGN = KEY

    set_a = set(A[ALIGN])
    set_b = set(B[ALIGN])
    added = set_b - set_a
    removed = set_a - set_b
    shared = set_a & set_b
    rp("## Variant set" + (" (aligned on composite key)" if use_composite else ""))
    rp(f"- A rows: {len(A):,}   B rows: {len(B):,}")
    rp(f"- added (in B, not A): {len(added):,}")
    rp(f"- removed (in A, not B): {len(removed):,}")
    rp(f"- shared: {len(shared):,}")
    if len(added) + len(shared) != len(set_b):
        problems.append("set-diff recon failed: |added|+|shared| != |B keys|")
    if len(removed) + len(shared) != len(set_a):
        problems.append("set-diff recon failed: |removed|+|shared| != |A keys|")
    rp("")

    # ---- align shared rows on the chosen key (composite key is unique -> no keep-first loss) ----
    A_sh = A[A[ALIGN].isin(shared)].drop_duplicates(ALIGN).set_index(ALIGN).sort_index()
    B_sh = B[B[ALIGN].isin(shared)].drop_duplicates(ALIGN).set_index(ALIGN).sort_index()
    A_sh, B_sh = A_sh.align(B_sh, join="inner", axis=0)

    # ---- per-column value diffs on shared rows ----
    value_diff_counts: dict[str, int] = {}
    coord_changed_ids: list = []
    rp("## Shared-row value diffs")
    diff_cols = [c for c in shared_cols if c not in EXCLUDE_VALUE_DIFF and c != KEY]
    for c in diff_cols:
        allele = c in ALLELE_COLS
        eq = column_equal_series(A_sh[c], B_sh[c], allele=allele)
        n_diff = int((~np.asarray(eq)).sum())
        value_diff_counts[c] = n_diff
        if n_diff:
            rp(f"- `{c}`: {n_diff:,} changed"
               + (" (normalized-allele)" if allele else ""))
    if not any(value_diff_counts.values()):
        rp("- (no shared-row value differences on compared columns)")
    rp("")

    # coordinate change = any of chrom/pos/ref/alt differ on a shared variant_id
    coord_cols = [c for c in ("chrom", "pos", "ref", "alt") if c in diff_cols]
    if coord_cols:
        coord_mask = np.zeros(len(A_sh), dtype=bool)
        for c in coord_cols:
            eq = column_equal_series(A_sh[c], B_sh[c], allele=(c in ALLELE_COLS))
            coord_mask |= ~np.asarray(eq)
        coord_changed_ids = list(A_sh.index[coord_mask])
        rp(f"## Coordinate changes on shared variant_id: {len(coord_changed_ids):,}")
        if coord_changed_ids:
            rp("  (variant_id encodes coordinates -> any change warrants investigation)")
            problems.append(f"{len(coord_changed_ids)} shared variant_ids have coordinate changes")
        rp("")

    # ---- pathogenicity transition matrix (dimension B headline) ----
    reclass_ids = []
    tm = None
    if "pathogenicity" in diff_cols:
        old = A_sh["pathogenicity"].astype(str)
        new = B_sh["pathogenicity"].astype(str)
        tm = transition_matrix(old, new, PATH_CLASSES)
        changed = old.values != new.values
        reclass_ids = list(A_sh.index[changed])
        rp("## Pathogenicity transition matrix (old -> new)")
        rp("```")
        rp(tm.to_string())
        rp("```")
        rp(f"- reclassified (off-diagonal) total: {int(changed.sum()):,}")
        # G3 reconcile: matrix total == number of shared rows
        if int(tm.values.sum()) != len(A_sh):
            problems.append("transition matrix total != shared row count")
        rp("")

        # ---- POSITIONAL cross-check (labelfix: A and B are the same rows in the same order) ----
        # Independent method: compare pathogenicity row-by-row on the full frame. For an in-place
        # re-derivation this is the ground-truth alignment. The composite-key matrix's off-diagonal
        # must equal the positional off-diagonal; disagreement is a real anomaly, surfaced not hidden.
        if mode == "labelfix" and len(A) == len(B):
            pos_old = A["pathogenicity"].astype(str).values
            pos_new = B["pathogenicity"].astype(str).values
            pos_changed = int((pos_old != pos_new).sum())
            pos_p2u = int(((pos_old == "pathogenic") & (pos_new == "uncertain")).sum())
            ck_off = int(changed.sum())
            ck_p2u = int(tm.loc["pathogenic", "uncertain"])
            rp("## Positional cross-check (independent of composite key)")
            rp(f"- positional total changes: {pos_changed:,}   composite-key off-diagonal: {ck_off:,}")
            rp(f"- positional pathogenic->uncertain: {pos_p2u:,}   composite-key: {ck_p2u:,}")
            if pos_changed != ck_off:
                problems.append(f"cross-check MISMATCH: positional changes {pos_changed} != "
                                f"composite-key off-diagonal {ck_off}")
            if pos_p2u != ck_p2u:
                problems.append(f"cross-check MISMATCH: positional pathogenic->uncertain {pos_p2u} "
                                f"!= composite-key {ck_p2u}")
            if pos_changed == ck_off and pos_p2u == ck_p2u:
                rp("- AGREE: composite-key and positional methods match (full-frame count confirmed)")
            rp("")

    # ---- aggregate distribution shift (dimension C) ----
    rp("## Pathogenicity distribution")
    if "pathogenicity" in shared_cols:
        da = A["pathogenicity"].value_counts().reindex(PATH_CLASSES, fill_value=0)
        db = B["pathogenicity"].value_counts().reindex(PATH_CLASSES, fill_value=0)
        rp("```")
        rp(f"{'class':18s} {'A':>12s} {'B':>12s} {'delta':>12s}")
        for c in PATH_CLASSES:
            rp(f"{c:18s} {da[c]:>12,} {db[c]:>12,} {db[c]-da[c]:>+12,}")
        rp("```")
    rp("")

    # ---- mode-specific STRICT assertions (labelfix) ----
    if mode == "labelfix":
        rp("## Label-fix integrity assertions")
        if len(added) != 0:
            problems.append(f"labelfix: added != 0 ({len(added)})")
        if len(removed) != 0:
            problems.append(f"labelfix: removed != 0 ({len(removed)})")
        # every non-pathogenicity compared column must be identical
        for c, n in value_diff_counts.items():
            if c != "pathogenicity" and n != 0:
                problems.append(f"labelfix: non-pathogenicity column {c!r} changed ({n})")
        # only transition allowed: pathogenic -> uncertain
        if tm is not None:
            offdiag_bad = 0
            for i in PATH_CLASSES:
                for j in PATH_CLASSES:
                    if i != j and not (i == "pathogenic" and j == "uncertain"):
                        offdiag_bad += int(tm.loc[i, j])
            if offdiag_bad != 0:
                problems.append(f"labelfix: unexpected transitions besides pathogenic->uncertain "
                                f"({offdiag_bad} rows)")
            rp(f"- pathogenic -> uncertain: {int(tm.loc['pathogenic','uncertain']):,}")
            rp(f"- other off-diagonal transitions: {offdiag_bad:,} (must be 0)")
        rp("")

    # ---- write artifacts ----
    def _annot(df_src, ids):
        keep = [c for c in (KEY, "gene_symbol", "chrom", "pos", "ref", "alt",
                            "pathogenicity", "clinical_sig") if c in df_src.columns]
        return df_src[df_src[KEY].isin(ids)][keep].copy()

    written = {}
    if mode == "snapshot":
        p_add = outdir / f"{prefix}_added.parquet"
        p_rem = outdir / f"{prefix}_removed.parquet"
        _annot(B, added).to_parquet(p_add, index=False)
        _annot(A, removed).to_parquet(p_rem, index=False)
        written["added"] = p_add.name
        written["removed"] = p_rem.name
    if reclass_ids:
        rc = A_sh.loc[reclass_ids, ["pathogenicity"]].rename(columns={"pathogenicity": "old_path"})
        rc["new_path"] = B_sh.loc[reclass_ids, "pathogenicity"]
        for extra in ("gene_symbol", "clinical_sig"):
            if extra in A_sh.columns:
                rc[extra] = A_sh.loc[reclass_ids, extra]
        p_rc = outdir / f"{prefix}_reclassified.parquet"
        rc.reset_index().to_parquet(p_rc, index=False)
        written["reclassified"] = p_rc.name
    if tm is not None:
        p_tm = outdir / f"{prefix}_reclass_matrix.tsv"
        tm.to_csv(p_tm, sep="\t")
        written["reclass_matrix"] = p_tm.name

    summary = {
        "mode": mode, "generated": stamp,
        "A": {"name": path_a.name, "md5": md5_a, "rows": int(len(A))},
        "B": {"name": path_b.name, "md5": md5_b, "rows": int(len(B))},
        "columns": {"shared": len(shared_cols), "only_a": only_a, "only_b": only_b,
                    "excluded_value_diff": excluded},
        "added": len(added), "removed": len(removed), "shared": len(shared),
        "value_diff_counts": value_diff_counts,
        "coord_changed": len(coord_changed_ids),
        "reclassified": len(reclass_ids),
        "written": written,
        "problems": problems,
    }
    p_sum = outdir / f"{prefix}_summary.json"
    p_sum.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    p_rep = outdir / f"{prefix}_report.md"
    p_rep.write_text("\n".join(report), encoding="utf-8")

    # ---- final verdict ----
    rp2 = print
    if problems:
        rp2(f"\n=== {mode} DIFF: {len(problems)} PROBLEM(S) ===")
        for p in problems:
            rp2(f"  - {p}")
    else:
        rp2(f"\n=== {mode} DIFF: OK (no guard violations) ===")
    rp2(f"WROTE: {p_rep.name}, {p_sum.name}" +
        (", " + ", ".join(written.values()) if written else ""))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Two-mode parquet diff (snapshot | labelfix).")
    ap.add_argument("--mode", required=True, choices=["snapshot", "labelfix"])
    ap.add_argument("--a", required=True, help="path to parquet A (baseline)")
    ap.add_argument("--b", required=True, help="path to parquet B (comparand)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--prefix", default=None, help="output filename prefix (default: mode)")
    ap.add_argument("--dup-policy", choices=["strict", "report"], default=None,
                    help="how to treat duplicate keys (default: report for labelfix, strict for snapshot)")
    a = ap.parse_args()
    prefix = a.prefix or (f"cohort_diff_{a.mode}")
    summary = run_diff(Path(a.a), Path(a.b), a.mode, Path(a.outdir), prefix,
                       dup_policy=a.dup_policy)
    # exit non-zero if labelfix integrity failed (so CI/automation catches it)
    if a.mode == "labelfix" and summary["problems"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
