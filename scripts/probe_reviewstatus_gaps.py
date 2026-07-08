#!/usr/bin/env python
"""
probe_reviewstatus_gaps.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
SUPERSEDES scripts/probe_reviewstatus_source.py, whose verdict logic had two
metric defects (documented below and corrected here).

WHY THIS EXISTS
    On 2026-07-08 a re-run of `clean_cohort.py --apply` silently dropped the
    top-level `ReviewStatus` column, because `_detect_column()` scans only
    df.columns and cannot see `metadata.review_status` nested in the struct.
    Investigating that, the first probe reported 90.35% "normalized-equal" and
    concluded the two sources were NOT equivalent. That conclusion was an
    artifact of the metric:

      DEFECT 1: equality was computed over ALL rows, so `metadata` was penalised
                for having a value where `ReviewStatus` is blank. The correct
                metric is agreement WHERE BOTH ARE POPULATED (which was 100.0000%).
      DEFECT 2: the literal token "-" (245,148 rows) was counted as "populated",
                inflating metadata coverage to a false 100.000%.
      DEFECT 3: probe `_tier_of("")` returned 5, but clean_cohort's `_review_tier`
                maps "" -> NaN -> 6. The probe did not model production semantics.

    All three are corrected here.

THE REAL QUESTION THIS ANSWERS
    The 424,516 rows with a blank `ReviewStatus` appear, from a five-row sample,
    to be INDELS. `augment_reviewstatus.py:64` does:
        df["ReviewStatus"] = key.map(vmap).fillna("")   # unmatched -> "" -> tier 5
    If the VCF join key mis-normalises indels (VCF left-aligns with a padding
    base), then unmatched indels are silently assigned the WORST review tier --
    not because ClinVar reviewed them poorly, but because the join missed them.
    Any run with --min-review-tier < 5 would then preferentially DISCARD INDELS.
    That is a systematic selection bias, and it is the finding that matters.

WHAT IT MEASURES
    1. Agreement where both populated (the correct equivalence metric).
    2. Coverage of each source, with missing tokens properly excluded.
    3. Cross-tab of blank-ReviewStatus vs variant type (SNV / insertion /
       deletion / MNV) -- the indel hypothesis, tested.
    4. Semantics of the "-" token in metadata.review_status.
    5. Whether blank-ReviewStatus correlates with the LABEL (bias check).
    6. Cohort retention at every min_review_tier, under BOTH production tier
       functions and BOTH candidate sources -- the impact of Fix (a).

DECISION RULE (corrected, and stated before the data is seen)
    * agreement-where-both-populated >= 99.9%  AND  metadata coverage > RS coverage
        -> the sources are CONSISTENT and metadata is a strict SUPERSET.
           Fix (a) (self-derive from metadata) is VIABLE -- but it changes the
           review tier of the blank rows, hence the training cohort. It must be
           adopted as a DELIBERATE, QUANTIFIED, DOCUMENTED change, never silently.
    * otherwise
        -> the sources genuinely diverge. Fix (b): augment_reviewstatus.py remains
           the supplier. Document the divergence before any further change.

    FIX (c) IS REQUIRED REGARDLESS AND IMMEDIATELY: clean_cohort.py must grow a
    hard pre-condition (no resolvable review column => raise) and a hard
    post-condition (ReviewStatus present in the written schema). It currently
    guards row integrity (dups, null alleles, reconciliation) but NOT its own
    output schema -- which is the only thing that changed.

USAGE (from project root, .venv312 active)
    python scripts/probe_reviewstatus_gaps.py
    python scripts/probe_reviewstatus_gaps.py --cohort data/processed/clinvar_grch38_clean.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Production constants, copied verbatim so the probe models real behaviour.
# NOTE: the key "no classification for the individual variant" is DEAD -- the
# data says "...for the single variant". Reproduced as-is, on purpose, so the
# probe measures what production actually does, not what it intended.
# ---------------------------------------------------------------------------
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

# Tokens that mean "absent", however a given writer chose to spell it.
MISSING_TOKENS = {"", "-", ".", "na", "nan", "none", "null", "<na>"}


def _norm(s: pd.Series) -> pd.Series:
    """Lowercase, underscores -> spaces, collapse whitespace. Nulls -> ''."""
    return (
        s.astype("string")
        .fillna("")
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _is_missing(s: pd.Series) -> pd.Series:
    return s.isin(MISSING_TOKENS)


def _extract_nested(md: pd.Series) -> pd.Series:
    def get(v):
        if isinstance(v, dict):
            return v.get("review_status")
        return getattr(v, "review_status", None)
    return pd.Series([get(v) for v in md], dtype="string")


def tier_exact(s: str) -> int:
    """clean_cohort.py `_review_tier` semantics: exact .map(), fillna(6)."""
    return REVIEW_STATUS_TIER.get(s, 6)


def tier_substr(s: str) -> int:
    """real_data_prep.py:479 and augment_reviewstatus.py:32 semantics."""
    return next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5)


def _variant_class(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    lr, la = r.str.len(), a.str.len()
    out = pd.Series("MNV/other", index=r.index, dtype="object")
    out[(lr == 1) & (la == 1)] = "SNV"
    out[(lr > 1) & (la == 1)] = "deletion"
    out[(lr == 1) & (la > 1)] = "insertion"
    return out


def _hr(title: str) -> None:
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ReviewStatus gap + indel-bias probe.")
    p.add_argument("--cohort", default="data/processed/clinvar_grch38_clean.parquet")
    args = p.parse_args(argv)

    cohort = Path(args.cohort)
    if not cohort.exists():
        print(f"ERROR: cohort not found: {cohort}", file=sys.stderr)
        return 2

    print("=" * 74)
    print(f"PROBE: ReviewStatus gaps, indel bias, cohort impact   ({cohort})")
    print("=" * 74)

    df = pd.read_parquet(
        cohort, columns=["variant_id", "ref", "alt", "pathogenicity", "ReviewStatus", "metadata"]
    )
    n = len(df)
    rs = _norm(df["ReviewStatus"])
    md = _norm(_extract_nested(df["metadata"]))
    rs_missing = _is_missing(rs)
    md_missing = _is_missing(md)
    vclass = _variant_class(df["ref"], df["alt"])

    rs_pop, md_pop = int((~rs_missing).sum()), int((~md_missing).sum())
    print(f"rows                                : {n:,}")
    print(f"ReviewStatus populated              : {rs_pop:,}  ({100*rs_pop/n:.3f}%)")
    print(f"metadata.review_status populated    : {md_pop:,}  ({100*md_pop/n:.3f}%)")
    print(f"  (missing tokens excluded: {sorted(MISSING_TOKENS)})")

    # --- 1. the CORRECT equivalence metric -------------------------------
    _hr("1. AGREEMENT WHERE BOTH POPULATED  (the corrected metric)")
    both = (~rs_missing) & (~md_missing)
    nb = int(both.sum())
    agree = int((rs[both] == md[both]).sum())
    pct_agree = 100 * agree / nb if nb else float("nan")
    print(f"rows with BOTH populated            : {nb:,}")
    print(f"  agreeing                          : {agree:,}  ({pct_agree:.4f}%)")
    print(f"  disagreeing                       : {nb - agree:,}")
    if nb - agree:
        d = df.loc[both & (rs != md), ["variant_id"]].head(5)
        print("  first true disagreements          :", list(d["variant_id"]))

    print(f"\nrows RS-missing but metadata present: {int((rs_missing & ~md_missing).sum()):,}")
    print(f"rows metadata-missing but RS present: {int((md_missing & ~rs_missing).sum()):,}")
    print(f"rows both missing                   : {int((rs_missing & md_missing).sum()):,}")

    # --- 2. the indel hypothesis -----------------------------------------
    _hr("2. INDEL HYPOTHESIS  (is blank ReviewStatus enriched for indels?)")
    ct = pd.crosstab(vclass, rs_missing, normalize=False)
    ct.columns = ["RS_present", "RS_blank"] if list(ct.columns) == [False, True] else ct.columns
    ct["pct_blank"] = (100 * ct["RS_blank"] / (ct["RS_blank"] + ct["RS_present"])).round(3)
    print(ct.to_string())
    overall = 100 * int(rs_missing.sum()) / n
    print(f"\noverall blank rate                  : {overall:.3f}%")
    print("INTERPRETATION: if pct_blank for deletion/insertion >> SNV, the VCF join")
    print("key mis-normalises indels and assigns them the WORST tier by accident.")

    # --- 3. what is "-" ? -------------------------------------------------
    _hr("3. SEMANTICS OF THE '-' TOKEN IN metadata.review_status")
    dash = md == "-"
    print(f"metadata == '-'                     : {int(dash.sum()):,}")
    if int(dash.sum()):
        print("\n  ReviewStatus values where metadata == '-' (top 6):")
        print(rs[dash].value_counts().head(6).to_string())
        print("\n  variant class where metadata == '-':")
        print(vclass[dash].value_counts().to_string())

    print("\n  metadata values where ReviewStatus is blank (top 8):")
    print(md[rs_missing].value_counts().head(8).to_string())

    # --- 4. label-bias check ---------------------------------------------
    _hr("4. BIAS CHECK  (does missing ReviewStatus correlate with the label?)")
    lab = df["pathogenicity"].astype("string").fillna("(null)")
    tab = pd.crosstab(lab, rs_missing)
    tab.columns = ["RS_present", "RS_blank"] if list(tab.columns) == [False, True] else tab.columns
    tab["pct_blank"] = (100 * tab["RS_blank"] / (tab["RS_blank"] + tab["RS_present"])).round(3)
    print(tab.to_string())
    print("\nINTERPRETATION: a materially higher pct_blank in pathogenic/benign than")
    print("in uncertain would mean tier filtering biases the LABEL distribution.")

    # --- 5. cohort impact of Fix (a) --------------------------------------
    _hr("5. COHORT RETENTION BY min_review_tier  (impact of switching source)")
    t_rs_sub = rs.map(tier_substr)     # real_data_prep semantics, current source
    t_md_sub = md.map(tier_substr)     # real_data_prep semantics, proposed source
    t_rs_exa = rs.map(tier_exact)      # clean_cohort semantics, current source
    print("Assumption: pipeline keeps rows with review_tier <= min_review_tier.")
    print("CONFIRM against real_data_prep before relying on these numbers:")
    print("  Select-String -Path src\\genomic_variant_classifier\\data\\real_data_prep.py "
          "-Pattern 'min_review_tier' -Context 2,2")
    print()
    rows = []
    for t in (1, 2, 3, 4, 5):
        keep_rs = int((t_rs_sub <= t).sum())
        keep_md = int((t_md_sub <= t).sum())
        rows.append({
            "min_review_tier": t,
            "keep_from_ReviewStatus": keep_rs,
            "keep_from_metadata": keep_md,
            "delta_rows": keep_md - keep_rs,
            "delta_pct_of_cohort": round(100 * (keep_md - keep_rs) / n, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\ntier distribution, three ways (rows):")
    dist = pd.DataFrame({
        "RS_substr(prod filter)": t_rs_sub.value_counts().sort_index(),
        "RS_exact(clean_cohort)": t_rs_exa.value_counts().sort_index(),
        "metadata_substr": t_md_sub.value_counts().sort_index(),
    }).fillna(0).astype(int)
    print(dist.to_string())
    print("\nNOTE: RS_substr vs RS_exact differ on blanks (5 vs 6). Three separate")
    print("implementations of review-tier mapping exist in the codebase; they should")
    print("be reconciled into ONE function with ONE documented default.")

    # --- verdict ----------------------------------------------------------
    print()
    print("=" * 74)
    consistent = (pct_agree >= 99.9) if nb else False
    superset = md_pop > rs_pop
    if consistent and superset:
        print("VERDICT: sources CONSISTENT (100% agreement where both populated) and")
        print("         metadata is a strict SUPERSET.")
        print("  -> Fix (a) self-derivation is VIABLE and technically superior, BUT it")
        print("     re-tiers the blank rows and therefore CHANGES THE TRAINING COHORT.")
        print("     Adopt only as a deliberate, quantified, documented change. Runs 1-17")
        print("     used the VCF-derived column; Run 18 would not be comparable.")
    elif consistent and not superset:
        print("VERDICT: sources CONSISTENT but metadata is NOT more complete.")
        print("  -> No benefit to switching. Fix (b): keep augment_reviewstatus.py.")
    else:
        print("VERDICT: sources DIVERGE where both are populated. Investigate before")
        print("         any change. Fix (b): keep augment_reviewstatus.py as supplier.")
    print()
    print("FIX (c) IS REQUIRED IMMEDIATELY AND UNCONDITIONALLY:")
    print("  clean_cohort.py needs a hard PRE-condition (no resolvable review column")
    print("  => raise, never silent all-tier-5) and a hard POST-condition (ReviewStatus")
    print("  present in the written schema). It guards rows but not its own schema.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
