#!/usr/bin/env python
"""
probe_tier_filter_impact.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
CONTEXT -- what probe_reviewstatus_gaps.py established on 2026-07-08

    98.834% of DELETIONS in the clean cohort carry a blank `ReviewStatus`
    (187,258 of 189,468). Insertions are unaffected (0.483% blank). The blanks
    arise in scripts/augment_reviewstatus.py:64 --

        df["ReviewStatus"] = key.map(vmap).fillna("")   # unmatched -> "" -> tier 5

    -- i.e. the VCF join misses deletions, and `.fillna("")` then assigns them
    the WORST review tier. real_data_prep.py:215 defaults `min_review_tier=3`
    and line 484 filters `df[df["review_tier"] <= min_review_tier]`. Blank ->
    tier 5 -> DROPPED.

    Consequently AT MOST 2,210 of 189,468 deletions (1.166%) can survive the
    default filter. The missingness is also label-correlated: pct_blank is
    22.3% for likely_pathogenic and 20.1% for pathogenic, versus 2.3% for
    likely_benign. The filter therefore removes pathogenic variants ~9.6x more
    often than likely-benign ones.

    `metadata.review_status` (already inside the source parquet's struct) agrees
    with the VCF-derived column on ALL 3,974,573 rows where both are populated
    (zero disagreements) and rescues 178,563 deletions -- 94.2% of every deletion
    in the cohort.

WHAT THIS PROBE MEASURES (the numbers the incident report needs)

    1. Agreement between the two sources BY VARIANT CLASS. The headline 100%
       agreement rests on only 2,210 deletions, because the join succeeded on so
       few. State the validation coverage honestly before relying on metadata
       for the very class it would rescue.
    2. EXACT retention by variant class, per min_review_tier, per source.
    3. EXACT retention by label, and the post-filter BINARY class balance
       (pathogenic + likely_pathogenic = 1; benign + likely_benign = 0;
       uncertain dropped) -- the balance that drives AUPRC, the primary metric.
    4. The deletion SHARE of the surviving training cohort, before and after.
    5. A live demonstration of the latent underscore bug in clean_cohort.py's
       PATHOGENIC_TERMS / BENIGN_TERMS (data says `likely_pathogenic`, the
       constants say `likely pathogenic`).

ASSUMPTIONS -- VERIFY BEFORE CITING ANY NUMBER FROM THIS PROBE
    * Production used min_review_tier=3 (the DataPrepConfig default). Confirm:
        Select-String -Path scripts\\run_phase2_eval.py, scripts\\smoke_all_models.py,
          scripts\\launch_run17_baseline.sh -Pattern 'min_review_tier|DataPrepConfig'
    * The label column consumed downstream is `pathogenicity` (what clean_cohort
      auto-detects). If real_data_prep uses `clinical_sig`, re-run with --label-col.
    * review_tier is computed with real_data_prep's SUBSTRING semantics
      (default 5), not clean_cohort's exact-map semantics (default 6). Both are
      reported so the divergence stays visible.

USAGE (from project root, .venv312 active)
    python scripts/probe_tier_filter_impact.py
    python scripts/probe_tier_filter_impact.py --min-review-tier 3 --label-col pathogenicity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# --- production constants, copied verbatim (dead key included on purpose) ----
REVIEW_STATUS_TIER = {
    "practice guideline": 1,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no assertion criteria provided": 5,
    "no classification provided": 6,
    "no classification for the individual variant": 6,   # DEAD: data says "single variant"
}

# clean_cohort.py's constants, verbatim -- note the SPACES
CC_PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
CC_BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}

# what the data actually contains (underscores)
DATA_PATHOGENIC = {"pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic"}
DATA_BENIGN = {"benign", "likely_benign", "benign/likely_benign"}

MISSING_TOKENS = {"", "-", ".", "na", "nan", "none", "null", "<na>"}


def _norm(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").fillna("").str.lower()
        .str.replace("_", " ", regex=False).str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _extract_nested(md: pd.Series) -> pd.Series:
    def get(v):
        if isinstance(v, dict):
            return v.get("review_status")
        return getattr(v, "review_status", None)
    return pd.Series([get(v) for v in md], dtype="string")


def tier_substr(s: str) -> int:
    """real_data_prep.py:479 semantics. Unmatched (incl. '' and '-') -> 5."""
    return next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5)


def tier_exact(s: str) -> int:
    """clean_cohort.py:139 semantics. Unmatched -> 6."""
    return REVIEW_STATUS_TIER.get(s, 6)


def _variant_class(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r, a = ref.astype("string").fillna(""), alt.astype("string").fillna("")
    lr, la = r.str.len(), a.str.len()
    out = pd.Series("MNV/other", index=r.index, dtype="object")
    out[(lr == 1) & (la == 1)] = "SNV"
    out[(lr > 1) & (la == 1)] = "deletion"
    out[(lr == 1) & (la > 1)] = "insertion"
    return out


def _binary_label(lab: pd.Series) -> pd.Series:
    """Underscore-aware. 1=path, 0=benign, -1=excluded (uncertain/other)."""
    s = lab.astype("string").fillna("").str.strip().str.lower()
    out = pd.Series(-1, index=s.index, dtype=int)
    out[s.isin(DATA_PATHOGENIC)] = 1
    out[s.isin(DATA_BENIGN)] = 0
    return out


def _hr(t: str) -> None:
    print("\n" + "-" * 76 + f"\n{t}\n" + "-" * 76)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quantify review-tier filter impact.")
    p.add_argument("--cohort", default="data/processed/clinvar_grch38_clean.parquet")
    p.add_argument("--label-col", default="pathogenicity")
    p.add_argument("--min-review-tier", type=int, default=3)
    args = p.parse_args(argv)

    cohort = Path(args.cohort)
    if not cohort.exists():
        print(f"ERROR: cohort not found: {cohort}", file=sys.stderr)
        return 2

    print("=" * 76)
    print(f"PROBE: review-tier filter impact   ({cohort})")
    print(f"assumed production min_review_tier = {args.min_review_tier}  (VERIFY -- see docstring)")
    print("=" * 76)

    df = pd.read_parquet(
        cohort, columns=["variant_id", "ref", "alt", args.label_col, "ReviewStatus", "metadata"]
    )
    n = len(df)
    rs = _norm(df["ReviewStatus"])
    md = _norm(_extract_nested(df["metadata"]))
    vcls = _variant_class(df["ref"], df["alt"])
    t_rs = rs.map(tier_substr)
    t_md = md.map(tier_substr)
    ybin = _binary_label(df[args.label_col])
    print(f"rows: {n:,}")

    # --- 1. agreement coverage BY CLASS -----------------------------------
    _hr("1. AGREEMENT BY VARIANT CLASS  (how well is metadata validated per class?)")
    both = (~rs.isin(MISSING_TOKENS)) & (~md.isin(MISSING_TOKENS))
    rows = []
    for c in ["SNV", "deletion", "insertion", "MNV/other"]:
        m = (vcls == c)
        nb = int((m & both).sum())
        ag = int((rs[m & both] == md[m & both]).sum())
        rows.append({"class": c, "class_total": int(m.sum()), "both_populated": nb,
                     "agreeing": ag, "disagreeing": nb - ag,
                     "validation_coverage_pct": round(100 * nb / int(m.sum()), 3) if int(m.sum()) else 0.0})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNOTE: metadata's DELETION values are validated only on the rows where the")
    print("VCF join happened to succeed. Low coverage there = thin evidence for the")
    print("very class metadata would rescue. State this in the incident report.")

    # --- 2. retention by variant class ------------------------------------
    _hr(f"2. RETENTION BY VARIANT CLASS  (keep rows with review_tier <= t)")
    for src, tier in (("ReviewStatus (current)", t_rs), ("metadata (proposed)", t_md)):
        print(f"\n  source: {src}")
        tab = []
        for c in ["SNV", "deletion", "insertion", "MNV/other"]:
            m = (vcls == c)
            tot = int(m.sum())
            row = {"class": c, "total": tot}
            for t in (1, 2, 3, 4, 5):
                row[f"t<={t}"] = int((m & (tier <= t)).sum())
            row[f"pct_kept@{args.min_review_tier}"] = round(
                100 * int((m & (tier <= args.min_review_tier)).sum()) / tot, 3) if tot else 0.0
            tab.append(row)
        print(pd.DataFrame(tab).to_string(index=False))

    _hr(f"2b. DELETION SHARE OF THE SURVIVING COHORT  (min_review_tier={args.min_review_tier})")
    for src, tier in (("ReviewStatus (current)", t_rs), ("metadata (proposed)", t_md)):
        keep = tier <= args.min_review_tier
        k = int(keep.sum())
        kd = int((keep & (vcls == "deletion")).sum())
        print(f"  {src:24s} kept={k:,}  deletions={kd:,}  "
              f"deletion share={100*kd/k:.4f}%" if k else f"  {src}: kept=0")

    # --- 3. label + binary class balance ----------------------------------
    _hr(f"3. LABEL RETENTION AND BINARY CLASS BALANCE  (min_review_tier={args.min_review_tier})")
    lab = df[args.label_col].astype("string").fillna("(null)")
    for src, tier in (("ReviewStatus (current)", t_rs), ("metadata (proposed)", t_md)):
        keep = tier <= args.min_review_tier
        print(f"\n  source: {src}")
        t = pd.DataFrame({"total": lab.value_counts(), "kept": lab[keep].value_counts()}).fillna(0).astype(int)
        t["pct_kept"] = (100 * t["kept"] / t["total"]).round(3)
        print(t.to_string())
        yk = ybin[keep]
        pos, neg = int((yk == 1).sum()), int((yk == 0).sum())
        tot = pos + neg
        print(f"    binary trainable rows : {tot:,}   pos={pos:,}  neg={neg:,}  "
              f"pos_rate={100*pos/tot:.4f}%" if tot else "    binary trainable rows : 0")

    yb_all = ybin
    pos0, neg0 = int((yb_all == 1).sum()), int((yb_all == 0).sum())
    print(f"\n  UNFILTERED binary rows  : {pos0+neg0:,}   pos={pos0:,}  neg={neg0:,}  "
          f"pos_rate={100*pos0/(pos0+neg0):.4f}%")
    print("  AUPRC is sensitive to pos_rate. A filter that changes it changes the")
    print("  primary metric's baseline. Any comparison across runs must hold it fixed.")

    # --- 4. the latent underscore bug -------------------------------------
    _hr("4. LATENT BUG: clean_cohort.py PATHOGENIC_TERMS / BENIGN_TERMS use SPACES")
    vals = sorted(set(lab.unique()))
    print(f"  actual label values          : {vals}")
    hit_cc = [v for v in vals if v.strip().lower() in (CC_PATHOGENIC_TERMS | CC_BENIGN_TERMS)]
    hit_data = [v for v in vals if v.strip().lower() in (DATA_PATHOGENIC | DATA_BENIGN)]
    print(f"  matched by clean_cohort terms: {hit_cc}")
    print(f"  matched by underscore-aware  : {hit_data}")
    missed = sorted(set(hit_data) - set(hit_cc))
    if missed:
        print(f"  ** clean_cohort SILENTLY maps these to -1 (uncertain): {missed}")
        print("  Currently inert (source has 0 duplicate variant_id, so the conflict")
        print("  machinery never runs) -- but it will mis-detect conflicts the moment")
        print("  a duplicate appears. Same defect class as the review-status underscores.")
    else:
        print("  no divergence detected.")

    # --- 5. tier semantics divergence -------------------------------------
    _hr("5. TIER SEMANTICS: substring(default 5) vs exact(default 6)")
    d = pd.DataFrame({
        "RS_substr(real_data_prep)": rs.map(tier_substr).value_counts().sort_index(),
        "RS_exact(clean_cohort)": rs.map(tier_exact).value_counts().sort_index(),
    }).fillna(0).astype(int)
    print(d.to_string())
    print("\n  Blank ReviewStatus is tier 5 to the FILTER and tier 6 to the COHORT")
    print("  BUILDER. Three implementations exist (clean_cohort, augment_reviewstatus,")
    print("  real_data_prep). Reconcile into one function, one documented default.")

    print("\n" + "=" * 76)
    print("This probe asserts nothing about causation in augment_reviewstatus.py.")
    print("Read that file's join key before concluding the mechanism.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
