#!/usr/bin/env python3
"""Step 1b measurement: the label-filter interaction for the real_data_prep rewire.

READ-ONLY. Writes one measurement file; touches no source, runs no git command.

WHY THIS EXISTS
===============
The consumer probe (CONSUMER_PROBE_2026-07-24.txt, commit cd635b2) measured the
review-tier filter IN ISOLATION and found that at the production threshold
min_review_tier=3 the substring->exact map change moves 157,229 rows by identity --
the criteria-provided-conflicting-classifications rows, tier 5 under the legacy
substring map, tier 3 under the unified exact map.

But real_data_prep._load_and_label applies THREE filters in sequence
(real_data_prep.py:505-561), and the tier filter is the SECOND, not the first:

  1. LABEL filter (line 516): keep rows whose clinical_sig is in PATHOGENIC_TERMS
     or BENIGN_TERMS; drop everything else (VUS, conflicting significance).
  2. TIER filter (line 536): among the survivors, keep review_tier <= min_review_tier.
     THIS is the local substring map being rewired.
  3. CONFLICT filter (line 557): if exclude_conflicting, drop clinical_sig containing
     "onflict".

Because the label filter runs FIRST, the training-set impact of the tier-map rewire
is NOT the 157,229. It is: among rows that SURVIVE the label filter, does the
substring->exact change move ANY row across the <= min_review_tier threshold? That
is PHASE1_SPEC section 6 measurement 3, and it is the prerequisite for the
real_data_prep rewire. This probe answers it BY ROW IDENTITY, not by count.

clinical_sig (the label source) and ReviewStatus (the tier source) are INDEPENDENT
columns: a Pathogenic variant can carry any review status. So the label-filtered set
carries the full range of ReviewStatus values and the tier map genuinely applies to
it -- the question of whether substring and exact disagree on those surviving rows is
real and is measured here, not assumed.

WHAT IT MEASURES, on the real cohort
------------------------------------
  * label-filter retention: |cohort| -> |labeled|, with the class balance.
  * of the 157,229 conflicting-classification rows, how many survive the label
    filter (expected 0, because their clinical_sig is in neither term set).
  * on the LABELED set, the tier distribution under the legacy substring map and
    under the unified resolver, side by side.
  * at every threshold 1..5, the kept-row IDENTITY sets under both maps and their
    symmetric difference -- the training-set delta of the rewire.
  * the resolution-path distribution and WOULD_RAISE count on the labeled set.

INPUTS
------
  data/processed/clinvar_grch38_clean.parquet  (the augmented cohort; needs the
  ReviewStatus column added by scripts/augment_reviewstatus.py).
Reads only the columns it needs, natively via pyarrow.

OUTPUT
------
  docs/measurements/LABEL_FILTER_MEASUREMENT_2026-07-24.txt

USAGE (PowerShell 5.1):
    python "C:\\Users\\monzi\\Downloads\\probe_label_filter_2026-07-24.py"

Exit 0 on success, 2 if the cohort or a required column is missing. Never edits.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
COHORT = REPO / "data" / "processed" / "clinvar_grch38_clean.parquet"
RDP = REPO / "src" / "genomic_variant_classifier" / "data" / "real_data_prep.py"
OUT = REPO / "docs" / "measurements" / "LABEL_FILTER_MEASUREMENT_2026-07-24.txt"

EXIT_OK = 0
EXIT_FAIL = 2

# Legacy substring map, transcribed from real_data_prep.py:132 on 2026-07-24.
# The probe re-reads the source at run time to confirm this is still current.
RDP_MAP = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "no assertion criteria provided": 4, "no classification provided": 5,
    "no classification for the individual variant": 5,
}

# Label term sets, transcribed from real_data_prep.py:142-152.
PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}

# Unified map + normalization mirrored from review_status.py.
UNIFIED_MAP = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 3,
    "criteria provided, conflicting interpretations": 3,
    "no assertion criteria provided": 4, "no classification provided": 5,
    "no classification for the single variant": 5,
    "no classification for the individual variant": 5,
    "no classifications from unflagged records": 5,
}
MISSING_TOKENS = frozenset({"", "-", ".", "na", "nan", "none", "null", "<na>"})
TIER_MISSING = 5


def norm_term(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower().replace("_", " ")
    return " ".join(s.split())


def legacy_tier(value: object) -> int:
    s = str(value).lower()
    for k, v in RDP_MAP.items():
        if k in s:
            return v
    return 5


def unified_tier(value: object) -> tuple[int | None, str]:
    key = norm_term(value)
    if key in MISSING_TOKENS:
        return TIER_MISSING, "missing_token"
    if key in UNIFIED_MAP:
        return UNIFIED_MAP[key], "explicit_status"
    return None, "unmatched"


def confirm_source_map() -> str:
    if not RDP.is_file():
        return f"DRIFT: {RDP} absent"
    tree = ast.parse(RDP.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "REVIEW_STATUS_TIER":
                    try:
                        found = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        found = None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "REVIEW_STATUS_TIER" and node.value is not None:
            try:
                found = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                found = None
    if found is None:
        return "DRIFT: could not read REVIEW_STATUS_TIER from real_data_prep.py"
    if found != RDP_MAP:
        return "DRIFT: real_data_prep REVIEW_STATUS_TIER changed -- re-transcribe the probe"
    return f"OK: real_data_prep REVIEW_STATUS_TIER matches transcription ({len(RDP_MAP)} keys)"


def main() -> int:
    import pyarrow.parquet as pq

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit("STEP 1b LABEL-FILTER MEASUREMENT -- 2026-07-24  (read-only)")
    emit("real_data_prep rewire prerequisite: PHASE1_SPEC section 6 measurement 3")
    emit("=" * 78)

    if not COHORT.is_file():
        print(f"  FAIL: cohort absent at {COHORT}")
        return EXIT_FAIL

    emit("\n-- source-map drift check --")
    emit(f"  {confirm_source_map()}")

    tbl = pq.read_table(COHORT)
    have = set(tbl.column_names)
    for need in ("variant_id", "ReviewStatus", "clinical_sig"):
        emit(f"  {'OK  ' if need in have else 'MISSING'} column {need}")
        if need not in have:
            print(f"  FAIL: required column {need} absent")
            return EXIT_FAIL

    vid = tbl.column("variant_id").to_pylist()
    rev = tbl.column("ReviewStatus").to_pylist()
    sig_raw = tbl.column("clinical_sig").to_pylist()
    n = len(vid)
    emit(f"\n  cohort rows: {n:,}")

    # ---- filter 1: label ---------------------------------------------------
    sig = [("" if s is None else str(s).strip()) for s in sig_raw]
    label_set = PATHOGENIC_TERMS | BENIGN_TERMS
    labeled_idx = [i for i in range(n) if sig[i] in label_set]
    n_path = sum(1 for i in labeled_idx if sig[i] in PATHOGENIC_TERMS)
    n_benign = sum(1 for i in labeled_idx if sig[i] in BENIGN_TERMS)
    emit("\n-- FILTER 1: label (clinical_sig in PATHOGENIC_TERMS | BENIGN_TERMS) --")
    emit(f"  cohort {n:,} -> labeled {len(labeled_idx):,} "
         f"({n - len(labeled_idx):,} VUS/conflicting/other removed)")
    emit(f"  class balance: pathogenic={n_path:,}  benign={n_benign:,}")

    # of the conflicting-classification rows, how many survive the label filter?
    conflict_sig_idx = [i for i in range(n)
                        if "onflict" in sig[i].lower()]
    conflict_in_labeled = [i for i in conflict_sig_idx if sig[i] in label_set]
    emit(f"  rows with 'conflict' in clinical_sig: {len(conflict_sig_idx):,}")
    emit(f"  of those, surviving the label filter: {len(conflict_in_labeled):,} "
         f"(expected 0 -- conflicting significance is in neither term set)")

    # ---- on the LABELED set: tier both ways --------------------------------
    emit("\n-- on the LABELED set: tier distribution, legacy vs unified --")
    leg_dist: Counter = Counter()
    uni_dist: Counter = Counter()
    uni_path: Counter = Counter()
    would_raise = 0
    unmatched_vals: Counter = Counter()
    for i in labeled_idx:
        leg_dist[legacy_tier(rev[i])] += 1
        t, pth = unified_tier(rev[i])
        uni_path[pth] += 1
        if t is None:
            would_raise += 1
            uni_dist["WOULD_RAISE"] += 1
            unmatched_vals[norm_term(rev[i])] += 1
        else:
            uni_dist[t] += 1
    emit("  legacy substring tier distribution (labeled rows):")
    for t in sorted(leg_dist):
        emit(f"    tier {t}: {leg_dist[t]:>10,}")
    emit("  unified resolver tier distribution (labeled rows):")
    for t in sorted(uni_dist, key=lambda x: (isinstance(x, str), x)):
        emit(f"    tier {t}: {uni_dist[t]:>10,}")
    emit("  unified resolution-path distribution (labeled rows):")
    for pth in sorted(uni_path):
        emit(f"    {pth}: {uni_path[pth]:>10,}")
    if unmatched_vals:
        emit("  WOULD_RAISE values on labeled rows:")
        for val, c in unmatched_vals.most_common():
            emit(f"    {c:>8,}  {val!r}")
    else:
        emit(f"  WOULD_RAISE on labeled rows: {would_raise} (strict resolve rejects nothing)")

    # ---- the training-set delta: identity sym-diff at each threshold -------
    emit("\n-- TRAINING-SET DELTA: kept-row identity, legacy vs unified, per threshold --")
    emit("  (this is the real_data_prep rewire's true impact: rows entering/leaving")
    emit("   the tier<=T training set purely because the map changed)")
    for thr in range(1, 6):
        keep_leg = {vid[i] for i in labeled_idx if legacy_tier(rev[i]) <= thr}
        keep_uni = set()
        for i in labeled_idx:
            t, _p = unified_tier(rev[i])
            # strict resolver: unmatched has no tier and cannot pass a <=T filter;
            # count it as excluded (it would raise in production, measured above).
            if t is not None and t <= thr:
                keep_uni.add(vid[i])
        sym = keep_leg ^ keep_uni
        only_leg = keep_leg - keep_uni
        only_uni = keep_uni - keep_leg
        emit(f"    thr {thr}: legacy_keep={len(keep_leg):>10,}  "
             f"unified_keep={len(keep_uni):>10,}  sym_diff={len(sym):>8,}  "
             f"(only_legacy={len(only_leg):,} only_unified={len(only_uni):,})")

    emit("\n" + "=" * 78)
    emit("INTERPRETATION")
    emit("-" * 78)
    emit("  The production threshold is min_review_tier=3. If sym_diff at thr 3 is 0, the")
    emit("  real_data_prep rewire moves NO training rows: substring and exact agree on")
    emit("  every review status that survives the label filter, so the map change is a")
    emit("  pure correctness improvement with no cohort-composition effect. If nonzero,")
    emit("  only_legacy and only_unified name exactly which rows change and by how much,")
    emit("  and the class balance of that delta must be examined before the rewire lands.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWROTE {OUT}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
