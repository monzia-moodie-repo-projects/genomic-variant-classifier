#!/usr/bin/env python3
"""Step 1b measurement: does the clean_cohort tier rewire change which row is kept?

READ-ONLY. Writes one measurement file; touches no source, runs no git command.

WHY THIS EXISTS
===============
clean_cohort is the certified cohort builder. Unlike augment and real_data_prep,
its review-tier map is NOT a near-copy of the unified resolver: its whole bottom
scale is shifted by one, and -- the part that matters -- it places conflicting
classifications at tier 4 while the unified resolver places them at tier 3.

The tier is used ONLY for relative ordering inside duplicate variant_id groups
(clean_cohort.py:380-396): among rows sharing a variant_id, the row with the
LOWEST tier is kept (sort_values("_tier").iloc[0]; and grp["_tier"].min() picks
the authoritative class in a label conflict). Absolute tier values are irrelevant;
only the ORDER within a group matters.

A monotonic scale shift preserves order and cannot change the kept row. But one
difference is NOT a monotonic shift:

  status                             clean_cohort   unified
  criteria provided, single submit.       3            3
  criteria provided, CONFLICTING          4            3   (TIES with single submitter)

Under clean_cohort's map, a single-submitter row (3) strictly outranks a
conflicting row (4). Under the unified map they TIE at 3, and the stable sort keeps
whichever appears first. So in any duplicate group containing both a
tier<=3 row and a conflicting row, the KEPT ROW can change. This probe measures --
by row identity, on the real cohort -- whether that actually happens, and how often.

It reconstructs clean_cohort's exact pipeline (resolve the review column, normalise,
compute the per-row tier, run the duplicate-group resolution) under BOTH the legacy
map (with TIER_UNMATCHED=6) and the unified resolver, and compares the kept-row
identity sets. It also reports the reconciliation counts under both, and confirms
WOULD_RAISE on the review column (the unified resolver's strict behaviour).

INPUTS
------
  data/processed/clinvar_grch38.parquet  (the RAW cohort clean_cohort operates on,
  BEFORE cleaning; it carries duplicate variant_ids and the nested
  metadata.review_status column). If that path is absent the probe tries the
  cleaned path and reports that duplicates may already be resolved there.

OUTPUT
------
  docs/measurements/CLEAN_COHORT_MEASUREMENT_2026-07-25.txt

USAGE (PowerShell 5.1):
    python "C:\\Users\\monzi\\Downloads\\probe_clean_cohort_2026-07-25.py"

Exit 0 on success, 2 if no usable cohort or required column is missing. Never edits.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
RAW = REPO / "data" / "processed" / "clinvar_grch38.parquet"
CLEAN = REPO / "data" / "processed" / "clinvar_grch38_clean.parquet"
CC = REPO / "scripts" / "clean_cohort.py"
OUT = REPO / "docs" / "measurements" / "CLEAN_COHORT_MEASUREMENT_2026-07-25.txt"

EXIT_OK = 0
EXIT_FAIL = 2

# clean_cohort's legacy map, transcribed from clean_cohort.py:126 on 2026-07-25.
CLEAN_MAP = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 4,
    "criteria provided, conflicting interpretations": 4,
    "no assertion criteria provided": 5,
    "no classification provided": 6,
    "no classification for the single variant": 6,
    "no classification for the individual variant": 6,
}
TIER_UNMATCHED = 6

# Unified resolver map, mirrored from review_status.py.
UNIFIED_MAP = {
    "practice guideline": 1, "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 3,
    "criteria provided, conflicting interpretations": 3,
    "no assertion criteria provided": 4,
    "no classification provided": 5,
    "no classification for the single variant": 5,
    "no classification for the individual variant": 5,
    "no classifications from unflagged records": 5,
}
MISSING_TOKENS = {"", "-", ".", "na", "nan", "none", "null", "<na>"}
TIER_MISSING = 5

# clean_cohort's label normalization (clean_cohort.py:140-141, _normalize_label).
PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}


def norm_term(v: object) -> str:
    if v is None:
        return ""
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return " ".join(str(v).strip().lower().replace("_", " ").split())


def norm_label(v: object) -> object:
    """Mirror _normalize_label: pathogenic->1, benign->0, else NaN (pd.NA sentinel)."""
    s = norm_term(v)
    if s in PATHOGENIC_TERMS:
        return 1
    if s in BENIGN_TERMS:
        return 0
    return None


def legacy_tier(value: object) -> int:
    key = norm_term(value)
    if key in MISSING_TOKENS:
        return TIER_UNMATCHED  # clean_cohort maps missing -> NaN -> fillna(TIER_UNMATCHED)
    return CLEAN_MAP.get(key, TIER_UNMATCHED)


def unified_tier(value: object) -> tuple[int | None, str]:
    key = norm_term(value)
    if key in MISSING_TOKENS:
        return TIER_MISSING, "missing_token"
    if key in UNIFIED_MAP:
        return UNIFIED_MAP[key], "explicit_status"
    return None, "unmatched"


def confirm_source_map() -> str:
    if not CC.is_file():
        return f"DRIFT: {CC} absent"
    tree = ast.parse(CC.read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "REVIEW_STATUS_TIER":
                    try:
                        found = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        found = None
    if found is None:
        return "DRIFT: could not read REVIEW_STATUS_TIER from clean_cohort.py"
    if found != CLEAN_MAP:
        return "DRIFT: clean_cohort REVIEW_STATUS_TIER changed -- re-transcribe the probe"
    return f"OK: clean_cohort REVIEW_STATUS_TIER matches transcription ({len(CLEAN_MAP)} keys)"


def resolve_duplicates(vids, labels, tiers):
    """Reconstruct clean_cohort's duplicate-group resolution. Returns the set of
    kept row indices and the reconciliation counts. `tiers` is a per-row list of
    ints (missing/unmatched already folded in by the caller)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, v in enumerate(vids):
        groups[v].append(i)

    kept: set[int] = set()
    n_exact_dup_dropped = 0
    n_conflict_resolved_dropped = 0
    n_conflict_rows = 0

    for v, idxs in groups.items():
        if len(idxs) == 1:
            kept.add(idxs[0])
            continue
        distinct = {labels[i] for i in idxs}
        is_conflict = (1 in distinct) and (0 in distinct)
        if not is_conflict:
            # keep the lowest tier; emulate sort_values("_tier", kind="stable").iloc[0]
            # -- a stable sort preserves original order among equal tiers, so the
            # first-appearing row at the minimum tier wins.
            best_i = sorted(idxs, key=lambda i: tiers[i])[0]
            kept.add(best_i)
            n_exact_dup_dropped += len(idxs) - 1
        else:
            best_tier = min(tiers[i] for i in idxs)
            at_best = [i for i in idxs if tiers[i] == best_tier]
            classes_at_best = {labels[i] for i in at_best}
            if len(classes_at_best) == 1 and classes_at_best <= {0, 1}:
                kept.add(at_best[0])
                n_conflict_resolved_dropped += len(idxs) - 1
            else:
                # conflict rows are NOT kept in clean; counted separately
                n_conflict_rows += len(idxs)
    return kept, n_exact_dup_dropped, n_conflict_resolved_dropped, n_conflict_rows


def main() -> int:
    import pyarrow.parquet as pq

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit("STEP 1b CLEAN_COHORT MEASUREMENT -- 2026-07-25  (read-only)")
    emit("does the tier rewire change which row is kept in a duplicate group?")
    emit("=" * 78)

    cohort = RAW if RAW.is_file() else (CLEAN if CLEAN.is_file() else None)
    if cohort is None:
        print(f"  FAIL: no cohort at {RAW} or {CLEAN}")
        return EXIT_FAIL
    emit(f"\n  cohort: {cohort.name}")
    if cohort == CLEAN:
        emit("  NOTE: raw cohort absent; using the CLEANED cohort. Duplicates may already")
        emit("        be resolved there, so the duplicate-group divergence may read 0 for")
        emit("        lack of duplicates rather than for lack of effect. Re-run against the")
        emit("        raw pre-clean parquet for the definitive measurement.")

    emit("\n-- source-map drift check --")
    emit(f"  {confirm_source_map()}")

    tbl = pq.read_table(cohort)
    have = set(tbl.column_names)
    emit(f"  columns available: {len(have)}")

    if "variant_id" not in have:
        print("  FAIL: no variant_id column")
        return EXIT_FAIL
    vid = tbl.column("variant_id").to_pylist()
    n = len(vid)
    emit(f"  rows: {n:,}")

    # resolve the review column the way clean_cohort does: prefer metadata.review_status
    rev = None
    src_desc = None
    if "metadata" in have:
        md = tbl.column("metadata").to_pylist()
        if md and isinstance(md[0], dict) and "review_status" in md[0]:
            rev = [(m.get("review_status") if isinstance(m, dict) else None) for m in md]
            src_desc = "metadata.review_status (nested)"
    if rev is None:
        for cand in ("review_status", "ReviewStatus", "clnrevstat"):
            if cand in have:
                rev = tbl.column(cand).to_pylist()
                src_desc = cand
                break
    if rev is None:
        print("  FAIL: no review column resolvable")
        return EXIT_FAIL
    emit(f"  review column: {src_desc}")

    # label column: prefer clinical_sig / clinical_significance
    lab = None
    lab_desc = None
    for cand in ("clinical_sig", "clinical_significance", "label", "clnsig"):
        if cand in have:
            lab = tbl.column(cand).to_pylist()
            lab_desc = cand
            break
    if lab is None:
        print("  FAIL: no label column resolvable")
        return EXIT_FAIL
    emit(f"  label column:  {lab_desc}")

    # per-row normalized label + both tiers
    labels = [norm_label(x) for x in lab]
    leg_tiers = [legacy_tier(r) for r in rev]
    uni_tiers_raw = [unified_tier(r) for r in rev]
    would_raise = sum(1 for t, _p in uni_tiers_raw if t is None)
    # for the resolution comparison, unmatched (None) must be given a sortable value.
    # The strict resolver would RAISE, aborting the build. To measure the ordering
    # effect in isolation we place unmatched at a sentinel WORSE than any real tier
    # (so it never wins) and report would_raise separately.
    SENTINEL = 99
    uni_tiers = [SENTINEL if t is None else t for t, _p in uni_tiers_raw]

    emit(f"\n  WOULD_RAISE (review statuses unified rejects): {would_raise:,}")
    if would_raise:
        emit("  *** unified resolver would ABORT the build on these; strict rewire changes")
        emit("      behaviour here. Listing distinct unmatched values:")
        um = Counter(norm_term(rev[i]) for i, (t, _p) in enumerate(uni_tiers_raw) if t is None)
        for val, c in um.most_common(20):
            emit(f"      {c:>8,}  {val!r}")

    # duplicate-group resolution under both maps
    emit("\n-- duplicate-group resolution: legacy map vs unified map --")
    from collections import Counter as C
    vc = C(vid)
    n_dup_groups = sum(1 for v, c in vc.items() if c > 1)
    n_dup_rows = sum(c for v, c in vc.items() if c > 1)
    emit(f"  duplicate variant_id groups: {n_dup_groups:,} ({n_dup_rows:,} rows)")

    keep_leg, leg_exact, leg_confres, leg_confrows = resolve_duplicates(vid, labels, leg_tiers)
    keep_uni, uni_exact, uni_confres, uni_confrows = resolve_duplicates(vid, labels, uni_tiers)

    emit(f"  legacy : kept={len(keep_leg):,}  exact_dup_dropped={leg_exact:,}  "
         f"conflict_resolved_dropped={leg_confres:,}  conflict_rows={leg_confrows:,}")
    emit(f"  unified: kept={len(keep_uni):,}  exact_dup_dropped={uni_exact:,}  "
         f"conflict_resolved_dropped={uni_confres:,}  conflict_rows={uni_confrows:,}")

    # THE governing comparison: do the kept-row identity sets differ?
    sym = keep_leg ^ keep_uni
    only_leg = keep_leg - keep_uni
    only_uni = keep_uni - keep_leg
    emit("\n-- KEPT-ROW IDENTITY DIVERGENCE (the certified-cohort delta) --")
    emit(f"  kept only under legacy : {len(only_leg):,}")
    emit(f"  kept only under unified: {len(only_uni):,}")
    emit(f"  symmetric difference   : {len(sym):,}")

    if sym:
        emit("\n  the rewire CHANGES the certified cohort. Sampling up to 10 affected groups:")
        affected_vids = {vid[i] for i in sym}
        shown = 0
        for v in affected_vids:
            idxs = [i for i in range(n) if vid[i] == v]
            leg_kept = [i for i in idxs if i in keep_leg]
            uni_kept = [i for i in idxs if i in keep_uni]
            emit(f"    variant_id={v!r}")
            for i in idxs:
                emit(f"      row {i}: review={norm_term(rev[i])!r} label={labels[i]} "
                     f"leg_tier={leg_tiers[i]} uni_tier={uni_tiers[i]}"
                     f"{'  [leg-kept]' if i in keep_leg else ''}"
                     f"{'  [uni-kept]' if i in keep_uni else ''}")
            shown += 1
            if shown >= 10:
                break
    else:
        emit("\n  the rewire keeps EXACTLY the same rows: no certified-cohort change from the")
        emit("  tier map difference. The conflicting-tier tie does not flip any kept row on")
        emit("  this cohort.")

    emit("\n" + "=" * 78)
    emit("INTERPRETATION")
    emit("-" * 78)
    emit("  clean_cohort keeps the lowest-tier row per duplicate variant_id group. The only")
    emit("  map difference that can reorder is conflicting-classifications moving from tier 4")
    emit("  (legacy, strictly below single-submitter) to tier 3 (unified, tied with it). If")
    emit("  the symmetric difference above is 0, the rewire produces a byte-identical certified")
    emit("  cohort and proceeds like the augment and real_data_prep rewires. If nonzero, each")
    emit("  affected group is listed with the competing rows, and the change of representative")
    emit("  is a deliberate decision (conflicting->3 is arguably more correct, but it alters a")
    emit("  certified artifact) rather than something to fold silently into a consolidation.")
    emit("  Separately, WOULD_RAISE > 0 would mean the strict resolver aborts the build on")
    emit("  vocabulary the legacy map tolerated at tier 6 -- a contract change to weigh too.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWROTE {OUT}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
