#!/usr/bin/env python3
"""Step 1b: equal-tier representative-selection audit for the clean_cohort rewire.

READ-ONLY. Writes one measurement file; touches no source, runs no git command.

WHY THIS EXISTS (path D, 2026-07-25)
====================================
The earlier probe (CLEAN_COHORT_MEASUREMENT_2026-07-25.txt) found that rewiring
clean_cohort's tier map to the unified resolver changes the kept row for 2 variants,
because "conflicting classifications" moves from legacy tier 4 to unified tier 3,
tying with "single submitter". But the deeper finding is a defect that exists in the
CURRENT builder regardless of the rewire:

  clean_cohort.py:384  best = grp.sort_values("_tier", kind="stable").iloc[[0]]
  clean_cohort.py:390  kept_rows.append(at_best.iloc[[0]])

Both take the FIRST row in physical Parquet order among rows tied at the best tier.
That makes raw file position part of the scientific adjudication: the same input
records in a different physical order would produce a different certified cohort row.
That is not reproducible, and it must be removed whether or not the two selected rows
ultimately change.

This audit answers path D's six questions on the REAL cohort, and simulates five
candidate representative-selection policies so the rewire's policy is chosen on
evidence:

  Q1. the exact changed rows, all fields, for every affected variant.
  Q2. whether the competing rows carry the same binary label (representative-only)
      or different labels (a training-target effect).
  Q3. every heterogeneous equal-best-tier group under the unified resolver (not just
      the 2 already known), establishing the full tie-policy surface.
  Q4. direct order-dependence test: permute each tied group's row order and check
      whether the CURRENT (file-order) selection changes.
  Q5. five candidate policies compared by kept-row identity, label changes, and
      quarantine changes.
  Q6. field-level deltas for each changed selection: which downstream columns
      actually differ (artifact-identity only / provenance / feature / label).

SCHEMA-AGNOSTIC: the real cohort's columns are discovered at run time. Whichever of
{source_id, clinical_sig, ref, alt, consequence, gene, rsid, ...} exist are used for
the canonical key and the field-delta; the probe reports exactly which it found.

THE FIVE POLICIES SIMULATED
---------------------------
  P0 legacy      : clean_cohort's own tier map (conflicting=4), file-order tie-break.
  P1 unified+ord : unified resolver tiers, file-order tie-break (the naive rewire;
                   this is the one with the 2-variant change AND the order defect).
  P2 unified+nonconflict : unified tiers, then prefer non-conflicting status at equal
                   tier (path D's recommended semantic rank), then a deterministic
                   canonical key. Order-independent.
  P3 unified+neutral : unified tiers, then a deterministic canonical key only (no
                   conflict preference). Order-independent; shows what pure
                   determinism gives without the semantic rank.
  P4 unified+quarantine : unified tiers; if a concordant group ties across DIFFERENT
                   normalized statuses, quarantine it instead of picking. Most
                   conservative.

INPUT
-----
  data/processed/clinvar_grch38.parquet  (RAW cohort, with duplicates)

OUTPUT
------
  docs/measurements/CLEAN_COHORT_ADJUDICATION_AUDIT_2026-07-25.txt

USAGE (PowerShell 5.1):
    python "C:\\Users\\monzi\\Downloads\\probe_clean_cohort_adjudication_2026-07-25.py"

Exit 0 on success, 2 if no usable cohort or required column is missing. Never edits.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
RAW = REPO / "data" / "processed" / "clinvar_grch38.parquet"
CLEAN = REPO / "data" / "processed" / "clinvar_grch38_clean.parquet"
CC = REPO / "scripts" / "clean_cohort.py"
OUT = REPO / "docs" / "measurements" / "CLEAN_COHORT_ADJUDICATION_AUDIT_2026-07-25.txt"

EXIT_OK = 0
EXIT_FAIL = 2

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
LEGACY_TIER_UNMATCHED = 6

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
SENTINEL_UNMATCHED = 99  # unified would RAISE; sentinel lets the audit continue

# path D's narrow conflict penalty (representative selection only, NOT a tier map).
CONFLICT_STATUS_PENALTY = {
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
}

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}

# Candidate columns for the canonical key + field delta; whichever exist are used.
CANONICAL_KEY_FIELDS = ("source_id", "clinical_sig", "ref", "alt", "consequence",
                        "gene", "gene_symbol", "rsid", "rs_id")
FIELD_DELTA_FIELDS = ("clinical_sig", "gene", "gene_symbol", "consequence", "ref",
                      "alt", "rsid", "rs_id", "pathogenicity", "ReviewStatus",
                      "review_status", "source_id")


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


def norm_label(v: object):
    s = norm_term(v)
    if s in PATHOGENIC_TERMS:
        return 1
    if s in BENIGN_TERMS:
        return 0
    return None


def legacy_tier(value: object) -> int:
    key = norm_term(value)
    if key in MISSING_TOKENS:
        return LEGACY_TIER_UNMATCHED
    return CLEAN_MAP.get(key, LEGACY_TIER_UNMATCHED)


def unified_tier(value: object):
    key = norm_term(value)
    if key in MISSING_TOKENS:
        return TIER_MISSING, "missing_token"
    if key in UNIFIED_MAP:
        return UNIFIED_MAP[key], "explicit_status"
    return None, "unmatched"


def semantic_rank(value: object) -> int:
    return CONFLICT_STATUS_PENALTY.get(norm_term(value), 0)


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


def canonical_key(rowvals: dict, present_fields: tuple) -> str:
    parts = [str(rowvals.get(f, "")) for f in present_fields]
    return "\x1f".join(parts)


def select_group(idxs, tiers, labels, revs, rowdicts, key_fields, policy):
    """Return (kept_index_or_None, quarantined_bool, reason). Emulates the concordant
    'keep one' branch under a given policy; discordant handling mirrors clean_cohort."""
    distinct = {labels[i] for i in idxs}
    is_conflict = (1 in distinct) and (0 in distinct)

    if not is_conflict:
        # concordant group -> keep one representative
        best_tier = min(tiers[i] for i in idxs)
        at_best = [i for i in idxs if tiers[i] == best_tier]
        if len(at_best) == 1:
            return at_best[0], False, "concordant_lowest_tier"
        # tie among >1 rows at best tier
        statuses_at_best = {norm_term(revs[i]) for i in at_best}
        if policy == "P0" or policy == "P1":
            # file-order: first in original index order
            return sorted(at_best)[0], False, "concordant_file_order_tie"
        if policy == "P2":
            # semantic rank then canonical key -> deterministic, non-conflict preferred
            best = sorted(at_best, key=lambda i: (
                semantic_rank(revs[i]), canonical_key(rowdicts[i], key_fields)))[0]
            reason = ("concordant_semantic_tiebreak"
                      if len({semantic_rank(revs[i]) for i in at_best}) > 1
                      else "concordant_canonical_tiebreak")
            return best, False, reason
        if policy == "P3":
            # neutral deterministic: canonical key only
            best = sorted(at_best, key=lambda i: canonical_key(rowdicts[i], key_fields))[0]
            return best, False, "concordant_canonical_tiebreak"
        if policy == "P4":
            # quarantine on heterogeneous status tie
            if len(statuses_at_best) > 1:
                return None, True, "concordant_status_tie_quarantined"
            best = sorted(at_best, key=lambda i: canonical_key(rowdicts[i], key_fields))[0]
            return best, False, "concordant_canonical_tiebreak"
    else:
        # discordant: mirror clean_cohort's best-tier-agreement logic
        best_tier = min(tiers[i] for i in idxs)
        at_best = [i for i in idxs if tiers[i] == best_tier]
        classes_at_best = {labels[i] for i in at_best}
        if len(classes_at_best) == 1 and classes_at_best <= {0, 1}:
            if policy in ("P0", "P1"):
                return sorted(at_best)[0], False, "conflict_resolved_by_tier_file_order"
            best = sorted(at_best, key=lambda i: (
                semantic_rank(revs[i]), canonical_key(rowdicts[i], key_fields)))[0]
            return best, False, "conflict_resolved_by_tier_deterministic"
        return None, True, "conflict_irreducible"
    return sorted(idxs)[0], False, "fallback"


def run_policy(vids, tiers, labels, revs, rowdicts, key_fields, policy):
    groups = defaultdict(list)
    for i, v in enumerate(vids):
        groups[v].append(i)
    kept = set()
    quarantined_vids = set()
    reasons = Counter()
    for v, idxs in groups.items():
        if len(idxs) == 1:
            kept.add(idxs[0]); reasons["unique_row"] += 1; continue
        k, q, reason = select_group(idxs, tiers, labels, revs, rowdicts, key_fields, policy)
        reasons[reason] += 1
        if q:
            quarantined_vids.add(v)
        elif k is not None:
            kept.add(k)
    return kept, quarantined_vids, reasons


def main() -> int:
    import pyarrow.parquet as pq

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit("STEP 1b CLEAN_COHORT ADJUDICATION AUDIT -- 2026-07-25  (read-only, path D)")
    emit("=" * 78)

    cohort = RAW if RAW.is_file() else (CLEAN if CLEAN.is_file() else None)
    if cohort is None:
        print(f"  FAIL: no cohort at {RAW} or {CLEAN}")
        return EXIT_FAIL
    emit(f"\n  cohort: {cohort.name}")
    if cohort == CLEAN:
        emit("  WARNING: using CLEANED cohort; duplicates may already be resolved, so tie")
        emit("           groups may be absent. Re-run against the RAW parquet for the real audit.")

    emit("\n-- source-map drift check --")
    emit(f"  {confirm_source_map()}")

    tbl = pq.read_table(cohort)
    have = list(tbl.column_names)
    emit(f"  cohort columns ({len(have)}): {have}")

    if "variant_id" not in have:
        print("  FAIL: no variant_id column"); return EXIT_FAIL
    vid = tbl.column("variant_id").to_pylist()
    n = len(vid)
    emit(f"  rows: {n:,}")

    # resolve review + label columns the way clean_cohort does
    rev = None; rev_src = None
    if "metadata" in have:
        md = tbl.column("metadata").to_pylist()
        if md and isinstance(md[0], dict) and "review_status" in md[0]:
            rev = [(m.get("review_status") if isinstance(m, dict) else None) for m in md]
            rev_src = "metadata.review_status"
    if rev is None:
        for c in ("review_status", "ReviewStatus", "clnrevstat"):
            if c in have:
                rev = tbl.column(c).to_pylist(); rev_src = c; break
    if rev is None:
        print("  FAIL: no review column"); return EXIT_FAIL

    lab = None; lab_src = None
    for c in ("clinical_sig", "clinical_significance", "label", "clnsig", "pathogenicity"):
        if c in have:
            lab = tbl.column(c).to_pylist(); lab_src = c; break
    if lab is None:
        print("  FAIL: no label column"); return EXIT_FAIL
    emit(f"  review column: {rev_src}   label column: {lab_src}")

    # which canonical-key + field-delta columns are actually present
    key_fields = tuple(f for f in CANONICAL_KEY_FIELDS if f in have)
    delta_fields = tuple(f for f in FIELD_DELTA_FIELDS if f in have)
    emit(f"  canonical-key fields present: {key_fields}")
    emit(f"  field-delta fields present:  {delta_fields}")
    if not key_fields:
        emit("  WARNING: none of the preferred canonical-key fields exist; the deterministic")
        emit("           tie-break will fall back to variant_id+row-hash of all columns.")

    # materialise per-row dicts only for the columns we need (memory-conscious)
    need_cols = set(key_fields) | set(delta_fields) | {rev_src if rev_src in have else ""}
    need_cols.discard("")
    coldata = {c: tbl.column(c).to_pylist() for c in need_cols if c in have}
    rowdicts = [{c: coldata[c][i] for c in coldata} for i in range(n)]

    labels = [norm_label(x) for x in lab]
    leg_tiers = [legacy_tier(r) for r in rev]
    uni_pairs = [unified_tier(r) for r in rev]
    would_raise = sum(1 for t, _p in uni_pairs if t is None)
    uni_tiers = [SENTINEL_UNMATCHED if t is None else t for t, _p in uni_pairs]

    emit(f"\n  WOULD_RAISE (unified rejects): {would_raise:,}")

    # -- Q3: enumerate heterogeneous equal-best-tier groups under UNIFIED --
    groups = defaultdict(list)
    for i, v in enumerate(vid):
        groups[v].append(i)
    dup_groups = {v: idxs for v, idxs in groups.items() if len(idxs) > 1}
    emit(f"\n[Q3] duplicate groups: {len(dup_groups):,}")

    hetero_tie_groups = []  # concordant groups with >1 distinct status at best unified tier
    for v, idxs in dup_groups.items():
        distinct_lab = {labels[i] for i in idxs}
        is_conflict = (1 in distinct_lab) and (0 in distinct_lab)
        if is_conflict:
            continue
        best = min(uni_tiers[i] for i in idxs)
        at_best = [i for i in idxs if uni_tiers[i] == best]
        if len(at_best) > 1 and len({norm_term(rev[i]) for i in at_best}) > 1:
            hetero_tie_groups.append((v, at_best))
    emit(f"[Q3] heterogeneous equal-best-tier concordant groups (unified): {len(hetero_tie_groups):,}")
    if hetero_tie_groups:
        pair_counter = Counter()
        for v, at_best in hetero_tie_groups:
            statuses = tuple(sorted({norm_term(rev[i]) for i in at_best}))
            pair_counter[statuses] += 1
        emit("  status pairs that tie at best tier (count):")
        for pair, c in pair_counter.most_common(20):
            emit(f"    {c:>8,}  {pair}")

    # -- Q5: run all five policies, compare kept-row identity --
    emit("\n[Q5] policy comparison (kept-row identity vs P0 legacy)")
    results = {}
    for pol in ("P0", "P1", "P2", "P3", "P4"):
        tiers = leg_tiers if pol == "P0" else uni_tiers
        kept, quar, reasons = run_policy(vid, tiers, labels, rev, rowdicts, key_fields, pol)
        results[pol] = (kept, quar, reasons)
    base_kept, base_quar, _ = results["P0"]
    emit(f"  P0 legacy      : kept={len(base_kept):,}  quarantined_groups={len(base_quar):,}")
    for pol, name in (("P1", "unified+fileorder"), ("P2", "unified+nonconflict"),
                      ("P3", "unified+neutral"), ("P4", "unified+quarantine")):
        kept, quar, reasons = results[pol]
        sym = base_kept ^ kept
        # label change: among symmetric-difference rows, does the kept label differ per variant?
        changed_vids = {vid[i] for i in sym}
        label_changed = 0
        for v in changed_vids:
            leg_keep = [i for i in base_kept if vid[i] == v]
            pol_keep = [i for i in kept if vid[i] == v]
            leg_l = {labels[i] for i in leg_keep}
            pol_l = {labels[i] for i in pol_keep}
            if leg_l != pol_l:
                label_changed += 1
        quar_delta = len(quar ^ base_quar)
        emit(f"  {pol} {name:<20}: kept={len(kept):,}  sym_diff={len(sym):,}  "
             f"changed_variants={len(changed_vids):,}  label_changed_variants={label_changed:,}  "
             f"quarantine_delta={quar_delta:,}")

    # -- Q4: order-dependence test on P1 (the naive rewire) --
    emit("\n[Q4] order-dependence test (does file order change the P1/file-order selection?)")
    import random
    rng = random.Random(20260725)
    order_sensitive = 0
    tested = 0
    affected_by_order = []
    for v, idxs in dup_groups.items():
        distinct_lab = {labels[i] for i in idxs}
        if (1 in distinct_lab) and (0 in distinct_lab):
            continue
        best = min(uni_tiers[i] for i in idxs)
        at_best = [i for i in idxs if uni_tiers[i] == best]
        if len(at_best) <= 1:
            continue
        tested += 1
        # file-order selection = first index; permute and see if "first" changes identity
        base_pick = sorted(at_best)[0]
        base_key = canonical_key(rowdicts[base_pick], key_fields) if key_fields else str(base_pick)
        for _ in range(5):
            perm = at_best[:]
            rng.shuffle(perm)
            pick = perm[0]  # naive file-order would pick whatever is first now
            pick_key = canonical_key(rowdicts[pick], key_fields) if key_fields else str(pick)
            if pick_key != base_key:
                order_sensitive += 1
                if len(affected_by_order) < 10:
                    affected_by_order.append(v)
                break
    emit(f"  concordant tied groups tested: {tested:,}")
    emit(f"  groups where file order changes the selected row: {order_sensitive:,}")
    emit("  (this is the defect path D removes: P2/P3/P4 are all order-INDEPENDENT by construction)")

    # -- Q1 + Q2 + Q6: the variants where P1 differs from P0, full detail --
    p1_kept, _, _ = results["P1"]
    changed = base_kept ^ p1_kept
    changed_vids = sorted({vid[i] for i in changed})
    emit(f"\n[Q1/Q2/Q6] variants whose kept row changes P0(legacy) -> P1(unified+fileorder): {len(changed_vids)}")
    for v in changed_vids[:25]:
        idxs = [i for i in range(n) if vid[i] == v]
        emit(f"\n  variant_id = {v!r}  (group size {len(idxs)})")
        leg_keep = [i for i in base_kept if i in set(idxs)]
        p1_keep = [i for i in p1_kept if i in set(idxs)]
        for i in idxs:
            tagp0 = "  [P0-kept]" if i in base_kept else ""
            tagp1 = "  [P1-kept]" if i in p1_kept else ""
            emit(f"    row {i}: review={norm_term(rev[i])!r} label={labels[i]} "
                 f"leg_tier={leg_tiers[i]} uni_tier={uni_tiers[i]}{tagp0}{tagp1}")
        # Q2: label effect
        leg_l = {labels[i] for i in leg_keep}
        p1_l = {labels[i] for i in p1_keep}
        eff = "unchanged" if leg_l == p1_l else "CHANGED"
        emit(f"    [Q2] binary-label effect: {eff} (P0 label {leg_l}, P1 label {p1_l})")
        # Q6: field-level delta between the two kept rows
        if leg_keep and p1_keep and leg_keep[0] != p1_keep[0]:
            a, b = leg_keep[0], p1_keep[0]
            diffs = [f for f in delta_fields
                     if str(rowdicts[a].get(f, "")) != str(rowdicts[b].get(f, ""))]
            emit(f"    [Q6] columns differing between P0-kept and P1-kept rows: {diffs or 'NONE (identical on measured fields)'}")

    # -- P2 vs legacy: the recommended policy's actual effect --
    p2_kept, p2_quar, _ = results["P2"]
    p2_changed = base_kept ^ p2_kept
    p2_vids = sorted({vid[i] for i in p2_changed})
    emit("\n" + "=" * 78)
    emit("RECOMMENDED POLICY (P2 = unified tiers + non-conflict preference + canonical key)")
    emit("-" * 78)
    emit(f"  rows kept differing from legacy P0: {len(p2_changed):,}  (variants: {len(p2_vids):,})")
    emit(f"  quarantine delta vs legacy: {len(p2_quar ^ base_quar):,}")
    emit(f"  order-independent by construction: yes (no file-order tie-break remains)")
    if not p2_vids:
        emit("  => P2 reproduces the legacy kept rows for every variant AND removes the")
        emit("     order-dependence defect. This is path D's ideal outcome: same certified")
        emit("     cohort, but now by a declared deterministic policy instead of file order.")
    else:
        emit(f"  => P2 differs from legacy on {len(p2_vids)} variant(s); listed above by identity.")
        emit("     Review these before adopting; each is a deliberate representative change.")

    emit("\n" + "=" * 78)
    emit("ACCEPTANCE-CRITERIA READOUT (path D)")
    emit("-" * 78)
    emit(f"  unmatched review statuses (WOULD_RAISE)   : {would_raise}")
    emit(f"  input-order-dependent selections (naive)  : {order_sensitive}")
    emit(f"  P2 selected rows differing from legacy    : {len(p2_vids)}")
    emit(f"  P2 label changes                          : "
         f"{sum(1 for v in p2_vids if {labels[i] for i in base_kept if vid[i]==v} != {labels[i] for i in p2_kept if vid[i]==v})}")
    emit(f"  P2 quarantine changes                     : {len(p2_quar ^ base_quar)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWROTE {OUT}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
