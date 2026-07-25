#!/usr/bin/env python3
"""Step 1b: P5 + P6 group-level evidence-adjudication audit for clean_cohort.

READ-ONLY. Writes one measurement file; touches no source, runs no git command.

WHY THIS EXISTS (decision 2026-07-25, superseding the P0-P4 audit)
==================================================================
The P0-P4 audit (CLEAN_COHORT_ADJUDICATION_AUDIT_2026-07-25.txt) established:
  * 1,467 of 1,502 concordant best-tier ties are order-sensitive (97.7%): the current
    certified builder selects the representative row by physical Parquet order;
  * P2 (unified tiers + non-conflict preference + canonical key) diverges from legacy
    on 210 variants and changes 53 binary labels -- because legacy was choosing many
    representatives by file-order accident, and a deterministic key reassigns them.

A first proposed fix -- P5, adding a "label eligibility" term to the representative
key so a trainable row is preferred -- removes the symptom but is scientifically
circular: it makes the foundational cohort prefer whichever row the CURRENT binary
task can use, entangling record adjudication with downstream model eligibility and
biasing against uncertainty/conflict. It would also make the cohort change whenever
the label ontology expands (ordinal, somatic, risk alleles), with ClinVar unchanged.

The correct architecture (P6) SEPARATES two independent decisions per variant_id:

  A. Canonical evidence state (the LABEL) -- adjudicated from the authority-qualified
     evidence group G* = {rows at the best tier}, NOT from any single chosen row.
  B. Representative metadata row -- chosen independently and deterministically; it
     supplies row-level metadata but NEVER determines the canonical label.

Governing invariant:
  The variant's clinical label is adjudicated from the full authority-qualified
  evidence group. The representative row supplies metadata only. Neither physical row
  order nor downstream label eligibility may manufacture, remove, or change the label.

This probe adds P5 (diagnostic counterfactual) and P6 (the recommended architecture),
classifies the label-changing groups into 7 mechanistic categories, emits the full
P0-P6 policy table, and runs P6 order-invariance under original / reverse / random
permutations. It does NOT rewire anything and it does NOT adopt a policy: it produces
the evidence for the cohort-v2 decision.

VOCABULARY-DRIVEN + FAIL-LOUD: the clinical_sig evidence-state classifier reports any
value it cannot classify rather than silently bucketing it, so no category hides.

INPUT   : data/processed/clinvar_grch38.parquet  (RAW cohort, with duplicates)
OUTPUT  : docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25.txt

USAGE (PowerShell 5.1):
    python "C:\\Users\\monzi\\Downloads\\probe_clean_cohort_p6_2026-07-25.py"

Exit 0 on success, 2 if no usable cohort or required column is missing. Never edits.
"""
from __future__ import annotations

import ast
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
RAW = REPO / "data" / "processed" / "clinvar_grch38.parquet"
CLEAN = REPO / "data" / "processed" / "clinvar_grch38_clean.parquet"
CC = REPO / "scripts" / "clean_cohort.py"
OUT = REPO / "docs" / "measurements" / "CLEAN_COHORT_P6_AUDIT_2026-07-25.txt"

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
SENTINEL_UNMATCHED = 99

CONFLICT_STATUS_PENALTY = {
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
}

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}

# clinical_sig evidence-state vocabulary (vocabulary-driven; unknowns are reported).
UNCERTAIN_TERMS = {"uncertain significance", "uncertain risk allele",
                   "uncertain significance/uncertain risk allele"}
CONFLICT_SIG_TERMS = {"conflicting classifications of pathogenicity",
                      "conflicting interpretations of pathogenicity",
                      "conflicting classifications of pathogenicity and risk allele"}
NONBINARY_TERMS = {"drug response", "association", "risk factor", "protective",
                   "affects", "other", "not provided", "confers sensitivity",
                   "association not found", "low penetrance"}
NOCLASS_TERMS = {"no classification provided",
                 "no classification for the single variant",
                 "no classifications from unflagged records", ""}

CANONICAL_KEY_FIELDS = ("source_id", "clinical_sig", "ref", "alt", "consequence",
                        "gene_symbol", "gene", "rsid", "rs_id")
DELTA_FIELDS = ("clinical_sig", "gene_symbol", "gene", "consequence", "ref", "alt",
                "rsid", "rs_id", "pathogenicity", "source_id")


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


# ---- evidence-state classifier (P6 rule A); returns (state, is_recognised) ----
def evidence_state(clinical_sig: object):
    s = norm_term(clinical_sig)
    if s in PATHOGENIC_TERMS:
        return "PATHOGENIC", True
    if s in BENIGN_TERMS:
        return "BENIGN", True
    if s in UNCERTAIN_TERMS:
        return "UNCERTAIN", True
    if s in CONFLICT_SIG_TERMS or "conflicting" in s:
        return "EXPLICIT_CONFLICT", True
    if s in NOCLASS_TERMS:
        return "NO_CLASSIFICATION", True
    if s in NONBINARY_TERMS:
        return "NON_BINARY", True
    return "UNRECOGNISED", False


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


def canonical_key(rowdict: dict, fields: tuple) -> str:
    return "\x1f".join(str(rowdict.get(f, "")) for f in fields)


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


# =====================================================================
# P0-P5: single-row representative selection (legacy conflation model)
# =====================================================================
def select_repr_row(idxs, tiers, labels, revs, rowdicts, kf, policy, order):
    """Return (kept_idx | None, quarantined, reason). 'order' is the row order to use
    for file-order policies (P0/P1)."""
    distinct = {labels[i] for i in idxs}
    is_conflict = (1 in distinct) and (0 in distinct)
    # File-order policies (P0/P1) need this group's rows in ascending original-index
    # order. That is exactly sorted(idxs); we must NOT scan the global row list here
    # (doing so was O(groups * N) -- accidentally quadratic on the full cohort).
    ordered = sorted(idxs)

    if not is_conflict:
        best = min(tiers[i] for i in idxs)
        at_best = [i for i in idxs if tiers[i] == best]
        if len(at_best) == 1:
            return at_best[0], False, "concordant_lowest_tier"
        if policy in ("P0", "P1"):
            best_set = set(at_best)
            return next(i for i in ordered if i in best_set), False, "concordant_file_order"
        if policy == "P2":
            best_i = sorted(at_best, key=lambda i: (semantic_rank(revs[i]), canonical_key(rowdicts[i], kf)))[0]
            return best_i, False, "concordant_semantic_then_canonical"
        if policy == "P3":
            return sorted(at_best, key=lambda i: canonical_key(rowdicts[i], kf))[0], False, "concordant_canonical"
        if policy == "P4":
            if len({norm_term(revs[i]) for i in at_best}) > 1:
                return None, True, "concordant_status_tie_quarantined"
            return sorted(at_best, key=lambda i: canonical_key(rowdicts[i], kf))[0], False, "concordant_canonical"
        if policy == "P5":
            # label-eligibility term FIRST, then semantic rank, then canonical key
            def lelig(i):
                return 0 if labels[i] in (0, 1) else 1
            best_i = sorted(at_best, key=lambda i: (lelig(i), semantic_rank(revs[i]), canonical_key(rowdicts[i], kf)))[0]
            return best_i, False, "concordant_labeleligible_then_semantic"
    else:
        best = min(tiers[i] for i in idxs)
        at_best = [i for i in idxs if tiers[i] == best]
        classes = {labels[i] for i in at_best}
        if len(classes) == 1 and classes <= {0, 1}:
            if policy in ("P0", "P1"):
                best_set = set(at_best)
                return next(i for i in ordered if i in best_set), False, "conflict_resolved_file_order"
            return sorted(at_best, key=lambda i: (semantic_rank(revs[i]), canonical_key(rowdicts[i], kf)))[0], False, "conflict_resolved_deterministic"
        return None, True, "conflict_irreducible"
    return sorted(idxs)[0], False, "fallback"


def run_single_row_policy(vids, tiers, labels, revs, rowdicts, kf, policy, order, groups):
    kept, quar = set(), set()
    for v, idxs in groups.items():
        if len(idxs) == 1:
            kept.add(idxs[0]); continue
        k, q, _r = select_repr_row(idxs, tiers, labels, revs, rowdicts, kf, policy, order)
        if q:
            quar.add(v)
        elif k is not None:
            kept.add(k)
    return kept, quar


# =====================================================================
# P6: group-level evidence adjudication + independent representative row
# =====================================================================
def adjudicate_label(idxs, uni_tiers, sigs):
    """Rule A: canonical label from G* = rows at best tier. Returns
    (canonical_state, binary_label|None, reason, explicit_conflict_present)."""
    best = min(uni_tiers[i] for i in idxs)
    gstar = [i for i in idxs if uni_tiers[i] == best]
    states = [evidence_state(sigs[i])[0] for i in gstar]
    has_path = "PATHOGENIC" in states
    has_benign = "BENIGN" in states
    has_conflict = "EXPLICIT_CONFLICT" in states
    # Rule 3: opposed binary at best tier
    if has_path and has_benign:
        return "IRREDUCIBLE_CONFLICT", None, "rule3_opposed_binary", has_conflict
    # Rule 4: equal-tier explicit-conflict + equal-tier binary
    if has_conflict and (has_path or has_benign):
        return "AMBIGUOUS_AT_BEST_TIER", None, "rule4_conflict_plus_binary", True
    # Rule 2: concordant binary
    if has_path:
        return "PATHOGENIC", 1, "rule2_concordant_pathogenic", has_conflict
    if has_benign:
        return "BENIGN", 0, "rule2_concordant_benign", has_conflict
    # Rule 6: no binary evidence at best tier
    return "NO_BINARY_AT_BEST_TIER", None, "rule6_no_binary", has_conflict


def select_repr_row_p6(idxs, uni_tiers, revs, rowdicts, kf, order):
    """Rule B: representative row, independent of label. Deterministic:
    (tier, semantic_rank, canonical_key). Never file order."""
    best = min(uni_tiers[i] for i in idxs)
    at_best = [i for i in idxs if uni_tiers[i] == best]
    return sorted(at_best, key=lambda i: (semantic_rank(revs[i]), canonical_key(rowdicts[i], kf)))[0]


def run_p6(vids, uni_tiers, sigs, revs, rowdicts, kf, order, groups):
    repr_kept = set()
    quar = set()
    labels_out = {}       # variant_id -> binary label | None
    states_out = {}       # variant_id -> canonical state
    for v, idxs in groups.items():
        if len(idxs) == 1:
            i = idxs[0]
            repr_kept.add(i)
            st, bl, _r, _ec = adjudicate_label(idxs, uni_tiers, sigs)
            labels_out[v] = bl; states_out[v] = st
            continue
        st, bl, reason, _ec = adjudicate_label(idxs, uni_tiers, sigs)
        states_out[v] = st; labels_out[v] = bl
        if st in ("IRREDUCIBLE_CONFLICT",):
            quar.add(v)
            continue
        r = select_repr_row_p6(idxs, uni_tiers, revs, rowdicts, kf, order)
        repr_kept.add(r)
    return repr_kept, quar, labels_out, states_out


def main() -> int:
    import pyarrow.parquet as pq
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s); print(s)

    emit("=" * 78)
    emit("STEP 1b CLEAN_COHORT P5+P6 EVIDENCE-ADJUDICATION AUDIT -- 2026-07-25 (read-only)")
    emit("=" * 78)

    cohort = RAW if RAW.is_file() else (CLEAN if CLEAN.is_file() else None)
    if cohort is None:
        print(f"  FAIL: no cohort at {RAW} or {CLEAN}"); return EXIT_FAIL
    emit(f"\n  cohort: {cohort.name}")
    if cohort == CLEAN:
        emit("  WARNING: cleaned cohort; duplicates may be pre-resolved. Use RAW for the real audit.")

    emit("\n-- source-map drift check --")
    emit(f"  {confirm_source_map()}")

    tbl = pq.read_table(cohort)
    have = list(tbl.column_names)
    if "variant_id" not in have:
        print("  FAIL: no variant_id"); return EXIT_FAIL
    vid = tbl.column("variant_id").to_pylist()
    n = len(vid)
    emit(f"  rows: {n:,}   columns: {len(have)}")

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

    sig_col = None
    for c in ("clinical_sig", "clinical_significance", "clnsig"):
        if c in have:
            sig_col = c; break
    if sig_col is None:
        print("  FAIL: no clinical_sig column"); return EXIT_FAIL
    sigs = tbl.column(sig_col).to_pylist()
    emit(f"  review column: {rev_src}   clinical_sig column: {sig_col}")

    kf = tuple(f for f in CANONICAL_KEY_FIELDS if f in have)
    df_fields = tuple(f for f in DELTA_FIELDS if f in have)
    emit(f"  canonical-key fields: {kf}")

    need = set(kf) | set(df_fields)
    coldata = {c: tbl.column(c).to_pylist() for c in need if c in have}
    rowdicts = [{c: coldata[c][i] for c in coldata} for i in range(n)]

    labels = [norm_label(x) for x in sigs]
    leg_tiers = [legacy_tier(r) for r in rev]
    uni_pairs = [unified_tier(r) for r in rev]
    would_raise = sum(1 for t, _p in uni_pairs if t is None)
    uni_tiers = [SENTINEL_UNMATCHED if t is None else t for t, _p in uni_pairs]
    emit(f"\n  WOULD_RAISE (unified rejects): {would_raise:,}")

    # -- fail-loud: report any UNRECOGNISED clinical_sig vocabulary --
    unrec = Counter()
    for s in sigs:
        st, ok = evidence_state(s)
        if not ok:
            unrec[norm_term(s)] += 1
    emit(f"\n-- clinical_sig evidence-state vocabulary check --")
    if unrec:
        emit(f"  UNRECOGNISED clinical_sig values: {len(unrec)} distinct "
             f"({sum(unrec.values()):,} rows) -- classifier must be extended before adoption:")
        for val, c in unrec.most_common(25):
            emit(f"    {c:>10,}  {val!r}")
    else:
        emit("  all clinical_sig values classified into known evidence states (none unrecognised)")

    order = list(range(n))

    # Build the variant_id -> [row indices] grouping ONCE and reuse it everywhere.
    # (Rebuilding it inside every policy runner was O(policies * N).)
    groups: "dict" = defaultdict(list)
    for i, v in enumerate(vid):
        groups[v].append(i)

    # -- P0-P5 single-row policies --
    emit("\n" + "=" * 78)
    emit("P0-P6 POLICY TABLE")
    emit("-" * 78)
    results = {}
    for pol in ("P0", "P1", "P2", "P3", "P4", "P5"):
        tiers = leg_tiers if pol == "P0" else uni_tiers
        kept, quar = run_single_row_policy(vid, tiers, labels, rev, rowdicts, kf, pol, order, groups)
        results[pol] = (kept, quar)
    p6_kept, p6_quar, p6_labels, p6_states = run_p6(vid, uni_tiers, sigs, rev, rowdicts, kf, order, groups)

    base_kept, base_quar = results["P0"]

    # legacy per-variant label = label of the kept row
    def variant_labels_from_kept(kept):
        out = {}
        for i in kept:
            out[vid[i]] = labels[i]
        return out
    base_vlabels = variant_labels_from_kept(base_kept)

    def count_labels(vlabels):
        pos = sum(1 for v in vlabels.values() if v == 1)
        neg = sum(1 for v in vlabels.values() if v == 0)
        trn = pos + neg
        return trn, pos, neg

    hdr = f"  {'measure':<34}" + "".join(f"{p:>9}" for p in ("P0","P1","P2","P3","P4","P5","P6"))
    emit(hdr)

    # order-sensitivity computed per policy (P0/P1 file-order = sensitive where ties>1)
    def order_sensitive_count(policy):
        if policy not in ("P0", "P1"):
            return 0
        tiers = leg_tiers if policy == "P0" else uni_tiers
        cnt = 0
        for v, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            # Both concordant "keep one" ties and conflict-resolved (discordant-but-
            # agreeing-at-best-tier) ties are resolved by iloc[0] under file order, so
            # any multi-row group with more than one row at the best tier is an
            # order-sensitive selection under P0/P1. (The earlier if/else here had two
            # identical branches -- collapsed; there is no concordant/discordant split
            # to make for this count.)
            best = min(tiers[i] for i in idxs)
            at = [i for i in idxs if tiers[i] == best]
            if len(at) > 1:
                cnt += 1
        return cnt

    rows_out = {}
    for pol in ("P0","P1","P2","P3","P4","P5"):
        kept, quar = results[pol]
        vlabels = variant_labels_from_kept(kept)
        trn, pos, neg = count_labels(vlabels)
        repr_changes = len({vid[i] for i in (kept ^ base_kept)})
        label_changes = sum(1 for v in vlabels if base_vlabels.get(v) != vlabels.get(v))
        rows_out[pol] = dict(order=order_sensitive_count(pol), repr=repr_changes,
                             label=label_changes, quar=len(quar ^ base_quar),
                             trn=trn, pos=pos, neg=neg)
    # P6. IMPORTANT: P6 has TWO distinct label notions, and they must not be conflated
    # with the P0-P5 kept-row label:
    #   (a) repr-row label  -- the label of P6's chosen representative row. This is the
    #       like-for-like counterpart to the P0-P5 kept-row label, and belongs on the
    #       same "kept-row label changes" line.
    #   (b) group-adjudicated label -- the canonical label from adjudicate_label(). This
    #       is the stricter, group-level notion (it withholds a label under Rule 4/6) and
    #       is reported on its OWN line so it is never compared against a kept-row label.
    p6_reprrow_label = {}
    for i in p6_kept:
        p6_reprrow_label[vid[i]] = labels[i]
    trn6, pos6, neg6 = count_labels(p6_labels)  # group-adjudicated trainability
    p6_repr_changes = len({vid[i] for i in (p6_kept ^ base_kept)})
    # like-for-like: P6 repr-row label vs legacy kept-row label
    p6_reprrow_label_changes = sum(1 for v in p6_reprrow_label
                                   if base_vlabels.get(v) != p6_reprrow_label.get(v))
    # group-level: P6 adjudicated label vs legacy kept-row label (DIFFERENT BASIS -- own line)
    p6_group_label_changes = sum(1 for v in p6_labels if base_vlabels.get(v) != p6_labels.get(v))
    p6_conflicts_preserved = sum(1 for st in p6_states.values()
                                 if st in ("IRREDUCIBLE_CONFLICT", "AMBIGUOUS_AT_BEST_TIER"))
    rows_out["P6"] = dict(order=0, repr=p6_repr_changes, label=p6_reprrow_label_changes,
                          quar=len(p6_quar ^ base_quar), trn=trn6, pos=pos6, neg=neg6)

    def row(label, key):
        return f"  {label:<38}" + "".join(f"{rows_out[p][key]:>9,}" for p in ("P0","P1","P2","P3","P4","P5","P6"))
    emit(row("order-sensitive selections", "order"))
    emit(row("representative-row changes vs P0", "repr"))
    emit(row("kept-row label changes vs P0", "label"))
    emit("  (the line above is like-for-like: each policy's KEPT/REPR row label vs legacy)")
    emit(row("quarantine changes vs P0", "quar"))
    emit(row("trainable-row count", "trn"))
    emit(row("positive count", "pos"))
    emit(row("negative count", "neg"))
    emit(f"  {'explicit conflicts preserved':<38}" +
         "".join(f"{'-':>9}" for _ in range(6)) + f"{p6_conflicts_preserved:>9,}")
    emit("")
    emit(f"  P6 GROUP-ADJUDICATED label changes vs legacy (STRICTER, different basis): "
         f"{p6_group_label_changes:,}")
    emit(f"    -- P6 withholds a binary label under Rule 4 (equal-tier conflict + binary) and")
    emit(f"       Rule 6 (no binary at best tier). This count is NOT comparable to the kept-row")
    emit(f"       label line above; it reflects group-level adjudication, not row choice.")

    # -- classify the label-changing groups (P2 vs legacy) into 7 categories --
    emit("\n" + "=" * 78)
    emit("LABEL-CHANGING GROUP CLASSIFICATION (7 mechanistic categories)")
    emit("-" * 78)
    emit("  Comparing legacy P0 kept-row label vs P6 canonical label, per variant.")
    cat = Counter()
    examples = defaultdict(list)
    for v, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        base_l = base_vlabels.get(v)
        p6_l = p6_labels.get(v)
        if base_l == p6_l:
            continue
        best = min(uni_tiers[i] for i in idxs)
        gstar = [i for i in idxs if uni_tiers[i] == best]
        gstar_states = {evidence_state(sigs[i])[0] for i in gstar}
        has_bin = ("PATHOGENIC" in gstar_states) or ("BENIGN" in gstar_states)
        if "PATHOGENIC" in gstar_states and "BENIGN" in gstar_states:
            c = "4_pathogenic_vs_benign"
        elif has_bin and "EXPLICIT_CONFLICT" in gstar_states:
            c = "1_binary_vs_explicit_conflict"
        elif has_bin and "UNCERTAIN" in gstar_states:
            c = "2_binary_vs_uncertain"
        elif has_bin and "NO_CLASSIFICATION" in gstar_states:
            c = "3_binary_vs_no_classification"
        elif "UNRECOGNISED" in gstar_states:
            c = "6_missing_or_malformed"
        else:
            c = "7_provenance_only_or_other"
        cat[c] += 1
        if len(examples[c]) < 3:
            examples[c].append(v)
    if cat:
        for c in sorted(cat):
            emit(f"  {c:<34} {cat[c]:>6}   e.g. {examples[c]}")
    else:
        emit("  no label-changing groups between legacy and P6 (label independent of row choice)")

    # -- P6 ORDER-INVARIANCE sabotage: original / reverse / random perms --
    emit("\n" + "=" * 78)
    emit("P6 ORDER-INVARIANCE (sabotage: original / reverse / 3 random permutations)")
    emit("-" * 78)
    # P6's selection is deterministic on (semantic_rank, canonical_key) and ignores
    # input order by construction. To TEST that claim rather than assume it, we permute
    # the row indices WITHIN each duplicate group (that is the only place order could
    # leak in) and rebuild a permuted groups map, then re-run P6 and compare. Permuting
    # only within groups is both the correct test surface and O(dup rows), not O(N).
    # P6's selection ignores input order by construction (it sorts each group by
    # (semantic_rank, canonical_key)). We TEST that claim on the only surface where
    # order could leak in -- the multi-row groups -- by permuting each such group's row
    # order and re-adjudicating JUST those groups. Singletons are order-trivial (one
    # row) and are excluded, which keeps the test O(dup rows) rather than O(N) per perm.
    rng = random.Random(20260725)
    multi = {v: idxs for v, idxs in groups.items() if len(idxs) > 1}

    def adjudicate_multi(perm_map):
        """Re-run P6's per-group decisions over the multi-row groups only, in the given
        per-group row order. Returns (repr_set, quar_set, labels, states) for those."""
        rset, qset, labs, sts = set(), set(), {}, {}
        for v, idxs in perm_map.items():
            st, bl, _reason, _ec = adjudicate_label(idxs, uni_tiers, sigs)
            sts[v] = st; labs[v] = bl
            if st == "IRREDUCIBLE_CONFLICT":
                qset.add(v); continue
            rset.add(select_repr_row_p6(idxs, uni_tiers, rev, rowdicts, kf, order))
        return rset, qset, labs, sts

    base = adjudicate_multi(multi)
    perms = {"within-group reverse": {v: list(reversed(ix)) for v, ix in multi.items()}}
    for k in range(3):
        perms[f"within-group random{k+1}"] = {v: rng.sample(ix, len(ix)) for v, ix in multi.items()}

    all_ok = True
    for name, pm in perms.items():
        got = adjudicate_multi(pm)
        repr_same = (got[0] == base[0])
        q_same = (got[1] == base[1])
        lab_same = (got[2] == base[2])
        st_same = (got[3] == base[3])
        ok = repr_same and lab_same and st_same and q_same
        all_ok = all_ok and ok
        emit(f"  {name:<22}: repr={repr_same} labels={lab_same} states={st_same} quarantine={q_same}  -> {'OK' if ok else 'FAIL'}")
    emit(f"\n  P6 order-invariant across all permutations: {all_ok}")

    emit("\n" + "=" * 78)
    emit("ACCEPTANCE-CRITERIA READOUT (for the v2 decision)")
    emit("-" * 78)
    emit(f"  unmatched review statuses (WOULD_RAISE)      : {would_raise}")
    emit(f"  unrecognised clinical_sig values             : {len(unrec)}")
    emit(f"  P0 order-sensitive selections                : {rows_out['P0']['order']}")
    emit(f"  P6 order-invariant                           : {all_ok}")
    emit(f"  P6 representative-row changes vs legacy       : {rows_out['P6']['repr']}")
    emit(f"  P6 canonical-label changes vs legacy          : {rows_out['P6']['label']}")
    emit(f"  P6 quarantine changes vs legacy               : {rows_out['P6']['quar']}")
    emit(f"  P6 explicit conflicts preserved (not discarded): {p6_conflicts_preserved}")
    emit("\n  P6 separates label adjudication (from the best-tier evidence group) from")
    emit("  representative-row selection (deterministic metadata only). Neither file order")
    emit("  nor label-eligibility can manufacture, remove, or change a canonical label.")
    emit("  If order-invariant=True and every label change is explained by a declared rule")
    emit("  above, P6 is the basis for the separately-certified cohort v2. Do not adopt any")
    emit("  policy or rewire clean_cohort from this probe alone -- this is the evidence.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWROTE {OUT}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
