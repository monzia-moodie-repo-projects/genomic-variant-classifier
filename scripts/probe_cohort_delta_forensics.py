#!/usr/bin/env python3
"""What did clean_cohort actually remove, what review statuses actually exist, and did the sequence branch actually succeed?

WHY THIS SCRIPT EXISTS
======================
The cohort schema census of 2026-07-24 produced three numbers that do not agree
with the project's own record, and one that was never measured at all.

  1. THE 1,103-ROW GAP. `clinvar_grch38.parquet` holds 4,420,180 rows and
     `clinvar_grch38_clean.parquet` holds 4,399,089, a removal of 21,091. The
     project record attributes that removal to structural and copy-number-variant
     (CNV) rows carrying null or empty alleles, and 21,091 is the figure recorded
     for them. But the census measured only **19,988** rows with a null `ref` and
     19,988 with a null `alt`. **1,103 rows were removed for a reason nothing in
     the record accounts for.** Until those rows are identified, the statement
     "the clean cohort is the raw cohort minus its null-allele rows" is false as
     written, and every downstream claim resting on it inherits the error.

  2. THE REVIEW-STATUS VALUE INVENTORY. Both `ReviewStatus` and the nested
     `metadata.review_status` carry exactly **10 distinct values** with zero
     nulls. The tier map in `review_status.py` has exactly 10 keys. Those two
     tens have never been shown to be the SAME ten. `OBSERVED_2026_07_24` in that
     module freezes only eight values and documents itself as "the top eight of
     the blank-ReviewStatus subset". If any cohort value is not a map key, the
     raise that Step 1b introduces fires on the first production run. That must
     be known before the raise lands, not discovered by it.

  3. THE SEQUENCE-BRANCH GATE. `clinvar_grch38_clean_seq.parquet` adds `ok`
     (boolean) and `reason` (string). Nothing has measured how many rows carry
     `ok = false` or why. A 399 MiB addition whose success rate is unknown is a
     stub with a large disk footprint until that rate is measured.

  4. PROVENANCE TIMESTAMPS. Every artifact stamps `created_by` as
     'parquet-cpp-arrow version 23.0.1', which is identical across all three and
     therefore distinguishes nothing. The filesystem modification time is the
     only ordering evidence available, and it bears directly on the finding that
     the `pathogenicity` column's contents are inconsistent with the mapping code
     in force since 2026-07-10.

A CORRECTION THIS SCRIPT ALSO MAKES
-----------------------------------
`probe_cohort_schema_census.py` reported distinct counts using
`pyarrow.compute.unique`, which COUNTS NULL AS A DISTINCT VALUE. Any distinct
figure it printed beside a non-zero null count is therefore inflated by exactly
one: raw `ref` is 38,496 real distinct values, not 38,497, and raw `alt` is
20,667, not 20,668. This script reports null count and non-null distinct count as
two separate figures so the reader is never required to make that adjustment
mentally.

EXIT CODES
----------
    0  Every measurement completed and every finding is benign: the removal is
       fully accounted for, every cohort review-status value is a tier-map key,
       and the sequence branch has no failures.
    1  A measurement completed and found something that blocks: an unaccounted
       removal, a cohort review-status value absent from the tier map, or
       sequence-branch failures. The report names each.
    2  Contract or environment failure. Nothing measured, no verdict issued.

READ-ONLY. `--self-test` round-trips synthetic tables through the system
temporary directory and touches nothing under the repository.

Parquet is read with `pyarrow.parquet`, never `pandas.read_parquet`, per commit
644a184 (2026-07-23). Only the columns each measurement needs are materialised.

Acronyms on first use. AST = abstract syntax tree. CNV = copy-number variant.
ClinVar = the National Center for Biotechnology Information's Clinical Variation
archive. VCF = variant call format.

USAGE (PowerShell 5.1)
----------------------
    python "C:\\Users\\monzi\\Downloads\\probe_cohort_delta_forensics.py" --self-test

    python "C:\\Users\\monzi\\Downloads\\probe_cohort_delta_forensics.py" `
        --repo "C:\\Projects\\genomic-variant-classifier"

Companion to
probe_label_column_terms.py and probe_cohort_schema_census.py.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

EXIT_CLEAN = 0
EXIT_FINDING = 1
EXIT_ENVIRONMENT = 2

RAW_REL = "data/processed/clinvar_grch38.parquet"
CLEAN_REL = "data/processed/clinvar_grch38_clean.parquet"
SEQ_REL = "data/processed/clinvar_grch38_clean_seq.parquet"

#: Exactly the tokens real_data_prep._assert_clean_cohort rejects, line 476.
BAD_TOKENS = ("", "nan", "none", "na", ".", "null", "-")

#: Where a tier map may be read from, most authoritative first. The first that
#: parses wins, and the report says which was used -- never a silent fallback.
TIER_MAP_SOURCES = (
    ("src/genomic_variant_classifier/data/review_status.py", "REVIEW_STATUS_TIER"),
    ("scripts/clean_cohort.py", "REVIEW_STATUS_TIER"),
    ("src/genomic_variant_classifier/data/real_data_prep.py", "REVIEW_STATUS_TIER"),
)

_WHITESPACE = re.compile(r"\s+")


class ContractError(RuntimeError):
    """The repository or artifacts do not look the way this probe requires."""


def normalise(value: object) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value).lower().replace("_", " ")).strip()


# ---------------------------------------------------------------------------
# Tier map, read by AST
# ---------------------------------------------------------------------------
def read_missing_tokens(repo: Path) -> tuple[frozenset[str], str]:
    """MISSING_TOKENS from review_status.py, or the documented default.

    WHY THIS EXISTS -- a defect in the first version of this probe, 2026-07-24.
    That version compared every cohort review-status value against the tier map
    ALONE and printed, beside the count, the claim that Step 1b's raise would
    "abort on these rows". For '' (424,516 rows) and '-' (245,148 rows) that
    claim is FALSE: `tier_of` tests MISSING_TOKENS BEFORE consulting the map and
    returns TIER_MISSING without raising. The count was literally true -- those
    values are not map keys -- and the sentence printed next to it was wrong for
    669,664 of the 669,918 rows it named. A probe whose prose over-claims what
    its own number means is the defect class this project exists to eliminate,
    so the probe now classifies into three buckets and only the third is a
    finding.
    """
    rel = "src/genomic_variant_classifier/data/review_status.py"
    path = repo / rel
    if path.is_file():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                targets = list(node.targets) if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else [])
                for tgt in targets:
                    if not (isinstance(tgt, ast.Name) and tgt.id == "MISSING_TOKENS"):
                        continue
                    val = node.value
                    if isinstance(val, ast.Call) and val.args:
                        val = val.args[0]
                    if isinstance(val, (ast.Set, ast.List, ast.Tuple)):
                        items = [e.value for e in val.elts
                                 if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                        if items:
                            return frozenset(items), f"{rel}:{node.lineno}"
    return frozenset(BAD_TOKENS), "built-in default (review_status.py unavailable)"


def read_tier_map(repo: Path) -> tuple[dict[str, int], str]:
    """The tier map and the file it came from. Never a silent fallback."""
    tried: list[str] = []
    for rel, name in TIER_MAP_SOURCES:
        path = repo / rel
        if not path.is_file():
            tried.append(f"{rel} (absent)")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            tried.append(f"{rel} (unparseable: {type(exc).__name__})")
            continue
        for node in ast.walk(tree):
            targets = list(node.targets) if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else [])
            for tgt in targets:
                if not (isinstance(tgt, ast.Name) and tgt.id == name):
                    continue
                val = node.value
                if not isinstance(val, ast.Dict):
                    continue
                out: dict[str, int] = {}
                ok = True
                for k, v in zip(val.keys, val.values):
                    if not (isinstance(k, ast.Constant) and isinstance(k.value, str)
                            and isinstance(v, ast.Constant) and isinstance(v.value, int)):
                        ok = False
                        break
                    out[k.value] = v.value
                if ok and out:
                    return out, f"{rel}:{node.lineno}"
        tried.append(f"{rel} (no literal {name})")
    raise ContractError(
        "no tier map could be read. Tried: " + "; ".join(tried) +
        ". Without it, coverage of the cohort's review-status values cannot be assessed.")


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def require_columns(path: Path, needed: tuple[str, ...]) -> None:
    """Raise ContractError if any needed column is absent from the artifact.

    Checked against the schema, which costs no row read. Without this, a missing
    column surfaces as an unhandled pyarrow.lib.ArrowInvalid traceback and the
    process exits 1 -- the code reserved for a measured finding. A contract
    problem reported as a finding is a worse failure than no check at all,
    because the operator acts on it.
    """
    import pyarrow.parquet as pq

    present = set(pq.read_schema(path).names)
    missing = [c for c in needed if c not in present]
    if missing:
        raise ContractError(
            f"{path}: required column(s) absent -- {', '.join(missing)}. "
            f"Present: {sorted(present)}. Nothing was measured from this artifact."
        )


def nulls_and_distinct(arr) -> tuple[int, int]:
    """(null_count, NON-NULL distinct count).

    pyarrow.compute.unique counts null as a distinct value. Reporting that number
    beside a null count invites a reader to double-count absence, which is the
    defect this function exists to remove.
    """
    import pyarrow.compute as pc

    nulls = arr.null_count
    distinct = len(pc.unique(arr))
    return nulls, distinct - (1 if nulls else 0)


def classify_removed(repo: Path, out: list[str]) -> tuple[int, int]:
    """Identify every row present in the raw cohort and absent from the clean one.

    Returns (removed_total, unaccounted). The classification is ORDERED and each
    row is counted once, under the first reason that applies, so the categories
    sum to the total exactly and cannot double-count.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    raw_p, clean_p = repo / RAW_REL, repo / CLEAN_REL
    for p in (raw_p, clean_p):
        if not p.is_file():
            raise ContractError(f"{p} is absent; the removal cannot be characterised.")

    # Validate against the schema BEFORE reading. A missing column must surface as
    # a contract failure, not as an unhandled Arrow traceback that exits with the
    # code reserved for findings -- which is precisely the confusion these exit
    # codes exist to prevent.
    require_columns(raw_p, ("variant_id", "ref", "alt"))
    require_columns(clean_p, ("variant_id",))

    raw = pq.read_table(raw_p, columns=["variant_id", "ref", "alt"])
    clean = pq.read_table(clean_p, columns=["variant_id"])

    def as_str(col):
        c = col.combine_chunks()
        return c.cast(pa.string()) if c.type != pa.string() else c

    raw_vid = as_str(raw.column("variant_id"))
    clean_vid = as_str(clean.column("variant_id"))
    raw_ref = as_str(raw.column("ref"))
    raw_alt = as_str(raw.column("alt"))

    out.append("=" * 78)
    out.append("MEASUREMENT 1 -- WHAT clean_cohort ACTUALLY REMOVED")
    out.append("=" * 78)
    out.append(f"  raw rows        : {raw.num_rows:,}")
    out.append(f"  clean rows      : {clean.num_rows:,}")
    removed_total = raw.num_rows - clean.num_rows
    out.append(f"  removal (rows)  : {removed_total:,}")
    out.append("")

    for label, arr in (("ref", raw_ref), ("alt", raw_alt)):
        n, d = nulls_and_distinct(arr)
        out.append(f"  raw {label:<3}: nulls {n:>9,}   non-null distinct {d:>9,}")
    out.append("")

    kept_mask = pc.is_in(raw_vid, value_set=pc.unique(clean_vid))
    removed_mask = pc.invert(pc.fill_null(kept_mask, False))
    n_removed_rows = pc.sum(pc.cast(removed_mask, pa.int64())).as_py() or 0
    out.append(f"  rows in raw whose variant_id is ABSENT from clean : {n_removed_rows:,}")
    if n_removed_rows != removed_total:
        out.append("")
        out.append(f"  NOTE: that is not equal to the row-count delta ({removed_total:,}).")
        out.append("  The difference means clean contains DUPLICATE variant_id values, or")
        out.append("  raw does, so a row-count delta and an identity delta are not the same")
        out.append("  quantity. Both are reported; neither is assumed to stand for the other.")
    out.append("")

    idx = [i for i, v in enumerate(removed_mask.to_pylist()) if v]
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    ref_l, alt_l, vid_l = raw_ref.to_pylist(), raw_alt.to_pylist(), raw_vid.to_pylist()
    for i in idx:
        r, a = ref_l[i], alt_l[i]
        if r is None or a is None:
            reason = "null ref and/or alt"
        elif str(r).strip().lower() in BAD_TOKENS or str(a).strip().lower() in BAD_TOKENS:
            reason = "ref/alt is a rejected token"
        else:
            reason = "UNACCOUNTED -- ref and alt are both present and valid"
        reasons[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 5:
            examples[reason].append(f"variant_id={vid_l[i]!r} ref={r!r} alt={a!r}")

    out.append("  classification of every removed row, each counted once under the")
    out.append("  FIRST reason that applies (so the categories sum to the total exactly):")
    for reason, n in reasons.most_common():
        out.append(f"    {n:>9,}  {reason}")
        for ex in examples[reason]:
            out.append(f"               e.g. {ex}")
    out.append(f"    {sum(reasons.values()):>9,}  TOTAL")
    unaccounted = reasons.get("UNACCOUNTED -- ref and alt are both present and valid", 0)
    out.append("")
    if unaccounted:
        out.append(f"  *** FINDING: {unaccounted:,} removed row(s) have a valid ref AND a valid")
        out.append("  alt. They are not null-allele rows. The statement that the clean cohort")
        out.append("  is the raw cohort minus its null-allele rows is therefore incomplete,")
        out.append("  and the reason these rows were dropped is not recorded anywhere.")
    else:
        out.append("  Every removed row is accounted for by a null or rejected allele.")
    return removed_total, unaccounted



def _sum_mask(mask) -> int:
    """Row count of a boolean mask, nulls counted as False."""
    import pyarrow as pa
    import pyarrow.compute as pc

    return int(pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0)


def source_comparison(repo: Path, tier_map: dict[str, int],
                      missing_tokens: frozenset[str], out: list[str]) -> int:
    """Row-level comparison of the two candidate review-status sources.

    WHY THIS EXISTS -- a defect in the first version of this probe, 2026-07-24.
    That version summed per-column unmatched counts, 121 from ReviewStatus and 133
    from metadata.review_status, and printed the total as "254 row(s)". Those are
    FIELD OCCURRENCES ACROSS TWO COLUMNS. The distinct-row union is bounded below
    by 133 and above by 254 and was never measured, so the sentence asserted
    something the arithmetic did not support. A sum of two column counts is a row
    count only when the two conditions are disjoint, which nothing had shown.

    This function measures the union directly, and reconciles it three ways so a
    miscount cannot pass:

        old_count = overlap + old_only
        new_count = overlap + new_only
        union     = overlap + old_only + new_only

    It also builds the full transition cross-tabulation between the two sources.
    That answers the question that actually justifies the Phase 1 source change and
    that no measurement so far distinguishes: does the nested source FILL blanks the
    variant-call-format join left empty, or does it CHANGE statuses the join already
    populated? Filling is a repair. Changing is a disagreement, and a disagreement
    between two sources of the same fact needs adjudicating before either is trusted.

    Both columns are read once. The cross-tabulation is computed by encoding each
    column to integer indices and counting the combined code, so no 4.4-million-
    element Python list is ever materialised.

    Returns the number of rows in the unmatched union, which is the figure that
    belongs in a verdict.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = repo / CLEAN_REL
    if not path.is_file():
        raise ContractError(f"{path} is absent; the sources cannot be compared.")
    require_columns(path, ("ReviewStatus", "metadata"))

    table = pq.read_table(path, columns=["ReviewStatus", "metadata"])
    rows = table.num_rows
    old = table.column("ReviewStatus").combine_chunks()
    meta = table.column("metadata").combine_chunks()
    try:
        new = meta.field("review_status")
    except Exception as exc:                                     # noqa: BLE001
        raise ContractError(
            f"{path}: metadata has no review_status child ({type(exc).__name__})."
        ) from exc

    normalised_keys = {normalise(k) for k in tier_map}
    normalised_missing = {normalise(t) for t in missing_tokens} | {""}

    values = sorted({v for v in pc.unique(old).to_pylist() if v is not None} |
                    {v for v in pc.unique(new).to_pylist() if v is not None})
    unmatched = [v for v in values
                 if normalise(v) not in normalised_keys
                 and normalise(v) not in normalised_missing]

    out.append("")
    out.append("=" * 78)
    out.append("MEASUREMENT 2b -- UNMATCHED VALUES, MEASURED AS ROWS NOT OCCURRENCES")
    out.append("=" * 78)
    out.append(f"  cohort : {path.name}   ({rows:,} rows)")
    out.append("")
    if not unmatched:
        out.append("  No value in either column is unmatched. Nothing to reconcile.")
        union_total = 0
    else:
        union_total = 0
        for value in unmatched:
            m_old = pc.fill_null(pc.equal(old, value), False)
            m_new = pc.fill_null(pc.equal(new, value), False)
            old_c = _sum_mask(m_old)
            new_c = _sum_mask(m_new)
            overlap = _sum_mask(pc.and_(m_old, m_new))
            old_only = _sum_mask(pc.and_not(m_old, m_new))
            new_only = _sum_mask(pc.and_not(m_new, m_old))
            union = _sum_mask(pc.or_(m_old, m_new))
            union_total += union
            out.append(f"  value: {value!r}")
            out.append(f"    ReviewStatus (legacy source)        occurrences : {old_c:>9,}")
            out.append(f"    metadata.review_status (repaired)   occurrences : {new_c:>9,}")
            out.append(f"    naive sum of the two -- NOT a row count         : {old_c + new_c:>9,}")
            out.append(f"    overlap  (both columns, same row)               : {overlap:>9,}")
            out.append(f"    old only (legacy only)                          : {old_only:>9,}")
            out.append(f"    new only (repaired only)                        : {new_only:>9,}")
            out.append(f"    UNION    (distinct rows affected)               : {union:>9,}")
            ok1 = old_c == overlap + old_only
            ok2 = new_c == overlap + new_only
            ok3 = union == overlap + old_only + new_only
            for label, ok in (("old = overlap + old_only", ok1),
                              ("new = overlap + new_only", ok2),
                              ("union = overlap + old_only + new_only", ok3)):
                out.append(f"    reconcile  {label:<40} {'OK' if ok else 'FAILED'}")
            if not (ok1 and ok2 and ok3):
                raise ContractError(
                    f"union reconciliation failed for {value!r}; the counts are not "
                    f"self-consistent and no verdict is issued."
                )
            out.append("")
        out.append(f"  The figure that belongs in a verdict is the UNION: {union_total:,} row(s).")
        out.append("  The repaired production source alone would encounter the 'new' count.")

    # ---- transition cross-tabulation -------------------------------------
    out.append("")
    out.append("=" * 78)
    out.append("MEASUREMENT 2c -- SOURCE TRANSITION TABLE")
    out.append("  ReviewStatus (legacy) -> metadata.review_status (repaired), by row.")
    out.append("  Distinguishes a repair that FILLS a blank from one that CHANGES a")
    out.append("  status the legacy source already populated. Only the first is")
    out.append("  unambiguously an improvement.")
    out.append("=" * 78)

    vs = pa.array(values, type=pa.string())
    k = len(values)
    idx_old = pc.fill_null(pc.cast(pc.index_in(old, value_set=vs), pa.int64()), k)
    idx_new = pc.fill_null(pc.cast(pc.index_in(new, value_set=vs), pa.int64()), k)
    code = pc.add(pc.multiply(idx_old, k + 1), idx_new)
    counts: dict[tuple[int, int], int] = {}
    for entry in pc.value_counts(code):
        c = entry["values"].as_py()
        counts[(c // (k + 1), c % (k + 1))] = entry["counts"].as_py()
    total = sum(counts.values())
    if total != rows:
        raise ContractError(
            f"transition counts sum to {total:,} but the table has {rows:,} rows."
        )

    def label(i: int) -> str:
        return "<null>" if i == k else repr(values[i])

    def is_missing(i: int) -> bool:
        return i == k or normalise(values[i]) in normalised_missing

    out.append("")
    out.append(f"  {len(counts)} distinct transition(s), sorted by row count:")
    for (a, b), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        arrow = "==" if a == b else "->"
        out.append(f"    {n:>12,}  {label(a):<56} {arrow} {label(b)}")

    summary = {"unchanged": 0, "FILLED (blank -> populated)": 0,
               "CHANGED (populated -> different populated)": 0,
               "EMPTIED (populated -> blank)": 0, "blank in both": 0}
    for (a, b), n in counts.items():
        ma, mb = is_missing(a), is_missing(b)
        if ma and mb:
            summary["blank in both"] += n
        elif ma and not mb:
            summary["FILLED (blank -> populated)"] += n
        elif mb and not ma:
            summary["EMPTIED (populated -> blank)"] += n
        elif a == b:
            summary["unchanged"] += n
        else:
            summary["CHANGED (populated -> different populated)"] += n
    out.append("")
    out.append("  summary:")
    for label_, n in summary.items():
        out.append(f"    {n:>12,}  ({100.0 * n / rows:6.3f}%)  {label_}")
    out.append(f"    {sum(summary.values()):>12,}  TOTAL")
    if sum(summary.values()) != rows:
        raise ContractError("transition summary does not sum to the row count.")
    out.append("")
    if summary["CHANGED (populated -> different populated)"]:
        out.append("  *** FINDING: the repaired source does not only fill blanks. It assigns a")
        out.append("  DIFFERENT status to rows the legacy source already populated. Those rows")
        out.append("  are a disagreement between two sources of the same fact, and the")
        out.append("  specification's claim of zero disagreement where both are populated must")
        out.append("  be reconciled against this count before the source change lands.")
    else:
        out.append("  The repaired source only fills blanks; it changes no populated status.")
        out.append("  That is the strongest form the specification's zero-disagreement claim")
        out.append("  can take, and it is now measured rather than asserted.")
    if summary["EMPTIED (populated -> blank)"]:
        out.append("")
        out.append("  *** FINDING: some rows LOSE a populated status under the repaired source.")
        out.append("  The repair must not reduce coverage; these rows need explanation.")
    return union_total


def review_status_inventory(repo: Path, tier_map: dict[str, int], origin: str,
                            missing_tokens: frozenset[str], missing_origin: str,
                            out: list[str]) -> int:
    """Every review-status value, classified into three buckets.

    Returns ONLY the count that would actually raise: values that are neither a
    tier-map key nor a missing token. A value handled by MISSING_TOKENS resolves
    to TIER_MISSING silently and by design, and reporting it as a would-raise
    row is an over-report."""
    import pyarrow.parquet as pq

    out.append("")
    out.append("=" * 78)
    out.append("MEASUREMENT 2 -- REVIEW-STATUS VALUES vs THE TIER MAP")
    out.append("=" * 78)
    out.append(f"  tier map read from      : {origin}   ({len(tier_map)} key(s))")
    out.append(f"  MISSING_TOKENS read from: {missing_origin}   "
               f"({len(missing_tokens)} token(s))")
    out.append("")
    out.append("  Each value falls in exactly one bucket:")
    out.append("    MAP KEY        resolves through REVIEW_STATUS_TIER")
    out.append("    MISSING TOKEN  resolves to TIER_MISSING before the map is consulted")
    out.append("    WOULD RAISE    neither -- this is the only bucket that is a finding")
    out.append("")

    path = repo / CLEAN_REL
    if not path.is_file():
        raise ContractError(f"{path} is absent; the inventory cannot be taken.")
    require_columns(path, ("ReviewStatus", "metadata"))
    normalised_keys = {normalise(k) for k in tier_map}
    normalised_missing = {normalise(t) for t in missing_tokens} | {""}
    uncovered_total = 0

    for column in ("ReviewStatus", "metadata"):
        table = pq.read_table(path, columns=[column])
        arr = table.column(column).combine_chunks()
        label = column
        if column == "metadata":
            try:
                arr = arr.field("review_status")
                label = "metadata.review_status"
            except Exception:                                    # noqa: BLE001
                out.append(f"  {column}: no review_status child; skipped.")
                continue
        counts = Counter(arr.to_pylist())
        nulls = counts.pop(None, 0)
        out.append(f"  -- {label} --")
        out.append(f"     nulls {nulls:,}   non-null distinct {len(counts):,}")
        buckets: Counter[str] = Counter()
        for value, n in counts.most_common():
            key = normalise(value)
            if key in normalised_missing:
                mark = "MISSING TOKEN"
            elif key in normalised_keys:
                mark = "MAP KEY"
            else:
                mark = "WOULD RAISE"
                uncovered_total += n
            buckets[mark] += n
            out.append(f"     {n:>12,}  {value!r:<58} {mark}")
        out.append(f"     {'':>12}  {'-- bucket totals --':<58}")
        for b in ("MAP KEY", "MISSING TOKEN", "WOULD RAISE"):
            out.append(f"     {buckets.get(b, 0):>12,}  {b}")
        out.append("")

    if uncovered_total:
        out.append(f"  *** FINDING: {uncovered_total:,} FIELD OCCURRENCE(S) across the two")
        out.append("  columns carry a review status that is NEITHER a tier-map key NOR a")
        out.append("  missing token. That total is a sum over two columns and is NOT a row")
        out.append("  count -- the distinct-row union is measured in section 2b below, and")
        out.append("  it is the union, not this sum, that belongs in a verdict.")
        out.append("  Once Step 1b makes an unmatched status RAISE, a production run over")
        out.append("  the source it consumes aborts on that column's rows.")
    else:
        out.append("  Every review-status value resolves, through the map or through")
        out.append("  MISSING_TOKENS. The raise introduced by Step 1b will not fire here.")
    return uncovered_total


def sequence_branch_gate(repo: Path, out: list[str]) -> int:
    """How many sequence windows failed, and why."""
    import pyarrow.parquet as pq

    out.append("")
    out.append("=" * 78)
    out.append("MEASUREMENT 3 -- THE SEQUENCE-BRANCH GATE (ok / reason)")
    out.append("=" * 78)
    path = repo / SEQ_REL
    if not path.is_file():
        out.append(f"  {path} is absent; nothing to measure.")
        return 0
    names = set(pq.read_schema(path).names)
    if "ok" not in names:
        out.append("  no 'ok' column in this artifact; nothing to measure.")
        return 0
    cols = ["ok"] + (["reason"] if "reason" in names else [])
    table = pq.read_table(path, columns=cols)
    ok_list = table.column("ok").combine_chunks().to_pylist()
    counts = Counter(ok_list)
    total = len(ok_list)
    n_false = counts.get(False, 0)
    n_true = counts.get(True, 0)
    n_null = counts.get(None, 0)
    out.append(f"  rows      : {total:,}")
    out.append(f"  ok = true : {n_true:>12,}  ({100.0 * n_true / total:6.3f}%)")
    out.append(f"  ok = false: {n_false:>12,}  ({100.0 * n_false / total:6.3f}%)")
    out.append(f"  ok = null : {n_null:>12,}  ({100.0 * n_null / total:6.3f}%)")
    if "reason" in cols and (n_false or n_null):
        reasons = Counter(r for r, o in zip(
            table.column("reason").combine_chunks().to_pylist(), ok_list) if o is not True)
        out.append("")
        out.append("  reasons given for every row that is not ok = true:")
        for r, n in reasons.most_common(25):
            out.append(f"    {n:>12,}  {r!r}")
    out.append("")
    if n_false or n_null:
        out.append(f"  *** FINDING: {n_false + n_null:,} row(s) did not produce a usable")
        out.append("  sequence window. Any model consuming fasta_seq_ref or fasta_seq_alt")
        out.append("  must state how these rows are handled; silently training on them is a")
        out.append("  defect, and silently dropping them changes the cohort.")
    else:
        out.append("  Every row produced a usable sequence window.")
    return n_false + n_null


def provenance(repo: Path, out: list[str]) -> None:
    out.append("")
    out.append("=" * 78)
    out.append("MEASUREMENT 4 -- ARTIFACT MODIFICATION TIMES")
    out.append("  created_by is identical across all three artifacts and therefore orders")
    out.append("  nothing. Modification time is the only ordering evidence available, and")
    out.append("  it bears on whether pathogenicity predates the 2026-07-10 mapping fix.")
    out.append("=" * 78)
    rows: list[tuple[str, str, float]] = []
    for rel in (RAW_REL, CLEAN_REL, SEQ_REL):
        p = repo / rel
        if not p.is_file():
            out.append(f"  ABSENT  {rel}")
            continue
        st = p.stat()
        mtime = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        rows.append((rel, mtime, st.st_mtime))
    for rel, mtime, _ in sorted(rows, key=lambda r: r[2]):
        out.append(f"  {mtime}   {rel}")
    out.append("")
    out.append("  The 2026-07-10 mapping fix (database_connectors.py, the guard returning")
    out.append("  'uncertain' for any status beginning 'conflicting') is the reference date.")
    out.append("  An artifact written BEFORE it cannot carry the corrected mapping.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _check(ok: bool, label: str, got: object, want: object, failures: list[str]) -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}: got {got!r}, want {want!r}")
    if not ok:
        failures.append(label)


def self_test() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("=" * 78)
    print("SELF-TEST -- cohort delta forensics")
    print("=" * 78)
    failures: list[str] = []

    print("\n-- nulls and NON-NULL distinct are reported separately --")
    a = pa.array(["x", "y", None, "x"])
    _check(nulls_and_distinct(a) == (1, 2), "null inflation removed",
           nulls_and_distinct(a), (1, 2), failures)
    b = pa.array(["x", "y"])
    _check(nulls_and_distinct(b) == (0, 2), "no nulls, no adjustment",
           nulls_and_distinct(b), (0, 2), failures)

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "data/processed").mkdir(parents=True)
        (repo / "scripts").mkdir()

        # raw: 6 rows -- 1 null allele, 1 rejected token, 1 valid-but-removed, 3 kept
        raw = pa.table({
            "variant_id": pa.array(["v1", "v2", "v3", "v4", "v5", "v6"]),
            "ref": pa.array([None, "-", "A", "C", "G", "T"]),
            "alt": pa.array([None, "A", "T", "G", "A", "C"]),
        })
        clean = pa.table({
            "variant_id": pa.array(["v4", "v5", "v6"]),
            "ReviewStatus": pa.array(["criteria provided, single submitter",
                                      "reviewed by expert panel", "a future clinvar wording"]),
            "metadata": pa.array(
                [{"review_status": "practice guideline"},
                 {"review_status": "criteria provided, single submitter"},
                 {"review_status": "practice guideline"}],
                type=pa.struct([("review_status", pa.string())])),
        })
        pq.write_table(raw, repo / RAW_REL)
        pq.write_table(clean, repo / CLEAN_REL)
        (repo / "scripts/clean_cohort.py").write_text(
            'REVIEW_STATUS_TIER = {\n'
            '    "practice guideline": 1,\n'
            '    "reviewed by expert panel": 1,\n'
            '    "criteria provided, single submitter": 3,\n'
            '}\n', encoding="utf-8")

        print("\n-- the tier map is read by AST, and its origin is named --")
        tmap, origin = read_tier_map(repo)
        _check(len(tmap) == 3, "map keys", len(tmap), 3, failures)
        _check(origin.startswith("scripts/clean_cohort.py:"), "origin named",
               origin.split(":")[0], "scripts/clean_cohort.py", failures)

        print("\n-- removed rows classify into ordered, non-overlapping reasons --")
        out: list[str] = []
        removed, unaccounted = classify_removed(repo, out)
        joined = "\n".join(out)
        _check(removed == 3, "removal total", removed, 3, failures)
        _check(unaccounted == 1, "exactly one unaccounted row", unaccounted, 1, failures)
        _check("null ref and/or alt" in joined, "null category present",
               "null ref and/or alt" in joined, True, failures)
        _check("rejected token" in joined, "token category present",
               "rejected token" in joined, True, failures)
        _check("v3" in joined, "the unaccounted row is named by identity",
               "v3" in joined, True, failures)

        print("\n-- a cohort value absent from the map is flagged, with its row count --")
        out = []
        mt, mo = read_missing_tokens(repo)
        uncovered = review_status_inventory(repo, tmap, origin, mt, mo, out)
        joined = "\n".join(out)
        _check(uncovered == 1, "one uncovered row", uncovered, 1, failures)
        _check("WOULD RAISE" in joined, "the uncovered value is marked",
               "WOULD RAISE" in joined, True, failures)
        _check("a future clinvar wording" in joined, "the uncovered value is named",
               "a future clinvar wording" in joined, True, failures)
        _check("metadata.review_status" in joined, "the nested column is inventoried too",
               "metadata.review_status" in joined, True, failures)

        print("\n-- a MISSING TOKEN is not a would-raise row (the 2026-07-24 over-report) --")
        clean_mt = clean.set_column(
            clean.schema.get_field_index("ReviewStatus"), "ReviewStatus",
            pa.array(["", "-", "practice guideline"]))
        pq.write_table(clean_mt, repo / CLEAN_REL)
        out = []
        mt2, mo2 = read_missing_tokens(repo)
        n_raise = review_status_inventory(repo, tmap, origin, mt2, mo2, out)
        joined = "\n".join(out)
        _check(n_raise == 0, "'' and '-' do NOT count as would-raise", n_raise, 0, failures)
        _check("MISSING TOKEN" in joined, "they are labelled MISSING TOKEN",
               "MISSING TOKEN" in joined, True, failures)
        _check("WOULD RAISE" in joined, "the three-bucket totals are printed",
               "WOULD RAISE" in joined, True, failures)
        pq.write_table(clean, repo / CLEAN_REL)

        print("\n-- a fully covered cohort produces no finding --")
        clean_ok = clean.set_column(
            clean.schema.get_field_index("ReviewStatus"), "ReviewStatus",
            pa.array(["criteria provided, single submitter", "reviewed by expert panel",
                      "practice guideline"]))
        pq.write_table(clean_ok, repo / CLEAN_REL)
        out = []
        _check(review_status_inventory(repo, tmap, origin, mt, mo, out) == 0,
               "no uncovered rows when every value is a key", 0, 0, failures)

        print("\n-- sequence-branch gate counts failures and reports reasons --")
        pq.write_table(pa.table({
            "ok": pa.array([True, False, True, None]),
            "reason": pa.array([None, "window ran off contig", None, "no reference"]),
        }), repo / SEQ_REL)
        out = []
        bad = sequence_branch_gate(repo, out)
        joined = "\n".join(out)
        _check(bad == 2, "false plus null counted", bad, 2, failures)
        _check("window ran off contig" in joined, "reasons are reported",
               "window ran off contig" in joined, True, failures)

        print("\n-- an absent artifact is reported, not silently treated as clean --")
        (repo / SEQ_REL).unlink()
        out = []
        _check(sequence_branch_gate(repo, out) == 0 and "absent" in "\n".join(out),
               "absent sequence artifact is stated", "stated", "stated", failures)

        print("\n-- union is measured, not summed; transitions are classified --")
        old_col = ["", "",  "a", "a", "b", "c", "c", "", "-", "a", "c", "b"]
        new_col = ["a", "-", "a", "b", "",  "c", "a", "c", "a", "a", "c", "b"]
        pq.write_table(pa.table({
            "ReviewStatus": pa.array(old_col),
            "metadata": pa.array([{"review_status": v} for v in new_col],
                                 type=pa.struct([("review_status", pa.string())])),
        }), repo / CLEAN_REL)
        out = []
        union = source_comparison(repo, {"a": 1, "b": 2}, frozenset({"", "-"}), out)
        joined = "\n".join(out)
        # 'c' appears in 3 legacy rows and 3 repaired rows, overlapping on 2.
        _check(union == 4, "union is 4, not the naive sum of 6", union, 4, failures)
        _check("naive sum of the two -- NOT a row count" in joined,
               "the naive sum is shown and labelled as not a row count",
               "naive sum" in joined, True, failures)
        _check(joined.count("OK") >= 3, "all three reconciliations reported",
               joined.count("OK") >= 3, True, failures)
        for cat, want in (("unchanged", 5), ("FILLED (blank -> populated)", 3),
                          ("CHANGED (populated -> different populated)", 2),
                          ("EMPTIED (populated -> blank)", 1), ("blank in both", 1)):
            line = [l for l in out if l.rstrip().endswith(cat)]
            got = int(line[0].split()[0].replace(",", "")) if line else -1
            _check(got == want, f"transition category {cat!r}", got, want, failures)
        _check("does not only fill blanks" in joined, "a CHANGED row raises a finding",
               "does not only fill blanks" in joined, True, failures)
        _check("LOSE a populated status" in joined, "an EMPTIED row raises a finding",
               "LOSE a populated status" in joined, True, failures)

        print("\n-- a fill-only source produces no disagreement finding --")
        pq.write_table(pa.table({
            "ReviewStatus": pa.array(["", "", "a"]),
            "metadata": pa.array([{"review_status": v} for v in ["a", "b", "a"]],
                                 type=pa.struct([("review_status", pa.string())])),
        }), repo / CLEAN_REL)
        out = []
        source_comparison(repo, {"a": 1, "b": 2}, frozenset({"", "-"}), out)
        joined = "\n".join(out)
        _check("only fills blanks" in joined, "fill-only is stated positively",
               "only fills blanks" in joined, True, failures)
        _check("does not only fill blanks" not in joined, "no false disagreement finding",
               "does not only fill blanks" not in joined, True, failures)

        print("\n-- no tier map anywhere is a CONTRACT failure, not a pass --")
        (repo / "scripts/clean_cohort.py").unlink()
        try:
            read_tier_map(repo)
            _check(False, "missing tier map raises", "no raise", "ContractError", failures)
        except ContractError:
            _check(True, "missing tier map raises", "ContractError", "ContractError", failures)

    print()
    print("=" * 78)
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} check(s): {', '.join(failures)}")
        print("=" * 78)
        return 1
    print("SELF-TEST PASSED")
    print("=" * 78)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="What was removed, what statuses exist, did the sequence branch work?")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    repo = args.repo
    if repo is None:
        guess = Path(__file__).resolve().parents[1]
        repo = guess if (guess / "src" / "genomic_variant_classifier").is_dir() else None
    if repo is None:
        print("ERROR: pass --repo explicitly, for example", file=sys.stderr)
        print('       --repo "C:\\Projects\\genomic-variant-classifier"', file=sys.stderr)
        return EXIT_ENVIRONMENT
    repo = repo.resolve()

    out: list[str] = [
        "=" * 78,
        "COHORT DELTA FORENSICS",
        "READ-ONLY. Writes nothing.",
        "=" * 78,
        f"  repository : {repo}",
        "",
    ]
    try:
        tier_map, origin = read_tier_map(repo)
        _, unaccounted = classify_removed(repo, out)
        missing_tokens, missing_origin = read_missing_tokens(repo)
        uncovered = review_status_inventory(repo, tier_map, origin,
                                            missing_tokens, missing_origin, out)
        union_rows = source_comparison(repo, tier_map, missing_tokens, out)
        seq_bad = sequence_branch_gate(repo, out)
        provenance(repo, out)

        out.append("")
        out.append("=" * 78)
        out.append("VERDICT")
        out.append("=" * 78)
        out.append(f"  unaccounted removed rows        : {unaccounted:,}")
        out.append(f"  unmatched field occurrences, summed over 2 columns: {uncovered:,}")
        out.append(f"  DISTINCT ROWS affected (measured union)          : {union_rows:,}")
        out.append("    (neither a tier-map key nor a missing token. A value in")
        out.append("     MISSING_TOKENS resolves to TIER_MISSING before the map is")
        out.append("     consulted, so it is NOT counted here. The first version of")
        out.append("     this probe counted it and over-reported by 669,664 rows.)")
        out.append(f"  rows without a usable sequence window          : {seq_bad:,}")
        out.append("")
        code = EXIT_FINDING if (unaccounted or uncovered or seq_bad) else EXIT_CLEAN
        if code == EXIT_CLEAN:
            out.append("  EXIT 0 -- every measurement completed and every finding is benign.")
        else:
            out.append("  EXIT 1 -- at least one finding above requires resolution before the")
            out.append("  Phase 1 repair proceeds. Each is named in its own section.")
        print("\n".join(out))
        return code
    except ContractError as exc:
        print("\n".join(out))
        print("")
        print("=" * 78)
        print("CONTRACT FAILURE -- nothing conclusive was measured, and no verdict is issued.")
        print("=" * 78)
        print(f"  {exc}")
        return EXIT_ENVIRONMENT


if __name__ == "__main__":
    raise SystemExit(main())
