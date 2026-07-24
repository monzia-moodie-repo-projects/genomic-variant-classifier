#!/usr/bin/env python3
"""Which cohort artifact can actually support the section 2 recomputation, and what wrote it?

WHY THIS SCRIPT EXISTS
======================
`docs/PHASE1_SPEC_2026-07-24_deletion-repair.md` section 2 predicts the effect of
the review-status repair with five figures -- deletions carrying a populated
review status, deletions surviving the tier filter, the deletion share of the
surviving cohort, binary trainable rows, and the positive rate. Section 6,
measurement 5 of that same specification records that **every one of those
figures was measured on the `pathogenicity` column while production labels from
`clinical_sig`**.

On 2026-07-24 those two columns were measured against all three cohort artifacts
by `probe_label_column_terms.py`. `clinical_sig` diverges nowhere; `pathogenicity`
diverges everywhere, and its contents are arithmetically inconsistent with the
mapping code that has been in force since 2026-07-10. Section 2 therefore has to
be recomputed on `clinical_sig` before it can justify anything.

Recomputing it requires columns. This script establishes, by measurement rather
than by assumption, WHICH artifact carries the columns that recomputation needs,
and what the Parquet files themselves say about their own provenance. It is the
view-first step before the recomputation, not the recomputation.

WHAT IT REPORTS
---------------
  1. File-level Parquet metadata for each artifact: row count, row groups,
     format version, and the `created_by` string the writer stamped in. That
     string is the only provenance the files carry, and it is the first evidence
     to consult about when a column was written.
  2. The complete schema of each artifact, including nested struct fields, so a
     nested `metadata.review_status` is visible rather than inferred.
  3. A presence matrix for the columns the recomputation needs, naming exactly
     which artifacts can support it and which cannot.
  4. Schema deltas between artifacts -- what `_clean_seq` adds over `_clean`,
     and what `_clean` differs from the raw file by.
  5. For each required column that is present: null count and distinct count.

WHAT IT DOES NOT DO
-------------------
It does not compute the section 2 figures. Establishing feasibility and computing
a result are two jobs, and a tool that did both would report a number even when
the inputs could not support one -- which is how section 2 came to rest on the
wrong column in the first place.

EXIT CODES
----------
    0  At least one artifact carries every column the recomputation requires.
       The report names which.
    4  INSUFFICIENT SUPPORT. Every artifact is missing at least one required
       column. The probe ran; its inputs could not answer the question. The
       report names the gaps. See docs/standards/PROBE_EXIT_CODES.md.
    2  Contract or environment failure: no artifact found, or an artifact could
       not be read. An absence of measurement is never reported as a result.

READ-ONLY in every mode. `--self-test` round-trips small synthetic tables through
the system temporary directory and touches nothing under the repository.

Parquet is read with `pyarrow.parquet`, never `pandas.read_parquet`, per commit
644a184 (2026-07-23). Schema and file metadata are read without touching row data;
only the required columns are materialised, and only when they exist.

Acronyms on first use. AST = abstract syntax tree. ClinVar = the National Center
for Biotechnology Information's Clinical Variation archive. VCF = variant call
format. CNV = copy-number variant.

USAGE (PowerShell 5.1)
----------------------
    python "C:\\Users\\monzi\\Downloads\\probe_cohort_schema_census.py" --self-test

    python "C:\\Users\\monzi\\Downloads\\probe_cohort_schema_census.py" `
        --repo "C:\\Projects\\genomic-variant-classifier"

Author: written for Monzia Moodie, 2026-07-24. Companion to
probe_label_column_terms.py.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

EXIT_FEASIBLE = 0
EXIT_ENVIRONMENT = 2
#: 4, not 3. Per docs/standards/PROBE_EXIT_CODES.md, 3 means a NON-BLOCKING
#: FINDING and 4 means INSUFFICIENT SUPPORT -- the probe ran but its required
#: inputs were absent. 'No artifact carries every required column' is the
#: second, not the first: nothing was found, because nothing could be looked
#: at. Changed 2026-07-24 when the shared standard was written. NOTE that
#: pytest also exits 4 on a command-line usage error, so orchestration reading
#: both must not treat the two as one signal.
EXIT_NOT_FEASIBLE = 4

DEFAULT_COHORTS = (
    "data/processed/clinvar_grch38.parquet",
    "data/processed/clinvar_grch38_clean.parquet",
    "data/processed/clinvar_grch38_clean_seq.parquet",
)

#: What recomputing specification section 2 on `clinical_sig` requires, and why.
#: Nested paths use dotted notation and are matched against the flattened schema.
REQUIRED: tuple[tuple[str, str], ...] = (
    ("clinical_sig", "the labelling column; the whole point of the recomputation"),
    ("ref",          "classify deletions -- len(ref) > len(alt)"),
    ("alt",          "classify deletions -- len(ref) > len(alt)"),
    ("ReviewStatus", "the BEFORE review-status source (the VCF-derived column)"),
    ("metadata.review_status", "the AFTER review-status source (the nested column)"),
)
OPTIONAL: tuple[tuple[str, str], ...] = (
    ("variant_id", "row identity; duplicate detection"),
    ("gene",       "gene-level reporting, not required for section 2"),
    ("pathogenicity", "the column section 2 was WRONGLY measured on; reported for contrast"),
)

MAX_SCHEMA_LINES = 400


class ContractError(RuntimeError):
    """The artifacts do not look the way this census requires. Exit code 2."""


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------
def flatten_field(field, prefix: str = "") -> list[tuple[str, str]]:
    """Every leaf of a possibly-nested Arrow field, as (dotted_path, type).

    Struct columns are descended into so that a nested `metadata.review_status`
    is reported as a real, addressable path. A census that only listed top-level
    names would report `metadata` as present and leave the reader to assume what
    is inside it -- which is exactly the kind of assumption this project keeps
    paying for.
    """
    import pyarrow as pa

    name = f"{prefix}{field.name}"
    out: list[tuple[str, str]] = []
    if pa.types.is_struct(field.type):
        out.append((name, f"struct<{field.type.num_fields} field(s)>"))
        for i in range(field.type.num_fields):
            out.extend(flatten_field(field.type.field(i), prefix=f"{name}."))
    else:
        out.append((name, str(field.type)))
    return out


def flatten_schema(schema) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(len(schema)):
        out.extend(flatten_field(schema.field(i)))
    return out


def column_stats(path: Path, dotted: str) -> tuple[int, int] | None:
    """(null_count, distinct_count) for one possibly-nested column, or None.

    Only the one column is materialised. Nested paths are resolved by reading the
    struct and taking the child field, because `read_table(columns=["a.b"])` is
    not supported for struct children in every pyarrow version and silently
    reading the wrong thing would be worse than not reading at all.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    top = dotted.split(".", 1)[0]
    try:
        table = pq.read_table(path, columns=[top])
    except Exception:                                            # noqa: BLE001
        return None
    arr = table.column(top).combine_chunks()
    if "." in dotted:
        child = dotted.split(".", 1)[1]
        try:
            arr = arr.field(child)
        except Exception:                                        # noqa: BLE001
            return None
    nulls = arr.null_count
    try:
        # pyarrow.compute.unique COUNTS NULL AS A DISTINCT VALUE. Subtracting it
        # when nulls are present makes this the NON-NULL distinct count, which is
        # what a reader sitting beside a null count expects, and which matches
        # probe_cohort_delta_forensics.nulls_and_distinct so the two tools cannot
        # disagree. Corrected 2026-07-24: the first version inflated raw ref to
        # 38,497 when the true figure is 38,496, and raw alt to 20,668 when it is
        # 20,667.
        distinct = len(pc.unique(arr)) - (1 if nulls else 0)
    except Exception:                                            # noqa: BLE001
        distinct = -1
    return nulls, distinct


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------
class CohortCensus:
    def __init__(self, path: Path) -> None:
        import pyarrow.parquet as pq

        self.path = path
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        self.num_rows = meta.num_rows
        self.num_row_groups = meta.num_row_groups
        self.num_columns = meta.num_columns
        self.format_version = meta.format_version
        self.created_by = meta.created_by
        self.size_mib = path.stat().st_size / (1024 * 1024)
        self.flat = flatten_schema(pf.schema_arrow)
        self.names = {n for n, _ in self.flat}

    def missing_required(self) -> list[tuple[str, str]]:
        return [(n, why) for n, why in REQUIRED if n not in self.names]

    @property
    def feasible(self) -> bool:
        return not self.missing_required()

    def render(self, out: list[str]) -> None:
        out.append("#" * 78)
        out.append(f"COHORT: {self.path}")
        out.append("#" * 78)
        out.append("  -- file-level Parquet metadata (read without touching row data) --")
        out.append(f"    rows            : {self.num_rows:,}")
        out.append(f"    row groups      : {self.num_row_groups:,}")
        out.append(f"    leaf columns    : {self.num_columns:,}")
        out.append(f"    size on disk    : {self.size_mib:,.1f} MiB")
        out.append(f"    format version  : {self.format_version}")
        out.append(f"    created_by      : {self.created_by!r}")
        out.append("")
        out.append(f"  -- schema, flattened, {len(self.flat)} path(s) --")
        for name, typ in self.flat[:MAX_SCHEMA_LINES]:
            depth = name.count(".")
            out.append(f"    {'  ' * depth}{name:<44} {typ}")
        if len(self.flat) > MAX_SCHEMA_LINES:
            out.append(f"    ... and {len(self.flat) - MAX_SCHEMA_LINES:,} further path(s), "
                       f"suppressed")
        out.append("")
        out.append("  -- columns the section 2 recomputation REQUIRES --")
        for name, why in REQUIRED:
            mark = "PRESENT" if name in self.names else "ABSENT "
            out.append(f"    [{mark}] {name:<26} {why}")
        out.append("")
        out.append("  -- columns reported for context, not required --")
        for name, why in OPTIONAL:
            mark = "PRESENT" if name in self.names else "ABSENT "
            out.append(f"    [{mark}] {name:<26} {why}")
        out.append("")
        if self.feasible:
            out.append("  VERDICT: this artifact CAN support the section 2 recomputation.")
        else:
            gaps = ", ".join(n for n, _ in self.missing_required())
            out.append(f"  VERDICT: this artifact CANNOT support it. Missing: {gaps}")


def render_deltas(censuses: list[CohortCensus], out: list[str]) -> None:
    """Pairwise schema differences, so 'what did _seq add' is answered, not guessed."""
    out.append("=" * 78)
    out.append("SCHEMA DELTAS BETWEEN ARTIFACTS")
    out.append("=" * 78)
    if len(censuses) < 2:
        out.append("  fewer than two artifacts inspected; no delta to report.")
        return
    for i in range(len(censuses) - 1):
        a, b = censuses[i], censuses[i + 1]
        only_a = sorted(a.names - b.names)
        only_b = sorted(b.names - a.names)
        out.append("")
        out.append(f"  {a.path.name}  ->  {b.path.name}")
        out.append(f"    rows {a.num_rows:,} -> {b.num_rows:,}  "
                   f"(delta {b.num_rows - a.num_rows:+,})")
        out.append(f"    paths {len(a.flat):,} -> {len(b.flat):,}")
        out.append(f"    ADDED   ({len(only_b)}): "
                   f"{', '.join(only_b[:25]) if only_b else 'none'}"
                   f"{' ...' if len(only_b) > 25 else ''}")
        out.append(f"    REMOVED ({len(only_a)}): "
                   f"{', '.join(only_a[:25]) if only_a else 'none'}"
                   f"{' ...' if len(only_a) > 25 else ''}")


def render_stats(censuses: list[CohortCensus], out: list[str]) -> None:
    out.append("")
    out.append("=" * 78)
    out.append("NULL AND DISTINCT COUNTS FOR THE REQUIRED COLUMNS")
    out.append("  A required column that is present but wholly null cannot support the")
    out.append("  recomputation either, so presence alone is not the test.")
    out.append("=" * 78)
    for c in censuses:
        out.append("")
        out.append(f"  {c.path.name}   ({c.num_rows:,} rows)")
        for name, _ in REQUIRED:
            if name not in c.names:
                out.append(f"    {name:<26} ABSENT")
                continue
            stats = column_stats(c.path, name)
            if stats is None:
                out.append(f"    {name:<26} PRESENT but could not be materialised")
                continue
            nulls, distinct = stats
            pct = 0.0 if not c.num_rows else 100.0 * nulls / c.num_rows
            dtxt = "unknown" if distinct < 0 else f"{distinct:,}"
            flag = "   <== WHOLLY NULL" if nulls == c.num_rows else ""
            out.append(f"    {name:<26} nulls {nulls:>12,} ({pct:6.3f}%)   "
                       f"distinct {dtxt:>12}{flag}")


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
    print("SELF-TEST -- schema census")
    print("=" * 78)
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        print("\n-- a nested struct column is flattened to an addressable path --")
        nested = pa.table({
            "clinical_sig": pa.array(["Pathogenic", "Benign", None]),
            "ref": pa.array(["A", "AT", "G"]),
            "alt": pa.array(["T", "A", "C"]),
            "ReviewStatus": pa.array(["criteria provided, single submitter", "", "-"]),
            "metadata": pa.array(
                [{"review_status": "reviewed by expert panel", "src": "x"},
                 {"review_status": None, "src": "y"},
                 {"review_status": "practice guideline", "src": "z"}],
                type=pa.struct([("review_status", pa.string()), ("src", pa.string())])),
        })
        full = root / "full.parquet"
        pq.write_table(nested, full)
        c = CohortCensus(full)
        _check("metadata.review_status" in c.names, "nested path is discovered",
               "metadata.review_status" in c.names, True, failures)
        _check("metadata" in c.names, "the struct itself is also listed",
               "metadata" in c.names, True, failures)
        _check(c.feasible, "a complete artifact is judged feasible", c.feasible, True, failures)
        _check(c.missing_required() == [], "no required column reported missing",
               c.missing_required(), [], failures)
        _check(c.num_rows == 3, "row count from file metadata", c.num_rows, 3, failures)
        _check(isinstance(c.created_by, str) and len(c.created_by) > 0,
               "created_by provenance string is captured", bool(c.created_by), True, failures)

        print("\n-- null and distinct counts, including through a struct child --")
        # (1, 2), not (1, 3). Each fixture column holds two non-null distinct
        # values and one null. The original expectation of 3 encoded the very
        # defect D2 removed -- pyarrow.compute.unique counting null as a distinct
        # value -- so this self-test ASSERTED the defective behaviour and would
        # have gone red the moment the production path was fixed. Corrected
        # 2026-07-24, logged as D9. Lesson: a correction is not complete until the
        # tool's own self-test has been re-run against it.
        st = column_stats(full, "metadata.review_status")
        _check(st == (1, 2), "nested child: nulls and NON-NULL distinct", st, (1, 2), failures)
        st = column_stats(full, "clinical_sig")
        _check(st == (1, 2), "top-level: nulls and NON-NULL distinct", st, (1, 2), failures)
        st = column_stats(full, "ref")
        _check(st == (0, 3), "no nulls means no adjustment", st, (0, 3), failures)

        print("\n-- an artifact missing a required column is judged INFEASIBLE --")
        partial = root / "partial.parquet"
        pq.write_table(nested.drop_columns(["ReviewStatus"]), partial)
        c2 = CohortCensus(partial)
        _check(not c2.feasible, "missing ReviewStatus -> infeasible", c2.feasible, False, failures)
        _check([n for n, _ in c2.missing_required()] == ["ReviewStatus"],
               "the gap is named exactly", [n for n, _ in c2.missing_required()],
               ["ReviewStatus"], failures)

        print("\n-- exit-code discrimination --")
        _check(decide_exit([c]) == EXIT_FEASIBLE, "one feasible artifact -> 0",
               decide_exit([c]), EXIT_FEASIBLE, failures)
        _check(decide_exit([c2]) == EXIT_NOT_FEASIBLE, "no feasible artifact -> 4 (insufficient support)",
               decide_exit([c2]), EXIT_NOT_FEASIBLE, failures)
        _check(EXIT_NOT_FEASIBLE == 4, "insufficient support is 4, not 3",
               EXIT_NOT_FEASIBLE, 4, failures)
        _check(decide_exit([c2, c]) == EXIT_FEASIBLE, "any feasible artifact -> 0",
               decide_exit([c2, c]), EXIT_FEASIBLE, failures)
        _check(decide_exit([]) == EXIT_ENVIRONMENT, "nothing inspected -> 2",
               decide_exit([]), EXIT_ENVIRONMENT, failures)

        print("\n-- schema delta between two artifacts --")
        out: list[str] = []
        render_deltas([c2, c], out)
        joined = "\n".join(out)
        _check("ReviewStatus" in joined, "the added column is named in the delta",
               "ReviewStatus" in joined, True, failures)
        _check("ADDED   (1)" in joined, "exactly one column added",
               "ADDED   (1)" in joined, True, failures)

        print("\n-- a wholly-null required column is flagged, not silently accepted --")
        allnull = root / "allnull.parquet"
        pq.write_table(nested.set_column(
            nested.schema.get_field_index("ReviewStatus"), "ReviewStatus",
            pa.array([None, None, None], type=pa.string())), allnull)
        out = []
        render_stats([CohortCensus(allnull)], out)
        _check("WHOLLY NULL" in "\n".join(out), "wholly-null column is flagged",
               "WHOLLY NULL" in "\n".join(out), True, failures)

    print()
    print("=" * 78)
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} check(s): {', '.join(failures)}")
        print("=" * 78)
        return 1
    print("SELF-TEST PASSED")
    print("=" * 78)
    return 0


def decide_exit(censuses: list[CohortCensus]) -> int:
    if not censuses:
        return EXIT_ENVIRONMENT
    return EXIT_FEASIBLE if any(c.feasible for c in censuses) else EXIT_NOT_FEASIBLE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Which cohort artifact can support the section 2 recomputation?")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--cohort", action="append", default=None, metavar="PATH")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    repo = args.repo
    if repo is None:
        guess = Path(__file__).resolve().parents[1]
        marker = guess / "src" / "genomic_variant_classifier"
        repo = guess if marker.is_dir() else None
    if repo is None:
        print("ERROR: pass --repo explicitly, for example", file=sys.stderr)
        print('       --repo "C:\\Projects\\genomic-variant-classifier"', file=sys.stderr)
        return EXIT_ENVIRONMENT
    repo = repo.resolve()

    out: list[str] = []
    try:
        cohort_args = args.cohort if args.cohort else list(DEFAULT_COHORTS)
        resolved: list[Path] = []
        absent: list[str] = []
        for c in cohort_args:
            p = Path(c)
            p = p if p.is_absolute() else (repo / c)
            if p.is_file():
                resolved.append(p)
            else:
                absent.append(str(p))

        out.append("=" * 78)
        out.append("COHORT SCHEMA AND PROVENANCE CENSUS")
        out.append("READ-ONLY. Writes nothing.")
        out.append("=" * 78)
        out.append(f"  repository : {repo}")
        out.append(f"  artifacts  : {len(resolved)} present, {len(absent)} absent")
        for p in resolved:
            out.append(f"    PRESENT  {p}")
        for p in absent:
            out.append(f"    ABSENT   {p}")
        if not resolved:
            raise ContractError("no cohort artifact found; nothing was measured.")

        censuses: list[CohortCensus] = []
        for p in resolved:
            try:
                censuses.append(CohortCensus(p))
            except Exception as exc:                             # noqa: BLE001
                raise ContractError(f"{p}: could not be read -- "
                                    f"{type(exc).__name__}: {exc}") from exc
        for c in censuses:
            out.append("")
            c.render(out)

        out.append("")
        render_deltas(censuses, out)
        render_stats(censuses, out)

        code = decide_exit(censuses)
        out.append("")
        out.append("=" * 78)
        out.append("VERDICT")
        out.append("=" * 78)
        for c in censuses:
            gaps = ", ".join(n for n, _ in c.missing_required()) or "-"
            out.append(f"  {'FEASIBLE  ' if c.feasible else 'INFEASIBLE'}  "
                       f"{c.path.name}   missing: {gaps}")
        out.append("")
        if code == EXIT_FEASIBLE:
            names = ", ".join(c.path.name for c in censuses if c.feasible)
            out.append(f"  EXIT 0 -- the section 2 recomputation can proceed from: {names}")
        else:
            out.append("  EXIT 4 -- INSUFFICIENT SUPPORT: no artifact carries every required")
            out.append("  column. The recomputation cannot proceed until the gap is closed.")
        print("\n".join(out))
        return code
    except ContractError as exc:
        print("\n".join(out))
        print("")
        print("=" * 78)
        print("CONTRACT FAILURE -- nothing was measured, and no verdict is issued.")
        print("=" * 78)
        print(f"  {exc}")
        return EXIT_ENVIRONMENT


if __name__ == "__main__":
    raise SystemExit(main())
